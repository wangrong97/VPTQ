# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Hessian collection for DeepSeek-V4-Flash-BF16 on Ascend NPU / CPU.

Drives the vendored native model layer-major (streaming residency), and
for each decoder layer builds a HessianCollector whose hooks accumulate
H = E[x x^T] per linear input — including per-expert Hessians for the
256 routed experts (experts' inputs are the routed token subsets, which
the hooks capture naturally).

Note: the MoE router (`ffn.gate`) applies its weight via a functional
linear (no module), so it has no hook; its input is identical to the
shared expert's input, whose `ffn.shared_experts.mlp_in` Hessian can be
reused for the router.

Example (single NPU, streaming):
    python -m vptq.tools.deepseek_v4.collect_hessian \
        --ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
        --data-dir /mnt/share/rr08002/weights/RedPajama-Data-1T-Sample \
        --output-dir ./hessians/dsv4-flash \
        --nsamples 32 --seqlen 4096 --device npu --residency stream
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers

from vptq.tools.deepseek_v4.loader import build_model, streamed_forward
from vptq.tools.deepseek_v4.model import Linear
from vptq.tools.hessian import HessianCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.deepseek_v4.hessian")


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect Hessians for DeepSeek-V4-Flash-BF16."
    )
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory with calibration .jsonl (optionally .zst) files.",
    )
    p.add_argument(
        "--data-file",
        default=None,
        help=(
            "Single .jsonl calibration file (e.g. benchmark-aligned "
            "corpus). Overrides the RedPajama slice mix: lines are read "
            "in file order, concatenated and chunked. Baseline "
            "collection mechanics (all groups hooked, no token floor)."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--residency",
        default="stream",
        choices=["cpu", "stream", "map"],
        help="Weight residency strategy (see loader.py).",
    )
    p.add_argument(
        "--device",
        default="npu",
        help="Compute device for stream mode (e.g. npu, npu:0, cpu).",
    )
    p.add_argument(
        "--layer-range",
        default=None,
        help="Only save Hessians for layers start:end "
        "(all layers still run).",
    )
    p.add_argument(
        "--accum-dtype", default="fp32", choices=["fp32", "fp64"]
    )
    p.add_argument(
        "--save-inv",
        action="store_true",
        help="Also compute inverse Hessians (expensive for 256 experts "
        "per layer; default off).",
    )
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="ModelArgs.max_seq_len (sizes KV buffers).",
    )
    p.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for residency=map, e.g. "
        "npu:0,npu:1,...,npu:7. Layers are pinned contiguously.",
    )
    p.add_argument(
        "--n-layer-limit",
        type=int,
        default=None,
        help="Build/load only the first N decoder layers (smoke tests).",
    )
    p.add_argument(
        "--token-floor",
        type=int,
        default=0,
        help="Top-up mode: continue collecting expert groups whose "
        "n_tokens is below this floor, restoring state from the saved "
        "layer files. 0 disables (original one-shot collection). Only "
        "ffn.experts.N.mlp_in/w2 groups are topped up; all other groups "
        "keep their saved statistics unchanged.",
    )
    p.add_argument(
        "--topup-seed",
        type=int,
        default=None,
        help="Sampling seed for top-up calibration data (default: seed+1). "
        "Note: only affects multi-file slice ordering (common_crawl); "
        "fresh text across rounds comes from --topup-round.",
    )
    p.add_argument(
        "--topup-round",
        type=int,
        default=0,
        help="Which consecutive segment of each slice's document stream "
        "the top-up samples come from (0 = same text as the original "
        "collection; k>0 = the k-th fresh batch). This is the knob that "
        "actually varies the data — see build_samples docstring.",
    )
    p.add_argument(
        "--max-extra-samples",
        type=int,
        default=512,
        help="Calibration sequences for top-up mode (upper bound; "
        "collection stops consuming once every group reaches the floor "
        "via the hook-side cap check).",
    )
    p.add_argument(
        "--act-ckpt",
        default=None,
        help="Override path for the activation checkpoint. With 512 "
        "samples the ckpt is ~64GB per layer write + read-back on every "
        "watchdog restart; on NFS that alone eats most of the killer "
        "window. Point it at local NVMe for top-up runs.",
    )
    p.add_argument(
        "--save-staging",
        default=None,
        help="Write layer files here first and upload them to "
        "--output-dir from a background thread (one at a time, at most "
        "MAX_STAGING_LAYERS pending). Moves the 2-4min/layer NFS write "
        "off the critical path. Point at local NVMe; must survive "
        "process kills (residuals are flushed on next start before the "
        "resume scan).",
    )
    p.add_argument(
        "--prefetch-next",
        action="store_true",
        help="In top-up mode, advise the kernel to read-ahead the next "
        "layer's Hessian file while the current layer forwards, so "
        "restore_state hits page cache instead of NFS.",
    )
    return p.parse_args()


# Official RedPajama-Data-1T slice proportions (token share, NeurIPS'24).
# The official VPTQ Hessians were collected on RedPajama-Data-1T-Sample,
# which preserves this mix; we reproduce it per-slice. The "book" slice is
# absent from the local mirror, so proportions are renormalized over the
# available slices.
_REDPJ_SLICE_MIX = {
    "common_crawl": 0.731,
    "c4": 0.146,
    "github": 0.049,
    "arxiv": 0.023,
    "wikipedia": 0.020,
    "stackexchange": 0.017,
    # "book": 0.022  (not mirrored)
}

# Expert-input groups subject to the top-up floor: per-expert mlp_in
# (w1/w3 shared input) and w2 both see the routed token subset of that
# expert, so both must be topped up together. Everything else (attn,
# shared experts/router, wo_a) is frozen by cap 0 to keep the top-up a
# controlled change on top of the existing product.
_EXPERT_KEY_RE = re.compile(r"^ffn\.experts\.\d+\.(mlp_in|w2)$")

_SLICE_OF_FILE = (
    ("common_crawl", "common_crawl"),
    ("c4", "c4"),
    ("github", "github"),
    ("arxiv", "arxiv"),
    ("wiki", "wikipedia"),
    ("stackexchange", "stackexchange"),
)


def _slice_of(path: str) -> str:
    name = os.path.basename(path)
    for marker, slice_name in _SLICE_OF_FILE:
        if marker in name:
            return slice_name
    return "unknown"


def iter_texts(data_dir: str, seed: int, only_files=None):
    """Yield texts from .jsonl / .jsonl.zst files (optionally restricted)."""
    files = sorted(
        glob.glob(os.path.join(data_dir, "*.jsonl"))
        + glob.glob(os.path.join(data_dir, "*.jsonl.zst"))
    )
    if only_files is not None:
        files = [f for f in files if f in only_files]
    rng = random.Random(seed)
    rng.shuffle(files)
    for path in files:
        opener = open
        if path.endswith(".zst"):
            try:
                import zstandard
            except ImportError:
                logger.warning("zstandard missing, skipping %s", path)
                continue

            def opener(p, mode="rt"):
                return zstandard.open(p, mode)

        logger.info("reading calibration texts from %s", path)
        with opener(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue


def build_samples(tokenizer, data_dir, nsamples, seqlen, seed, round_idx=0, data_file=None):
    """Build (nsamples, seqlen) token sequences following the official
    RedPajama slice mix: each slice contributes round(nsamples * share)
    sequences, built by tokenizing that slice's documents and chunking.

    round_idx: which consecutive segment of each slice's document token
    stream to take — segment k = tokens [k*quota*seqlen, (k+1)*quota*seqlen).
    Round 0 reproduces the original collection; top-up rounds consume
    FRESH text. Seed-shuffling files alone is NOT sufficient here: 5 of
    the 6 slices are single-file mirrors, so their document order is
    seed-independent (verified empirically: seeds 1/2/3 yielded 100%
    identical samples, and seed 1 overlapped the original seed-0 batch
    by the entire single-file quota, 26%).

    data_file: single-file mode (e.g. a benchmark-aligned calibration
    corpus). The file's lines are read IN FILE ORDER (deterministic,
    matching the single-slice path of the mix above), tokenized,
    concatenated and chunked at seqlen — same concatenation scheme,
    minus slice quotas (one file IS the mix). Text exhaustion yields
    fewer than nsamples samples (no padding).
    """
    if data_file is not None:
        need = nsamples * seqlen
        start = round_idx * need
        target = start + need
        ids = []
        logger.info("reading calibration texts from %s", data_file)
        with open(data_file, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                ids.extend(tokenizer(text).input_ids)
                if len(ids) >= target:
                    break
        n_full = min(nsamples, max(0, len(ids) - start) // seqlen)
        if n_full < nsamples:
            logger.warning(
                "data_file: only %d/%d sequences filled (not enough text)",
                n_full, nsamples,
            )
        return [
            ids[start + i * seqlen : start + (i + 1) * seqlen]
            for i in range(n_full)
        ]
    files = sorted(
        glob.glob(os.path.join(data_dir, "*.jsonl"))
        + glob.glob(os.path.join(data_dir, "*.jsonl.zst"))
    )
    by_slice = {}
    for path in files:
        by_slice.setdefault(_slice_of(path), []).append(path)
    total_share = sum(
        share for s, share in _REDPJ_SLICE_MIX.items() if s in by_slice
    )
    rng = random.Random(seed)
    all_seqs = []
    stats = {}
    for slice_name, share in _REDPJ_SLICE_MIX.items():
        if slice_name not in by_slice:
            logger.warning("slice %s not found in %s", slice_name, data_dir)
            continue
        quota = max(1, round(nsamples * share / total_share))
        need = quota * seqlen
        start = round_idx * need
        target = start + need
        ids = []
        for text in iter_texts(data_dir, seed, only_files=by_slice[slice_name]):
            ids.extend(tokenizer(text).input_ids)
            if len(ids) >= target:
                break
        avail = max(0, len(ids) - start)
        n_full = min(quota, avail // seqlen)
        if n_full < quota:
            logger.warning(
                "slice %s: only %d/%d sequences filled (not enough text)",
                slice_name,
                n_full,
                quota,
            )
        for i in range(n_full):
            all_seqs.append(ids[start + i * seqlen : start + (i + 1) * seqlen])
        stats[slice_name] = n_full
    rng.shuffle(all_seqs)
    samples = torch.tensor(all_seqs[:nsamples])
    logger.info("built %d samples of length %d (round %d); mix: %s",
                samples.shape[0], seqlen, round_idx, stats)
    return samples


def _acquire_lock(output_dir: str) -> str:
    """Single-instance lock: refuse to start if a live process already
    works on this output dir (prevents the duplicate-process storms that
    contended for NPU memory). Stale locks of dead PIDs are reclaimed."""
    os.makedirs(output_dir, exist_ok=True)
    lock_path = os.path.join(output_dir, ".collect.lock")
    if os.path.exists(lock_path):
        try:
            old_pid = int(open(lock_path).read().strip())
            os.kill(old_pid, 0)
            raise SystemExit(
                f"another collect_hessian process (pid {old_pid}) is "
                f"already running on {output_dir}; exiting"
            )
        except ProcessLookupError:
            logger.warning("reclaiming stale lock of pid %d", old_pid)
        except ValueError:
            pass
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))
    import atexit

    atexit.register(lambda: os.path.exists(lock_path)
                    and os.remove(lock_path))
    return lock_path


# staging backlog cap: each pending layer is ~45GB, keep local NVMe
# usage bounded while the upload thread drains it to NFS
MAX_STAGING_LAYERS = 4


def _upload_layer(staging_path: str, output_dir: str) -> None:
    """Copy one staged layer file to the output dir atomically."""
    name = os.path.basename(staging_path)
    os.makedirs(output_dir, exist_ok=True)
    tmp = os.path.join(output_dir, name + ".tmp")
    shutil.copyfile(staging_path, tmp)
    os.replace(tmp, os.path.join(output_dir, name))
    os.remove(staging_path)
    logger.info("uploaded %s to output dir", name)


def _flush_staging(staging_dir: str, output_dir: str) -> None:
    """Upload residuals from a killed run before anything reads the
    output dir — otherwise the resume scan would see stale layers.
    Torn .tmp files (killed mid-save or mid-upload) are deleted: their
    layer will simply be re-collected on this run."""
    if not os.path.isdir(staging_dir):
        return
    for name in sorted(os.listdir(staging_dir)):
        path = os.path.join(staging_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".tmp"):
            os.remove(path)
            logger.warning("removed torn staging file %s", name)
        else:
            _upload_layer(path, output_dir)


def _prefetch_file(path: str) -> None:
    """Advise the kernel to read-ahead a file (best effort)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _scan_layers(output_dir: str, token_floor: int):
    """Mmap-scan saved layer files.

    Returns (done_layers, under_floor): the set of layer indices with a
    readable file, and {idx: {group_key: n_tokens}} for expert groups
    below the floor (token_floor>0 only). Mmap keeps the scan at
    seconds per layer regardless of file size; a layer that cannot be
    mmap-opened falls back to a full validation load (non-top-up only).
    """
    done_layers = set()
    under_floor = {}
    for path in glob.glob(os.path.join(output_dir, "layer_*.pt")):
        idx = int(os.path.basename(path)[6:10])
        try:
            payload = torch.load(path, mmap=True, map_location="cpu")
        except Exception:
            if token_floor > 0:
                logger.warning(
                    "layer %d not mmap-readable; cannot top it up", idx
                )
                continue
            try:
                torch.load(path, weights_only=False)
                done_layers.add(idx)
            except Exception:
                logger.warning("layer %d file corrupt, will redo", idx)
            continue
        if token_floor > 0:
            under = {
                k: entry["n_tokens"]
                for k, entry in payload.items()
                if _EXPERT_KEY_RE.match(k)
                and entry["n_tokens"] < token_floor
            }
            if under:
                under_floor[idx] = under
        del payload
        done_layers.add(idx)
    return done_layers, under_floor


def main():
    args = parse_args()
    _acquire_lock(args.output_dir)
    dtype_map = {"fp32": torch.float32, "fp64": torch.float64}
    device = torch.device(args.device)
    layer_range = None
    if args.layer_range:
        start, end = args.layer_range.split(":")
        layer_range = (int(start), int(end))

    topup = args.token_floor > 0
    if args.data_file and topup:
        p_error = "--data-file is baseline collection only (all groups "
        p_error += "hooked, no floor); use a plain RedPajama dir to top up"
        raise SystemExit(p_error)
    # staging residuals from a killed run must land in the output dir
    # BEFORE the resume scan, or the scan would see stale layers
    if args.save_staging:
        os.makedirs(args.save_staging, exist_ok=True)
        _flush_staging(args.save_staging, args.output_dir)
    # scan before building tokenizer/model so a no-op top-up exits fast
    done_layers, under_floor = _scan_layers(
        args.output_dir, args.token_floor if topup else 0
    )
    if topup:
        total_under = sum(len(v) for v in under_floor.values())
        logger.info(
            "top-up: %d/%d layers have %d expert groups below floor %d",
            len(under_floor),
            len(done_layers),
            total_under,
            args.token_floor,
        )
        if not under_floor:
            logger.info("top-up: everything already above floor; done")
            return
        sample_seed = (
            args.topup_seed if args.topup_seed is not None
            else args.seed + 1
        )
        nsamples = args.max_extra_samples
    else:
        sample_seed = args.seed
        nsamples = args.nsamples
        if done_layers:
            logger.info(
                "resuming: %d layers already done (%s)",
                len(done_layers),
                sorted(done_layers),
            )

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.ckpt)
    except Exception as err:
        # unknown model_type (deepseek_v4) can break config parsing in
        # AutoTokenizer; fall back to the raw fast tokenizer
        logger.warning("AutoTokenizer failed (%s); using tokenizer.json", err)
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(args.ckpt, "tokenizer.json")
        )
    samples = build_samples(
        tokenizer,
        args.data_dir,
        nsamples,
        args.seqlen,
        sample_seed,
        round_idx=args.topup_round if topup else 0,
        data_file=args.data_file,
    )
    if isinstance(samples, list):
        # single-file mode returns a list of equal-length token lists
        # (no padding); main (meta/forward) expects a (n, seqlen) tensor
        samples = torch.tensor(samples, dtype=torch.long)

    model = build_model(
        args.ckpt,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device if args.residency == "map" else "cpu",
        n_layer_limit=args.n_layer_limit,
    )

    if args.residency == "map":
        devices = [
            torch.device(d.strip()) for d in args.devices.split(",")
        ]
        n_layers = len(model.layers)
        # balanced contiguous assignment: first (n % k) devices take one
        # extra layer
        base, rem = divmod(n_layers, len(devices))
        bounds, start = [], 0
        for di in range(len(devices)):
            cnt = base + (1 if di < rem else 0)
            bounds.append((start, start + cnt))
            start += cnt
        for di, (lo, hi) in enumerate(bounds):
            for i in range(lo, hi):
                model.layers[i].to(devices[di])
            if lo < hi:
                logger.info("layers %d..%d -> %s", lo, hi - 1, devices[di])
        # embed goes to the last (lightest) device; activations start there
        model.embed.to(devices[-1])
        device = devices[-1]

    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "nsamples": int(samples.shape[0]),
        "seqlen": args.seqlen,
        "seed": sample_seed,
        "residency": args.residency,
        "convention": "hessian = E[x x^T]",
        "router_note": (
            "ffn.gate (router) has no hook (functional linear); reuse "
            "ffn.shared_experts.mlp_in — identical input."
        ),
    }
    if topup:
        meta["topup"] = {
            "token_floor": args.token_floor,
            "base_seed": args.seed,
            "layers_pending_before": {
                str(idx): len(groups)
                for idx, groups in sorted(under_floor.items())
            },
        }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    state = {"collector": None, "prev_payload": None, "before": None}

    # off-critical-path layer upload (staging mode): one worker, with a
    # backlog cap so local NVMe never fills up
    upload_pool = (
        ThreadPoolExecutor(max_workers=1) if args.save_staging else None
    )
    upload_queue = deque()

    # v3.4: off-critical-path layer SAVE too — the 21GB torch.save is
    # ~55s of pure IO per layer and overlaps the next layer's forward.
    # One worker keeps layer files written in order; depth-2 backpressure
    # bounds staging space and payload RAM. Drained before process exit
    # (the layer file IS the cross-round checkpoint).
    save_pool = ThreadPoolExecutor(max_workers=1)
    save_futures = deque()

    def _submit_upload(name: str):
        # backpressure: wait for the oldest upload when too many pending
        while len(upload_queue) >= MAX_STAGING_LAYERS:
            upload_queue.popleft().result()
        upload_queue.append(
            upload_pool.submit(
                _upload_layer,
                os.path.join(args.save_staging, name),
                args.output_dir,
            )
        )

    def cap_fn(key: str):
        # top-up mode: expert groups collect until the floor; everything
        # else is frozen at cap 0 so the run changes only what it means
        # to change (controlled comparison against the base product)
        if _EXPERT_KEY_RE.match(key):
            return args.token_floor
        return 0

    def on_layer_start(idx: int):
        if topup and args.prefetch_next:
            threading.Thread(
                target=_prefetch_file,
                args=(
                    os.path.join(
                        args.output_dir, f"layer_{idx + 1:04d}.pt"
                    ),
                ),
                daemon=True,
            ).start()
        if topup:
            if idx not in under_floor:
                return  # satisfied (or unreadable): pass-through layer
        elif idx in done_layers:
            logger.info("layer %d already collected, skipping hooks", idx)
            return
        if layer_range and not (layer_range[0] <= idx < layer_range[1]):
            return
        # accumulators stay on CPU. NPU residency was tried (09-20) and
        # works compute-wise (~0 accumulation overhead), but a card
        # holding 6 resident layers (~76GB) + the worst layer's 136
        # below-floor groups (~6.5GB fp32) + the even-layer indexer's
        # ~5GB workspace tops out HBM — and this NPU stack hangs to an
        # aicpu timeout instead of raising OOM (layer 16, twice). The
        # per-group transfer cost is addressed instead with a single
        # bulk copy in the hook (see _make_hook).
        collector = HessianCollector(
            model,
            layer_range=(idx, idx + 1),
            accum_dtype=dtype_map[args.accum_dtype],
            accum_device=torch.device("cpu"),
            linear_types=(Linear,),
            token_caps=cap_fn if topup else None,
        )
        if topup:
            t_restore = time.time()
            prev = torch.load(
                os.path.join(args.output_dir, f"layer_{idx:04d}.pt"),
                mmap=True,
                map_location="cpu",
            )
            t_open = time.time()
            restored = collector.restore_state(idx, prev)
            logger.info(
                "layer %d: restore open %.0fs, restore_state %.0fs",
                idx,
                t_open - t_restore,
                time.time() - t_open,
            )
            state["prev_payload"] = prev
            state["before"] = {
                k: collector.accumulators[idx][k].n_tokens
                for k in restored
            }
            logger.info(
                "layer %d: restored %d groups below floor "
                "(coldest %d tokens)",
                idx,
                len(restored),
                min(state["before"].values()) if state["before"] else -1,
            )
        state["collector"] = collector

    def on_layer_done(idx: int):
        collector = state["collector"]
        if collector is None:
            return
        prev = state.pop("prev_payload", None)
        before = state.pop("before", None)
        extra_meta = None
        if topup:
            pending = collector.pending_under_caps()
            after = {
                k: acc.n_tokens
                for k, acc in collector.accumulators[idx].items()
                if before and k in before
            }
            gained = sum(1 for k, n in after.items() if n > before[k])
            reached = sum(
                1 for k, n in after.items() if n >= args.token_floor
            )
            logger.info(
                "layer %d top-up: %d/%d groups gained tokens, "
                "%d reached floor, %d still below",
                idx,
                gained,
                len(before or {}),
                reached,
                len(pending),
            )
            extra_meta = {
                "topup": {
                    "token_floor": args.token_floor,
                    "topup_seed": sample_seed,
                    "extra_samples": int(samples.shape[0]),
                    "last_layer": idx,
                    "last_layer_still_pending": len(pending),
                }
            }
        collector.remove_hooks()
        # hand the save to the worker thread; the collector reference
        # moves with the job and dies when it ends (collector.save pops
        # each accumulator right after finalizing it, so NPU residency
        # does not stack across the overlap window)
        while len(save_futures) >= 2:
            save_futures.popleft().result()

        def _save_job(c=collector, p=prev, m=extra_meta, i=idx):
            c.save(
                args.save_staging or args.output_dir,
                model_name=args.ckpt,
                save_inv=args.save_inv,
                damp=args.damp,
                extra_meta=m,
                prev_payload=p,
            )
            if upload_pool is not None:
                _submit_upload(f"layer_{i:04d}.pt")

        save_futures.append(save_pool.submit(_save_job))
        state["collector"] = None
        del collector

    streamed_forward(
        model,
        samples,
        device=device,
        residency=args.residency,
        batch_size=args.batch_size,
        on_layer_start=on_layer_start,
        on_layer_done=on_layer_done,
        act_ckpt_path=args.act_ckpt
        or os.path.join(args.output_dir, "act_ckpt.pt"),
    )
    # drain pending saves first: exiting before a layer file lands would
    # silently drop that layer's round contribution (the layer file is
    # the checkpoint every later attempt/round resumes from)
    while save_futures:
        save_futures.popleft().result()
    save_pool.shutdown()
    if upload_pool is not None:
        while upload_queue:
            upload_queue.popleft().result()
        # pick up meta.json and anything else the kills left in staging
        _flush_staging(args.save_staging, args.output_dir)
        upload_pool.shutdown()
    if topup:
        logger.info(
            "top-up pass written. Run compute_inverse next to fill "
            "inv_hessian for the updated groups (existing inverses are "
            "kept and skipped automatically)."
        )
    logger.info("done. Hessians written to %s", args.output_dir)


if __name__ == "__main__":
    main()
