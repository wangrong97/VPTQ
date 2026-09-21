# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Config mapping and weight loading for the vendored DeepSeek-V4 model,
plus a layer-streaming forward driver that keeps memory bounded for the
272B BF16 checkpoint.

Residency modes:
  - "cpu":    everything stays on CPU (slow, always works);
  - "stream": decoder layers live on CPU and are moved to the compute
              device one at a time, layer-major (each layer processes ALL
              samples before being offloaded). Works with a single
              partially-free accelerator;
  - "map":    layers are pinned to devices once (e.g. sharded over
              npu:0..7); activations flow across devices. Needs enough
              free memory to hold the full model.
"""

import glob
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from vptq.tools.deepseek_v4.model import ModelArgs, Transformer

logger = logging.getLogger("vptq.deepseek_v4")


def model_args_from_config(
    config_path: str,
    max_batch_size: int = 1,
    max_seq_len: int = 8192,
    n_layer_limit: Optional[int] = None,
) -> ModelArgs:
    """Build ModelArgs from the checkpoint's inference/config.json,
    forcing the BF16 weight format (the shipped inference/config.json
    still describes the FP8/FP4 original).

    `n_layer_limit` truncates the decoder to its first layers (the
    numbering is preserved, so checkpoint tensors still match) — meant
    for pipeline smoke tests without loading all 543GB."""
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.update(
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
        expert_dtype=None,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if n_layer_limit is not None:
        cfg["n_layers"] = n_layer_limit
    return ModelArgs(**cfg)


class ShardedWeightLoader:
    """Load HF-style sharded safetensors (model.safetensors.index.json)
    into the vendored Transformer, casting to each parameter's dtype.

    keepalive_device: if set, a tiny tensor op is issued on that device
    after every shard. This environment appears to reap processes whose
    accelerator sits idle through the (10+ minute) loading phase; the
    keep-alive prevents that."""

    def __init__(self, ckpt_dir: str, keepalive_device=None):
        index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.ckpt_dir = ckpt_dir
        self.keepalive_device = keepalive_device

    def _keepalive(self):
        if self.keepalive_device is not None:
            x = torch.ones(8, device=self.keepalive_device)
            del x

    def load(
        self, model: Transformer, strict: bool = True, workers: int = 16
    ) -> None:
        """Load all shards, several in parallel.

        The per-tensor path (safe_open get_tensor -> dtype cast ->
        param copy) is single-threaded inside a shard, which left the
        543GB load at ~160MB/s — 55 minutes wall clock, longer than
        this environment's process-killer cycle, so a killed run could
        never finish loading. Shards touch disjoint parameters, so
        loading them concurrently is race-free; the keep-alive stays on
        the main thread (NPU ops are issued from one thread only).
        """
        params = dict(model.named_parameters())
        # name -> shard file
        shards: Dict[str, List[str]] = {}
        for name, shard in self.weight_map.items():
            shards.setdefault(shard, []).append(name)
        loaded = set()
        lock = threading.Lock()
        wanted = 0

        def load_one(item):
            shard, names = item
            path = os.path.join(self.ckpt_dir, shard)
            hits = [n for n in names if n in params]
            if not hits:
                if strict:
                    raise KeyError(
                        f"unexpected tensors in {shard}: {names[:4]}..."
                    )
                return 0
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in hits:
                    tensor = f.get_tensor(name)
                    param = params[name]
                    if tensor.shape != param.shape:
                        raise ValueError(
                            f"shape mismatch for {name}: "
                            f"ckpt {tuple(tensor.shape)} vs "
                            f"param {tuple(param.shape)}"
                        )
                    param.data.copy_(tensor.to(param.dtype))
                    with lock:
                        loaded.add(name)
            logger.info("loaded shard %s (%d tensors)", shard, len(hits))
            return len(hits)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map() yields in submission order; keep-alive from the main
            # thread once per completed shard
            for n_hits in pool.map(load_one, sorted(shards.items())):
                wanted += n_hits
                self._keepalive()
        missing = set(params) - loaded
        # non-persistent buffers (kv caches etc.) are not parameters and
        # are expected to stay uninitialized.
        if missing and strict:
            raise KeyError(f"missing tensors: {sorted(missing)[:8]}...")


def build_model(
    ckpt_dir: str,
    max_batch_size: int = 1,
    max_seq_len: int = 8192,
    device: str = "cpu",
    strict: bool = True,
    n_layer_limit: Optional[int] = None,
) -> Transformer:
    """Construct the Transformer on CPU and load sharded BF16 weights.

    With `n_layer_limit`, only the first decoder layers are built and
    loaded (strict is relaxed automatically)."""
    args = model_args_from_config(
        os.path.join(ckpt_dir, "inference", "config.json"),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        n_layer_limit=n_layer_limit,
    )
    logger.info("ModelArgs: %s", args)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # keep construction on CPU regardless of the runtime device
        model = Transformer(args)
    finally:
        torch.set_default_dtype(prev_dtype)
    keepalive = None
    if device not in ("cpu", None):
        keepalive = torch.device(device)
    ShardedWeightLoader(ckpt_dir, keepalive_device=keepalive).load(
        model, strict=strict and n_layer_limit is None
    )
    model.eval()
    return model


@torch.inference_mode()
def streamed_forward(
    model: Transformer,
    samples: torch.Tensor,
    device: torch.device,
    residency: str = "stream",
    batch_size: int = 1,
    on_layer_start=None,
    on_layer_done=None,
    act_ckpt_path: Optional[str] = None,
) -> None:
    """Run all samples through embedding + decoder layers.

    Only the compute-path modules (embed + layers) are driven; the LM
    head is skipped (logits are irrelevant for Hessian collection).

    Args:
        model: Transformer on CPU (residency "stream"/"cpu") or already
            placed ("map": caller moved modules to their devices first).
        samples: (n_samples, seq_len) long tensor on CPU.
        device: compute device for "stream" mode.
        residency: "cpu" | "stream" | "map".
        batch_size: sequences per forward inside each layer.
        on_layer_start/on_layer_done: optional callbacks(layer_idx)
            bracketing each layer (used to save/free Hessians).
        act_ckpt_path: optional activation-checkpoint file. After each
            layer the hidden states are written there (atomically); if
            the file already exists and matches (nsamples, seqlen,
            batch_size), forward resumes directly from that layer
            instead of re-running everything.
    """
    n_samples = samples.shape[0]
    device_cpu = torch.device("cpu")
    compute = device_cpu if residency == "cpu" else device

    def run_embedding(ids: torch.Tensor) -> torch.Tensor:
        if residency == "stream":
            model.embed.to(compute)
        h = model.embed(ids.to(compute))
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        if residency == "stream":
            model.embed.to(device_cpu)
        return h.to(device_cpu)

    start_layer = 0
    hidden = None
    if act_ckpt_path and os.path.exists(act_ckpt_path):
        try:
            ckpt = torch.load(act_ckpt_path, weights_only=False)
            if (
                ckpt["nsamples"],
                ckpt["seqlen"],
                ckpt["batch_size"],
            ) == (n_samples, samples.shape[1], batch_size):
                hidden = ckpt["hidden"]
                start_layer = ckpt["next_layer"]
                logger.info(
                    "resuming activations from layer %d (%s)",
                    start_layer,
                    act_ckpt_path,
                )
            else:
                logger.warning("act ckpt mismatch, starting from layer 0")
        except Exception as err:
            logger.warning("act ckpt unreadable (%s), starting over", err)

    if hidden is None:
        logger.info("embedding %d samples ...", n_samples)
        hidden = [
            run_embedding(samples[i : i + batch_size])
            for i in range(0, n_samples, batch_size)
        ]

    for layer_idx, layer in enumerate(model.layers):
        if layer_idx < start_layer:
            continue
        if on_layer_start is not None:
            on_layer_start(layer_idx)
        layer_device = compute
        if residency == "map":
            layer_device = next(layer.parameters()).device
        elif residency == "stream":
            layer.to(compute)
        logger.info(
            "layer %d/%d on %s ...",
            layer_idx,
            len(model.layers),
            layer_device,
        )
        t_layer = time.time()
        for bi, h in enumerate(hidden):
            ids = samples[bi * batch_size : (bi + 1) * batch_size]
            hidden[bi] = layer(
                h.to(layer_device), 0, ids.to(layer_device)
            ).to(device_cpu)
            if (bi + 1) % 16 == 0 or bi + 1 == len(hidden):
                logger.info(
                    "layer %d: %d/%d batches (%.0fs elapsed)",
                    layer_idx,
                    bi + 1,
                    len(hidden),
                    time.time() - t_layer,
                )
        if residency == "stream":
            layer.to(device_cpu)
        if on_layer_done is not None:
            on_layer_done(layer_idx)
        if act_ckpt_path:
            tmp = act_ckpt_path + ".tmp"
            t_save = time.time()
            torch.save(
                {
                    "hidden": hidden,
                    "next_layer": layer_idx + 1,
                    "nsamples": n_samples,
                    "seqlen": samples.shape[1],
                    "batch_size": batch_size,
                },
                tmp,
            )
            os.replace(tmp, act_ckpt_path)
            logger.info(
                "layer %d: act ckpt saved in %.0fs",
                layer_idx,
                time.time() - t_save,
            )
