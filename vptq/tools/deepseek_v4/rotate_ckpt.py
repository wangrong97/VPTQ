# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Checkpoint-level random Hadamard absorption for DeepSeek-V4-Flash-BF16.

Rewrites a BF16 checkpoint so the model runs in the Hadamard-rotated
residual basis (see vptq/tools/quantize/hadamard.py for the scheme).
Streams shards; per tensor name applies:

  embed.weight                      -> E @ R
  head.weight                       -> (W @ diag(g_final)) @ R   (final norm folded)
  layers.N.attn_norm.weight         -> 1  (gain folded into attn inputs)
  layers.N.ffn_norm.weight          -> 1  (gain folded into ffn consumers)
  layers.N.attn.{wq_a,wkv,
               compressor.{wkv,wgate},
               indexer.weights_proj}.weight
                                    -> (W @ diag(g_attn)) @ R
  layers.N.attn.wo_b.weight         -> R^T @ W        (output side)
  layers.N.ffn.gate.weight          -> (W @ diag(g_ffn)) @ R
  layers.N.ffn.shared_experts.{w1,w3}.weight -> (W @ diag(g_ffn)) @ R
  layers.N.ffn.shared_experts.w2.weight     -> R^T @ W
  layers.N.ffn.experts.E.{w1,w3}.weight     -> (W @ diag(g_ffn)) @ R
  layers.N.ffn.experts.E.w2.weight          -> R^T @ W
  layers.N.hc_{attn,ffn}_fn         -> W @ R_blk      (R_blk = blockdiag x4)
  hc_head_fn                        -> W @ R_blk
  mtp.0.*                           -> same block rules; e_proj/h_proj input side
  norm.weight (final)               -> 1  (folded into head)

Everything else (wo_a, indexer.wq_b, q_norm/kv_norm, attn_sink, biases,
tid2eid, hc_*_base/hc_*_scale, e_proj/enorm internals...) passes through.

The rotation matrix and metadata are saved as
  <out>/rotation.safetensors  {global_rotation: (4096,4096) fp32}
  <out>/rotation_meta.json    {scheme, seed, dim, hc_mult}

Usage:
  python -m vptq.tools.deepseek_v4.rotate_ckpt \
      --ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
      --output-dir <out> --seed 0 [--layers 0:2] [--dry-run]
"""

import argparse
import json
import logging
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vptq.tools.quantize.hadamard import (
    block_diag, fold_gain, hadamard_matrix, left_rot, right_rot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.rotate_ckpt")

LAYER_RE = re.compile(r"(layers|mtp)\.(\d+)\.")
ATTN_INPUT_LEAVES = (
    "attn.wq_a.weight",
    "attn.wkv.weight",
    "attn.compressor.wkv.weight",
    "attn.compressor.wgate.weight",
    "attn.indexer.weights_proj.weight",
    # 内层 indexer.compressor 同样直接读残差 x（Indexer.forward 里
    # self.compressor(x, start_pos)），必须与外层 compressor 一样吸收
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.compressor.wgate.weight",
)
FFN_INPUT_LEAVES = (
    "ffn.gate.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w3.weight",
)


def is_expert_leaf(name: str, leaves) -> bool:
    return any(
        re.fullmatch(r"ffn\.experts\.\d+\." + re.escape(leaf), name)
        for leaf in leaves
    )


def rotate_ckpt(args):
    R = hadamard_matrix(args.dim, seed=args.seed)
    HC = args.hc_mult
    R_blk = block_diag(R, HC)

    with open(os.path.join(args.ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    os.makedirs(args.output_dir, exist_ok=True)

    # final norm gain (for head folding)
    with safe_open(
        os.path.join(args.ckpt, wmap["norm.weight"]), framework="pt",
        device="cpu",
    ) as f:
        g_final = f.get_tensor("norm.weight").float()

    shards = sorted(set(wmap.values()))
    lo, hi = args.layers
    stats = {"right": 0, "left": 0, "fold": 0, "norm1": 0, "hc": 0, "copy": 0}
    for shard in shards:
        dst = os.path.join(args.output_dir, shard)
        if os.path.exists(dst):
            logger.info("%s exists, skip (resume)", shard)
            continue
        out = {}
        prev = (
            stats["right"] + stats["left"] + stats["fold"]
            + stats["hc"] + stats["norm1"]
        )
        with safe_open(
            os.path.join(args.ckpt, shard), framework="pt", device="cpu"
        ) as f:
            names = list(f.keys())
            # per-layer gains needed by this shard
            gains = {}
            for n in names:
                m = LAYER_RE.match(n)
                if m:
                    gains.setdefault(m.group(1) + "." + m.group(2), {})
            for blk in gains:
                for kind in ("attn_norm.weight", "ffn_norm.weight"):
                    key = f"{blk}.{kind}"
                    if key in wmap:
                        with safe_open(
                            os.path.join(args.ckpt, wmap[key]),
                            framework="pt", device="cpu",
                        ) as gf:
                            gains[blk][kind] = gf.get_tensor(key).float()

            for name in names:
                t = f.get_tensor(name)
                m = LAYER_RE.match(name)
                blk = f"{m.group(1)}.{m.group(2)}" if m else None
                lidx = int(m.group(2)) if m else None
                in_range = m is None or (lo <= lidx < hi)
                leaf = (
                    name[len(blk) + 1:]
                    if blk and name.startswith(blk + ".")
                    else name
                )

                if not in_range:
                    out[name] = t
                    stats["copy"] += 1
                    continue

                if name == "embed.weight":
                    out[name] = right_rot(t, R); stats["right"] += 1
                elif name == "head.weight":
                    out[name] = right_rot(fold_gain(t, g_final), R)
                    stats["fold"] += 1; stats["right"] += 1
                elif name == "norm.weight":
                    out[name] = torch.ones_like(t); stats["norm1"] += 1
                elif leaf in ("attn_norm.weight", "ffn_norm.weight"):
                    out[name] = torch.ones_like(t); stats["norm1"] += 1
                elif leaf in ATTN_INPUT_LEAVES:
                    g = gains.get(blk, {}).get("attn_norm.weight")
                    out[name] = right_rot(fold_gain(t, g) if g is not None else t, R)
                    stats["fold"] += g is not None; stats["right"] += 1
                elif leaf == "attn.wo_b.weight":
                    out[name] = left_rot(t, R); stats["left"] += 1
                elif leaf in FFN_INPUT_LEAVES or is_expert_leaf(
                    leaf, ("w1.weight", "w3.weight")
                ):
                    g = gains.get(blk, {}).get("ffn_norm.weight")
                    out[name] = right_rot(fold_gain(t, g) if g is not None else t, R)
                    stats["fold"] += g is not None; stats["right"] += 1
                elif leaf == "ffn.shared_experts.w2.weight" or is_expert_leaf(
                    leaf, ("w2.weight",)
                ):
                    out[name] = left_rot(t, R); stats["left"] += 1
                elif leaf in (
                    "hc_attn_fn.weight", "hc_ffn_fn.weight", "hc_head_fn.weight"
                ) or leaf in ("hc_attn_fn", "hc_ffn_fn") or name == "hc_head_fn":
                    out[name] = right_rot(t, R_blk); stats["hc"] += 1
                elif blk and blk.startswith("mtp") and leaf in (
                    "e_proj.weight", "h_proj.weight"
                ):
                    out[name] = right_rot(t, R); stats["right"] += 1
                else:
                    out[name] = t
                    stats["copy"] += 1
        if args.dry_run:
            logger.info("[dry-run] %s would be written", shard)
            continue
        touched = (
            stats["right"] + stats["left"] + stats["fold"]
            + stats["hc"] + stats["norm1"]
        ) - prev
        if touched == 0 and not any(
            n in ("embed.weight", "head.weight", "norm.weight", "hc_head_fn")
            for n in names
        ):
            # 无旋转张量的分片直接链接/复制（省 I/O）
            src = os.path.join(args.ckpt, shard)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            logger.info("shard %s linked (no rotated tensors)", shard)
            continue
        save_file(out, dst + ".tmp", metadata={"format": "pt"})
        os.replace(dst + ".tmp", dst)
        logger.info("shard %s done", shard)

    # non-safetensors 文件与配置目录
    for f in os.listdir(args.ckpt):
        src = os.path.join(args.ckpt, f)
        dst = os.path.join(args.output_dir, f)
        if f.endswith(".safetensors"):
            continue
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src) and f in ("inference", "tokenizer"):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

    if not args.dry_run:
        save_file(
            {"global_rotation": R},
            os.path.join(args.output_dir, "rotation.safetensors"),
        )
    with open(os.path.join(args.output_dir, "rotation_meta.json"), "w") as f:
        json.dump(
            {"scheme": "hadamard-global-v1", "seed": args.seed,
             "dim": args.dim, "hc_mult": args.hc_mult,
             "layers_rotated": args.layers},
            f, indent=2,
        )
    logger.info("ROTATION DONE: %s", stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--hc-mult", type=int, default=4)
    p.add_argument("--layers", default="0:43", help="layer range to rotate")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.layers = tuple(int(x) for x in args.layers.split(":"))
    rotate_ckpt(args)


if __name__ == "__main__":
    main()
