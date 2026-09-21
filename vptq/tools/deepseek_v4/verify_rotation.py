# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end invariance check for Hadamard checkpoint absorption.

Rotates a prefix of the checkpoint (default layers 0-1) into a temp dir,
builds the vendored model from BOTH checkpoints with n_layer_limit, and
verifies: hidden_rotated ≈ hidden_original @ R after the rotated layers.

Usage:
  python -m vptq.tools.deepseek_v4.verify_rotation \
      --ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
      --work-dir /tmp/rot_verify --layers 0:2 --seed 0
"""

import argparse
import logging
import os

import torch

from vptq.tools.deepseek_v4.loader import build_model, model_args_from_config
from vptq.tools.quantize.hadamard import hadamard_matrix

logger = logging.getLogger("vptq.verify_rotation")


@torch.inference_mode()
def hidden_after_layers(ckpt: str, n_layers: int, ids: torch.Tensor,
                        device: str) -> torch.Tensor:
    """Run embed + first n_layers blocks of the vendored model on CPU."""
    model = build_model(ckpt, max_batch_size=ids.shape[0],
                        max_seq_len=ids.shape[1] + 8, device=device,
                        strict=True, n_layer_limit=n_layers)
    h = model.embed(ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    for i in range(n_layers):
        h = model.layers[i](h, 0, ids)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--work-dir", default="/tmp/rot_verify")
    p.add_argument("--layers", default="0:2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=4096)
    args = p.parse_args()
    lo, hi = (int(x) for x in args.layers.split(":"))
    n_layers = hi

    # 1) rotate the prefix (idempotent; resume-safe)
    from vptq.tools.deepseek_v4.rotate_ckpt import rotate_ckpt

    class _A:
        pass
    a = _A()
    a.ckpt, a.output_dir = args.ckpt, args.work_dir
    a.seed, a.dim, a.hc_mult = args.seed, args.dim, 4
    a.layers, a.dry_run = (lo, hi), False
    rotate_ckpt(a)

    # 2) compare hidden states
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 50000, (2, 64), generator=g)
    h0 = hidden_after_layers(args.ckpt, n_layers, ids, "cpu")
    h1 = hidden_after_layers(args.work_dir, n_layers, ids, "cpu")
    R = hadamard_matrix(args.dim, seed=args.seed)
    expected = h0.float() @ R
    got = h1.float()
    err = (got - expected).abs().max().item()
    rel = err / expected.abs().max().item()
    print(f"hidden after {n_layers} layers: max|diff|={err:.4e} "
          f"(rel {rel:.3e})")
    # bf16 精度下限：旋转权重经 fp32→bf16 转换，单层相对误差 ~1e-2 量级
    # （roundtrip 抽查：wq_a 3.4e-3、w2 1.7e-2），2 层累计 ~2e-2 属精度噪声
    print("INVARIANCE:", "PASS" if rel < 3e-2 else "FAIL")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
