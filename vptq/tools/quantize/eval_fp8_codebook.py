# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Evaluate FP8 codebook (centroid) variants for VPTQ, using collected
Hessians as the proxy-error metric — no model inference needed.

Simulates a simplified VPTQ quantizer with TILE grouping: the weight
matrix W (out, in) is split into column tiles of `tile_size` in-features;
each tile owns its codebook(s). Vectors are length-`vector_len` segments
along out_features (VPTQ's vector_quant_dim="out" convention). K-means is
Hessian-diagonal weighted (each vector inherits h_ii of its in-column).

Configs compared (proxy error = tr(D^T D H) / tr(W^T W H), D = What - W):
  fp16              : centroids in fp16 (baseline)
  fp8               : centroids rounded to fp8-e4m3 after training
  fp8+norm          : per-channel normalization before K-means, fp8
                      centroids (unnormalized after reconstruction)
  fp8+res           : fp8 main centroids + fp16 residual codebook whose
                      residual is computed AFTER fp8 rounding (absorbs
                      the fp8 error)
  fp8-aware         : fp8 rounding inside every Lloyd iteration
  fp16+res          : fp16 main + fp16 residual (reference)

Example:
    python -m vptq.tools.quantize.eval_fp8_codebook \
        --ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
        --tensor layers.2.ffn.experts.0.w1.weight \
        --hessian-dir /mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-rpmix \
        --layer 2 --group ffn.experts.0.mlp_in \
        --vector-len 16 --num-centroids 65536 --res-centroids 4096 \
        --tile-size 512 --device npu:3
"""

import argparse
import json
import logging
import math
import os

import torch
from safetensors import safe_open

from vptq.tools.deepseek_v4.kernels_torch import _to_fp8_e4m3_and_back

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.eval_fp8")


def fp8_round(x: torch.Tensor) -> torch.Tensor:
    return _to_fp8_e4m3_and_back(x)


def load_weight(ckpt: str, name: str) -> torch.Tensor:
    index = json.load(
        open(os.path.join(ckpt, "model.safetensors.index.json"))
    )["weight_map"]
    shard = index[name]
    with safe_open(
        os.path.join(ckpt, shard), framework="pt", device="cpu"
    ) as f:
        return f.get_tensor(name)


def load_hessian(hessian_dir: str, layer: int, group: str) -> torch.Tensor:
    payload = torch.load(
        os.path.join(hessian_dir, f"layer_{layer:04d}.pt"),
        weights_only=False,
    )
    return payload[group]["hessian"]


@torch.no_grad()
def assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 16384):
    """argmin ||x - c||^2 via ||x||^2 - 2 x.c + ||c||^2, chunked."""
    c_norm = (C * C).sum(-1)
    idx = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        x = X[s : s + chunk]
        d = (x * x).sum(-1, keepdim=True) - 2 * (x @ C.t()) + c_norm
        idx[s : s + chunk] = d.argmin(-1)
        del x, d
    return idx


@torch.no_grad()
def kmeans(
    X: torch.Tensor,
    weights: torch.Tensor,
    K: int,
    iters: int,
    fp8_in_loop: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Weighted K-means (Lloyd). Returns centroids (K, v)."""
    n, v = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = min(n, max(4 * K, 65536))
    perm = torch.randperm(n, generator=g)[:sub].to(X.device)
    Xs = X[perm]
    if K > 1024:
        # kmeans++ is O(K^2 * n * v) — prohibitive for large K; random
        # init is standard practice for VPTQ-scale codebooks
        centroids = Xs[
            torch.randperm(sub, generator=g).to(X.device)[:K]
        ].clone()
    else:
        centroids = Xs[
            torch.randperm(sub, generator=g).to(X.device)[:K]
        ].clone()
        for i in range(1, K):
            d = torch.cdist(Xs, centroids[:i]).min(-1).values
            probs = (d * d).clamp(min=1e-12)
            centroids[i] = Xs[
                torch.multinomial(probs, 1, generator=g).to(X.device)
            ]
    for it in range(iters):
        idx = assign(X, centroids)
        new = torch.zeros_like(centroids)
        wsum = torch.zeros(K, device=X.device)
        new.index_add_(0, idx, X * weights.unsqueeze(-1))
        wsum.index_add_(0, idx, weights)
        alive = wsum > 0
        new[alive] = new[alive] / wsum[alive].unsqueeze(-1)
        # dead centroids keep old values
        centroids = torch.where(alive.unsqueeze(-1), new, centroids)
        if fp8_in_loop:
            centroids = fp8_round(centroids)
        logger.info("  iter %d/%d done", it + 1, iters)
    return centroids


@torch.no_grad()
def quantize_tile(
    X: torch.Tensor,
    w: torch.Tensor,
    K: int,
    Kr: int,
    iters: int,
    centroid_dtype: str,
    fp8_aware: bool,
    res_absorb: bool,
    seed: int,
):
    """Quantize one tile of vectors. Returns reconstruction of X."""
    main = kmeans(X, w, K, iters, fp8_in_loop=fp8_aware, seed=seed)
    if centroid_dtype == "fp8" and not fp8_aware:
        main_q = fp8_round(main)
    elif centroid_dtype == "fp8":
        main_q = main  # already rounded in loop
    else:
        main_q = main.half().float()
    idx = assign(X, main_q)
    rec = main_q[idx]
    if Kr > 0:
        # residual computed AFTER main-centroid rounding: the residual
        # stage absorbs fp8 storage error (res_absorb=True) or is computed
        # against the unrounded centroids (False, ablation)
        base = main_q[idx] if res_absorb else main[idx]
        R = X - base
        res = kmeans(R, w, Kr, iters, seed=seed + 1)
        res = res.half().float()  # residual centroids stay fp16
        ridx = assign(R, res)
        rec = rec + res[ridx]
    return rec


@torch.no_grad()
def evaluate(
    W: torch.Tensor,
    H: torch.Tensor,
    vector_len: int,
    K: int,
    Kr: int,
    tile_size: int,
    iters: int,
    use_norm: bool,
    centroid_dtype: str,
    fp8_aware: bool,
    res_absorb: bool,
    device: torch.device,
    seed: int = 0,
) -> float:
    """Relative Hessian proxy error of one config. W: (out, in)."""
    out_f, in_f = W.shape
    pad = (-out_f) % vector_len
    W = W.to(device).float()
    H = H.to(device).float()
    if pad:
        W = torch.nn.functional.pad(W, (0, 0, 0, pad))
    scale = torch.ones(in_f, device=device)
    bias = torch.zeros(in_f, device=device)
    Wn = W
    if use_norm:
        # per-in-channel standardization (weights shared per column)
        bias = W.mean(dim=0)
        scale = W.var(dim=0).sqrt().clamp(min=1e-6)
        Wn = (W - bias) / scale
    hdiag = H.diag().clamp(min=0)
    rec = torch.empty_like(Wn)
    if tile_size <= 0:
        tile_size = in_f
    n_tiles = (in_f + tile_size - 1) // tile_size
    for t in range(n_tiles):
        c0, c1 = t * tile_size, min((t + 1) * tile_size, in_f)
        # vectors along out-dim: (cols, out) -> (cols*out/v, v)
        Xt = Wn[:, c0:c1].t().reshape(-1, vector_len)
        wt = hdiag[c0:c1].repeat_interleave(Xt.shape[0] // (c1 - c0))
        logger.info(
            "tile %d/%d: %d vectors, K=%d Kr=%d (ratio %.1f:1)",
            t + 1, n_tiles, Xt.shape[0], K, Kr,
            Xt.shape[0] / max(K, 1),
        )
        rec[:, c0:c1] = (
            quantize_tile(
                Xt, wt, K, Kr, iters, centroid_dtype, fp8_aware,
                res_absorb, seed + t,
            )
            .reshape(c1 - c0, -1)
            .t()
        )
        del Xt, wt
    del Wn
    What = rec * scale + bias
    del rec
    if pad:
        What = What[:-pad]
        W = W[:-pad]
    D = What - W
    err = torch.sum(D.t() @ D * H) / torch.sum(W.t() @ W * H)
    return err.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--hessian-dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--vector-len", type=int, default=16)
    p.add_argument("--num-centroids", type=int, default=65536)
    p.add_argument("--res-centroids", type=int, default=4096)
    p.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="In-features per tile (one codebook each). 0 = whole "
        "matrix in one tile (VPTQ group semantics). Keep vectors/"
        "centroid >= ~8 or quantization is degenerate.",
    )
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--configs",
        default="fp16,fp8,fp8+norm,fp8+res,fp8-aware,fp16+res",
    )
    args = p.parse_args()
    device = torch.device(args.device)

    W = load_weight(args.ckpt, args.tensor)
    H = load_hessian(args.hessian_dir, args.layer, args.group)
    logger.info(
        "W %s, H %s, tensor=%s group=%s",
        tuple(W.shape), tuple(H.shape), args.tensor, args.group,
    )

    table = {}
    for cfg in args.configs.split(","):
        use_norm = "+norm" in cfg
        fp8_aware = cfg.endswith("aware")
        with_res = cfg.endswith("+res") or "+res" in cfg
        dtype = "fp8" if cfg.startswith("fp8") else "fp16"
        Kr = args.res_centroids if with_res else 0
        logger.info("=== config %s (norm=%s aware=%s Kr=%d dtype=%s)",
                    cfg, use_norm, fp8_aware, Kr, dtype)
        err = evaluate(
            W, H, args.vector_len, args.num_centroids, Kr,
            args.tile_size, args.iters, use_norm, dtype, fp8_aware,
            res_absorb=True, device=device, seed=args.seed,
        )
        table[cfg] = err
        logger.info("=== %s: rel proxy error = %.6f", cfg, err)
        from vptq.utils.device import empty_cache

        empty_cache()

    print("\n==== RESULTS (relative Hessian proxy error, lower=better) ====")
    print(f"tensor={args.tensor}  group={args.group}")
    print(f"v{args.vector_len}-k{args.num_centroids}-{args.res_centroids}"
          f" tile={args.tile_size} iters={args.iters}")
    base = table.get("fp16")
    for cfg, err in table.items():
        ratio = f"{err / base:.3f}x" if base else ""
        print(f"  {cfg:12s} {err:.6f}   ({ratio} vs fp16)")


if __name__ == "__main__":
    main()
