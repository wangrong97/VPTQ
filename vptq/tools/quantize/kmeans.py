# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Hessian-weighted K-means for VPTQ codebook initialization (§3.2.1).

The layer-wise proxy error decomposes as

    tr(ΔWᵀ ΔW ⊙ H) = Σ_j h_jj ‖ΔW:,j‖² + cross terms (≈ diagonal),

so the codebook should minimize the *weighted* distortion
Σ_d h_d (v_d − c_d)² with per-dimension weights h from the Hessian
diagonal. Assignment uses weighted distances; the centroid update is the
plain cluster mean (per-dimension weights cancel).
"""

import logging

import torch

logger = logging.getLogger("vptq.quantize.kmeans")


@torch.no_grad()
def weighted_distances(
    vectors: torch.Tensor,
    centroids: torch.Tensor,
    weights: torch.Tensor,
    chunk: int = 65536,
) -> torch.Tensor:
    """argmin_k Σ_d w_d (v_d − c_d)² for each row of `vectors`.

    ‖v−c‖²_w = (v²·w)·1 + w·(c²)ᵀ − 2(v∘w)·cᵀ — computed as GEMMs,
    chunked over rows to bound memory. `weights` may be (v,) shared or
    (n, v) per-vector.
    """
    n, v = vectors.shape
    k = centroids.shape[0]
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    if weights.shape[0] == 1:
        a = (vectors.square() * weights).sum(-1)  # (n,)
        b = weights @ centroids.square().t()  # (1, k)
        out = torch.empty(n, dtype=torch.long, device=vectors.device)
        for s in range(0, n, chunk):
            cross = (vectors[s : s + chunk] * weights) @ centroids.t()
            out[s : s + chunk] = (
                a[s : s + chunk].unsqueeze(1) + b - 2 * cross
            ).argmin(dim=-1)
        return out
    # per-vector weights
    out = torch.empty(n, dtype=torch.long, device=vectors.device)
    c2 = centroids.square()
    for s in range(0, n, chunk):
        vs = vectors[s : s + chunk]
        ws = weights[s : s + chunk]
        a = (vs.square() * ws).sum(-1, keepdim=True)  # (m, 1)
        b = ws @ c2.t()  # (m, k)
        cross = (vs * ws) @ centroids.t()  # (m, k)
        out[s : s + chunk] = (a + b - 2 * cross).argmin(dim=-1)
    return out


@torch.no_grad()
def weighted_kmeans(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    iters: int = 10,
    sample: int = 262144,
    seed: int = 0,
) -> torch.Tensor:
    """Lloyd's algorithm with Hessian-weighted assignment.

    Args:
        vectors: (n, v) training vectors (fp32).
        weights: (v,) or (n, v) non-negative per-dimension weights
            (Hessian diagonal of the owning input channel).
        k: number of centroids.
        iters: Lloyd iterations.
        sample: subsample size per iteration (n >> sample for big
            codebooks); the full set is used when smaller.
        seed: RNG seed.

    Returns:
        (k, v) fp32 centroids.
    """
    n, v = vectors.shape
    device = vectors.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)

    # init: k distinct random training vectors
    perm = torch.randperm(n, generator=g)[:k].to(device)
    centroids = vectors[perm].clone()

    for it in range(iters):
        if n > sample:
            sel = torch.randperm(n, generator=g)[:sample].to(device)
            vs, ws = vectors[sel], weights[sel] if weights.shape[0] > 1 else weights
        else:
            vs, ws = vectors, weights
        assign = weighted_distances(vs, centroids, ws)
        # M step: plain mean per cluster (per-dim weights cancel)
        new = torch.zeros_like(centroids)
        count = torch.zeros(k, dtype=torch.float32, device=device)
        new.index_add_(0, assign, vs)
        count.index_add_(0, assign, torch.ones_like(count))
        empty = count == 0
        new[empty] = centroids[empty]  # keep empty clusters in place
        centroids = new / count.clamp(min=1).unsqueeze(-1)
        logger.debug(
            "kmeans iter %d: %d empty clusters", it, int(empty.sum())
        )
    return centroids
