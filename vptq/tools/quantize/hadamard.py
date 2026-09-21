# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Random Hadamard rotation utilities (QuaRot-style absorption).

The residual stream of the model is re-based by an orthogonal matrix R
(hidden := hidden @ R). All weights are then folded so the model is
bit-exact equivalent while the quantizer sees rotated (Gaussianized)
weights:

  - consumers of the residual stream (wq_a/wkv/compressor/router/
    w1/w3/shared): W -> W @ R        (input side, use H' = R^T H R)
  - producers (wo_b, down w2):       W -> R^T @ W    (output side, H unchanged)
  - RMSNorm gains: folded into the consumers (W -> W @ diag(g)), set to 1
  - hc (hyper-connection) mixing matrices: hc_fn -> hc_fn @ R_blk
    (R_blk = blockdiag over the hc streams)
  - embed: E -> E @ R;  head: W -> W @ R

R for this model is a single 4096x4096 signed Hadamard.
"""

import torch


def hadamard_matrix(n: int, seed: int = 0,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Signed normalized Hadamard matrix of order n (n must be a power of 2).

    R = H · diag(±1) / sqrt(n), so R R^T = I and R = R^T.
    """
    assert n & (n - 1) == 0, f"n must be a power of 2, got {n}"
    g = torch.Generator().manual_seed(seed)
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    signs = (torch.rand(n, generator=g) > 0.5).to(torch.float64) * 2 - 1
    R = H * signs.unsqueeze(1) / (n ** 0.5)
    return R.to(dtype)


def right_rot(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Input-side absorption: W -> W @ R (weights stored as (out, in))."""
    return (W.to(R.dtype) @ R).to(W.dtype)


def left_rot(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Output-side rotation: W -> R^T @ W (weights stored as (out, in))."""
    return (R.t() @ W.to(R.dtype)).to(W.dtype)


def fold_gain(W: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Fold an RMSNorm gain into a following linear's input side:
    (x ⊙ g) @ Wᵀ == x @ (W @ diag(g))ᵀ  =>  W -> W @ diag(g).
    """
    return (W.to(torch.float32) * gain.to(torch.float32).unsqueeze(0)).to(W.dtype)


def rot_hessian(H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate a Hessian for an input-side-rotated linear: H' = R^T H R."""
    return R.t() @ H.to(R.dtype) @ R


def block_diag(R: torch.Tensor, times: int) -> torch.Tensor:
    """blockdiag(R, ..., R) `times` along the diagonal (for hc streams)."""
    return torch.block_diag(*([R] * times))


_WRAP_R_CACHE: dict = {}


def wrap_rotation(dim: int, seed: int = 0) -> torch.Tensor:
    """Per-input-dim rotation for wrap mode (W -> W @ R, Q(WR) @ R^T).

    Wrap mode is self-contained per matrix, so each input dimension gets
    its own signed Hadamard (residual-stream weights: 4096; expert w2
    reading the 2048-d SwiGLU intermediate: 2048). dim=4096, seed=0 is
    bit-identical to rotate_ckpt's global_rotation. Cached per dim.
    """
    key = (dim, seed)
    if key not in _WRAP_R_CACHE:
        _WRAP_R_CACHE[key] = hadamard_matrix(dim, seed=seed)
    return _WRAP_R_CACHE[key]
