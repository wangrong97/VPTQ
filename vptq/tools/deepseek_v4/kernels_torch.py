# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pure-PyTorch replacements for the tilelang kernels used by DeepSeek-V4
native inference (`inference/kernel.py`), so the model runs on any device
(Ascend NPU, CPU) without a GPU compiler toolchain.

Covered (everything the BF16 code path touches):
  - act_quant         : block-wise FP8-E4M3 quantize+dequantize (QAT sim)
  - fp4_act_quant     : block-wise FP4-E2M1 quantize+dequantize (QAT sim)
  - sparse_attn       : top-k index-gathered attention with attention sink
  - hc_split_sinkhorn : Hyper-Connections mixing (sinkhorn normalization)
  - hadamard_transform: Walsh-Hadamard rotation (replaces the CUDA package)

fp8_gemm / fp4_gemm are only used for FP8/FP4 *weights*; the BF16 model
never calls them, so they raise if invoked.
"""

import math
from typing import Optional

import torch

_FP8_MAX = 448.0
_FP4_MAX = 6.0
# positive FP4-E2M1 grid (8 values incl. 0)
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_scale_pow2(amax: torch.Tensor, inv_max: float) -> torch.Tensor:
    """2^ceil(log2(amax * inv_max)), matching fast_round_scale bit tricks."""
    return torch.pow(2.0, torch.ceil(torch.log2(amax * inv_max)))


def _to_fp8_e4m3_and_back(x: torch.Tensor) -> torch.Tensor:
    """Round x to the FP8-E4M3 grid. Uses the native dtype cast when
    available, otherwise falls back to arithmetic emulation."""
    try:
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    except (RuntimeError, TypeError):
        pass
    # e4m3: 4 exp bits (bias 7), 3 mantissa bits; normals >= 2^-6,
    # subnormals down to 2^-9, max 448.
    sign = torch.sign(x)
    ax = x.abs().clamp(max=_FP8_MAX)
    out = torch.zeros_like(ax)
    # subnormal region: step 2^-9 below 2^-6
    sub = ax < 2**-6
    out = torch.where(sub, (ax * 2**9).round() * 2**-9, out)
    # normal region: mantissa step depends on exponent
    exp = torch.floor(torch.log2(ax.clamp(min=2**-6)))
    step = torch.pow(2.0, exp - 3)
    norm = (ax / step).round() * step
    out = torch.where(sub, out, norm.clamp(max=_FP8_MAX))
    return sign * out


def _to_fp4_e2m1_and_back(x: torch.Tensor) -> torch.Tensor:
    """Round x to the FP4-E2M1 grid {0,.5,1,1.5,2,3,4,6} x sign."""
    grid = _FP4_GRID.to(device=x.device, dtype=x.dtype)
    sign = torch.sign(x)
    ax = x.abs().clamp(max=_FP4_MAX)
    idx = torch.searchsorted(grid, ax.contiguous())
    idx = idx.clamp(max=grid.numel() - 1)
    lo = grid[idx - 1].where(idx > 0, grid[0])
    hi = grid[idx]
    pick_hi = (ax - lo) >= (hi - ax)
    nearest = torch.where(pick_hi, hi, lo)
    nearest = torch.where(idx == 0, grid[0], nearest)
    return sign * nearest


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8-E4M3 quantization along the last dim.

    inplace=True fuses quant+dequant back to the input dtype (QAT sim),
    which is the only mode the BF16 model uses. Otherwise returns
    (y_fp8, scales) like the tilelang version.
    """
    shape = x.shape
    N = shape[-1]
    assert N % block_size == 0
    z = x.detach().reshape(-1, N).float()
    blocks = z.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    if scale_fmt is not None:
        s = _round_scale_pow2(amax, 1.0 / _FP8_MAX)
    else:
        s = amax / _FP8_MAX
    q = (blocks / s).clamp(-_FP8_MAX, _FP8_MAX)
    if inplace:
        y = _to_fp8_e4m3_and_back(q) * s
        out = y.reshape(shape).to(x.dtype)
        x.copy_(out)
        return x
    y = q.reshape(shape).to(torch.float8_e4m3fn)
    return y, s.reshape(*shape[:-1], N // block_size).to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False
):
    """Block-wise FP4-E2M1 quantization (power-of-2 scales), QAT-sim mode."""
    shape = x.shape
    N = shape[-1]
    assert N % block_size == 0
    z = x.detach().reshape(-1, N).float()
    blocks = z.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=6 * 2**-126)
    s = _round_scale_pow2(amax, 1.0 / _FP4_MAX)
    q = (blocks / s).clamp(-_FP4_MAX, _FP4_MAX)
    if inplace:
        y = _to_fp4_e2m1_and_back(q) * s
        out = y.reshape(shape).to(x.dtype)
        x.copy_(out)
        return x
    raise NotImplementedError(
        "non-inplace fp4_act_quant is not needed for BF16 inference"
    )


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    s_chunk: int = 512,
) -> torch.Tensor:
    """Top-k index-gathered attention with attention sink.

    Args:
        q: (b, s, h, d)
        kv: (b, n, d) — shared latent KV (MLA style, single "head")
        attn_sink: (h,) fp32 — extra logit added to the softmax
            denominator only.
        topk_idxs: (b, s, t) int — indices into dim 1 of kv; -1 = masked.
        softmax_scale: score scale.
        s_chunk: chunk size along s to bound memory.

    Matches sparse_attn_kernel: scores in fp32, probabilities cast back
    to kv dtype before the value matmul.
    """
    b, s, h, d = q.shape
    topk_idxs = topk_idxs.to(device=kv.device)
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    for s0 in range(0, s, s_chunk):
        s1 = min(s0 + s_chunk, s)
        idx = topk_idxs[:, s0:s1]  # (b, sc, t)
        valid = idx != -1
        safe = idx.clamp(min=0)
        kv_sel = kv[
            torch.arange(b, device=kv.device).view(b, 1, 1), safe
        ]  # (b, sc, t, d)
        scores = torch.einsum(
            "bshd,bstd->bsht", q[:, s0:s1].float(), kv_sel.float()
        )
        scores = scores * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        m = torch.maximum(scores.amax(dim=-1), sink)  # (b, sc, h)
        p = torch.exp(scores - m.unsqueeze(-1))
        denom = p.sum(dim=-1) + torch.exp(sink - m)  # (b, sc, h)
        p = (p / denom.unsqueeze(-1)).to(kv.dtype)
        out[:, s0:s1] = torch.einsum("bsht,bstd->bshd", p, kv_sel)
    return out


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Hyper-Connections mixing weights (see hc_split_sinkhorn_kernel).

    mixes: (..., (2 + hc_mult) * hc_mult) fp32
    returns (pre, post, comb) of shapes (..., hc), (..., hc),
    (..., hc, hc).
    """
    hc = hc_mult
    pre = (
        torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    )
    post = 2 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = mixes[..., 2 * hc :].unflatten(-1, (hc, hc)) * hc_scale[
        2
    ] + hc_base[2 * hc :].unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """ Walsh-Hadamard transform along the last dim (butterfly, O(d log d)).

    Replaces the fast_hadamard_transform CUDA package. The last dimension
    must be a power of two (128/512 in DeepSeek-V4).
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, "hadamard dim must be a power of two"
    y = x.float()
    h = 1
    while h < d:
        y = y.unflatten(-1, (d // (2 * h), 2, h))
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).flatten(-3)
        h *= 2
    return (y * scale).to(x.dtype)


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("fp8_gemm is unused with BF16 weights")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("fp4_gemm is unused with BF16 weights")
