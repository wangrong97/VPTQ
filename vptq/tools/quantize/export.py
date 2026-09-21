# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Export QuantizedWeight results into VQuantLinear layers (and back).

Layout note: vq.QuantizedWeight stores indices as
(group_num, cols_in_group, nvec) while VQuantLinear expects per-codebook
(num_indices, group_size) — the transpose happens here.
"""

import torch

from vptq.layers.vqlinear import VQuantLinear
from vptq.tools.quantize.config import QuantConfig
from vptq.tools.quantize.vq import QuantizedWeight


def build_vqlinear(
    qw: QuantizedWeight,
    in_features: int,
    out_features: int,
    config: QuantConfig,
    device=None,
    dtype=torch.float16,
) -> VQuantLinear:
    """Instantiate a VQuantLinear and fill it with a QuantizedWeight.

    Indices are stored unpacked (int16); bit-packing is a separate
    offline step (vptq.utils.pack.pack_index). Perm/norm/outliers are
    disabled (not produced by vq.vptq_quantize).
    """
    group_num = qw.centroids.shape[0]
    layer = VQuantLinear(
        in_features=in_features,
        out_features=out_features,
        vector_lens=(-1, config.vector_len),
        num_centroids=(-1, config.num_centroids),
        num_res_centroids=(-1, config.num_res_centroids),
        group_num=group_num,
        group_size=in_features // group_num,
        outlier_size=0,
        indices_as_float=False,
        is_indice_packed=False,
        enable_norm=False,
        enable_perm=False,
        enable_proxy_error=False,
        device=device,
        dtype=dtype,
    )
    # (g, cols, nvec) -> {g+1: (nvec, cols)} for init_parameters.
    # init_parameters unconditionally skips dict key 0 (the outlier
    # slot), so a placeholder must be present even without outliers.
    indices = {0: torch.zeros(1, 1, dtype=torch.int32)}
    indices.update(
        {
            g + 1: qw.indices[g].t().contiguous().to(torch.int32)
            for g in range(group_num)
        }
    )
    centroids = {0: torch.zeros(1, config.vector_len, dtype=dtype)}
    centroids.update(
        {g + 1: qw.centroids[g].to(dtype) for g in range(group_num)}
    )
    res_centroids = res_indices = None
    if qw.res_centroids is not None:
        res_centroids = {0: torch.zeros(1, config.vector_len, dtype=dtype)}
        res_centroids.update(
            {g + 1: qw.res_centroids[g].to(dtype) for g in range(group_num)}
        )
        res_indices = {0: torch.zeros(1, 1, dtype=torch.int32)}
        res_indices.update(
            {
                g + 1: qw.res_indices[g].t().contiguous().to(torch.int32)
                for g in range(group_num)
            }
        )
    layer.init_parameters(
        centroids=centroids,
        indices=indices,
        res_centroids=res_centroids,
        res_indices=res_indices,
        weight_scale=None,
        weight_bias=None,
        perm=None,
    )
    return layer


@torch.no_grad()
def reconstruct_weight(
    qw: QuantizedWeight, out_features: int
) -> torch.Tensor:
    """Reconstruct Ŵ (out, in) fp32 from a QuantizedWeight:
    main codebook lookup + residual add, layout
    (g, cols, nvec) -> (out, in), row padding stripped."""
    group_num, cols, nvec = qw.indices.shape
    v = qw.centroids.shape[-1]
    q = (
        qw.centroids.float()[
            torch.arange(group_num).unsqueeze(-1).unsqueeze(-1),
            qw.indices.long(),
        ]
        .permute(0, 2, 1, 3)  # (g, nvec, cols, v)
        .permute(0, 1, 3, 2)  # (g, nvec, v, cols)
        .reshape(group_num, nvec * v, cols)
        .permute(1, 0, 2)
        .reshape(nvec * v, group_num * cols)
    )
    if qw.res_centroids is not None:
        r = (
            qw.res_centroids.float()[
                torch.arange(group_num).unsqueeze(-1).unsqueeze(-1),
                qw.res_indices.long(),
            ]
            .permute(0, 2, 1, 3)
            .permute(0, 1, 3, 2)
            .reshape(group_num, nvec * v, cols)
            .permute(1, 0, 2)
            .reshape(nvec * v, group_num * cols)
        )
        q = q + r
    return q[:out_features]
