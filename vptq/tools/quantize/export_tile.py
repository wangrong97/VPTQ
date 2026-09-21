# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Production export for the TILE quantization scheme (v2-k16-res8-fp8).

The development format (`QuantizedWeight`: uint16 indices + fp8 codebooks
per matrix) stores 4-bit indices in 16-bit slots — 4x waste. The production
bundle bit-packs indices into int32 streams (main and residual packed
separately since their vector lengths differ) and records everything the
Ascend inference kernel needs in a per-layer safetensors file plus a JSON
manifest.

Per weight the bundle holds:
  <name>.centroids      (T, k, v)      fp8_e4m3   main tile codebooks
  <name>.res_centroids  (T, k_res, v_r) fp8_e4m3  residual tile codebooks
  <name>.main_packed    (T, cols, w32)  int32     4-bit main indices
  <name>.res_packed     (T, cols, w32r) int32     4-bit residual indices
  <name>.shape          (2,)            int64     original (out, in)

Reconstruction (reference, mirrors what the kernel does):
  block = centroids[t][main_idx] + res_centroids[t][res_idx]
"""

import json
import logging
import os
from typing import Dict

import torch
from safetensors.torch import save_file

from vptq.tools.quantize.config import QuantConfig
from vptq.tools.quantize.vq import QuantizedWeight
from vptq.utils.pack import pack_index, unpack_index_tensor

logger = logging.getLogger("vptq.quantize.export_tile")


def pack_tile_weight(
    qw: QuantizedWeight, config: QuantConfig
) -> Dict[str, torch.Tensor]:
    """QuantizedWeight (tile layout) -> compact production tensors."""
    assert qw.res_centroids is not None, "tile scheme expects residual VQ"
    index_bits = config.index_bits
    res_bits = config.res_index_bits
    out = {}
    # (T, cols, nvec) -> per-tile packed int32 stream (nvec, w32)
    main_packed = []
    res_packed = []
    for t in range(qw.indices.shape[0]):
        ind = qw.indices[t].t().contiguous().unsqueeze(0)  # (1,nvec,cols)
        pk = pack_index(
            indice=ind, index_bits=index_bits, index_dtype=torch.uint16
        )
        main_packed.append(pk.squeeze(0))  # (nvec, w32)
        res = qw.res_indices[t].t().contiguous().unsqueeze(0)
        rp = pack_index(
            indice=res, index_bits=res_bits, index_dtype=torch.uint16
        )
        res_packed.append(rp.squeeze(0))
    out["centroids"] = qw.centroids
    out["res_centroids"] = qw.res_centroids
    out["main_packed"] = torch.stack(main_packed)
    out["res_packed"] = torch.stack(res_packed)
    return out


def unpack_tile_weight(
    bundle: Dict[str, torch.Tensor],
    group_cols: int,
    config: QuantConfig,
) -> QuantizedWeight:
    """Inverse of pack_tile_weight (for verification / reference dequant)."""
    index_bits = config.index_bits
    res_bits = config.res_index_bits
    indices, res_indices = [], []
    for t in range(bundle["main_packed"].shape[0]):
        pk = bundle["main_packed"][t].unsqueeze(0)  # (1,nvec,w32)
        rp = bundle["res_packed"][t].unsqueeze(0)
        ind, _ = unpack_index_tensor(
            pk, index_bits, group_cols, 0, group_cols
        )
        res, _ = unpack_index_tensor(
            rp, res_bits, group_cols, 0, group_cols
        )
        indices.append(ind.squeeze(0).t())  # (cols, nvec)
        res_indices.append(res.squeeze(0).t())
    return QuantizedWeight(
        centroids=bundle["centroids"],
        indices=torch.stack(indices),
        res_centroids=bundle["res_centroids"],
        res_indices=torch.stack(res_indices),
        proxy_error=float("nan"),
        mse=float("nan"),
    )


def reconstruct_tile_weight(
    centroids: torch.Tensor,
    indices: torch.Tensor,
    res_centroids: torch.Tensor,
    res_indices: torch.Tensor,
    group_num: int,
    row_tile: int,
    out_f: int,
) -> torch.Tensor:
    """Reference reconstruction from tile-layout tensors -> (out_f, in_f)."""
    T, k, v = centroids.shape
    n_row = T // group_num
    cols = indices.shape[1]
    in_f = cols * group_num
    Q = torch.zeros(n_row * row_tile, in_f)
    for gi in range(group_num):
        for t in range(n_row):
            tile = gi * n_row + t
            cb = centroids[tile].float()
            block = cb[indices[tile].long()]  # (cols, nvec, v)
            block = block.permute(1, 2, 0).reshape(-1, cols)
            if res_centroids is not None:
                rcb = res_centroids[tile].float()
                rv = rcb.shape[-1]
                rblock = rcb[res_indices[tile].long()]
                rblock = rblock.permute(1, 2, 0).reshape(-1, cols)
                block = block + rblock
            Q[t * row_tile : (t + 1) * row_tile,
              gi * cols : (gi + 1) * cols] = block
    return Q[:out_f]


def export_layer(
    layer_file: str,
    out_dir: str,
    config: QuantConfig,
    layer_idx: int,
    weight_shapes: Dict[str, tuple],
) -> dict:
    """Convert a merged quant_layer file into a production safetensors
    bundle + manifest entry. Returns the manifest dict for the layer."""
    quant = torch.load(layer_file, weights_only=False)
    tensors: Dict[str, torch.Tensor] = {}
    manifest = {"layer": layer_idx, "weights": {}}
    group_cols = None
    for name, qw in quant.items():
        packed = pack_tile_weight(qw, config)
        for suffix, t in packed.items():
            tensors[f"{name}.{suffix}"] = t
        T, cols, nvec = qw.indices.shape
        out_f, in_f = weight_shapes[name]
        group_cols = cols
        tensors[f"{name}.shape"] = torch.tensor([out_f, in_f])
        manifest["weights"][name] = {
            "tiles": T,
            "cols_per_group": cols,
            "nvec_main": nvec,
            "nvec_res": qw.res_indices.shape[-1],
            "out_features": out_f,
            "in_features": in_f,
            "proxy_error": qw.proxy_error,
            "mse": qw.mse,
        }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"layer_{layer_idx:04d}.safetensors")
    save_file(tensors, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    raw_mb = (
        sum(
            w["out_features"] * w["in_features"]
            for w in manifest["weights"].values()
        )
        * 2
        / 1e6
    )
    logger.info(
        "layer %d: %d weights -> %s (%.1f MB, %.2fx compression)",
        layer_idx, len(quant), out_path, size_mb, raw_mb / max(size_mb, 1e-9),
    )
    manifest["file"] = os.path.basename(out_path)
    manifest["size_mb"] = round(size_mb, 2)
    return manifest
