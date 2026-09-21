# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compute inverse Hessians for already-collected layer_XXXX.pt files.

Reads each layer file produced by the collector, computes the damped
Cholesky inverse per group (float64 internally), and writes it back
into the same file as `inv_hessian` (float32). Runs group-by-group so
peak memory stays at one layer file plus one inverse at a time.

Usage:
    python -m vptq.tools.hessian.compute_inverse \
        --input-dir /path/to/hessians --device npu --damp 0.01
"""

import argparse
import glob
import logging
import os

import torch

from vptq.tools.hessian.collector import damped_inverse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.hessian.inv")


def compute_file_inverse(path: str, device: torch.device, damp: float):
    try:
        payload = torch.load(path, weights_only=False)
    except Exception as err:
        logger.warning("cannot load %s (%s); skipping", path, err)
        return 0
    n_done = 0
    for group_key, entry in payload.items():
        if "inv_hessian" in entry:
            continue
        hessian = entry["hessian"]
        if not torch.isfinite(hessian).all():
            logger.warning(
                "%s/%s has non-finite Hessian; leaving inv_hessian "
                "absent (re-collect this group)",
                path,
                group_key,
            )
            continue
        try:
            inv = damped_inverse(hessian.to(device), damp=damp).cpu()
        except Exception as err:
            logger.warning(
                "%s/%s failed on %s (%s); retrying on CPU",
                path,
                group_key,
                device,
                err,
            )
            inv = damped_inverse(hessian, damp=damp)
        entry["inv_hessian"] = inv
        n_done += 1
        if n_done % 64 == 0:
            logger.info("%s: %d/%d groups", path, n_done, len(payload))
    torch.save(payload, path)
    return n_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument(
        "--layer-range",
        default=None,
        help="Only process layers start:end (for parallel sharding).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    files = sorted(glob.glob(os.path.join(args.input_dir, "layer_*.pt")))
    if args.layer_range:
        start, end = (int(x) for x in args.layer_range.split(":"))
        files = [
            f
            for f in files
            if start <= int(os.path.basename(f)[6:10]) < end
        ]
    logger.info("processing %d files on %s", len(files), device)
    for path in files:
        n = compute_file_inverse(path, device, args.damp)
        logger.info("done %s (%d inverses)", path, n)
    logger.info("all done")


if __name__ == "__main__":
    main()
