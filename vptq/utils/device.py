# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Device abstraction for CUDA / Ascend NPU / CPU.

VPTQ historically assumed CUDA (hardcoded `.to("cuda")` and
`torch.cuda.empty_cache()`). This module centralizes device selection so
the same code runs on NVIDIA GPUs (CUDA), Ascend NPUs (torch_npu) and
plain CPU.
"""

import functools
import logging

import torch

logger = logging.getLogger("vptq.device")

_IS_NPU_AVAILABLE = None


def is_npu_available() -> bool:
    """Whether torch_npu is installed and an NPU is visible."""
    global _IS_NPU_AVAILABLE
    if _IS_NPU_AVAILABLE is None:
        try:
            import torch_npu  # noqa: F401

            _IS_NPU_AVAILABLE = bool(
                hasattr(torch, "npu") and torch.npu.is_available()
            )
        except ImportError:
            _IS_NPU_AVAILABLE = False
    return _IS_NPU_AVAILABLE


def is_accelerator_available() -> bool:
    return torch.cuda.is_available() or is_npu_available()


@functools.lru_cache(maxsize=1)
def default_device() -> torch.device:
    """Best available compute device: CUDA > NPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_npu_available():
        return torch.device("npu")
    return torch.device("cpu")


def empty_cache() -> None:
    """Release cached allocator memory on whichever accelerator exists."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()


def device_of(module: torch.nn.Module) -> torch.device:
    """Device of a module's first parameter/buffer (CPU if none)."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")
