# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.metadata

from vptq.layers import AutoModelForCausalLM, VQuantLinear

try:
    __version__ = importlib.metadata.version("vptq")
except importlib.metadata.PackageNotFoundError:
    # running from a source checkout without installation
    __version__ = "0.0.0.dev0"

__all__ = [
    "AutoModelForCausalLM",
    "VQuantLinear",
]
