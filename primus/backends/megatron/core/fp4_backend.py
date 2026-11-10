# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Explicit quantizer selection; both implementations obey the FP4 contract."""

from . import fp4_reference


def quantizer(config):
    if config.quantizer == "torch":
        return fp4_reference
    if config.quantizer == "triton":
        from . import fp4_triton

        return fp4_triton
    raise ValueError(f"Unsupported FP4 quantizer: {config.quantizer}")


def qdq(x, config, operand, **kwargs):
    return quantizer(config).qdq(x, config, operand, **kwargs)


def compute_outer_scale(x, config):
    return quantizer(config).compute_outer_scale(x, config)
