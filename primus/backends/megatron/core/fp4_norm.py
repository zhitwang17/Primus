# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Shared eager RMSNorm arithmetic for explicit FP4 comparison recipes."""


def comparison_rmsnorm(x, effective_gamma, epsilon):
    """Preserve the caller's effective-gamma dtype and return the input dtype.

    Callers retain their existing zero-centered ``weight + 1`` operation before
    entering this helper. Never downcast a FP32 norm Parameter to BF16 here.
    No parameters, buffers or random state are created.
    """
    import torch

    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
    return (normalized * effective_gamma.float()).to(x.dtype)
