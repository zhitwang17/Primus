# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Experimental dense GEMM consuming actual two-level FP4 packed operands.

E2M1 payload and the FP32 inner-grid/outer scales are decoded in registers,
rounded to BF16 and multiplied with BF16 matrix instructions. This avoids
materializing full dequantized matrices but is NOT native FP4 MFMA and is not
the default training backend. Only forward A[M,K] @ W[N,K].T is implemented.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp4_triton import _e2m1_value


@triton.jit
def _scaled_gemm(
    A,
    AS,
    AO,
    B,
    BS,
    BO,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AKP: tl.constexpr,
    BKP: tl.constexpr,
    B_TWO_D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    outer_a, outer_b = tl.load(AO), tl.load(BO)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        kvals = tile * BK + reduction
        adata = tl.load(
            A + rows[:, None] * (AKP // 2) + kvals[None, :] // 2,
            (rows[:, None] < M) & (kvals[None, :] < K),
            0,
        ).to(tl.int32)
        bdata = tl.load(
            B + cols[:, None] * (BKP // 2) + kvals[None, :] // 2,
            (cols[:, None] < N) & (kvals[None, :] < K),
            0,
        ).to(tl.int32)
        acode = (adata >> ((kvals[None, :] & 1) * 4)) & 15
        bcode = (bdata >> ((kvals[None, :] & 1) * 4)) & 15
        ascale = tl.load(
            AS + rows[:, None] * (AKP // 16) + kvals[None, :] // 16,
            (rows[:, None] < M) & (kvals[None, :] < K),
            0.0,
        )
        if B_TWO_D:
            brows = cols // 16
        else:
            brows = cols
        bscale = tl.load(
            BS + brows[:, None] * (BKP // 16) + kvals[None, :] // 16,
            (cols[:, None] < N) & (kvals[None, :] < K),
            0.0,
        )
        aval = _e2m1_value(acode & 7)
        bval = _e2m1_value(bcode & 7)
        aval = tl.where((acode & 8) != 0, -aval, aval) * (ascale * outer_a)
        bval = tl.where((bcode & 8) != 0, -bval, bval) * (bscale * outer_b)
        accumulator = tl.dot(aval.to(tl.bfloat16), tl.trans(bval.to(tl.bfloat16)), accumulator)
    tl.store(C + rows[:, None] * N + cols[None, :], accumulator, (rows[:, None] < M) & (cols[None, :] < N))


@torch.no_grad()
def scaled_gemm(a, b, *, out_dtype=torch.bfloat16):
    """Consume packed fwd input and weight tensors without an intermediate QDQ buffer.

    The numerical product uses BF16 decoded operands and FP32 accumulation.
    This prototype deliberately rejects unsupported directions/grouped layouts;
    callers must select it explicitly and must not infer native FP4 throughput.
    """
    a.validate()
    b.validate()
    if a.data.device.type != "cuda" or b.data.device != a.data.device:
        raise ValueError("Packed-scale GEMM requires operands on the same CUDA/ROCm device")
    if a.config != b.config:
        raise ValueError("Packed-scale GEMM requires identical numerical recipes")
    if len(a.layout.original_shape) != 2 or len(b.layout.original_shape) != 2:
        raise ValueError("Packed-scale GEMM currently supports 2-D dense matrices only")
    if a.layout.operand != a.config.operand("fwd", "input") or b.layout.operand != b.config.operand(
        "fwd", "weight"
    ):
        raise ValueError("Packed-scale GEMM currently supports forward input/weight operands only")
    if a.layout.group_offsets or b.layout.group_offsets:
        raise ValueError("Packed-scale GEMM does not implement grouped expert offsets")
    m, k = a.layout.original_shape
    n, wk = b.layout.original_shape
    if k != wk:
        raise ValueError("Packed-scale GEMM contracting dimensions differ")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Packed-scale GEMM outputs BF16 or FP32")
    out = torch.empty((m, n), device=a.data.device, dtype=out_dtype)
    if not m * n:
        return out
    if not k:
        return out.zero_()
    _scaled_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
        a.data,
        a.inner_scale,
        a.outer_scale,
        b.data,
        b.inner_scale,
        b.outer_scale,
        out,
        m,
        n,
        k,
        a.layout.padded_shape[-1],
        b.layout.padded_shape[-1],
        b.layout.operand.two_dimensional,
        16,
        32,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def capability_report():
    return {
        "backend": "experimental_packed_scale_bf16_dot",
        "payload": "E2M1 low-nibble-first",
        "scale_storage": "FP32 E4M3FN/UE5M3 grid plus FP32 outer",
        "matrix_operands": "BF16 decoded in registers",
        "accumulation": "FP32",
        "native_fp4_matrix_instructions": False,
        "directions": ["dense forward"],
        "training_default": False,
    }
