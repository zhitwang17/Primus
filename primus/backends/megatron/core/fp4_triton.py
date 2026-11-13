# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Triton packing and fused QDQ for the two-level FP4 reference contract.

This backend consumes E4M3FN/UE5M3 grid values and a separate FP32 outer scale.
It never passes these scales to an MXFP4/E8M0 matrix instruction. ``qdq`` fuses
block amax, scale rounding, E2M1 rounding and reconstruction without writing an
intermediate packed payload; ``quantize``/``dequantize`` expose the actual packed
contract for parity checks and future consumers. GEMMs remain a separate choice.

By default SR uses the reference's versioned CPU generator and identical padded
logical shape. Explicit sr_rng=triton_philox instead generates Philox4x32-10
uniforms inside the QDQ kernel, without a host random array or a uniform tensor.
RHT remains the paired reference operation. Public value checks are enabled by
default; ``check_values=False`` is an explicit finite-input profiling/internal
contract, not a different numerical recipe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp4_contract import FP4TensorLayout, TwoLevelFP4Tensor
from .fp4_reference import _check_random, _generator
from .fp4_rng import reserve_philox_sr


@triton.jit
def _philox_uniform(seed, block_offset, logical_index):
    # randint4x splits the uint64 offset into two uint32 counter words.
    block = block_offset.to(tl.uint64) + logical_index.to(tl.uint64) // 4
    a, b, c, d = tl.randint4x(seed, block, n_rounds=10)
    lane = logical_index % 4
    word = tl.where(lane == 0, a, tl.where(lane == 1, b, tl.where(lane == 2, c, d)))
    return (word >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _input_offsets(
    index, ROWS: tl.constexpr, COLS: tl.constexpr, SB: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr
):
    batch = index // (ROWS * COLS)
    row = index // COLS % ROWS
    col = index % COLS
    return batch * SB + row * SR + col * SC


@triton.jit
def _amax_input(
    X,
    PARTIAL,
    ERROR,
    N: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    CHECK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + _input_offsets(idx, R, C, SB, SR, SC), idx < N, 0).to(tl.float32)
    finite = tl.abs(value) < float("inf")
    if CHECK:
        bad = tl.sum(((idx < N) & ~finite).to(tl.int32), 0)
        if bad:
            tl.atomic_or(ERROR, 1)
    magnitude = tl.where(finite, tl.abs(value), 0.0)
    tl.store(PARTIAL + tl.program_id(0), tl.max(magnitude, 0))


@triton.jit
def _amax_reduce(X, OUT, N: tl.constexpr, FINAL: tl.constexpr, DENOM: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.max(tl.load(X + idx, idx < N, 0.0), 0)
    if FINAL:
        # Torch's division by a Python scalar multiplies by its FP32 reciprocal.
        value = tl.maximum(value * (1.0 / DENOM), 1e-30)
    tl.store(OUT + tl.program_id(0), value)


@triton.jit
def _validate_input_kernel(
    X,
    ERROR,
    N: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + _input_offsets(idx, R, C, SB, SR, SC), idx < N, 0).to(tl.float32)
    bad = tl.sum(((idx < N) & ~(tl.abs(value) < float("inf"))).to(tl.int32), 0)
    if bad:
        tl.atomic_or(ERROR, 1)


@triton.jit
def _validate_outer_kernel(OUTER, ERROR):
    outer = tl.load(OUTER)
    if ~((outer > 0.0) & (outer < float("inf"))):
        tl.atomic_or(ERROR + 1, 1)


@triton.jit
def _inner_round(x, MIN_NORMAL: tl.constexpr, SUBNORMAL_INV: tl.constexpr):
    """Positive finite 3-mantissa-bit RNE, including AMD subnormal scales."""
    bits = x.to(tl.uint32, bitcast=True)
    rounded_bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & 0xFFF00000
    normal = rounded_bits.to(tl.float32, bitcast=True)
    units = x * SUBNORMAL_INV
    lower = tl.floor(units).to(tl.int32)
    fraction = units - lower.to(tl.float32)
    upper = (fraction > 0.5) | ((fraction == 0.5) & ((lower & 1) != 0))
    subnormal = (lower + upper.to(tl.int32)).to(tl.float32) * (1.0 / SUBNORMAL_INV)
    return tl.where(x < MIN_NORMAL, subnormal, normal)


@triton.jit
def _e2m1_value(code):
    return tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            0.5,
            tl.where(
                code == 2,
                1.0,
                tl.where(
                    code == 3,
                    1.5,
                    tl.where(code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )


@triton.jit
def _e2m1_code(magnitude, uniform, STOCHASTIC: tl.constexpr):
    if STOCHASTIC:
        # bucketize(left): exact grid values select themselves with probability1.
        upper = (magnitude > 0.0).to(tl.int32)
        upper += (magnitude > 0.5).to(tl.int32)
        upper += (magnitude > 1.0).to(tl.int32)
        upper += (magnitude > 1.5).to(tl.int32)
        upper += (magnitude > 2.0).to(tl.int32)
        upper += (magnitude > 3.0).to(tl.int32)
        upper += (magnitude > 4.0).to(tl.int32)
        lower = tl.maximum(upper - 1, 0)
        lo, hi = _e2m1_value(lower), _e2m1_value(upper)
        probability = tl.div_rn(magnitude - lo, tl.where(hi > lo, hi - lo, 1.0))
        return tl.where(uniform < probability, upper, lower)
    else:
        # Alternate strict/non-strict midpoint comparisons implement encoded RNE.
        code = (magnitude > 0.25).to(tl.int32)
        code += (magnitude >= 0.75).to(tl.int32)
        code += (magnitude > 1.25).to(tl.int32)
        code += (magnitude >= 1.75).to(tl.int32)
        code += (magnitude > 2.5).to(tl.int32)
        code += (magnitude >= 3.5).to(tl.int32)
        code += (magnitude > 5.0).to(tl.int32)
        return code


@triton.jit
def _quantize_kernel(
    X,
    OUTER,
    PACKED,
    INNER,
    OUT,
    UNIFORM,
    ERROR,
    RNG_SEED,
    RNG_OFFSET,
    R: tl.constexpr,
    C: tl.constexpr,
    RP: tl.constexpr,
    CP: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    OB: tl.constexpr,
    OR: tl.constexpr,
    OC: tl.constexpr,
    TWO_D: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    GPU_RANDOM: tl.constexpr,
    DIRECT: tl.constexpr,
    CHECK: tl.constexpr,
    INNER_FLOOR: tl.constexpr,
    INNER_MAX: tl.constexpr,
    MIN_NORMAL: tl.constexpr,
    SUBNORMAL_INV: tl.constexpr,
    DEFERRED: tl.constexpr = False,
):
    ct = CP // 16
    rt = tl.cdiv(RP, 16)
    pid = tl.program_id(0)
    batch = pid // (rt * ct)
    tile_r = pid // ct % rt
    tile_c = pid % ct
    rows = tile_r * 16 + tl.arange(0, 16)
    cols = tile_c * 16 + tl.arange(0, 16)
    value = tl.load(
        X + batch * SB + rows[:, None] * SR + cols[None, :] * SC,
        (rows[:, None] < R) & (cols[None, :] < C),
        0.0,
    ).to(tl.float32)
    if DEFERRED:
        # A finite amax/scale alone cannot establish finite input: the amax
        # reduction intentionally excludes nonfinite values. Keep this check
        # inside the data-reading kernel, including when OUTER was supplied.
        valid = (rows[:, None] < R) & (cols[None, :] < C)
        bad_input = tl.sum(tl.sum((valid & ~(tl.abs(value) < float("inf"))).to(tl.int32), 1), 0)
        if bad_input:
            tl.atomic_or(ERROR, 1)
    maximum = tl.max(tl.abs(value), 1)
    if TWO_D:
        maximum = tl.max(maximum, 0)
    outer = tl.load(OUTER)
    if DEFERRED:
        if ~((outer > 0.0) & (outer < float("inf"))):
            tl.atomic_or(ERROR + 1, 1)
    raw_inner = tl.div_rn(maximum, outer) * (1.0 / 6.0)
    raw_inner = tl.minimum(tl.maximum(raw_inner, INNER_FLOOR), INNER_MAX)
    inner = _inner_round(raw_inner, MIN_NORMAL, SUBNORMAL_INV)
    combined = inner * outer
    if CHECK:
        invalid = (combined <= 0.0) | (tl.abs(combined) == float("inf")) | (combined != combined)
        if TWO_D:
            bad = invalid.to(tl.int32)
        else:
            bad = tl.sum(((rows < RP) & invalid).to(tl.int32), 0)
        if bad:
            tl.atomic_or(ERROR + (2 if DEFERRED else 0), 1)
    if not TWO_D:
        combined = combined[:, None]
    scaled = tl.div_rn(value, combined)
    magnitude = tl.minimum(tl.abs(scaled), 6.0)
    if STOCHASTIC:
        logical_index = (batch.to(tl.uint64) * RP + rows[:, None].to(tl.uint64)) * CP + cols[None, :]
        if GPU_RANDOM:
            uniform = _philox_uniform(RNG_SEED, RNG_OFFSET, logical_index)
        else:
            uniform = tl.load(UNIFORM + logical_index, rows[:, None] < RP, 0.0)
    else:
        uniform = tl.full((16, 16), 0.0, tl.float32)
    code = _e2m1_code(magnitude, uniform, STOCHASTIC)
    sign = value.to(tl.uint32, bitcast=True) >> 31
    if DIRECT:
        reconstructed = tl.where(sign != 0, -_e2m1_value(code), _e2m1_value(code)) * combined
        tl.store(
            OUT + batch * OB + rows[:, None] * OR + cols[None, :] * OC,
            reconstructed,
            (rows[:, None] < R) & (cols[None, :] < C),
        )
    else:
        nibble = code | (sign.to(tl.int32) << 3)
        pairs = tl.reshape(nibble, (16, 8, 2))
        shifts = tl.arange(0, 2)[None, None, :] * 4
        packed = tl.sum(pairs << shifts, 2)
        packed_cols = tile_c * 8 + tl.arange(0, 8)
        tl.store(
            PACKED + (batch * RP + rows[:, None]) * (CP // 2) + packed_cols[None, :],
            packed,
            rows[:, None] < RP,
        )
        if TWO_D:
            tl.store(INNER + (batch * (RP // 16) + tile_r) * ct + tile_c, inner)
        else:
            tl.store(INNER + (batch * RP + rows) * ct + tile_c, inner, rows < RP)


@triton.jit
def _dequantize_kernel(
    PACKED,
    INNER,
    OUTER,
    OUT,
    N: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    RP: tl.constexpr,
    CP: tl.constexpr,
    OB: tl.constexpr,
    OR: tl.constexpr,
    OC: tl.constexpr,
    TWO_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = idx // (R * C)
    row = idx // C % R
    col = idx % C
    byte = tl.load(PACKED + (batch * RP + row) * (CP // 2) + col // 2, idx < N, 0).to(tl.int32)
    code = (byte >> ((col % 2) * 4)) & 15
    if TWO_D:
        scale_idx = (batch * (RP // 16) + row // 16) * (CP // 16) + col // 16
    else:
        scale_idx = (batch * RP + row) * (CP // 16) + col // 16
    inner = tl.load(INNER + scale_idx, idx < N, 0.0)
    combined = inner * tl.load(OUTER)
    value = _e2m1_value(code & 7)
    value = tl.where((code & 8) != 0, -value, value) * combined
    tl.store(OUT + batch * OB + row * OR + col * OC, value, idx < N)


def _input(x):
    if (
        not isinstance(x, torch.Tensor)
        or x.device.type != "cuda"
        or x.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("Triton FP4 requires a CUDA/ROCm float32 or bfloat16 tensor")
    if x.ndim not in (2, 3):
        raise ValueError("Triton FP4 requires a 2-D or 3-D tensor")


def _shape_stride(x, axis=None):
    dims = list(range(x.ndim))
    if axis is not None:
        dims.remove(axis)
        dims.append(axis)
    shape = tuple(x.shape[d] for d in dims)
    strides = tuple(x.stride(d) for d in dims)
    if x.ndim == 2:
        return (1, *shape), (0, *strides)
    return shape, strides


class DeferredValueChecks:
    """Bounded checks for ONE grouped forward or backward call.

    The three sticky device flags are input, outer-scale and combined-scale
    errors. No tensor values or per-expert lists are retained. A normal context
    exit transfers these three integers once and raises before autograd returns.
    Explicit objects avoid global state, cross-step deferral and nested-scope
    interference. Ordinary public QDQ calls do not use this object implicitly.
    """

    def __init__(self, device):
        self.flags = torch.zeros(3, dtype=torch.int32, device=device)
        self.closed = False

    def _require_open(self, x):
        if self.closed:
            raise RuntimeError("FP4 deferred validation scope is already closed")
        if x.device != self.flags.device:
            raise ValueError("FP4 deferred validation cannot cross devices")

    def check_input(self, x):
        self._require_open(x)
        _input(x)
        if x.numel():
            (_, r, c), strides = _shape_stride(x)
            _validate_input_kernel[(triton.cdiv(x.numel(), 4096),)](
                x, self.flags, x.numel(), r, c, *strides, 4096
            )

    def finish(self):
        if self.closed:
            raise RuntimeError("FP4 deferred validation scope is already closed")
        self.closed = True
        flags = self.flags.cpu().tolist()
        failures = [
            description
            for flag, description in zip(
                flags, ("non-finite input", "invalid outer scale", "invalid combined inner/outer scale")
            )
            if flag
        ]
        if failures:
            raise ValueError("FP4 grouped deferred validation: " + "; ".join(failures))

    def __enter__(self):
        self._require_open(self.flags)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.finish()
        else:
            # Preserve the original failure; this object is never reused.
            self.closed = True
        return False


def _validation_flags(validation, x, check_values):
    if not isinstance(validation, DeferredValueChecks):
        raise TypeError("FP4 validation requires a DeferredValueChecks scope")
    if not check_values:
        raise ValueError("Deferred FP4 validation cannot disable value checks")
    validation._require_open(x)
    return validation.flags


@torch.no_grad()
def compute_outer_scale(x, config, *, check_values=True, validation=None):
    """Whole-tensor outer amax with bounded-size recursive Triton reductions."""
    _input(x)
    error = (
        _validation_flags(validation, x, check_values)
        if validation is not None
        else torch.zeros(1, device=x.device, dtype=torch.int32)
    )
    if not x.numel():
        return torch.full((1,), 1e-30, device=x.device, dtype=torch.float32)
    (_, r, c), (sb, sr, sc) = _shape_stride(x)
    count = triton.cdiv(x.numel(), 4096)
    partial = torch.empty(count, device=x.device, dtype=torch.float32)
    _amax_input[(count,)](x, partial, error, x.numel(), r, c, sb, sr, sc, check_values, 4096)
    while count > 4096:
        next_count = triton.cdiv(count, 4096)
        reduced = torch.empty(next_count, device=x.device, dtype=torch.float32)
        _amax_reduce[(next_count,)](partial, reduced, count, False, 1.0, 4096)
        partial, count = reduced, next_count
    outer = torch.empty(1, device=x.device, dtype=torch.float32)
    _amax_reduce[(1,)](partial, outer, count, True, 6.0 * config.inner_max, triton.next_power_of_2(count))
    if validation is None and check_values and error.item():
        raise ValueError("Triton FP4 rejects non-finite input")
    return outer


def _prepare(x, config, operand, outer_scale, random, check_values, validation=None):
    _input(x)
    if validation is not None:
        _validation_flags(validation, x, check_values)
    if operand != config.operand(operand.direction, operand.role):
        raise ValueError("FP4 operand does not match its recipe")
    if operand.stochastic_rounding:
        _check_random(random, config)
    if outer_scale is None:
        outer_scale = compute_outer_scale(x, config, check_values=check_values, validation=validation)
    else:
        if (
            not isinstance(outer_scale, torch.Tensor)
            or outer_scale.device != x.device
            or outer_scale.dtype != torch.float32
            or outer_scale.shape != (1,)
        ):
            raise ValueError("outer_scale must be FP32 [1] on the input device")
        if check_values and validation is None:
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Triton FP4 rejects non-finite input")
            if not bool((torch.isfinite(outer_scale) & (outer_scale > 0)).all()):
                raise ValueError("outer_scale must be positive and finite")
    outer = outer_scale.detach().clone().contiguous()
    if validation is not None and not x.numel():
        # Empty QDQ has no data kernel, but its explicit outer must still be
        # valid. Do not silently drop an invalid scale on an empty expert.
        _validate_outer_kernel[(1,)](outer, validation.flags)
    axis = operand.axis % x.ndim
    physical, strides = _shape_stride(x, axis)
    b, r, c = physical
    rp = triton.cdiv(r, 16) * 16 if operand.two_dimensional else r
    alignment = max(16, operand.hadamard_size)
    cp = triton.cdiv(c, alignment) * alignment
    physical_padded = [b, rp, cp] if x.ndim == 3 else [rp, cp]
    last = physical_padded.pop()
    physical_padded.insert(axis, last)
    layout = FP4TensorLayout(tuple(x.shape), tuple(physical_padded), operand)
    layout.validate(config)
    if operand.stochastic_rounding and b * rp * cp:
        if config.sr_rng == "triton_philox":
            uniform = reserve_philox_sr(random, b * rp * cp)
        else:
            uniform = torch.rand(
                (b, rp, cp),
                generator=_generator(random, "sr", b * rp * cp),
                dtype=torch.float32,
                device="cpu",
            ).to(x.device)
    else:
        uniform = outer
    return outer, layout, physical, (b, rp, cp), strides, uniform


def _launch(
    x,
    config,
    operand,
    outer,
    physical,
    padded,
    strides,
    uniform,
    packed,
    inner,
    out,
    direct,
    check_values,
    validation=None,
):
    b, r, c = physical
    _, rp, cp = padded
    if not b * rp * cp:
        return
    _, out_strides = _shape_stride(out, operand.axis % out.ndim) if direct else ((0, 0, 0), (0, 0, 0))
    error = (
        validation.flags
        if validation is not None
        else torch.zeros(1, dtype=torch.int32, device=x.device) if check_values else outer
    )
    gpu_random = config.sr_rng == "triton_philox" and operand.stochastic_rounding
    seed, offset = (uniform["seed"], uniform["offset"]) if gpu_random else (0, 0)
    _quantize_kernel[(b * triton.cdiv(rp, 16) * (cp // 16),)](
        x,
        outer,
        packed,
        inner,
        out,
        outer if gpu_random else uniform,
        error,
        seed,
        offset,
        r,
        c,
        rp,
        cp,
        *strides,
        *out_strides,
        operand.two_dimensional,
        operand.stochastic_rounding,
        gpu_random,
        direct,
        check_values,
        config.inner_floor,
        config.inner_max,
        2.0 ** (-6 if config.recipe == "nvfp4" else -14),
        2.0 ** (9 if config.recipe == "nvfp4" else 17),
        validation is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if validation is None and check_values and error.item():
        raise ValueError("FP4 combined inner/outer scale must be positive and finite in FP32")


@torch.no_grad()
def quantize(x, config, operand, *, outer_scale=None, random=None, check_values=True, validation=None):
    outer, layout, physical, padded, strides, uniform = _prepare(
        x, config, operand, outer_scale, random, check_values, validation
    )
    packed_shape, scale_shape = layout.storage_shapes(config)
    data = torch.empty(packed_shape, dtype=torch.uint8, device=x.device)
    inner = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    _launch(
        x,
        config,
        operand,
        outer,
        physical,
        padded,
        strides,
        uniform,
        data,
        inner,
        outer,
        False,
        check_values,
        validation,
    )
    return TwoLevelFP4Tensor(data, inner, outer, config, layout).validate()


@torch.no_grad()
def dequantize(tensor, dtype=None):
    tensor.validate()
    if tensor.data.device.type != "cuda":
        raise ValueError("Triton FP4 dequantization requires CUDA/ROCm storage")
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Triton FP4 dequantization supports float32 or bfloat16")
    out = torch.empty(tensor.layout.original_shape, device=tensor.data.device, dtype=dtype)
    (_, r, c), strides = _shape_stride(out, tensor.layout.operand.axis % out.ndim)
    physical = tensor.layout.validate(tensor.config)
    rp, cp = physical[-2:]
    if out.numel():
        _dequantize_kernel[(triton.cdiv(out.numel(), 256),)](
            tensor.data,
            tensor.inner_scale,
            tensor.outer_scale,
            out,
            out.numel(),
            r,
            c,
            rp,
            cp,
            *strides,
            tensor.layout.operand.two_dimensional,
            256,
            enable_fp_fusion=False,
        )
    return out


@torch.no_grad()
def qdq(x, config, operand, *, outer_scale=None, random=None, dtype=None, check_values=True, validation=None):
    """Fused new-format quant/dequant with no packed or inner tensor writeback."""
    dtype = x.dtype if dtype is None else dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Triton FP4 QDQ supports float32 or bfloat16")
    outer, layout, physical, padded, strides, uniform = _prepare(
        x, config, operand, outer_scale, random, check_values, validation
    )
    out = torch.empty(layout.original_shape, dtype=dtype, device=x.device)
    _launch(
        x,
        config,
        operand,
        outer,
        physical,
        padded,
        strides,
        uniform,
        outer,
        outer,
        out,
        True,
        check_values,
        validation,
    )
    return out
