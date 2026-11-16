# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Opt-in Philox/Bernoulli gradient SR with native MXFP4 storage and GEMMs.

Turbo first produces its ordinary RNE data, scales and grouped layout. Only an
operand requesting SR is overwritten, using the same Philox and E2M1 decision
functions as the two-level FP4 Triton quantizer. This intentionally pays for an
extra gradient quantization pass: native scale selection, including UOS, and
the producer's actual group padding remain authoritative. No Turbo source or
GEMM is replaced. Fused producers bypassing the two public quantizers must be
disabled by the integration guard before installing this adapter.

The private stream is rank-local and launch-ordered, with explicit checkpoint
state. Shared algorithm does not imply identical random samples across formats
whose operand shapes, padding, RHT or launch ordering differ.
"""

from __future__ import annotations

import copy
import importlib
import threading

import torch
import triton
import triton.language as tl

from .fp4_contract import FP4RandomState
from .fp4_rng import PHILOX_SR_RNG_VERSION, reserve_philox_sr
from .fp4_triton import _e2m1_code, _philox_uniform

_IDENTITY = "mxfp4-shared-philox-gradient-v1"
_RNG = None
_TOPOLOGY = None
_LAUNCHES = 0
_PATCHES = []
_LOCK = threading.RLock()


@triton.jit
def _h16(value):
    # Native H4 within each four values, then H4 across four groups; no BF16
    # intermediate cast. The 32-element block contains two independent H16s.
    lane = tl.arange(0, 32)
    for bit in tl.static_range(4):
        peer = tl.gather(value, tl.broadcast_to(lane[None, :] ^ (1 << bit), (16, 32)), 1)
        value = tl.where((lane[None, :] & (1 << bit)) == 0, value + peer, peer - value)
    return value * 0.25


@triton.jit(
    do_not_specialize=[
        "R",
        "C",
        "CP",
        "XS0",
        "XS1",
        "OS0",
        "SS0",
        "SEED",
        "OFFSET",
        "LOGICAL_BASE",
        "LOGICAL_STRIDE",
    ]
)
def _overwrite(
    X,
    OUT,
    SCALE,
    R,
    C,
    CP,
    XS0,
    XS1,
    OS0,
    SS0,
    SEED,
    OFFSET,
    LOGICAL_BASE,
    LOGICAL_STRIDE,
    RHT: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    value = tl.load(
        X + rows[:, None] * XS0 + cols[None, :] * XS1, (rows[:, None] < R) & (cols[None, :] < C), 0.0
    ).to(tl.float32)
    if RHT:
        value = _h16(value)
    exponent = tl.load(SCALE + rows * SS0 + tl.program_id(1), rows < R, 127).to(tl.int32)
    # E8M0 has no zero: encoded 0 represents 2**-127. Its normal FP32 bitcast
    # needs this one special case; encoded 255 is outside finite-input use.
    scale = tl.where(exponent == 0, 2.0**-127, (exponent << 23).to(tl.float32, bitcast=True))
    magnitude = tl.minimum(tl.abs(tl.div_rn(value, scale[:, None])), 6.0)
    logical = LOGICAL_BASE.to(tl.uint64) + rows[:, None].to(tl.uint64) * LOGICAL_STRIDE + cols[None, :]
    uniform = _philox_uniform(SEED, OFFSET, logical)
    code = _e2m1_code(magnitude, uniform, True)
    sign = value.to(tl.uint32, bitcast=True) >> 31
    nibble = code | (sign.to(tl.int32) << 3)
    pair = tl.reshape(nibble, (16, 16, 2))
    packed = tl.sum(pair << (tl.arange(0, 2)[None, None, :] * 4), 2)
    packed_cols = tl.program_id(1) * 16 + tl.arange(0, 16)
    tl.store(
        OUT + rows[:, None] * OS0 + packed_cols[None, :],
        packed,
        (rows[:, None] < R) & (packed_cols[None, :] < CP // 2),
    )


def is_installed():
    return _RNG is not None


def state_dict():
    with _LOCK:
        if _RNG is None:
            raise ValueError("Shared MXFP4 SR is not installed")
        return copy.deepcopy(
            {
                "version": 1,
                "policy": "shared_philox",
                "rng_implementation": PHILOX_SR_RNG_VERSION,
                "topology": _TOPOLOGY,
                "rng": _RNG.state_dict(),
                "launches": _LAUNCHES,
            }
        )


def validate_state_dict(state):
    expected = state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Shared MXFP4 SR checkpoint fields mismatch")
    for key in ("version", "policy", "rng_implementation", "topology"):
        if state[key] != expected[key]:
            raise ValueError(f"Shared MXFP4 SR checkpoint {key} mismatch")
    saved = state["rng"]
    if not isinstance(saved, dict) or set(saved) != set(expected["rng"]):
        raise ValueError("Shared MXFP4 SR checkpoint RNG fields mismatch")
    for key in ("version", "recipe_fingerprint", "seed", "stream_id", "rht_offset"):
        if saved[key] != expected["rng"][key]:
            raise ValueError(f"Shared MXFP4 SR checkpoint {key} mismatch")
    if type(saved["sr_offset"]) is not int or not 0 <= saved["sr_offset"] <= 2**64:
        raise ValueError("Shared MXFP4 SR checkpoint counter is invalid")
    if type(state["launches"]) is not int or state["launches"] < 0:
        raise ValueError("Shared MXFP4 SR checkpoint launch count is invalid")


def load_state_dict(state):
    global _RNG, _LAUNCHES
    with _LOCK:
        validate_state_dict(state)
        fields = dict(state["rng"])
        fields.pop("version")
        restored = FP4RandomState(**fields)
        _RNG, _LAUNCHES = restored, state["launches"]


def _reserve(words):
    global _LAUNCHES
    with _LOCK:
        if _RNG is None:
            raise ValueError("Shared MXFP4 SR requires installation before execution")
        reservation = reserve_philox_sr(_RNG, words)
        _LAUNCHES += 1
        return reservation


def _check_recipe(recipe):
    if recipe.use_sr and (recipe.use_2d_block or recipe.shuffle_out or recipe.shuffle_scale):
        raise ValueError("Shared MXFP4 gradient SR requires unshuffled one-dimensional blocks")


def _launch(x, packed, scales, reservation, *, rht, cp=None, logical_base=0, logical_stride=None):
    r, c = x.shape
    cp = packed.shape[-1] * 2 if cp is None else cp
    if not r or not cp:
        return
    if cp % 32 or c > cp or scales.shape[-1] < cp // 32:
        raise ValueError("Native MXFP4 packed layout cannot cover the SR operand")
    _overwrite[(triton.cdiv(r, 16), cp // 32)](
        x,
        packed.view(torch.uint8),
        scales.view(torch.uint8),
        r,
        c,
        cp,
        *x.stride(),
        packed.stride(0),
        scales.stride(0),
        reservation["seed"],
        reservation["offset"],
        logical_base,
        cp if logical_stride is None else logical_stride,
        rht,
        num_warps=4,
    )


def _rewrite(x, packed, scales, *, axis, rht, group_lens=None, group_offs=None, padded_offs=None):
    if x.ndim not in (2, 3) or axis % x.ndim not in (x.ndim - 2, x.ndim - 1):
        raise ValueError("Shared MXFP4 SR requires a matrix or batched matrix operand")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Shared MXFP4 SR requires floating-point input")
    if packed.dtype != torch.float4_e2m1fn_x2 or scales.dtype != torch.float8_e8m0fnu:
        raise ValueError("Shared MXFP4 SR requires native packed E2M1/E8M0 outputs")
    transpose = axis % x.ndim == x.ndim - 2
    if group_lens is not None:
        if x.ndim != 2:
            raise ValueError("Shared MXFP4 grouped SR requires packed-M two-dimensional input")
        lengths = group_lens.tolist()
        offsets = group_offs.tolist()
        if len(offsets) != len(lengths) + 1 or offsets[0] != 0 or offsets[-1] != x.shape[0]:
            raise ValueError("Shared MXFP4 grouped SR routing offsets mismatch")
        if any(
            type(n) is not int or n < 0 or offsets[i + 1] - offsets[i] != n for i, n in enumerate(lengths)
        ):
            raise ValueError("Shared MXFP4 grouped SR routing lengths mismatch")
        if transpose:
            pads = padded_offs.tolist()
            cp = packed.shape[-1] * 2
            if len(pads) != len(offsets) or pads[0] != 0 or pads[-1] > cp:
                raise ValueError("Shared MXFP4 grouped SR padded offsets mismatch")
            if any((pads[i + 1] - pads[i]) % 32 or pads[i + 1] - pads[i] < n for i, n in enumerate(lengths)):
                raise ValueError("Shared MXFP4 grouped SR padded lengths mismatch")
            reservation = _reserve(packed.numel() * 2)
            for i, n in enumerate(lengths):
                start, end = pads[i], pads[i + 1]
                part = x[offsets[i] : offsets[i + 1]].T
                _launch(
                    part,
                    packed[:, start // 2 :],
                    scales[:, start // 32 :],
                    reservation,
                    rht=rht,
                    cp=end - start,
                    logical_base=start,
                    logical_stride=cp,
                )
            return
    reservation = _reserve(packed.numel() * 2)
    if x.ndim == 2:
        _launch(x.T if transpose else x, packed, scales, reservation, rht=rht)
    else:
        for batch in range(x.shape[0]):
            part = x[batch].T if transpose else x[batch]
            _launch(
                part,
                packed[batch],
                scales[batch],
                reservation,
                rht=rht,
                logical_base=batch * packed[batch].numel() * 2,
            )


def _dense_adapter(original):
    def quantize(
        x,
        out_dtype,
        axis,
        block_size,
        with_trans=False,
        scaling_recipe=None,
        scaling_recipe_for_trans=None,
        scale_rounding_mode=0,
    ):
        from primus_turbo.pytorch.core.low_precision import ScalingRecipe

        row = scaling_recipe or ScalingRecipe()
        col = (scaling_recipe_for_trans or ScalingRecipe()) if with_trans else row
        if not row.use_sr and not col.use_sr:
            return original(
                x,
                out_dtype,
                axis,
                block_size,
                with_trans,
                scaling_recipe,
                scaling_recipe_for_trans,
                scale_rounding_mode,
            )
        _check_recipe(row)
        _check_recipe(col)
        result = original(
            x,
            out_dtype,
            axis,
            block_size,
            with_trans,
            row._replace(use_sr=False),
            col._replace(use_sr=False),
            scale_rounding_mode,
        )
        if with_trans:
            if row.use_sr:
                _rewrite(x, result[0], result[1], axis=-1, rht=row.use_rht)
            if col.use_sr:
                _rewrite(x, result[2], result[3], axis=-2, rht=col.use_rht)
        else:
            _rewrite(x, result[0], result[1], axis=axis, rht=row.use_rht)
        return result

    return quantize


def _grouped_adapter(original):
    def quantize(
        x,
        out_dtype,
        axis,
        block_size,
        group_lens,
        group_offs,
        with_trans=False,
        scaling_recipe=None,
        scaling_recipe_for_trans=None,
        scale_rounding_mode=0,
        dest2src=None,
        permuted_probs=None,
        total_m=None,
    ):
        from primus_turbo.pytorch.core.low_precision import ScalingRecipe

        row = scaling_recipe or ScalingRecipe()
        col = (scaling_recipe_for_trans or ScalingRecipe()) if with_trans else row
        if not row.use_sr and not col.use_sr:
            return original(
                x,
                out_dtype,
                axis,
                block_size,
                group_lens,
                group_offs,
                with_trans,
                scaling_recipe,
                scaling_recipe_for_trans,
                scale_rounding_mode,
                dest2src=dest2src,
                permuted_probs=permuted_probs,
                total_m=total_m,
            )
        _check_recipe(row)
        _check_recipe(col)
        if dest2src is not None or permuted_probs is not None or total_m is not None:
            raise ValueError("Shared MXFP4 SR requires materialized gradients; fused gather is unsupported")
        result = original(
            x,
            out_dtype,
            axis,
            block_size,
            group_lens,
            group_offs,
            with_trans,
            row._replace(use_sr=False),
            col._replace(use_sr=False),
            scale_rounding_mode,
        )
        if with_trans:
            if row.use_sr:
                _rewrite(
                    x,
                    result[0],
                    result[1],
                    axis=1,
                    rht=row.use_rht,
                    group_lens=group_lens,
                    group_offs=group_offs,
                )
            if col.use_sr:
                _rewrite(
                    x,
                    result[2],
                    result[3],
                    axis=0,
                    rht=col.use_rht,
                    group_lens=group_lens,
                    group_offs=group_offs,
                    padded_offs=result[7],
                )
        else:
            _rewrite(
                x,
                result[0],
                result[1],
                axis=axis,
                rht=row.use_rht,
                group_lens=group_lens,
                group_offs=group_offs,
                padded_offs=result[3] if axis == 0 else None,
            )
        return result

    return quantize


def install_shared_mxfp4_sr(seed, *, rank=None, world_size=None):
    """Install before model execution; repeat installation never resets a stream."""
    global _RNG, _TOPOLOGY
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = (torch.distributed.get_rank() if initialized else 0) if rank is None else rank
    world_size = (
        (torch.distributed.get_world_size() if initialized else 1) if world_size is None else world_size
    )
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid shared MXFP4 SR topology")
    candidate = FP4RandomState(_IDENTITY, seed, f"rank{rank}/mxfp4-gradient")
    with _LOCK:
        if _RNG is not None:
            if seed != _RNG.seed or _TOPOLOGY != {"rank": rank, "world_size": world_size}:
                raise ValueError("Shared MXFP4 SR already installed with a different seed/topology")
            return
        # Both namespaces are intentional: ops holds imported aliases, while
        # callers importing the implementation module use its own globals.
        ops = importlib.import_module("primus_turbo.pytorch.ops.quantization")
        impl = importlib.import_module("primus_turbo.pytorch.kernels.quantization.quantization_impl")
        patches = []
        for name, adapter in (
            ("quantize_mxfp4_impl", _dense_adapter),
            ("grouped_quantize_mxfp4_impl", _grouped_adapter),
        ):
            original = getattr(impl, name)
            if getattr(ops, name) is not original:
                raise ValueError(f"Turbo {name} alias changed; refuse incomplete shared SR installation")
            replacement = adapter(original)
            patches.extend((module, name, original, replacement) for module in (impl, ops))
        for module, name, original, replacement in patches:
            setattr(module, name, replacement)
        _PATCHES.extend(patches)
        _RNG, _TOPOLOGY = candidate, {"rank": rank, "world_size": world_size}


def uninstall_shared_mxfp4_sr():
    """Testing/explicit teardown only; training must checkpoint instead of resetting."""
    global _RNG, _TOPOLOGY, _LAUNCHES
    with _LOCK:
        for module, name, original, replacement in _PATCHES:
            if getattr(module, name) is not replacement:
                raise ValueError("Shared MXFP4 SR patch ownership changed")
        for module, name, original, replacement in reversed(_PATCHES):
            setattr(module, name, original)
        _PATCHES.clear()
        _RNG, _TOPOLOGY, _LAUNCHES = None, None, 0
