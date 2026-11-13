# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Executable Torch reference for the Primus two-level FP4 contract.

These operators pack real E2M1 bytes with E4M3FN/UE5M3 inner scales and an
explicit FP32 outer scale. They run on CPU or GPU but are correctness tools,
not native FP4 GEMM kernels: QDQ consumers subsequently use high precision
GEMMs. Value checks synchronize a GPU, and reference random values are made
on CPU before transfer. Quantization is deliberately not an autograd STE;
the Megatron adapter supplies its three-direction backward explicitly.

The scale policy follows the ALTO reference described in fp4_contract.py;
the implementation here uses independent table-based rounding. Its random
mapping is versioned below and does not reproduce ALTO's Triton Philox stream.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from .fp4_contract import FP4RandomState, FP4TensorLayout, TwoLevelFP4Tensor
from .fp4_rng import REFERENCE_RNG_VERSION

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_DTYPES = (torch.float32, torch.bfloat16)


def _check_input(x, validation=None):
    if not isinstance(x, torch.Tensor) or x.dtype not in _DTYPES:
        raise ValueError("FP4 reference input must be a float32 or bfloat16 tensor")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("FP4 reference requires a real CPU/CUDA/ROCm tensor")
    if validation is not None:
        validation.check_input(x)
    elif not bool(torch.isfinite(x).all()):
        raise ValueError("FP4 reference rejects non-finite input")


def _check_random(random, config):
    if not isinstance(random, FP4RandomState):
        raise ValueError("Stochastic FP4 operations require an explicit FP4RandomState")
    if random.recipe_fingerprint != config.fingerprint or random.seed != config.rng_seed:
        raise ValueError("FP4 random state does not match the recipe")


def _generator(random, stream, words):
    """Map one reservation to an isolated CPU generator, without skip-ahead.

    A block reserves four logical random words. Each call hashes the stream
    key and starting block into a fresh CPU-generator seed; disjoint logical
    ranges do not reuse a generator. This is not a claim about the number of
    words internally consumed by Torch randperm or a native Philox kernel.
    Replaying the same call sequence, device-independent shape and Torch
    version reproduces the random values. Global Torch RNG is never touched.
    """
    reservation = random.reserve(stream, (words + 3) // 4)
    identity = (
        f"{REFERENCE_RNG_VERSION}:{reservation['seed']}:" f"{reservation['stream']}:{reservation['offset']}"
    )
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") & (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


@torch.no_grad()
def compute_outer_scale(x, config):
    """Return one FP32 scale for the entire supplied tensor, including a bank.

    Empty tensors use the same floor as an all-zero tensor. The reference
    rejects NaN/Inf rather than silently sanitizing them; this is intentionally
    stricter than ALTO's non-finite amax policy.
    """
    _check_input(x)
    amax = x.float().abs().amax() if x.numel() else x.new_zeros((), dtype=torch.float32)
    return (amax / (6.0 * config.inner_max)).clamp_min(1e-30).reshape(1).contiguous()


def _round_on_grid(value, grid):
    """Nearest-even by encoded code, including subnormal/normal boundaries."""
    upper = torch.bucketize(value.contiguous(), grid).clamp_max(grid.numel() - 1)
    lower = (upper - 1).clamp_min(0)
    left = value - grid[lower]
    right = grid[upper] - value
    choose_upper = (right < left) | ((right == left) & ((upper & 1) == 0))
    return torch.where(choose_upper, upper, lower)


def _inner_grid(config, device):
    bias, last_code = (7, 126) if config.recipe == "nvfp4" else (15, 254)
    code = torch.arange(last_code + 1, dtype=torch.int32, device=device)
    mantissa = (code & 7).float() / 8
    normal = torch.ldexp(1.0 + mantissa, (code >> 3) - bias)
    return torch.where(code < 8, mantissa * 2.0 ** (1 - bias), normal)


def _expand_inner(inner, config, operand):
    if operand.two_dimensional:
        inner = inner.repeat_interleave(config.block_size, dim=-2)
    return inner.repeat_interleave(config.block_size, dim=-1)


@torch.no_grad()
def quantize(x, config, operand, *, outer_scale=None, random=None):
    """Pack a 2-D/3-D tensor in contiguous axis-last storage.

    ``operand`` must be obtained from ``config.operand(direction, role)``.
    Quantization pads only the required matrix dimensions; a grouped caller
    must first pad each expert separately and retain its routing offsets.
    ``outer_scale`` can be shared by per-expert calls, preserving a whole-bank
    amax scope. The payload owns a clone and never mutates the caller's scale.
    SR requires a private state; deterministic operands consume no RNG.
    RHT is a separate paired operation: use ``rht_pair`` before quantizing
    wgrad operands, so both receive exactly the same transform.
    """
    if config.sr_rng != "torch_cpu":
        raise ValueError("Torch reference quantization requires sr_rng=torch_cpu")
    _check_input(x)
    if x.ndim not in (2, 3):
        raise ValueError("FP4 reference quantization requires a 2-D or 3-D tensor")
    if operand != config.operand(operand.direction, operand.role):
        raise ValueError("FP4 operand does not match its recipe")
    if operand.stochastic_rounding:
        _check_random(random, config)
    if outer_scale is None:
        outer_scale = compute_outer_scale(x, config)
    elif (
        not isinstance(outer_scale, torch.Tensor)
        or outer_scale.dtype != torch.float32
        or outer_scale.shape != (1,)
        or outer_scale.device != x.device
        or not bool((torch.isfinite(outer_scale) & (outer_scale > 0)).all())
    ):
        raise ValueError("outer_scale must be a positive finite FP32 [1] tensor on the input device")
    outer_scale = outer_scale.detach().clone().contiguous()

    axis = operand.axis % x.ndim
    physical = x.float().movedim(axis, -1).contiguous()
    alignment = max(config.block_size, operand.hadamard_size)
    pad_last = (-physical.shape[-1]) % alignment
    pad_rows = (-physical.shape[-2]) % config.block_size if operand.two_dimensional else 0
    if physical.numel():
        physical = F.pad(physical, (0, pad_last, 0, pad_rows))
    else:
        # torch.nn.functional.pad rejects some zero-length matrix dimensions.
        shape = (*physical.shape[:-2], physical.shape[-2] + pad_rows, physical.shape[-1] + pad_last)
        physical = physical.new_zeros(shape)
    padded = list(physical.movedim(-1, axis).shape)
    layout = FP4TensorLayout(tuple(x.shape), tuple(padded), operand)
    layout.validate(config)

    if operand.two_dimensional:
        rows, cols = physical.shape[-2:]
        grouped = physical.abs().reshape(*physical.shape[:-2], rows // 16, 16, cols // 16, 16)
        amax = grouped.amax(dim=(-3, -1))
    else:
        grouped = physical.abs().reshape(*physical.shape[:-1], physical.shape[-1] // 16, 16)
        amax = grouped.amax(dim=-1)
    raw_scale = (amax / outer_scale / 6.0).clamp(config.inner_floor, config.inner_max)
    grid = _inner_grid(config, x.device)
    inner_scale = grid[_round_on_grid(raw_scale, grid)].contiguous()
    # Combine in FP32 before division, matching the stated ALTO quant-scale
    # order. Two successive divisions can round differently near E2M1 ties.
    quant_scale = inner_scale * outer_scale
    if not bool((torch.isfinite(quant_scale) & (quant_scale > 0)).all()):
        raise ValueError("FP4 combined inner/outer scale must be positive and finite in FP32")
    scaled = physical / _expand_inner(quant_scale, config, operand)
    magnitude = scaled.abs().clamp_max(6.0)
    values = torch.tensor(_E2M1, dtype=torch.float32, device=x.device)
    if operand.stochastic_rounding and magnitude.numel():
        upper = torch.bucketize(magnitude.contiguous(), values).clamp_max(7)
        lower = (upper - 1).clamp_min(0)
        distance = values[upper] - values[lower]
        probability = (magnitude - values[lower]) / torch.where(distance > 0, distance, 1.0)
        uniform = torch.rand(
            magnitude.shape,
            generator=_generator(random, "sr", magnitude.numel()),
            dtype=torch.float32,
            device="cpu",
        ).to(x.device)
        code = torch.where(uniform < probability, upper, lower)
    else:
        code = _round_on_grid(magnitude, values)
    code = code.to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) << 3)
    data = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    return TwoLevelFP4Tensor(data, inner_scale, outer_scale, config, layout).validate()


@torch.no_grad()
def dequantize(tensor, dtype=None):
    """Decode and return original coordinates/shape, with padding removed."""
    tensor.validate()
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in _DTYPES:
        raise ValueError("FP4 reference dequantization supports float32 or bfloat16")
    low = tensor.data & 15
    high = tensor.data >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    values = torch.tensor(_E2M1, dtype=torch.float32, device=tensor.data.device)
    magnitude = values[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -magnitude, magnitude)
    quant_scale = tensor.inner_scale * tensor.outer_scale
    physical = values * _expand_inner(quant_scale, tensor.config, tensor.layout.operand)
    result = physical.movedim(-1, tensor.layout.operand.axis % physical.ndim)
    return result[tuple(slice(0, size) for size in tensor.layout.original_shape)].to(dtype).contiguous()


@torch.no_grad()
def qdq(x, config, operand, *, outer_scale=None, random=None, dtype=None):
    """Explicit quant/dequant; defaults to the input dtype, without an STE."""
    return dequantize(
        quantize(x, config, operand, outer_scale=outer_scale, random=random),
        dtype=x.dtype if dtype is None else dtype,
    )


@lru_cache(maxsize=2)
def _rht_base(size):
    """Immutable CPU Walsh base; random signs/permutations are never cached."""
    index = torch.arange(size, dtype=torch.int64, device="cpu")
    bits = index[:, None] & index[None, :]
    parity = torch.zeros((size, size), dtype=torch.int64, device="cpu")
    for shift in range(size.bit_length() - 1):
        parity ^= (bits >> shift) & 1
    return (1 - 2 * parity).float()


@torch.no_grad()
def rht_pair(x, dy, config, *, random=None, validation=None):
    """Apply the SAME orthogonal block transform on both wgrad token axes.

    Returns full padded rows: cropping a transformed tail before the wgrad
    product breaks orthogonal cancellation. Inputs are transformed in FP32
    and returned in their original dtype, matching the intended BF16 operand
    boundary. H32 is the default; H16/disabled are distinct recipe settings.
    Randomized H uses row signs and a shared row/column permutation, freshly
    selected for this pair. Empty input and disabled RHT consume no RNG.
    """
    _check_input(x, validation)
    _check_input(dy, validation)
    if x.ndim != 2 or dy.ndim != 2 or x.shape[0] != dy.shape[0] or x.device != dy.device:
        raise ValueError("RHT requires two 2-D operands with equal token count and device")
    if config.rht_size == 0 or x.shape[0] == 0:
        return x, dy
    size = config.rht_size
    if config.rht_randomized:
        _check_random(random, config)
    # Walsh entries H[i,j] = (-1)**popcount(i & j). All following operations
    # are out-of-place, so the cached deterministic base is never mutated.
    h = _rht_base(size)
    if config.rht_randomized:
        generator = _generator(random, "rht", 2 * size)
        signs = 2 * torch.randint(2, (size,), generator=generator, dtype=torch.int64) - 1
        permutation = torch.randperm(size, generator=generator)
        h = (h * signs[:, None])[permutation][:, permutation]
    h = h.to(x.device)
    tail = (-x.shape[0]) % size

    def transform(value):
        padded = F.pad(value.float(), (0, 0, 0, tail))
        blocks = padded.reshape(-1, size, value.shape[-1])
        # Division after matmul avoids an early BF16 H32 normalization round.
        return ((h @ blocks) / math.sqrt(size)).reshape(-1, value.shape[-1]).to(value.dtype)

    return transform(x), transform(dy)
