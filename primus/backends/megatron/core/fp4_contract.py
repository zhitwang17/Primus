# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""NVFP4/AMDFP4 numerical and storage contracts, independent of GPU libraries.

These are two-level formats, not Turbo Float4QuantConfig/MXFP4. The reference
storage carries FP32 inner-grid values AND an FP32 outer scale. Native packed
scale layouts require a separate contract/backend; uint8 alone is not a dtype.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def _name(value):
    return getattr(value, "value", value)


def _boolean(value, name):
    if isinstance(value, bool):
        return value
    if value in ("true", "True", "1", 1):
        return True
    if value in ("false", "False", "0", 0):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class FP4Operand:
    direction: str
    role: str
    axis: int
    two_dimensional: bool
    stochastic_rounding: bool
    hadamard_size: int


@dataclass(frozen=True)
class TwoLevelFP4Config:
    recipe: str
    block_size: int = 16
    weight_2d: bool = True
    outer_scope: str = "tensor"
    inner_storage: str = "float32_grid"
    rht_size: int = 32
    rht_randomized: bool = True
    gradient_sr: bool = True
    rng_seed: int = 1234
    quantizer: str = "torch"
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "recipe", _name(self.recipe))
        if self.recipe not in ("nvfp4", "amdfp4"):
            raise ValueError("TwoLevelFP4Config requires nvfp4 or amdfp4")
        if self.quantizer not in ("torch", "triton"):
            raise ValueError("Two-level FP4 quantizer must be torch or triton")
        for key in ("weight_2d", "rht_randomized", "gradient_sr"):
            object.__setattr__(self, key, _boolean(getattr(self, key), key))
        if self.block_size != 16 or isinstance(self.block_size, bool):
            raise ValueError("NVFP4/AMDFP4 inner block_size must be 16")
        if self.outer_scope != "tensor" or self.inner_storage != "float32_grid":
            raise ValueError("Reference contract supports tensor outer scope and float32_grid inner storage")
        if type(self.rht_size) is not int or self.rht_size not in (0, 16, 32):
            raise ValueError("rht_size must be 0 (disabled), 16, or 32")
        if type(self.rng_seed) is not int or not 0 <= self.rng_seed < 2**63:
            raise ValueError("rng_seed must be an integer in [0, 2**63)")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported FP4 contract version")

    @property
    def inner_format(self):
        return "e4m3fn" if self.recipe == "nvfp4" else "ue5m3"

    @property
    def inner_max(self):
        return 448.0 if self.recipe == "nvfp4" else 114688.0

    @property
    def inner_floor(self):
        # Match ALTO's current policy: NV uses min NORMAL, AMD min SUBNORMAL.
        return 2.0**-6 if self.recipe == "nvfp4" else 2.0**-17

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def rng_fingerprint(self):
        """Kernel selection must not perturb a recipe's stochastic sequence."""
        numerical = self.to_dict()
        numerical.pop("quantizer")
        return hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()

    def operand(self, direction, role):
        axes = {
            ("fwd", "input"): -1,
            ("fwd", "weight"): -1,
            ("dgrad", "grad_output"): -1,
            ("dgrad", "weight"): -2,
            ("wgrad", "input"): 0,
            ("wgrad", "grad_output"): 0,
        }
        if (direction, role) not in axes:
            raise ValueError(f"Invalid FP4 operand {direction}/{role}")
        return FP4Operand(
            direction,
            role,
            axes[direction, role],
            role == "weight" and self.weight_2d,
            role == "grad_output" and self.gradient_sr,
            self.rht_size if direction == "wgrad" else 0,
        )

    def outer_scale(self, tensor_amax):
        """Scalar CPU specification. Tensor reductions belong to the backend."""
        if not math.isfinite(tensor_amax) or tensor_amax < 0:
            raise ValueError("tensor_amax must be finite and nonnegative")
        return max(tensor_amax / (6.0 * self.inner_max), 1e-30)

    def inner_scale(self, block_amax, outer_scale):
        """Scalar round-to-nearest-even oracle, not a GPU quantizer."""
        if not math.isfinite(block_amax) or block_amax < 0:
            raise ValueError("block_amax must be finite and nonnegative")
        if not math.isfinite(outer_scale) or outer_scale <= 0:
            raise ValueError("outer_scale must be positive and finite")
        value = min(max(block_amax / outer_scale / 6.0, self.inner_floor), self.inner_max)
        bias, last_code = (7, 126) if self.recipe == "nvfp4" else (15, 254)
        grid = [
            (
                (code & 7) / 8 * 2.0 ** (1 - bias)
                if code < 8
                else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - bias)
            )
            for code in range(last_code + 1)
        ]
        hi = bisect.bisect_left(grid, value)
        if hi == 0 or grid[hi] == value:
            return grid[hi]
        lo = hi - 1
        left, right = value - grid[lo], grid[hi] - value
        return grid[hi if right < left or (right == left and hi % 2 == 0) else lo]


def padded_group_offsets(lengths, alignment=128):
    """Per-expert padding; empty experts keep a zero-length interval."""
    if type(alignment) is not int or alignment < 128 or alignment % 128:
        raise ValueError("Reference grouped alignment must be a positive multiple of 128")
    original, padded = [0], [0]
    for size in lengths:
        if type(size) is not int or size < 0:
            raise ValueError("Expert lengths must be nonnegative integers")
        original.append(original[-1] + size)
        padded.append(padded[-1] + ((size + alignment - 1) // alignment) * alignment)
    return tuple(original), tuple(padded)


@dataclass(frozen=True)
class FP4TensorLayout:
    """Original coordinates plus physical axis-last, contiguous reference storage.

    The quantization axis is moved to the last dimension before packing. A
    transpose requires a new layout/quantization, not just transposing bytes.
    Group offsets refer to axis 0 of the ORIGINAL [M,K] activation/gradient.
    """

    original_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    operand: FP4Operand
    group_offsets: tuple[int, ...] = ()
    padded_offsets: tuple[int, ...] = ()

    def validate(self, config):
        if self.operand != config.operand(self.operand.direction, self.operand.role):
            raise ValueError("Operand contract does not match recipe")
        if len(self.original_shape) not in (2, 3) or len(self.original_shape) != len(self.padded_shape):
            raise ValueError("FP4 tensors require 2-D or 3-D shape metadata")
        for original, padded in zip(self.original_shape, self.padded_shape):
            if type(original) is not int or type(padded) is not int or original < 0 or padded < original:
                raise ValueError("Invalid original/padded dimensions")
        axis = self.operand.axis % len(self.padded_shape)
        physical = tuple(v for i, v in enumerate(self.padded_shape) if i != axis) + (self.padded_shape[axis],)
        if physical[-1] % config.block_size:
            raise ValueError("Quantization axis must be padded to block16")
        if self.operand.two_dimensional and physical[-2] % config.block_size:
            raise ValueError("2-D weights require both matrix dimensions padded to block16")
        if self.operand.hadamard_size and self.padded_shape[0] % self.operand.hadamard_size:
            raise ValueError("wgrad token axis must be padded for RHT")
        if bool(self.group_offsets) != bool(self.padded_offsets):
            raise ValueError("Both original and padded expert offsets are required")
        if self.group_offsets:
            if len(self.original_shape) != 2 or len(self.group_offsets) < 2:
                raise ValueError("Expert offsets describe routed 2-D tensors")
            if len(self.group_offsets) != len(self.padded_offsets):
                raise ValueError("Expert offset counts differ")
            for offsets, total in (
                (self.group_offsets, self.original_shape[0]),
                (self.padded_offsets, self.padded_shape[0]),
            ):
                if offsets[0] != 0 or offsets[-1] != total:
                    raise ValueError("Expert offsets must span the token buffer")
                if any(type(x) is not int or x < 0 for x in offsets):
                    raise ValueError("Expert offsets must be nonnegative integers")
                if any(a > b for a, b in zip(offsets, offsets[1:])):
                    raise ValueError("Expert offsets must be monotonic")
            lengths = [b - a for a, b in zip(self.group_offsets, self.group_offsets[1:])]
            padded_lengths = [b - a for a, b in zip(self.padded_offsets, self.padded_offsets[1:])]
            if any(p < n or p % 128 for n, p in zip(lengths, padded_lengths)):
                raise ValueError("Each expert must be independently padded to 128 rows")
        return physical

    def storage_shapes(self, config):
        physical = self.validate(config)
        packed = physical[:-1] + (physical[-1] // 2,)
        scale = physical[:-1] + (physical[-1] // 16,)
        if self.operand.two_dimensional:
            scale = physical[:-2] + (physical[-2] // 16, physical[-1] // 16)
        return packed, scale


@dataclass(frozen=True)
class TwoLevelFP4Tensor:
    data: Any
    inner_scale: Any
    outer_scale: Any
    config: TwoLevelFP4Config
    layout: FP4TensorLayout

    def metadata(self):
        return {
            "version": 1,
            "config": self.config.to_dict(),
            "fingerprint": self.config.fingerprint,
            "layout": asdict(self.layout),
            "payload_format": "e2m1_low_nibble_first",
            "physical_layout": "axis_last_contiguous",
            "inner_format": self.config.inner_format,
            "outer_dtype": "float32",
        }

    @classmethod
    def from_metadata(cls, data, inner_scale, outer_scale, metadata):
        config = TwoLevelFP4Config(**metadata["config"])
        layout = dict(metadata["layout"])
        layout["operand"] = FP4Operand(**layout["operand"])
        for key in ("original_shape", "padded_shape", "group_offsets", "padded_offsets"):
            layout[key] = tuple(layout[key])
        tensor = cls(data, inner_scale, outer_scale, config, FP4TensorLayout(**layout))
        # Includes fingerprint, format, layout and version; refuse stale or
        # relabelled scale metadata. JSON lists/tuples compare canonically.
        if json.dumps(tensor.metadata(), sort_keys=True) != json.dumps(metadata, sort_keys=True):
            raise ValueError("FP4 tensor metadata mismatch")
        return tensor.validate()

    def validate(self):
        """Structural validation without a device query or tensor host sync."""
        packed_shape, scale_shape = self.layout.storage_shapes(self.config)
        for tensor, dtype, shape in (
            (self.data, "torch.uint8", packed_shape),
            (self.inner_scale, "torch.float32", scale_shape),
            (self.outer_scale, "torch.float32", (1,)),
        ):
            if str(tensor.dtype) != dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"FP4 storage requires {dtype} {shape}, got {tensor.dtype} {tensor.shape}")
            if tensor.device != self.data.device or not tensor.is_contiguous():
                raise ValueError("FP4 storage must be contiguous and on the same device")
        return self


@dataclass
class FP4RandomState:
    """Private counter reservations; no consumption of global Torch/Python RNG.

    The reference backend maps reservations to private CPU generators; a future
    native Philox implementation needs a distinct checkpoint identity. Both
    wgrad operands must share the same RHT reservation. Reservations alone do
    not generate random numbers or establish parity with another backend.
    """

    recipe_fingerprint: str
    seed: int
    stream_id: str = "default"
    sr_offset: int = 0
    rht_offset: int = 0

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Invalid FP4 RNG seed")
        if any(type(v) is not int or v < 0 for v in (self.sr_offset, self.rht_offset)):
            raise ValueError("Invalid FP4 RNG offset")
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ValueError("FP4 RNG requires a nonempty stream_id")

    @classmethod
    def for_config(cls, config, stream_id="default"):
        return cls(config.fingerprint, config.rng_seed, stream_id)

    def reserve(self, stream, blocks):
        if stream not in ("sr", "rht") or type(blocks) is not int or blocks < 0:
            raise ValueError("Expected sr/rht stream and nonnegative integer block count")
        field = stream + "_offset"
        offset = getattr(self, field)
        setattr(self, field, offset + blocks)
        # Domain-separated numeric keys: a backend taking only seed/offset
        # cannot accidentally reuse the SR stream for RHT. The caller supplies
        # a stable rank/module stream_id. Python's randomized hash is not used.
        digest = hashlib.sha256(f"fp4-v1:{self.seed}:{self.stream_id}:{stream}".encode()).digest()
        key = int.from_bytes(digest[:8], "little") & (2**63 - 1)
        return {"seed": key, "stream": stream, "offset": offset, "blocks": blocks}

    def state_dict(self):
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_state_dict(cls, state, config, stream_id="default"):
        state = dict(state)
        if state.pop("version", None) != 1 or state.get("recipe_fingerprint") != config.fingerprint:
            raise ValueError("FP4 RNG checkpoint recipe/version mismatch")
        if state.get("seed") != config.rng_seed:
            raise ValueError("FP4 RNG checkpoint seed mismatch")
        if state.get("stream_id") != stream_id:
            raise ValueError("FP4 RNG checkpoint stream_id mismatch")
        return cls(**state)
