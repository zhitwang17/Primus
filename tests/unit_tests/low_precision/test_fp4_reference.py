# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Numerical CPU tests for real FP4 packing and the private reference RNG.

Run with --confcutdir=tests/unit_tests/low_precision to avoid GPU fixtures.
The loader bypasses training-package initialization, not the tested operators.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_primus_fp4_reference_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
sys.modules[PACKAGE] = package
C = importlib.import_module(PACKAGE + ".fp4_contract")
R = importlib.import_module(PACKAGE + ".fp4_reference")


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_e2m1_codebook_signed_zero_and_nearest_even(recipe):
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0, gradient_sr=False)
    codebook = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    x = torch.cat((codebook, -codebook)).reshape(1, 16)
    packed = R.quantize(x, cfg, cfg.operand("fwd", "input"), outer_scale=torch.ones(1))
    assert packed.data.tolist() == [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]]
    recovered = R.dequantize(packed)
    assert torch.equal(x, recovered)
    assert torch.equal(torch.signbit(x), torch.signbit(recovered))

    x = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 2])
    expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6] * 2], dtype=torch.float32)
    actual = R.qdq(x, cfg, cfg.operand("fwd", "input"), outer_scale=torch.ones(1))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_all_inner_grid_codes_and_midpoints_match_scalar_oracle(recipe):
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0, gradient_sr=False)
    bias, maximum = (7, 126) if recipe == "nvfp4" else (15, 254)
    grid = [
        (i & 7) / 8 * 2 ** (1 - bias) if i < 8 else (1 + (i & 7) / 8) * 2 ** ((i >> 3) - bias)
        for i in range(maximum + 1)
    ]
    samples = grid + [(a + b) / 2 for a, b in zip(grid[:-1], grid[1:])]
    x = (torch.tensor(samples, dtype=torch.float32) * 6)[:, None].repeat(1, 16)
    packed = R.quantize(x, cfg, cfg.operand("fwd", "input"), outer_scale=torch.ones(1))
    expected = torch.tensor([cfg.inner_scale(float(value), 1.0) for value in x[:, 0]])
    assert torch.equal(packed.inner_scale[:, 0], expected)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("direction,role", [("fwd", "input"), ("dgrad", "weight"), ("fwd", "weight")])
@pytest.mark.parametrize("shape", [(17, 33), (2, 17, 33)])
def test_axis_layout_padding_and_outer_roundtrip(recipe, direction, role, shape):
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0, gradient_sr=False)
    x = torch.linspace(-5, 7, steps=torch.tensor(shape).prod().item()).reshape(shape)
    x = x.transpose(-1, -2)
    packed = R.quantize(x, cfg, cfg.operand(direction, role))
    packed.validate()
    assert tuple(packed.data.shape) == packed.layout.storage_shapes(cfg)[0]
    assert packed.layout.original_shape == tuple(x.shape)
    recovered = R.dequantize(packed)
    assert recovered.shape == x.shape
    assert recovered.is_contiguous()
    assert torch.isfinite(recovered).all()
    assert (recovered - x).square().mean().sqrt() / x.square().mean().sqrt() < 0.2
    expected_outer = torch.tensor([cfg.outer_scale(float(x.abs().max()))])
    torch.testing.assert_close(packed.outer_scale, expected_outer, rtol=1e-7, atol=0)
    restored = C.TwoLevelFP4Tensor.from_metadata(
        packed.data, packed.inner_scale, packed.outer_scale, packed.metadata()
    )
    assert torch.equal(R.dequantize(restored), recovered)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_weight_2d_blocks_share_one_scale_and_bank_outer(recipe):
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0, gradient_sr=False)
    weights = torch.arange(3 * 19 * 35, dtype=torch.float32).reshape(3, 19, 35) / 100
    outer = R.compute_outer_scale(weights, cfg)
    bank = R.quantize(weights, cfg, cfg.operand("fwd", "weight"), outer_scale=outer)
    assert bank.inner_scale.shape == (3, 2, 3)
    for expert in range(3):
        separate = R.quantize(weights[expert], cfg, cfg.operand("fwd", "weight"), outer_scale=outer)
        assert torch.equal(bank.data[expert], separate.data)
        assert torch.equal(bank.inner_scale[expert], separate.inner_scale)
    outer.zero_()
    assert bank.outer_scale.item() > 0  # Caller-owned mutable buffer is not aliased.


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_weight_1d_and_2d_scaling_are_distinct(recipe):
    rowwise = C.TwoLevelFP4Config(recipe, weight_2d=False, rht_size=0, gradient_sr=False)
    tiled = C.TwoLevelFP4Config(recipe, weight_2d=True, rht_size=0, gradient_sr=False)
    x = torch.ones(16, 16)
    x[-1] = 1000
    outer = R.compute_outer_scale(x, rowwise)
    a = R.quantize(x, rowwise, rowwise.operand("fwd", "weight"), outer_scale=outer)
    b = R.quantize(x, tiled, tiled.operand("fwd", "weight"), outer_scale=outer)
    assert a.inner_scale.shape == (16, 1) and b.inner_scale.shape == (1, 1)
    assert a.inner_scale[0] < b.inner_scale[0]
    assert not torch.equal(R.dequantize(a), R.dequantize(b))


@pytest.mark.parametrize("tokens", [0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129])
def test_wgrad_axis_keeps_shape_and_pads_rht_alignment(tokens):
    cfg = C.TwoLevelFP4Config("amdfp4", gradient_sr=False)
    x = torch.arange(tokens * 7, dtype=torch.float32).reshape(tokens, 7)
    packed = R.quantize(x, cfg, cfg.operand("wgrad", "input"))
    assert packed.layout.original_shape == (tokens, 7)
    assert packed.layout.padded_shape == (((tokens + 31) // 32) * 32, 7)
    assert packed.data.shape == (7, packed.layout.padded_shape[0] // 2)
    assert R.dequantize(packed).shape == x.shape


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("shape", [(0, 17), (3, 0), (0, 0), (2, 0, 17)])
def test_zero_length_and_all_zero_inputs(recipe, shape):
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0, gradient_sr=False)
    for role in ("input", "weight"):
        x = torch.zeros(shape)
        packed = R.quantize(x, cfg, cfg.operand("fwd", role))
        assert torch.equal(R.dequantize(packed), x)
        assert packed.outer_scale.item() == pytest.approx(1e-30)
    x = torch.zeros(3, 17)
    packed = R.quantize(x, cfg, cfg.operand("fwd", "input"))
    assert torch.equal(packed.inner_scale, torch.full_like(packed.inner_scale, cfg.inner_floor))


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_stochastic_rounding_is_unbiased_replayable_and_private(recipe):
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0)
    random = C.FP4RandomState.for_config(cfg, "rank0/layer0")
    x = torch.full((4096, 16), 1.25)
    x[:, -1] = 6
    before = torch.random.get_rng_state().clone()
    saved = random.state_dict()
    first = R.qdq(x, cfg, cfg.operand("dgrad", "grad_output"), outer_scale=torch.ones(1), random=random)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert abs(first[:, :-1].mean().item() - 1.25) < 0.006
    assert set(first[:, :-1].unique().tolist()) == {1.0, 1.5}
    assert torch.equal(first[:, -1], x[:, -1])
    assert random.sr_offset == x.numel() // 4 and random.rht_offset == 0
    restored = C.FP4RandomState.from_state_dict(saved, cfg, "rank0/layer0")
    replay = R.qdq(x, cfg, cfg.operand("dgrad", "grad_output"), outer_scale=torch.ones(1), random=restored)
    assert torch.equal(first, replay)
    second = R.qdq(x, cfg, cfg.operand("dgrad", "grad_output"), outer_scale=torch.ones(1), random=random)
    assert not torch.equal(first, second)


@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("randomized", [False, True])
def test_rht_preserves_wgrad_product_and_tail_rows(size, randomized):
    cfg = C.TwoLevelFP4Config("nvfp4", rht_size=size, rht_randomized=randomized)
    rng = C.FP4RandomState.for_config(cfg)
    x = torch.linspace(-1, 1, 17 * 7).reshape(17, 7)
    dy = torch.cos(torch.arange(17 * 9, dtype=torch.float32)).reshape(17, 9)
    global_state = torch.random.get_rng_state().clone()
    saved = rng.state_dict()
    xh, dyh = R.rht_pair(x, dy, cfg, random=rng)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert xh.shape == (32, 7) and dyh.shape == (32, 9)
    torch.testing.assert_close(dyh.T @ xh, dy.T @ x, atol=2e-6, rtol=2e-5)
    if randomized:
        assert rng.rht_offset == 2 * size // 4
    else:
        assert rng.rht_offset == 0
    assert rng.sr_offset == 0
    replay_rng = C.FP4RandomState.from_state_dict(saved, cfg)
    replay = R.rht_pair(x, dy, cfg, random=replay_rng)
    assert torch.equal(xh, replay[0]) and torch.equal(dyh, replay[1])


def test_bf16_rounding_boundary_and_deterministic_calls_do_not_consume_rng():
    cfg = C.TwoLevelFP4Config("amdfp4", gradient_sr=False, rht_size=0)
    random = C.FP4RandomState.for_config(cfg)
    x = torch.linspace(-1, 1, 17 * 13).reshape(17, 13).bfloat16()
    before = random.state_dict()
    result = R.qdq(x, cfg, cfg.operand("wgrad", "input"), random=random)
    assert result.dtype == torch.bfloat16 and result.shape == x.shape
    assert random.state_dict() == before
    xh, yh = R.rht_pair(x, x, cfg, random=random)
    assert xh is x and yh is x


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_are_rejected(bad):
    cfg = C.TwoLevelFP4Config("nvfp4")
    with pytest.raises(ValueError, match="non-finite"):
        R.quantize(torch.tensor([[bad]]), cfg, cfg.operand("fwd", "input"))


def test_invalid_outer_random_identity_and_shape_are_rejected():
    cfg = C.TwoLevelFP4Config("nvfp4")
    x = torch.ones(3, 17)
    for outer in (torch.zeros(1), torch.tensor([float("nan")]), torch.ones(2), torch.ones(1).bfloat16()):
        with pytest.raises(ValueError, match="outer_scale"):
            R.quantize(x, cfg, cfg.operand("fwd", "input"), outer_scale=outer)
    with pytest.raises(ValueError, match="combined inner/outer"):
        R.quantize(torch.zeros_like(x), cfg, cfg.operand("fwd", "input"), outer_scale=torch.tensor([1e-45]))
    with pytest.raises(ValueError, match="explicit FP4RandomState"):
        R.quantize(x, cfg, cfg.operand("dgrad", "grad_output"))
    wrong = C.FP4RandomState.for_config(C.TwoLevelFP4Config("amdfp4"))
    with pytest.raises(ValueError, match="does not match"):
        R.quantize(x, cfg, cfg.operand("dgrad", "grad_output"), random=wrong)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        R.quantize(torch.ones(3), cfg, cfg.operand("fwd", "input"))
