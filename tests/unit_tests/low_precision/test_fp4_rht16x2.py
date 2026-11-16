# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Deterministic H16x2 recipe checks without allocating a GPU.

Turbo applies two independent H16 butterflies in each 32-value quantization
microblock. Its transform is FP32 until FP4 conversion; this reference route
retains the existing input-dtype boundary before quantization. Test both the
shared geometry and the boundary explicitly, without implying bitwise parity
between distinct quantization/GEMM implementations.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fp4_preflight_rht16x2", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, D = preflight.load_contracts()
R = importlib.import_module(C.__package__ + ".fp4_reference")
A = importlib.import_module(C.__package__ + ".fp4_autograd")


def butterfly_h16(value):
    """Independent scalar-lane oracle for native H4-local/H4-cross grouping."""
    padded = F.pad(value.float(), (0, 0, 0, (-value.shape[0]) % 16))
    blocks = padded.reshape(-1, 16, value.shape[-1])
    lanes = list(blocks.unbind(1))
    for bit in (1, 2, 4, 8):
        previous = lanes
        lanes = [
            previous[i] + previous[i ^ bit] if i & bit == 0 else previous[i ^ bit] - previous[i]
            for i in range(16)
        ]
    return (torch.stack(lanes, 1) * 0.25).reshape(-1, value.shape[-1])


def aligned_config(recipe, **options):
    return C.TwoLevelFP4Config(recipe, rht_size=16, rht_randomized=False, **options)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 31, 32, 33, 65])
def test_h16x2_geometry_tail_and_paired_cancellation(recipe, tokens):
    cfg = aligned_config(recipe)
    # Integer-valued data makes all FP32 sums exact, separating the transform
    # geometry from unrelated matrix-multiplication reduction order.
    x = (torch.arange(tokens * 5).reshape(tokens, 5) % 19 - 9).float()
    dy = (torch.arange(tokens * 7).reshape(tokens, 7) % 23 - 11).float()
    rng = C.FP4RandomState.for_config(cfg)
    before_private = rng.state_dict()
    before_global = torch.random.get_rng_state().clone()
    xh, dyh = R.rht_pair(x, dy, cfg, random=rng)
    torch.testing.assert_close(xh, butterfly_h16(x), rtol=0, atol=0)
    torch.testing.assert_close(dyh, butterfly_h16(dy), rtol=0, atol=0)
    torch.testing.assert_close(dyh.T @ xh, dy.T @ x, rtol=0, atol=0)
    assert rng.state_dict() == before_private
    assert torch.equal(torch.random.get_rng_state(), before_global)
    # Deterministic RHT needs no private RNG at all, even with gradient SR on.
    no_rng = R.rht_pair(x, dy, cfg)
    assert torch.equal(no_rng[0], xh) and torch.equal(no_rng[1], dyh)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_h16x2_does_not_mix_32_value_halves_or_expert_boundaries(recipe):
    cfg = aligned_config(recipe)
    x = torch.zeros(32, 2)
    x[0, 0] = 4
    x[16, 1] = 8
    xh, _ = R.rht_pair(x, x, cfg)
    assert torch.equal(xh[:16, 0], torch.ones(16))
    assert torch.equal(xh[16:, 1], torch.full((16,), 2.0))
    assert torch.count_nonzero(xh[:16, 1]) == torch.count_nonzero(xh[16:, 0]) == 0

    # Grouped autograd pads each expert independently before calling rht_pair.
    # A one-token first expert must not share an H16 with the following expert.
    lengths = (1, 0, 17)
    routed = torch.cat([torch.full((1, 2), 4.0), torch.full((17, 2), 8.0)])
    parts = A._split_pad(routed, lengths)
    transformed = [R.rht_pair(part, part, cfg)[0] for part in parts]
    assert [part.shape[0] for part in transformed] == [128, 0, 128]
    assert torch.equal(transformed[0][:16], torch.ones(16, 2))
    assert torch.count_nonzero(transformed[0][16:]) == 0
    for part, actual in zip(parts, transformed):
        if part.numel():
            torch.testing.assert_close(actual, butterfly_h16(part), rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_reference_bf16_intermediate_is_explicit_and_preserved(recipe):
    cfg = aligned_config(recipe)
    x = torch.linspace(-0.9, 1.3, 32 * 17).reshape(32, 17).bfloat16()
    xh, dyh = R.rht_pair(x, x, cfg)
    native_fp32 = butterfly_h16(x)
    assert xh.dtype == dyh.dtype == torch.bfloat16
    torch.testing.assert_close(xh, native_fp32.bfloat16(), rtol=0, atol=0)
    assert torch.count_nonzero(xh.float() != native_fp32) > 0


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("grouped", [False, True])
def test_actual_autograd_uses_deterministic_h16_and_only_sr_stream(recipe, grouped, monkeypatch):
    cfg = aligned_config(recipe)
    rng = C.FP4RandomState.for_config(cfg)
    x = torch.linspace(-1, 1, 33 * 17).reshape(33, 17).bfloat16().requires_grad_()
    shape = (3, 19, 17) if grouped else (19, 17)
    weight = torch.linspace(-0.8, 0.9, 19 * 17 * (3 if grouped else 1)).reshape(shape)
    weight = weight.bfloat16().requires_grad_()
    observed = []
    original = A.rht_pair

    def capture(a, b, config, **kwargs):
        result = original(a, b, config, **kwargs)
        observed.append((a.shape[0], config.rht_size, config.rht_randomized))
        if a.numel():
            torch.testing.assert_close(result[0], butterfly_h16(a).to(a.dtype), rtol=0, atol=0)
        return result

    monkeypatch.setattr(A, "rht_pair", capture)
    before_global = torch.random.get_rng_state().clone()
    y = A.grouped_linear(x, weight, [1, 0, 32], cfg, rng) if grouped else A.linear(x, weight, cfg, rng)
    y.float().square().mean().backward()
    assert observed == (
        [(128, 16, False), (0, 16, False), (128, 16, False)] if grouped else [(33, 16, False)]
    )
    assert rng.rht_offset == 0 and rng.sr_offset > 0
    assert torch.equal(torch.random.get_rng_state(), before_global)
    assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_configuration_and_checkpoint_identity_distinguish_h16x2(recipe):
    cfg = D.build_two_level_config({"fp4_recipe": recipe, "fp4_rht_size": 16, "fp4_rht_randomized": False})
    assert cfg == aligned_config(recipe)
    assert C.TwoLevelFP4Config(**cfg.to_dict()) == cfg
    old_h32 = C.TwoLevelFP4Config(recipe)
    random_h16 = C.TwoLevelFP4Config(recipe, rht_size=16)
    assert len({cfg.fingerprint, old_h32.fingerprint, random_h16.fingerprint}) == 3
    saved = C.FP4RandomState.for_config(old_h32, "rank0/layer0").state_dict()
    with pytest.raises(ValueError):
        C.FP4RandomState.from_state_dict(saved, cfg, "rank0/layer0")
