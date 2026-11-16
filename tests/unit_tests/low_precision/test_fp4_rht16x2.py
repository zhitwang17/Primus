# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Deterministic H16x2 recipe checks without allocating a GPU.

Turbo applies two independent H16 butterflies in each 32-value quantization
microblock. Explicit mxfp4 mode preserves its FP32 transform until conversion;
legacy mode retains the previous input-dtype boundary and checkpoint identity.
Shared RHT arithmetic does not imply bitwise parity of distinct FP4 formats.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fp4_preflight_rht16x2", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, D = preflight.load_contracts()
R = importlib.import_module(C.__package__ + ".fp4_reference")
A = importlib.import_module(C.__package__ + ".fp4_autograd")
S = importlib.import_module(C.__package__ + ".fp4_state")


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
@pytest.mark.parametrize("mode", ["legacy", "mxfp4"])
def test_h16x2_geometry_tail_and_paired_cancellation(recipe, tokens, mode):
    cfg = aligned_config(recipe, rht_mode=mode)
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
@pytest.mark.parametrize("mode", ["legacy", "mxfp4"])
def test_h16x2_does_not_mix_32_value_halves_or_expert_boundaries(recipe, mode):
    cfg = aligned_config(recipe, rht_mode=mode)
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
@pytest.mark.parametrize("mode", ["legacy", "mxfp4"])
def test_actual_autograd_uses_deterministic_h16_and_only_sr_stream(recipe, grouped, mode, monkeypatch):
    cfg = aligned_config(recipe, rht_mode=mode)
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
            expected_dtype = torch.float32 if mode == "mxfp4" else a.dtype
            torch.testing.assert_close(result[0], butterfly_h16(a).to(expected_dtype), rtol=0, atol=0)
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


@pytest.mark.parametrize(
    "options",
    [
        {"rht_mode": "unsupported"},
        {"rht_mode": "mxfp4", "rht_size": 32, "rht_randomized": False},
        {"rht_mode": "mxfp4", "rht_size": 0, "rht_randomized": False},
        {"rht_mode": "mxfp4", "rht_size": 16, "rht_randomized": True},
    ],
)
def test_mxfp4_mode_rejects_other_geometry_or_randomization(options):
    with pytest.raises(ValueError, match="rht_mode"):
        C.TwoLevelFP4Config("nvfp4", **options)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_legacy_fingerprints_and_new_mode_dispatch(recipe):
    # Frozen from f3c346f32 before rht_mode existed; these are resume identities,
    # not hashes computed from the implementation being tested.
    historical = {
        "nvfp4": (
            "8c24d7213a97670a3471e844f426c1a8ea278a5687ece9d865ad024a3e6b41f3",
            "98c5bb4f221ed1275e84f62fc4dc171578f51c0c584cef1a59c6d220cfe258f4",
        ),
        "amdfp4": (
            "a4ce47386d69dd2fe71bb27a9ea6bb20201dc360f1de6dc5446fa29a7e14dc69",
            "ac665392b9bc05d397bf3e078d4207d5021580b5cf10dd4713801aed06638fd2",
        ),
    }
    assert C.TwoLevelFP4Config(recipe).fingerprint == historical[recipe][0]
    old = aligned_config(recipe)
    assert old.fingerprint == historical[recipe][1]
    assert "rht_mode" not in old.to_dict()
    new = D.build_two_level_config(
        {
            "fp4_recipe": recipe,
            "fp4_rht_size": 16,
            "fp4_rht_randomized": False,
            "fp4_rht_mode": "mxfp4",
        }
    )
    assert new.rht_mode == "mxfp4" and new.to_dict()["rht_mode"] == "mxfp4"
    assert old.fingerprint != new.fingerprint
    assert old.rng_fingerprint == new.rng_fingerprint
    assert C.TwoLevelFP4Config(**new.to_dict()) == new


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("quantizer,sr_rng", [("torch", "torch_cpu"), ("triton", "triton_philox")])
def test_bound_model_sr_stream_is_shared_but_checkpoint_modes_are_distinct(recipe, quantizer, sr_rng):
    models = []
    for mode in ("legacy", "mxfp4"):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.zeros(16, 16, dtype=torch.bfloat16))
        model.fp4_config = aligned_config(recipe, rht_mode=mode, quantizer=quantizer, sr_rng=sr_rng)
        model.fp4_configs = (model.fp4_config,)
        S.bind_fp4_model(model, rank=3)
        models.append(model)
    old, new = models
    assert old.fp4_rng.stream_id == new.fp4_rng.stream_id
    assert old.fp4_rng.recipe_fingerprint != new.fp4_rng.recipe_fingerprint
    for words in (16, 128, 17):
        assert old.fp4_rng.reserve("sr", words) == new.fp4_rng.reserve("sr", words)
    assert old.fp4_rng.rht_offset == new.fp4_rng.rht_offset == 0
    before = S.fp4_state_dict(new, rank=3)
    with pytest.raises(ValueError, match="recipes"):
        S.load_fp4_state_dict(new, S.fp4_state_dict(old, rank=3), rank=3)
    assert S.fp4_state_dict(new, rank=3) == before


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("backward_inside", [False, True])
def test_mxfp4_wgrad_promotes_mixed_autocast_operands(grouped, backward_inside):
    cfg = aligned_config("nvfp4", rht_mode="mxfp4", gradient_sr=False)
    # Outside-autocast backward requests only wgrad: the older reference dgrad
    # route independently requires a matching autocast context for mixed dtypes.
    x = torch.linspace(-1, 1, 33 * 17).reshape(33, 17).requires_grad_(backward_inside)
    # Keep both experts nonempty: the legacy empty-part concatenation promotes
    # its BF16 matmul outputs to x.dtype, which would hide this mixed-dtype case.
    shape = (2, 19, 17) if grouped else (19, 17)
    weight = torch.linspace(-0.8, 0.9, 19 * 17 * (2 if grouped else 1)).reshape(shape).requires_grad_()
    products = []

    class ObserveMatmul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func is torch.ops.aten.mm.default:
                products.append((args[0].dtype, args[1].dtype, result.dtype))
            return result

    with ObserveMatmul():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = A.grouped_linear(x, weight, [1, 32], cfg) if grouped else A.linear(x, weight, cfg)
            assert y.dtype == torch.bfloat16
            loss = y.float().square().mean()
            if backward_inside:
                loss.backward()
        if not backward_inside:
            loss.backward()
    assert torch.isfinite(weight.grad).all() and weight.grad.dtype == torch.float32
    if backward_inside:
        assert torch.isfinite(x.grad).all()
    count = 2 if grouped else 1
    assert products[-count:] == [(torch.float32, torch.float32, torch.float32)] * count
