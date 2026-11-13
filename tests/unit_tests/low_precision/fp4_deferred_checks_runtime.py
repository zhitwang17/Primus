# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Real Triton kernels under CPU interpretation or explicitly allocated hardware.

Called explicitly by test_fp4_deferred_checks.py, not normal test discovery.
CPU interpretation bypasses only the CUDA placement gate; all validation,
quantization, RHT and autograd operations execute. Device mode patches nothing.
"""

import importlib
import os
from dataclasses import replace

import pytest
import torch
from test_fp4_deferred_checks import C, R

T = importlib.import_module(C.__package__ + ".fp4_triton")
A = importlib.import_module(C.__package__ + ".fp4_autograd")
DEVICE = torch.device(os.environ["FP4_VALUE_CHECKS_DEVICE"])


@pytest.fixture(autouse=True)
def interpreter_placement_only(monkeypatch):
    if DEVICE.type == "cpu":
        assert os.environ.get("TRITON_INTERPRET") == "1"

        def input_on_cpu(x):
            assert isinstance(x, torch.Tensor) and x.device.type == "cpu"
            assert x.dtype in (torch.float32, torch.bfloat16) and x.ndim in (2, 3)

        monkeypatch.setattr(T, "_input", input_on_cpu)
    else:
        assert DEVICE.type == "cuda" and torch.cuda.is_available()
        assert os.environ.get("TRITON_INTERPRET") == "0"


def tensor(rows=17, cols=19):
    return torch.linspace(-2, 3, rows * cols, device=DEVICE).reshape(rows, cols)


def rng_position(rng):
    return rng.seed, rng.stream_id, rng.sr_offset, rng.rht_offset


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_amax_sanitization_cannot_erase_nonfinite_evidence(recipe, bad):
    cfg = C.TwoLevelFP4Config(recipe, quantizer="triton")
    x = tensor()
    x[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite input"):
        with T.DeferredValueChecks(DEVICE) as validation:
            outer = T.compute_outer_scale(x, cfg, validation=validation)
            # The reduction deliberately replaces invalid values by zero; its
            # apparently valid scale must not make the original input acceptable.
            assert bool(torch.isfinite(outer).all()) and bool((outer > 0).all())
            T.qdq(tensor(), cfg, cfg.operand("fwd", "input"), validation=validation)
            assert validation.flags[0].item() == 1
            assert validation.flags.shape == (3,) and validation.flags.dtype == torch.int32
    assert validation.closed


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_valid_supplied_outer_does_not_bypass_nonfinite_input_checks(recipe):
    cfg = C.TwoLevelFP4Config(recipe, quantizer="triton")
    x = tensor()
    x[0, 0] = float("nan")
    outer = torch.ones(1, dtype=torch.float32, device=DEVICE)
    with pytest.raises(ValueError, match="non-finite input"):
        with T.DeferredValueChecks(DEVICE) as validation:
            # No amax reduction is called: this must be caught in the QDQ kernel.
            T.qdq(x, cfg, cfg.operand("fwd", "input"), outer_scale=outer, validation=validation)
            assert validation.flags[0].item() == 1
            assert validation.flags[1].item() == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
@pytest.mark.parametrize("rows", [0, 17])
def test_invalid_supplied_outer_is_detected_even_for_empty_input(rows, bad):
    cfg = C.TwoLevelFP4Config("nvfp4", quantizer="triton")
    x = tensor(rows)
    outer = torch.tensor([bad], dtype=torch.float32, device=DEVICE)
    with pytest.raises(ValueError, match="outer scale"):
        with T.DeferredValueChecks(DEVICE) as validation:
            T.qdq(x, cfg, cfg.operand("fwd", "input"), outer_scale=outer, validation=validation)
            assert validation.flags[0].item() == 0
            assert validation.flags[1].item() == 1


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_combined_underflow_is_not_mistaken_for_valid_positive_outer(recipe):
    cfg = C.TwoLevelFP4Config(recipe, quantizer="triton")
    x = torch.zeros(17, 19, device=DEVICE)
    outer = torch.tensor([2.0**-149], dtype=torch.float32, device=DEVICE)
    assert outer.item() > 0
    with pytest.raises(ValueError, match="combined"):
        T.qdq(x, cfg, cfg.operand("fwd", "input"), outer_scale=outer)
    with pytest.raises(ValueError, match="combined"):
        with T.DeferredValueChecks(DEVICE) as validation:
            T.qdq(x, cfg, cfg.operand("fwd", "input"), outer_scale=outer, validation=validation)
            assert validation.flags.cpu().tolist() == [0, 0, 1]


def test_direct_calls_are_strict_even_with_deferred_config_and_nested_scope():
    cfg = C.TwoLevelFP4Config("nvfp4", quantizer="triton", value_checks="grouped_deferred")
    x = tensor()
    x[0, 0] = float("nan")
    with T.DeferredValueChecks(DEVICE) as outer:
        for operation in (T.qdq, T.quantize):
            with pytest.raises(ValueError, match="non-finite"):
                operation(x, cfg, cfg.operand("fwd", "input"))
        with pytest.raises(ValueError, match="non-finite"):
            T.compute_outer_scale(x, cfg)
        assert outer.flags.cpu().tolist() == [0, 0, 0]


def test_nested_explicit_scopes_are_independent_and_closed_scopes_reject_reuse():
    cfg = C.TwoLevelFP4Config("nvfp4", quantizer="triton")
    bad = tensor()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        with T.DeferredValueChecks(DEVICE) as outer:
            outer.check_input(bad)
            with T.DeferredValueChecks(DEVICE) as inner:
                T.qdq(tensor(), cfg, cfg.operand("fwd", "input"), validation=inner)
            assert inner.closed and not outer.closed
            assert outer.flags[0].item() == 1
    for scope in (inner, outer):
        with pytest.raises(RuntimeError, match="closed"):
            scope.check_input(tensor())
        with pytest.raises(RuntimeError, match="closed"):
            scope.finish()


def test_original_exception_is_preserved_without_leaking_pending_flags():
    bad = tensor()
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="original operation failed"):
        with T.DeferredValueChecks(DEVICE) as validation:
            validation.check_input(bad)
            raise RuntimeError("original operation failed")
    assert validation.closed
    with T.DeferredValueChecks(DEVICE) as clean:
        clean.check_input(tensor())
        assert clean.flags.cpu().tolist() == [0, 0, 0]


def test_cross_device_and_explicit_check_disabling_cannot_bypass_validation():
    cfg = C.TwoLevelFP4Config("nvfp4", quantizer="triton")
    with T.DeferredValueChecks(DEVICE) as validation:
        with pytest.raises(ValueError, match="cross devices"):
            validation.check_input(torch.empty((16, 16), device="meta"))
        with pytest.raises(ValueError, match="disable value checks"):
            T.qdq(tensor(), cfg, cfg.operand("fwd", "input"), validation=validation, check_values=False)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("sr_rng", ["torch_cpu", "triton_philox"])
def test_all_qdq_directions_preserve_outputs_and_rng_reservations(recipe, sr_rng):
    cfg = C.TwoLevelFP4Config(recipe, quantizer="triton", sr_rng=sr_rng)
    strict, deferred = (C.FP4RandomState.for_config(cfg, "independent-parity") for _ in range(2))
    for direction, role in (
        ("fwd", "input"),
        ("fwd", "weight"),
        ("dgrad", "grad_output"),
        ("dgrad", "weight"),
        ("wgrad", "input"),
        ("wgrad", "grad_output"),
    ):
        x = tensor()
        if role == "weight":
            x = torch.stack((x, x * 17))  # Bank scale, tail blocks and both weight axes.
        operand = cfg.operand(direction, role)
        expected = T.qdq(x, cfg, operand, random=strict)
        with T.DeferredValueChecks(DEVICE) as validation:
            actual = T.qdq(x, cfg, operand, random=deferred, validation=validation)
        assert torch.equal(actual, expected), (recipe, sr_rng, direction, role)
        assert strict.state_dict() == deferred.state_dict()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("sr_rng", ["torch_cpu", "triton_philox"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_grouped_autograd_matches_strict_and_flushes_before_return(recipe, sr_rng, dtype, monkeypatch):
    cfg = C.TwoLevelFP4Config(recipe, quantizer="triton", sr_rng=sr_rng)
    lengths = [1, 0, 33]
    inputs = tensor(sum(lengths), 17).to(dtype)
    weights = tensor(3 * 19, 17).reshape(3, 19, 17).to(dtype)
    dy = tensor(sum(lengths), 19).to(dtype)
    snapshots, closed = [], []
    finish = T.DeferredValueChecks.finish

    def observe_finish(self):
        result = finish(self)
        closed.append(self.closed)
        return result

    monkeypatch.setattr(T.DeferredValueChecks, "finish", observe_finish)
    global_cpu_rng = torch.get_rng_state().clone()
    global_device_rng = torch.cuda.get_rng_state().clone() if DEVICE.type == "cuda" else None
    for mode in ("strict", "grouped_deferred"):
        current = replace(cfg, value_checks=mode)
        rng = C.FP4RandomState.for_config(current, "actual-grouped/" + current.rng_fingerprint)
        x, w = inputs.clone().requires_grad_(), weights.clone().requires_grad_()
        y = A.grouped_linear(x, w, lengths, current, rng)
        assert y.dtype == dtype
        assert len(closed) == (0 if mode == "strict" else 1)
        y.backward(dy)
        assert len(closed) == (0 if mode == "strict" else 2)
        snapshots.append((y.detach(), x.grad, w.grad, rng_position(rng)))
    for a, b in zip(snapshots[0][:3], snapshots[1][:3]):
        assert torch.equal(a, b)
    assert snapshots[0][3] == snapshots[1][3]
    assert all(closed)
    assert torch.equal(global_cpu_rng, torch.get_rng_state())
    if DEVICE.type == "cuda":
        assert torch.equal(global_device_rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize("where", ["input", "weights", "gradient"])
def test_grouped_invalid_values_raise_before_result_or_leaf_grad_hooks(where):
    cfg = C.TwoLevelFP4Config("nvfp4", quantizer="triton", value_checks="grouped_deferred")
    x = tensor(17, 16)
    weights = tensor(2 * 19, 16).reshape(2, 19, 16)
    dy = tensor(17, 19)
    {"input": x, "weights": weights, "gradient": dy}[where].reshape(-1)[0] = float("nan")
    x.requires_grad_()
    weights.requires_grad_()
    hooks = []
    x.register_hook(lambda grad: hooks.append("x"))
    weights.register_hook(lambda grad: hooks.append("weights"))
    rng = C.FP4RandomState.for_config(cfg)
    with pytest.raises(ValueError, match="non-finite"):
        y = A.grouped_linear(x, weights, [1, 16], cfg, rng)
        assert where == "gradient", "Invalid forward input escaped the grouped boundary"
        y.backward(dy)
    assert hooks == []
    assert x.grad is None and weights.grad is None


def test_rht_and_empty_experts_still_use_bounded_validation():
    cfg = C.TwoLevelFP4Config("amdfp4", quantizer="triton", value_checks="grouped_deferred")
    rng = C.FP4RandomState.for_config(cfg)
    x = torch.empty(0, 16, device=DEVICE, requires_grad=True)
    weights = tensor(2 * 19, 16).reshape(2, 19, 16).requires_grad_()
    A.grouped_linear(x, weights, [0, 0], cfg, rng).sum().backward()
    assert x.grad.shape == x.shape and torch.equal(weights.grad, torch.zeros_like(weights))
    assert rng.sr_offset == rng.rht_offset == 0
    bad = tensor()
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        with T.DeferredValueChecks(DEVICE) as validation:
            R.rht_pair(bad, tensor(), cfg, random=rng, validation=validation)
