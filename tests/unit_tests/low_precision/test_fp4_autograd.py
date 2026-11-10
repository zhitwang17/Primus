# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Reference three-direction, routed-tail, autograd and recompute checks."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fp4_preflight_autograd", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, D = preflight.load_contracts()
A = importlib.import_module(C.__package__ + ".fp4_autograd")
Q = importlib.import_module(C.__package__ + ".fp4_reference")


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_dense_three_directions_and_parameter_hooks(recipe):
    torch.manual_seed(31)
    cfg = C.TwoLevelFP4Config(recipe, rht_size=0, gradient_sr=False)
    x = torch.randn(2, 17, 19, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(23, 19))
    dy = torch.randn(2, 17, 23)
    observed = []
    w.register_hook(lambda grad: observed.append(grad.clone()))
    w.main_grad = torch.zeros_like(w, dtype=torch.float32)

    def accumulate(grad):
        w.main_grad.add_(grad.float())

    w.register_hook(accumulate)
    y = A.linear(x, w, cfg)
    assert not torch.equal(y, torch.nn.functional.linear(x, w))
    y.backward(dy)
    q = lambda t, direction, role: Q.qdq(t, cfg, cfg.operand(direction, role))
    expected_dx = q(dy.reshape(-1, 23), "dgrad", "grad_output") @ q(w, "dgrad", "weight")
    expected_dw = q(dy.reshape(-1, 23), "wgrad", "grad_output").t() @ q(x.reshape(-1, 19), "wgrad", "input")
    torch.testing.assert_close(x.grad, expected_dx.reshape_as(x), rtol=0, atol=0)
    torch.testing.assert_close(w.grad, expected_dw, rtol=0, atol=0)
    assert len(observed) == 1
    torch.testing.assert_close(w.main_grad, expected_dw, rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_grouped_empty_and_all_tail_boundaries(recipe):
    torch.manual_seed(14)
    lengths = [0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129]
    cfg = C.TwoLevelFP4Config(recipe, gradient_sr=False, rht_size=0)
    x = torch.randn(sum(lengths), 17, requires_grad=True)
    weights = torch.nn.Parameter(torch.randn(len(lengths), 19, 17))
    # Make the bank-wide outer statistic observably different from each expert's.
    with torch.no_grad():
        weights[4] *= 1000
    saved_id = id(weights)
    output = A.grouped_linear(x, weights, lengths, cfg)
    assert output.shape == (sum(lengths), 19)
    wq = Q.qdq(weights, cfg, cfg.operand("fwd", "weight"))
    xq = Q.qdq(x, cfg, cfg.operand("fwd", "input"))
    expected = torch.cat([a @ b.t() for a, b in zip(xq.split(lengths), wq)])
    # Padded/unpadded GEMM shapes can select different FP32 reduction orders.
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    output.square().mean().backward()
    assert id(weights) == saved_id and torch.isfinite(weights.grad).all()
    assert torch.count_nonzero(weights.grad[0]) == 0
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    # A second microbatch adds to existing grads, including a zero empty expert.
    grad = weights.grad.clone()
    A.grouped_linear(x, weights, lengths, cfg).square().mean().backward()
    torch.testing.assert_close(weights.grad, grad * 2, rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_all_empty_experts_have_explicit_zero_wgrad(recipe):
    cfg = C.TwoLevelFP4Config(recipe)
    rng = C.FP4RandomState.for_config(cfg, "empty")
    x = torch.empty(0, 16, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(3, 32, 16))
    A.grouped_linear(x, w, [0, 0, 0], cfg, rng).sum().backward()
    assert x.grad.shape == x.shape
    assert torch.equal(w.grad, torch.zeros_like(w))
    assert rng.sr_offset == rng.rht_offset == 0


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("reentrant", [True, False])
def test_activation_recompute_preserves_private_rng_and_gradients(recipe, reentrant):
    torch.manual_seed(71)
    cfg = C.TwoLevelFP4Config(recipe)
    x = torch.randn(33, 16, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(32, 16))
    rng = C.FP4RandomState.for_config(cfg, "rank0/layer1")
    initial_rng = rng.state_dict()
    torch_rng = torch.get_rng_state().clone()
    A.linear(x, w, cfg, rng).square().mean().backward()
    expected_x, expected_w, expected_rng = x.grad.clone(), w.grad.clone(), rng.state_dict()
    assert torch.equal(torch_rng, torch.get_rng_state())
    x.grad = w.grad = None
    replay = C.FP4RandomState.from_state_dict(initial_rng, cfg, "rank0/layer1")
    checkpoint(lambda z: A.linear(z, w, cfg, replay), x, use_reentrant=reentrant).square().mean().backward()
    torch.testing.assert_close(x.grad, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(w.grad, expected_w, rtol=0, atol=0)
    assert replay.state_dict() == expected_rng
    assert torch.equal(torch_rng, torch.get_rng_state())


def test_wrong_rng_and_group_lengths_fail_before_gemm():
    cfg = C.TwoLevelFP4Config("nvfp4")
    x, w = torch.randn(3, 16), torch.randn(32, 16)
    with pytest.raises(ValueError, match="private FP4RandomState"):
        A.linear(x, w, cfg)
    rng = C.FP4RandomState.for_config(C.TwoLevelFP4Config("amdfp4"))
    with pytest.raises(ValueError, match="does not match"):
        A.linear(x, w, cfg, rng)
    with pytest.raises(ValueError, match="span"):
        A.grouped_linear(x, w.unsqueeze(0), [2], cfg, rng)
