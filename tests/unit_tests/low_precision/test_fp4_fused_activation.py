"""Verify reference experts route to the native fused activation contract.

The kernel boundary is replaced by differentiable CPU math; GPU kernel
qualification remains a separate admission check.
"""

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from test_fp4_modules import config, modules  # noqa: F401


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("activation", [F.silu, F.gelu])
def test_fused_route_forwards_counts_and_preserves_gradients(modules, monkeypatch, recipe, activation):
    calls = []

    def fused(x, bias, probs, lengths, name, clamp):
        calls.append((bias, lengths.tolist(), name, clamp))
        gate, up = x.chunk(2, dim=-1)
        return activation(gate) * up * probs.unsqueeze(-1)

    stub = ModuleType("primus.backends.megatron.core.extensions.primus_turbo")
    stub.fused_bias_act_with_probs = fused
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    cfg = config(
        fp4_recipe=recipe,
        use_turbo_fused_act_with_probs=True,
        activation_func=activation,
        glu_linear_offset=0.0,
        activation_func_clamp_value=None,
        add_bias_linear=False,
    )
    model = modules.M.TwoLevelFP4GroupedMLP(2, cfg)
    x = torch.randn(5, 32, requires_grad=True)
    probs = torch.rand(5, 1, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    expected_probs = probs.detach().clone().requires_grad_()
    actual = model._activate(x, probs, [3, 2])
    gate, up = expected_x.chunk(2, dim=-1)
    expected = activation(gate) * up * expected_probs
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, expected_probs.grad, rtol=0, atol=0)
    assert calls == [(None, [3, 2], "silu" if activation is F.silu else "gelu", None)]
    assert model.fp4_fused_activation_calls == 1


@pytest.mark.parametrize(
    "update",
    [
        dict(glu_linear_offset=1.0),
        dict(moe_mlp_glu_interleave_size=2),
        dict(activation_func_fp8_input_store=True),
        dict(gated_linear_unit=False),
    ],
)
def test_unsupported_fused_contract_is_rejected(modules, update):
    values = dict(use_turbo_fused_act_with_probs=True, glu_linear_offset=0.0, add_bias_linear=False)
    values.update(update)
    model = modules.M.TwoLevelFP4GroupedMLP(2, config(**values))
    with pytest.raises(ValueError, match="Turbo FP4 activation requires"):
        model._activate(torch.randn(5, 32), torch.rand(5), [3, 2])


def test_fused_route_requires_counts(modules):
    model = modules.M.TwoLevelFP4GroupedMLP(
        2, config(use_turbo_fused_act_with_probs=True, glu_linear_offset=0.0, add_bias_linear=False)
    )
    with pytest.raises(ValueError, match="expert token counts"):
        model._activate(torch.randn(5, 32), torch.rand(5))
