"""The reference QKV must consume externally normalized inputs exactly once."""

import pytest
import torch
from test_fp4_modules import config, modules  # noqa: F401


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_external_rmsnorm_preserves_output_and_single_gamma_gradient(
    modules, monkeypatch, recipe, zero_centered
):
    monkeypatch.setattr(modules.M.TwoLevelFP4Linear, "forward", lambda self, x, **kw: (x, kw.get("tag")))
    cfg = config(fp4_recipe=recipe, layernorm_zero_centered_gamma=zero_centered)
    layer = modules.M.TwoLevelFP4LayerNormColumnParallelLinear(
        16, 32, config=cfg, init_method=cfg.init_method, bias=False
    )
    with torch.no_grad():
        layer.layer_norm_weight.copy_(torch.linspace(0.2, 1.9, 16))
    x = torch.randn(3, 16, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    expected_gamma = layer.layer_norm_weight.detach().clone().requires_grad_()

    def norm(value, weight):
        gamma = weight + 1 if zero_centered else weight
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + cfg.layernorm_epsilon) * gamma

    normalized = norm(x, layer.layer_norm_weight)
    layer._skip_fused_norm = True
    actual, tag = layer(normalized, tag="forwarded")
    expected = norm(expected_x, expected_gamma)
    assert actual is normalized and tag == "forwarded"
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(layer.layer_norm_weight.grad, expected_gamma.grad, rtol=0, atol=0)
    assert layer.fp4_external_rmsnorm_calls == 1
    layer._skip_fused_norm = False
    ordinary, _ = layer(expected_x)
    torch.testing.assert_close(ordinary, expected, rtol=0, atol=0)
    assert layer.fp4_external_rmsnorm_calls == 1


def test_external_layernorm_is_not_silently_accepted(modules):
    cfg = config(normalization="LayerNorm")
    layer = modules.M.TwoLevelFP4LayerNormColumnParallelLinear(
        16, 32, config=cfg, init_method=cfg.init_method, bias=False
    )
    layer._skip_fused_norm = True
    with pytest.raises(ValueError, match="requires RMSNorm"):
        layer(torch.randn(2, 16))
