# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Comparison norm routing: production method bodies, GEMM boundary intercepted.

The CPU tests do not import TE/Turbo runtime or execute GEMMs. Device tests run
both actual norm routes on GPU, but still intercept GEMMs to isolate the norm.
"""
import ast
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_fp4_modules import ROOT, config, modules  # noqa: F401


@pytest.fixture
def mx_class(monkeypatch):
    path = ROOT / "primus/backends/megatron/core/extensions/primus_turbo.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "PrimusTurboLayerNormColumnParallelLinear"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_internal")
    node = ast.ClassDef(
        name="MXNormRoute",
        bases=[
            ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="nn", ctx=ast.Load()),
                attr="Module",
                ctx=ast.Load(),
            )
        ],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = {
        "torch": torch,
        "_take_fused_mxfp4_activation": lambda x: None,
        "_use_split_wgrad_op": lambda: False,
        "_fuse_wgrad_accum_pattern": lambda *args: None,
        "PrimusTurboLowPrecisionGlobalStateManager": SimpleNamespace(
            is_turbo_fp8_enabled=lambda: False, is_turbo_fp4_enabled=lambda: False
        ),
        "primus_turbo_torch": SimpleNamespace(ops=SimpleNamespace(gemm=lambda x, *args, **kwargs: x)),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace["MXNormRoute"], namespace


def bytes_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def make_nv(modules, monkeypatch, gamma_dtype, zero_centered, backend, device, hidden_size):
    # Intercept only the parent GEMM boundary, leaving the production norm body intact.
    monkeypatch.setattr(modules.M.TwoLevelFP4Linear, "forward", lambda self, x, **kwargs: (x, None))
    cfg = config(
        hidden_size=hidden_size,
        params_dtype=gamma_dtype,
        layernorm_zero_centered_gamma=zero_centered,
        fp4_comparison_rmsnorm_backend=backend,
    )
    result = modules.M.TwoLevelFP4LayerNormColumnParallelLinear(
        hidden_size, 16, config=cfg, init_method=lambda p: torch.nn.init.ones_(p), bias=False
    )
    return result.to(device)


def make_mx(mx_class, gamma, zero_centered, backend):
    cls, _ = mx_class
    result = cls()
    result.config = SimpleNamespace(normalization="RMSNorm")
    result.fp4_comparison_rmsnorm_backend = backend
    result.fp4_comparison_rmsnorm_calls = 0
    result.layer_norm_weight = torch.nn.Parameter(gamma.detach().clone())
    result.weight = torch.nn.Parameter(torch.empty(1, device=gamma.device))
    result.zero_centered_gamma = zero_centered
    result.eps = 1e-5
    result.use_bias = False
    return result


def check_routes(modules, monkeypatch, mx_class, device, gamma_dtype, zero_centered, strided, hidden_size=16):
    nv = make_nv(modules, monkeypatch, gamma_dtype, zero_centered, "torch", device, hidden_size)
    baseline = make_nv(modules, monkeypatch, gamma_dtype, zero_centered, "native", device, hidden_size)
    values = torch.linspace(-0.09, 0.09, hidden_size, dtype=torch.float32, device=device) + 0.0031
    with torch.no_grad():
        nv.layer_norm_weight.copy_(values)
        baseline.layer_norm_weight.copy_(values)
    mx = make_mx(mx_class, nv.layer_norm_weight, zero_centered, "torch")
    generator = torch.Generator(device="cpu").manual_seed(91)
    original = torch.randn((7, 3, hidden_size if not strided else hidden_size * 2), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    if strided:
        original = original[..., ::2]
        assert not original.is_contiguous()
    inputs = [original.detach().requires_grad_() for _ in range(3)]
    ids = [id(layer.layer_norm_weight) for layer in (mx, nv, baseline)]
    keys = [tuple(layer.state_dict()) for layer in (mx, nv, baseline)]
    results = [mx.forward_internal(inputs[0])[0], nv(inputs[1])[0], baseline(inputs[2])[0]]
    assert [layer.fp4_comparison_rmsnorm_calls for layer in (mx, nv, baseline)] == [1, 1, 0]
    dy = torch.randn(results[0].shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    gradients = [
        torch.autograd.grad(y, (x, layer.layer_norm_weight), dy)
        for y, x, layer in zip(results, inputs, (mx, nv, baseline))
    ]
    for result in results[1:]:
        bytes_equal(results[0], result)
    for grads in gradients[1:]:
        bytes_equal(gradients[0][0], grads[0])
        bytes_equal(gradients[0][1], grads[1])
    assert gradients[0][1].dtype == gamma_dtype
    assert ids == [id(layer.layer_norm_weight) for layer in (mx, nv, baseline)]
    assert keys == [tuple(layer.state_dict()) for layer in (mx, nv, baseline)]
    assert keys[1] == keys[2] == ("layer_norm_weight", "weight")
    # Existing state remains loadable across the opt-in, with no extra norm keys.
    nv.load_state_dict(baseline.state_dict(), strict=True)
    bytes_equal(nv.layer_norm_weight, baseline.layer_norm_weight)


@pytest.mark.parametrize("gamma_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_production_norm_routes_and_gradients(
    modules, monkeypatch, mx_class, gamma_dtype, zero_centered, strided
):
    check_routes(modules, monkeypatch, mx_class, "cpu", gamma_dtype, zero_centered, strided)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an explicitly allocated GPU")
@pytest.mark.parametrize("gamma_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_device_kernel_production_norm_routes_and_gradients(
    modules, monkeypatch, mx_class, gamma_dtype, zero_centered, strided
):
    check_routes(
        modules, monkeypatch, mx_class, "cuda", gamma_dtype, zero_centered, strided, hidden_size=2880
    )


def test_default_mx_calls_existing_turbo_and_comparison_rejects_skip(monkeypatch, mx_class):
    fake = ModuleType("primus_turbo.pytorch.ops.normalization")
    calls = []
    fake.rmsnorm = lambda x, gamma, epsilon: calls.append((gamma.clone(), epsilon)) or x
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    gamma = torch.arange(16, dtype=torch.float32)
    mx = make_mx(mx_class, gamma, True, "native")
    x = torch.randn(2, 3, 16)
    assert mx.forward_internal(x)[0] is not None
    assert len(calls) == 1 and mx.fp4_comparison_rmsnorm_calls == 0
    bytes_equal(calls[0][0], gamma + 1)
    mx.fp4_comparison_rmsnorm_backend = "torch"
    mx._skip_fused_norm = True
    with pytest.raises(ValueError, match="already normalized"):
        mx.forward_internal(x)


@pytest.mark.parametrize("recipe", ["mxfp4", "nvfp4", "amdfp4"])
def test_comparison_selection_and_rng_identity(modules, recipe):
    options = dict(
        fp4="e2m1",
        fp4_recipe=recipe,
        normalization="RMSNorm",
        fp4_backend="auto" if recipe == "mxfp4" else "reference",
        enable_primus_turbo=True,
        use_turbo_gemm=recipe == "mxfp4",
        fp4_comparison_rmsnorm_backend="torch",
    )
    assert modules.D.comparison_rmsnorm_backend(options) == "torch"
    assert modules.D.comparison_rmsnorm_backend({}, fallback=options) == "torch"
    modules.D.validate_fp4_options(options, require_runtime=False)
    assert modules.D.comparison_rmsnorm_backend({}) == "native"
    if recipe != "mxfp4":
        a = modules.D.build_two_level_config(options)
        b = modules.D.build_two_level_config({**options, "fp4_comparison_rmsnorm_backend": "native"})
        assert a.fingerprint == b.fingerprint and a.rng_fingerprint == b.rng_fingerprint
    for override in (
        {"fp4": None},
        {"normalization": "LayerNorm"},
        {"fp4_comparison_rmsnorm_backend": "typo"},
    ):
        with pytest.raises(ValueError):
            modules.D.validate_fp4_options({**options, **override}, require_runtime=False)
    for key in (
        "PRIMUS_FUSED_RMSNORM_MXFP4_QKV",
        "PRIMUS_FUSED_RESIDUAL_NORM",
        "PRIMUS_FUSED_RESIDUAL_NORM_V2",
    ):
        with pytest.raises(ValueError, match=key):
            modules.D.validate_fp4_options(options, environ={key: "1"}, require_runtime=False)


def test_recipe_option_is_explicit_and_default_stays_native():
    spec = importlib.util.spec_from_file_location("_norm_recipe", ROOT / "tools/fp4_training_recipe.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    source = ROOT
    for recipe in ("mxfp4", "nvfp4", "amdfp4"):
        default, old_manifest = generator.build_recipe(source, recipe=recipe)
        controlled, manifest = generator.build_recipe(
            source, recipe=recipe, comparison_rmsnorm_backend="torch"
        )
        before = default["modules"]["pre_trainer"]["overrides"]
        after = controlled["modules"]["pre_trainer"]["overrides"]
        assert "fp4_comparison_rmsnorm_backend" not in before
        assert after.pop("fp4_comparison_rmsnorm_backend") == "torch"
        assert before == after and "rmsnorm_control" not in old_manifest
        assert "rmsnorm_control" in manifest
