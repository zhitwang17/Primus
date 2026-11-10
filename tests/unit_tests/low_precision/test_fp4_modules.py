# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU tests of the actual reference module implementations.

External Megatron initialization and provider imports are isolated. Quantizers,
GEMM autograd, module forwards, recipe selection, and Torch parameter hooks are
real. These tests do not claim a real Megatron DDP or GPU execution.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "primus/backends/megatron/core"


@pytest.fixture
def modules(monkeypatch):
    def package(name, path=None, **members):
        module = ModuleType(name)
        module.__path__ = [] if path is None else [str(path)]
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    for name, path in (
        ("primus", ROOT / "primus"),
        ("primus.backends", ROOT / "primus/backends"),
        ("primus.backends.megatron", CORE.parent),
        ("primus.backends.megatron.core", CORE),
        ("primus.backends.megatron.core.extensions", CORE / "extensions"),
    ):
        package(name, path)

    class MegatronModule(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    package("megatron")
    package("megatron.core")
    package("megatron.core.parallel_state", get_tensor_model_parallel_group=lambda **kwargs: None)
    package("megatron.core.quantization")
    package("megatron.core.quantization.utils", get_quant_config_or_none=lambda *args: None)
    package("megatron.core.tensor_parallel")
    package(
        "megatron.core.tensor_parallel.random",
        get_cuda_rng_tracker=lambda: None,
        get_expert_parallel_rng_tracker_name=lambda: "expert",
    )
    package("megatron.core.transformer")
    package("megatron.core.transformer.module", MegatronModule=MegatronModule)
    package("megatron.core.extensions")
    package(
        "megatron.core.extensions.transformer_engine",
        TEQuantizationParams=SimpleNamespace(parse_from_config=lambda value: value),
    )
    package(
        "primus.backends.megatron.core.extensions.transformer_engine_spec_provider",
        PrimusTurboSpecProvider=type("Provider", (), {}),
        TEGroupedMLPSubmodules=SimpleNamespace,
        _experts_builder=lambda module, submodules: lambda *a, **kw: module(*a, submodules=submodules, **kw),
        _MOE_TAKES_EXPERTS_BUILDER=True,
    )
    prefix = "primus.backends.megatron.core"
    contract = load(prefix + ".fp4_contract", CORE / "fp4_contract.py")
    load(prefix + ".fp4_reference", CORE / "fp4_reference.py")
    load(prefix + ".fp4_backend", CORE / "fp4_backend.py")
    dispatch = load(prefix + ".fp4_dispatch", CORE / "fp4_dispatch.py")
    autograd = load(prefix + ".fp4_autograd", CORE / "fp4_autograd.py")
    impl = load(prefix + ".extensions.two_level_fp4", CORE / "extensions/two_level_fp4.py")
    return SimpleNamespace(C=contract, D=dispatch, A=autograd, M=impl)


def config(**updates):
    values = dict(
        fp4="e2m1",
        fp4_recipe="nvfp4",
        fp4_backend="reference",
        fp8=None,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=True,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        hidden_size=16,
        ffn_hidden_size=16,
        moe_ffn_hidden_size=16,
        gated_linear_unit=True,
        add_bias_linear=True,
        activation_func=F.silu,
        activation_func_clamp_value=1.0,
        glu_linear_offset=1.0,
        moe_router_topk=2,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        init_method=lambda p: torch.nn.init.normal_(p, std=0.2),
        output_layer_init_method=lambda p: torch.nn.init.normal_(p, std=0.2),
    )
    values.update(updates)
    return SimpleNamespace(**values)


def bind(test, module):
    for name, child in module.named_modules():
        for cfg in getattr(child, "fp4_configs", ()):
            child.fp4_rngs[cfg.fingerprint] = test.C.FP4RandomState.for_config(
                cfg, f"test/{name}/{cfg.recipe}"
            )
        if hasattr(child, "fp4_config"):
            child.fp4_rng = child.fp4_rngs[child.fp4_config.fingerprint]


def plan(test, cfg, kind="dense"):
    return test.D.FP4ExecutionPlan(test.D.build_two_level_config(cfg), "reference", kind)


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_composite_norm_retains_output_bias_and_parameter_gradients(modules, normalization, zero_centered):
    cfg = config(normalization=normalization, layernorm_zero_centered_gamma=zero_centered)
    layer = modules.M.TwoLevelFP4LayerNormColumnParallelLinear(
        16, 32, config=cfg, init_method=cfg.init_method, bias=True, skip_bias_add=True
    )
    with torch.no_grad():
        layer.bias.fill_(0.25)
        layer.layer_norm_weight.uniform_(0.3, 0.9)
    x = torch.randn(3, 2, 16, requires_grad=True)
    result, bias = layer(x)
    gamma = layer.layer_norm_weight + int(zero_centered)
    if normalization == "RMSNorm":
        expected_norm = F.rms_norm(x, (16,), gamma, cfg.layernorm_epsilon)
    else:
        expected_norm = F.layer_norm(x, (16,), gamma, layer.layer_norm_bias, cfg.layernorm_epsilon)
    expected = F.linear(expected_norm, layer.weight)
    torch.testing.assert_close(result, expected)
    assert bias is layer.bias
    (result + bias).square().mean().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_real_dense_and_grouped_execution_preserves_parameters_and_gradient_hooks(modules, recipe):
    cfg = config(fp4_recipe=recipe)
    dense = modules.M.TwoLevelFP4ColumnParallelLinear(16, 16, config=cfg, init_method=cfg.init_method)
    grouped = modules.M.TwoLevelFP4ColumnParallelGroupedLinear(
        3, 16, 16, config=cfg, init_method=cfg.init_method
    )
    model = torch.nn.ModuleList([dense, grouped])
    bind(modules, model)
    weights = [dense.weight, grouped.weights]
    identities = [id(weight) for weight in weights]
    calls = [0, 0]
    for index, weight in enumerate(weights):
        weight.main_grad = torch.zeros_like(weight)

        def hook(gradient, index=index, weight=weight):
            weight.main_grad.add_(gradient)
            calls[index] += 1
            return gradient

        weight.register_hook(hook)
    expected_accum = [torch.zeros_like(weight) for weight in weights]
    for _ in range(2):
        x = torch.randn(17, 16, requires_grad=True)
        with modules.D.fp4_recipe_context(plan(modules, cfg)):
            intermediate, _ = dense(x)
            output, _ = grouped(intermediate, [1, 0, 16])
        output.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for i, weight in enumerate(weights):
            expected_accum[i].add_(weight.grad)
            weight.grad = None
    assert calls == [2, 2] and identities == [id(weight) for weight in weights]
    for expected, weight in zip(expected_accum, weights):
        assert torch.equal(expected, weight.main_grad)
        assert weight._primus_two_level_fp4_config.recipe == recipe
    counts = modules.A.execution_counts()
    for kind in ("dense", "grouped"):
        for direction in ("fwd", "dgrad", "wgrad"):
            assert counts[f"{recipe}.{kind}.{direction}"] == 2
    assert torch.count_nonzero(grouped.weights.main_grad[1]) == 0


@pytest.mark.parametrize("interleave", [None, 1, 4])
@pytest.mark.parametrize("probs_on_input", [False, True])
def test_grouped_mlp_glu_clamp_offset_bias_and_probs_gradient(modules, interleave, probs_on_input):
    cfg = config(
        moe_mlp_glu_interleave_size=interleave,
        moe_apply_probs_on_input=probs_on_input,
        moe_router_topk=1 if probs_on_input else 2,
    )
    model = modules.M.TwoLevelFP4GroupedMLP(3, cfg)
    with torch.no_grad():
        model.linear_fc1.bias.uniform_(-0.2, 0.2)
        model.linear_fc2.bias.uniform_(-0.2, 0.2)
    x = torch.randn(7, 16, requires_grad=True)
    probs = torch.rand(7, requires_grad=True)
    lengths = [1, 0, 6]
    actual, bias = model(x, lengths, probs)
    expected_parts = []
    offset = 0
    for expert, count in enumerate(lengths):
        p = probs[offset : offset + count, None]
        input_ = x[offset : offset + count] * p if probs_on_input else x[offset : offset + count]
        h = F.linear(input_, model.linear_fc1.weights[expert], model.linear_fc1.bias[expert])
        if interleave is None:
            gate, up = h[:, :16], h[:, 16:]
        else:
            chunks = h.split(interleave, dim=-1)
            gate, up = torch.cat(chunks[::2], -1), torch.cat(chunks[1::2], -1)
        act = F.silu(torch.minimum(gate, torch.ones_like(gate))) * (up.clip(-1, 1) + 1)
        out = F.linear(act, model.linear_fc2.weights[expert], model.linear_fc2.bias[expert])
        expected_parts.append(out if probs_on_input else out * p)
        offset += count
    expected = torch.cat(expected_parts)
    torch.testing.assert_close(actual, expected)
    assert bias is None
    targets = (x, probs, *model.parameters())
    grad_actual = torch.autograd.grad(actual.square().sum(), targets, retain_graph=True)
    grad_expected = torch.autograd.grad(expected.square().sum(), targets)
    for got, want in zip(grad_actual, grad_expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(grad_actual[1]) > 0


def override(recipe=None, *, quantized=True, nonquantized=False):
    return SimpleNamespace(
        fp4_quantization_recipe=recipe,
        fp8_quantization_recipe=None,
        override_quantized_autocast=quantized,
        override_nonquantized_autocast=nonquantized,
        fp4_param=False,
        fp8_param=False,
    )


def test_intentional_bf16_override_and_override_flags(modules):
    cfg = config()
    layer = modules.M.TwoLevelFP4Linear(16, 16, config=cfg, init_method=cfg.init_method, bias=False)
    layer.finish_init(SimpleNamespace(training_recipe=override(), evaluation_recipe=None))
    bind(modules, layer)
    x = torch.randn(2, 16)
    with modules.D.fp4_recipe_context(plan(modules, cfg)):
        actual, _ = layer(x)
    assert torch.equal(actual, F.linear(x, layer.weight))
    assert modules.A.execution_counts() == {}
    assert not layer.will_execute_quantized(True)
    layer.te_quant_params.training_recipe.override_quantized_autocast = False
    with modules.D.fp4_recipe_context(plan(modules, cfg)):
        layer(x)
    assert modules.A.execution_counts()["nvfp4.dense.fwd"] == 1
    assert layer.will_execute_quantized(True)


def test_per_module_cross_recipe_uses_matching_private_rng(modules):
    cfg = config()
    layer = modules.M.TwoLevelFP4Linear(16, 16, config=cfg, init_method=cfg.init_method, bias=False)
    layer.finish_init(
        SimpleNamespace(
            training_recipe=override("amdfp4", nonquantized=True),
            evaluation_recipe=override("nvfp4", nonquantized=True),
        )
    )
    bind(modules, layer)
    assert {c.recipe for c in layer.fp4_configs} == {"nvfp4", "amdfp4"}
    assert layer.fp4_config.recipe == "amdfp4"
    layer(torch.randn(3, 16, requires_grad=True))[0].sum().backward()
    assert modules.A.execution_counts()["amdfp4.dense.wgrad"] == 1
    layer.eval()
    layer(torch.randn(3, 16, requires_grad=True))[0].sum().backward()
    assert modules.A.execution_counts()["nvfp4.dense.wgrad"] == 1
    for state in layer.fp4_rngs.values():
        assert state.sr_offset > 0 and state.rht_offset > 0


def test_routing_identity_distinguishes_precision_roles_and_override_flags(modules):
    cfg = config()
    layer = modules.M.TwoLevelFP4Linear(16, 16, config=cfg, init_method=cfg.init_method)
    layer.finish_init(
        SimpleNamespace(training_recipe=override("nvfp4"), evaluation_recipe=override("amdfp4"))
    )
    first_identity = layer.fp4_routing_identity()
    first_set = {c.fingerprint for c in layer.fp4_configs}
    layer.finish_init(
        SimpleNamespace(training_recipe=override("amdfp4"), evaluation_recipe=override("nvfp4"))
    )
    assert {c.fingerprint for c in layer.fp4_configs} == first_set
    assert layer.fp4_routing_identity() != first_identity
    assert layer.fp4_config.recipe == "amdfp4"

    layer.finish_init(
        SimpleNamespace(
            training_recipe=override("amdfp4", quantized=False), evaluation_recipe=override("nvfp4")
        )
    )
    ignored_identity = layer.fp4_routing_identity()
    assert layer.fp4_config.recipe == "nvfp4"
    assert ignored_identity != first_identity
    bind(modules, layer)
    with modules.D.fp4_recipe_context(plan(modules, cfg)):
        layer(torch.randn(3, 16))
    assert layer._primus_fp4_last_config.recipe == "nvfp4"

    layer.finish_init(SimpleNamespace(training_recipe=override(), evaluation_recipe=None))
    bf16_identity = layer.fp4_routing_identity()
    layer.te_quant_params.training_recipe.override_quantized_autocast = False
    assert layer.fp4_routing_identity() != bf16_identity


def test_unbound_rng_and_unsupported_topology_fail(modules):
    cfg = config()
    layer = modules.M.TwoLevelFP4Linear(16, 16, config=cfg, init_method=cfg.init_method)
    with modules.D.fp4_recipe_context(plan(modules, cfg)):
        with pytest.raises(ValueError, match="private FP4RandomState"):
            layer(torch.randn(2, 16))
    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        modules.M.TwoLevelFP4Linear(
            16, 16, config=config(tensor_model_parallel_size=2), init_method=cfg.init_method
        )
    with pytest.raises(ValueError, match="moe_grouped_gemm"):
        modules.M.TwoLevelFP4SpecProvider().grouped_mlp_modules(False)


def test_moe_activation_recompute_retains_probs_gradients(modules):
    cfg = config(recompute_granularity="selective", recompute_modules=["moe_act"])
    model = modules.M.TwoLevelFP4GroupedMLP(2, cfg)
    x = torch.randn(17, 16, requires_grad=True)
    probs = torch.rand(17, requires_grad=True)
    output, _ = model(x, [1, 16], probs)
    output.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(probs.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
