# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Real smoke argument/config wiring and real reference norm execution on CPU."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_fp4_modules import modules  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("_norm_smoke", ROOT / "tools/fp4_smoke.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def make_config(monkeypatch, recipe, backend):
    for name, members in (
        ("megatron.core.transformer.enums", {"AttnBackend": SimpleNamespace(unfused="unfused")}),
        ("megatron.core.transformer.transformer_config", {"TransformerConfig": SimpleNamespace}),
        ("primus.backends.megatron.core.enums", {"Fp4Recipe": lambda value: value}),
    ):
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    args = SMOKE.parse_args(
        ["--recipe", recipe, "--output", "/tmp/unused.json", "--comparison-rmsnorm-backend", backend]
    )
    return SMOKE._make_config(args)


def make_model(modules, monkeypatch, config):
    monkeypatch.setattr(modules.M.TwoLevelFP4Linear, "forward", lambda self, x, **kwargs: (x, None))

    def norm():
        return modules.M.TwoLevelFP4LayerNormColumnParallelLinear(
            config.hidden_size,
            16,
            config=config,
            init_method=lambda parameter: torch.nn.init.ones_(parameter),
            bias=False,
        )

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList()
    for index in range(config.num_layers):
        layer = torch.nn.Module()
        layer.self_attention = torch.nn.Module()
        layer.self_attention.linear_qkv = norm()
        if index == 0:
            layer.mlp = torch.nn.Module()
            layer.mlp.linear_fc1 = norm()
        model.decoder.layers.append(layer)
    return model


def execute_norms(model, x):
    return [module(x)[0] for module in model.modules() if hasattr(module, "fp4_comparison_rmsnorm_backend")]


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("backend", ["native", "torch"])
def test_cli_config_actual_norm_and_fresh_replay_route(modules, monkeypatch, recipe, backend):
    config = make_config(monkeypatch, recipe, backend)
    assert config.fp4_comparison_rmsnorm_backend == backend
    original = make_model(modules, monkeypatch, config)
    state = copy.deepcopy(original.state_dict())
    start = SMOKE._rmsnorm_snapshot(original, backend, config.num_layers, pristine=True)
    assert start["expected_qkv_slots"] == config.num_layers
    assert len(start["slots"]) == config.num_layers + 1  # all combined slots, including dense FC1
    assert all(slot["calls"] == 0 for slot in start["slots"].values())
    x = torch.randn(2, 3, config.hidden_size).bfloat16()
    outputs = execute_norms(original, x)
    execute_norms(original, x)  # diagnostic call counts need not match replay length
    trained = SMOKE._rmsnorm_snapshot(original, backend, config.num_layers, executed=True)
    assert all(slot["calls"] == (2 if backend == "torch" else 0) for slot in trained["slots"].values())
    restored = make_model(modules, monkeypatch, config)
    restored.load_state_dict(state, strict=True)
    restored_start = SMOKE._rmsnorm_snapshot(restored, backend, config.num_layers, pristine=True)
    assert restored_start == start
    for expected, actual in zip(outputs, execute_norms(restored, x)):
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    replayed = SMOKE._rmsnorm_snapshot(restored, backend, config.num_layers, executed=True)
    SMOKE._assert_rmsnorm_replay(trained, replayed)
    assert all(slot["calls"] == (1 if backend == "torch" else 0) for slot in replayed["slots"].values())
    assert set(state) == set(restored.state_dict())


def test_torch_claim_cannot_pass_without_real_execution(modules, monkeypatch):
    config = make_config(monkeypatch, "nvfp4", "torch")
    model = make_model(modules, monkeypatch, config)
    with pytest.raises(AssertionError, match="did not execute"):
        SMOKE._rmsnorm_snapshot(model, "torch", config.num_layers, executed=True)
    qkv = model.decoder.layers[0].self_attention.linear_qkv
    qkv.fp4_comparison_rmsnorm_backend = "native"
    with pytest.raises(AssertionError, match="Invalid actual"):
        SMOKE._rmsnorm_snapshot(model, "torch", config.num_layers)
    qkv.fp4_comparison_rmsnorm_backend = "torch"
    qkv.fp4_comparison_rmsnorm_calls = -1
    with pytest.raises(AssertionError, match="Invalid actual"):
        SMOKE._rmsnorm_snapshot(model, "torch", config.num_layers)
    qkv.fp4_comparison_rmsnorm_calls = 0
    del model.decoder.layers[0].self_attention.linear_qkv
    with pytest.raises(AssertionError, match="slots differ"):
        SMOKE._rmsnorm_snapshot(model, "torch", config.num_layers)


def test_replay_rejects_changed_route_and_counter_inherited_as_state(modules, monkeypatch):
    config = make_config(monkeypatch, "nvfp4", "torch")
    model = make_model(modules, monkeypatch, config)
    execute_norms(model, torch.randn(2, 3, config.hidden_size).bfloat16())
    snapshot = SMOKE._rmsnorm_snapshot(model, "torch", config.num_layers, executed=True)
    with pytest.raises(AssertionError, match="not zero"):
        SMOKE._rmsnorm_snapshot(model, "torch", config.num_layers, pristine=True)
    mismatched = copy.deepcopy(snapshot)
    next(iter(mismatched["slots"].values()))["backend"] = "native"
    with pytest.raises(AssertionError, match="rmsnorm_replay_routes"):
        SMOKE._assert_rmsnorm_replay(snapshot, mismatched)


def test_default_native_and_mxfp4_stays_unsupported():
    assert (
        SMOKE.parse_args(["--recipe", "nvfp4", "--output", "unused"]).comparison_rmsnorm_backend == "native"
    )
    with pytest.raises(SystemExit):
        SMOKE.parse_args(["--recipe", "mxfp4", "--output", "unused"])
    with pytest.raises(SystemExit):
        SMOKE.parse_args(["--recipe", "nvfp4", "--output", "unused", "--comparison-rmsnorm-backend", "typo"])
