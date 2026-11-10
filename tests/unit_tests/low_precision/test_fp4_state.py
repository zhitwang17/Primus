# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

import copy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fp4_preflight_state", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, D = preflight.load_contracts()
A = importlib.import_module(C.__package__ + ".fp4_autograd")
S = importlib.import_module(C.__package__ + ".fp4_state")


class Layer(torch.nn.Module):
    def __init__(self, recipe="nvfp4"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(32, 16))
        self.fp4_config = C.TwoLevelFP4Config(recipe)
        self.fp4_configs = (self.fp4_config,)
        self.fp4_rngs = {}

    def forward(self, x):
        return A.linear(x, self.weight, self.fp4_config, self.fp4_rng)


def step(model, optimizer, x):
    optimizer.zero_grad()
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    return loss.item()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_checkpoint_exact_optimizer_and_random_continuation(tmp_path, recipe):
    torch.manual_seed(27)
    model = Layer(recipe)
    assert S.bind_fp4_model(model) == 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(33, 16, requires_grad=True)
    step(model, optimizer, x)
    path = S.save_fp4_sidecar(model, optimizer, tmp_path, 1)
    assert path.is_file()
    checkpoint = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    expected_loss = step(model, optimizer, x)
    expected_weight, expected_state = model.weight.detach().clone(), S.fp4_state_dict(model)
    clone = Layer(recipe)
    S.bind_fp4_model(clone)
    clone_optimizer = torch.optim.Adam(clone.parameters(), lr=1e-3)
    clone.load_state_dict(checkpoint["model"])
    clone_optimizer.load_state_dict(checkpoint["optimizer"])
    S.load_fp4_sidecar(clone, clone_optimizer, tmp_path, 1)
    assert step(clone, clone_optimizer, x) == expected_loss
    assert torch.equal(clone.weight, expected_weight)
    assert S.fp4_state_dict(clone) == expected_state


def test_binding_is_stable_private_and_idempotent():
    model = torch.nn.Sequential(Layer(), Layer())
    S.bind_fp4_model(model, rank=3)
    assert model[0].fp4_rng.stream_id != model[1].fp4_rng.stream_id
    model[0].fp4_rng.reserve("sr", 9)
    S.bind_fp4_model(SimpleNamespace(module=model), rank=3)
    assert model[0].fp4_rng.sr_offset == 9
    with pytest.raises(ValueError, match="moved"):
        S.bind_fp4_model(model, rank=4)


@pytest.mark.parametrize("field", ["version", "backend", "rng_implementation", "torch_version", "topology"])
def test_checkpoint_rejects_identity_drift_without_mutating_rng(field):
    model = Layer()
    S.bind_fp4_model(model)
    state = S.fp4_state_dict(model)
    state[field] = "wrong"
    before = S.fp4_state_dict(model)
    with pytest.raises(ValueError, match=field):
        S.load_fp4_state_dict(model, state)
    assert S.fp4_state_dict(model) == before


def test_checkpoint_rejects_recipe_shape_and_missing_history(tmp_path):
    model = Layer()
    S.bind_fp4_model(model)
    with pytest.raises(FileNotFoundError, match="Missing FP4"):
        S.load_fp4_sidecar(model, None, tmp_path, 1)
    S.save_fp4_sidecar(model, None, tmp_path, 1)
    with pytest.raises(ValueError, match="iteration"):
        S.load_fp4_sidecar(model, None, tmp_path, 2)
    other = Layer("amdfp4")
    S.bind_fp4_model(other)
    with pytest.raises(ValueError, match="recipes"):
        S.load_fp4_state_dict(other, S.fp4_state_dict(model))
    state = S.fp4_state_dict(model)
    state["modules"]["model0/<root>"]["parameters"]["weight"]["shape"] = [16, 32]
    with pytest.raises(ValueError, match="parameters"):
        S.load_fp4_state_dict(model, state)


def test_checkpoint_rejects_routing_and_frozen_parameter_drift():
    model = Layer()
    S.bind_fp4_model(model)
    state = S.fp4_state_dict(model)
    model.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="parameters"):
        S.load_fp4_state_dict(model, state)
    model.weight.requires_grad_(True)
    state["modules"]["model0/<root>"]["routing"]["training_config"] = "wrong"
    with pytest.raises(ValueError, match="routing"):
        S.load_fp4_state_dict(model, state)


def test_missing_reference_modules_fail_closed():
    with pytest.raises(ValueError, match="spec provider"):
        S.bind_fp4_model(torch.nn.Linear(2, 2))


def test_bf16_nested_layer_context_restores_parent():
    plan = D.FP4ExecutionPlan(C.TwoLevelFP4Config("amdfp4"), "reference", "dense")
    with D.fp4_recipe_context(plan):
        with D.fp4_recipe_context(None):
            assert D.current_fp4_plan() is None
        assert D.current_fp4_plan() is plan
