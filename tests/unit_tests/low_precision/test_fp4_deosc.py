# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU DeOsc numerics, restore and Megatron copy-boundary contract tests.

Optimizer mocks exercise the real installer/wrappers without importing the
GPU training stack. The reduced-model smoke covers an installed Megatron
optimizer separately.
"""

import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_primus_fp4_deosc_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
sys.modules[PACKAGE] = package
C = importlib.import_module(PACKAGE + ".fp4_contract")
R = importlib.import_module(PACKAGE + ".fp4_reference")
D = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc")


def binding(recipe="nvfp4", shape=(16, 16)):
    cfg = C.TwoLevelFP4Config(recipe)
    model = torch.nn.Parameter(torch.full(shape, 6.0, dtype=torch.bfloat16))
    master = model.detach().float().clone()
    master.view(-1)[0] = 0.74609375
    return model, master, cfg


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("shape", [(16, 16), (3, 16, 16)])
def test_bin_flip_snap_preserves_master_and_recomputes_full_qdq(recipe, shape, monkeypatch):
    model, master, cfg = binding(recipe, shape)
    model_before = model.detach().clone()
    runner = D.TwoLevelDeOsc({"weight": (model, master, cfg)}, period=1, ratio=4)
    calls = []
    original = D.qdq

    def recording_qdq(x, *args, **kwargs):
        calls.append(x.shape)
        return original(x, *args, **kwargs)

    monkeypatch.setattr(D, "qdq", recording_qdq)
    global_rng = torch.random.get_rng_state().clone()
    runner.step()  # Seed after the first optimizer update.
    assert runner.stats["periods"] == 0
    master.view(-1)[0] = 0.75390625
    runner.step()
    assert runner.stats["periods"] == 1 and runner.stats["snapped"] == 1
    assert master.view(-1)[0].item() == 1.0
    assert torch.equal(model.detach(), model_before)  # Installer owns the later copy.
    assert master.dtype == torch.float32
    assert calls == [model.shape, model.shape, model.shape]
    assert torch.equal(global_rng, torch.random.get_rng_state())
    saved = runner.state_dict()["states"]["weight"]
    expected = R.qdq(master.bfloat16(), cfg, cfg.operand("fwd", "weight"))
    assert torch.equal(saved["prev_q"], expected)
    assert torch.equal(saved["prev"], master.bfloat16())
    assert saved["dist_w"].count_nonzero() == 0 and saved["dist_q"].count_nonzero() == 0


def test_snapshot_denominator_matches_existing_bf16_policy():
    model, master, cfg = binding()
    runner = D.TwoLevelDeOsc({"weight": (model, master, cfg)}, period=3)
    runner.step()
    master.view(-1)[0] += 1e-5  # Below a BF16 ulp: existing policy sees no movement.
    runner.step()
    state = runner.state_dict()
    assert "bf16-snapshots-bf16-deltas" in state["snapshot_policy"]
    assert state["states"]["weight"]["dist_w"].count_nonzero() == 0


def test_start_step_bf16_override_and_actual_recipe_guard():
    model, master, cfg = binding()
    owner = SimpleNamespace(_primus_fp4_last_config=cfg)
    runner = D.TwoLevelDeOsc(
        {"weight": (model, master, cfg)}, period=2, start_step=3, active_modules={"weight": owner}
    )
    runner.step()
    runner.step()
    assert not runner.state_dict()["states"]
    runner.step()
    assert runner.stats["observations"] == 1
    owner._primus_fp4_last_config = None
    runner.step()
    assert not runner.state_dict()["states"]
    owner._primus_fp4_last_config = cfg
    runner.step()
    assert runner.state_dict()["states"]["weight"]["window_steps"] == 0
    owner._primus_fp4_last_config = C.TwoLevelFP4Config("amdfp4")
    previous_steps = runner.step_count
    with pytest.raises(ValueError, match="last forward recipe"):
        runner.step()
    assert runner.step_count == previous_steps


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_state_equal(a[key], b[key])
    else:
        assert a == b


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_checkpoint_resume_matches_uninterrupted_period(recipe):
    model, master, cfg = binding(recipe)
    runner = D.TwoLevelDeOsc({"layer.weight": (model, master, cfg)}, period=2, ratio=4)
    runner.step()
    master.view(-1)[0] = 0.75390625
    runner.step()
    saved, saved_master = runner.state_dict(), master.clone()
    master.view(-1)[0] = 0.74609375
    runner.step()
    expected_state, expected_master = runner.state_dict(), master.clone()
    resumed = D.TwoLevelDeOsc({"layer.weight": (model, saved_master, cfg)}, period=2, ratio=4)
    resumed.validate_state_dict(saved)
    assert resumed.step_count == 0  # Validation must not mutate state.
    resumed.load_state_dict(saved)
    saved_master.view(-1)[0] = 0.74609375
    resumed.step()
    assert torch.equal(saved_master, expected_master)
    assert_state_equal(resumed.state_dict(), expected_state)
    saved["states"]["layer.weight"]["prev"].zero_()
    assert resumed.state_dict()["states"]["layer.weight"]["prev"].count_nonzero() > 0


@pytest.mark.parametrize(
    "corruption", ["recipe", "parameter", "settings", "dtype", "shape", "window", "negative"]
)
def test_checkpoint_rejection_is_strict_and_atomic(corruption):
    model, master, cfg = binding()
    runner = D.TwoLevelDeOsc({"weight": (model, master, cfg)}, period=2)
    runner.step()
    saved = runner.state_dict()
    before = copy.deepcopy(saved)
    if corruption == "recipe":
        saved["bindings"]["weight"]["recipe_fingerprint"] = "wrong"
    elif corruption == "parameter":
        del saved["bindings"]["weight"]
    elif corruption == "settings":
        saved["settings"]["period"] += 1
    elif corruption == "dtype":
        saved["states"]["weight"]["prev"] = saved["states"]["weight"]["prev"].float()
    elif corruption == "shape":
        saved["states"]["weight"]["prev"] = torch.zeros(1, dtype=torch.bfloat16)
    elif corruption == "window":
        saved["states"]["weight"]["window_steps"] = 2
    else:
        saved["states"]["weight"]["dist_q"].fill_(-1)
    with pytest.raises(ValueError):
        runner.load_state_dict(saved)
    assert_state_equal(runner.state_dict(), before)


class ReferenceModule(torch.nn.Module):
    def __init__(self, recipe="nvfp4", stream="rank0/model0/layer0"):
        super().__init__()
        model, _, cfg = binding(recipe)
        self.weight = model
        self.bias = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
        self.fp4_config, self.fp4_stream_id = cfg, stream
        # No last-forward tag until an actual forward activates the recipe.


class MockFloat16Optimizer:
    def __init__(self, model, *, distributed=False):
        self.model = model
        self.master = model.weight.detach().float().clone()
        self.master.view(-1)[0] = 0.74609375
        self.config = SimpleNamespace()
        self.is_stub_optimizer = False
        self.successful = True
        self.copy_count = 0
        if distributed:
            self.master = self.master.flatten()
            self.model_float16_groups = [[model.weight]]
            self.shard_fp32_from_float16_groups = [[self.master]]
        else:
            self.float16_groups = [[model.weight]]
            self.fp32_from_float16_groups = [[self.master]]

    def _copy_main_params_to_model_params(self):
        self.copy_count += 1
        with torch.no_grad():
            self.model.weight.copy_(self.master.view_as(self.model.weight))

    def step_with_ready_grads(self):
        if not self.successful:
            return False
        self._copy_main_params_to_model_params()
        return True


def args(**updates):
    return SimpleNamespace(weight_deosc=True, weight_deosc_period=1, weight_deosc_ratio=4, **updates)


@pytest.mark.parametrize("distributed", [False, True])
def test_installer_runs_after_update_before_copy_and_skips_reload_and_failed_step(distributed):
    model = ReferenceModule()
    optimizer = MockFloat16Optimizer(model, distributed=distributed)
    runners = D.install_megatron_deosc(optimizer, [model], args())
    runner = runners[0]
    assert optimizer._primus_two_level_deosc is runner
    assert optimizer._primus_two_level_deosc_runners is runners
    optimizer._copy_main_params_to_model_params()  # Constructor/reload copy.
    assert runner.step_count == 0
    model._primus_fp4_last_config = model.fp4_config
    optimizer.step_with_ready_grads()
    optimizer.successful = False
    assert optimizer.step_with_ready_grads() is False
    assert runner.step_count == 1
    optimizer.master.view(-1)[0] = 0.75390625  # Already-updated FP32 optimizer master.
    optimizer.successful = True
    assert optimizer.step_with_ready_grads() is True
    assert runner.step_count == 2 and runner.stats["snapped"] == 1
    assert optimizer.master.view(-1)[0].item() == model.weight.view(-1)[0].item() == 1
    assert optimizer.copy_count == 3
    with pytest.raises(ValueError, match="already installed"):
        D.install_megatron_deosc(optimizer, [model], args())


def test_installer_chained_optimizers_and_pre_forward_restore():
    first, second = ReferenceModule(), ReferenceModule("amdfp4", "rank0/model0/layer1")
    a, b = MockFloat16Optimizer(first), MockFloat16Optimizer(second)
    chain = SimpleNamespace(chained_optimizers=[a, b])
    runners = D.install_megatron_deosc(chain, [first, second], args())
    assert len(runners) == 2
    a.step_with_ready_grads()  # No forward marker yet: no FP4 window.
    assert not runners[0].state_dict()["states"]
    saved = runners[1].state_dict()
    runners[1].load_state_dict(saved)
    second._primus_fp4_last_config = second.fp4_config
    b.step_with_ready_grads()
    assert len(runners[1].state_dict()["states"]) == 1


def test_grouped_2d_bias_is_not_a_quantized_weight_binding():
    model = ReferenceModule()
    del model.weight
    model.weights = torch.nn.Parameter(torch.full((3, 16, 16), 6.0, dtype=torch.bfloat16))
    model.bias = torch.nn.Parameter(torch.zeros((3, 16), dtype=torch.bfloat16))
    master = model.weights.detach().float().clone()
    optimizer = SimpleNamespace(
        float16_groups=[[model.weights, model.bias]],
        fp32_from_float16_groups=[[master, model.bias.detach().float().clone()]],
        step_with_ready_grads=lambda: True,
        _copy_main_params_to_model_params=lambda: None,
    )
    runners = D.install_megatron_deosc(optimizer, [model], args())
    assert list(runners[0].bindings) == [model.fp4_stream_id + "/weights"]
    assert runners[0].stats["tracked_parameters"] == 1


@pytest.mark.parametrize(
    "change",
    ["partial", "master_bf16", "missing", "dp2", "precision_aware", "reuse", "unbound", "fusion"],
)
def test_installer_rejects_unsupported_contracts_before_mutating_optimizer(change):
    model = ReferenceModule()
    optimizer = MockFloat16Optimizer(model)
    options = args()
    if change == "partial":
        optimizer.fp32_from_float16_groups = [[optimizer.master.flatten()[:10]]]
    elif change == "master_bf16":
        optimizer.fp32_from_float16_groups = [[optimizer.master.bfloat16()]]
    elif change == "missing":
        optimizer.float16_groups, optimizer.fp32_from_float16_groups = [[]], [[]]
    elif change == "dp2":
        options.data_parallel_size = 2
    elif change == "precision_aware":
        optimizer.config.use_precision_aware_optimizer = True
    elif change == "reuse":
        optimizer.config.reuse_grad_buf_for_mxfp8_param_ag = True
    elif change == "unbound":
        del model.fp4_stream_id
    else:
        options.weight_deosc_fusion = True
    original_step = optimizer.step_with_ready_grads
    with pytest.raises(ValueError):
        D.install_megatron_deosc(optimizer, [model], options)
    assert optimizer.step_with_ready_grads == original_step
    assert not hasattr(optimizer, "_primus_two_level_deosc")
