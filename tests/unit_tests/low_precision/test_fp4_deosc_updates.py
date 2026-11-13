# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Exercise DeOsc's update boundary with real CPU autograd and optimizers.

Only Megatron's model/master adapter is reduced here. SGD/AdamW, accumulated
gradients, master updates, DeOsc, and checkpoint serialization are executable.
Run with --confcutdir=tests/unit_tests/low_precision to avoid GPU fixtures.
"""

import copy
import importlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_primus_fp4_deosc_updates_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
sys.modules[PACKAGE] = package
C = importlib.import_module(PACKAGE + ".fp4_contract")
D = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc")


class Weight(torch.nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((17, 19), 6.0, dtype=torch.bfloat16))
        with torch.no_grad():
            self.weight[0, 0] = 0.75390625
        self.fp4_config = C.TwoLevelFP4Config(recipe)
        self.fp4_stream_id = "rank0/model0/weight"
        self._primus_fp4_last_config = self.fp4_config


class CPUOptimizerAdapter:
    """The ordinary Megatron FP32-master/copy contract, backed by real SGD/AdamW."""

    def __init__(self, model, kind="sgd"):
        self.model = model
        self.master = torch.nn.Parameter(model.weight.detach().float().clone())
        self.float16_groups = [[model.weight]]
        self.fp32_from_float16_groups = [[self.master]]
        self.config = SimpleNamespace()
        self.successful = True
        self.copy_count = 0
        if kind == "sgd":
            self.optimizer = torch.optim.SGD([self.master], lr=1 / 128)
        else:
            self.optimizer = torch.optim.AdamW([self.master], lr=1 / 128, betas=(0.5, 0.9), weight_decay=0.01)

    @torch.no_grad()
    def _copy_main_params_to_model_params(self):
        self.copy_count += 1
        self.model.weight.copy_(self.master)

    def step_with_ready_grads(self):
        if not self.successful:
            return False
        self.master.grad = self.model.weight.grad.float().clone()
        self.optimizer.step()
        self._copy_main_params_to_model_params()
        return True

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)


def install(model, optimizer, period):
    args = SimpleNamespace(weight_deosc=True, weight_deosc_period=period, weight_deosc_ratio=4)
    return D.install_megatron_deosc(optimizer, [model], args)[0]


def accumulate(model, coefficient, microbatches=4):
    for _ in range(microbatches):
        (model.weight[0, 0].float() * (coefficient / microbatches)).backward()


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_tree_equal(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_accumulation_and_failed_update_do_not_advance_or_snap(recipe):
    model = Weight(recipe)
    optimizer = CPUOptimizerAdapter(model)
    runner = install(model, optimizer, period=1)
    original_master = optimizer.master.detach().clone()

    accumulate(model, 1)
    assert model.weight.grad[0, 0].item() == 1
    assert runner.step_count == 0 and runner.state_dict()["states"] == {}
    assert torch.equal(optimizer.master, original_master)
    assert optimizer.step_with_ready_grads()
    assert runner.step_count == 1 and runner.stats["periods"] == 0
    assert optimizer.master[0, 0].item() == 0.74609375
    optimizer.zero_grad()

    accumulate(model, -1)
    before_state = runner.state_dict()
    before_master = optimizer.master.detach().clone()
    optimizer.successful = False
    assert optimizer.step_with_ready_grads() is False
    assert optimizer.copy_count == 1
    assert_tree_equal(runner.state_dict(), before_state)
    assert torch.equal(optimizer.master, before_master)

    optimizer.successful = True
    assert optimizer.step_with_ready_grads()
    assert runner.step_count == 2 and runner.stats["snapped"] == 1
    assert optimizer.master[0, 0].item() == model.weight[0, 0].item() == 1.0
    assert optimizer.copy_count == 2


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_adamw_and_partial_deosc_window_replay_from_serialized_checkpoint(recipe):
    model = Weight(recipe)
    optimizer = CPUOptimizerAdapter(model, kind="adamw")
    runner = install(model, optimizer, period=3)
    coefficients = [1, -1, 1, -1, -1, 1, -1]

    def advance(target_model, target_optimizer, gradient):
        target_optimizer.zero_grad()
        accumulate(target_model, gradient)
        assert target_optimizer.step_with_ready_grads()

    for coefficient in coefficients[:2]:
        advance(model, optimizer, coefficient)
    assert next(iter(runner.state_dict()["states"].values()))["window_steps"] == 1

    checkpoint = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "master": optimizer.master.detach().clone(),
            "optimizer": optimizer.optimizer.state_dict(),
            "deosc": runner.state_dict(),
        },
        checkpoint,
    )
    for coefficient in coefficients[2:]:
        advance(model, optimizer, coefficient)
    expected_model = copy.deepcopy(model.state_dict())
    expected_master = optimizer.master.detach().clone()
    expected_optimizer = copy.deepcopy(optimizer.optimizer.state_dict())
    expected_deosc = runner.state_dict()
    assert runner.step_count == len(coefficients) and runner.stats["periods"] == 2

    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored_model = Weight(recipe)
    restored_model.load_state_dict(saved["model"])
    restored_optimizer = CPUOptimizerAdapter(restored_model, kind="adamw")
    with torch.no_grad():
        restored_optimizer.master.copy_(saved["master"])
    restored_optimizer.optimizer.load_state_dict(saved["optimizer"])
    restored_runner = install(restored_model, restored_optimizer, period=3)
    restored_runner.load_state_dict(saved["deosc"])
    for coefficient in coefficients[2:]:
        advance(restored_model, restored_optimizer, coefficient)

    assert_tree_equal(restored_model.state_dict(), expected_model)
    assert torch.equal(restored_optimizer.master, expected_master)
    assert_tree_equal(restored_optimizer.optimizer.state_dict(), expected_optimizer)
    assert_tree_equal(restored_runner.state_dict(), expected_deosc)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize(
    "accumulator_dtype,expected_weight_distance,expected_quantized_distance",
    [("bf16", 2.0, 128.0), ("fp32", 300 / 128, 150.0)],
)
def test_accumulator_precision_has_documented_long_window_rounding(
    recipe, accumulator_dtype, expected_weight_distance, expected_quantized_distance
):
    # These exactly representable weights alternate around the 0.5/1.0 FP4
    # decision boundary. Each transition moves raw BF16 by 1/128 and QDQ by
    # 1/2. After 256 additions, a BF16 accumulator rounds another half-ulp
    # back to its even code; FP32 continues adding. This distinguishes the
    # ALTO-compatible accumulation policy from FP32 instead of just checking
    # that a dtype option was accepted.
    cfg = C.TwoLevelFP4Config(recipe)
    model = torch.full((1, 16), 6.0, dtype=torch.bfloat16)
    master = model.float()
    master[0, 0] = 0.74609375
    runner = D.TwoLevelDeOsc(
        {"weight": (model, master, cfg)}, period=1000, accumulator_dtype=accumulator_dtype
    )
    runner.step()
    for transition in range(300):
        master[0, 0] = 0.75390625 if transition % 2 == 0 else 0.74609375
        runner.step()
    state = runner.state_dict()["states"]["weight"]
    assert state["window_steps"] == 300
    assert state["dist_w"][0, 0].item() == expected_weight_distance
    assert state["dist_q"][0, 0].item() == expected_quantized_distance
    assert state["dist_w"].count_nonzero() == state["dist_q"].count_nonzero() == 1
    assert runner.stats["snapped"] == 0
