# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""The supplemental prefix retains default schedule and never activates DeOsc."""
import pytest
from test_fp4_training_recipe import _build


@pytest.mark.parametrize("recipe", ["mxfp4", "nvfp4", "amdfp4"])
def test_prefix_changes_only_exit_and_symmetric_evaluation_cadence(recipe):
    _, default, _ = _build(recipe)
    _, args, manifest = _build(recipe, steps=128, supplemental_dp4_prefix=True)
    assert {k: v for k, v in args.items() if k not in ("exit_interval", "eval_interval")} == {
        k: v for k, v in default.items() if k not in ("exit_interval", "eval_interval")
    }
    assert args["exit_interval"] == 128
    assert args["eval_interval"] == args["eval_iters"] == 32
    assert args["train_iters"] == 1200000 and args["lr_decay_iters"] == 1199883
    assert args["weight_deosc_start_step"] == 768 and args["weight_deosc_period"] == 100
    assert manifest["scope"] == "supplemental_dp4_prefix"
    assert manifest["validation_steps"] == [32, 64, 96, 128]
    assert manifest["evaluation_interval"] == manifest["evaluation_iters"] == 32
    assert manifest["expected_world_size"] == 4 and manifest["gradient_accumulation_steps"] == 2
    assert manifest["environment"]["GPUS_PER_NODE"] == "4"
    assert manifest["deosc_activated_in_prefix"] is False
    assert manifest["default_dp8_3000_comparison"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 8},
        {"steps": 3000},
        {"admission": True},
        {"supplemental_dp4": True, "admission": True, "steps": 8},
        {"recompute": False},
    ],
)
def test_prefix_rejects_shortcuts_and_other_scopes(override):
    options = dict(steps=128, supplemental_dp4_prefix=True)
    options.update(override)
    with pytest.raises(ValueError, match="Supplemental DP4"):
        _build("nvfp4", **options)
