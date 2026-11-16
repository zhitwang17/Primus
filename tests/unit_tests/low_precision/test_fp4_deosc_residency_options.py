# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU-only validation of operational DeOsc history placement."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("deosc_residency_preflight", ROOT / "tools/fp4_preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
C, D = PREFLIGHT.load_contracts()


def options(recipe="nvfp4", **updates):
    path = ROOT / f"examples/megatron/low_precision/two_level_fp4/{recipe}.yaml"
    values = yaml.safe_load(path.read_text())
    return {**values, "weight_deosc": True, **updates}


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("residency", ["device", "cpu"])
def test_supported_history_placement_preserves_recipe_and_rng_identity(recipe, residency):
    baseline = D.validate_fp4_options(options(recipe), require_runtime=False)
    selected = D.validate_fp4_options(
        options(recipe, weight_deosc_state_residency=residency), require_runtime=False
    )
    assert len(selected) == len(baseline) == 2
    for before, after in zip(baseline, selected):
        assert before.to_dict() == after.to_dict()
        assert before.config.fingerprint == after.config.fingerprint
        assert before.config.rng_fingerprint == after.config.rng_fingerprint
        assert "weight_deosc_state_residency" not in after.config.to_dict()


@pytest.mark.parametrize("residency", ["host", "CPU", "", 0, True])
@pytest.mark.parametrize("recipe", [None, "mxfp4", "nvfp4", "amdfp4"])
def test_invalid_residency_rejected_even_for_non_two_level_routes(recipe, residency):
    args = options() if recipe in ("nvfp4", "amdfp4") else {"fp4": "e2m1" if recipe else None}
    args.update(fp4_recipe=recipe, weight_deosc_state_residency=residency)
    with pytest.raises(ValueError, match="weight_deosc_state_residency must be device or cpu"):
        D.validate_fp4_options(args, require_runtime=False)


@pytest.mark.parametrize(
    "update,reason",
    [
        ({"fp4_recipe": "mxfp4"}, "enabled NVFP4/AMDFP4"),
        ({"fp4": None}, "enabled NVFP4/AMDFP4"),
        ({"weight_deosc": False}, "weight_deosc=true"),
        ({"weight_deosc": "false"}, "weight_deosc=true"),
        ({"fp4_backend": "auto"}, "fp4_backend=reference"),
        ({"fp4_backend": "te"}, "fp4_backend=reference"),
    ],
)
def test_cpu_history_fails_closed_for_unimplemented_routes(update, reason):
    args = options(weight_deosc_state_residency="cpu", **update)
    with pytest.raises(ValueError, match=reason):
        D.validate_fp4_options(args, require_runtime=False)
    with pytest.raises(ValueError, match=reason):
        D.resolve_fp4_plan(args)


def test_residency_fallback_and_legacy_mxfp4_default():
    args = options(weight_deosc_state_residency="cpu")
    assert D.deosc_state_residency({}, args) == "cpu"
    assert D.deosc_state_residency(SimpleNamespace(weight_deosc_state_residency="device"), args) == "device"
    assert D.resolve_fp4_plan({}, fallback=args).config == D.build_two_level_config(args)
    assert D.deosc_state_residency({}) == "device"
    assert D.validate_fp4_options({"fp4": "e2m1", "fp4_recipe": "mxfp4"}, require_runtime=False) == ()


def test_registered_default_keeps_legacy_device_history():
    path = ROOT / "primus/configs/modules/megatron/primus_megatron_module.yaml"
    assert yaml.safe_load(path.read_text())["weight_deosc_state_residency"] == "device"
