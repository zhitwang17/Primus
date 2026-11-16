# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Shared SR opts into the existing two-level numerical/RNG contract exactly."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("fp4_shared_policy_preflight", ROOT / "tools/fp4_preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
C, D = PREFLIGHT.load_contracts()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_shared_policy_preserves_existing_philox_and_rht_streams(recipe):
    options = dict(fp4_recipe=recipe, fp4_quantizer="triton", fp4_sr_rng="triton_philox")
    old = D.build_two_level_config(options)
    shared = D.build_two_level_config({**options, "fp4_gradient_rounding_policy": "shared_philox"})
    assert shared == old
    assert shared.fingerprint == old.fingerprint
    assert shared.rng_fingerprint == old.rng_fingerprint
    a, b = C.FP4RandomState.for_config(old, "rank0/layer1"), C.FP4RandomState.for_config(
        shared, "rank0/layer1"
    )
    for stream, blocks in (("sr", 17), ("rht", 8), ("sr", 31)):
        assert a.reserve(stream, blocks) == b.reserve(stream, blocks)


@pytest.mark.parametrize(
    "override",
    [
        dict(fp4_gradient_sr=False),
        dict(fp4_sr_rng="torch_cpu"),
        dict(fp4_quantizer="torch", fp4_sr_rng="torch_cpu"),
    ],
)
def test_shared_policy_refuses_silent_recipe_fallback(override):
    options = dict(
        fp4_recipe="nvfp4",
        fp4_quantizer="triton",
        fp4_sr_rng="triton_philox",
        fp4_gradient_rounding_policy="shared_philox",
    )
    with pytest.raises(ValueError, match="shared_philox requires"):
        D.build_two_level_config({**options, **override})


def test_shared_policy_validates_fallback_and_unknown_options():
    fallback = dict(fp4_gradient_rounding_policy="shared_philox")
    with pytest.raises(ValueError, match="shared_philox requires"):
        D.build_two_level_config(dict(fp4_recipe="amdfp4"), fallback)
    with pytest.raises(ValueError, match="fp4_gradient_rounding_policy"):
        D.build_two_level_config(dict(fp4_recipe="amdfp4", fp4_gradient_rounding_policy="typo"))
