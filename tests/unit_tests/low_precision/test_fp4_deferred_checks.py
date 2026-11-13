# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Independent contracts and real interpreter/device tests for grouped checks."""

import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fp4_preflight_checks", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, D = preflight.load_contracts()
R = importlib.import_module(C.__package__ + ".fp4_reference")


def test_default_config_and_old_rng_checkpoint_keep_their_frozen_identity():
    legacy = {
        "recipe": "nvfp4",
        "block_size": 16,
        "weight_2d": True,
        "outer_scope": "tensor",
        "inner_storage": "float32_grid",
        "rht_size": 32,
        "rht_randomized": True,
        "gradient_sr": True,
        "rng_seed": 1234,
        "quantizer": "torch",
        "version": 1,
    }
    cfg = C.TwoLevelFP4Config(**legacy)
    assert cfg.value_checks == "strict"
    assert cfg.to_dict() == legacy == replace(cfg, value_checks="strict").to_dict()
    # Captured before adding deferred checks, not derived from the new dataclass.
    assert cfg.fingerprint == "8c24d7213a97670a3471e844f426c1a8ea278a5687ece9d865ad024a3e6b41f3"
    assert cfg.rng_fingerprint == "d62ab140328ca3db30437dc61ca7d52f78382d116edfd41b16d200404315ef2a"
    saved = {
        "version": 1,
        "recipe_fingerprint": "8c24d7213a97670a3471e844f426c1a8ea278a5687ece9d865ad024a3e6b41f3",
        "seed": 1234,
        "stream_id": "rank0/layer1",
        "sr_offset": 11,
        "rht_offset": 7,
    }
    assert C.FP4RandomState.from_state_dict(saved, cfg, "rank0/layer1").state_dict() == saved


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("sr_rng", ["torch_cpu", "triton_philox"])
def test_opt_in_identity_separates_checkpoints_without_changing_random_streams(recipe, sr_rng):
    strict = C.TwoLevelFP4Config(recipe, quantizer="triton", sr_rng=sr_rng)
    deferred = replace(strict, value_checks="grouped_deferred")
    assert "value_checks" not in strict.to_dict()
    assert deferred.to_dict()["value_checks"] == "grouped_deferred"
    assert strict.fingerprint != deferred.fingerprint
    assert strict.rng_fingerprint == deferred.rng_fingerprint
    stream = "rank0/layer1/" + strict.rng_fingerprint
    a, b = (C.FP4RandomState.for_config(cfg, stream) for cfg in (strict, deferred))
    for domain in ("rht", "sr", "sr", "rht"):
        assert a.reserve(domain, 19) == b.reserve(domain, 19)
    before = copy.deepcopy(b.state_dict())
    with pytest.raises(ValueError, match="recipe/version"):
        C.FP4RandomState.from_state_dict(a.state_dict(), deferred, stream)
    assert b.state_dict() == before
    # Validation mode also must not alter randomized H32 or its reservations.
    x = torch.linspace(-1, 1, 33 * 17).reshape(33, 17)
    dy = torch.linspace(2, -2, 33 * 19).reshape(33, 19)
    ax, ay = R.rht_pair(x, dy, strict, random=a)
    bx, by = R.rht_pair(x, dy, deferred, random=b)
    assert torch.equal(ax, bx) and torch.equal(ay, by)
    assert (a.sr_offset, a.rht_offset) == (b.sr_offset, b.rht_offset)


@pytest.mark.parametrize("options", [{"value_checks": "off"}, {"value_checks": "grouped_deferred"}])
def test_invalid_mode_or_torch_deferred_request_fails_closed(options):
    with pytest.raises(ValueError, match="value_checks"):
        C.TwoLevelFP4Config("nvfp4", **options)


def test_cpu_reference_remains_strict_for_qdq_and_rht():
    cfg = C.TwoLevelFP4Config("nvfp4")
    x = torch.ones(17, 16)
    x[1, 1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        R.qdq(x, cfg, cfg.operand("fwd", "input"))
    with pytest.raises(ValueError, match="non-finite"):
        R.rht_pair(x, torch.ones_like(x), cfg, random=C.FP4RandomState.for_config(cfg))


def test_default_torch_grouped_fallback_needs_no_triton_scope(monkeypatch):
    autograd = importlib.import_module(C.__package__ + ".fp4_autograd")
    monkeypatch.setitem(sys.modules, C.__package__ + ".fp4_triton", None)
    cfg = C.TwoLevelFP4Config("nvfp4", rht_size=0, gradient_sr=False)
    x = torch.linspace(-1, 1, 17 * 16).reshape(17, 16).requires_grad_()
    weights = torch.ones(2, 19, 16, requires_grad=True)
    result = autograd.grouped_linear(x, weights, [0, 17], cfg)
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights.grad).all()
    assert torch.equal(weights.grad[0], torch.zeros_like(weights.grad[0]))
    with torch.no_grad():
        x[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        autograd.grouped_linear(x, weights, [0, 17], cfg)


def run_runtime_suite(device):
    environment = dict(
        os.environ,
        TRITON_INTERPRET="1" if device == "cpu" else "0",
        FP4_VALUE_CHECKS_DEVICE=device,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(Path(__file__).with_name("fp4_deferred_checks_runtime.py")),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "passed" in run.stdout and "skipped" not in run.stdout, run.stdout
    print(json.dumps({"value_checks_runtime": device, "pytest": run.stdout.strip()}))


def test_interpreter_value_checks_and_grouped_autograd():
    pytest.importorskip("triton")
    run_runtime_suite("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an explicitly allocated GPU")
def test_device_kernel_value_checks_and_grouped_autograd():
    pytest.importorskip("triton")
    run_runtime_suite("cuda")
