# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU contract tests; the interpreter exercises Triton math without a GPU."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fp4_preflight_gpu_rng", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, D = preflight.load_contracts()
R = importlib.import_module(C.__package__ + ".fp4_reference")
G = importlib.import_module(C.__package__ + ".fp4_rng")
S = importlib.import_module(C.__package__ + ".fp4_state")


def philox_words(seed, blocks):
    """Independent NumPy Philox4x32-10 oracle with explicit 32-bit arithmetic."""
    blocks = np.asarray(blocks, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    c0, c1 = blocks & mask, blocks >> np.uint64(32)
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    k0, k1 = np.uint64(seed & 0xFFFFFFFF), np.uint64(seed >> 32)
    for _ in range(10):
        p0, p2 = c0 * np.uint64(0xD2511F53), c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (p2 >> np.uint64(32)) ^ c1 ^ k0,
            p2 & mask,
            (p0 >> np.uint64(32)) ^ c3 ^ k1,
            p0 & mask,
        )
        k0 = (k0 + np.uint64(0x9E3779B9)) & mask
        k1 = (k1 + np.uint64(0xBB67AE85)) & mask
    return np.stack((c0, c1, c2, c3), axis=-1).astype(np.uint32)


def philox_uniform(seed, offset, count):
    blocks = np.arange((count + 3) // 4, dtype=np.uint64) + np.uint64(offset)
    words = philox_words(seed, blocks).reshape(-1)[:count]
    return (words >> np.uint32(8)).astype(np.float32) * np.float32(2.0**-24)


def test_legacy_fingerprint_bitstream_and_checkpoint_identity_are_unchanged():
    cfg = C.TwoLevelFP4Config("nvfp4")
    legacy = dict(
        recipe="nvfp4",
        block_size=16,
        weight_2d=True,
        outer_scope="tensor",
        inner_storage="float32_grid",
        rht_size=32,
        rht_randomized=True,
        gradient_sr=True,
        rng_seed=1234,
        quantizer="torch",
        version=1,
    )
    assert cfg.to_dict() == legacy
    assert cfg.fingerprint == hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    # Frozen from the pre-GPU-RNG contract, not computed from the new dataclass.
    assert cfg.fingerprint == "8c24d7213a97670a3471e844f426c1a8ea278a5687ece9d865ad024a3e6b41f3"
    assert cfg.rng_fingerprint == "d62ab140328ca3db30437dc61ca7d52f78382d116edfd41b16d200404315ef2a"
    old_rng, new_rng = C.FP4RandomState.for_config(cfg), C.FP4RandomState.for_config(cfg)
    reservation = old_rng.reserve("sr", 8)
    identity = f"torch-cpu-reservation-sha256-v1:{reservation['seed']}:sr:0"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") & (2**63 - 1)
    expected = torch.rand(32, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(torch.rand(32, generator=R._generator(new_rng, "sr", 32)), expected)
    assert G.rng_implementation([cfg]) == "torch-cpu-reservation-sha256-v1"


def test_philox_known_vector_uniform_distribution_and_counter_high_word():
    # Random123's zero-key/zero-counter Philox4x32-10 known-answer vector.
    assert philox_words(0, [0]).tolist() == [[0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]]
    values = philox_uniform(1234, 0, 2**18)
    assert 0 <= values.min() < values.max() < 1
    assert abs(float(values.mean()) - 0.5) < 0.003
    assert abs(float(values.var()) - 1 / 12) < 0.001
    bins, _ = np.histogram(values, bins=16, range=(0, 1))
    assert np.max(np.abs(bins - values.size / 16)) < 600
    assert abs(float(np.corrcoef(values[:-1], values[1:])[0, 1])) < 0.01
    assert not np.array_equal(philox_words(9, [0, 1]), philox_words(9, [2**32, 2**32 + 1]))


def test_reservations_restore_domain_separation_and_overflow_are_strict():
    cfg = C.TwoLevelFP4Config("amdfp4", quantizer="triton", sr_rng="triton_philox")
    rng = C.FP4RandomState.for_config(cfg, "rank0/layer1")
    before_global = torch.get_rng_state().clone()
    first = G.reserve_philox_sr(rng, 17)
    assert first["offset"] == 0 and first["blocks"] == 5
    clone = C.FP4RandomState.from_state_dict(json.loads(json.dumps(rng.state_dict())), cfg, rng.stream_id)
    second = G.reserve_philox_sr(rng, 64)
    assert G.reserve_philox_sr(clone, 64) == second
    assert not np.array_equal(
        philox_uniform(second["seed"], second["offset"], 16),
        philox_uniform(first["seed"], first["offset"], 16),
    )
    other = G.reserve_philox_sr(C.FP4RandomState.for_config(cfg, "rank1/layer1"), 17)
    assert first["seed"] != other["seed"]
    assert rng.reserve("rht", 1)["seed"] != first["seed"]
    assert torch.equal(before_global, torch.get_rng_state())
    rng.sr_offset = 2**64 - 1
    assert G.reserve_philox_sr(rng, 4)["offset"] == 2**64 - 1
    with pytest.raises(ValueError, match="64-bit"):
        G.reserve_philox_sr(rng, 1)
    assert rng.sr_offset == 2**64


@pytest.mark.parametrize("kwargs", [dict(sr_rng="bad"), dict(sr_rng="triton_philox")])
def test_invalid_rng_config_is_rejected(kwargs):
    with pytest.raises(ValueError, match="sr_rng"):
        C.TwoLevelFP4Config("nvfp4", **kwargs)


def test_dispatch_selects_gpu_sr_only_with_the_triton_quantizer():
    options = dict(fp4_recipe="nvfp4", fp4_quantizer="triton", fp4_sr_rng="triton_philox")
    assert D.build_two_level_config(options).sr_rng == "triton_philox"
    options["fp4_quantizer"] = "torch"
    with pytest.raises(ValueError, match="requires quantizer=triton"):
        D.build_two_level_config(options)


def test_gpu_rng_checkpoint_identity_and_mismatch_prevent_partial_restore():
    def layer(mode):
        module = torch.nn.Linear(16, 16, bias=False)
        module.fp4_config = C.TwoLevelFP4Config("nvfp4", quantizer="triton", sr_rng=mode)
        module.fp4_configs = (module.fp4_config,)
        module.fp4_rngs = {}
        S.bind_fp4_model(module)
        return module

    model = layer("triton_philox")
    G.reserve_philox_sr(model.fp4_rng, 99)
    saved = copy.deepcopy(S.fp4_state_dict(model))
    assert saved["rng_implementation"] == {
        "sr": ["philox4x32-10-counter64-u24-v1"],
        "rht": "torch-cpu-reservation-sha256-v1",
    }
    clone = layer("triton_philox")
    S.load_fp4_state_dict(clone, saved)
    assert G.reserve_philox_sr(clone.fp4_rng, 31) == G.reserve_philox_sr(model.fp4_rng, 31)
    legacy = layer("torch_cpu")
    before = S.fp4_state_dict(legacy)
    with pytest.raises(ValueError, match="rng_implementation"):
        S.load_fp4_state_dict(legacy, saved)
    assert S.fp4_state_dict(legacy) == before
    cfg = model.fp4_config
    with pytest.raises(ValueError, match="Torch reference"):
        R.qdq(torch.ones(16, 16), cfg, cfg.operand("fwd", "weight"))


def test_triton_interpreter_matches_oracle_quantization_and_resume():
    pytest.importorskip("triton")
    environment = dict(os.environ, TRITON_INTERPRET="1")
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fp4_gpu_rng_interpreter.py"))],
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "GPU RNG interpreter parity passed" in run.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an explicitly allocated GPU")
def test_device_kernel_matches_oracle_quantization_and_resume():
    pytest.importorskip("triton")
    environment = dict(os.environ, TRITON_INTERPRET="0", FP4_GPU_RNG_DEVICE="cuda")
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fp4_gpu_rng_interpreter.py"))],
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "GPU RNG device parity passed" in run.stdout
