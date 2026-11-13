# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Real Triton specialization/cache keys and virtual large-offset CPU checks."""

import importlib
import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import compute_cache_key, create_function_from_signature

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("_fp4_dynamic_preflight", ROOT / "tools/fp4_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
contract, _ = preflight.load_contracts()
T = importlib.import_module(contract.__package__ + ".fp4_triton")
R = importlib.import_module(contract.__package__ + ".fp4_reference")

DYNAMIC = {
    "_quantize_kernel": "RNG_SEED RNG_OFFSET R C RP CP SB SR SC OB OR OC".split(),
    "_amax_input": "N R C SB SR SC".split(),
    "_validate_input_kernel": "N R C SB SR SC".split(),
    "_amax_reduce": ["N"],
    "_dequantize_kernel": "N R C RP CP OB OR OC".split(),
}


@pytest.mark.parametrize("name", DYNAMIC)
def test_real_triton_cache_key_reused_for_dynamic_shape_stride_rng_values(name):
    kernel = getattr(T, name)
    parameters = {parameter.name: parameter for parameter in kernel.params}
    for key in DYNAMIC[name]:
        assert not parameters[key].is_constexpr
        assert parameters[key].do_not_specialize
    # Construct an offline compiler backend. No driver/device initialization,
    # compilation or GPU execution is involved in the actual JIT argument binder.
    backend = HIPBackend(GPUTarget("hip", "gfx950", 64))
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    pointer = torch.empty(16)
    arguments = {
        parameter.name: (False if parameter.is_constexpr else pointer) for parameter in kernel.params
    }
    cache = {}
    keys = []
    for value in (1, 17, 32, 129, 4096, 8192):
        arguments.update({key: value for key in DYNAMIC[name]})
        _, specialization, options = binder(**arguments)
        keys.append(compute_cache_key(cache, specialization, options))
        for key in DYNAMIC[name]:
            assert specialization[kernel.arg_names.index(key)][1] is None
    assert len(set(keys)) == 1
    # The optimization must not erase real format/control specialization.
    control = next(parameter.name for parameter in kernel.params if parameter.is_constexpr)
    arguments[control] = True
    _, specialization, options = binder(**arguments)
    assert compute_cache_key(cache, specialization, options) != keys[0]


@triton.jit
def _virtual_offset_kernel(OUT, BASE, R, C, SB, SR, SC):
    index = BASE + tl.arange(0, 4).to(tl.int64)
    offset = T._input_offsets(index, R, C, SB, SR, SC)
    tl.store(OUT + tl.arange(0, 4), offset)


def _offset_check():
    # Only four int64 outputs are allocated. The logical tensor/strides exceed
    # 32-bit indexing; native interpreter arithmetic must match Python integers.
    for base, rows, cols, sb, sr, sc in (
        (2**32 + 7, 65536, 65536, 2**33, 65536, 1),
        (2**31 - 2, 3, 7, 2**35, 2**32, 3),
        (0, 2, 3, 2**34, 2**31, 5),
    ):
        output = torch.empty(4, dtype=torch.int64)
        _virtual_offset_kernel[(1,)](output, base, rows, cols, sb, sr, sc)
        expected = torch.tensor(
            [
                (index // (rows * cols)) * sb + (index // cols % rows) * sr + (index % cols) * sc
                for index in range(base, base + 4)
            ],
            dtype=torch.int64,
        )
        assert torch.equal(output, expected), (output, expected)


def test_dynamic_indexing_exceeds_int32_without_allocating_large_tensors():
    environment = {**os.environ, "TRITON_INTERPRET": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--offset-check"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real accelerator required; no GPU emulation")
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("direction,role", [("fwd", "weight"), ("wgrad", "grad_output")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_kernel_dynamic_cache_reuse_and_noncontiguous_exact_parity(recipe, direction, role, dtype):
    from test_fp4_gpu_rng import philox_uniform

    assert os.environ.get("TRITON_INTERPRET", "0") == "0"
    cfg = contract.TwoLevelFP4Config(recipe, quantizer="triton", sr_rng="triton_philox")
    oracle_cfg = replace(cfg, quantizer="torch", sr_rng="torch_cpu")
    operand = cfg.operand(direction, role)
    cached_keys = None
    for case, rows in enumerate((17, 33, 65)):
        # Noncontiguous input with a stable aligned base pointer; pointer alignment
        # is legitimately specialized independently of dynamic dimensions/strides.
        base = torch.linspace(-5.9, 5.9, rows * 19 * 2, device="cuda", dtype=dtype).reshape(rows, 38)
        x = base[:, ::2]
        assert x.shape == (rows, 19) and not x.is_contiguous()
        rng = contract.FP4RandomState.for_config(cfg)
        rng.sr_offset = 2**32 + case
        saved = rng.state_dict()
        reservation = contract.FP4RandomState.from_state_dict(saved, cfg).reserve("sr", 0)
        packed = T.quantize(x, cfg, operand, random=rng)
        direct_rng = contract.FP4RandomState.from_state_dict(saved, cfg)
        direct = T.qdq(x, cfg, operand, random=direct_rng)
        assert direct_rng.state_dict() == rng.state_dict()
        decoded = T.dequantize(packed, dtype=dtype)

        def oracle_uniform(shape, **kwargs):
            # Independent NumPy Philox oracle, never the production Triton RNG.
            values = philox_uniform(reservation["seed"], reservation["offset"], math.prod(shape))
            return torch.from_numpy(values.reshape(shape))

        with patch.object(torch, "rand", side_effect=oracle_uniform):
            expected = R.quantize(
                x.cpu(), oracle_cfg, operand, random=contract.FP4RandomState.for_config(oracle_cfg)
            )
        torch.cuda.synchronize()
        assert torch.equal(packed.data.cpu(), expected.data)
        assert torch.equal(packed.inner_scale.cpu(), expected.inner_scale)
        assert torch.equal(packed.outer_scale.cpu(), expected.outer_scale)
        expected_output = R.dequantize(expected, dtype=dtype)
        assert torch.equal(direct.cpu(), expected_output)
        assert torch.equal(decoded.cpu(), expected_output)
        # Inspect the real launch cache after warm-up of packed and direct variants.
        # Subsequent odd dimensions and new RNG offsets must add no specialization.
        keys = set(T._quantize_kernel.device_caches[torch.cuda.current_device()][0])
        assert keys
        if cached_keys is not None:
            assert keys == cached_keys
        cached_keys = keys


if __name__ == "__main__" and "--offset-check" in sys.argv:
    _offset_check()
