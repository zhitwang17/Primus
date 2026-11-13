# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Run with TRITON_INTERPRET=1: no accelerator or CUDA context is used."""

import copy
import importlib
import os
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch
import triton
import triton.language as tl
from test_fp4_gpu_rng import C, R, philox_uniform

T = importlib.import_module(C.__package__ + ".fp4_triton")


@triton.jit
def sample(OUT, SEED, OFFSET, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = T._philox_uniform(SEED, OFFSET, index)
    tl.store(OUT + index, value, index < N)


def main():
    device = os.environ.get("FP4_GPU_RNG_DEVICE", "cpu")
    before_global = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    for offset in (0, 2**32 - 1, 2**32 + 19, 2**63 + 7):
        for block in (64, 256):
            out = torch.empty(1031, dtype=torch.float32, device=device)
            sample[(triton.cdiv(out.numel(), block),)](out, 0x123456789ABCDEF, offset, out.numel(), block)
            np.testing.assert_array_equal(
                out.cpu().numpy(), philox_uniform(0x123456789ABCDEF, offset, out.numel())
            )

    # Only bypass the GPU placement gate for Triton's CPU interpreter. Every
    # preparation, reservation, quantize/QDQ kernel and finite check still runs.
    with patch.object(T, "_input", lambda x: None) if device == "cpu" else nullcontext():
        for recipe in ("nvfp4", "amdfp4"):
            cfg = C.TwoLevelFP4Config(recipe, quantizer="triton", sr_rng="triton_philox")
            for direction in ("dgrad", "wgrad"):
                operand = cfg.operand(direction, "grad_output")
                x = torch.linspace(-5.9, 5.9, 33 * 65).reshape(33, 65).to(device)
                rng = C.FP4RandomState.for_config(cfg)
                rng.sr_offset = 2**32 - 1
                saved = copy.deepcopy(rng.state_dict())
                reservation = C.FP4RandomState.from_state_dict(saved, cfg).reserve("sr", 0)
                with patch.object(torch, "rand", side_effect=AssertionError("GPU SR allocated a CPU array")):
                    packed = T.quantize(x, cfg, operand, random=rng)
                clone = C.FP4RandomState.from_state_dict(saved, cfg)
                direct = T.qdq(x, cfg, operand, random=clone)
                assert clone.state_dict() == rng.state_dict()
                assert torch.equal(direct, R.dequantize(packed))
                oracle_cfg = replace(cfg, quantizer="torch", sr_rng="torch_cpu")
                oracle_rng = C.FP4RandomState.for_config(oracle_cfg)

                def uniform(shape, **kwargs):
                    values = philox_uniform(reservation["seed"], reservation["offset"], int(np.prod(shape)))
                    return torch.from_numpy(values.reshape(shape))

                with patch.object(torch, "rand", side_effect=uniform):
                    expected = R.quantize(x.cpu(), oracle_cfg, operand, random=oracle_rng)
                assert torch.equal(packed.data.cpu(), expected.data)
                assert torch.equal(packed.inner_scale.cpu(), expected.inner_scale)
                assert torch.equal(direct.cpu(), R.dequantize(expected))
    assert torch.equal(before_global, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(before_cuda, torch.cuda.get_rng_state())
    print(f"GPU RNG {'device' if device == 'cuda' else 'interpreter'} parity passed")


if __name__ == "__main__":
    main()
