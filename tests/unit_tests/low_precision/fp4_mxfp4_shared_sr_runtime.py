# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Small numerical/consumer validation, not a model-training job.

CPU: TRITON_INTERPRET=1 python .../fp4_mxfp4_shared_sr_runtime.py
GPU: FP4_SHARED_SR_DEVICE=cuda python .../fp4_mxfp4_shared_sr_runtime.py
"""

import copy
import importlib
import json
import os
from unittest.mock import patch

import numpy as np
import torch
from test_fp4_gpu_rng import C, philox_uniform

S = importlib.import_module(C.__package__ + ".fp4_mxfp4_shared_sr")


def decoded(data, scale):
    packed = data.view(torch.uint8).cpu().numpy()
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(packed.shape[0], -1)
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    values = grid[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    scales = np.exp2(scale.view(torch.uint8).cpu().numpy().astype(np.float32) - 127)
    return values * np.repeat(scales, 32, axis=-1)


def reference(x, scales, reservation, *, rht=False, logical_base=0, logical_stride=None):
    x = x.float().cpu().numpy()
    r, c = x.shape
    cp = scales.shape[-1] * 32
    values = np.zeros((r, cp), dtype=np.float32)
    values[:, :c] = x
    if rht:
        # Independent fixed Sylvester H16 matrix; binary-fraction test inputs
        # make every intermediate exact, independent of reduction ordering.
        h = np.ones((1, 1), dtype=np.float32)
        for _ in range(4):
            h = np.block([[h, h], [h, -h]])
        values = (values.reshape(r, -1, 16) @ (h * np.float32(0.25))).reshape(r, cp)
    scale = np.exp2(scales.view(torch.uint8).cpu().numpy().astype(np.float32) - 127)
    magnitude = np.minimum(np.abs(values / np.repeat(scale, 32, axis=1)), np.float32(6))
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    upper = np.searchsorted(grid, magnitude, side="left")
    lower = np.maximum(upper - 1, 0)
    lo, hi = grid[lower], grid[upper]
    probability = (magnitude - lo) / np.where(hi > lo, hi - lo, np.float32(1))
    logical_stride = cp if logical_stride is None else logical_stride
    indices = logical_base + np.arange(r)[:, None] * logical_stride + np.arange(cp)[None, :]
    samples = philox_uniform(reservation["seed"], reservation["offset"], int(indices.max()) + 1)
    code = np.where(samples[indices] < probability, upper, lower).astype(np.uint8)
    code |= np.signbit(values).astype(np.uint8) << 3
    return code[:, 0::2] | (code[:, 1::2] << 4)


def kernel_checks(device):
    reservation = {"seed": 0x123456789ABCDEF, "offset": 2**32 + 19}
    for shape in ((17, 65), (32, 128), (1, 33)):
        x = ((torch.arange(np.prod(shape)).reshape(shape) % 97 - 48) / 16).to(device)
        for transpose in (False, True):
            oriented = x.T if transpose else x
            r, c = oriented.shape
            cp = (c + 127) // 128 * 128
            scales = torch.full((r, cp // 32), 127, dtype=torch.uint8, device=device).view(
                torch.float8_e8m0fnu
            )
            for rht in (False, True):
                out = torch.empty((r, cp // 2), dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
                S._launch(oriented, out, scales, reservation, rht=rht)
                expected = reference(oriented, scales, reservation, rht=rht)
                np.testing.assert_array_equal(out.view(torch.uint8).cpu().numpy(), expected)
    # Exercise real grouped rewrite/routing with an empty expert, unequal
    # padded spans, and an unused upper-bound allocation tail. Each expert's
    # H16 begins at its own unpadded source offset.
    x = ((torch.arange(49 * 65).reshape(49, 65) % 97 - 48) / 16).to(device)
    lengths = torch.tensor([17, 0, 32], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 17, 17, 49], dtype=torch.int64, device=device)
    pads = torch.tensor([0, 128, 128, 384], dtype=torch.int64, device=device)
    scales = torch.full((65, 512 // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    for rht in (False, True):
        packed = torch.full((65, 512 // 2), 0xA5, dtype=torch.uint8, device=device).view(
            torch.float4_e2m1fn_x2
        )
        with patch.object(S, "_reserve", return_value=reservation):
            S._rewrite(
                x, packed, scales, axis=0, rht=rht, group_lens=lengths, group_offs=offsets, padded_offs=pads
            )
        for first, last, start, end in ((0, 17, 0, 128), (17, 49, 128, 384)):
            expected = reference(
                x[first:last].T,
                scales[:, start // 32 : end // 32],
                reservation,
                rht=rht,
                logical_base=start,
                logical_stride=512,
            )
            np.testing.assert_array_equal(
                packed[:, start // 2 : end // 2].view(torch.uint8).cpu().numpy(), expected
            )
        assert torch.all(packed[:, 384 // 2 :].view(torch.uint8) == 0xA5)
    print(
        "shared SR kernel: independent Philox/E2M1 oracle, H16, tails, transpose and high counter passed",
        flush=True,
    )


def native_checks():
    from primus_turbo.pytorch.core.low_precision import ScalingRecipe
    from primus_turbo.pytorch.kernels.quantization import quantization_impl as Q

    original_dense, original_grouped = Q.quantize_mxfp4_impl, Q.grouped_quantize_mxfp4_impl
    S.install_shared_mxfp4_sr(1234)
    before_torch = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    row = ScalingRecipe(use_sr=True)
    col = ScalingRecipe(use_sr=True, use_rht=True)
    x = ((torch.arange(256 * 128).reshape(256, 128) % 97 - 48) / 16).to("cuda", torch.bfloat16)
    for data in (x, torch.stack((x, -x))):
        original = original_dense(
            data,
            torch.float4_e2m1fn_x2,
            None,
            32,
            True,
            row._replace(use_sr=False),
            col._replace(use_sr=False),
            2,
        )
        saved = copy.deepcopy(S.state_dict())
        actual = Q.quantize_mxfp4_impl(data, torch.float4_e2m1fn_x2, None, 32, True, row, col, 2)
        assert S.state_dict()["launches"] == saved["launches"] + 2
        for i in (1, 3):
            assert torch.equal(actual[i].view(torch.uint8), original[i].view(torch.uint8))
        after = S.state_dict()
        S.load_state_dict(saved)
        replay = Q.quantize_mxfp4_impl(data, torch.float4_e2m1fn_x2, None, 32, True, row, col, 2)
        assert S.state_dict() == after
        for a, b in zip(actual, replay):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    lengths = torch.tensor([0, 17, 111, 128], dtype=torch.int64, device="cuda")
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device="cuda"), lengths.cumsum(0)))
    original = original_grouped(
        x,
        torch.float4_e2m1fn_x2,
        None,
        32,
        lengths,
        offsets,
        True,
        row._replace(use_sr=False),
        col._replace(use_sr=False),
        2,
    )
    saved = copy.deepcopy(S.state_dict())
    native_before = {}

    def native_with_snapshot(*args, **kwargs):
        produced = original_grouped(*args, **kwargs)
        # Capture the SAME allocation before SR rewriting. Unlike comparing
        # separate calls, this verifies even uninitialized allocation-tail
        # bytes are untouched without requiring them to have a fixed value.
        native_before.update({i: produced[i].view(torch.uint8).clone() for i in (1, 3, 4, 5, 6, 7)})
        return produced

    actual = S._grouped_adapter(native_with_snapshot)(
        x, torch.float4_e2m1fn_x2, None, 32, lengths, offsets, True, row, col, 2
    )
    used_col_values = int(actual[7][-1].item())
    for i in (1, 3, 4, 5, 6, 7):
        a, b = actual[i].view(torch.uint8), original[i].view(torch.uint8)
        assert torch.equal(a, native_before[i]), f"SR overwrote producer scale/metadata output {i}"
        mismatch = a != b
        # Grouped producer allocates an upper bound, larger than the sum of
        # actual padded expert spans. No GEMM reads the final unused columns.
        live = mismatch[:, : used_col_values // 32] if i == 3 else mismatch
        print(
            json.dumps(
                {
                    "grouped_output": i,
                    "shape": list(a.shape),
                    "padded_offsets": actual[7].tolist(),
                    "same_allocation_unchanged": True,
                    "separate_allocation_mismatches": int(mismatch.sum().item()),
                    "consumer_region_mismatches": int(live.sum().item()),
                    "first_mismatches": mismatch.nonzero()[:8].tolist(),
                }
            ),
            flush=True,
        )
        assert not live.any(), f"Native producer changed scale/metadata {i} within the consumer region"
    S.load_state_dict(saved)
    replay = Q.grouped_quantize_mxfp4_impl(
        x, torch.float4_e2m1fn_x2, None, 32, lengths, offsets, True, row, col, 2
    )
    # Unused upper-bound allocation tail is not part of any GEMM operand.
    used_col_bytes = used_col_values // 2
    assert torch.equal(actual[0].view(torch.uint8), replay[0].view(torch.uint8))
    assert torch.equal(
        actual[2][:, :used_col_bytes].view(torch.uint8), replay[2][:, :used_col_bytes].view(torch.uint8)
    )
    for bad in (ScalingRecipe(use_sr=True, use_2d_block=True), ScalingRecipe(use_sr=True, shuffle_out=True)):
        saved = S.state_dict()
        try:
            Q.quantize_mxfp4_impl(x, torch.float4_e2m1fn_x2, 1, 32, False, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported SR recipe was accepted")
        assert S.state_dict() == saved
    assert torch.equal(before_torch, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state())
    print(
        "native producer: dense/batched/grouped UOS scales and actual padding preserved; replay and guards passed",
        flush=True,
    )
    native_consumers(x)
    S.uninstall_shared_mxfp4_sr()


def native_consumers(x):
    from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
    from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
    from primus_turbo.pytorch.ops.gemm_fp4 import gemm_fp4
    from primus_turbo.pytorch.ops.grouped_gemm_fp4 import grouped_gemm_fp4

    cfg = Float4QuantConfig(use_gradient_sr=True, use_preshuffle=False, scale_rounding_mode=2)
    generator = torch.Generator(device="cpu").manual_seed(20261116)

    def sample(shape):
        return torch.randn(shape, generator=generator).to(x.device, torch.bfloat16)

    def compare(actual, expected):
        assert torch.isfinite(actual).all()
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        # Quantization is intentionally lossy. This catches broken storage or
        # group offsets in the actual consumer; exact packed math is checked
        # separately against the independent oracle above.
        assert error.item() < 0.6, float(error)

    GlobalBackendManager.set_auto_tune(False)
    GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL)
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    try:
        a = sample((256, 128)).requires_grad_()
        b = sample((128, 128)).requires_grad_()
        before = S.state_dict()["launches"]
        out = gemm_fp4(a, b, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=cfg)
        assert S.state_dict()["launches"] == before  # Forward still native RNE.
        dy = sample(out.shape)
        out.backward(dy)
        torch.cuda.synchronize()
        assert S.state_dict()["launches"] > before
        compare(out, a.float() @ b.float().T)
        compare(a.grad, dy.float() @ b.float())
        compare(b.grad, dy.float().T @ a.float())
        a = sample((256, 128)).requires_grad_()
        b = sample((2, 128, 128)).requires_grad_()
        lengths = torch.tensor([17, 239], dtype=torch.int64, device=x.device)
        before = S.state_dict()["launches"]
        out = grouped_gemm_fp4(a, b, lengths, trans_b=True, out_dtype=torch.bfloat16, config=cfg)
        assert S.state_dict()["launches"] == before
        dy = sample(out.shape)
        out.backward(dy)
        torch.cuda.synchronize()
        assert S.state_dict()["launches"] > before
        for index, (start, end) in enumerate(((0, 17), (17, 256))):
            aa, bb, dd = a[start:end].float(), b[index].float(), dy[start:end].float()
            compare(out[start:end], aa @ bb.T)
            compare(a.grad[start:end], dd @ bb)
            compare(b.grad[index], dd.T @ aa)
        print(
            "native GEMM consumers: dense and ragged grouped forward/dgrad/wgrad completed with shared-SR backward",
            flush=True,
        )
    finally:
        GlobalBackendManager.reset()


if __name__ == "__main__":
    device = os.environ.get("FP4_SHARED_SR_DEVICE", "cpu")
    kernel_checks(device)
    if device == "cuda":
        native_checks()
