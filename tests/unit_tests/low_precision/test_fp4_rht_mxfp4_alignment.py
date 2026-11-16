# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Independent native-order H16 oracle and real QDQ/GEMM-boundary checks.

These are CPU numerical tests. They establish neither ROCm kernel execution nor
bitwise identity between native FP4 GEMMs and reference QDQ/BF16 GEMMs.
"""
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "fp4_preflight_native_rht_test", ROOT / "tools/fp4_preflight.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
C, _ = preflight.load_contracts()
R = importlib.import_module(C.__package__ + ".fp4_reference")
A = importlib.import_module(C.__package__ + ".fp4_autograd")
S = importlib.import_module(C.__package__ + ".fp4_state")


def native_h16_oracle(value):
    """Scalar NumPy FP32 H4 then simultaneous XOR1/XOR2 lane exchanges.

    Order is taken from Turbo csrc/kernels/quantization/quantization_mxfp4.cu
    rht16_inplace, independently of the new Torch implementation. No GEMM or
    production helper implements this oracle.
    """
    array = value.detach().float().cpu().numpy()
    rows, cols = array.shape
    padded = np.zeros(((rows + 15) // 16 * 16, cols), dtype=np.float32)
    padded[:rows] = array
    for begin in range(0, len(padded), 16):
        for column in range(cols):
            registers = np.empty((4, 4), dtype=np.float32)
            for lane in range(4):
                v0, v1, v2, v3 = padded[begin + lane * 4 : begin + lane * 4 + 4, column]
                a0, a1 = np.float32(v0 + v1), np.float32(v0 - v1)
                a2, a3 = np.float32(v2 + v3), np.float32(v2 - v3)
                registers[lane] = [
                    np.float32(a0 + a2),
                    np.float32(a1 + a3),
                    np.float32(a0 - a2),
                    np.float32(a1 - a3),
                ]
            for xor in (1, 2):
                previous = registers.copy()
                for lane in range(4):
                    for element in range(4):
                        peer, local = previous[lane ^ xor, element], previous[lane, element]
                        registers[lane, element] = np.float32(peer - local if lane & xor else peer + local)
            padded[begin : begin + 16, column] = (registers * np.float32(0.25)).reshape(16)
    return torch.from_numpy(padded)


def config(recipe, **kwargs):
    return C.TwoLevelFP4Config(recipe, rht_size=16, rht_randomized=False, rht_mode="mxfp4", **kwargs)


def exponent_stress(rows, cols):
    # Exact BF16 powers/mantissas with cancellations, widely separated exponents
    # and subnormal-sensitive small magnitudes; all native additions stay finite.
    sequence = np.array(
        [
            2.0**80,
            1.0,
            -(2.0**80),
            3.0,
            2.0**-120,
            -(2.0**-110),
            2.0**-110,
            -(2.0**-120),
            1.125,
            -0.0625,
            2.0**30,
            -(2.0**30),
            -8.0,
            0.125,
            5.0,
            -0.5,
        ],
        dtype=np.float32,
    )
    array = np.resize(sequence, rows * cols).reshape(rows, cols)
    # Offset columns to avoid each reduction seeing only repeated magnitudes.
    for column in range(cols):
        array[:, column] = np.roll(array[:, column], column)
    return torch.from_numpy(array).bfloat16()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("rows", [0, 1, 15, 16, 17, 31, 32, 33, 65])
@pytest.mark.parametrize("autocast", [False, True])
def test_native_order_fp32_oracle_with_tails_and_cpu_autocast(recipe, rows, autocast):
    cfg = config(recipe)
    x, dy = exponent_stress(rows, 5), exponent_stress(rows, 7)
    private = C.FP4RandomState.for_config(cfg)
    before_private, before_global = private.state_dict(), torch.random.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual_x, actual_dy = R.rht_pair(x, dy, cfg, random=private)
    assert actual_x.dtype == actual_dy.dtype == torch.float32
    # Compare FP32 representations, including sign of zero, not an FP16/BF16 tolerance.
    assert torch.equal(actual_x.view(torch.int32), native_h16_oracle(x).view(torch.int32))
    assert torch.equal(actual_dy.view(torch.int32), native_h16_oracle(dy).view(torch.int32))
    assert private.state_dict() == before_private
    assert torch.equal(torch.random.get_rng_state(), before_global)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_paired_transform_preserves_exact_integer_dot_and_separates_halves(recipe):
    x = (torch.arange(33 * 3).reshape(33, 3) % 19 - 9).float()
    dy = (torch.arange(33 * 5).reshape(33, 5) % 23 - 11).float()
    xh, dyh = R.rht_pair(x, dy, config(recipe))
    torch.testing.assert_close(dyh.T @ xh, dy.T @ x, rtol=0, atol=0)
    impulse = torch.zeros(32, 2, dtype=torch.bfloat16)
    impulse[0, 0], impulse[16, 1] = 4, 8
    result, _ = R.rht_pair(impulse, impulse, config(recipe))
    assert torch.equal(result[:16, 0], torch.ones(16))
    assert torch.equal(result[16:, 1], torch.full((16,), 2.0))
    assert torch.count_nonzero(result[:16, 1]) == torch.count_nonzero(result[16:, 0]) == 0


def test_cancellation_vector_distinguishes_native_order_from_fp32_gemm():
    x = exponent_stress(32, 5).float()
    expected = native_h16_oracle(x)
    # Merely changing legacy GEMM's output dtype to FP32 would still be wrong.
    dot_order = (R._rht_base(16) @ x.reshape(-1, 16, 5)).reshape_as(x) * 0.25
    assert torch.count_nonzero(expected.view(torch.int32) != dot_order.view(torch.int32)) > 0
    actual, _ = R.rht_pair(x, x, config("nvfp4"))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("fold_scale", [False, True])
def test_installed_flydsl_stock_scalar_and_packed_helpers_match_oracle(fold_scale):
    # Read the actual installed backend without importing FlyDSL or its GPU
    # runtime. Execute only these pure arithmetic helper bodies on FP32 arrays.
    relative = Path("primus_turbo/flydsl/quantization/mxfp4_quant_kernel_stock.py")
    paths = [Path(entry) / relative for entry in sys.path if (Path(entry) / relative).is_file()]
    if not paths:
        pytest.skip("Installed Turbo stock helper source is unavailable")
    tree = ast.parse(paths[0].read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_h4", "_rht16", "_rht16_pair"}
    ]
    assert len(selected) == 3

    class FP32Constants(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, float):
                return ast.copy_location(
                    ast.Call(func=ast.Name(id="f32", ctx=ast.Load()), args=[node], keywords=[]), node
                )
            return node

    class Vector:
        @staticmethod
        def from_elements(elements, dtype):
            return np.asarray(elements, dtype=dtype)

    class Scalars:
        Float32 = np.float32

    module = ast.fix_missing_locations(FP32Constants().visit(ast.Module(body=selected, type_ignores=[])))
    namespace = {"range_constexpr": range, "Vec": Vector, "fx": Scalars, "f32": np.float32}
    exec(compile(module, str(paths[0]), "exec"), namespace)
    value = exponent_stress(32, 5).float()
    expected = native_h16_oracle(value)
    scalar, packed = torch.empty_like(value), torch.empty_like(value)
    for column in range(value.shape[1]):
        values = list(value[:, column].numpy())
        result = namespace["_rht16"](values[:16], post_scale=not fold_scale)
        result += namespace["_rht16"](values[16:], post_scale=not fold_scale)
        pair = namespace["_rht16_pair"](values, post_scale=not fold_scale)
        normalization = np.float32(0.25 if fold_scale else 1.0)
        scalar[:, column] = torch.from_numpy(np.asarray(result, dtype=np.float32) * normalization)
        packed[:, column] = torch.from_numpy(np.asarray(pair, dtype=np.float32) * normalization)
    assert torch.equal(scalar.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(packed.view(torch.int32), expected.view(torch.int32))


class MatmulDtypes(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.calls.append((args[0].dtype, args[1].dtype, result.dtype))
        return result


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_real_qdq_sees_fp32_rht_but_matmul_is_bf16_and_outer_is_whole_buffer(
    recipe, grouped, autocast, monkeypatch
):
    cfg = config(recipe, gradient_sr=False)
    lengths = [1, 0, 17]
    rows = sum(lengths)
    x = torch.linspace(-0.91, 1.13, rows * 17).reshape(rows, 17).bfloat16()
    x[0] *= 0.01  # Distinct expert-local maxima must not become separate outers.
    x.requires_grad_()
    dy = torch.linspace(0.17, -1.31, rows * 19).reshape(rows, 19).bfloat16()
    dy[0] *= 0.01
    weight_shape = (3, 19, 17) if grouped else (19, 17)
    weights = torch.linspace(-0.81, 0.93, int(np.prod(weight_shape))).reshape(weight_shape).bfloat16()
    weights.requires_grad_()
    original_quantize, original_qdq = R.quantize, A.qdq
    quantized, qdq_calls = [], []

    def capture_quantize(value, active, operand, **kwargs):
        packed = original_quantize(value, active, operand, **kwargs)
        quantized.append(
            (operand.direction, operand.role, value.detach().clone(), packed.outer_scale.clone())
        )
        return packed

    def capture_qdq(value, active, operand, **kwargs):
        output = original_qdq(value, active, operand, **kwargs)
        qdq_calls.append((operand.direction, operand.role, value.dtype, kwargs.get("dtype"), output.dtype))
        return output

    monkeypatch.setattr(R, "quantize", capture_quantize)
    monkeypatch.setattr(A, "qdq", capture_qdq)
    recorder = MatmulDtypes()
    with recorder, torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        y = A.grouped_linear(x, weights, lengths, cfg) if grouped else A.linear(x, weights, cfg)
        y.backward(dy)
    assert recorder.calls and all(call == (torch.bfloat16,) * 3 for call in recorder.calls)
    assert x.grad.dtype == weights.grad.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights.grad).all()
    assert all(row[3:] == (torch.bfloat16, torch.bfloat16) for row in qdq_calls)
    assert all(row[2] == (torch.float32 if row[0] == "wgrad" else torch.bfloat16) for row in qdq_calls)

    parts_x = A._split_pad(x.detach(), lengths) if grouped else [x.detach()]
    parts_dy = A._split_pad(dy, lengths) if grouped else [dy]
    expected = {
        "input": [native_h16_oracle(part) for part in parts_x],
        "grad_output": [native_h16_oracle(part) for part in parts_dy],
    }
    for role in expected:
        all_values = torch.cat(expected[role])
        expected_outer = (all_values.abs().amax() / (6.0 * cfg.inner_max)).clamp_min(1e-30).reshape(1)
        records = [row for row in quantized if row[:2] == ("wgrad", role)]
        used = [part for part in expected[role] if part.numel()]
        assert len(records) == len(used)
        for record, wanted in zip(records, used):
            torch.testing.assert_close(record[2], wanted, rtol=0, atol=0)
            torch.testing.assert_close(record[3], expected_outer, rtol=0, atol=0)
        assert torch.count_nonzero(all_values != all_values.bfloat16().float()) > 0
    if grouped:
        assert torch.count_nonzero(weights.grad[1]) == 0
        bank = [row for row in quantized if row[:2] == ("fwd", "weight")]
        assert len(bank) == 1 and bank[0][2].shape == weights.shape
        expected_outer = (weights.detach().float().abs().amax() / (6.0 * cfg.inner_max)).reshape(1)
        torch.testing.assert_close(bank[0][3], expected_outer, rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_entirely_empty_grouped_backward_is_finite_and_does_not_do_matmul(recipe):
    cfg = config(recipe, gradient_sr=False)
    x = torch.empty((0, 17), dtype=torch.bfloat16, requires_grad=True)
    weights = torch.zeros((3, 19, 17), dtype=torch.bfloat16, requires_grad=True)
    recorder = MatmulDtypes()
    with recorder:
        out = A.grouped_linear(x, weights, [0, 0, 0], cfg)
        out.backward(torch.empty_like(out))
    assert out.shape == (0, 19) and out.dtype == torch.bfloat16
    assert x.grad.shape == x.shape and torch.count_nonzero(weights.grad) == 0
    assert recorder.calls == []


class StatefulLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((16, 16), dtype=torch.bfloat16))
        self.fp4_config, self.fp4_configs, self.fp4_rngs = cfg, (cfg,), {}


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_legacy_serialized_identity_and_sr_keys_survive_new_mode(recipe):
    # Literal pre-feature schema; this would fail if a default mode field leaked
    # into existing checkpoint identities or private RNG stream names.
    prior = dict(
        recipe=recipe,
        block_size=16,
        weight_2d=True,
        outer_scope="tensor",
        inner_storage="float32_grid",
        rht_size=16,
        rht_randomized=False,
        gradient_sr=True,
        rng_seed=1234,
        quantizer="torch",
        version=1,
    )
    legacy = C.TwoLevelFP4Config(**prior)
    assert legacy.to_dict() == prior
    assert legacy.fingerprint == hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()
    aligned = config(recipe)
    assert aligned.to_dict()["rht_mode"] == "mxfp4"
    assert legacy.fingerprint != aligned.fingerprint
    assert legacy.rng_fingerprint == aligned.rng_fingerprint
    left, right = StatefulLayer(legacy), StatefulLayer(aligned)
    S.bind_fp4_model(left, rank=2)
    S.bind_fp4_model(right, rank=2)
    assert left.fp4_rng.stream_id == right.fp4_rng.stream_id
    assert left.fp4_rng.reserve("sr", 29) == right.fp4_rng.reserve("sr", 29)
    saved, before = copy.deepcopy(S.fp4_state_dict(left, rank=2)), copy.deepcopy(
        S.fp4_state_dict(right, rank=2)
    )
    with pytest.raises(ValueError, match="recipes"):
        S.load_fp4_state_dict(right, saved, rank=2)
    assert S.fp4_state_dict(right, rank=2) == before
    old_rng = C.FP4RandomState.for_config(legacy, "same").state_dict()
    with pytest.raises(ValueError):
        C.FP4RandomState.from_state_dict(old_rng, aligned, "same")


def triton_interpreter_check():
    """Check Triton FP32 math; characterize its CPU-only implicit BF16 cast.

    The pinned interpreter's cast_impl passes rounding_mode=None for implicit
    FP32->BF16 stores, so _convert_float truncates low 16 bits. This differs
    from Torch round-to-nearest-even and cannot qualify device BF16 stores.
    """
    # Triton rebuilds the interpreted function with module globals, so these
    # kernel dependencies must be globals rather than closure-only names.
    global shared, tl
    import triton
    import triton.language as tl
    from test_fp4_gpu_rng import philox_uniform

    T = importlib.import_module(C.__package__ + ".fp4_triton")
    shared = importlib.import_module(C.__package__ + ".fp4_mxfp4_shared_sr")

    @triton.jit
    def capture_shared_h16(X, Y):
        offsets = tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]
        output = shared._h16(tl.load(X + offsets))
        tl.store(Y + offsets, output)

    native_input = exponent_stress(32, 16).float().t().contiguous()
    native_output = torch.empty_like(native_input)
    capture_shared_h16[(1,)](native_input, native_output)
    assert torch.equal(
        native_output.t().contiguous().view(torch.int32),
        native_h16_oracle(native_input.t()).view(torch.int32),
    )
    value = torch.linspace(-0.91, 1.13, 17 * 19).reshape(17, 19).bfloat16()
    for recipe in ("nvfp4", "amdfp4"):
        cfg = config(recipe, quantizer="triton", sr_rng="triton_philox")
        transformed, _ = R.rht_pair(value, value, cfg)
        assert torch.equal(transformed.view(torch.int32), native_h16_oracle(value).view(torch.int32))
        assert torch.count_nonzero(transformed != transformed.bfloat16().float()) > 0
        # Bypass only the device-placement gate for the official CPU interpreter.
        # Real outer reduction, scale selection, Philox, pack and fused QDQ run.
        with patch.object(T, "_input", lambda x: None):
            for role in ("input", "grad_output"):
                operand = cfg.operand("wgrad", role)
                rng = C.FP4RandomState.for_config(cfg)
                rng.sr_offset = 2**32 + 19
                saved = copy.deepcopy(rng.state_dict())
                reservation = C.FP4RandomState.from_state_dict(saved, cfg).reserve("sr", 0)
                packed = T.quantize(transformed, cfg, operand, random=rng)
                clone = C.FP4RandomState.from_state_dict(saved, cfg)
                actual = T.qdq(transformed, cfg, operand, random=clone, dtype=torch.bfloat16)
                fp32_rng = C.FP4RandomState.from_state_dict(saved, cfg)
                actual_fp32 = T.qdq(transformed, cfg, operand, random=fp32_rng, dtype=torch.float32)
                oracle_cfg = replace(cfg, quantizer="torch", sr_rng="torch_cpu")
                oracle_rng = C.FP4RandomState.for_config(oracle_cfg)

                def uniforms(shape, **kwargs):
                    return torch.from_numpy(
                        philox_uniform(
                            reservation["seed"], reservation["offset"], int(np.prod(shape))
                        ).reshape(shape)
                    )

                with patch.object(torch, "rand", side_effect=uniforms):
                    expected = R.quantize(transformed, oracle_cfg, operand, random=oracle_rng)
                assert torch.equal(packed.data, expected.data)
                assert torch.equal(packed.inner_scale, expected.inner_scale)
                assert torch.equal(packed.outer_scale, expected.outer_scale)
                assert actual.dtype == torch.bfloat16
                expected_fp32 = R.dequantize(expected, dtype=torch.float32)
                assert torch.equal(actual_fp32.view(torch.int32), expected_fp32.view(torch.int32))
                truncated = (actual_fp32.view(torch.int32) >> 16).to(torch.int16)
                assert torch.equal(actual.view(torch.int16), truncated)
                assert torch.count_nonzero(actual != expected_fp32.bfloat16()) > 0
                assert clone.state_dict() == fp32_rng.state_dict() == rng.state_dict()
    print(
        "Native-order RHT Triton packed/scales/FP32 QDQ parity passed; "
        "CPU interpreter BF16 truncation characterized, device BF16 validation pending"
    )


def test_fp32_rht_triton_math_and_known_interpreter_bf16_cast():
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__)), "--triton-interpreter"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "packed/scales/FP32 QDQ parity passed" in result.stdout
    assert "device BF16 validation pending" in result.stdout


if __name__ == "__main__":
    assert sys.argv[1:] == ["--triton-interpreter"]
    triton_interpreter_check()
