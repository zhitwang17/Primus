# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""One-GPU exact packed/grid/QDQ/RNG parity for the opt-in Triton quantizer."""

import argparse
import importlib
import json
import traceback
from pathlib import Path

import torch
from fp4_preflight import load_contracts


def _equal(a, b, label):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        if a.shape == b.shape and a.numel():
            mismatch = int((a != b).sum())
            maximum = float((a.float() - b.float()).abs().max())
            raise AssertionError(f"{label}: {mismatch} mismatches, max abs {maximum}")
        raise AssertionError(f"{label}: shape/dtype/value mismatch")
    if a.is_floating_point() and not torch.equal(torch.signbit(a), torch.signbit(b)):
        raise AssertionError(f"{label}: sign mismatch (including signed zero)")


def run(quick=False, gemm=False):
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for Triton parity")
    C, _ = load_contracts()
    R = importlib.import_module(C.__package__ + ".fp4_reference")
    T = importlib.import_module(C.__package__ + ".fp4_triton")
    rows = []

    def check(x, cfg, operand, name, outer=None):
        state = C.FP4RandomState.for_config(cfg, "triton-parity")
        initial = state.state_dict()
        a = R.quantize(x, cfg, operand, outer_scale=outer, random=state)
        ref_state = state.state_dict()
        replay = C.FP4RandomState.from_state_dict(initial, cfg, "triton-parity")
        b = T.quantize(x, cfg, operand, outer_scale=outer, random=replay)
        for attr in ("data", "inner_scale", "outer_scale"):
            _equal(getattr(a, attr), getattr(b, attr), f"{name}/{attr}")
        if a.metadata() != b.metadata() or replay.state_dict() != ref_state:
            raise AssertionError(f"{name}: metadata/RNG reservation differs")
        expected = R.dequantize(a, dtype=x.dtype)
        _equal(expected, T.dequantize(b, dtype=x.dtype), f"{name}/decode")
        replay = C.FP4RandomState.from_state_dict(initial, cfg, "triton-parity")
        direct = T.qdq(x, cfg, operand, outer_scale=outer, random=replay)
        _equal(expected, direct, f"{name}/direct-qdq")
        if replay.state_dict() != ref_state:
            raise AssertionError(f"{name}: direct QDQ RNG reservation differs")
        rows.append({"case": name, "shape": list(x.shape), "dtype": str(x.dtype), "status": "passed"})

    roles = [
        ("fwd", "input"),
        ("fwd", "weight"),
        ("dgrad", "weight"),
        ("dgrad", "grad_output"),
        ("wgrad", "input"),
        ("wgrad", "grad_output"),
    ]
    for recipe in ("nvfp4", "amdfp4"):
        cfg = C.TwoLevelFP4Config(recipe)
        for dtype in (torch.float32, torch.bfloat16):
            for shape in [(17, 33)] if quick else [(17, 33), (2, 17, 33)]:
                x = torch.sin(torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32) / 3)
                x = x.reshape(shape).to(device="cuda", dtype=dtype).transpose(-1, -2)
                for direction, role in roles:
                    check(
                        x, cfg, cfg.operand(direction, role), f"{recipe}/{dtype}/{shape}/{direction}/{role}"
                    )
        deterministic = C.TwoLevelFP4Config(recipe, gradient_sr=False, rht_size=0)
        one = torch.ones(1, device="cuda", dtype=torch.float32)
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
        )
        check(
            values.reshape(1, 16),
            deterministic,
            deterministic.operand("fwd", "input"),
            f"{recipe}/e2m1-codes",
            one,
        )
        grid = R._inner_grid(cfg, "cuda")
        midpoint = (grid[:-1] + grid[1:]) / 2
        candidates = torch.cat(
            [
                grid,
                midpoint,
                torch.nextafter(midpoint, torch.full_like(midpoint, float("inf"))),
                torch.nextafter(midpoint, torch.zeros_like(midpoint)),
            ]
        )
        x = (candidates * 6)[:, None].expand(-1, 16).contiguous()
        check(
            x, deterministic, deterministic.operand("fwd", "input"), f"{recipe}/inner-grid-and-midpoints", one
        )
        for shape in ((0, 17), (3, 0), (2, 0, 17)):
            x = torch.empty(shape, device="cuda")
            check(x, deterministic, deterministic.operand("fwd", "weight"), f"{recipe}/empty/{shape}")
        for bad in (float("nan"), float("inf")):
            x = torch.tensor([[bad]], device="cuda")
            for function in (T.compute_outer_scale,):
                try:
                    function(x, cfg)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Non-finite input was accepted")
        try:
            T.qdq(
                torch.zeros((1, 16), device="cuda"),
                deterministic,
                deterministic.operand("fwd", "input"),
                outer_scale=torch.tensor([1e-45], device="cuda"),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Underflowed combined scale was accepted")
    result = {
        "status": "passed",
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "cases": rows,
        "case_count": len(rows),
        "parity": "exact packed bytes, scales, metadata, decoded/fused values and RNG offsets",
    }
    if gemm:
        G = importlib.import_module(C.__package__ + ".fp4_scaled_gemm")
        gemm_rows = []
        shapes = [(17, 33, 19), (32, 64, 64)]
        if not quick:
            shapes.append((16, 5120, 2880))
        generator = torch.Generator(device="cpu").manual_seed(917)
        for recipe in ("nvfp4", "amdfp4"):
            for two_d in (False, True):
                cfg = C.TwoLevelFP4Config(recipe, weight_2d=two_d, rht_size=0, gradient_sr=False)
                for m, n, k in shapes:
                    a = torch.randn((m, k), generator=generator).cuda().bfloat16()
                    b = torch.randn((n, k), generator=generator).cuda().bfloat16()
                    aq = T.quantize(a, cfg, cfg.operand("fwd", "input"))
                    bq = T.quantize(b, cfg, cfg.operand("fwd", "weight"))
                    expected = R.dequantize(aq, torch.bfloat16) @ R.dequantize(bq, torch.bfloat16).t()
                    actual = G.scaled_gemm(aq, bq)
                    # Different BF16 GEMM reduction orders can differ by one output ULP.
                    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=2e-4)
                    error = actual.float() - expected.float()
                    gemm_rows.append(
                        {
                            "recipe": recipe,
                            "weight_2d": two_d,
                            "shape_m_n_k": [m, n, k],
                            "status": "passed",
                            "bit_equal": torch.equal(actual, expected),
                            "max_abs_error": float(error.abs().max()),
                            "relative_l2_error": float(error.norm() / expected.float().norm()),
                        }
                    )
        result["packed_scale_gemm"] = {
            "capability": G.capability_report(),
            "cases": gemm_rows,
            "comparison": "BF16 decoded operands + ordinary BF16 GEMM; rtol2^-7/atol2e-4",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--gemm", action="store_true", help="Also validate the experimental packed-scale GEMM"
    )
    args = parser.parse_args()
    result = {"status": "failed"}
    try:
        result = run(args.quick, args.gemm)
    except Exception:
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Triton FP4 parity {result['status']}: {args.output}")


if __name__ == "__main__":
    main()
