# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Single-GPU component timings for the two-level FP4 training reference.

Reports synchronized wall time and CUDA-event stream spans separately. Event
spans may contain host dispatch gaps in multi-launch Python operations; they
are not the sum of active kernel durations. No full-model TTT is inferred.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import platform
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Shape:
    name: str
    experts: int
    output_width: int
    input_width: int

    @property
    def weight_shape(self):
        if self.experts == 1:
            return (self.output_width, self.input_width)
        return (self.experts, self.output_width, self.input_width)

    def to_dict(self, rows):
        return {
            "name": self.name,
            "weight_shape": self.weight_shape,
            "input_shape": (rows * self.experts, self.input_width),
            "output_shape": (rows * self.experts, self.output_width),
            "rows_per_expert": rows,
            "experts": self.experts,
            "gemm_flops_per_direction": 2 * self.experts * rows * self.input_width * self.output_width,
            "gemm_implementation": "torch.mm" if self.experts == 1 else "per-expert torch.mm + cat/stack",
        }


def shapes_for(profile):
    if profile == "gpt-oss":
        return (
            Shape("qkv", 1, 5120, 2880),
            Shape("o", 1, 2880, 4096),
            Shape("moe_fc1", 32, 5760, 2880),
            Shape("moe_fc2", 32, 2880, 2880),
        )
    return (
        Shape("qkv", 1, 256, 128),
        Shape("o", 1, 128, 128),
        Shape("moe_fc1", 4, 256, 128),
        Shape("moe_fc2", 4, 128, 128),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("nvfp4", "amdfp4"), required=True)
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--shape-profile", choices=("tiny", "gpt-oss"), default="gpt-oss")
    parser.add_argument("--cases", nargs="+", choices=("qkv", "o", "moe_fc1", "moe_fc2"))
    parser.add_argument("--rows", type=int, default=128, help="Token rows per dense input or per expert")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--benchmarks", nargs="+", choices=("outer", "quant", "gemm", "deosc"))
    parser.add_argument("--directions", choices=("fwd", "all"), default="all")
    parser.add_argument(
        "--unchecked", action="store_true", help="Also time explicitly unchecked Triton calls"
    )
    parser.add_argument(
        "--packed-gemm",
        action="store_true",
        help="Also compare the experimental dense packed-scale BF16-dot consumer (not native FP4)",
    )
    parser.add_argument(
        "--describe", action="store_true", help="Print workload descriptions without Torch/GPU"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.rows <= 0 or args.rows % 128:
        parser.error("rows must be a positive multiple of128 (the grouped adapter's physical alignment)")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if args.unchecked and args.backend != "triton":
        parser.error("--unchecked is defined only for the explicit Triton backend")
    if args.packed_gemm and args.backend != "triton":
        parser.error("--packed-gemm requires --backend triton")
    if args.packed_gemm and args.benchmarks is not None and "gemm" not in args.benchmarks:
        parser.error("--packed-gemm requires gemm in --benchmarks")
    if not args.describe and args.output is None:
        parser.error("--output is required for GPU measurements")
    return args


def _load_modules(backend):
    # Use the same actual source-module loader as CPU preflight, avoiding a
    # full Primus/Megatron/TE startup in a low-level component benchmark.
    spec = importlib.util.spec_from_file_location("_fp4_profile_preflight", ROOT / "tools/fp4_preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    contract, _ = preflight.load_contracts()
    package = contract.__package__
    reference = importlib.import_module(package + ".fp4_reference")
    selected = reference if backend == "torch" else importlib.import_module(package + ".fp4_triton")
    deosc = importlib.import_module(package + ".optimizer.two_level_fp4_deosc")
    return contract, reference, selected, deosc


def summarize(samples):
    if not samples:
        raise ValueError("A timing summary requires at least one sample")
    return {
        "samples": list(samples),
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "max": max(samples),
    }


def measure(function, *, warmup, repeats, prepare=None):
    import torch

    for _ in range(warmup):
        if prepare is not None:
            prepare()
        output = function()
        torch.cuda.synchronize()
        del output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    event_samples, wall_samples = [], []
    for _ in range(repeats):
        if prepare is not None:
            prepare()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_begin = time.perf_counter()
        begin.record()
        output = function()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_begin) * 1000)
        event_samples.append(begin.elapsed_time(end))
        del output
    return {
        "cuda_event_ms": summarize(event_samples),
        "synchronized_wall_ms": summarize(wall_samples),
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _gemm(x, weight, shape, rows, direction="fwd", dy=None):
    import torch

    if shape.experts == 1:
        if direction == "fwd":
            return x @ weight.t()
        if direction == "dgrad":
            return dy @ weight
        return dy.t() @ x
    xs = x.split(rows)
    if direction == "fwd":
        return torch.cat([part @ w.t() for part, w in zip(xs, weight)])
    dys = dy.split(rows)
    if direction == "dgrad":
        return torch.cat([part @ w for part, w in zip(dys, weight)])
    return torch.stack([grad.t() @ part for grad, part in zip(dys, xs)])


def _assert_equal(a, b, description):
    import torch

    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        raise AssertionError(f"Triton/reference parity failed for {description}")


def profile_case(args, shape, contract, reference, selected, deosc):
    import torch

    fields = {"recipe": args.recipe}
    if "quantizer" in contract.TwoLevelFP4Config.__dataclass_fields__:
        fields["quantizer"] = args.backend
    elif args.backend == "triton" and (args.benchmarks is None or "deosc" in args.benchmarks):
        raise RuntimeError("Triton DeOsc profiling requires the config.quantizer routing implementation")
    config = contract.TwoLevelFP4Config(**fields)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weight = torch.randn(shape.weight_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(
        (shape.experts * args.rows, shape.input_width),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    operand = config.operand("fwd", "weight")
    input_operand = config.operand("fwd", "input")
    outer = reference.compute_outer_scale(weight, config)
    outer_x = reference.compute_outer_scale(x, config)
    result = {**shape.to_dict(args.rows), "recipe_config": config.to_dict(), "timings": {}}
    metrics = result["timings"]

    def timed(name, function, *, prepare=None):
        metrics[name] = measure(function, warmup=args.warmup, repeats=args.repeats, prepare=prepare)

    # Verify the full selected shape once, outside timed regions. Source code
    # and artifact hashes make the corresponding validation auditable.
    if args.backend == "triton":
        for name, tensor, policy, scale in (
            ("weight", weight, operand, outer),
            ("input", x, input_operand, outer_x),
        ):
            expected = reference.quantize(tensor, config, policy, outer_scale=scale)
            actual = selected.quantize(tensor, config, policy, outer_scale=scale)
            for key in ("data", "inner_scale", "outer_scale"):
                _assert_equal(getattr(expected, key), getattr(actual, key), name + "/" + key)
            _assert_equal(
                reference.dequantize(expected, dtype=tensor.dtype),
                selected.qdq(tensor, config, policy, outer_scale=scale),
                name + "/direct_qdq",
            )
            del expected, actual
        result["checked_triton_reference_parity"] = "exact packed bytes/scales and QDQ"

    benchmarks = args.benchmarks or ("outer", "quant", "gemm", "deosc")
    if "outer" in benchmarks:
        timed(
            "outer_amax_scale_torch_no_value_checks",
            lambda: (weight.float().abs().amax() / (6.0 * config.inner_max)).clamp_min(1e-30),
        )
        timed("outer_scale_selected_checked", lambda: selected.compute_outer_scale(weight, config))
        if args.unchecked:
            timed(
                "outer_scale_triton_unchecked",
                lambda: selected.compute_outer_scale(weight, config, check_values=False),
            )
    if "quant" in benchmarks:
        timed(
            "weight_quantize_fixed_outer_checked",
            lambda: selected.quantize(weight, config, operand, outer_scale=outer),
        )
        packed = selected.quantize(weight, config, operand, outer_scale=outer)
        timed("weight_dequantize_prepacked", lambda: selected.dequantize(packed, dtype=weight.dtype))
        del packed
        timed(
            "weight_qdq_fixed_outer_checked", lambda: selected.qdq(weight, config, operand, outer_scale=outer)
        )
        timed("weight_qdq_dynamic_outer_checked", lambda: selected.qdq(weight, config, operand))
        timed("input_qdq_dynamic_outer_checked", lambda: selected.qdq(x, config, input_operand))
        if args.directions == "all":
            timed(
                "weight_dgrad_qdq_dynamic_outer_checked",
                lambda: selected.qdq(weight, config, config.operand("dgrad", "weight")),
            )
        if args.unchecked:
            timed(
                "weight_qdq_fixed_outer_triton_unchecked",
                lambda: selected.qdq(weight, config, operand, outer_scale=outer, check_values=False),
            )
            timed(
                "weight_quantize_fixed_outer_triton_unchecked",
                lambda: selected.quantize(weight, config, operand, outer_scale=outer, check_values=False),
            )
    if "gemm" in benchmarks:
        xq = selected.qdq(x, config, input_operand)
        wq = selected.qdq(weight, config, operand)
        timed("bf16_gemm_fwd_predequantized", lambda: _gemm(xq, wq, shape, args.rows))
        if args.directions == "all":
            dy = torch.randn(
                (shape.experts * args.rows, shape.output_width),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            timed("bf16_gemm_dgrad_only", lambda: _gemm(xq, wq, shape, args.rows, "dgrad", dy))
            timed("bf16_gemm_wgrad_only", lambda: _gemm(xq, wq, shape, args.rows, "wgrad", dy))
            del dy
        del xq, wq

        def quantized_forward():
            qx = selected.qdq(x, config, input_operand)
            qw = selected.qdq(weight, config, operand)
            return _gemm(qx, qw, shape, args.rows)

        timed("dynamic_qdq_pair_plus_bf16_gemm_fwd", quantized_forward)
        if args.packed_gemm:
            if shape.experts != 1:
                result["packed_scale_gemm"] = {"status": "skipped", "reason": "dense forward only"}
            else:
                consumer = importlib.import_module(contract.__package__ + ".fp4_scaled_gemm")
                px = selected.quantize(x, config, input_operand)
                pw = selected.quantize(weight, config, operand)
                expected = (
                    selected.dequantize(px, dtype=x.dtype) @ selected.dequantize(pw, dtype=weight.dtype).t()
                )
                actual = consumer.scaled_gemm(px, pw)
                torch.testing.assert_close(actual, expected, rtol=2**-7, atol=2e-4)
                error = actual.float() - expected.float()
                result["packed_scale_gemm"] = {
                    "status": "passed",
                    "capability": consumer.capability_report(),
                    "max_abs_error": error.abs().max().item(),
                    "relative_l2_error": (error.norm() / expected.float().norm().clamp_min(1e-30)).item(),
                    "parity_tolerance": {"rtol": 2**-7, "atol": 2e-4},
                }
                del expected, actual, error
                timed("packed_scale_bf16_dot_fwd_prepacked", lambda: consumer.scaled_gemm(px, pw))
                del px, pw

                def packed_forward():
                    packed_x = selected.quantize(x, config, input_operand)
                    packed_w = selected.quantize(weight, config, operand)
                    return consumer.scaled_gemm(packed_x, packed_w)

                timed("dynamic_quant_pair_plus_packed_scale_bf16_dot_fwd", packed_forward)
    if "deosc" in benchmarks:
        master = weight.float()
        bindings = {"profile/weight": (weight, master, config)}
        runner = deosc.TwoLevelDeOsc(bindings, period=1_000_000)
        runner.step()  # Allocate/seed before measuring ordinary observation.
        timed("deosc_observe_dynamic_qdq", runner.step, prepare=lambda: master.add_(1e-3))
        result["deosc_observe_stats"] = runner.stats
        del runner
        runner = deosc.TwoLevelDeOsc(bindings, period=1)
        runner.step()  # Every subsequent step closes exactly one window.
        timed("deosc_close_dynamic_qdq", runner.step, prepare=lambda: master.add_(1e-3))
        result["deosc_close_stats"] = runner.stats
        del runner, bindings, master
    torch.cuda.synchronize()
    return result


def _source_hashes():
    names = (
        "tools/fp4_profile.py",
        "primus/backends/megatron/core/fp4_contract.py",
        "primus/backends/megatron/core/fp4_reference.py",
        "primus/backends/megatron/core/fp4_backend.py",
        "primus/backends/megatron/core/fp4_triton.py",
        "primus/backends/megatron/core/fp4_scaled_gemm.py",
        "primus/backends/megatron/core/optimizer/two_level_fp4_deosc.py",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in names
        if (ROOT / name).exists()
    }


def main():
    args = parse_args()
    shapes = [
        shape for shape in shapes_for(args.shape_profile) if args.cases is None or shape.name in args.cases
    ]
    if args.describe:
        print(json.dumps([shape.to_dict(args.rows) for shape in shapes], indent=2))
        return
    result = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": args.recipe,
        "backend": args.backend,
        "quantizer": args.backend,
        "training_gemm_backend": "reference_qdq_bf16",
        "shape_profile": args.shape_profile,
        "packed_scale_gemm_requested": args.packed_gemm,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "timing_notes": [
            "All values are isolated component measurements, not full-model throughput or TTT.",
            "CUDA events measure a stream span and can include CPU launch gaps between multiple kernels.",
            "Synchronized wall time additionally includes host-side validation, dispatch and final wait.",
            "Outer-amax, quantization, QDQ and DeOsc rows overlap semantically; do not sum every row.",
            "GEMM-only rows use already prepared BF16 inputs; they exclude gradient SR/RHT and quantization.",
            "DeOsc master perturbation is performed before timing; closure includes post-snap QDQ.",
            "Grouped inputs use equal aligned rows per expert to isolate component cost, not real routing skew.",
        ],
        "source_sha256": _source_hashes(),
        "cases": [],
    }
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one available GPU before running this profiler")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        result["environment"] = {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "device": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "total_memory_bytes": properties.total_memory,
        }
        modules = _load_modules(args.backend)
        with torch.no_grad():
            for shape in shapes:
                result["cases"].append(profile_case(args, shape, *modules))
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Profiled {args.recipe}/{args.backend}/{shape.name}", flush=True)
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
