# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Two-GPU Triton/NCCL validation of sharded DeOsc, without model training.

Run with torchrun --standalone --nproc-per-node=2 and --output-dir PATH.
The oracle is CPU Torch QDQ of a whole tensor with nonowned values masked to
zero and an explicit whole-bank outer. Collective traces cover runner.step()
only; constructor metadata and this validator's result agreement are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
from fp4_preflight import load_contracts


def cases():
    # (name, full shape, cut between two rank-owned flat intervals, aligned)
    return (
        ("cross_tile", (17, 19), 7, False),
        ("cross_expert", (3, 17, 19), 330, False),
        ("empty_owner", (2, 17, 19), 646, True),
        ("aligned_dense", (32, 32), 512, True),
        ("aligned_experts", (2, 17, 19), 323, True),
    )


def weights(shape):
    result = torch.full(shape, 6.0, dtype=torch.float32)
    matrices = result.reshape(-1, shape[-2] * shape[-1])
    matrices[:, :7] = torch.arange(1, 8).float() / 8
    result.flatten()[-1] = 48.0  # Another owner's maximum sets the whole-bank outer.
    return result


def cpu_oracle(full_master, start, end, config, reference, *, masked):
    config = replace(config, quantizer="torch", sr_rng="torch_cpu", value_checks="strict")
    current = full_master.detach().cpu().bfloat16()
    outer = reference.compute_outer_scale(current, config)
    source = current
    if masked:
        source = torch.zeros_like(current)
        source.flatten()[start:end].copy_(current.flatten()[start:end])
    return reference.qdq(source, config, config.operand("fwd", "weight"), outer_scale=outer).flatten()[
        start:end
    ]


def equal(actual, expected, label):
    if isinstance(expected, torch.Tensor):
        actual = actual.detach().cpu()
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.equal(actual, expected)
        ):
            raise AssertionError(f"{label}: tensor shape/dtype/value mismatch")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{label}: dictionary keys differ")
        for key in expected:
            equal(actual[key], expected[key], f"{label}/{key}")
    elif actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def traced_step(runner):
    calls = []
    original = dist.all_reduce

    def traced(tensor, op=dist.ReduceOp.SUM, **kwargs):
        name = "MAX" if op == dist.ReduceOp.MAX else "SUM" if op == dist.ReduceOp.SUM else str(op)
        calls.append(
            {
                "op": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": tensor.numel() * tensor.element_size(),
            }
        )
        return original(tensor, op=op, **kwargs)

    with mock.patch.object(dist, "all_reduce", side_effect=traced):
        runner.step()
    return calls


def agree(error, label):
    """Every rank checks a completed phase before any enters the next phase."""
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        detail = "; ".join(f"rank{i}: {message}" for i, message in enumerate(errors) if message)
        raise RuntimeError(f"{label}: {detail}")


def quantization_case(case, config, shard_module, reference, device):
    name, shape, cut, aligned = case
    rank = dist.get_rank()
    start, end = (0, cut) if rank == 0 else (cut, math.prod(shape))
    full = weights(shape)
    model = torch.full(shape, float("nan"), device=device, dtype=torch.bfloat16)
    binding = shard_module.FP4ShardBinding(
        model, full.flatten()[start:end].to(device).clone(), config, start, end
    )
    zero = shard_module.ShardedTwoLevelDeOsc(
        {"weight": binding}, process_group=dist.group.WORLD, boundary_mode="zero_fill"
    )
    exact = shard_module.ShardedTwoLevelDeOsc(
        {"weight": binding}, process_group=dist.group.WORLD, boundary_mode="exact"
    )
    zero_calls = traced_step(zero)
    exact_calls = traced_step(exact)
    # Finish both distributed operations before any local oracle assertion.
    error = None
    row = {
        "case": name,
        "recipe": config.recipe,
        "outer_scope": config.outer_scope,
        "weight_2d": config.weight_2d,
        "shape": list(shape),
        "owned_interval": [start, end],
        "zero_fill_collectives": zero_calls,
        "exact_collectives": exact_calls,
    }
    try:
        actual = zero.state_dict()["states"]["weight"]["prev_q"]
        exact_q = exact.state_dict()["states"]["weight"]["prev_q"]
        equal(actual, cpu_oracle(full, start, end, config, reference, masked=True), "zero-fill CPU oracle")
        equal(exact_q, cpu_oracle(full, start, end, config, reference, masked=False), "exact CPU oracle")
        if len(zero_calls) != 2 or any(call["op"] != "MAX" for call in zero_calls):
            raise AssertionError("Zero-fill must use only active/window and outer/finite MAX collectives")
        if zero_calls[1]["shape"] != [2] or zero_calls[1]["dtype"] != "torch.float32":
            raise AssertionError("Outer/finite statistics must remain two FP32 values")
        if any(plan.stripes for plan in zero._plans.values()):
            raise AssertionError("Zero-fill must not prepare boundary stripes")
        if aligned:
            equal(actual, exact_q, "Aligned zero-fill/exact")
            if any(call["op"] == "SUM" for call in exact_calls):
                raise AssertionError("Aligned exact shards unexpectedly exchanged weights")
        elif not any(call["op"] == "SUM" for call in exact_calls):
            raise AssertionError("Cross-boundary exact mode did not exchange boundary weights")
        equal(binding.master, full.flatten()[start:end], "Master unchanged by seeding")
        if not bool(torch.isnan(model).all()):
            raise AssertionError("DeOsc changed the deliberately stale model parameter")
        row.update(
            status="passed",
            boundary_difference_elements=int((actual != exact_q).sum()),
            zero_fill_collective_bytes=sum(c["bytes"] for c in zero_calls),
            exact_collective_bytes=sum(c["bytes"] for c in exact_calls),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    agree(error, f"{config.recipe}/{config.outer_scope}/{config.weight_2d}/{name}")
    return row


def checkpoint_case(config, shard_module, device):
    rank = dist.get_rank()
    full = torch.full((2, 16, 16), 6.0)
    full.flatten()[0] = 0.74609375
    full.flatten()[-1] = 48.0
    start, end = (0, 13) if rank == 0 else (13, full.numel())
    binding = shard_module.FP4ShardBinding(
        full.bfloat16().to(device), full.flatten()[start:end].to(device).clone(), config, start, end
    )
    options = dict(process_group=dist.group.WORLD, boundary_mode="zero_fill", period=2, ratio=4)
    runner = shard_module.ShardedTwoLevelDeOsc({"weight": binding}, **options)
    calls = []

    def update(b, index):
        if b.start == 0:
            b.master[0] = 0.74609375 if index % 2 == 0 else 0.75390625
        if b.end == full.numel():
            b.master[-1] = 52.0 if index in (2, 3) else 48.0

    for index in range(8):
        update(binding, index)
        calls.extend(traced_step(runner))
        if index == 1:
            buffer = io.BytesIO()
            torch.save({"state": runner.state_dict(), "master": binding.master.detach().cpu()}, buffer)
    expected = runner.state_dict()
    expected_master = binding.master.detach().cpu().clone()
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    replay_binding = replace(binding, master=saved["master"].to(device))
    replay = shard_module.ShardedTwoLevelDeOsc({"weight": replay_binding}, **options)
    replay.load_state_dict(saved["state"])
    for index in range(2, 8):
        update(replay_binding, index)
        calls.extend(traced_step(replay))
    # No collective operation follows local assertions until agree().
    error = None
    try:
        equal(replay.state_dict(), expected, "Checkpoint replay state")
        equal(replay_binding.master, expected_master, "Checkpoint replay master")
        if expected["version"] != 4 or expected["settings"]["shard_boundary_mode"] != "zero_fill":
            raise AssertionError("Checkpoint does not identify the approximate boundary policy")
        if any(call["op"] != "MAX" for call in calls):
            raise AssertionError("A snap/replay step exchanged boundary weights")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    agree(error, f"{config.recipe}/{config.outer_scope}/checkpoint")
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, runner.stats["snapped"])
    if sum(counts) <= 0:
        raise AssertionError("Qualification requires an actual DeOsc snap")
    return {
        "recipe": config.recipe,
        "outer_scope": config.outer_scope,
        "status": "passed",
        "updates": 8,
        "saved_after_update": 2,
        "replayed_updates": 6,
        "local_snapped_elements": runner.stats["snapped"],
        "global_snapped_elements": sum(counts),
        "checkpoint_version": expected["version"],
        "replay_exact": True,
        "weight_sum_collectives": 0,
        "serialization": "torch.save/torch.load(weights_only=True)",
    }


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [
        "tools/fp4_deosc_shard_validate.py",
        *[
            f"primus/backends/megatron/core/{name}.py"
            for name in (
                "fp4_contract",
                "fp4_reference",
                "fp4_backend",
                "fp4_triton",
                "optimizer/two_level_fp4_deosc",
                "optimizer/two_level_fp4_deosc_shard",
            )
        ],
    ]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "rank": rank,
        "status": "failed",
        "cases": [],
        "checkpoints": [],
        "scope": "small-tensor DeOsc Triton/NCCL validation; not a training/convergence test",
        "source_sha256": source_hashes(),
    }
    exit_code = 1
    try:
        if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
            raise RuntimeError("Use torchrun --nproc-per-node=2 with two available GPUs")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        device = torch.device("cuda", local_rank)
        contract, _ = load_contracts()
        reference = importlib.import_module(contract.__package__ + ".fp4_reference")
        shard = importlib.import_module(contract.__package__ + ".optimizer.two_level_fp4_deosc_shard")
        report.update(
            world_size=dist.get_world_size(),
            backend=str(dist.get_backend()),
            device=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
            hip_version=torch.version.hip,
            quantizer="triton",
            sr_rng="triton_philox",
        )
        for recipe in ("nvfp4", "amdfp4"):
            for outer_scope in ("tensor", "none"):
                for tiled in (False, True):
                    config = contract.TwoLevelFP4Config(
                        recipe,
                        outer_scope=outer_scope,
                        weight_2d=tiled,
                        quantizer="triton",
                        sr_rng="triton_philox",
                    )
                    for case in cases():
                        report["cases"].append(quantization_case(case, config, shard, reference, device))
                report["checkpoints"].append(checkpoint_case(config, shard, device))
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, report["cases"])
        for recipe in ("nvfp4", "amdfp4"):
            for outer_scope in ("tensor", "none"):
                for tiled in (False, True):
                    differences = sum(
                        r["boundary_difference_elements"]
                        for items in rows
                        for r in items
                        if (r["recipe"], r["outer_scope"], r["weight_2d"]) == (recipe, outer_scope, tiled)
                    )
                    if differences <= 0:
                        raise AssertionError(
                            f"No boundary approximation observed for {recipe}/{outer_scope}/{tiled}"
                        )
        report["status"], exit_code = "passed", 0
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        write_json(args.output_dir / f"rank{rank}.json", report)
        if dist.is_initialized():
            try:
                reports = [None] * dist.get_world_size()
                dist.all_gather_object(reports, report)
                if rank == 0:
                    summary = {
                        "schema_version": 1,
                        "status": "passed" if all(r["status"] == "passed" for r in reports) else "failed",
                        "ranks": reports,
                    }
                    write_json(args.output_dir / "summary.json", summary)
                    print(
                        json.dumps(
                            {
                                "status": summary["status"],
                                "output_dir": str(args.output_dir),
                                "cases_per_rank": len(report["cases"]),
                                "checkpoints_per_rank": len(report["checkpoints"]),
                            }
                        ),
                        flush=True,
                    )
            finally:
                dist.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
