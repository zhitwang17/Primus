# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU checks for geometry, complete-state comparison and DP smoke assertions."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("_fp4_smoke_harness", ROOT / "tools/fp4_smoke.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_gpt_oss_profile_preserves_pinned_hidden_head_and_expert_dimensions():
    actual = SMOKE.model_geometry("gpt-oss")
    source = yaml.safe_load((ROOT / "primus/configs/models/megatron/gpt_oss_20B.yaml").read_text())
    for local, upstream in (
        ("hidden_size", "hidden_size"),
        ("attention_heads", "num_attention_heads"),
        ("query_groups", "num_query_groups"),
        ("kv_channels", "kv_channels"),
        ("num_experts", "num_experts"),
        ("topk", "moe_router_topk"),
        ("ffn_hidden_size", "moe_ffn_hidden_size"),
    ):
        assert actual[local] == source[upstream]
    assert actual["layers"] == 1 and actual["layer_pattern"] == [1]
    assert SMOKE.model_geometry("gpt-oss", 2)["layer_pattern"] == [1, 1]
    assert (actual["attention_heads"] + 2 * actual["query_groups"]) * actual["kv_channels"] == 5120


def test_profile_defaults_and_backward_compatible_tiny():
    base = ["--recipe", "nvfp4", "--output", "/tmp/result.json"]
    tiny = SMOKE.parse_args(base)
    assert (tiny.batch_size, tiny.seq_length, tiny.lr) == (2, 64, 0.002)
    assert tiny.gradient_accumulation_steps == 1 and tiny.deosc_accumulator_dtype == "fp32"
    assert SMOKE.model_geometry("tiny")["layer_pattern"] == [0, 1]
    full = SMOKE.parse_args(base + ["--model-profile", "gpt-oss", "--quantizer", "triton"])
    assert (full.batch_size, full.seq_length, full.lr, full.quantizer) == (1, 32, 5.7e-4, "triton")
    assert full.checkpoint_format == "bundle"


def test_grouped_deferred_validation_is_opt_in_and_requires_triton():
    base = ["--recipe", "nvfp4", "--output", "/tmp/result.json"]
    assert SMOKE.parse_args(base).value_checks == "strict"
    with pytest.raises(SystemExit):
        SMOKE.parse_args(base + ["--value-checks", "grouped_deferred"])
    options = SMOKE.parse_args(base + ["--quantizer", "triton", "--value-checks", "grouped_deferred"])
    assert options.value_checks == "grouped_deferred"


@pytest.mark.parametrize("extra", [["--batch-size", "0"], ["--vocab-size", "9"], ["--lr", "0"]])
def test_invalid_scale_options_rejected(extra):
    with pytest.raises(SystemExit):
        SMOKE.parse_args(["--recipe", "nvfp4", "--output", "/tmp/result.json", *extra])


def test_distributed_optimizer_requires_distributed_checkpoint_and_allows_accumulation():
    base = ["--recipe", "nvfp4", "--output", "/tmp/result.json", "--distributed-optimizer"]
    with pytest.raises(SystemExit):
        SMOKE.parse_args(base)
    options = SMOKE.parse_args(
        base
        + [
            "--checkpoint-format",
            "torch_dist",
            "--gradient-accumulation-steps",
            "4",
            "--deosc-accumulator-dtype",
            "bf16",
        ]
    )
    assert options.distributed_optimizer
    assert options.checkpoint_format == "torch_dist"
    assert options.gradient_accumulation_steps == 4
    assert options.deosc_accumulator_dtype == "bf16"


@pytest.mark.parametrize("count", ["0", "-1"])
def test_smoke_rejects_nonpositive_accumulation(count):
    with pytest.raises(SystemExit):
        SMOKE.parse_args(
            ["--recipe", "amdfp4", "--output", "/tmp/result.json", "--gradient-accumulation-steps", count]
        )


def test_complete_fingerprint_detects_unsampled_values_dtype_shape_and_metadata(monkeypatch):
    monkeypatch.setattr(SMOKE, "_CHECK_CHUNK_ELEMENTS", 4)
    tensor = torch.zeros(13)
    before = SMOKE._state_fingerprint({"weight": tensor, "routing": "nvfp4"})
    tensor[-1] = 1
    assert before != SMOKE._state_fingerprint({"weight": tensor, "routing": "nvfp4"})
    tensor[-1] = 0
    assert before == SMOKE._state_fingerprint({"routing": "nvfp4", "weight": tensor})
    assert before != SMOKE._state_fingerprint({"weight": tensor, "routing": "amdfp4"})
    assert SMOKE._state_fingerprint(tensor) != SMOKE._state_fingerprint(tensor.reshape(1, 13))
    assert SMOKE._state_fingerprint(tensor) != SMOKE._state_fingerprint(tensor.to(torch.bfloat16))
    matrix = torch.arange(12).reshape(3, 4)
    assert SMOKE._state_fingerprint(matrix.t()) == SMOKE._state_fingerprint(matrix.t().contiguous())


def test_distributed_optimizer_fingerprint_includes_master_and_moments_hidden_by_wrapper(monkeypatch):
    monkeypatch.setattr(SMOKE, "_CHECK_CHUNK_ELEMENTS", 4)
    master = torch.nn.Parameter(torch.zeros(13))
    adam = torch.optim.Adam([master], lr=0.01, foreach=False)
    SMOKE._initialize_adam_state(adam)

    class MetadataOnlyOptimizer:
        optimizer = adam

        def state_dict(self):
            return {"param_state_sharding_type": "dp_zero_gather_scatter", "step": 7}

        def get_parameters(self):
            return [master]

    optimizer = MetadataOnlyOptimizer()
    wrapper_before = SMOKE._state_fingerprint(optimizer.state_dict())
    complete_before = SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(optimizer))
    with torch.no_grad():
        master[-1] = 1.0
    assert SMOKE._state_fingerprint(optimizer.state_dict()) == wrapper_before
    assert SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(optimizer)) != complete_before
    with torch.no_grad():
        master[-1] = 0.0
    assert SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(optimizer)) == complete_before
    adam.state[master]["exp_avg"][-1] = 0.125
    assert SMOKE._state_fingerprint(optimizer.state_dict()) == wrapper_before
    assert SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(optimizer)) != complete_before
    adam.state[master]["exp_avg"][-1] = 0.0
    adam.state[master]["exp_avg_sq"][-1] = 0.0625
    assert SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(optimizer)) != complete_before


def test_adam_load_initialization_preserves_existing_moments():
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.Adam([parameter], lr=0.01, foreach=False)
    SMOKE._initialize_adam_state(optimizer)
    initial = optimizer.state[parameter]
    assert initial["step"].device.type == "cpu"
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = SMOKE._state_fingerprint(optimizer.state_dict())
    SMOKE._initialize_adam_state(optimizer)
    assert SMOKE._state_fingerprint(optimizer.state_dict()) == before
    assert optimizer.state[parameter]["step"].item() == 1


def test_replay_comparison_excludes_only_wall_time():
    a = [{"loss": 3.1, "global_grad_norm": 1.2, "elapsed_seconds_including_checks": 7.0}]
    b = [{"loss": 3.1, "global_grad_norm": 1.2, "elapsed_seconds_including_checks": 5.0}]
    assert SMOKE._comparable_steps(a) == SMOKE._comparable_steps(b)
    b[0]["global_grad_norm"] = 1.3
    assert SMOKE._comparable_steps(a) != SMOKE._comparable_steps(b)


def _dp_worker(rank, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        SMOKE._CHECK_CHUNK_ELEMENTS = 4
        tensor = torch.zeros((3, 5))
        if mode == "value" and rank == 1:
            tensor[-1, -1] = 1
        elif mode == "metadata" and rank == 1:
            tensor = tensor.reshape(5, 3)
        if mode == "same":
            result = SMOKE._assert_replicated([("all_zero_gradient", tensor)], "CPU test")
            assert result == {"status": "exact", "tensors": 1, "elements": 15}
        else:
            with pytest.raises(AssertionError, match="differs"):
                SMOKE._assert_replicated([("all_zero_gradient", tensor)], "CPU test")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["same", "value", "metadata"])
def test_real_two_process_complete_replica_check(tmp_path, mode):
    mp.spawn(_dp_worker, args=(str(tmp_path / (mode + ".rendezvous")), mode), nprocs=2, join=True)
