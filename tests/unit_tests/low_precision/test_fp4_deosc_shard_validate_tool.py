# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Execute the device validator's reporting/replay flow on CPU/Gloo only."""

import importlib
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
V = importlib.import_module("fp4_deosc_shard_validate")


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        contract, _ = V.load_contracts()
        reference = importlib.import_module(contract.__package__ + ".fp4_reference")
        shard = importlib.import_module(contract.__package__ + ".optimizer.two_level_fp4_deosc_shard")
        for recipe in ("nvfp4", "amdfp4"):
            for outer in ("tensor", "none"):
                for tiled in (False, True):
                    config = contract.TwoLevelFP4Config(recipe, outer_scope=outer, weight_2d=tiled)
                    for case in V.cases():
                        row = V.quantization_case(case, config, shard, reference, torch.device("cpu"))
                        assert row["status"] == "passed"
                        assert row["zero_fill_collective_bytes"] == 56
                row = V.checkpoint_case(config, shard, torch.device("cpu"))
                assert row["replay_exact"] and row["global_snapped_elements"] > 0
        # Validate that a local numerical failure is surfaced on both peers,
        # without sending the other rank into the next quantization collective.
        with pytest.raises(RuntimeError, match="rank1: deliberate local failure"):
            V.agree("deliberate local failure" if rank == 1 else None, "negative gate")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_validator_case_matrix_replay_and_failure_agreement_on_cpu(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "tool"),), nprocs=2, join=True)
