# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Real CPU/Gloo equivalence tests for optimizer-sharded two-level FP4 DeOsc.

The oracle is the existing complete-parameter runner, whose QDQ has independent
scalar/codebook tests. No sharded quantizer or communication logic is duplicated
here. Every rank compares its master and observation-window slices with that
complete-parameter result after each update, including post-snap scale changes.
"""

import copy
import importlib
import io
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_primus_fp4_deosc_distributed_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
sys.modules[PACKAGE] = package
C = importlib.import_module(PACKAGE + ".fp4_contract")
D = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc")
S = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc_shard")


def _init(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )


def _tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _tree_equal(a, e)
    else:
        assert actual == expected


def _inputs(rank, empty_rank, config):
    specs = [
        ("dense", (17, 19), ((0, 7), (7, 255), (255, 323))),
        ("bank", (3, 17, 19), ((0, 330), (330, 805), (805, 969))),
        ("small", (1, 16), ((0, 9), (9, 16), (0, 0))),
    ]
    reference, shards, ranges = {}, {}, {}
    for key, shape, intervals in specs:
        master = torch.full(shape, 6.0, dtype=torch.float32)
        if key == "bank":
            master[0].fill_(0.03125)
            master[1].fill_(1.0)
            master[2].fill_(48.0)
        # The small parameter guarantees actual bin flips in every recipe/layout.
        for index in {0, 6, 7, 15, 16, 254, 255, 322, 329, 330, 804, 805}:
            if index < master.numel() - 1:
                master.flatten()[index] = 0.74609375
        model = torch.nn.Parameter(master.bfloat16())
        reference[key] = (model, master, config)
        if empty_rank:
            cut = min(13, master.numel())
            intervals = ((0, cut), (cut, master.numel()), (0, 0))
        start, end = intervals[rank]
        ranges[key] = (start, end)
        shards[key] = S.FP4ShardBinding(
            model=model.detach().clone(),
            master=master.flatten()[start:end].clone(),
            config=config,
            start=start,
            end=end,
        )
    return reference, shards, ranges


def _delta(key, master, update):
    result = torch.zeros_like(master)
    if update == 0:
        return result
    sign = 1 if update % 2 else -1
    for index in {0, 6, 7, 15, 16, 254, 255, 322, 329, 330, 804, 805}:
        if index < master.numel() - 1:
            result.flatten()[index] = sign / 128
    if key == "bank":
        # The global outer scale is owned by another rank for most shards.
        # Change it during an open window and at a period boundary.
        result.flatten()[-1] = 4 if update in (2, 3) else -2
    return result


def _equivalent(reference, sharded, full_bindings, bindings, ranges, accumulator_dtype):
    expected, actual = reference.state_dict(), sharded.state_dict()
    assert reference.step_count == sharded.step_count
    assert actual["window_presence"] == expected["window_presence"]
    for key, (_, full_master, _) in full_bindings.items():
        start, end = ranges[key]
        assert torch.equal(bindings[key].master, full_master.flatten()[start:end]), key
        assert actual["states"][key]["window_steps"] == expected["states"][key]["window_steps"]
        for field in ("prev", "prev_q", "dist_w", "dist_q"):
            local = actual["states"][key][field]
            assert torch.equal(local, expected["states"][key][field].flatten()[start:end]), (key, field)
            if field.startswith("dist_"):
                assert local.dtype == (torch.bfloat16 if accumulator_dtype == "bf16" else torch.float32)
    snapped = torch.tensor(sharded.stats["snapped"], dtype=torch.int64)
    dist.all_reduce(snapped)
    assert snapped.item() == reference.stats["snapped"]


def _equivalence_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        for recipe in ("nvfp4", "amdfp4"):
            for tiled in (False, True):
                for accumulator_dtype in ("fp32", "bf16"):
                    for empty_rank in (False, True):
                        cfg = C.TwoLevelFP4Config(recipe, weight_2d=tiled)
                        full, bindings, ranges = _inputs(rank, empty_rank, cfg)
                        options = dict(period=3, ratio=4, accumulator_dtype=accumulator_dtype)
                        reference = D.TwoLevelDeOsc(full, **options)
                        sharded = S.ShardedTwoLevelDeOsc(bindings, process_group=dist.group.WORLD, **options)
                        original_models = {key: b.model.clone() for key, b in bindings.items()}
                        rng_before = torch.random.get_rng_state().clone()
                        checkpoint = io.BytesIO()
                        for update in range(8):
                            for key, (_, master, _) in full.items():
                                delta = _delta(key, master, update)
                                master.add_(delta)
                                start, end = ranges[key]
                                bindings[key].master.add_(delta.flatten()[start:end])
                            reference.step()
                            sharded.step()
                            _equivalent(reference, sharded, full, bindings, ranges, accumulator_dtype)
                            if update == 1:
                                torch.save(
                                    {
                                        "deosc": sharded.state_dict(),
                                        "masters": {key: b.master.clone() for key, b in bindings.items()},
                                    },
                                    checkpoint,
                                )
                        assert reference.stats["snapped"] > 0
                        assert torch.equal(torch.random.get_rng_state(), rng_before)
                        for key, b in bindings.items():
                            assert torch.equal(b.model, original_models[key])
                        expected = sharded.state_dict()
                        expected_masters = {key: b.master.clone() for key, b in bindings.items()}

                        checkpoint.seek(0)
                        saved = torch.load(checkpoint, weights_only=True)
                        restored_bindings = {
                            key: S.FP4ShardBinding(
                                model=b.model,
                                master=saved["masters"][key],
                                config=cfg,
                                start=b.start,
                                end=b.end,
                            )
                            for key, b in bindings.items()
                        }
                        restored = S.ShardedTwoLevelDeOsc(
                            restored_bindings, process_group=dist.group.WORLD, **options
                        )
                        restored.validate_state_dict(saved["deosc"])
                        assert restored.step_count == 0
                        restored.load_state_dict(saved["deosc"])
                        for update in range(2, 8):
                            for key, (_, master, _) in full.items():
                                start, end = ranges[key]
                                restored_bindings[key].master.add_(
                                    _delta(key, master, update).flatten()[start:end]
                                )
                            restored.step()
                        _tree_equal(restored.state_dict(), expected)
                        for key, b in restored_bindings.items():
                            assert torch.equal(b.master, expected_masters[key])

                        # A rank-local history must not silently load on another rank.
                        payload = [saved["deosc"] if rank == 0 else None]
                        dist.broadcast_object_list(payload, src=0)
                        if rank != 0:
                            before = restored.state_dict()
                            with pytest.raises(ValueError):
                                restored.load_state_dict(payload[0])
                            _tree_equal(restored.state_dict(), before)

                        # Local validation is strict and atomic, including empty owners.
                        for mutation in ("missing_window", "wrong_dtype", "negative_distance"):
                            invalid = copy.deepcopy(restored.state_dict())
                            if mutation == "missing_window":
                                del invalid["states"]["small"]
                            elif mutation == "wrong_dtype":
                                invalid["states"]["small"]["prev"] = invalid["states"]["small"][
                                    "prev"
                                ].float()
                            else:
                                if not bindings["small"].master.numel():
                                    continue
                                invalid["states"]["small"]["dist_w"].fill_(-1)
                            with pytest.raises(ValueError):
                                restored.load_state_dict(invalid)
                            _tree_equal(restored.state_dict(), expected)
                        dist.barrier()
    finally:
        dist.destroy_process_group()


def _invalid_partition_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        for corruption in ("overlap", "hole", "shape", "recipe", "keys", "master_dtype"):
            cfg = C.TwoLevelFP4Config("amdfp4" if corruption == "recipe" and rank == 2 else "nvfp4")
            model = torch.full((1, 16), 6.0, dtype=torch.bfloat16)
            if corruption == "shape" and rank == 2:
                model = model.reshape(2, 8)
            intervals = [(0, 8), (8, 12), (12, 16)]
            if corruption == "overlap":
                intervals[1] = (7, 12)
            elif corruption == "hole":
                intervals[1] = (9, 12)
            start, end = intervals[rank]
            key = "different_weight" if corruption == "keys" and rank == 2 else "weight"
            master = model.flatten()[start:end].float()
            if corruption == "master_dtype" and rank == 1:
                master = master.bfloat16()
            bindings = {key: S.FP4ShardBinding(model, master, cfg, start, end)}
            with pytest.raises(ValueError):
                S.ShardedTwoLevelDeOsc(bindings, process_group=dist.group.WORLD, period=2)
            # All ranks must reject, and the process group must remain usable.
            dist.barrier()

        for mismatch in ("active", "recipe", "step", "window", "nonfinite"):
            cfg = C.TwoLevelFP4Config("nvfp4")
            model = torch.full((1, 16), 6.0, dtype=torch.bfloat16)
            start, end = [(0, 8), (8, 12), (12, 16)][rank]
            master = model.flatten()[start:end].float()
            owner = SimpleNamespace(_primus_fp4_last_config=cfg)
            runner = S.ShardedTwoLevelDeOsc(
                {"weight": S.FP4ShardBinding(model, master, cfg, start, end)},
                process_group=dist.group.WORLD,
                period=2,
                active_modules={"weight": owner},
            )
            runner.step()
            if rank == 0:
                if mismatch == "active":
                    owner._primus_fp4_last_config = None
                elif mismatch == "recipe":
                    owner._primus_fp4_last_config = C.TwoLevelFP4Config("amdfp4")
                elif mismatch in ("step", "window"):
                    # Each sidecar is structurally valid on its own; loading
                    # different update positions across ranks must fail before
                    # post-snap collectives can take different paths.
                    saved = runner.state_dict()
                    if mismatch == "step":
                        saved["step_count"] += 1
                    else:
                        saved["states"]["weight"]["window_steps"] = 1
                    runner.load_state_dict(saved)
                else:
                    master[0] = float("nan")
            with pytest.raises(ValueError):
                runner.step()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_shards_match_full_parameters_and_resume_across_real_gloo_ranks(tmp_path):
    mp.spawn(_equivalence_worker, args=(str(tmp_path / "equivalence"),), nprocs=3, join=True)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_invalid_partitions_are_rejected_collectively(tmp_path):
    mp.spawn(_invalid_partition_worker, args=(str(tmp_path / "invalid"),), nprocs=3, join=True)
