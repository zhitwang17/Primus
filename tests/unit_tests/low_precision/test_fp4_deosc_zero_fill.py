# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU/Gloo checks for the intentionally approximate DeOsc boundary policy.

The oracle masks nonowned values in the original whole tensor, uses the real
whole-tensor outer, and invokes the independently tested complete QDQ. It does
not reproduce the production shard slab/stripe construction.
"""

import copy
from dataclasses import replace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_fp4_deosc_distributed import C, D, S, _delta, _init, _inputs, _tree_equal

R = __import__(C.__package__ + ".fp4_reference", fromlist=["qdq"])


def _masked_reference(master, start, end, cfg):
    current = master.bfloat16()
    outer = R.compute_outer_scale(current, cfg)
    local = torch.zeros_like(current)
    local.flatten()[start:end].copy_(current.flatten()[start:end])
    return R.qdq(local, cfg, cfg.operand("fwd", "weight"), outer_scale=outer).flatten()[start:end]


def _cases(rank):
    # Nonaligned widths, cuts within a block, a shard spanning multiple experts,
    # and a rank that owns none of a complete logical parameter.
    for shape in ((17, 19), (3, 17, 19)):
        count = torch.tensor(shape).prod().item()
        for empty in (False, True):
            partitions = (
                (0, 7),
                (7, count if empty else count - 23),
                (0, 0) if empty else (count - 23, count),
            )
            yield shape, partitions[rank], False
    # Both tile-row aligned dense cuts and whole-expert cuts are exact even
    # under zero_fill, since no inner quantization block loses values.
    for shape, stride in (((48, 32), 16 * 32), ((3, 17, 19), 17 * 19)):
        yield shape, (rank * stride, (rank + 1) * stride), True


def _weight(shape):
    master = torch.full(shape, 6.0)
    master.flatten()[:7] = torch.arange(1, 8).float() / 8
    master.flatten()[-1] = 48.0
    return master


def _quantization_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        for recipe in ("nvfp4", "amdfp4"):
            for outer_scope in ("tensor", "none"):
                for tiled in (False, True):
                    cfg = C.TwoLevelFP4Config(recipe, outer_scope=outer_scope, weight_2d=tiled)
                    differences = 0
                    for shape, (start, end), aligned in _cases(rank):
                        master = _weight(shape)
                        # A deliberately stale model must never supply boundary
                        # values or the outer statistic at the optimizer boundary.
                        model = torch.full(shape, float("nan"), dtype=torch.bfloat16)
                        binding = S.FP4ShardBinding(
                            model, master.flatten()[start:end].clone(), cfg, start, end
                        )
                        zero = S.ShardedTwoLevelDeOsc({"weight": binding}, process_group=dist.group.WORLD)
                        exact = S.ShardedTwoLevelDeOsc(
                            {"weight": binding}, process_group=dist.group.WORLD, boundary_mode="exact"
                        )
                        calls = []
                        all_reduce = dist.all_reduce

                        def traced(tensor, op=dist.ReduceOp.SUM, **kwargs):
                            calls.append((op, tensor.dtype, tuple(tensor.shape), tensor.clone()))
                            return all_reduce(tensor, op=op, **kwargs)

                        with mock.patch.object(dist, "all_reduce", side_effect=traced):
                            zero.step()
                        assert zero.boundary_mode == "zero_fill"
                        assert all(plan.stripes == () for plan in zero._plans.values())
                        assert len(calls) == 2  # active/window agreement + outer/finite summary
                        assert all(op == dist.ReduceOp.MAX for op, _, _, _ in calls)
                        assert calls[1][1:3] == (torch.float32, (2,))
                        if outer_scope == "none":
                            assert calls[1][3][0] == 0  # no hidden per-shard amax policy
                        else:
                            expected_amax = (
                                binding.master.bfloat16().abs().amax().float() if end > start else 0
                            )
                            assert calls[1][3][0] == expected_amax
                        actual = zero.state_dict()["states"]["weight"]["prev_q"]
                        assert torch.equal(actual, _masked_reference(master, start, end, cfg))
                        calls.clear()
                        with mock.patch.object(dist, "all_reduce", side_effect=traced):
                            exact.step()
                        full_q = R.qdq(master.bfloat16(), cfg, cfg.operand("fwd", "weight")).flatten()[
                            start:end
                        ]
                        assert torch.equal(exact.state_dict()["states"]["weight"]["prev_q"], full_q)
                        if aligned:
                            assert torch.equal(actual, full_q)
                            assert not any(op == dist.ReduceOp.SUM for op, _, _, _ in calls)
                        else:
                            assert any(op == dist.ReduceOp.SUM for op, _, _, _ in calls)
                            differences += int(not torch.equal(actual, full_q))
                        assert torch.equal(binding.master, master.flatten()[start:end])
                        assert torch.isnan(model).all()
                    difference_count = torch.tensor(differences)
                    dist.all_reduce(difference_count)
                    assert difference_count.item() > 0  # approximation must be explicit, not an exact alias
    finally:
        dist.destroy_process_group()


def _continuation_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        for recipe in ("nvfp4", "amdfp4"):
            for outer_scope in ("tensor", "none"):
                for tiled in (False, True):
                    cfg = C.TwoLevelFP4Config(recipe, outer_scope=outer_scope, weight_2d=tiled)
                    full, bindings, ranges = _inputs(rank, True, cfg)
                    options = dict(period=3, ratio=4, boundary_mode="zero_fill")
                    runner = S.ShardedTwoLevelDeOsc(bindings, process_group=dist.group.WORLD, **options)
                    rng = torch.random.get_rng_state().clone()
                    for update in range(8):
                        for key, binding in bindings.items():
                            start, end = ranges[key]
                            binding.master.add_(_delta(key, full[key][1], update).flatten()[start:end])
                        runner.step()
                        if update == 1:
                            saved = copy.deepcopy(runner.state_dict())
                            saved_bindings = {
                                key: replace(b, master=b.master.clone()) for key, b in bindings.items()
                            }
                    expected = runner.state_dict()
                    assert expected["version"] == 4
                    assert expected["settings"]["shard_boundary_mode"] == "zero_fill"
                    count = torch.tensor(runner.stats["snapped"])
                    dist.all_reduce(count)
                    assert count.item() > 0
                    restored = S.ShardedTwoLevelDeOsc(
                        saved_bindings, process_group=dist.group.WORLD, **options
                    )
                    restored.load_state_dict(saved)
                    for update in range(2, 8):
                        for key, binding in saved_bindings.items():
                            start, end = ranges[key]
                            binding.master.add_(_delta(key, full[key][1], update).flatten()[start:end])
                        restored.step()
                    _tree_equal(restored.state_dict(), expected)
                    for key in bindings:
                        assert torch.equal(saved_bindings[key].master, bindings[key].master)
                    assert torch.equal(torch.random.get_rng_state(), rng)

                    exact = S.ShardedTwoLevelDeOsc(
                        saved_bindings,
                        process_group=dist.group.WORLD,
                        period=3,
                        ratio=4,
                        boundary_mode="exact",
                    )
                    exact_state = exact.state_dict()
                    assert exact_state["version"] == 3
                    assert "shard_boundary_mode" not in exact_state["settings"]
                    exact.load_state_dict(exact_state)  # original v3 schema remains accepted
                    with pytest.raises(ValueError, match="identity mismatch"):
                        exact.load_state_dict(saved)
                    with pytest.raises(ValueError, match="identity mismatch"):
                        restored.load_state_dict(exact_state)
                    _tree_equal(restored.state_dict(), expected)
                    assert exact.step_count == 0
        # A policy disagreement or invalid mode must reject on all peers before
        # any participant enters a boundary exchange that another would skip.
        for invalid in (False, True):
            cfg = C.TwoLevelFP4Config("nvfp4")
            full, bindings, _ = _inputs(rank, True, cfg)
            mode = ("typo" if invalid else "exact") if rank == 2 else "zero_fill"
            with pytest.raises(ValueError):
                S.ShardedTwoLevelDeOsc(bindings, process_group=dist.group.WORLD, boundary_mode=mode)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("outer_scope", ["tensor", "none"])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("shape", [(17, 19), (3, 17, 19)])
def test_dp1_zero_fill_is_exact_full_parameter(recipe, outer_scope, tiled, shape):
    cfg = C.TwoLevelFP4Config(recipe, outer_scope=outer_scope, weight_2d=tiled)
    master = _weight(shape)
    binding = S.FP4ShardBinding(
        torch.zeros(shape, dtype=torch.bfloat16), master.flatten().clone(), cfg, 0, master.numel()
    )
    runner = S.ShardedTwoLevelDeOsc({"weight": binding})
    runner.step()
    expected = R.qdq(master.bfloat16(), cfg, cfg.operand("fwd", "weight")).flatten()
    assert torch.equal(runner.state_dict()["states"]["weight"]["prev_q"], expected)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_zero_fill_has_no_weight_collective_and_matches_masked_full_tensor(tmp_path):
    mp.spawn(_quantization_worker, args=(str(tmp_path / "quantization"),), nprocs=3, join=True)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_zero_fill_checkpoint_resume_and_collective_mode_validation(tmp_path):
    mp.spawn(_continuation_worker, args=(str(tmp_path / "continuation"),), nprocs=3, join=True)
