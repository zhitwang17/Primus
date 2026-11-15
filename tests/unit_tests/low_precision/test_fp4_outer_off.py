# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU numerics and real Gloo partitions for NVFP4/AMDFP4 outer ablation.

Run with --confcutdir=tests/unit_tests/low_precision. The scalar oracle below
implements the documented scale/E2M1 grids independently of either quantizer.
No GPU devices, framework initialization, or installed training jobs are used.
"""

import copy
import importlib
import math
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_primus_fp4_outer_off_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
sys.modules[PACKAGE] = package
C = importlib.import_module(PACKAGE + ".fp4_contract")
R = importlib.import_module(PACKAGE + ".fp4_reference")
A = importlib.import_module(PACKAGE + ".fp4_autograd")
D = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc")
S = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc_shard")
STATE = importlib.import_module(PACKAGE + ".fp4_state")

OPERANDS = (
    ("fwd", "input", -1, False),
    ("fwd", "weight", -1, True),
    ("dgrad", "grad_output", -1, False),
    ("dgrad", "weight", -2, True),
    ("wgrad", "input", 0, False),
    ("wgrad", "grad_output", 0, False),
)
SCALE_GRIDS = {
    recipe: tuple((code & 7) / 8 * 2.0 ** (1 - bias) if code < 8
                  else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - bias)
                  for code in range(last + 1))
    for recipe, bias, last in (("nvfp4", 7, 126), ("amdfp4", 15, 254))
}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def config(recipe="amdfp4", **kwargs):
    return C.TwoLevelFP4Config(
        recipe, outer_scope="none", gradient_sr=False, rht_size=0, **kwargs
    )


def nearest_even(value, grid):
    index = min(range(len(grid)), key=lambda i: (abs(value - grid[i]), i % 2))
    return grid[index]


def oracle(x, axis, tiled, outer=1.0, recipe="amdfp4"):
    """Scalar tile quantization; no production layout/grid/scaling helper calls."""
    moved = x.detach().float().movedim(axis, -1).contiguous()
    bank = moved.unsqueeze(0) if moved.ndim == 2 else moved
    result = torch.empty_like(bank)
    height = 16 if tiled else 1
    for expert in range(bank.shape[0]):
        for row in range(0, bank.shape[1], height):
            for col in range(0, bank.shape[2], 16):
                tile = bank[expert, row : row + height, col : col + 16]
                maximum, floor = (448.0, 2.0**-6) if recipe == "nvfp4" else (114688.0, 2.0**-17)
                raw = max(min(float(tile.abs().max()) / outer / 6, maximum), floor)
                scale = float(torch.tensor(nearest_even(raw, SCALE_GRIDS[recipe])) * outer)
                values = [math.copysign(nearest_even(abs(v) / scale, E2M1) * scale, v)
                          for v in tile.flatten().tolist()]
                result[expert, row : row + height, col : col + 16] = torch.tensor(values).reshape(tile.shape)
    result = result[0] if moved.ndim == 2 else result
    return result.movedim(-1, axis).contiguous().to(x.dtype)


@pytest.mark.parametrize("direction,role,axis,tiled", OPERANDS)
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_actual_quantized_operands_use_fp32_one_and_independent_oracle(direction, role, axis, tiled, recipe):
    cfg = config(recipe)
    x = (torch.sin(torch.arange(17 * 19).float()) * 7).reshape(17, 19).bfloat16()
    packed = R.quantize(x, cfg, cfg.operand(direction, role))
    assert packed.outer_scale.dtype == torch.float32
    assert packed.outer_scale.shape == (1,) and packed.outer_scale.item() == 1.0
    assert torch.equal(R.dequantize(packed, dtype=x.dtype), oracle(x, axis, tiled, recipe=recipe))


@pytest.mark.parametrize("shape", [(0, 17), (3, 0), (0, 0), (2, 0, 17), (3, 17)])
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_empty_and_zero_operands_keep_one_not_the_tensor_amax_floor(shape, recipe):
    cfg = config(recipe)
    x = torch.zeros(shape, dtype=torch.bfloat16)
    for role in ("input", "weight"):
        packed = R.quantize(x, cfg, cfg.operand("fwd", role))
        assert packed.outer_scale.dtype == torch.float32 and packed.outer_scale.tolist() == [1.0]
        assert torch.equal(R.dequantize(packed, dtype=x.dtype), x)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_expert_bank_weights_use_independent_16_by_16_scales_with_fixed_outer(recipe):
    cfg = config(recipe)
    bank = (torch.arange(3 * 17 * 19).reshape(3, 17, 19).float() % 37 - 19).bfloat16()
    bank[0] *= 0.03125
    bank[2] *= 16
    packed = R.quantize(bank, cfg, cfg.operand("fwd", "weight"))
    assert packed.outer_scale.tolist() == [1.0]
    assert packed.inner_scale.shape == (3, 2, 2)
    assert torch.equal(R.dequantize(packed, dtype=bank.dtype), oracle(bank, -1, True, recipe=recipe))
    for expert in range(3):
        separate = R.quantize(bank[expert], cfg, cfg.operand("fwd", "weight"))
        assert torch.equal(separate.data, packed.data[expert])
        assert torch.equal(separate.inner_scale, packed.inner_scale[expert])


@pytest.mark.parametrize("outer", [0.5, 2.0, 0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("shape", [(3, 17), (0, 17)])
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_none_rejects_explicit_nonunit_outer_even_for_empty_input(outer, shape, recipe):
    cfg = config(recipe)
    with pytest.raises(ValueError, match="outer"):
        R.quantize(torch.ones(shape), cfg, cfg.operand("fwd", "input"), outer_scale=torch.tensor([outer]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_fixed_one_does_not_bypass_nonfinite_input_validation(value, recipe):
    cfg = config(recipe)
    x = torch.tensor([[value, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        R.compute_outer_scale(x, cfg)
    with pytest.raises(ValueError, match="non-finite"):
        R.qdq(x, cfg, cfg.operand("fwd", "input"))


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_packed_metadata_cannot_restore_outer_off_with_a_nonunit_scale(recipe):
    cfg = config(recipe)
    packed = R.quantize(torch.ones(17, 19), cfg, cfg.operand("fwd", "weight"))
    with pytest.raises(ValueError, match="outer"):
        C.TwoLevelFP4Tensor.from_metadata(
            packed.data, packed.inner_scale, torch.tensor([2.0]), packed.metadata()
        )


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_tensor_outer_retains_dynamic_formula_and_existing_quantization(recipe):
    cfg = replace(config(recipe), outer_scope="tensor")
    x = torch.linspace(-0.9, 1.7, 17 * 19).reshape(17, 19)
    packed = R.quantize(x, cfg, cfg.operand("fwd", "weight"))
    expected_outer = (x.abs().amax() / (6 * (448 if recipe == "nvfp4" else 114688))).reshape(1)
    assert torch.equal(packed.outer_scale, expected_outer)
    torch.testing.assert_close(
        R.dequantize(packed), oracle(x, -1, True, float(expected_outer), recipe), rtol=1e-7, atol=1e-7
    )
    off_cfg = config(recipe)
    off = R.qdq(x, off_cfg, off_cfg.operand("fwd", "weight"))
    assert not torch.equal(off, R.dequantize(packed))


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_dense_and_grouped_autograd_execute_all_directions_with_one(grouped, recipe, monkeypatch):
    cfg = config(recipe)
    q = lambda x, axis, tiled: oracle(x, axis, tiled, recipe=recipe)
    lengths = (0, 1, 17)
    x = (torch.sin(torch.arange(18 * 17).float()) * 2).reshape(18, 17).bfloat16().requires_grad_()
    shape = (3, 19, 17) if grouped else (19, 17)
    weights = torch.cos(torch.arange(math.prod(shape)).float()).reshape(shape).bfloat16().requires_grad_()
    dy = torch.cos(torch.arange(18 * 19).float()).reshape(18, 19).bfloat16()
    captured = []
    original = R.quantize

    def quantize(*args, **kwargs):
        value = original(*args, **kwargs)
        operand = args[2]
        captured.append((operand.direction, operand.role, value.outer_scale.clone()))
        return value

    monkeypatch.setattr(R, "quantize", quantize)
    y = A.grouped_linear(x, weights, lengths, cfg) if grouped else A.linear(x, weights, cfg)
    y.backward(dy)
    assert {(d, r) for d, r, _ in captured} == {(d, r) for d, r, _, _ in OPERANDS}
    assert all(v.dtype == torch.float32 and v.tolist() == [1.0] for _, _, v in captured)
    if grouped:
        ys, dxs, dws = [], [], []
        for xp, dyp, weight in zip(x.split(lengths), dy.split(lengths), weights):
            ys.append(q(xp, -1, False) @ q(weight, -1, True).T)
            dxs.append(q(dyp, -1, False) @ q(weight, -2, True))
            dws.append(q(dyp, 0, False).T @ q(xp, 0, False))
        expected = torch.cat(ys), torch.cat(dxs), torch.stack(dws)
        assert weights.grad[0].count_nonzero() == 0
    else:
        expected = (q(x, -1, False) @ q(weights, -1, True).T,
                    q(dy, -1, False) @ q(weights, -2, True),
                    q(dy, 0, False).T @ q(x, 0, False))
    for actual, wanted in zip((y, x.grad, weights.grad), expected):
        torch.testing.assert_close(actual, wanted, rtol=0.01, atol=0.01)


class Layer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(16, 16))
        self.fp4_config = cfg
        self.fp4_configs = (cfg,)
        self.fp4_rngs = {}


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_mode_changes_identity_but_not_rng_numbers_or_rht_transform(recipe):
    off = C.TwoLevelFP4Config(recipe, outer_scope="none")
    on = replace(off, outer_scope="tensor")
    assert off.fingerprint != on.fingerprint
    assert off.rng_fingerprint == on.rng_fingerprint
    models = Layer(off), Layer(on)
    for model in models:
        STATE.bind_fp4_model(model, rank=2)
    assert models[0].fp4_rng.stream_id == models[1].fp4_rng.stream_id
    r_off, r_on = (C.FP4RandomState.for_config(c, "rank0/layer4") for c in (off, on))
    for stream in ("sr", "rht"):
        assert r_off.reserve(stream, 17) == r_on.reserve(stream, 17)
    x = torch.arange(17 * 7).reshape(17, 7).float()
    dy = torch.sin(x[:, :5])
    before = torch.random.get_rng_state().clone()
    a = R.rht_pair(x, dy, off, random=r_off)
    b = R.rht_pair(x, dy, on, random=r_on)
    assert all(torch.equal(left, right) for left, right in zip(a, b))
    assert torch.equal(before, torch.random.get_rng_state())
    assert (r_off.sr_offset, r_off.rht_offset) == (r_on.sr_offset, r_on.rht_offset)
    # Equal explicit scales isolate the actual stochastic-rounding samples
    # from the intended change in quantization scale.
    samples = torch.full((33, 16), 1.25)
    samples[:, -1] = 6
    sr_off = R.qdq(samples, off, off.operand("dgrad", "grad_output"),
                   outer_scale=torch.ones(1), random=r_off)
    sr_on = R.qdq(samples, on, on.operand("dgrad", "grad_output"),
                  outer_scale=torch.ones(1), random=r_on)
    assert torch.equal(sr_off, sr_on)
    assert r_off.sr_offset == r_on.sr_offset
    with pytest.raises(ValueError):
        C.FP4RandomState.from_state_dict(r_off.state_dict(), on, "rank0/layer4")


@pytest.mark.parametrize("source_mode,target_mode", [("none", "tensor"), ("tensor", "none")])
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_mode_mismatch_rejects_rng_and_deosc_checkpoint_without_mutation(source_mode, target_mode, recipe):
    source_cfg = replace(config(recipe), outer_scope=source_mode)
    target_cfg = replace(config(recipe), outer_scope=target_mode)
    source, target = Layer(source_cfg), Layer(target_cfg)
    STATE.bind_fp4_model(source)
    STATE.bind_fp4_model(target)
    before = STATE.fp4_state_dict(target)
    with pytest.raises(ValueError, match="recipes"):
        STATE.load_fp4_state_dict(target, STATE.fp4_state_dict(source))
    assert STATE.fp4_state_dict(target) == before
    model = torch.full((16, 16), 6.0, dtype=torch.bfloat16)
    source_runner = D.TwoLevelDeOsc({"w": (model, model.float().clone(), source_cfg)}, period=2)
    target_runner = D.TwoLevelDeOsc({"w": (model, model.float().clone(), target_cfg)}, period=2)
    source_runner.step()
    assert source_runner.state_dict()["bindings"]["w"]["outer_scope"] == (
        "none" if source_mode == "none" else "complete_parameter"
    )
    with pytest.raises(ValueError):
        target_runner.load_state_dict(source_runner.state_dict())
    assert target_runner.step_count == 0 and not target_runner.state_dict()["states"]


def _tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _tree_equal(a, b)
    else:
        assert actual == expected


def _partition_worker(rank, rendezvous, recipe):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=3, timeout=timedelta(seconds=45))
    try:
        for tiled in (False, True):
            cfg = config(recipe, weight_2d=tiled)
            specs = {"dense": ((17, 19), ((0, 7), (7, 255), (255, 323))),
                     "bank": ((2, 17, 19), ((0, 17), (17, 330), (330, 646))),
                     "empty_owner": ((1, 16), ((0, 9), (9, 16), (0, 0)))}
            full, shards = {}, {}
            for key, (shape, partitions) in specs.items():
                master = torch.full(shape, 6.0)
                master.flatten()[0] = 0.74609375
                model = master.bfloat16()
                full[key] = model, master, cfg
                start, end = partitions[rank]
                shards[key] = S.FP4ShardBinding(model.clone(), master.flatten()[start:end].clone(),
                                                cfg, start, end)
            ref = D.TwoLevelDeOsc(full, period=2, ratio=4)
            actual = S.ShardedTwoLevelDeOsc(shards, process_group=dist.group.WORLD, period=2, ratio=4)
            saved = saved_masters = None
            for step, first in enumerate((0.74609375, 0.75390625, 0.74609375, 0.75390625)):
                for key, (_, master, _) in full.items():
                    master.flatten()[0] = first
                    shard = shards[key]
                    if shard.start == 0 and shard.end:
                        shard.master[0] = first
                ref.step()
                actual.step()
                ref_state, state = ref.state_dict(), actual.state_dict()
                for key, (_, master, _) in full.items():
                    shard = shards[key]
                    assert torch.equal(shard.master, master.flatten()[shard.start:shard.end])
                    q = oracle(master.bfloat16(), -1, tiled, recipe=recipe)
                    assert torch.equal(state["states"][key]["prev_q"], q.flatten()[shard.start:shard.end])
                    for field in ("prev", "prev_q", "dist_w", "dist_q"):
                        assert torch.equal(state["states"][key][field],
                                           ref_state["states"][key][field].flatten()[shard.start:shard.end])
                if step == 1:
                    saved = copy.deepcopy(state)
                    saved_masters = {key: b.master.clone() for key, b in shards.items()}
            assert ref.stats["snapped"] > 0
            restored_bindings = {key: replace(b, master=saved_masters[key]) for key, b in shards.items()}
            restored = S.ShardedTwoLevelDeOsc(restored_bindings, process_group=dist.group.WORLD,
                                             period=2, ratio=4)
            restored.load_state_dict(saved)
            for first in (0.74609375, 0.75390625):
                for shard in restored_bindings.values():
                    if shard.start == 0 and shard.end:
                        shard.master[0] = first
                restored.step()
            _tree_equal(restored.state_dict(), actual.state_dict())
            for key in shards:
                assert torch.equal(restored_bindings[key].master, shards[key].master)
            wrong = {key: replace(b, config=replace(cfg, outer_scope="tensor"))
                     for key, b in restored_bindings.items()}
            wrong_runner = S.ShardedTwoLevelDeOsc(wrong, process_group=dist.group.WORLD, period=2, ratio=4)
            with pytest.raises(ValueError):
                wrong_runner.load_state_dict(saved)
            assert wrong_runner.step_count == 0
            if rank == 1:
                restored_bindings["bank"].master[0] = float("nan")
            with pytest.raises(ValueError, match="nonfinite"):
                restored.step()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required")
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_three_rank_deosc_boundary_partitions_empty_owner_and_checkpoint(tmp_path, recipe):
    mp.spawn(_partition_worker, args=(str(tmp_path / "rendezvous"), recipe), nprocs=3, join=True)
