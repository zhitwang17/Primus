# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU/Gloo verification of the actual Megatron shard-metadata adapter.

The adapter has Megatron's public buffer/range shapes, real CPU SGD and a real
collective parameter copy. Native Megatron GPU DDP remains a separate smoke.
Bucket offsets, inter-parameter padding and empty owners deliberately differ
from logical parameter coordinates so using a grad-buffer offset cannot pass.
"""

import importlib
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_primus_fp4_deosc_installer_distributed_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
sys.modules[PACKAGE] = package
C = importlib.import_module(PACKAGE + ".fp4_contract")
D = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc")
S = importlib.import_module(PACKAGE + ".optimizer.two_level_fp4_deosc_shard")


class Layer(torch.nn.Module):
    def __init__(self, recipe, rank, index, shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full(shape, 6.0, dtype=torch.bfloat16))
        with torch.no_grad():
            self.weight.flatten()[0] = 0.75390625
        self.fp4_config = C.TwoLevelFP4Config(recipe)
        self.fp4_stream_id = f"rank{rank}/model0/layers.{index}"
        self._primus_fp4_last_config = self.fp4_config


class BucketOptimizer:
    def __init__(self, model, rank, empty_rank, selected=(0, 1, 2)):
        self.model = model
        self.layers = [model.layers[index] for index in selected]
        self.config = SimpleNamespace(
            use_precision_aware_optimizer=True,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        )
        self.data_parallel_group = dist.group.WORLD
        self.copy_count = 0
        self.successful = True
        self.is_stub_optimizer = False
        self.model_param_gbuf_map = {}
        self.ranges = {}
        self.masters = {}
        # This is one bucket within a larger global grad buffer; its nonzero
        # offset must never be interpreted as an index in an individual weight.
        bucket_offset = 4096
        shard_size = 768 if empty_rank else 512
        shard_begin = bucket_offset + rank * shard_size
        shard_end = shard_begin + shard_size
        parameter_starts = (13, 355, 1350)
        param_index_map = {}
        owned_parameters, masters = [], []
        for index in selected:
            layer, relative_start = model.layers[index], parameter_starts[index]
            parameter = layer.weight
            world_start = bucket_offset + relative_start
            world_end = world_start + parameter.numel()
            param_index_map[parameter] = (world_start, world_end, 0)
            lo, hi = max(shard_begin, world_start), min(shard_end, world_end)
            if hi <= lo:
                continue
            start, end = lo - world_start, hi - world_start
            master = torch.nn.Parameter(parameter.detach().flatten()[start:end].float().clone())
            self.masters[parameter] = master
            owned_parameters.append(parameter)
            masters.append(master)
            self.model_param_gbuf_map[parameter] = (0, (torch.bfloat16, torch.float32), 0)
            self.ranges[parameter] = {
                "param": SimpleNamespace(start=start, end=end, size=end - start),
                "gbuf_world": SimpleNamespace(start=lo, end=hi, size=hi - lo),
                "gbuf_world_in_bucket": SimpleNamespace(start=lo - bucket_offset, end=hi - bucket_offset),
                "gbuf_local": SimpleNamespace(start=lo - shard_begin, end=hi - shard_begin),
            }
        self.buffers = [
            SimpleNamespace(param_index_map=param_index_map, data_parallel_group=dist.group.WORLD)
        ]
        self.model_float16_groups = [owned_parameters]
        self.shard_fp32_from_float16_groups = [masters]
        # An explicit empty parameter group is valid on a zero-owner rank.
        self.optimizer = torch.optim.SGD([{"params": masters}], lr=1 / 128)

    def _get_model_param_range_map(self, parameter):
        return self.ranges[parameter]

    @torch.no_grad()
    def _copy_main_params_to_model_params(self):
        self.copy_count += 1
        for layer in self.layers:
            parameter = layer.weight
            gathered = torch.zeros_like(parameter)
            if parameter in self.masters:
                interval = self.ranges[parameter]["param"]
                gathered.flatten()[interval.start : interval.end].copy_(self.masters[parameter])
            dist.all_reduce(gathered)
            parameter.copy_(gathered)

    def step_with_ready_grads(self):
        if not self.successful:
            return False
        for parameter, master in self.masters.items():
            interval = self.ranges[parameter]["param"]
            master.grad = parameter.grad.flatten()[interval.start : interval.end].float().clone()
        self.optimizer.step()
        self._copy_main_params_to_model_params()
        return True


def _installer_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        for recipe in ("nvfp4", "amdfp4"):
            for accumulator_dtype in ("fp32", "bf16"):
                for empty_rank in (False, True):
                    model = torch.nn.Module()
                    model.layers = torch.nn.ModuleList(
                        Layer(recipe, rank, i, shape)
                        for i, shape in enumerate(((17, 19), (3, 17, 19), (1, 16)))
                    )
                    optimizer = BucketOptimizer(model, rank, empty_rank)
                    args = SimpleNamespace(
                        weight_deosc=True,
                        weight_deosc_period=1,
                        weight_deosc_ratio=4,
                        weight_deosc_accumulator_dtype=accumulator_dtype,
                        data_parallel_size=3,
                        use_distributed_optimizer=True,
                    )
                    runner = D.install_megatron_deosc(optimizer, [model], args)[0]
                    assert isinstance(runner, S.ShardedTwoLevelDeOsc)
                    full = {
                        f"model0/layers.{i}/weight": (
                            layer.weight.detach().clone(),
                            layer.weight.detach().float().clone(),
                            layer.fp4_config,
                        )
                        for i, layer in enumerate(model.layers)
                    }
                    reference = D.TwoLevelDeOsc(full, period=1, ratio=4, accumulator_dtype=accumulator_dtype)
                    assert set(runner.bindings) == set(full)
                    if empty_rank and rank == 2:
                        assert not optimizer.masters
                        assert runner.stats["local_elements"] == 0

                    for update, coefficient in enumerate((1, -1, 1, -1)):
                        model.zero_grad(set_to_none=True)
                        optimizer.optimizer.zero_grad(set_to_none=True)
                        for _ in range(4):
                            sum(
                                layer.weight.flatten()[0].float() * coefficient / 4 for layer in model.layers
                            ).backward()
                            assert runner.step_count == update
                        if update == 1:
                            optimizer.successful = False
                            assert optimizer.step_with_ready_grads() is False
                            assert runner.step_count == update and optimizer.copy_count == update
                            optimizer.successful = True
                        for i, layer in enumerate(model.layers):
                            full[f"model0/layers.{i}/weight"][1].sub_(layer.weight.grad.float() / 128)
                        reference.step()
                        assert optimizer.step_with_ready_grads()
                        assert runner.step_count == update + 1
                        assert optimizer.copy_count == update + 1
                        local_state, full_state = runner.state_dict(), reference.state_dict()
                        for i, layer in enumerate(model.layers):
                            key = f"model0/layers.{i}/weight"
                            expected = full[key][1]
                            assert torch.equal(layer.weight, expected.bfloat16())
                            binding = runner.shard_bindings[key]
                            assert torch.equal(
                                binding.master, expected.flatten()[binding.start : binding.end]
                            )
                            for field in ("prev", "prev_q", "dist_w", "dist_q"):
                                assert torch.equal(
                                    local_state["states"][key][field],
                                    full_state["states"][key][field].flatten()[binding.start : binding.end],
                                )
                    assert reference.stats["snapped"] >= 3
                    dist.barrier()

        # Dense and expert optimizer leaves have distinct parameter membership;
        # some ranks own no master at all in a leaf but must join its QDQ calls.
        for recipe in ("nvfp4", "amdfp4"):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList(
                Layer(recipe, rank, i, shape) for i, shape in enumerate(((17, 19), (3, 17, 19), (1, 16)))
            )
            leaves = [BucketOptimizer(model, rank, False, selected=(index,)) for index in range(3)]
            chain = SimpleNamespace(chained_optimizers=leaves)
            args = SimpleNamespace(
                weight_deosc=True,
                weight_deosc_period=1,
                weight_deosc_ratio=4,
                weight_deosc_accumulator_dtype="bf16",
                data_parallel_size=3,
                use_distributed_optimizer=True,
            )
            runners = D.install_megatron_deosc(chain, [model], args)
            assert len(runners) == 3 and all(isinstance(runner, S.ShardedTwoLevelDeOsc) for runner in runners)
            assert [list(runner.bindings) for runner in runners] == [
                [f"model0/layers.{index}/weight"] for index in range(3)
            ]
            for coefficient in (1, -1):
                model.zero_grad(set_to_none=True)
                for leaf in leaves:
                    leaf.optimizer.zero_grad(set_to_none=True)
                sum(layer.weight.flatten()[0].float() * coefficient for layer in model.layers).backward()
                for leaf in leaves:
                    assert leaf.step_with_ready_grads()
            for layer in model.layers:
                assert layer.weight.flatten()[0].item() == 1.0
            snapped = torch.tensor(sum(runner.stats["snapped"] for runner in runners))
            dist.all_reduce(snapped)
            assert snapped.item() == 3

        # Bad metadata on only one owner must reject on every rank without
        # leaving its peers blocked inside the collective runner constructor.
        for corruption in ("missing_master", "shard_length", "membership"):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList(
                Layer("nvfp4", rank, i, shape) for i, shape in enumerate(((17, 19), (3, 17, 19), (1, 16)))
            )
            optimizer = BucketOptimizer(model, rank, False)
            if rank == 1:
                if corruption == "missing_master":
                    optimizer.model_float16_groups = [[]]
                    optimizer.shard_fp32_from_float16_groups = [[]]
                elif corruption == "shard_length":
                    optimizer.shard_fp32_from_float16_groups[0][0] = optimizer.shard_fp32_from_float16_groups[
                        0
                    ][0].detach()[:-1]
                else:
                    del optimizer.buffers[0].param_index_map[model.layers[0].weight]
            original_step = optimizer.step_with_ready_grads
            with pytest.raises(ValueError):
                D.install_megatron_deosc(optimizer, [model], args)
            assert optimizer.step_with_ready_grads == original_step
            assert not hasattr(optimizer, "_primus_two_level_deosc")
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_megatron_shard_coordinates_and_empty_owners_use_real_update_boundary(tmp_path):
    mp.spawn(_installer_worker, args=(str(tmp_path / "installer"),), nprocs=3, join=True)
