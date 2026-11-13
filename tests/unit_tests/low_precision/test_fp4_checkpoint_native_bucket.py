# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU checks of the pinned native DistOpt checkpoint implementation.

Only GPU framework initialization is isolated: the mapping classes and selected
native optimizer method bodies are compiled directly from the checked-out MCore
source, without rewriting them. Real CPU AdamW supplies masters, moments and step
state. This checks descriptor construction and native restore, not GPU DCP I/O.
"""

import ast
import copy
import dataclasses
import importlib
import logging
import sys
import typing
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MCORE = ROOT / "third_party/Megatron-LM/megatron/core"


def _compile_nodes(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        declared = {getattr(node, "name", None)}
        if isinstance(node, ast.Assign):
            declared.update(target.id for target in node.targets if isinstance(target, ast.Name))
        if declared.intersection(names):
            selected.append(node)
    assert len(selected) == len(names), (path, names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def native(monkeypatch):
    module = ModuleType("_fp4_checkpoint_native_bucket")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    ns.update({key: value for key, value in vars(typing).items() if not key.startswith("__")})
    ns.update(
        ABC=ABC,
        abstractmethod=abstractmethod,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        replace=dataclasses.replace,
        chain=chain,
        torch=torch,
        logging=logging,
        logger=logging.getLogger(__name__),
        log_single_rank=lambda *args, **kwargs: None,
        HAVE_APEX_OR_TE=False,
        USING_APEX_OPTIMIZER=False,
        USING_TE_OPTIMIZER=False,
        HybridDeviceOptimizer=type("UnusedHybridDeviceOptimizer", (), {}),
    )
    _compile_nodes(MCORE / "dist_checkpointing/core.py", {"CheckpointingException"}, ns)
    _compile_nodes(
        MCORE / "dist_checkpointing/mapping.py",
        {"ShardedBase", "ShardedTensor", "ShardedObject", "LocalNonpersistentObject"},
        ns,
    )
    _compile_nodes(MCORE / "optimizer_param_scheduler.py", {"ParamGroupOverride"}, ns)
    ns["_ParamGroupOverride"] = ns["ParamGroupOverride"]
    _compile_nodes(
        MCORE / "optimizer/optimizer.py", {"_param_group_override_keys", "param_group_identifier_keys"}, ns
    )
    path = MCORE / "optimizer/distrib_optimizer.py"
    source_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "DistributedOptimizer"
    )
    methods = {
        "state_dict",
        "load_state_dict",
        "sharded_state_dict",
        "_get_main_param_and_optimizer_states",
        "_set_main_param_and_optimizer_states",
        "get_parameter_state_dp_reshardable",
        "sharded_param_state_dp_reshardable",
        "load_parameter_state_from_dp_reshardable",
    }
    selected = [
        node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    assert len(selected) == len(methods)
    selected += [
        node
        for node in source_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "checkpoint_fully_reshardable_formats"
    ]
    # Replace the GPU-only base; each tested method body remains exactly native.
    source_class.bases, source_class.keywords, source_class.body = [], [], selected
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            source_class,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return module


def _optimizer(native, rank, world):
    optimizer = native.DistributedOptimizer()
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fp16=False)
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.grad_scaler = None
    optimizer.distributed_optimizer_instance_id = 0
    optimizer.data_parallel_group_idx = 0
    optimizer.data_parallel_group = SimpleNamespace(rank=lambda: rank, size=lambda: world)
    # Global bucket: padding [0,2), A [2,9), padding [9,11), B [11,14), tail [14,16).
    # DP2 cuts parameter A at global offset 8. Both ranks have real ownership.
    local_size, local_start = 16 // world, rank * (16 // world)
    param_map, masters = {}, []
    optimizer.model_param_group_index_map = {}
    for start, end in ((2, 9), (11, 14)):
        lo, hi = max(start, local_start), min(end, local_start + local_size)
        if lo >= hi:
            continue
        model_param = torch.zeros(end - start, dtype=torch.bfloat16)
        master = torch.arange(lo, hi, dtype=torch.float32)
        optimizer.model_param_group_index_map[model_param] = (0, len(masters))
        masters.append(master)
        param_map[model_param] = {"gbuf_local": SimpleNamespace(start=lo - local_start, end=hi - local_start)}
    optimizer.optimizer = torch.optim.AdamW(masters, lr=0.03, weight_decay=0.02)
    for iteration in range(2):
        for master in masters:
            master.grad = torch.arange(master.numel(), dtype=torch.float32) / 10 + iteration + 1
        optimizer.optimizer.step()
    dtype = (torch.bfloat16, torch.float32)
    optimizer.gbuf_ranges = [{dtype: [{"param_map": param_map}]}]
    optimizer.buffers = [
        SimpleNamespace(buckets=[SimpleNamespace(numel_unpadded=14, grad_data=torch.empty(16))])
    ]
    optimizer.per_bucket_numel = [{dtype: [16]}]
    optimizer.per_bucket_numel_unpadded = [{dtype: [14]}]
    return optimizer


def _materialize(native, value):
    if isinstance(value, (native.ShardedTensor, native.ShardedObject)):
        return _materialize(native, value.data)
    if isinstance(value, native.LocalNonpersistentObject):
        return _materialize(native, value.unwrap())
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _materialize(native, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(native, item) for item in value]
    return copy.deepcopy(value)


def _tensors(native, value):
    if isinstance(value, native.ShardedTensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(native, item)
    elif isinstance(value, list):
        for item in value:
            yield from _tensors(native, item)


@pytest.mark.parametrize("world", [1, 2])
def test_native_bucket_master_adam_step_roundtrip_and_padding(native, world):
    coverage = {}
    for rank in range(world):
        optimizer = _optimizer(native, rank, world)
        saved = optimizer.sharded_state_dict({}, metadata={"distrib_optim_sharding_type": "dp_reshardable"})
        assert saved["param_state_sharding_type"] == "dp_reshardable"
        descriptors = list(_tensors(native, saved))
        assert {item.key.rsplit(".", 1)[-1] for item in descriptors} == {"param", "exp_avg", "exp_avg_sq"}
        for item in descriptors:
            assert item.flattened_range is None and item.dtype == torch.float32
            assert item.global_shape == (14,) and len(item.local_shape) == 1
            item.validate_metadata_integrity()
            counts = coverage.setdefault(item.key, torch.zeros(14, dtype=torch.int32))
            counts[item.global_offset[0] : item.global_offset[0] + item.local_shape[0]] += 1
        materialized = _materialize(native, saved)
        before = [
            (param.clone(), {key: value.clone() for key, value in optimizer.optimizer.state[param].items()})
            for param in optimizer.optimizer.param_groups[0]["params"]
        ]
        with torch.no_grad():
            for param in optimizer.optimizer.param_groups[0]["params"]:
                param.fill_(-71)
                for value in optimizer.optimizer.state[param].values():
                    value.fill_(-39)
            optimizer.load_state_dict(materialized)
        for param, (expected_master, expected_state) in zip(
            optimizer.optimizer.param_groups[0]["params"], before
        ):
            assert torch.equal(param, expected_master)
            assert expected_state.keys() == optimizer.optimizer.state[param].keys()
            for key, value in expected_state.items():
                assert torch.equal(optimizer.optimizer.state[param][key], value), key
    assert len(coverage) == 3
    assert all(torch.equal(counts, torch.ones_like(counts)) for counts in coverage.values())


def test_native_bucket_rejects_changed_unpadded_layout(native):
    optimizer = _optimizer(native, 0, 2)
    state = optimizer.get_parameter_state_dp_reshardable()
    state["per_bucket_numel_unpadded"] = [{(torch.bfloat16, torch.float32): [15]}]
    with pytest.raises(AssertionError, match="unpadded elements"):
        optimizer.load_parameter_state_from_dp_reshardable(state)


def test_removed_bucket_alias_and_flattened_descriptors_are_rejected(native):
    optimizer = _optimizer(native, 0, 1)
    with pytest.raises(NotImplementedError, match="Unknown sharding_type"):
        optimizer.sharded_state_dict(
            {}, metadata={"distrib_optim_sharding_type": "fully_sharded_bucket_space"}
        )
    with pytest.raises(native.CheckpointingException, match="flattened_range is not supported"):
        native.ShardedTensor(
            "broken", torch.ones(2), torch.float32, (2,), (2,), (0,), None, flattened_range=slice(0, 2)
        )


@pytest.mark.parametrize("is_loading", [False, True])
def test_fp4_helper_uses_native_bucket_path_on_save_and_load(native, monkeypatch, is_loading):
    package_name = "_fp4_native_bucket_checkpoint_core"
    package = ModuleType(package_name)
    package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
    monkeypatch.setitem(sys.modules, package_name, package)
    helper = importlib.import_module(package_name + ".fp4_checkpoint")
    # This helper needs only ShardedObject from the externally initialized framework.
    mapping = ModuleType("megatron.core.dist_checkpointing.mapping")
    mapping.ShardedObject = native.ShardedObject
    monkeypatch.setitem(sys.modules, "megatron.core.dist_checkpointing.mapping", mapping)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    model = torch.nn.Linear(2, 2)
    model.sharded_state_dict = lambda: {}
    optimizer = _optimizer(native, 0, 1)
    _, state = helper._state(model, optimizer, 2, {"rng": "rank-local"}, is_loading=is_loading)
    assert state["optimizer"]["param_state_sharding_type"] == "dp_reshardable"
    assert all(item.flattened_range is None for item in _tensors(native, state["optimizer"]))
    assert len(list(_tensors(native, state["optimizer"]))) >= 6
