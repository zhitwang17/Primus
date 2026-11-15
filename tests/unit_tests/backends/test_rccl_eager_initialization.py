###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU coverage of group ordering and failure handling; no device qualification."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.distributed import rccl_sdma_param_gather as runtime
from primus.backends.megatron.patches.parallelism import rccl_sdma_param_all_gather_patches as patch


@dataclass(eq=False)
class Group:
    group_name: str
    ranks: tuple
    backend: str = "nccl"
    group_desc: str = "test"
    backend_config: str = "cpu:gloo,cuda:nccl"


@pytest.fixture
def fake_runtime(monkeypatch):
    world = Group("0", (0, 1, 2, 3))
    ce = Group("ce", world.ranks, "nccl-lazy")
    registry = {world: ("nccl", None)}
    calls = []
    dist = torch.distributed
    monkeypatch.setattr(patch, "_EAGER_RUNTIME_INITIALIZED", False)
    monkeypatch.setattr(patch, "_EAGER_INITIALIZED_GROUPS", set())
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(dist.distributed_c10d, "_world", SimpleNamespace(pg_map=registry))
    monkeypatch.setattr(
        dist,
        "get_rank",
        lambda group=None: 3 if group is None else (group.ranks.index(3) if 3 in group.ranks else -1),
    )
    monkeypatch.setattr(dist, "get_backend", lambda group: group.backend)
    monkeypatch.setattr(dist, "get_backend_config", lambda group: group.backend_config)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(group.ranks))
    monkeypatch.setattr(
        dist,
        "broadcast",
        lambda tensor, src, group: calls.append(
            ("broadcast", group.group_name, src, tensor.item(), tensor.dtype)
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device.index)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1234, 5678))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 200)
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: zeros(*args, **dict(kwargs, device="cpu")))

    def prepare(group, device):
        calls.append(("ce_pool", group.group_name, device.index))
        registry[ce] = (ce.backend, None)
        return ce, object()

    monkeypatch.setattr(runtime, "prepare_direct_param_buffer_pool", prepare)
    return SimpleNamespace(world=world, ce=ce, registry=registry, calls=calls, prepare=prepare)


def test_creation_order_singleton_and_global_broadcast_source(fake_runtime, capsys):
    env = fake_runtime
    # Lexical sorting would put 10 before 2; getter order could put TP before DP.
    dp = Group("2", (1, 3))
    tp = Group("10", (3,), "nccl-lazy", "TENSOR_MODEL_PARALLEL_GROUP")
    env.registry.update({dp: None, tp: None})
    assert patch.eager_initialize_runtime()
    assert env.calls == [
        ("broadcast", "0", 0, 0, torch.int32),
        ("sync", 0),
        ("broadcast", "2", 1, 0, torch.int32),
        ("sync", 0),
        ("broadcast", "10", 3, 0, torch.int32),
        ("sync", 0),
        ("ce_pool", "0", 0),
        ("sync", 0),
    ]
    output = capsys.readouterr().out
    assert "name=10 description='TENSOR_MODEL_PARALLEL_GROUP' backend=nccl-lazy ranks=[3]" in output
    assert "free_bytes=1234 total_bytes=5678 allocated_bytes=100 reserved_bytes=200" in output


def test_filters_nonmembers_cpu_groups_and_duplicates(fake_runtime):
    env = fake_runtime
    gloo = Group("gloo", (0, 3), "gloo", backend_config="cpu:gloo")
    nonmember = Group("other", (0, 1), "nccl")
    env.registry.update(
        {gloo: None, nonmember: None, None: None, torch.distributed.GroupMember.NON_GROUP_MEMBER: None}
    )
    assert patch._registered_nccl_groups() == [(env.world, "nccl", [0, 1, 2, 3])]


@pytest.mark.parametrize(
    "backend,config",
    [
        ("nccl", "cuda:nccl"),
        ("nccl-lazy", "cuda:nccl-lazy"),
        ("nccl2", "cuda:nccl2"),
        ("undefined", "cpu:gloo,cuda:nccl"),
    ],
)
def test_nccl_backend_variants(fake_runtime, backend, config):
    group = Group("tp", (3,), backend, backend_config=config)
    fake_runtime.registry[group] = None
    assert patch._registered_nccl_groups()[-1] == (group, backend, [3])


def test_idempotent_but_initializes_groups_added_after_first_call(fake_runtime):
    assert patch.eager_initialize_runtime()
    calls = list(fake_runtime.calls)
    assert patch.eager_initialize_runtime()
    assert fake_runtime.calls == calls
    late_group = Group("tp", (3,))
    fake_runtime.registry[late_group] = None
    assert patch.eager_initialize_runtime()
    assert fake_runtime.calls[len(calls) :] == [
        ("broadcast", "tp", 3, 0, torch.int32),
        ("sync", 0),
        ("ce_pool", "0", 0),
        ("sync", 0),
    ]


def test_not_initialized_does_not_touch_cuda_or_registry(fake_runtime, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("CUDA must not be touched"))
    assert not patch.eager_initialize_runtime()
    assert fake_runtime.calls == []
    assert not patch._EAGER_RUNTIME_INITIALIZED


@pytest.mark.parametrize("failure_point", ["broadcast", "synchronize"])
def test_failure_keeps_group_uninitialized_and_preserves_cause(fake_runtime, monkeypatch, failure_point):
    failure = RuntimeError("ncclCuMemAlloc: out of memory")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        torch.distributed if failure_point == "broadcast" else torch.cuda, failure_point, fail
    )
    with pytest.raises(RuntimeError, match="name=0.*ncclCuMemAlloc") as caught:
        patch.eager_initialize_runtime()
    assert caught.value.__cause__ is failure
    assert not patch._EAGER_RUNTIME_INITIALIZED
    assert not patch._EAGER_INITIALIZED_GROUPS
    assert not any(call[0] == "ce_pool" for call in fake_runtime.calls)


def test_successful_prefix_is_not_repeated_after_later_group_failure(fake_runtime, monkeypatch):
    group = Group("tp", (3,))
    fake_runtime.registry[group] = None
    original = torch.distributed.broadcast

    def fail_second(tensor, src, group):
        if group.group_name == "tp":
            raise RuntimeError("first TP allocation failed")
        return original(tensor, src, group)

    monkeypatch.setattr(torch.distributed, "broadcast", fail_second)
    with pytest.raises(RuntimeError, match="name=tp"):
        patch.eager_initialize_runtime()
    assert not patch._EAGER_RUNTIME_INITIALIZED
    assert patch._EAGER_INITIALIZED_GROUPS == {(fake_runtime.world, 0)}
    monkeypatch.setattr(torch.distributed, "broadcast", original)
    assert patch.eager_initialize_runtime()
    assert [call[1] for call in fake_runtime.calls if call[0] == "broadcast"] == ["0", "tp"]


def test_ce_pool_failure_does_not_mark_runtime_ready(fake_runtime, monkeypatch):
    def fail(*args):
        raise RuntimeError("CE pool failure")

    monkeypatch.setattr(runtime, "prepare_direct_param_buffer_pool", fail)
    with pytest.raises(RuntimeError, match="CE pool failure"):
        patch.eager_initialize_runtime()
    assert not patch._EAGER_RUNTIME_INITIALIZED
    assert (fake_runtime.ce, 0) not in patch._EAGER_INITIALIZED_GROUPS


def test_missing_registry_fails_without_claiming_success(fake_runtime, monkeypatch):
    monkeypatch.setattr(torch.distributed.distributed_c10d, "_world", SimpleNamespace())
    with pytest.raises(RuntimeError, match="process-group registry"):
        patch.eager_initialize_runtime()
    assert not patch._EAGER_RUNTIME_INITIALIZED
    assert fake_runtime.calls == []


def test_wrapper_waits_for_ordinary_initializer_and_preserves_return(monkeypatch):
    calls = []

    def initialize(*args, **kwargs):
        calls.append(("initialize", args, kwargs))
        return 42

    monkeypatch.setattr(patch, "eager_initialize_runtime", lambda: calls.append("eager") or True)
    wrapped = patch.make_eager_initialize_megatron(initialize)
    assert wrapped(7, option=True) == 42
    assert calls == [("initialize", (7,), {"option": True}), "eager"]


def test_wrapper_waits_for_deferred_initializer_and_preserves_return(monkeypatch):
    calls = []

    def deferred(*args, **kwargs):
        calls.append(("deferred", args, kwargs))
        return 42

    monkeypatch.setattr(patch, "eager_initialize_runtime", lambda: calls.append("eager") or True)
    wrapped = patch.make_eager_initialize_megatron(lambda: deferred)
    pending = wrapped()
    assert calls == []
    assert pending(7, option=True) == 42
    assert calls == [("deferred", (7,), {"option": True}), "eager"]


@pytest.mark.parametrize("deferred", [False, True])
def test_wrapper_rejects_missing_distributed_initialization(monkeypatch, deferred):
    monkeypatch.setattr(patch, "eager_initialize_runtime", lambda: False)
    wrapped = patch.make_eager_initialize_megatron(lambda: (lambda: None) if deferred else None)
    with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
        pending = wrapped()
        if deferred:
            pending()
