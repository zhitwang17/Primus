###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Megatron parameter AllGather directly through RCCL CE.

The replacement touches only ``_ParamAndGradBucketGroup.start_param_sync``.
Gradient ReduceScatter can independently opt into the same dedicated process
group with ``MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma``. Gradient-norm AllReduce
and other collectives retain their original process groups and algorithms.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from unittest import mock

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

BACKEND_ENV = "MEGATRON_PARAM_GATHER_BACKEND"
RCCL_SDMA_BACKEND = "rccl_sdma"
_EAGER_RUNTIME_INITIALIZED = False
_EAGER_INITIALIZED_GROUPS: set[tuple[object, int]] = set()

# Keep the real allocator available when another __init__ wrapper temporarily
# patches torch.zeros around this one. In particular, the direct-gradient
# wrapper must receive the constructor's second allocation, not this wrapper's
# internal param_data allocation.
_REAL_TORCH_ZEROS = torch.zeros


def rccl_sdma_param_gather_enabled(_ctx: PatchContext | None = None) -> bool:
    return os.getenv(BACKEND_ENV, "").strip().lower() == RCCL_SDMA_BACKEND


def validate_global_cta_policy() -> None:
    """Reject a process-wide policy that would override per-group settings."""
    global_policy = os.getenv("NCCL_CTA_POLICY")
    if global_policy:
        raise RuntimeError(
            "MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma requires "
            "NCCL_CTA_POLICY to be unset. A process-wide CTA policy is applied "
            "while every RCCL communicator is initialized and cannot be "
            "safely undone during the before_train patch phase."
        )


def make_start_param_sync(original):
    """Build the RCCL CE replacement for Megatron's parameter gather."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer
    from torch.distributed.distributed_c10d import _coalescing_manager

    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        get_sdma_process_group,
        is_direct_param_buffer,
    )

    def start_param_sync(self, force_sync: bool = False):
        if not self.ddp_config.use_distributed_optimizer:
            return original(self, force_sync=force_sync)
        if force_sync and self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            return
        if not force_sync:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        jobs = []
        for index, bucket in enumerate(self.buckets):
            if self.cached_param_buffer_shard_list[index] is None:
                self.cached_param_buffer_shard_list[index] = shard_buffer(
                    bucket.param_data,
                    self.intra_distributed_optimizer_instance_size,
                )
            local_data = self.cached_param_buffer_shard_list[index][
                self.intra_distributed_optimizer_instance_rank
            ]
            jobs.append((bucket.param_data, local_data))

        if jobs:
            group = get_sdma_process_group(self.intra_distributed_optimizer_instance_group)
            if not all(is_direct_param_buffer(output) for output, _input in jobs):
                raise RuntimeError(
                    "RCCL-SDMA requires every Megatron parameter bucket to use "
                    "the direct symmetric parameter buffer"
                )
            # Match Megatron's native parameter-gather stream semantics. RCCL's
            # synchronous null-stream fallback crashes on imported ROCr VMM
            # pointers, so even synchronous callers launch through the
            # ProcessGroupNCCL async stream and synchronize before returning.
            with _coalescing_manager(group, async_ops=True) as cm:
                for output, local_input in jobs:
                    torch.distributed.all_gather_into_tensor(
                        output,
                        local_input,
                        group=group,
                        async_op=True,
                    )
            if async_op:
                self.param_gather_handle = cm
            else:
                cm.wait()
                torch.cuda.current_stream(jobs[0][0].device).synchronize()
                self.param_gather_handle = None
        else:
            self.param_gather_handle = None
        self.param_gather_dispatched = True

    return start_param_sync


def make_param_and_grad_buffer_init(original):
    """Allocate eligible Megatron parameter buffers from the symmetric pool."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        ddp_config = bound.arguments["ddp_config"]
        if "params_with_names" in bound.arguments:  # Megatron-LM >= 0.19
            params = [param for param, _ in bound.arguments["params_with_names"]]
        else:
            params = bound.arguments["params"]
        original_group = bound.arguments["data_parallel_group"]
        nccl_ub = bound.arguments["nccl_ub"]

        # nccl_ub owns a separate registered pool. MXFP8 may alias parameter
        # and gradient storage; neither layout is supported by direct gather.
        from megatron.core.distributed.param_and_grad_buffer import is_mxfp8tensor

        if not ddp_config.use_distributed_optimizer:
            raise RuntimeError("RCCL-SDMA direct gather requires the distributed optimizer")
        if nccl_ub:
            raise RuntimeError("RCCL-SDMA direct gather does not support Megatron NCCL user buffers")
        if any(is_mxfp8tensor(param) for param in params):
            raise RuntimeError("RCCL-SDMA direct gather does not support MXFP8 parameter storage")

        from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
            mark_direct_param_buffer,
            prepare_direct_param_buffer_pool,
            recommended_eager_param_bytes,
            rendezvous_direct_param_buffer,
            take_direct_param_buffer,
        )

        device = params[0].device
        group, pool = prepare_direct_param_buffer_pool(original_group, device)
        original_zeros = torch.zeros
        allocation_thread = threading.get_ident()
        param_data_allocated = False

        def allocate_param_data(*zeros_args, **zeros_kwargs):
            nonlocal param_data_allocated
            if threading.get_ident() == allocation_thread and not param_data_allocated:
                param_data_allocated = True
                dtype = zeros_kwargs.get("dtype")
                if dtype is None:
                    raise RuntimeError("RCCL-SDMA direct param_data allocation requires an explicit dtype")
                eager_tensor = take_direct_param_buffer(
                    group,
                    device,
                    zeros_args[0],
                    dtype,
                )
                if eager_tensor is not None:
                    return eager_tensor
                numel = int(zeros_args[0])
                size_bytes = numel * torch.empty((), dtype=dtype).element_size()
                recommended_bytes = recommended_eager_param_bytes(size_bytes)
                try:
                    with torch.cuda.use_mem_pool(pool):
                        return _REAL_TORCH_ZEROS(*zeros_args, **zeros_kwargs)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "RCCL-SDMA could not allocate the direct symmetric parameter "
                        f"buffer ({size_bytes} bytes). Rerun with "
                        "MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES="
                        f"{recommended_bytes} to reserve it before model construction."
                    ) from exc
            return original_zeros(*zeros_args, **zeros_kwargs)

        # Megatron allocates param_data first and grad_data second in this
        # eligible non-MXFP8 constructor path. Only param_data participates in
        # direct AllGather, so keep grad_data on the normal allocator.
        with mock.patch.object(torch, "zeros", allocate_param_data):
            result = original(self, *args, **kwargs)

        if not param_data_allocated or self.param_data is None:
            raise RuntimeError("RCCL-SDMA direct gather did not allocate param_data")
        symmetric_memory = rendezvous_direct_param_buffer(self.param_data, group)
        # Keep the pool and rendezvous handle alive for the buffer lifetime.
        self._primus_rccl_sdma_pool = pool
        self._primus_rccl_sdma_symmetric_memory = symmetric_memory
        mark_direct_param_buffer(self.param_data)
        for bucket in self.buckets:
            if bucket.param_data is not None:
                mark_direct_param_buffer(bucket.param_data)
        return result

    return wrapped


def _registered_nccl_groups():
    """Return this rank's NCCL groups in global group-creation order.

    There is no public API to enumerate groups. The c10d registry preserves
    creation order; enumerating Megatron getters instead can reorder overlapping
    groups and omit singleton TP groups. Keep this private-API dependency here
    and use public APIs for backend and rank queries. Never sort by group name
    (numeric names sort lexically) or skip size-one groups: their first broadcast
    still initializes a communicator and allocates RCCL device memory.
    """
    import torch.distributed as dist

    registry = getattr(getattr(dist.distributed_c10d, "_world", None), "pg_map", None)
    if registry is None:
        raise RuntimeError("RCCL eager initialization requires the c10d process-group registry")
    seen = set()
    groups = []
    for group in (dist.group.WORLD, *tuple(registry)):
        if group is None or group == dist.GroupMember.NON_GROUP_MEMBER or id(group) in seen:
            continue
        seen.add(id(group))
        if dist.get_rank(group) < 0:
            continue
        backend = str(dist.get_backend(group)).lower()
        nccl_backends = {"nccl", "nccl-lazy", "nccl2"}
        if backend not in nccl_backends:
            # A default multi-device group can be reported as "undefined".
            config = str(dist.get_backend_config(group)).lower()
            device_backends = dict(item.split(":", 1) for item in config.split(",") if ":" in item)
            if device_backends.get("cuda") not in nccl_backends:
                continue
        ranks = dist.get_process_group_ranks(group)
        if not ranks:
            raise RuntimeError("RCCL eager initialization found an empty member process group")
        groups.append((group, backend, ranks))
    return groups


def _eager_memory_description(device):
    free, total = torch.cuda.mem_get_info(device)
    return (
        f"free_bytes={free} total_bytes={total} "
        f"allocated_bytes={torch.cuda.memory_allocated(device)} "
        f"reserved_bytes={torch.cuda.memory_reserved(device)}"
    )


def eager_initialize_runtime() -> bool:
    """Initialize existing NCCL groups and CE before model allocations.

    This changes initialization timing only. In particular, lazy TP1 broadcasts
    must not first allocate their RCCL cuMem pool after large model/optimizer
    buffers have been allocated. Synchronize each probe before entering the next group
    so overlapping communicators use the same creation order on every rank.
    """
    global _EAGER_RUNTIME_INITIALIZED

    import torch
    import torch.distributed as dist

    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        prepare_direct_param_buffer_pool,
    )

    if not dist.is_initialized():
        return False
    was_initialized = _EAGER_RUNTIME_INITIALIZED
    _EAGER_RUNTIME_INITIALIZED = False
    device = torch.device("cuda", torch.cuda.current_device())
    groups = _registered_nccl_groups()
    pending = [entry for entry in groups if (entry[0], device.index) not in _EAGER_INITIALIZED_GROUPS]
    if was_initialized and not pending:
        _EAGER_RUNTIME_INITIALIZED = True
        return True
    probe = torch.zeros(1, dtype=torch.int32, device=device)
    for group, backend, ranks in pending:
        description = (
            f"rank={dist.get_rank()} name={group.group_name} "
            f"description={getattr(group, 'group_desc', '')!r} backend={backend} ranks={ranks}"
        )
        print(
            f"[RCCL-SDMA:Megatron] eager communicator begin {description} {_eager_memory_description(device)}",
            flush=True,
        )
        try:
            dist.broadcast(probe, src=ranks[0], group=group)
            torch.cuda.synchronize(device)
        except Exception as exc:
            raise RuntimeError(
                f"RCCL eager communicator initialization failed: {description}: {exc}"
            ) from exc
        _EAGER_INITIALIZED_GROUPS.add((group, device.index))
        print(
            f"[RCCL-SDMA:Megatron] eager communicator complete {description} "
            f"{_eager_memory_description(device)}",
            flush=True,
        )
    try:
        group, _pool = prepare_direct_param_buffer_pool(dist.group.WORLD, device)
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"RCCL eager CE pool initialization failed: rank={dist.get_rank()} "
            f"parent_group={dist.group.WORLD.group_name}: {exc}"
        ) from exc
    _EAGER_INITIALIZED_GROUPS.add((group, device.index))
    _EAGER_RUNTIME_INITIALIZED = True
    return True


def make_eager_initialize_megatron(original_initialize_megatron):
    """Cover ordinary and deferred initialization, including newly added groups."""

    @functools.wraps(original_initialize_megatron)
    def wrapped(*args, **kwargs):
        result = original_initialize_megatron(*args, **kwargs)
        if callable(result):
            deferred_initialize = result

            @functools.wraps(deferred_initialize)
            def wrapped_deferred(*deferred_args, **deferred_kwargs):
                deferred_result = deferred_initialize(*deferred_args, **deferred_kwargs)
                if not eager_initialize_runtime():
                    raise RuntimeError(
                        "torch.distributed is not initialized after deferred Megatron initialization"
                    )
                return deferred_result

            return wrapped_deferred
        if not eager_initialize_runtime():
            raise RuntimeError("torch.distributed is not initialized after Megatron initialization")
        return result

    return wrapped


@register_patch(
    "megatron.distributed.rccl_sdma_param_all_gather",
    backend="megatron",
    phase="before_train",
    description=(
        "Route distributed-optimizer parameter AllGather through a dedicated "
        "zero-CTA RCCL copy-engine process group."
    ),
    condition=rccl_sdma_param_gather_enabled,
)
def patch_rccl_sdma_param_all_gather(ctx: PatchContext) -> None:
    del ctx

    if os.getenv("ENABLE_SDMA_ALLGATHER", "0") == "1":
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_param_all_gather] "
            "disabling ENABLE_SDMA_ALLGATHER to avoid the direct-HIP SDMA path"
        )
        os.environ["ENABLE_SDMA_ALLGATHER"] = "0"

    # The dedicated process group selects zero CTA through pg_options. A global
    # policy may already have affected WORLD/DP communicators by this phase, so
    # fail instead of silently removing it too late.
    validate_global_cta_policy()
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    os.environ["NCCL_LOCAL_REGISTER"] = "0"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "true"

    eager_param_bytes = int(os.getenv("MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES", "0"))
    if int(os.getenv("RANK", "0")) == 0:
        print(
            "[RCCL-SDMA:Megatron] eager direct configuration " f"bytes={eager_param_bytes}",
            flush=True,
        )
    if eager_param_bytes:
        import torch.distributed._symmetric_memory as symm_mem

        from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
            reserve_direct_param_buffer,
        )

        device = torch.device("cuda", int(os.getenv("LOCAL_RANK", torch.cuda.current_device())))
        torch.cuda.set_device(device)
        symm_mem.set_backend("NCCL")
        pool = symm_mem.get_mem_pool(device)
        reserve_direct_param_buffer(
            None,
            pool,
            device,
            eager_param_bytes,
        )

    try:
        import megatron.core.distributed.param_and_grad_buffer as pgb
    except ImportError as exc:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_param_all_gather] "
            f"Megatron distributed modules are unavailable; skipping: {exc}"
        )
        return

    bucket_group = getattr(pgb, "_ParamAndGradBucketGroup", None)
    if bucket_group is None:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_param_all_gather] "
            "_ParamAndGradBucketGroup is unavailable; skipping"
        )
        return

    marker = "_primus_rccl_sdma_param_gather_patched"
    if not getattr(bucket_group, marker, False):
        bucket_group.start_param_sync = make_start_param_sync(bucket_group.start_param_sync)
        setattr(bucket_group, marker, True)

    param_and_grad_buffer = getattr(pgb, "_ParamAndGradBuffer", None)
    if param_and_grad_buffer is None:
        raise RuntimeError("RCCL-SDMA direct gather requires _ParamAndGradBuffer")
    direct_marker = "_primus_rccl_sdma_direct_allocation_patched"
    if not getattr(param_and_grad_buffer, direct_marker, False):
        param_and_grad_buffer.__init__ = make_param_and_grad_buffer_init(param_and_grad_buffer.__init__)
        setattr(param_and_grad_buffer, direct_marker, True)

    if os.getenv("MEGATRON_RCCL_SDMA_EAGER_INIT", "0") == "1":
        # Even if WORLD was initialized by the caller, Megatron may create more
        # groups later. Revisit the registry after initialize_megatron returns.
        eager_initialize_runtime()
        import megatron.training.training as training_module

        eager_marker = "_primus_rccl_sdma_eager_init_patched"
        if not getattr(training_module, eager_marker, False):
            training_module.initialize_megatron = make_eager_initialize_megatron(
                training_module.initialize_megatron
            )
            setattr(training_module, eager_marker, True)

    log_rank_0("[Patch:megatron.distributed.rccl_sdma_param_all_gather] installed")
