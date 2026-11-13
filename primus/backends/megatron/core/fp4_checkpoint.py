# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Real Megatron torch_dist checkpoint helpers for reference FP4 smoke jobs.

Model/master/moment tensors use Megatron sharded-state descriptors and the
torch_dist backend. Rank-local global RNG is a ShardedObject; private FP4/DeOsc
state uses the same strict sidecar helpers as the Primus training entrypoints.
This supports continuation with the same model, optimizer and parallel topology.
"""

from pathlib import Path

import torch

from .fp4_state import _models, bind_fp4_model, load_fp4_sidecar, save_fp4_sidecar


def _state(models, optimizer, iteration, extra_state=None, *, is_loading=False):
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    model_list = list(_models(models))
    if len(model_list) != 1:
        raise ValueError("FP4 torch_dist helper currently supports one PP1 model chunk")
    model = model_list[0]
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    model_state = model.sharded_state_dict()
    # Match the current training checkpoint path's DP bucket layout. MCore's
    # deprecated default (fully_sharded_model_space) still constructs nonempty
    # flattened_range descriptors, which current ShardedTensor rejects. The DP
    # format saves local FP32 masters AND every Adam state tensor without that
    # representation; loading must request the same layout. Replicated mixed-
    # precision optimizers accept this metadata and use their ordinary layout.
    optimizer_metadata = {"distrib_optim_sharding_type": "dp_reshardable"}
    state = {
        "model": model_state,
        "optimizer": optimizer.sharded_state_dict(
            model_state, is_loading=is_loading, metadata=optimizer_metadata
        ),
        "iteration": int(iteration),
        "fp4_checkpoint_version": 1,
        "rank_state": ShardedObject("fp4_rank_state", extra_state, (world,), (rank,)),
    }
    return model, state


def save_fp4_checkpoint(models, optimizer, directory, iteration, extra_state=None):
    """Synchronous collective save; every DP rank participates."""
    from megatron.core import dist_checkpointing

    if not torch.distributed.is_initialized():
        raise ValueError("FP4 torch_dist checkpoint requires initialized process groups")
    bind_fp4_model(models)
    directory = Path(directory)
    tensor_directory = directory / "torch_dist"
    if torch.distributed.get_rank() == 0:
        tensor_directory.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    # All rank-local state is complete before tensor checkpoint metadata is
    # published. A sidecar failure is a failed job, never a partial success.
    save_fp4_sidecar(models, optimizer, directory, iteration)
    torch.distributed.barrier()
    _, state = _state(models, optimizer, iteration, extra_state)
    dist_checkpointing.save(state, str(tensor_directory), async_sharded_save=False)
    torch.distributed.barrier()


def load_fp4_checkpoint(models, optimizer, directory, iteration):
    """Collective model/optimizer restore plus strict rank-local state restore."""
    from megatron.core import dist_checkpointing

    bind_fp4_model(models)
    model, state = _state(models, optimizer, iteration, is_loading=True)
    restored = dist_checkpointing.load(state, str(Path(directory) / "torch_dist"), strict="raise_all")
    if restored.get("fp4_checkpoint_version") != 1 or restored.get("iteration") != int(iteration):
        raise ValueError("FP4 torch_dist checkpoint version/iteration mismatch")
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    load_fp4_sidecar(models, optimizer, directory, iteration)
    torch.distributed.barrier()
    return restored["rank_state"]
