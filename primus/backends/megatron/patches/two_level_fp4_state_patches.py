# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Bind private FP4 streams and persist strict state in Primus training."""

from functools import wraps

from primus.backends.megatron.patches.two_level_fp4_patches import _enabled
from primus.core.patches import PatchContext, get_args, register_patch


def install_two_level_fp4_training_state(args):
    import megatron.training.training as training
    from megatron.training.checkpointing import get_checkpoint_name

    from primus.backends.megatron.core.fp4_state import (
        bind_fp4_model,
        load_fp4_sidecar,
        save_fp4_sidecar,
    )

    original_model = training.get_model
    if getattr(original_model, "_primus_two_level_fp4_state", False):
        return

    @wraps(original_model)
    def get_model(*positional, **keywords):
        models = original_model(*positional, **keywords)
        bind_fp4_model(models)
        return models

    get_model._primus_two_level_fp4_state = True
    training.get_model = get_model
    original_optimizer = training.get_megatron_optimizer

    @wraps(original_optimizer)
    def get_optimizer(*positional, **keywords):
        models = keywords.get("model_chunks", positional[1] if len(positional) > 1 else None)
        if models is None:
            raise ValueError("FP4 optimizer installation requires model_chunks")
        bind_fp4_model(models)
        optimizer = original_optimizer(*positional, **keywords)
        if getattr(args, "weight_deosc", False):
            from primus.backends.megatron.core.optimizer.two_level_fp4_deosc import (
                install_megatron_deosc,
            )

            install_megatron_deosc(optimizer, models, args)
        return optimizer

    training.get_megatron_optimizer = get_optimizer
    original_save, original_load = training.save_checkpoint, training.load_checkpoint

    @wraps(original_save)
    def save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context=None,
        pipeline_rank=None,
        expert_rank=None,
        tensor_rank=None,
        pipeline_parallel=None,
        expert_parallel=None,
        non_persistent_ckpt=False,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
        release=False,
        *positional,
        **keywords,
    ):
        forwarded = (
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            pipeline_rank,
            expert_rank,
            tensor_rank,
            pipeline_parallel,
            expert_parallel,
            non_persistent_ckpt,
            train_data_iterator,
            preprocess_common_state_dict_fn,
            release,
            *positional,
        )
        if getattr(args, "disable_last_saving", False) and iteration == getattr(args, "train_iters", None):
            return original_save(*forwarded, **keywords)
        if non_persistent_ckpt:
            raise ValueError("Reference FP4 state currently requires a persistent checkpoint")
        if not getattr(args, "save", None):
            raise ValueError("Reference FP4 checkpoint requires an explicit save directory")
        directory = get_checkpoint_name(args.save, iteration, release=release, return_base_dir=True)
        # A failed state save must prevent the standard checkpoint tracker from
        # advancing. The state snapshot corresponds to the same completed step.
        save_fp4_sidecar(model, optimizer, directory, iteration)
        return original_save(*forwarded, **keywords)

    @wraps(original_load)
    def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg="load", *positional, **keywords):
        bind_fp4_model(ddp_model)
        result = original_load(ddp_model, optimizer, opt_param_scheduler, load_arg, *positional, **keywords)
        iteration = result[0] if isinstance(result, (tuple, list)) else getattr(args, "iteration", 0)
        if iteration and not getattr(args, "finetune", False):
            directory = get_checkpoint_name(getattr(args, load_arg), iteration, return_base_dir=True)
            load_fp4_sidecar(ddp_model, optimizer, directory, iteration)
        return result

    training.save_checkpoint, training.load_checkpoint = save_checkpoint, load_checkpoint


@register_patch(
    "megatron.two_level_fp4.training_state",
    backend="megatron",
    phase="before_train",
    description="Bind NVFP4/AMDFP4 RNG, strict checkpoint state and pre-copy DeOsc",
    condition=_enabled,
    priority=100,  # Wrap the existing model, optimizer and checkpoint patches.
)
def patch_two_level_fp4_training_state(ctx: PatchContext):
    install_two_level_fp4_training_state(get_args(ctx))


def install_two_level_fp4_warmup_state(warmup_module=None):
    """The MLPerf trainer imports this function locally immediately before use."""
    if warmup_module is None:
        import importlib

        try:
            warmup_module = importlib.import_module("primus_mllog.warmup")
        except ModuleNotFoundError as error:
            if error.name in ("primus_mllog", "primus_mllog.warmup"):
                return False  # Optional MLPerf plugin is not needed for ordinary training.
            raise
    from primus.backends.megatron.core.fp4_warmup import wrap_synthetic_warmup
    from primus.core.utils.module_utils import log_rank_0

    warmup_module.run_synthetic_warmup = wrap_synthetic_warmup(warmup_module.run_synthetic_warmup, log_rank_0)
    return True


@register_patch(
    "megatron.two_level_fp4.warmup_state",
    backend="megatron",
    phase="before_train",
    description="Restore reference FP4 state after synthetic warmup; skip warmup on continuation",
    condition=_enabled,
    priority=101,
)
def patch_two_level_fp4_warmup_state(ctx: PatchContext):
    install_two_level_fp4_warmup_state()
