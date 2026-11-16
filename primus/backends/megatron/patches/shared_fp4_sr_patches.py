# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Opt-in native MXFP4 shared-Philox routing and continuation state.

This aligns the RNG family and E2M1 rounding law with the two-level recipes.
Different block geometries and execution orders do not have identical streams.
The default ``recipe`` policy does not install wrappers or import Turbo.
"""

from __future__ import annotations

import copy
import json
import os
from functools import wraps
from pathlib import Path

from primus.backends.megatron.core.fp4_dispatch import gradient_rounding_policy
from primus.core.patches import PatchContext, get_args, register_patch

_MARKER = "_primus_shared_fp4_sr"
_DENSE_FUSION_FLAGS = ("turbo_fused_gemm", "use_turbo_fused_dense_mlp")
_DENSE_FUSION_ENV = ("PRIMUS_TURBO_FUSED_DENSE_MLP", "PRIMUS_TURBO_FUSED_GEMM")
_BYPASS_FLAGS = (
    "turbo_fused_grouped_gemm",
    "moe_permute_quant_fusion",
    "moe_backward_permute_quant_fusion",
    "use_turbo_mega_moe",
    "use_turbo_deepep",
    "fp4_use_native_te_autocast",
    "moe_use_legacy_grouped_gemm",
    "use_te_op_fuser",
    "use_transformer_engine_op_fuser",
    "use_grouped_gemm_for_dense_mlp",
)
_CONFIG_FIELDS = (
    "fp4",
    "fp4_recipe",
    "fp4_gradient_rounding_policy",
    "seed",
    "fp4_rng_seed",
    "transformer_impl",
    "mxfp4_gradient_stochastic_rounding",
    "mxfp4_scale_rounding_mode",
    "enable_primus_turbo",
    "use_turbo_gemm",
    "use_turbo_grouped_gemm",
    "first_last_layers_bf16",
    "num_layers_at_start_in_bf16",
    "num_layers_at_end_in_bf16",
    "num_layers",
    "num_experts",
    "moe_router_topk",
    "turbo_grouped_gemm_without_padding",
    "moe_router_padding_for_quantization",
    "moe_pad_expert_input_to_capacity",
    "moe_expert_capacity_factor",
    "micro_batch_size",
    "global_batch_size",
    "seq_length",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "context_parallel_size",
    "data_parallel_size",
    "moe_grouped_gemm",
    "cuda_graph_impl",
    "turbo_sync_free_moe_stage",
    *_DENSE_FUSION_FLAGS,
    *_BYPASS_FLAGS,
)


def _recipe(args):
    value = getattr(args, "fp4_recipe", None)
    return getattr(value, "value", value)


def validate_shared_mxfp4_args(args):
    """Reject un-intercepted MX routes instead of silently using mixed SR."""
    if gradient_rounding_policy(args) != "shared_philox":
        return False
    if _recipe(args) in ("nvfp4", "amdfp4"):
        return False  # Their reference dispatch validates the same policy.
    if not getattr(args, "fp4", False) or _recipe(args) != "mxfp4":
        raise ValueError("shared_philox requires an enabled MXFP4/NVFP4/AMDFP4 recipe")
    rht_mode = getattr(args, "fp4_rht_mode", "legacy")
    if rht_mode not in ("legacy", "mxfp4"):
        raise ValueError("fp4_rht_mode must be legacy or mxfp4")
    if rht_mode == "mxfp4" and (
        type(getattr(args, "fp4_rht_size", None)) is not int
        or args.fp4_rht_size != 16
        or getattr(args, "fp4_rht_randomized", None) is not False
    ):
        raise ValueError("Aligned MXFP4 RHT requires fp4_rht_size=16 and fp4_rht_randomized=false")
    for name in ("enable_primus_turbo", "use_turbo_gemm", "mxfp4_gradient_stochastic_rounding"):
        if not getattr(args, name, False):
            raise ValueError(f"MXFP4 shared_philox requires {name}=true")
    if getattr(args, "num_experts", None):
        if not getattr(args, "use_turbo_grouped_gemm", False):
            raise ValueError("MXFP4 shared_philox MoE requires use_turbo_grouped_gemm=true")
        if not getattr(args, "moe_grouped_gemm", True):
            raise ValueError("MXFP4 shared_philox MoE requires moe_grouped_gemm=true")
    if getattr(args, "transformer_impl", "transformer_engine") != "transformer_engine":
        raise ValueError("MXFP4 shared_philox currently requires the Turbo transformer_engine spec route")
    from primus.backends.megatron.patches.turbo.dense_mlp_fp4_patches import _fused_dense_mlp_requested

    # Use the producer's actual priority and false spellings. In particular,
    # an explicit false environment override disables even a true YAML flag.
    if _fused_dense_mlp_requested(args):
        raise ValueError(
            "MXFP4 shared_philox cannot intercept fused dense MLP: "
            "turbo_fused_gemm/use_turbo_fused_dense_mlp or "
            "PRIMUS_TURBO_FUSED_DENSE_MLP/PRIMUS_TURBO_FUSED_GEMM enables this route"
        )
    unsupported = [name for name in _BYPASS_FLAGS if getattr(args, name, False)]
    if unsupported:
        raise ValueError(
            "MXFP4 shared_philox cannot intercept these fused/alternate routes: " + ", ".join(unsupported)
        )
    if getattr(args, "turbo_sync_free_moe_stage", 0):
        raise ValueError("MXFP4 shared_philox requires turbo_sync_free_moe_stage=0")
    if getattr(args, "cuda_graph_impl", None) not in (None, False, "none"):
        raise ValueError("MXFP4 shared_philox private Python counters do not support CUDA graph replay")
    # This independent Flux warmup hook includes the first real train_step, so
    # wrapping its entire call would rewind real reservations as well.
    if getattr(args, "warmup_train_steps", 0):
        raise ValueError("MXFP4 shared_philox supports primus_mllog synthetic warmup, not warmup_train_steps")
    return True


def _runtime():
    from primus.backends.megatron.core import fp4_mxfp4_shared_sr

    return fp4_mxfp4_shared_sr


def _ensure_runtime(args):
    import torch.distributed as dist

    initialized = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if initialized else int(os.environ.get("RANK", "0"))
    world = dist.get_world_size() if initialized else int(os.environ.get("WORLD_SIZE", "1"))
    seed = getattr(args, "fp4_rng_seed", None)
    seed = getattr(args, "seed", 1234) if seed is None else seed
    runtime = _runtime()
    runtime.install_shared_mxfp4_sr(seed, rank=rank, world_size=world)
    return runtime


def _plain(value):
    value = getattr(value, "value", value)
    if isinstance(value, (tuple, list)):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported shared SR config identity value: {type(value).__name__}")


def _identity(args, models):
    parameters = {}
    for index, model in enumerate(models if isinstance(models, (list, tuple)) else [models]):
        while hasattr(model, "module"):
            model = model.module
        for name, parameter in model.named_parameters():
            parameters[f"model{index}/{name}"] = {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
    if not parameters:
        raise ValueError("Cannot checkpoint shared MXFP4 SR without model parameters")
    identity = {
        "config": {name: _plain(getattr(args, name, None)) for name in _CONFIG_FIELDS},
        "parameters": parameters,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PRIMUS_TURBO_MXFP4_SCALE_ROUNDING",
                "NVTE_MXFP4_USE_HADAMARD",
                *_DENSE_FUSION_ENV,
            )
        },
    }
    # Preserve the exact old sidecar identity for the legacy/default recipe.
    # The explicit alignment policy is a new continuation contract, even though
    # native MX already uses this deterministic FP32 H16 butterfly.
    if getattr(args, "fp4_rht_mode", "legacy") != "legacy":
        identity["rht_alignment"] = {
            "mode": _plain(getattr(args, "fp4_rht_mode", None)),
            "size": _plain(getattr(args, "fp4_rht_size", None)),
            "randomized": _plain(getattr(args, "fp4_rht_randomized", None)),
        }
    return identity


def shared_sr_checkpoint_state(args, models, iteration):
    import torch

    return {
        "version": 1,
        "iteration": int(iteration),
        "policy": "shared_philox",
        "torch_version": str(torch.__version__),
        "identity": _identity(args, models),
        "runtime": copy.deepcopy(_runtime().state_dict()),
    }


def validate_shared_sr_checkpoint(args, models, state, iteration):
    expected = shared_sr_checkpoint_state(args, models, iteration)
    if not isinstance(state, dict):
        raise ValueError("Invalid shared SR checkpoint")
    for name in ("version", "iteration", "policy", "torch_version", "identity"):
        if state.get(name) != expected[name]:
            raise ValueError(f"Shared SR checkpoint {name} mismatch")
    _runtime().validate_state_dict(state.get("runtime"))


def _sidecar_path(directory):
    rank = _runtime().state_dict()["topology"]["rank"]
    return Path(directory) / "shared_fp4_sr" / f"rank_{rank}.json"


def save_shared_sr_sidecar(args, models, directory, iteration):
    state = shared_sr_checkpoint_state(args, models, iteration)
    target = _sidecar_path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".tmp.{os.getpid()}")
    try:
        with temporary.open("w") as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return target


def load_shared_sr_sidecar(args, models, directory, iteration, *, validate_only=False):
    path = _sidecar_path(directory)
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared MXFP4 SR continuation sidecar: {path}")
    state = json.loads(path.read_text())
    validate_shared_sr_checkpoint(args, models, state, iteration)
    if not validate_only:
        _runtime().load_state_dict(state["runtime"])
    return state


def _resume_target(args, load_arg, checkpointing):
    root = getattr(args, load_arg, None)
    if not root or getattr(args, "finetune", False):
        return None
    step = getattr(args, "ckpt_step", None)
    if step is None:
        tracker = Path(checkpointing.get_checkpoint_tracker_filename(root))
        if not tracker.is_file():
            return None
        text = tracker.read_text().strip()
        if text == "release":
            return None  # Megatron resets iteration for release/finetuning loads.
        step = int(text)
    if int(step) <= 0:
        return None
    return checkpointing.get_checkpoint_name(root, int(step), return_base_dir=True), int(step)


def install_shared_mxfp4_training_state(args):
    if not validate_shared_mxfp4_args(args):
        return False
    from megatron.training import checkpointing, training

    if getattr(training.save_checkpoint, _MARKER, False):
        return True
    # before_train runs before Megatron's distributed/model setup. Install the
    # actual interceptor once get_model executes with its final rank topology.
    original_model = training.get_model

    @wraps(original_model)
    def get_model(*positional, **keywords):
        # Other setup hooks may set environment overrides after before_train.
        # Recheck immediately before the producer chooses its fused route.
        validate_shared_mxfp4_args(args)
        _ensure_runtime(args)
        return original_model(*positional, **keywords)

    training.get_model = get_model
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
        if not (
            getattr(args, "disable_last_saving", False) and iteration == getattr(args, "train_iters", None)
        ):
            if non_persistent_ckpt or not getattr(args, "save", None):
                raise ValueError("Shared MXFP4 SR requires a persistent checkpoint save directory")
            directory = checkpointing.get_checkpoint_name(
                args.save, iteration, release=release, return_base_dir=True
            )
            # Persist first: an error must not advance the main checkpoint tracker.
            save_shared_sr_sidecar(args, model, directory, iteration)
        return original_save(*forwarded, **keywords)

    @wraps(original_load)
    def load_checkpoint(models, optimizer, opt_param_scheduler, load_arg="load", *positional, **keywords):
        _ensure_runtime(args)
        before = copy.deepcopy(_runtime().state_dict())
        target = _resume_target(args, load_arg, checkpointing)
        prepared = None
        if target is not None:
            prepared = load_shared_sr_sidecar(args, models, *target, validate_only=True)
        try:
            result = original_load(models, optimizer, opt_param_scheduler, load_arg, *positional, **keywords)
            iteration = result[0] if isinstance(result, (tuple, list)) else getattr(args, "iteration", 0)
            if iteration and not getattr(args, "finetune", False):
                if prepared is not None and int(iteration) == target[1]:
                    _runtime().load_state_dict(prepared["runtime"])
                else:
                    directory = checkpointing.get_checkpoint_name(
                        getattr(args, load_arg), iteration, return_base_dir=True
                    )
                    load_shared_sr_sidecar(args, models, directory, iteration)
            return result
        except BaseException:
            _runtime().load_state_dict(before)
            raise

    setattr(save_checkpoint, _MARKER, True)
    training.save_checkpoint, training.load_checkpoint = save_checkpoint, load_checkpoint
    return True


def _native_deosc_runners(optimizer):
    # Avoid importing the optional native GPU stack for optimizers with no
    # native DeOsc runner (including CPU-only checkpoint-hook tests).
    leaves = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    if not any(getattr(leaf, "_primus_weight_deosc_runner", None) is not None for leaf in leaves):
        return []
    from primus.backends.megatron.core.optimizer.weight_deosc import iter_deosc_runners

    return iter_deosc_runners(optimizer)


def _restore_native_deosc_warmup(optimizer, snapshots):
    errors = []
    try:
        runners = _native_deosc_runners(optimizer)
        same = len(runners) == len(snapshots) and all(
            current is saved[0] and current.config is saved[3] for current, saved in zip(runners, snapshots)
        )
    except Exception as error:
        errors.append(error)
        same = False
    for runner, state, eligible, config, enabled in snapshots:
        try:
            runner.load_state_dict(state)
        except Exception as error:
            errors.append(error)
        finally:
            runner._eligible_ids = eligible
            config.enable = enabled
    if not same:
        raise RuntimeError("Native MXFP4 DeOsc runner/config identity changed during synthetic warmup")
    if errors:
        raise RuntimeError("Failed to restore native MXFP4 DeOsc after synthetic warmup") from errors[0]


def wrap_shared_sr_warmup(original):
    if getattr(original, _MARKER, False):
        return original

    @wraps(original)
    def warmup(
        train_step_func, forward_step_func, model, optimizer, opt_param_scheduler, config, megatron_args
    ):
        call = (
            train_step_func,
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            megatron_args,
        )
        if not validate_shared_mxfp4_args(megatron_args):
            return original(*call)
        from primus.backends.megatron.core.fp4_warmup import (
            _capture_rng,
            _collective_resume_reason,
            _restore_rng,
            _resume_reason,
        )
        from primus.core.utils.module_utils import log_rank_0

        # The upstream warmup clears floating optimizer state. A resumed Adam
        # trajectory must skip it, including ranks that locally own no moments.
        native_runners = _native_deosc_runners(optimizer)
        reason = _resume_reason(megatron_args, optimizer, opt_param_scheduler)
        if not reason and any(
            runner._global_step or runner._state or runner._loaded_params for runner in native_runners
        ):
            reason = "native MXFP4 DeOsc continuation state"
        reason = _collective_resume_reason(reason)
        if reason:
            log_rank_0(f"[SharedFP4SR] Synthetic warmup skipped: {reason}; continuation state retained")
            return None
        runtime = _ensure_runtime(megatron_args)
        saved = copy.deepcopy(runtime.state_dict())
        global_rng = _capture_rng()
        # Capture every original flag before disabling any runner: multiple
        # optimizers can share one WeightDeOscConfig instance. Warmup must not
        # snap real masters or allocate/advance DeOsc history, even temporarily.
        native_snapshots = [
            (
                runner,
                copy.deepcopy(runner.state_dict()),
                None if runner._eligible_ids is None else set(runner._eligible_ids),
                runner.config,
                runner.config.enable,
            )
            for runner in native_runners
        ]
        for runner in native_runners:
            runner.config.enable = False
        try:
            return original(*call)
        finally:
            try:
                try:
                    _restore_native_deosc_warmup(optimizer, native_snapshots)
                finally:
                    runtime.load_state_dict(saved)
            finally:
                _restore_rng(global_rng)

    setattr(warmup, _MARKER, True)
    return warmup


def install_shared_sr_warmup(warmup_module=None):
    if warmup_module is None:
        import importlib

        try:
            warmup_module = importlib.import_module("primus_mllog.warmup")
        except ModuleNotFoundError as error:
            if error.name in ("primus_mllog", "primus_mllog.warmup"):
                return False
            raise
    warmup_module.run_synthetic_warmup = wrap_shared_sr_warmup(warmup_module.run_synthetic_warmup)
    return True


def _enabled(ctx):
    return validate_shared_mxfp4_args(get_args(ctx))


@register_patch(
    "megatron.shared_fp4_sr.training_state",
    backend="megatron",
    phase="before_train",
    condition=_enabled,
    priority=102,
    description="Opt-in shared Philox MXFP4 SR with strict continuation and synthetic warmup state",
)
def patch_shared_mxfp4_sr(ctx: PatchContext):
    install_shared_mxfp4_training_state(get_args(ctx))
    install_shared_sr_warmup()
