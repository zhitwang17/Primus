# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Isolate reference FP4 state from optional MLPerf synthetic warmup."""

from __future__ import annotations

import copy
import os
from functools import wraps

from .fp4_dispatch import is_two_level, option

_MISSING = object()
_MARKER = "_primus_fp4_warmup_active"


def _inner_optimizers(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        for child in children:
            yield from _inner_optimizers(child)
    else:
        yield getattr(optimizer, "optimizer", optimizer)


def _resume_reason(args, optimizer, scheduler):
    for name in ("iteration", "consumed_train_samples"):
        if getattr(args, name, 0):
            return f"{name} is already nonzero"
    if scheduler is not None and getattr(scheduler, "num_steps", 0):
        return "the LR scheduler has already advanced"
    if any(runner.step_count for runner in getattr(optimizer, "_primus_two_level_deosc_runners", [])):
        return "DeOsc has already observed an optimizer update"
    if any(bool(getattr(inner, "state", {})) for inner in _inner_optimizers(optimizer)):
        # The external warmup unconditionally zeros floating optimizer state.
        # Even preallocated zeros are conservatively protected: skipping only
        # omits performance warmup, and does not skip any real training steps.
        return "optimizer state already exists (including preallocated zero state)"
    return None


def _collective_resume_reason(reason):
    """An empty-owner rank must not warm up while another rank skips."""
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        device = torch.cuda.current_device() if dist.get_backend() == "nccl" else "cpu"
        skip = torch.tensor([int(reason is not None)], dtype=torch.int32, device=device)
        # Two-level reference training currently requires TP/PP/EP/CP=1;
        # every world rank participates in the same data-parallel warmup.
        dist.all_reduce(skip, op=dist.ReduceOp.MAX)
        if skip.item():
            return reason or "another data-parallel rank has continuation/optimizer state"
    return reason


def _capture_rng():
    import random

    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    tracker = get_cuda_rng_tracker()
    tracker = tracker if tracker is not None and tracker.is_initialized() else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        # Each Megatron rank trains on its current device. Do not initialize
        # other visible devices simply to take a warmup snapshot.
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        "tracker": tracker,
        "tracker_states": copy.deepcopy(tracker.get_states()) if tracker is not None else None,
    }


def _restore_rng(saved):
    import random

    import numpy as np
    import torch

    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"])
    if saved["tracker"] is not None:
        saved["tracker"].set_states(copy.deepcopy(saved["tracker_states"]))


def _restore_attributes(target, attributes):
    for name, value in attributes.items():
        if value is _MISSING:
            if hasattr(target, name):
                delattr(target, name)
        else:
            setattr(target, name, value)


def wrap_synthetic_warmup(original, log):
    """Wrap the installed MLPerf API, preserving its model/optimizer handling.

    Exceptions propagate. The external implementation itself does not promise
    model/Adam rollback after a failed warmup, so such failures must abort the
    run; this wrapper still restores FP4/private/global RNG and its own flags.
    """
    if getattr(original, "_primus_fp4_warmup_state", False):
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
        if int(os.getenv("SYNTH_WARMUP_STEPS", "3")) <= 0:
            return original(*call)
        if not is_two_level(megatron_args) or option(megatron_args, "fp4_backend") != "reference":
            return original(*call)
        reason = _collective_resume_reason(_resume_reason(megatron_args, optimizer, opt_param_scheduler))
        if reason:
            log(f"[TwoLevelFP4] Synthetic warmup skipped: {reason}; real training continues normally")
            return None
        if os.getenv("WARMUP_RECIPE", "").strip() or os.getenv("WARMUP_FP8_RECIPE", "").strip():
            raise ValueError(
                "Reference FP4 synthetic warmup uses the model recipe; leave WARMUP_RECIPE "
                "and WARMUP_FP8_RECIPE empty (foreign TE autocast is unsupported)"
            )

        from .fp4_state import _modules, fp4_state_dict, load_fp4_state_dict

        private = fp4_state_dict(model, optimizer)
        routing = [
            (module, getattr(module, "_primus_fp4_last_config", _MISSING)) for _, module in _modules(model)
        ]
        rng = _capture_rng()
        flags = [
            (optimizer, {_MARKER: getattr(optimizer, _MARKER, _MISSING)}),
            (
                megatron_args,
                {
                    name: getattr(megatron_args, name, _MISSING)
                    for name in (
                        _MARKER,
                        "iteration",
                        "consumed_train_samples",
                        "skipped_train_samples",
                    )
                },
            ),
            (
                config,
                {
                    name: getattr(config, name, _MISSING)
                    for name in (
                        "finalize_model_grads_func",
                        "grad_scale_func",
                        "no_sync_func",
                    )
                },
            ),
        ]
        setattr(optimizer, _MARKER, True)
        setattr(megatron_args, _MARKER, True)
        try:
            return original(*call)
        finally:
            try:
                load_fp4_state_dict(model, private, optimizer)
                for module, previous in routing:
                    _restore_attributes(module, {"_primus_fp4_last_config": previous})
            finally:
                try:
                    _restore_rng(rng)
                finally:
                    for target, attributes in flags:
                        _restore_attributes(target, attributes)

    warmup._primus_fp4_warmup_state = True
    return warmup
