# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Stable per-module FP4 randomness and strict same-topology continuation state.

Forward QDQ consumes no randomness; backward captures each module's private
state. Recomputing a forward therefore leaves reservations unchanged. Model and
optimizer tensors remain in the standard Megatron checkpoint; this state holds
only the reference recipe/RNG identity and optional DeOsc history.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .fp4_contract import FP4RandomState
from .fp4_reference import REFERENCE_RNG_VERSION


def _models(models):
    for model in models if isinstance(models, (list, tuple)) else [models]:
        # Megatron DDP/Float16 wrappers add only a .module layer. Strip these so
        # constructing the model with or without DDP preserves module identities.
        while hasattr(model, "module"):
            model = model.module
        yield model


def _modules(models):
    for index, model in enumerate(_models(models)):
        for name, module in model.named_modules():
            if hasattr(module, "fp4_configs"):
                yield f"model{index}/{name or '<root>'}", module


def _topology(rank=None):
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    actual_rank = torch.distributed.get_rank() if initialized else 0
    return {
        "rank": actual_rank if rank is None else rank,
        "world_size": torch.distributed.get_world_size() if initialized else 1,
    }


def bind_fp4_model(models, rank=None):
    """Call after model construction and before optimizer/checkpoint/forward.

    Rebinding the same module is idempotent, not an RNG reset. Train/eval
    overrides use independent streams keyed by their complete recipe identity.
    """
    topology = _topology(rank)
    count = 0
    for name, module in _modules(models):
        stream = f"rank{topology['rank']}/{name}"
        previous = getattr(module, "fp4_stream_id", stream)
        if previous != stream:
            raise ValueError("FP4 module moved after binding; restore into the same named topology")
        module.fp4_stream_id = stream
        states = getattr(module, "fp4_rngs", {})
        for config in module.fp4_configs:
            if config.fingerprint not in states:
                states[config.fingerprint] = FP4RandomState.for_config(
                    config, stream + "/" + config.rng_fingerprint
                )
        module.fp4_rngs = states
        module.fp4_rng = states[module.fp4_config.fingerprint]
        count += 1
    if not count:
        raise ValueError("No two-level FP4 modules found; the reference spec provider was not installed")
    return count


def _runners(optimizer):
    return getattr(optimizer, "_primus_two_level_deosc_runners", []) if optimizer is not None else []


def fp4_state_dict(models, optimizer=None, *, rank=None):
    modules = {}
    for name, module in _modules(models):
        if not getattr(module, "fp4_stream_id", None):
            raise ValueError("Bind FP4 model before saving continuation state")
        modules[name] = {
            "stream_id": module.fp4_stream_id,
            "routing": (
                module.fp4_routing_identity()
                if hasattr(module, "fp4_routing_identity")
                else {"training_config": module.fp4_config.fingerprint}
            ),
            "parameters": {
                key: {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                }
                for key, param in module.named_parameters(recurse=False)
            },
            "recipes": {config.fingerprint: config.to_dict() for config in module.fp4_configs},
            "rng": {key: state.state_dict() for key, state in module.fp4_rngs.items()},
        }
    if not modules:
        raise ValueError("Cannot checkpoint an empty FP4 module set")
    return {
        "version": 1,
        "backend": "reference",
        "rng_implementation": REFERENCE_RNG_VERSION,
        "torch_version": str(torch.__version__),
        "topology": _topology(rank),
        "modules": modules,
        "deosc": [runner.state_dict() for runner in _runners(optimizer)],
    }


def load_fp4_state_dict(models, state, optimizer=None, *, rank=None):
    """Reject recipe, layout, RNG implementation or topology drift on resume."""
    expected = fp4_state_dict(models, optimizer, rank=rank)
    for key in ("version", "backend", "rng_implementation", "torch_version", "topology"):
        if state.get(key) != expected[key]:
            raise ValueError(f"FP4 checkpoint {key} mismatch")
    saved = state.get("modules", {})
    if saved.keys() != expected["modules"].keys():
        raise ValueError("FP4 checkpoint module set mismatch")
    restored = []
    for name, module in _modules(models):
        current, checkpoint = expected["modules"][name], saved[name]
        for key in ("stream_id", "parameters", "recipes", "routing"):
            if checkpoint.get(key) != current[key]:
                raise ValueError(f"FP4 checkpoint {name}/{key} mismatch")
        configs = {c.fingerprint: c for c in module.fp4_configs}
        if checkpoint.get("rng", {}).keys() != configs.keys():
            raise ValueError(f"FP4 checkpoint {name} RNG set mismatch")
        rngs = {
            key: FP4RandomState.from_state_dict(
                checkpoint["rng"][key], config, module.fp4_stream_id + "/" + config.rng_fingerprint
            )
            for key, config in configs.items()
        }
        restored.append((module, rngs))
    runners = _runners(optimizer)
    if len(state.get("deosc", [])) != len(runners):
        raise ValueError("FP4 checkpoint DeOsc runner set mismatch")
    for runner, history in zip(runners, state["deosc"]):
        runner.validate_state_dict(history)
    for runner, history in zip(runners, state["deosc"]):
        runner.load_state_dict(history)
    for module, rngs in restored:
        module.fp4_rngs = rngs
        module.fp4_rng = rngs[module.fp4_config.fingerprint]


def save_fp4_sidecar(models, optimizer, checkpoint_dir, iteration):
    """Write atomically before Megatron publishes its checkpoint tracker."""
    state = fp4_state_dict(models, optimizer)
    state["iteration"] = int(iteration)
    directory = Path(checkpoint_dir) / "two_level_fp4"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank_{state['topology']['rank']}.pt"
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_fp4_sidecar(models, optimizer, checkpoint_dir, iteration):
    path = Path(checkpoint_dir) / "two_level_fp4" / f"rank_{_topology()['rank']}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FP4 continuation state: {path}; use explicit finetune to reset")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("iteration") != int(iteration):
        raise ValueError("FP4 checkpoint iteration mismatch")
    load_fp4_state_dict(models, state, optimizer)
