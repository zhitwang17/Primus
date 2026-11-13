# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""DeOsc for the NVFP4/AMDFP4 reference adapters.

The detector retains Primus' BF16 snapshot/delta policy and defaults to FP32
distance accumulators. Explicit BF16 accumulation reproduces ALTO's distance
policy. Its QDQ is the *forward* two-level weight QDQ, including BF16
pre-rounding, block16/16x16 and whole expert-bank outer scale. There is no
MXFP4 import, boundary-shard approximation, stochastic rounding or RNG use.

Call ``step`` after a successful optimizer update and before master-to-model
copy. At a period boundary the master is snapped to current BF16 QDQ centers;
then QDQ is recomputed over the entire weight because its dynamic scale may
change. A checkpoint records the numerical recipe and snapshot policy as well
as each observation window. Replicated masters use ``TwoLevelDeOsc``;
DistributedOptimizer shards use the exact collective implementation in
``two_level_fp4_deosc_shard``. Neither path approximates boundary blocks.
"""

from __future__ import annotations

import math
from functools import wraps
from typing import Mapping

import torch

from ..fp4_backend import qdq
from ..fp4_contract import TwoLevelFP4Config

_ALGORITHM = "two-level-fp4-dist-ratio-v1"
_SNAPSHOT_POLICY = "bf16-snapshots-bf16-deltas-fp32-accumulators"
_RNG_POLICY = "deterministic-forward-weight-qdq-no-rng"
_EPS = 1e-12


def _get(source, key, default=None):
    return source.get(key, default) if isinstance(source, Mapping) else getattr(source, key, default)


class TwoLevelDeOsc:
    """Track ``key -> (BF16 model parameter, full FP32 master, config)``.

    A full flattened DP1 master is accepted and reshaped to model coordinates;
    partial slices, noncontiguous masters and dtype/device mismatches are not.
    Optional owning modules supply ``_primus_fp4_last_config``: BF16 executions
    reset that parameter's partial window, and recipe changes fail explicitly.
    ``period`` counts deltas after the first seeding observation; ``start_step``
    is a successful optimizer-update index (0 begins on the first update).
    """

    def __init__(
        self,
        bindings,
        *,
        period=200,
        ratio=4.0,
        start_step=0,
        active_modules=None,
        accumulator_dtype="fp32",
        epsilon=None,
    ):
        if type(period) is not int or period <= 0:
            raise ValueError("weight_deosc_period must be a positive integer")
        if isinstance(ratio, bool) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("weight_deosc_ratio must be finite and positive")
        if type(start_step) is not int or start_step < 0:
            raise ValueError("weight_deosc_start_step must be a nonnegative integer")
        if not bindings or any(not isinstance(key, str) or not key for key in bindings):
            raise ValueError("DeOsc requires nonempty stable parameter keys")
        self.bindings = dict(bindings)
        self.active_modules = None if active_modules is None else dict(active_modules)
        if self.active_modules is not None and set(self.active_modules) != set(self.bindings):
            raise ValueError("DeOsc owning modules must cover all parameter bindings")
        dtype_names = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            torch.float32: torch.float32,
            torch.bfloat16: torch.bfloat16,
        }
        if accumulator_dtype not in dtype_names:
            raise ValueError("DeOsc accumulator_dtype must be fp32 or bf16")
        self.accumulator_dtype = dtype_names[accumulator_dtype]
        if epsilon is None:
            epsilon = 1e-30 if self.accumulator_dtype == torch.bfloat16 else _EPS
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("DeOsc epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.period, self.ratio, self.start_step = period, float(ratio), start_step
        self.step_count = 0
        self._states = {}
        self._stats = {"observations": 0, "periods": 0, "snapped": 0}
        seen_models, seen_masters = set(), set()
        for key, (model, master, config) in self.bindings.items():
            self._validate_binding(key, model, master, config)
            if id(model) in seen_models or id(master) in seen_masters:
                raise ValueError("DeOsc bindings must not duplicate a model or master parameter")
            seen_models.add(id(model))
            seen_masters.add(id(master))

    def _validate_binding(self, key, model, master, config):
        if not isinstance(config, TwoLevelFP4Config):
            raise ValueError(f"{key}: DeOsc requires an explicit two-level FP4 recipe")
        if not isinstance(model, torch.Tensor) or model.ndim not in (2, 3):
            raise ValueError(f"{key}: DeOsc needs a full 2-D weight or 3-D expert bank")
        if model.dtype != torch.bfloat16:
            raise ValueError(f"{key}: DeOsc requires BF16 model weights")
        if not isinstance(master, torch.Tensor) or master.dtype != torch.float32:
            raise ValueError(f"{key}: DeOsc requires an explicit FP32 master")
        if master.numel() != model.numel() or not master.is_contiguous():
            raise ValueError(f"{key}: partial/noncontiguous master shards are unsupported")
        if model.device != master.device:
            raise ValueError(f"{key}: model and master must reside on the same device")

    def _master_view(self, model, master):
        return master.view_as(model)

    def _state_shape(self, key):
        return self.bindings[key][0].shape

    def _active_bindings(self):
        return {key: self._active(key, cfg) for key, (_, _, cfg) in self.bindings.items()}

    @property
    def stats(self):
        # Periods are parameter-window closures, which can differ across mixed
        # BF16/FP4 modules; they are not a global training-step counter.
        return {**self._stats, "tracked_parameters": len(self.bindings)}

    def _identity(self):
        return {
            "version": 2,
            "algorithm": _ALGORITHM,
            "snapshot_policy": (
                "bf16-snapshots-bf16-deltas-"
                + ("bf16" if self.accumulator_dtype == torch.bfloat16 else "fp32")
                + "-accumulators"
            ),
            "rng_policy": _RNG_POLICY,
            "settings": {
                "period": self.period,
                "ratio": self.ratio,
                "start_step": self.start_step,
                "accumulator_dtype": str(self.accumulator_dtype),
                "epsilon": self.epsilon,
            },
            "bindings": {
                key: {
                    "shape": list(model.shape),
                    "model_dtype": str(model.dtype),
                    "master_dtype": str(master.dtype),
                    "outer_scope": "complete_parameter",
                    "recipe": config.to_dict(),
                    "recipe_fingerprint": config.fingerprint,
                }
                for key, (model, master, config) in self.bindings.items()
            },
        }

    def _active(self, key, config):
        if self.active_modules is None:
            return True
        actual = getattr(self.active_modules[key], "_primus_fp4_last_config", None)
        if actual is None:
            return False
        if not isinstance(actual, TwoLevelFP4Config) or actual.fingerprint != config.fingerprint:
            raise ValueError(f"{key}: DeOsc recipe differs from the last forward recipe")
        return True

    @staticmethod
    def _weight_qdq(master, model, config):
        # qdq defaults to its input dtype, exactly as the BF16 forward adapter.
        return qdq(master.view_as(model).to(torch.bfloat16), config, config.operand("fwd", "weight"))

    @torch.no_grad()
    def step(self):
        """Observe one successful master update; never perform the model copy."""
        active = self._active_bindings()
        self.step_count += 1
        if self.step_count < max(1, self.start_step):
            return self.stats
        for key, (model, master, config) in self.bindings.items():
            if not active[key]:
                self._states.pop(key, None)
                continue
            current = self._master_view(model, master).to(torch.bfloat16)
            quantized = self._weight_qdq(master, model, config)
            self._stats["observations"] += 1
            state = self._states.get(key)
            if state is None:
                self._states[key] = {
                    "prev": current.clone(),
                    "prev_q": quantized.clone(),
                    "dist_w": torch.zeros_like(current, dtype=self.accumulator_dtype),
                    "dist_q": torch.zeros_like(current, dtype=self.accumulator_dtype),
                    "window_steps": 0,
                }
                continue
            # Deliberately retain the existing BF16 subtract/abs rounding.
            state["dist_w"].add_((current - state["prev"]).abs())
            state["dist_q"].add_((quantized - state["prev_q"]).abs())
            state["window_steps"] += 1
            if state["window_steps"] == self.period:
                mask = (state["dist_w"] > 0) & (
                    state["dist_q"] / state["dist_w"].clamp_min(self.epsilon) >= self.ratio
                )
                master_view = self._master_view(model, master)
                master_view.copy_(torch.where(mask, quantized.float(), master_view))
                self._stats["snapped"] += int(mask.sum().item())
                self._stats["periods"] += 1
                # Recalculate the FULL parameter/bank, including changed outer
                # and inner scales. QDQ idempotency is not an assumption here.
                current = master_view.to(torch.bfloat16)
                quantized = self._weight_qdq(master, model, config)
                state["dist_w"].zero_()
                state["dist_q"].zero_()
                state["window_steps"] = 0
            state["prev"] = current.clone()
            state["prev_q"] = quantized.clone()
        return self.stats

    def state_dict(self):
        return {
            **self._identity(),
            "step_count": self.step_count,
            "stats": dict(self._stats),
            # Every binding is explicit, including inactive BF16 overrides.
            # Otherwise a missing window is indistinguishable from inactivity
            # and resume can silently re-seed a partially observed period.
            "window_presence": {key: key in self._states for key in self.bindings},
            "states": {
                key: {
                    name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                    for name, value in state.items()
                }
                for key, state in self._states.items()
            },
        }

    def validate_state_dict(self, saved):
        """Validate the entire state without modifying the runner or tensors."""
        identity = self._identity()
        if not isinstance(saved, dict) or set(saved) != set(identity) | {
            "step_count",
            "stats",
            "window_presence",
            "states",
        }:
            raise ValueError("DeOsc checkpoint fields mismatch")
        if any(saved[name] != value for name, value in identity.items()):
            raise ValueError("DeOsc checkpoint recipe/shape/settings/snapshot identity mismatch")
        if type(saved["step_count"]) is not int or saved["step_count"] < 0:
            raise ValueError("Invalid DeOsc checkpoint step count")
        if (
            not isinstance(saved["stats"], dict)
            or set(saved["stats"]) != set(self._stats)
            or any(type(value) is not int or value < 0 for value in saved["stats"].values())
        ):
            raise ValueError("Invalid DeOsc checkpoint statistics")
        presence = saved["window_presence"]
        if (
            not isinstance(presence, dict)
            or set(presence) != set(self.bindings)
            or any(type(value) is not bool for value in presence.values())
        ):
            raise ValueError("Invalid DeOsc checkpoint window presence manifest")
        states = saved["states"]
        if not isinstance(states, dict) or set(states) != {
            key for key, present in presence.items() if present
        }:
            raise ValueError("Invalid DeOsc checkpoint parameter windows")
        if saved["step_count"] == 0 and states:
            raise ValueError("Unstarted DeOsc checkpoint cannot contain observation windows")
        for key, state in states.items():
            shape = self._state_shape(key)
            if not isinstance(state, dict) or set(state) != {
                "prev",
                "prev_q",
                "dist_w",
                "dist_q",
                "window_steps",
            }:
                raise ValueError(f"{key}: invalid DeOsc window fields")
            if type(state["window_steps"]) is not int or not 0 <= state["window_steps"] < self.period:
                raise ValueError(f"{key}: invalid DeOsc window length")
            for name, dtype in (
                ("prev", torch.bfloat16),
                ("prev_q", torch.bfloat16),
                ("dist_w", self.accumulator_dtype),
                ("dist_q", self.accumulator_dtype),
            ):
                value = state[name]
                if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
                    raise ValueError(f"{key}/{name}: DeOsc checkpoint tensor shape/dtype mismatch")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{key}/{name}: DeOsc checkpoint contains nonfinite values")
                if name.startswith("dist_") and bool((value < 0).any()):
                    raise ValueError(f"{key}/{name}: DeOsc distances must be nonnegative")

    def load_state_dict(self, saved):
        self.validate_state_dict(saved)
        restored = {
            key: {
                name: (
                    value.to(self.bindings[key][1].device).clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in state.items()
            }
            for key, state in saved["states"].items()
        }
        self._states = restored
        self.step_count = saved["step_count"]
        self._stats = dict(saved["stats"])


def _leaf_optimizers(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield optimizer
    else:
        for child in children:
            yield from _leaf_optimizers(child)


def _model_weights(models):
    weights = {}
    for model in models if isinstance(models, (tuple, list)) else [models]:
        for _, module in model.named_modules():
            config = getattr(module, "fp4_config", None)
            if not isinstance(config, TwoLevelFP4Config):
                continue
            stream = getattr(module, "fp4_stream_id", None)
            if not stream:
                raise ValueError("Bind FP4 model stream identities before installing DeOsc")
            for name, parameter in module.named_parameters(recurse=False):
                # Grouped biases can also be 2-D [E,N]. Only the adapter's
                # matrix-weight parameters are quantized by the forward path.
                if (
                    name not in ("weight", "weights")
                    or parameter.ndim not in (2, 3)
                    or not parameter.requires_grad
                ):
                    continue
                key = f"{stream}/{name}"
                if id(parameter) in weights:
                    raise ValueError("Aliased FP4 parameters need an explicit shared DeOsc owner")
                weights[id(parameter)] = (key, parameter, config, module)
    if not weights:
        raise ValueError("No explicitly configured two-level FP4 weights found for DeOsc")
    return weights


def _pairs(opt):
    if hasattr(opt, "model_float16_groups"):
        models = opt.model_float16_groups
        masters = getattr(opt, "shard_fp32_from_float16_groups", None)
    else:
        models = getattr(opt, "float16_groups", None)
        masters = getattr(opt, "fp32_from_float16_groups", None)
    if models is None or masters is None or len(models) != len(masters):
        raise ValueError("DeOsc requires Megatron model-to-FP32-master parameter groups")
    for model_group, master_group in zip(models, masters):
        if len(model_group) != len(master_group):
            raise ValueError("DeOsc model/master parameter-group lengths differ")
        yield from zip(model_group, master_group)


def _logical_weight_key(key):
    """DeOsc has no rank-local RNG; collective keys use logical model names."""
    prefix, separator, rest = key.partition("/")
    return rest if separator and prefix.startswith("rank") and prefix[4:].isdigit() else key


def _distributed_bindings_local(opt, weights):
    """Read actual MCore bucket membership and parameter-coordinate shard ranges.

    ``model_float16_groups`` contains only locally owned parameters. In
    contrast, every ``buffer.param_index_map`` contains all logical parameters
    in that optimizer buffer, including zero-owner weights on this rank.
    Neither bucket offsets nor grad-buffer padding are parameter coordinates.
    """
    from .two_level_fp4_deosc_shard import FP4ShardBinding

    buffers = getattr(opt, "buffers", None)
    range_map = getattr(opt, "_get_model_param_range_map", None)
    if buffers is None or not callable(range_map):
        raise ValueError("Sharded DeOsc needs Megatron buffers and model-parameter range mapping")
    membership, groups = set(), []
    for buffer in buffers:
        parameters = getattr(buffer, "param_index_map", None)
        if parameters is None:
            raise ValueError("Sharded DeOsc needs each Megatron buffer's complete param_index_map")
        tracked = {id(parameter) for parameter in parameters} & set(weights)
        if tracked:
            membership.update(tracked)
            groups.append(getattr(buffer, "data_parallel_group", getattr(opt, "data_parallel_group", None)))
    if not membership:
        return {}, {}, None, set()
    group = groups[0]
    # DistOpt buffers can use an intra-instance subgroup. Use the group which
    # actually partitions their ranges, rather than assuming global DP/WORLD.
    if torch.distributed.is_initialized():

        def group_members(value):
            if value is None:
                return tuple(range(torch.distributed.get_world_size()))
            return tuple(torch.distributed.get_process_group_ranks(value))

        if any(group_members(value) != group_members(group) for value in groups):
            raise ValueError("One DeOsc leaf optimizer cannot mix differently partitioned DP groups")
    elif any(value is not None for value in groups):
        raise ValueError("Sharded DeOsc process groups require initialized torch.distributed")
    local_pairs = {}
    for parameter, master in _pairs(opt):
        if id(parameter) in local_pairs:
            raise ValueError("A model parameter has multiple local FP32 master shards")
        local_pairs[id(parameter)] = master
    bindings, owners = {}, {}
    for parameter_id in sorted(membership, key=lambda value: _logical_weight_key(weights[value][0])):
        key, parameter, config, module = weights[parameter_id]
        key = _logical_weight_key(key)
        if key in bindings:
            raise ValueError(f"Duplicate logical sharded DeOsc parameter key: {key}")
        if parameter_id in local_pairs:
            master = local_pairs[parameter_id]
            if (
                not isinstance(master, torch.Tensor)
                or master.dtype != torch.float32
                or not master.is_contiguous()
            ):
                raise ValueError(f"{key}: DeOsc needs an explicit contiguous FP32 master shard")
            coordinates = range_map(parameter)["param"]
            start, end = coordinates.start, coordinates.end
            if end - start != master.numel():
                raise ValueError(f"{key}: Megatron parameter range differs from its FP32 master size")
            master = master.view(-1)
        else:
            # Absence is valid only for a genuine non-owner, never as a silent
            # substitute for a missing optimizer master mapping.
            if parameter in getattr(opt, "model_param_gbuf_map", {}):
                raise ValueError(f"{key}: owned parameter is missing its FP32 master")
            master = torch.empty(0, dtype=torch.float32, device=parameter.device)
            start = end = 0
        bindings[key] = FP4ShardBinding(parameter, master, config, start, end)
        owners[key] = module
    return bindings, owners, group, membership


def _distributed_bindings(opt, weights):
    """Make rank-local optimizer mapping failures collective before installation."""
    group = getattr(opt, "data_parallel_group", None)
    error, result = None, None
    try:
        buffers = getattr(opt, "buffers", None)
        if buffers:
            # These groups are present even when this rank owns no part of a
            # tracked parameter. Establish the collective before reading the
            # rank-local master/range map, which may be malformed on one rank.
            for buffer in buffers:
                parameters = getattr(buffer, "param_index_map", {})
                if any(id(parameter) in weights for parameter in parameters):
                    group = getattr(buffer, "data_parallel_group", group)
                    break
        result = _distributed_bindings_local(opt, weights)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    payload = {"error": error, "keys": sorted(result[0]) if result is not None else None}
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size(group) if initialized else 1
    gathered = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(gathered, payload, group=group)
    else:
        gathered[0] = payload
    errors = [f"group rank {rank}: {item['error']}" for rank, item in enumerate(gathered) if item["error"]]
    if errors:
        raise ValueError("Sharded DeOsc optimizer mapping failed: " + "; ".join(errors))
    if any(item["keys"] != gathered[0]["keys"] for item in gathered):
        raise ValueError("Sharded DeOsc optimizer buffers disagree on logical parameter membership")
    return result


def _wrap_optimizer(opt, runner):
    original_step = opt.step_with_ready_grads
    original_copy = opt._copy_main_params_to_model_params
    status = {"active": False, "copies": 0}

    @wraps(original_copy)
    def copy(*args, **kwargs):
        if status["active"]:
            if status["copies"]:
                raise RuntimeError(
                    "DeOsc requires exactly one main-to-model copy per successful optimizer step"
                )
            runner.step()
            status["copies"] += 1
        return original_copy(*args, **kwargs)

    @wraps(original_step)
    def step(*args, **kwargs):
        if status["active"]:
            raise RuntimeError("Reentrant optimizer step is unsupported by DeOsc")
        status.update(active=True, copies=0)
        try:
            successful = original_step(*args, **kwargs)
            if successful is True and status["copies"] != 1:
                raise RuntimeError("Optimizer skipped the required DeOsc main-to-model copy boundary")
            if successful is not True and status["copies"]:
                raise RuntimeError("Optimizer reported failure after copying updated DeOsc parameters")
            return successful
        finally:
            status["active"] = False

    opt._copy_main_params_to_model_params = copy
    opt.step_with_ready_grads = step
    opt._primus_two_level_deosc = runner


def install_megatron_deosc(optimizer, models, args):
    """Install on supported leaf optimizers; return runners for checkpointing.

    Supports ordinary Float16OptimizerWithFloat16Params and explicitly mapped
    DistributedOptimizer master shards. Constructor/load-time copies remain untouched.
    Configuration or binding errors raise, including missing tracked weights;
    no partial best-effort installation or hidden MXFP4 fallback is allowed.
    """
    if not _get(args, "weight_deosc", False):
        return []
    if hasattr(optimizer, "_primus_two_level_deosc_runners"):
        raise ValueError("Two-level FP4 DeOsc is already installed")
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "context_parallel_size",
    ):
        if _get(args, name, 1) not in (None, 1):
            raise ValueError(f"Two-level FP4 DeOsc currently requires {name}=1")
    for name in (
        "weight_deosc_fusion",
        "weight_deosc_direct_qdq",
        "use_precision_aware_optimizer_no_fp8_or_ds_fp8",
        "optimizer_cpu_offload",
        "reuse_grad_buf_for_mxfp8_param_ag",
        "use_megatron_fsdp",
    ):
        if _get(args, name, False):
            raise ValueError(f"Two-level FP4 DeOsc does not support {name}")
    weights = _model_weights(models)
    matched = set()
    pending = []
    for opt in _leaf_optimizers(optimizer):
        if hasattr(opt, "_primus_two_level_deosc"):
            raise ValueError("Two-level FP4 DeOsc is already installed on a leaf optimizer")
        opt_config = getattr(opt, "config", None)
        for name in (
            "use_precision_aware_optimizer_no_fp8_or_ds_fp8",
            "optimizer_cpu_offload",
            "reuse_grad_buf_for_mxfp8_param_ag",
        ):
            if _get(opt_config, name, False):
                raise ValueError(f"Two-level FP4 DeOsc does not support optimizer {name}")
        options = dict(
            period=_get(args, "weight_deosc_period", 200),
            ratio=_get(args, "weight_deosc_ratio", 4.0),
            start_step=_get(args, "weight_deosc_start_step", 0),
            accumulator_dtype=_get(args, "weight_deosc_accumulator_dtype", "fp32"),
            epsilon=_get(args, "weight_deosc_epsilon", None),
        )
        is_distributed = hasattr(opt, "model_float16_groups")
        if _get(args, "use_distributed_optimizer", False) and not is_distributed:
            raise ValueError("Configured distributed optimizer lacks Megatron FP32 shard metadata")
        if is_distributed and hasattr(opt, "buffers"):
            from .two_level_fp4_deosc_shard import ShardedTwoLevelDeOsc

            shard_bindings, owners, group, membership = _distributed_bindings(opt, weights)
            if not shard_bindings:
                continue
            if matched & membership:
                raise ValueError("FP4 parameter appears in multiple distributed optimizers")
            matched.update(membership)
            runner = ShardedTwoLevelDeOsc(
                shard_bindings, process_group=group, active_modules=owners, **options
            )
            if getattr(opt, "is_stub_optimizer", False):
                raise ValueError("DeOsc cannot run on a stub optimizer")
            if not callable(getattr(opt, "step_with_ready_grads", None)) or not callable(
                getattr(opt, "_copy_main_params_to_model_params", None)
            ):
                raise ValueError("DeOsc requires the Megatron successful-step/main-to-model-copy boundary")
            pending.append((opt, runner))
            continue
        if is_distributed:
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if max(world_size, _get(args, "data_parallel_size", 1) or 1) > 1:
                raise ValueError("Sharded DeOsc requires complete Megatron buffers/range metadata")
        bindings, owners = {}, {}
        for model_param, master in _pairs(opt):
            entry = weights.get(id(model_param))
            if entry is None:
                continue
            key, model_param, config, module = entry
            if id(model_param) in matched:
                raise ValueError(f"{key}: parameter appears in multiple optimizers")
            matched.add(id(model_param))
            bindings[key] = model_param, master, config
            owners[key] = module
        if not bindings:
            continue
        if getattr(opt, "is_stub_optimizer", False):
            raise ValueError("DeOsc cannot run on a stub optimizer")
        if not callable(getattr(opt, "step_with_ready_grads", None)) or not callable(
            getattr(opt, "_copy_main_params_to_model_params", None)
        ):
            raise ValueError("DeOsc requires the Megatron successful-step/main-to-model-copy boundary")
        runner = TwoLevelDeOsc(
            bindings,
            active_modules=owners,
            **options,
        )
        pending.append((opt, runner))
    if matched != set(weights):
        missing = sorted(weights[key][0] for key in set(weights) - matched)
        raise ValueError(f"DeOsc FP32 master mapping is incomplete: {missing}")
    for opt, runner in pending:
        _wrap_optimizer(opt, runner)
    runners = [runner for _, runner in pending]
    optimizer._primus_two_level_deosc_runners = runners
    return runners
