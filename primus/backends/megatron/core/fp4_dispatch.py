# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Explicit two-level FP4 recipe selection and reference execution guards."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Mapping

from .fp4_contract import TwoLevelFP4Config, _boolean, _name

_ACTIVE_PLAN = ContextVar("primus_two_level_fp4", default=None)
_LOG = logging.getLogger(__name__)


def option(config, key, default=None, fallback=None):
    for source in (config, fallback):
        if source is None:
            continue
        if isinstance(source, Mapping):
            if key in source and source[key] is not None:
                return source[key]
        elif hasattr(source, key) and getattr(source, key) is not None:
            return getattr(source, key)
    return default


def is_two_level(config):
    return _name(option(config, "fp4_recipe")) in ("nvfp4", "amdfp4")


def build_two_level_config(config, fallback=None):
    fields = {
        "block_size": ("fp4_block_size", 16),
        "weight_2d": ("fp4_weight_2d", True),
        "outer_scope": ("fp4_outer_scope", "tensor"),
        "inner_storage": ("fp4_inner_storage", "float32_grid"),
        "rht_size": ("fp4_rht_size", 32),
        "rht_randomized": ("fp4_rht_randomized", True),
        "gradient_sr": ("fp4_gradient_sr", True),
        "rng_seed": ("fp4_rng_seed", option(config, "seed", 1234, fallback)),
        "quantizer": ("fp4_quantizer", "torch"),
        "sr_rng": ("fp4_sr_rng", "torch_cpu"),
        "value_checks": ("fp4_value_checks", "strict"),
    }
    return TwoLevelFP4Config(
        recipe=_name(option(config, "fp4_recipe", fallback=fallback)),
        **{field: option(config, key, default, fallback) for field, (key, default) in fields.items()},
    )


@dataclass(frozen=True)
class FP4ExecutionPlan:
    config: TwoLevelFP4Config
    backend: str
    module_kind: str

    def to_dict(self):
        return {
            "recipe": self.config.to_dict() if self.backend == "reference" else {"recipe": "nvfp4"},
            "inner_format": self.config.inner_format,
            "fingerprint": self.config.fingerprint if self.backend == "reference" else None,
            "backend": self.backend,
            "module_kind": self.module_kind,
            "execution_status": (
                "QDQ reference consumer" if self.backend == "reference" else "TE device check required"
            ),
        }

    def require_execution(self):
        if self.backend == "reference":
            from .fp4_autograd import grouped_linear, linear

            assert callable(grouped_linear if self.module_kind == "grouped" else linear)
        # TE owns its actual device/version/kernel check. This does not certify
        # NVFP4 availability on ROCm; no AMDFP4 TE plan can be constructed.


def resolve_fp4_plan(config, module_kind="dense", fallback=None, environ=None):
    if module_kind not in ("dense", "grouped"):
        raise ValueError(f"Unknown FP4 module kind: {module_kind}")
    recipe = build_two_level_config(config, fallback)
    backend = option(config, "fp4_backend", "auto", fallback)
    turbo = _boolean(option(config, "enable_primus_turbo", False, fallback), "enable_primus_turbo")
    consumer = "use_turbo_gemm" if module_kind == "dense" else "use_turbo_grouped_gemm"
    turbo_consumer = turbo and _boolean(option(config, consumer, False, fallback), consumer)
    if backend == "auto":
        if turbo_consumer:
            backend = "turbo"
        elif recipe.recipe == "nvfp4":
            backend = "te"
        else:
            raise ValueError("AMDFP4 requires explicit fp4_backend=reference")
    if backend == "turbo":
        raise ValueError(f"Turbo {module_kind} GEMM supports MXFP4 block32/E8M0, not {recipe.recipe}")
    if backend not in ("reference", "te"):
        raise ValueError(f"Unknown fp4_backend={backend!r}; expected auto/reference/te/turbo")
    if option(config, "transformer_impl", "transformer_engine", fallback) != "transformer_engine":
        raise ValueError("Two-level FP4 routing requires transformer_impl=transformer_engine")
    if backend == "te":
        if recipe.quantizer != "torch":
            raise ValueError("fp4_quantizer=triton requires fp4_backend=reference; TE owns its quantizer")
        if recipe.recipe != "nvfp4":
            raise ValueError("Transformer Engine has no AMDFP4 recipe")
        if turbo_consumer:
            raise ValueError(f"fp4_backend=te requires {consumer}=false; Turbo would bypass TE quantization")
    else:
        if _name(option(config, "fp4", "e2m1", fallback)) != "e2m1":
            raise ValueError("Two-level reference recipes require fp4=e2m1")
        if option(config, "fp8", None, fallback):
            raise ValueError("fp4_backend=reference requires fp8=null")
        for name in ("fp4_param", "fp4_param_gather", "fp4_use_native_te_autocast"):
            if _boolean(option(config, name, False, fallback), name):
                raise ValueError(f"fp4_backend=reference requires {name}=false/null")
        for name in (
            "tensor_model_parallel_size",
            "expert_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ):
            if option(config, name, 1, fallback) != 1:
                raise ValueError(f"FP4 reference consumer requires {name}=1")
        for name in (
            "moe_use_legacy_grouped_gemm",
            "sequence_parallel",
            "use_te_op_fuser",
            "use_transformer_engine_op_fuser",
            "use_grouped_gemm_for_dense_mlp",
            "cuda_graph_impl",
            "use_torch_fsdp2",
            "use_custom_fsdp",
        ):
            if option(config, name, False, fallback) not in (False, None, "none"):
                raise ValueError(f"FP4 reference consumer does not support {name}")
        if option(config, "turbo_sync_free_moe_stage", 0, fallback):
            raise ValueError("FP4 reference consumer requires turbo_sync_free_moe_stage=0")
        if module_kind == "grouped" and not _boolean(
            option(config, "moe_grouped_gemm", True, fallback), "moe_grouped_gemm"
        ):
            raise ValueError("FP4 reference MoE requires moe_grouped_gemm=true")
        if _boolean(option(config, "weight_deosc", False, fallback), "weight_deosc"):
            # DistributedOptimizer's ordinary FP32 master shards are supported.
            # The installer validates their actual parameter ranges and DP group;
            # hidden masters, offload and foreign parameter-storage contracts are not.
            for name in (
                "use_precision_aware_optimizer_no_fp8_or_ds_fp8",
                "optimizer_cpu_offload",
                "reuse_grad_buf_for_mxfp4_param_ag",
                "reuse_grad_buf_for_mxfp8_param_ag",
                "use_megatron_fsdp",
            ):
                if _boolean(option(config, name, False, fallback), name):
                    raise ValueError(f"Reference FP4 DeOsc does not support {name}")
            accumulator = option(config, "weight_deosc_accumulator_dtype", "fp32", fallback)
            if accumulator not in ("fp32", "bf16"):
                raise ValueError("weight_deosc_accumulator_dtype must be fp32 or bf16")
            epsilon = option(config, "weight_deosc_epsilon", None, fallback)
            if epsilon is not None and (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon <= 0
            ):
                raise ValueError("weight_deosc_epsilon must be finite and positive")
    # These producers/optimizer hooks embed MXFP4 assumptions even when the
    # GEMM itself is delegated to TE. Guard them for BOTH new-format routes.
    for name in (
        "turbo_fused_grouped_gemm",
        "moe_permute_quant_fusion",
        "moe_backward_permute_quant_fusion",
        "weight_deosc_fusion",
        "weight_deosc_direct_qdq",
        "use_turbo_mega_moe",
        "use_turbo_deepep",
    ):
        if _boolean(option(config, name, False, fallback), name):
            raise ValueError(f"{name} is not wired for {recipe.recipe}; disable it for two-level FP4")
    if backend == "te" and _boolean(option(config, "weight_deosc", False, fallback), "weight_deosc"):
        raise ValueError("weight_deosc is only wired for fp4_backend=reference, not TE NVFP4")
    for name in (
        "PRIMUS_FUSED_RMSNORM_MXFP4_QKV",
        "PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT",
        "PRIMUS_TURBO_GRAD_OWNERSHIP",
    ):
        if _boolean((environ or {}).get(name, "0"), name):
            raise ValueError(f"{name} must be disabled for two-level FP4")
    return FP4ExecutionPlan(recipe, backend, module_kind)


def validate_fp4_options(args, environ=None, require_runtime=True):
    """Called by argument validation and by the CPU preflight tool."""
    if not option(args, "fp4"):
        if option(args, "fp4_backend", "auto") != "auto":
            raise ValueError("An explicit fp4_backend requires fp4 to be enabled")
        return ()
    precision = (environ or {}).get("PRIMUS_LINEAR_PRECISION")
    recipe = _name(option(args, "fp4_recipe"))
    if precision and precision != recipe:
        raise ValueError(f"PRIMUS_LINEAR_PRECISION={precision} disagrees with fp4_recipe={recipe}")
    if not is_two_level(args):
        if option(args, "fp4_backend", "auto") != "auto":
            raise ValueError("Explicit fp4_backend is currently defined only for NVFP4/AMDFP4")
        if option(args, "use_turbo_gemm", False) and recipe != "mxfp4":
            raise ValueError(f"Turbo GEMM requires mxfp4; got {recipe!r}")
        return ()
    kinds = ("dense", "grouped") if option(args, "num_experts", 0) else ("dense",)
    plans = tuple(resolve_fp4_plan(args, kind, environ=environ) for kind in kinds)
    for plan in plans:
        _LOG.info("Primus FP4 route: %s", plan.to_dict())
        if require_runtime:
            plan.require_execution()
    return plans


def current_fp4_plan():
    return _ACTIVE_PLAN.get()


@contextmanager
def fp4_recipe_context(plan):
    """Scoped metadata state; callers executing operators must gate separately."""
    token = _ACTIVE_PLAN.set(plan)
    try:
        yield plan
    finally:
        _ACTIVE_PLAN.reset(token)


def module_fp4_context(qrecipe, options, module_kind="dense", quantized=False):
    """Respect TE module override flags before selecting a new recipe.

    Used by the reference module boundary. Native TE modules have their own
    context; metadata alone cannot turn a reference module into a TE consumer.
    """
    from contextlib import nullcontext

    active = quantized or current_fp4_plan() is not None
    flag = "override_quantized_autocast" if active else "override_nonquantized_autocast"
    if not option(qrecipe, flag, False):
        return nullcontext()
    selected = _name(option(qrecipe, "fp4_quantization_recipe"))
    if selected is None and option(qrecipe, "fp8_quantization_recipe") is None:
        return fp4_recipe_context(None)
    if selected not in ("nvfp4", "amdfp4"):
        raise ValueError("Two-level module context requires NVFP4/AMDFP4 or BF16 override")
    merged = dict(vars(options)) if not isinstance(options, Mapping) else dict(options)
    merged["fp4_recipe"] = selected
    plan = resolve_fp4_plan(merged, module_kind)
    if plan.backend == "te":
        raise ValueError("TE NVFP4 must use TE modules, not the reference per-module override")
    plan.require_execution()
    return fp4_recipe_context(plan)
