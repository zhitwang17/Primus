# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Megatron TP1/EP1 modules for the executable two-level FP4 reference backend.

Parameters remain ordinary tensors. QDQ and all three GEMM directions live in
``fp4_autograd``; normal autograd/DDP hooks accumulate ``main_grad``. These
modules deliberately have no TE/Turbo quantized-weight cache or native FP4 GEMM.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule

from primus.backends.megatron.core.fp4_autograd import grouped_linear, linear
from primus.backends.megatron.core.fp4_contract import _name
from primus.backends.megatron.core.fp4_dispatch import (
    build_two_level_config,
    comparison_rmsnorm_backend,
    current_fp4_plan,
    module_fp4_context,
)


def _options(config):
    try:
        from megatron.training.global_vars import get_args

        result = dict(vars(get_args()))
    except (AssertionError, ImportError):
        result = {}
    result.update(vars(config))
    # TransformerConfig does not declare the Primus-only backend selector.
    result.setdefault("fp4_backend", "reference")
    return result


def _check_topology(config, group=None):
    for key in ("tensor_model_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size"):
        if getattr(config, key, 1) not in (None, 1):
            raise ValueError(f"Two-level FP4 reference modules require {key}=1")
    if group is not None and group.size() != 1:
        raise ValueError("Two-level FP4 reference modules cannot perform TP/EP collectives")
    if getattr(config, "sequence_parallel", False):
        raise ValueError("Two-level FP4 reference modules require sequence_parallel=false")
    if getattr(config, "delay_wgrad_compute", False):
        raise ValueError("Two-level FP4 reference modules do not implement delayed wgrad")
    if getattr(config, "init_model_with_meta_device", False):
        raise ValueError("Two-level FP4 reference modules do not implement meta-device initialization")


def _parameter(
    shape,
    config,
    init_method,
    *,
    is_expert=False,
    partition_dim=0,
    stride=1,
    expert_slices=False,
    cpu_master=True,
    track_rng=True,
):
    device = "cpu" if getattr(config, "use_cpu_initialization", False) else torch.cuda.current_device()
    parameter = torch.nn.Parameter(torch.empty(shape, dtype=config.params_dtype, device=device))
    parameter.allreduce = True  # TP1/EP1: expert DP and ordinary DP are the same group.
    parameter.tensor_model_parallel = True
    parameter.partition_dim = partition_dim
    parameter.partition_stride = stride
    if getattr(config, "perform_initialization", True):
        with torch.no_grad():
            # TE initializes each weight{i} separately, in params_dtype, with
            # a separate expert RNG fork; Turbo only stacks them afterwards.
            # A single normal_(whole_bank) changes launch/counter consumption.
            # Views preserve TE's call shapes without a second complete bank.
            for target in parameter.unbind(0) if expert_slices else (parameter,):
                if getattr(config, "use_cpu_initialization", False) and cpu_master:
                    master = torch.empty(target.shape, dtype=torch.float32, device="cpu")
                    init_method(master)
                    target.copy_(master)
                else:
                    tracker = (
                        get_cuda_rng_tracker()
                        if track_rng and not getattr(config, "use_cpu_initialization", False)
                        else None
                    )
                    context = nullcontext()
                    if tracker is not None and tracker.is_initialized():
                        context = (
                            tracker.fork(get_expert_parallel_rng_tracker_name())
                            if is_expert
                            else tracker.fork()
                        )
                    with context:
                        init_method(target)
    return parameter


class _FP4Module(MegatronModule):
    """Recipe selection and explicitly bound, per-module RNG state."""

    def _init_fp4(self, config, name):
        self._fp4_options = _options(config)
        self._base_fp4_config = build_two_level_config(self._fp4_options)
        self.fp4_config = self._base_fp4_config
        self.fp4_configs = (self.fp4_config,)
        self.fp4_rng = None
        self.fp4_rngs = {}
        self._primus_fp4_last_config = None
        self.te_quant_params = None
        self.is_first_microbatch = True
        self.finish_init(get_quant_config_or_none(name, getattr(config, "quant_recipe", None)))

    def finish_init(self, quantization_config):
        if quantization_config is None:
            self.te_quant_params = None
            self.fp4_config = self._base_fp4_config
            self.fp4_configs = (self.fp4_config,)
            return
        from megatron.core.extensions.transformer_engine import TEQuantizationParams

        self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)
        self.fp4_config = self._base_fp4_config
        configs = {self._base_fp4_config.fingerprint: self._base_fp4_config}
        for recipe in (self.te_quant_params.training_recipe, self.te_quant_params.evaluation_recipe):
            if recipe is None:
                continue
            if recipe.fp8_quantization_recipe is not None or recipe.fp4_param or recipe.fp8_param:
                raise ValueError("Reference FP4 modules support two-level FP4/BF16 overrides only")
            selected = _name(recipe.fp4_quantization_recipe)
            if selected is not None:
                if selected not in ("nvfp4", "amdfp4"):
                    raise ValueError(f"Unsupported reference module recipe override: {selected}")
                config = replace(self._base_fp4_config, recipe=selected)
                configs[config.fingerprint] = config
                if recipe is self.te_quant_params.training_recipe and recipe.override_quantized_autocast:
                    self.fp4_config = config
        self.fp4_configs = tuple(configs.values())

    def fp4_routing_identity(self):
        """Describe precision roles, not just the unordered set of recipes.

        Checkpoint continuation must distinguish exchanged training/evaluation
        recipes and BF16 override flags even when the available RNG set is the
        same. Enum/string forms normalize to the same serialized identity.
        """

        def recipe_identity(recipe):
            if recipe is None:
                return None
            return {key: _name(value) for key, value in vars(recipe).items()}

        return {
            "base_recipe": self._base_fp4_config.fingerprint,
            "training_config": self.fp4_config.fingerprint,
            "training_override": recipe_identity(
                None if self.te_quant_params is None else self.te_quant_params.training_recipe
            ),
            "evaluation_override": recipe_identity(
                None if self.te_quant_params is None else self.te_quant_params.evaluation_recipe
            ),
        }

    def _quant_context(self, kind):
        if self.te_quant_params is None:
            return nullcontext()
        recipe = self.te_quant_params.training_recipe
        if not self.training and self.te_quant_params.evaluation_recipe is not None:
            recipe = self.te_quant_params.evaluation_recipe
        return module_fp4_context(recipe, self._fp4_options, kind)

    def will_execute_quantized(self, is_context_quantized):
        if self.te_quant_params is None:
            return is_context_quantized
        recipe = self.te_quant_params.training_recipe
        if not self.training and self.te_quant_params.evaluation_recipe is not None:
            recipe = self.te_quant_params.evaluation_recipe
        flag = "override_quantized_autocast" if is_context_quantized else "override_nonquantized_autocast"
        if not getattr(recipe, flag):
            return is_context_quantized
        return recipe.fp4_quantization_recipe is not None

    def _plan_and_random(self, parameter):
        plan = current_fp4_plan()
        if plan is None:
            self._primus_fp4_last_config = None
            return None, None
        if plan.backend != "reference":
            raise ValueError("Reference modules cannot execute the TE/Turbo FP4 backend")
        random = self.fp4_rngs.get(plan.config.fingerprint)
        if random is None and self.fp4_rng is not None:
            if self.fp4_rng.recipe_fingerprint == plan.config.fingerprint:
                random = self.fp4_rng
        self._primus_fp4_last_config = plan.config
        parameter._primus_two_level_fp4_config = plan.config
        return plan, random

    def backward_dw(self):
        # Ordinary autograd already produced wgrad and fired parameter hooks.
        if getattr(self.config, "delay_wgrad_compute", False):
            raise ValueError("Reference FP4 does not support delayed wgrad")


class TwoLevelFP4Linear(_FP4Module):
    """Dense linear with the Megatron (output, deferred_bias) interface."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias=True,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        parallel_mode=None,
        is_expert=False,
        tp_group=None,
        name=None,
        stride=1,
        gather_output=False,
        input_is_parallel=True,
        tp_comm_buffer_name=None,
        symmetric_ar_type=None,
        keep_master_weight_for_test=False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        disable_grad_reduce=False,
        gtp_remat_group=None,
    ):
        super().__init__(config)
        _check_topology(config, tp_group)
        if skip_weight_param_allocation:
            raise ValueError("Reference FP4 does not support external/tied weight allocation")
        if symmetric_ar_type is not None or gtp_remat_group is not None:
            raise ValueError("Reference FP4 does not support symmetric or rematerialized TP")
        if embedding_activation_buffer is not None or grad_output_buffer is not None:
            raise ValueError("Reference FP4 does not support deferred embedding wgrad")
        if disable_grad_reduce:
            raise ValueError("Reference FP4 requires ordinary DDP gradient reduction")
        self.in_features, self.out_features = input_size, output_size
        self.input_size, self.output_size = input_size, output_size
        self.parallel_mode = parallel_mode
        self.tp_group = tp_group or parallel_state.get_tensor_model_parallel_group(check_initialized=False)
        self._tp_group, self.tp_size = self.tp_group, 1
        self.skip_bias_add, self.use_bias = skip_bias_add, bias
        self._init_before_linear_parameters(input_size, config)
        partition_dim = 1 if parallel_mode == "row" else 0
        self.weight = _parameter(
            (output_size, input_size),
            config,
            init_method,
            is_expert=is_expert,
            partition_dim=partition_dim,
            stride=stride,
        )
        self.bias = None
        if bias:
            self.bias = _parameter(
                (output_size,), config, torch.nn.init.zeros_, partition_dim=0, track_rng=False
            )
            self.bias.tensor_model_parallel = parallel_mode != "row"
        self._init_fp4(config, name)

    def _init_before_linear_parameters(self, input_size, config):
        """Combined norm/linear subclasses register the constant norm first."""

    def forward(self, x, weight=None, runtime_gather_output=None):
        if weight is not None:
            raise ValueError("Reference FP4 does not support external/tied weights")
        with self._quant_context("dense"):
            plan, random = self._plan_and_random(self.weight)
            output = F.linear(x, self.weight) if plan is None else linear(x, self.weight, plan.config, random)
            if self.bias is not None and not self.skip_bias_add:
                output = output + self.bias
        self.is_first_microbatch = False
        return output, self.bias if self.skip_bias_add else None


class TwoLevelFP4ColumnParallelLinear(TwoLevelFP4Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, parallel_mode="column", **kwargs)


class TwoLevelFP4RowParallelLinear(TwoLevelFP4Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, parallel_mode="row", **kwargs)


class TwoLevelFP4LayerNormColumnParallelLinear(TwoLevelFP4ColumnParallelLinear):
    """TE's combined spec slot, implemented as separate Torch norm and GEMM.

    The upstream TE GPT spec relies on this slot to contain the norm. Returning an
    ordinary linear here would remove the QKV/pre-MLP norm from the model.
    """

    def _init_before_linear_parameters(self, input_size, config):
        self.fp4_comparison_rmsnorm_backend = comparison_rmsnorm_backend(_options(config))
        self.fp4_comparison_rmsnorm_calls = 0
        self.normalization = getattr(config, "normalization", "LayerNorm")
        if self.normalization not in ("LayerNorm", "RMSNorm"):
            raise ValueError(f"Unsupported reference normalization: {self.normalization}")
        self.layer_norm_epsilon = getattr(config, "layernorm_epsilon", 1e-5)
        self.zero_centered_gamma = getattr(config, "layernorm_zero_centered_gamma", False)
        device = "cpu" if getattr(config, "use_cpu_initialization", False) else torch.cuda.current_device()
        self.layer_norm_weight = torch.nn.Parameter(
            torch.full(
                (input_size,),
                0 if self.zero_centered_gamma else 1,
                dtype=config.params_dtype,
                device=device,
            )
        )
        self.layer_norm_weight.allreduce = True
        self.layer_norm_weight.tensor_model_parallel = False
        self.layer_norm_bias = None
        if self.normalization == "LayerNorm":
            self.layer_norm_bias = torch.nn.Parameter(torch.zeros_like(self.layer_norm_weight))
            self.layer_norm_bias.allreduce = True
            self.layer_norm_bias.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        gamma = self.layer_norm_weight + 1 if self.zero_centered_gamma else self.layer_norm_weight
        if self.normalization == "RMSNorm":
            if self.fp4_comparison_rmsnorm_backend == "torch":
                from primus.backends.megatron.core.fp4_norm import comparison_rmsnorm

                normalized = comparison_rmsnorm(x, gamma, self.layer_norm_epsilon)
                self.fp4_comparison_rmsnorm_calls += 1
            else:
                normalized = x.float() * torch.rsqrt(
                    x.float().square().mean(-1, keepdim=True) + self.layer_norm_epsilon
                )
                normalized = (normalized * gamma.float()).to(x.dtype)
        else:
            normalized = F.layer_norm(
                x.float(),
                (self.in_features,),
                gamma.float(),
                self.layer_norm_bias.float(),
                self.layer_norm_epsilon,
            ).to(x.dtype)
        return super().forward(normalized, **kwargs)


class TwoLevelFP4GroupedLinear(_FP4Module):
    """One ordinary [experts,out,in] Parameter and per-expert token lengths."""

    def __init__(
        self,
        num_gemms,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias=False,
        skip_bias_add=False,
        parallel_mode=None,
        is_expert=True,
        tp_comm_buffer_name=None,
        pg_collection=None,
        name=None,
    ):
        super().__init__(config)
        group = getattr(pg_collection, "expt_tp", None)
        _check_topology(config, group)
        if num_gemms <= 0:
            raise ValueError("Grouped FP4 requires at least one local expert")
        self.num_gemms, self.in_features, self.out_features = num_gemms, input_size, output_size
        self.parallel_mode, self.tp_group = parallel_mode, group
        self._tp_group, self.tp_size = group, 1
        self.use_bias, self.skip_bias_add = bias, skip_bias_add
        self.weights = _parameter(
            (num_gemms, output_size, input_size),
            config,
            init_method,
            is_expert=is_expert,
            partition_dim=2 if parallel_mode == "row" else 1,
            expert_slices=True,
            cpu_master=False,
        )
        self.bias = None
        if bias:
            self.bias = _parameter(
                (num_gemms, output_size),
                config,
                torch.nn.init.zeros_,
                is_expert=is_expert,
                partition_dim=1,
                expert_slices=True,
                cpu_master=False,
                track_rng=False,
            )
        self._init_fp4(config, name)

    def forward(self, x, m_splits):
        from primus.backends.megatron.core.fp4_autograd import _lengths

        lengths, _, _ = _lengths(m_splits, x.shape[0], self.num_gemms)
        with self._quant_context("grouped"):
            plan, random = self._plan_and_random(self.weights)
            if plan is None:
                output = torch.cat(
                    [F.linear(part, self.weights[i]) for i, part in enumerate(x.split(lengths))], dim=0
                )
            else:
                output = grouped_linear(x, self.weights, lengths, plan.config, random)
            if self.bias is not None and not self.skip_bias_add:
                output = torch.cat([part + self.bias[i] for i, part in enumerate(output.split(lengths))])
        self.is_first_microbatch = False
        return output, self.bias if self.skip_bias_add else None


class TwoLevelFP4ColumnParallelGroupedLinear(TwoLevelFP4GroupedLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, parallel_mode="column", **kwargs)


class TwoLevelFP4RowParallelGroupedLinear(TwoLevelFP4GroupedLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, parallel_mode="row", **kwargs)


class TwoLevelFP4GroupedMLP(MegatronModule):
    """Reference MoE MLP preserving eager GLU, clamp, offset and router gradients.

    Expert padding is inside grouped_linear, so neither TE's FP8 padding nor a
    global total-token alignment is allowed to mix experts' RHT/scale blocks.
    """

    def __init__(self, num_local_experts, config, submodules=None, pg_collection=None, name=None):
        super().__init__(config)
        _check_topology(config, getattr(pg_collection, "ep", None))
        if getattr(config, "fine_grained_activation_offloading", False):
            raise ValueError("Reference FP4 MLP does not implement activation offloading")
        if getattr(config, "use_transformer_engine_op_fuser", False):
            raise ValueError("Reference FP4 MLP does not implement TE op fusion")
        self.num_local_experts = num_local_experts
        self.tp_group = getattr(pg_collection, "expt_tp", None)
        self.ep_group = getattr(pg_collection, "ep", None)
        hidden = getattr(config, "moe_latent_size", None) or config.hidden_size
        width = getattr(config, "moe_ffn_hidden_size", None) or config.ffn_hidden_size
        fc1 = TwoLevelFP4ColumnParallelGroupedLinear if submodules is None else submodules.linear_fc1
        fc2 = TwoLevelFP4RowParallelGroupedLinear if submodules is None else submodules.linear_fc2
        self.linear_fc1 = fc1(
            num_local_experts,
            hidden,
            width * (2 if config.gated_linear_unit else 1),
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            pg_collection=pg_collection,
            name=f"{name}.linear_fc1" if name else None,
        )
        self.linear_fc2 = fc2(
            num_local_experts,
            width,
            hidden,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            pg_collection=pg_collection,
            name=f"{name}.linear_fc2" if name else None,
        )

    def _activate(self, x, probs):
        if self.config.gated_linear_unit:
            interleave = getattr(self.config, "moe_mlp_glu_interleave_size", None)
            if interleave is not None:
                if interleave <= 0 or x.shape[-1] % (2 * interleave):
                    raise ValueError("Invalid MoE GLU interleave size")
                x = x.reshape(*x.shape[:-1], -1, 2, interleave).transpose(-3, -2).reshape_as(x)
            gate, up = x.chunk(2, dim=-1)
            limit = getattr(self.config, "activation_func_clamp_value", None)
            if limit is not None:
                gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
            x = self.config.activation_func(gate) * (up + getattr(self.config, "glu_linear_offset", 0.0))
        else:
            x = self.config.activation_func(x)
        return (x * probs).to(x.dtype)

    def forward(
        self,
        permuted_local_hidden_states,
        tokens_per_expert,
        permuted_probs,
        output_buffer=None,
        grad_input_buffer=None,
    ):
        if output_buffer is not None or grad_input_buffer is not None:
            raise ValueError("Reference FP4 MLP does not consume TE op-fuser buffers")
        x = permuted_local_hidden_states
        probs = permuted_probs.reshape(-1, 1)
        if probs.shape[0] != x.shape[0]:
            raise ValueError("Router probabilities must match routed token rows")
        if getattr(self.config, "moe_apply_probs_on_input", False):
            if self.config.moe_router_topk != 1:
                raise ValueError("moe_apply_probs_on_input requires moe_router_topk=1")
            x = (x * probs).to(x.dtype)
            probs = torch.ones_like(probs)
        intermediate, _ = self.linear_fc1(x, tokens_per_expert)
        if getattr(self.config, "recompute_granularity", None) == "selective" and "moe_act" in (
            getattr(self.config, "recompute_modules", None) or []
        ):
            from torch.utils.checkpoint import checkpoint

            activated = checkpoint(self._activate, intermediate, probs, use_reentrant=False)
        else:
            activated = self._activate(intermediate, probs)
        output, bias = self.linear_fc2(activated, tokens_per_expert)
        if bias is not None:
            from primus.backends.megatron.core.fp4_autograd import _lengths

            lengths, _, _ = _lengths(tokens_per_expert, output.shape[0], self.num_local_experts)
            # FC2 bias is scaled by router probability, just like the FC2 output.
            output = torch.cat(
                [part + b * p for part, b, p in zip(output.split(lengths), bias, probs.split(lengths))]
            ).to(output.dtype)
        return output, None

    def backward_dw(self):
        self.linear_fc1.backward_dw()
        self.linear_fc2.backward_dw()


# Provider imports stay below the independent module definitions. They supply
# the existing attention/router/normalization choices, never FP4 GEMM execution.
from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (  # noqa: E402
    _MOE_TAKES_EXPERTS_BUILDER,
    PrimusTurboSpecProvider,
    TEGroupedMLPSubmodules,
    _experts_builder,
)


class TwoLevelFP4SpecProvider(PrimusTurboSpecProvider):
    def linear(self):
        return TwoLevelFP4Linear

    def column_parallel_linear(self):
        return TwoLevelFP4ColumnParallelLinear

    def row_parallel_linear(self):
        return TwoLevelFP4RowParallelLinear

    def column_parallel_linear_with_gather_output(self):
        return TwoLevelFP4ColumnParallelLinear

    def row_parallel_linear_with_scatter_input(self):
        return TwoLevelFP4RowParallelLinear

    def fuse_layernorm_and_linear(self):
        # This describes the spec slot, not a fused kernel: removing the norm
        # here would drop dense pre-MLP normalization in the upstream TE spec.
        return True

    def column_parallel_layer_norm_linear(self):
        return TwoLevelFP4LayerNormColumnParallelLinear

    def grouped_mlp_modules(self, moe_use_grouped_gemm, moe_use_legacy_grouped_gemm=None):
        if moe_use_legacy_grouped_gemm:
            raise ValueError("Reference FP4 does not use the legacy grouped GEMM implementation")
        if not moe_use_grouped_gemm:
            raise ValueError("Reference two-level FP4 MoE requires moe_grouped_gemm=true")
        submodules = TEGroupedMLPSubmodules(
            linear_fc1=TwoLevelFP4ColumnParallelGroupedLinear,
            linear_fc2=TwoLevelFP4RowParallelGroupedLinear,
        )
        if _MOE_TAKES_EXPERTS_BUILDER:
            return _experts_builder(TwoLevelFP4GroupedMLP, submodules)
        return TwoLevelFP4GroupedMLP, submodules
