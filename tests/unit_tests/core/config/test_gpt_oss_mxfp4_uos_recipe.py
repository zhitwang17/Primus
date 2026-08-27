###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path

from primus.core.config.yaml_loader import parse_yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE = (
    REPO_ROOT
    / "examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-UOS2-training-preflight.yaml"
)


def test_gpt_oss_mxfp4_uos2_recipe_resolves_target_training_contract():
    config = parse_yaml(str(RECIPE))
    overrides = config["modules"]["pre_trainer"]["overrides"]

    assert config["env"]["NVTE_MXFP4_USE_HADAMARD"] == "1"

    expected = {
        "fp8": None,
        "fp4": "e2m1",
        "fp4_recipe": "mxfp4",
        "first_last_layers_bf16": False,
        "num_layers_at_start_in_bf16": 0,
        "num_layers_at_end_in_bf16": 0,
        "mxfp4_gradient_stochastic_rounding": True,
        "mxfp4_scale_rounding_mode": 2,
        "train_iters": 1_200_000,
        "lr": 8.0e-4,
        "min_lr": 4.0e-5,
        "lr_warmup_iters": 128,
        "lr_decay_iters": 1_199_872,
        "lr_decay_style": "cosine",
        "init_method_std": 0.008,
        "micro_batch_size": 4,
        "global_batch_size": 32,
        "seq_length": 8192,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "none",
        "moe_aux_loss_coeff": 0.0,
        "moe_router_enable_expert_bias": False,
        "weight_deosc": True,
        "weight_deosc_start_step": 1000,
        "weight_deosc_period": 100,
        "weight_deosc_ratio": 3.0,
        "enable_primus_turbo": True,
        "use_turbo_gemm": True,
        "use_turbo_grouped_gemm": True,
        "turbo_gemm_backend": "fp4:hipblaslt",
        "turbo_grouped_gemm_backend": "fp4:flydsl",
    }

    assert {key: overrides[key] for key in expected} == expected


def test_gpt_oss_mxfp4_uos2_recipe_inherits_model_and_data_contract():
    config = parse_yaml(str(RECIPE))
    pre_trainer = config["modules"]["pre_trainer"]
    overrides = pre_trainer["overrides"]

    assert pre_trainer["framework"] == "megatron"
    assert pre_trainer["model"] == "gpt_oss_20B.yaml"
    assert overrides["num_layers"] == 24
    assert overrides["hidden_size"] == 2880
    assert overrides["num_experts"] == 32
    assert overrides["train_data_path"]
    assert overrides["valid_data_path"]
