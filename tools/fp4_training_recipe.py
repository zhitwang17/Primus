#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Materialize matched real-data GPT-OSS FP4 training recipes and their provenance.

The default MLPerf shell supplies the numerical schedule. The comparison stops
at an optimizer-update boundary without shrinking that schedule or rebuilding
the data index for a different training horizon. MX-control explicitly disables
format-specific fusions; it is not a performance claim against MX-best.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_RECIPES = ("mxfp4", "nvfp4", "amdfp4")


def expand(value, environment):
    if isinstance(value, dict):
        return {key: expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replacement(match):
        name, default = match.groups()
        if name not in environment and default is None:
            raise ValueError(f"Unresolved recipe variable: {name}")
        return environment.get(name, default or "")

    expanded = _VARIABLE.sub(replacement, value)
    return yaml.safe_load(expanded) if _VARIABLE.fullmatch(value) else expanded


def shell_defaults(source_root, runtime_primus, recipe):
    source = source_root / "examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1.sh"
    # A clean, non-secret environment avoids inheriting an unrelated experiment
    # override or storing credentials in a replay manifest.
    environment = {
        "PATH": os.defpath,
        "USER": "zhitwang",
        "LOGNAME": "zhitwang",
        "PRIMUS_PATH": runtime_primus,
        "PRIMUS_LINEAR_PRECISION": recipe,
    }
    output = subprocess.check_output(
        ["bash", "-c", 'set -e; source "$1"; env -0', "fp4-recipe", str(source)],
        env=environment,
    )
    keys = set(re.findall(r"\bexport\s+([A-Z][A-Z0-9_]*)", source.read_text()))
    keys.update(environment)
    result = {}
    for entry in output.decode().split("\0"):
        key, separator, val = entry.partition("=")
        if separator and key in keys:
            result[key] = val
    return result


def build_recipe(
    source_root,
    *,
    recipe,
    steps=3000,
    seed=1234,
    runtime_primus="/workspace/Primus",
    data_dir="/data",
    tokenizer_dir="/model",
    workspace="/training/output",
    recompute=True,
    admission=False,
    sr_rng="torch_cpu",
):
    if recipe not in _RECIPES or type(steps) is not int or steps <= 0:
        raise ValueError("Expected mxfp4/nvfp4/amdfp4 and a positive optimizer-update limit")
    if sr_rng not in ("torch_cpu", "triton_philox"):
        raise ValueError("Expected sr_rng=torch_cpu or triton_philox")
    source_root = Path(source_root)
    base = (
        source_root / "examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml"
    )
    environment = shell_defaults(source_root, runtime_primus, recipe)
    source_environment = environment.copy()
    # os.defpath was only for sourcing the shell. The training image owns PATH
    # (including its /opt/venv/bin); exporting the host path would hide torchrun.
    environment.pop("PATH", None)
    # Remove MX-only copy-engine tuning; the common stock communicator does
    # not consume these settings, and the replay environment should not imply it.
    for key in tuple(environment):
        if key.startswith("RCCL_CE_"):
            environment.pop(key)
    environment.update(
        EXP="/training/recipe.yaml",
        SEED=str(seed),
        PRIMUS_TOKENIZER_MODEL=tokenizer_dir,
        PRIMUS_WORKSPACE=workspace,
        PRIMUS_REBUILD_RCCL="0",
        MEGATRON_PARAM_GATHER_BACKEND="",
        MEGATRON_GRAD_REDUCE_BACKEND="",
        MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES="0",
        PRIMUS_USE_PRECISION_AWARE_OPTIMIZER="false",
        PRIMUS_EXP_AVG_DTYPE="fp32",
        PRIMUS_EXP_AVG_SQ_DTYPE="fp32",
        PRIMUS_FUSED_ADAM_MCORE_MASTER="0",
        PRIMUS_FUSED_ADAM_CLIP="0",
        PRIMUS_FUSED_ADAM_BF16_GRAD="0",
        PRIMUS_FUSED_ADAM_BF16_WRITEBACK="0",
        PRIMUS_STATIC_BF16_LOSS_SCALE_CPU="0",
        PRIMUS_FUSED_RMSNORM_MXFP4_QKV="0",
        PRIMUS_FUSED_RESIDUAL_NORM="0",
        PRIMUS_FUSED_RESIDUAL_NORM_V2="0",
        PRIMUS_TURBO_GRAD_OWNERSHIP="0",
        PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT="0",
        PRIMUS_TURBO_FUSED_WGRAD_ACCUM="0",
        PRIMUS_TURBO_AUTO_TUNE="0",
        RCCL_CE_REDUCESCATTER="0",
        RCCL_FORCE_CE_REDUCESCATTER="0",
        RCCL_DDA_ENABLE="0",
        NCCL_CUMEM_ENABLE="0",
        NCCL_LOCAL_REGISTER="0",
        WANDB_MODE="disabled",
        LOG_INTERVAL="10",
        MLLOG_TRAIN_LOSS_LOG_FREQ="10",
        MLLOG_SUBMISSION_DIVISION="open",
        MLLOG_OUTPUT_FILE="/training/mlperf_logging.out",
        # The installed MLPerf warmup restores TE state, but not reference FP4
        # private RNG counters or sharded DeOsc history. Do not count synthetic
        # updates as training or let them affect the real-data trajectory.
        SYNTH_WARMUP_STEPS="0",
    )
    original = expand(yaml.safe_load(base.read_text()), environment)
    result = copy.deepcopy(original)
    result.update(
        exp_name=f"fp4_{recipe}_{'admission' if admission else 'control'}_{steps}", workspace=workspace
    )
    overrides = result["modules"]["pre_trainer"]["overrides"]
    changes = dict(
        fp4_recipe=recipe,
        exit_interval=steps,
        seed=seed,
        tokenizer_model=tokenizer_dir,
        train_data_path=f"10 {data_dir}/c4-train.en_6_text_document",
        valid_data_path=f"{data_dir}/c4-validation-91205-samples.en_text_document",
        test_data_path=f"{data_dir}/c4-validation-91205-samples.en_text_document",
        data_cache_path="/training/data-cache",
        mock_data=False,
        wandb_project=None,
        save=None,
        load=None,
        auto_continue_train=False,
        disable_last_saving=True,
        ckpt_format="torch_dist",
        check_for_nan_in_loss_and_grad=True,
        gradient_accumulation_fusion=False,
        use_precision_aware_optimizer=False,
        exp_avg_dtype="fp32",
        exp_avg_sq_dtype="fp32",
        turbo_fused_grouped_gemm=False,
        moe_permute_quant_fusion=False,
        moe_backward_permute_quant_fusion=False,
        moe_router_grad_merge=False,
        moe_router_bwd_bf16=False,
        weight_deosc=True,
        weight_deosc_fusion=False,
        weight_deosc_direct_qdq=False,
        weight_deosc_accumulator_dtype="fp32",
        weight_deosc_epsilon=1e-12,
        weight_deosc_log_freq=1,
    )
    if recipe != "mxfp4":
        changes.update(
            fp4_backend="reference",
            fp4_quantizer="triton",
            fp4_block_size=16,
            fp4_weight_2d=True,
            fp4_outer_scope="tensor",
            fp4_inner_storage="float32_grid",
            fp4_rht_size=32,
            fp4_rht_randomized=True,
            fp4_gradient_sr=True,
            fp4_rng_seed=seed,
            fp4_sr_rng=sr_rng,
            fp4_param=False,
            fp4_param_gather=False,
            use_turbo_gemm=False,
            use_turbo_grouped_gemm=False,
            turbo_grouped_gemm_without_padding=False,
        )
    if recompute:
        changes.update(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    if admission:
        changes.update(weight_deosc_start_step=0, weight_deosc_period=2, eval_interval=4, eval_iters=1)
    overrides.update(changes)
    if overrides["train_iters"] <= steps:
        raise ValueError("Bounded comparison must retain the original full training horizon")
    manifest = {
        "schema_version": 1,
        "recipe": recipe,
        "scope": "admission_only" if admission else "MX-control_comparison",
        "source_template": str(base.relative_to(source_root)),
        "source_template_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "source_shell_sha256": hashlib.sha256(
            (source_root / "examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1.sh").read_bytes()
        ).hexdigest(),
        "optimizer_update_limit": steps,
        "full_schedule_train_iters": overrides["train_iters"],
        "lr_decay_iters": overrides["lr_decay_iters"],
        "seed": seed,
        "changed_overrides": {
            key: {"before": original["modules"]["pre_trainer"]["overrides"].get(key), "after": val}
            for key, val in changes.items()
        },
        "comparison_boundary": "Matched seed, data-index horizon, effective batch, LR schedule and evaluation steps; recipe comparison, not isolated format attribution",
        "deosc_boundary_difference": "Existing MXFP4 detector may approximate boundary shards; two-level detector uses exact global outer and boundary tile reconstruction",
        "checkpoint_policy": "Full-model checkpoints disabled for bounded runs; separate real-MCore smoke validates checkpoint continuation",
        "early_stop_policy": {
            "target_eval_loss": float(environment["MLLOG_TARGET_EVAL_LOSS"]),
            "target_reached_status": "converged",
            "bounded_exit_status": "completed_update_limit",
            "validation_steps_are_planned": True,
        },
        "synthetic_warmup": "disabled: installed MLPerf warmup does not restore reference FP4 private RNG or DeOsc history",
        "validation_steps": list(
            range(int(overrides["eval_interval"]), steps + 1, int(overrides["eval_interval"]))
        ),
        "recompute": recompute,
        "fp4_sr_rng": sr_rng if recipe != "mxfp4" else None,
        "environment_overrides": {
            key: {"before": source_environment.get(key), "after": environment.get(key)}
            for key in sorted(source_environment.keys() | environment.keys())
            if source_environment.get(key) != environment.get(key)
        },
        "environment": environment,
    }
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--recipe", choices=_RECIPES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--tokenizer-dir", default="/model")
    parser.add_argument("--recompute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--admission", action="store_true")
    parser.add_argument("--sr-rng", choices=("torch_cpu", "triton_philox"), default="torch_cpu")
    args = parser.parse_args()
    recipe, manifest = build_recipe(
        args.source_root,
        recipe=args.recipe,
        steps=args.steps,
        seed=args.seed,
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        recompute=args.recompute,
        admission=args.admission,
        sr_rng=args.sr_rng,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = args.output_dir / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    manifest["rendered_recipe_sha256"] = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "environment.list").write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(manifest["environment"].items()))
    )
    print(json.dumps({"recipe": str(recipe_path), "manifest": str(args.output_dir / "manifest.json")}))


if __name__ == "__main__":
    main()
