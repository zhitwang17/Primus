# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU contract checks for the actual real-data comparison recipe generator."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load("_fp4_training_recipe", "tools/fp4_training_recipe.py")
PREFLIGHT = _load("_fp4_training_preflight", "tools/fp4_preflight.py")


def _build(recipe, **kwargs):
    result, manifest = GENERATOR.build_recipe(ROOT, recipe=recipe, **kwargs)
    return result, result["modules"]["pre_trainer"]["overrides"], manifest


@pytest.mark.parametrize("recipe", ["mxfp4", "nvfp4", "amdfp4"])
def test_full_model_real_data_uses_default_shell_schedule_and_bounded_exit(recipe):
    result, args, manifest = _build(recipe)
    assert result["modules"]["pre_trainer"]["model"] == "gpt_oss_20B.yaml"
    assert args["num_layers"] == 24 and args["hidden_size"] == 2880
    assert args["num_experts"] == 32 and args["moe_router_topk"] == 4
    assert args["seq_length"] == 8192
    assert args["micro_batch_size"] == 4 and args["global_batch_size"] == 32
    assert args["tensor_model_parallel_size"] == args["pipeline_model_parallel_size"] == 1
    assert args["expert_model_parallel_size"] == 1
    assert args["use_distributed_optimizer"] is True
    assert args["mock_data"] is False
    assert args["train_data_path"] == "10 /data/c4-train.en_6_text_document"
    assert args["valid_data_path"] == "/data/c4-validation-91205-samples.en_text_document"
    assert args["test_data_path"] == args["valid_data_path"]
    assert args["data_cache_path"] == "/training/data-cache"
    assert args["tokenizer_model"] == "/model"

    # train_iters determines the dataset index horizon as well as the LR horizon.
    # A bounded comparison must stop without regenerating a short-run permutation.
    assert args["train_iters"] == 1_200_000 and args["exit_interval"] == 3000
    assert args["lr_decay_iters"] == 1_199_883
    assert args["lr"] == pytest.approx(5.7e-4) and args["min_lr"] == pytest.approx(5.7e-5)
    assert args["lr_warmup_iters"] == 117
    assert args["eval_interval"] == 384 and args["eval_iters"] == 32
    assert manifest["validation_steps"] == [384, 768, 1152, 1536, 1920, 2304, 2688]
    assert 3000 not in manifest["validation_steps"]
    assert args["weight_deosc_start_step"] == 768 and args["weight_deosc_period"] == 100
    assert args["weight_deosc_accumulator_dtype"] == "fp32"
    assert args["exp_avg_dtype"] == args["exp_avg_sq_dtype"] == "fp32"
    assert args["save"] is args["load"] is None
    assert args["disable_last_saving"] is True and args["auto_continue_train"] is False
    assert args["recompute_granularity"] == "full" and args["recompute_num_layers"] == 1
    assert "approximate boundary shards" in manifest["deosc_boundary_difference"]
    assert manifest["deosc_settings"] == {
        "enabled": True,
        "start_step": 768,
        "period": 100,
        "ratio": args["weight_deosc_ratio"],
        "accumulator_dtype": "fp32",
        "epsilon": 1e-12,
    }


def test_all_arms_share_data_seed_schedule_attention_and_optimizer_contract():
    arms = [_build(recipe, seed=2026, steps=2500) for recipe in ("mxfp4", "nvfp4", "amdfp4")]
    shared = (
        "train_data_path",
        "valid_data_path",
        "test_data_path",
        "train_iters",
        "lr_decay_iters",
        "seed",
        "micro_batch_size",
        "global_batch_size",
        "seq_length",
        "lr",
        "min_lr",
        "lr_warmup_iters",
        "eval_interval",
        "eval_iters",
        "exit_interval",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "weight_decay",
        "clip_grad",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
        "use_precision_aware_optimizer",
        "gradient_accumulation_fusion",
        "use_turbo_attention",
        "enable_turbo_attention_float8",
        "use_sink_attention",
        "sink_sliding_window",
        "sink_window_even_layers_only",
        "weight_deosc_start_step",
        "weight_deosc_period",
    )
    expected = {key: arms[0][1][key] for key in shared}
    for _, args, manifest in arms:
        assert {key: args[key] for key in shared} == expected
        assert args["seed"] == manifest["seed"] == 2026
        assert manifest["environment"]["SEED"] == "2026"
        assert manifest["validation_steps"] == [384, 768, 1152, 1536, 1920, 2304]
        assert manifest["full_schedule_train_iters"] == 1_200_000
        environment = manifest["environment"]
        assert "PATH" not in environment
        assert environment["EXP"] == "/training/recipe.yaml"
        assert environment["MLLOG_OUTPUT_FILE"] == "/training/mlperf_logging.out"
        assert environment["SYNTH_WARMUP_STEPS"] == "0"
        assert float(environment["MLLOG_TARGET_EVAL_LOSS"]) == 3.34
        assert float(environment["TARGET_EVAL_LOSS"]) == 3.34
        assert manifest["early_stop_policy"]["target_reached_status"] == "converged"
        assert manifest["early_stop_policy"]["validation_steps_are_planned"] is True
        for key in (
            "PRIMUS_STATIC_BF16_LOSS_SCALE_CPU",
            "PRIMUS_FUSED_ADAM_MCORE_MASTER",
            "PRIMUS_FUSED_ADAM_CLIP",
            "PRIMUS_FUSED_ADAM_BF16_GRAD",
            "PRIMUS_FUSED_ADAM_BF16_WRITEBACK",
            "RCCL_CE_REDUCESCATTER",
            "RCCL_FORCE_CE_REDUCESCATTER",
            "RCCL_DDA_ENABLE",
            "NCCL_CUMEM_ENABLE",
            "NCCL_LOCAL_REGISTER",
        ):
            assert environment[key] == "0"
        assert environment["MEGATRON_PARAM_GATHER_BACKEND"] == ""
        assert environment["MEGATRON_GRAD_REDUCE_BACKEND"] == ""
        assert sorted(key for key in environment if key.startswith("RCCL_CE_")) == ["RCCL_CE_REDUCESCATTER"]
    for _, args, _ in arms[1:]:
        assert args["fp4_rng_seed"] == 2026
        assert args["fp4_sr_rng"] == "torch_cpu"
    assert arms[0][2]["environment_overrides"]["RCCL_CE_AR_STAGING_BYTES"] == {
        "before": "268435456",
        "after": None,
    }
    assert arms[0][2]["environment_overrides"]["PRIMUS_STATIC_BF16_LOSS_SCALE_CPU"] == {
        "before": "1",
        "after": "0",
    }


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_generated_two_level_recipe_passes_real_dense_and_grouped_runtime_guards(recipe):
    _, args, manifest = _build(recipe)
    _, dispatch = PREFLIGHT.load_contracts()
    plans = dispatch.validate_fp4_options(args, environ=manifest["environment"], require_runtime=True)
    assert [plan.module_kind for plan in plans] == ["dense", "grouped"]
    assert all(plan.backend == "reference" and plan.config.quantizer == "triton" for plan in plans)
    assert all(plan.config.recipe == recipe for plan in plans)
    assert args["use_turbo_gemm"] is False and args["use_turbo_grouped_gemm"] is False
    assert args["fp4_param"] is False and args["fp4_param_gather"] is False
    assert args["fp8"] is None
    # Prove these are active production guards, not permissive validation stubs.
    with pytest.raises(ValueError, match="PRIMUS_FUSED_RMSNORM_MXFP4_QKV"):
        dispatch.validate_fp4_options(
            args, environ={**manifest["environment"], "PRIMUS_FUSED_RMSNORM_MXFP4_QKV": "1"}
        )


def test_shell_defaults_are_isolated_from_other_runs_and_credentials(monkeypatch):
    monkeypatch.setenv("PRIMUS_LR", "123")
    monkeypatch.setenv("PRIMUS_TRAIN_ITERS", "17")
    monkeypatch.setenv("SEED", "99")
    monkeypatch.setenv("PRIMUS_FUSED_RMSNORM_MXFP4_QKV", "1")
    monkeypatch.setenv("HF_TOKEN", "unit-test-credential-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "another-unit-test-credential")
    _, args, manifest = _build("nvfp4")
    assert args["lr"] == pytest.approx(5.7e-4)
    assert args["train_iters"] == 1_200_000 and args["seed"] == 1234
    assert manifest["environment"]["PRIMUS_FUSED_RMSNORM_MXFP4_QKV"] == "0"
    assert "HF_TOKEN" not in manifest["environment"]
    assert "AWS_SECRET_ACCESS_KEY" not in manifest["environment"]
    assert "unit-test-credential" not in json.dumps(manifest)


def test_admission_changes_detector_window_and_eval_without_shortening_data_horizon():
    _, standard, _ = _build("amdfp4")
    _, admission, manifest = _build("amdfp4", steps=8, admission=True, recompute=True)
    assert manifest["scope"] == "admission_only"
    for key in ("train_iters", "lr_decay_iters", "seed", "train_data_path", "lr", "lr_warmup_iters"):
        assert admission[key] == standard[key]
    assert admission["exit_interval"] == 8 and admission["weight_deosc_start_step"] == 0
    assert admission["weight_deosc_period"] == 2
    assert manifest["deosc_settings"]["enabled"] is True
    assert manifest["deosc_settings"]["start_step"] == 0
    assert manifest["deosc_settings"]["period"] == 2
    assert manifest["deosc_settings"]["ratio"] == standard["weight_deosc_ratio"]
    assert admission["eval_iters"] == 1 and manifest["validation_steps"] == [4, 8]
    assert admission["recompute_granularity"] == "full"
    assert admission["recompute_method"] == "uniform" and admission["recompute_num_layers"] == 1


def test_gpu_rng_is_explicit_and_does_not_claim_an_mxfp4_rng_change():
    for recipe in ("nvfp4", "amdfp4"):
        _, args, manifest = _build(recipe, sr_rng="triton_philox")
        assert args["fp4_sr_rng"] == manifest["fp4_sr_rng"] == "triton_philox"
    _, args, manifest = _build("mxfp4", sr_rng="triton_philox")
    assert "fp4_sr_rng" not in args and manifest["fp4_sr_rng"] is None
    with pytest.raises(ValueError, match="sr_rng"):
        _build("nvfp4", sr_rng="typo")


def test_grouped_value_checks_are_explicit_and_do_not_change_mxfp4():
    for recipe in ("nvfp4", "amdfp4"):
        _, strict, default_manifest = _build(recipe)
        _, deferred, manifest = _build(recipe, value_checks="grouped_deferred")
        assert strict["fp4_value_checks"] == default_manifest["fp4_value_checks"] == "strict"
        assert deferred["fp4_value_checks"] == manifest["fp4_value_checks"] == "grouped_deferred"
        assert {k: v for k, v in strict.items() if k != "fp4_value_checks"} == {
            k: v for k, v in deferred.items() if k != "fp4_value_checks"
        }
    _, args, manifest = _build("mxfp4", value_checks="grouped_deferred")
    assert "fp4_value_checks" not in args and manifest["fp4_value_checks"] is None
    with pytest.raises(ValueError, match="value_checks"):
        _build("nvfp4", value_checks="unchecked")


@pytest.mark.parametrize("steps", [0, -1, 1_200_000])
def test_invalid_bounded_horizon_rejected(steps):
    with pytest.raises(ValueError):
        _build("nvfp4", steps=steps)


def test_cli_artifacts_replay_exact_rendered_recipe_and_provenance(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/fp4_training_recipe.py"),
            "--recipe",
            "nvfp4",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "2026",
            "--data-dir",
            "/task/c4",
            "--tokenizer-dir",
            "/task/tokenizer",
            "--sr-rng",
            "triton_philox",
            "--value-checks",
            "grouped_deferred",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = (tmp_path / "recipe.yaml").read_bytes()
    args = yaml.safe_load(rendered)["modules"]["pre_trainer"]["overrides"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    environment = dict(
        line.split("=", 1) for line in (tmp_path / "environment.list").read_text().splitlines()
    )
    assert environment == manifest["environment"]
    assert "PATH" not in environment
    assert manifest["rendered_recipe_sha256"] == hashlib.sha256(rendered).hexdigest()
    assert (
        manifest["source_template_sha256"]
        == hashlib.sha256((ROOT / manifest["source_template"]).read_bytes()).hexdigest()
    )
    assert "${" not in rendered.decode()
    assert args["train_data_path"] == "10 /task/c4/c4-train.en_6_text_document"
    assert args["tokenizer_model"] == "/task/tokenizer"
    assert args["seed"] == args["fp4_rng_seed"] == 2026
    assert args["fp4_sr_rng"] == manifest["fp4_sr_rng"] == "triton_philox"
    assert args["fp4_value_checks"] == manifest["fp4_value_checks"] == "grouped_deferred"
    assert manifest["changed_overrides"]["exit_interval"]["after"] == args["exit_interval"]
    assert manifest["changed_overrides"]["weight_deosc_fusion"] == {"before": True, "after": False}
