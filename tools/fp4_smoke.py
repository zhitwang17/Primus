# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Single-GPU, reduced Megatron GPTModel smoke for two-level FP4 adapters.

This exercises real model/optimizer/DDP paths on synthetic tokens. It is not a
convergence or performance benchmark, and does not use FlyDSL attention. Expose
exactly one GPU to the process before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("nvfp4", "amdfp4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-length", type=int, choices=(32, 64), default=64)
    parser.add_argument("--glu-clamp", type=float, default=None)
    parser.add_argument("--glu-offset", type=float, default=0.0)
    parser.add_argument("--deosc", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="Exercise full-layer activation recompute")
    args = parser.parse_args()
    if args.steps < 2 or (args.deosc and args.steps < 4):
        parser.error("checkpoint smoke needs at least two updates; DeOsc needs at least four")
    if args.glu_clamp is not None and args.glu_clamp <= 0:
        parser.error("glu-clamp must be positive")
    return args


def _make_config(args):
    import torch
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    from primus.backends.megatron.core.enums import Fp4Recipe

    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=64,
        ffn_hidden_size=128,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_layer_freq=[0, 1],
        moe_router_topk=2,
        moe_router_pre_softmax=False,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_fusion=False,
        moe_permute_fusion=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=args.glu_clamp,
        glu_linear_offset=args.glu_offset,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        bias_dropout_fusion=False,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        apply_rope_fusion=False,
        gradient_accumulation_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        fp4="e2m1",
        fp4_recipe=Fp4Recipe(args.recipe),
        fp4_param=False,
        fp8=None,
        recompute_granularity="full" if args.recompute else None,
        recompute_method="uniform" if args.recompute else None,
        recompute_num_layers=1 if args.recompute else None,
    )
    # Primus options are not dataclass fields in upstream TransformerConfig.
    config.fp4_backend = "reference"
    config.fp4_rng_seed = args.seed
    config.enable_primus_turbo = False
    config.use_turbo_gemm = False
    config.use_turbo_grouped_gemm = False
    return config


def _gradient_summary(model):
    import torch

    router_norms = {}
    grads = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not hasattr(param, "main_grad"):
            raise AssertionError(f"Megatron DDP did not allocate main_grad for {name}")
        grad = param.main_grad
        if not torch.isfinite(grad).all():
            raise AssertionError(f"Non-finite main_grad: {name}")
        if param.grad is not None:
            raise AssertionError(f"DDP gradient hook did not consume param.grad for {name}")
        grads.append(grad.float().square().sum())
        if ".router." in name and name.endswith("weight"):
            router_norms[name] = float(grad.float().norm())
    if not router_norms or any(value <= 0 for value in router_norms.values()):
        raise AssertionError(f"MoE router gradients must be nonzero: {router_norms}")
    return {
        "main_grad_tensors": len(grads),
        "global_grad_norm": float(torch.stack(grads).sum().sqrt()),
        "router_grad_norms": router_norms,
    }


def _build_training_objects(args, config):
    import torch
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    from primus.backends.megatron.core.fp4_state import bind_fp4_model

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
        vocab_size=128,
        max_sequence_length=args.seq_length,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=150000,
    ).cuda()
    bind_fp4_model(model, rank=0)
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            use_distributed_optimizer=False,
            bucket_size=100000,
        ),
        model,
    )
    optimizer = Float16OptimizerWithFloat16Params(
        torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.0),
        OptimizerConfig(bf16=True, params_dtype=torch.bfloat16, lr=0.002, clip_grad=0.0),
        grad_scaler=None,
        init_state_fn=lambda *_: None,
    )
    runners = []
    if args.deosc:
        from primus.backends.megatron.core.optimizer.two_level_fp4_deosc import (
            install_megatron_deosc,
        )

        runners = install_megatron_deosc(
            optimizer,
            [model],
            SimpleNamespace(
                weight_deosc=True,
                weight_deosc_period=2,
                weight_deosc_ratio=4.0,
                weight_deosc_start_step=0,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
                context_parallel_size=1,
            ),
        )
        if not runners:
            raise AssertionError("DeOsc requested but no runners were installed")
    return model, ddp, optimizer, runners


def _rng_state():
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "megatron_cuda": get_cuda_rng_tracker().get_states(),
    }


def _load_rng_state(state):
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    get_cuda_rng_tracker().set_states(state["megatron_cuda"])


def _assert_equal(left, right, label):
    """Exact recursive comparison, including serialized metadata and optimizer tensors."""
    import io

    import torch

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype:
            raise AssertionError(f"Checkpoint replay tensor dtype differs: {label}")
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise AssertionError(f"Checkpoint replay tensor values differ: {label}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Checkpoint replay keys differ: {label}")
        for key in left:
            _assert_equal(left[key], right[key], f"{label}/{key}")
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Checkpoint replay sequence differs: {label}")
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            _assert_equal(item_left, item_right, f"{label}/{index}")
    elif isinstance(left, io.BytesIO):
        if not isinstance(right, io.BytesIO) or left.getvalue() != right.getvalue():
            raise AssertionError(f"Checkpoint replay TE metadata differs: {label}")
    elif left != right:
        raise AssertionError(f"Checkpoint replay value differs: {label}")


def _step(model, ddp, optimizer, input_ids, position_ids, mask, labels, step):
    import torch

    optimizer.zero_grad()
    ddp.zero_grad_buffer()
    logits = ddp(input_ids, position_ids, mask)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 128), labels.flatten())
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite language model cross entropy")
    loss.backward()
    ddp.finish_grad_sync()
    row = {"step": step + 1, "loss": float(loss.detach()), **_gradient_summary(model)}
    success, _, _ = optimizer.step()
    if not success:
        raise AssertionError("Optimizer skipped an update")
    if any(not torch.isfinite(param).all() for param in model.parameters()):
        raise AssertionError("Non-finite model parameter after optimizer update")
    return row


def run(args, result):
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one available GPU using ROCR_VISIBLE_DEVICES")
    torch.cuda.set_device(0)

    from megatron.core import parallel_state, tensor_parallel

    from primus.backends.megatron.core.fp4_autograd import execution_counts
    from primus.backends.megatron.core.fp4_dispatch import resolve_fp4_plan
    from primus.backends.megatron.core.fp4_state import (
        fp4_state_dict,
        load_fp4_sidecar,
        save_fp4_sidecar,
    )
    from primus.backends.megatron.patches.turbo.fp4_patches import (
        patch_enums,
        patch_fp4_context,
    )
    from primus.backends.megatron.patches.two_level_fp4_patches import (
        install_two_level_fp4_spec_provider,
    )

    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }
    with tempfile.TemporaryDirectory(prefix="primus-fp4-smoke-") as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/rendezvous", rank=0, world_size=1)
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=1,
            )
            torch.manual_seed(args.seed)
            tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
            patch_enums(None)
            patch_fp4_context(None)
            install_two_level_fp4_spec_provider()
            config = _make_config(args)
            result["plan"] = resolve_fp4_plan(config).to_dict()
            model, ddp, optimizer, runners = _build_training_objects(args, config)
            model.train()
            initial = {name: param.detach().clone() for name, param in model.named_parameters()}
            parameter_ids = {name: id(param) for name, param in model.named_parameters()}
            master_parameters = optimizer.get_parameters()
            if any(param.dtype != torch.float32 for param in master_parameters):
                raise AssertionError("Expected FP32 optimizer master parameters")
            result["optimizer"] = {
                "wrapper": type(optimizer).__name__,
                "inner": type(optimizer.optimizer).__name__,
                "master_parameter_count": len(master_parameters),
                "ddp": type(ddp).__name__,
                "overlap_grad_reduce": True,
            }
            data_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
            input_ids = torch.randint(0, 128, (2, args.seq_length), generator=data_rng).cuda()
            labels = input_ids.roll(-1, dims=-1)
            position_ids = torch.arange(args.seq_length, device="cuda").expand_as(input_ids)
            mask = torch.triu(
                torch.ones((1, 1, args.seq_length, args.seq_length), device="cuda", dtype=torch.bool),
                diagonal=1,
            )
            execution_counts(reset=True)
            result["steps"] = []
            checkpoint_step = args.steps // 2
            checkpoint_directory = Path(directory) / "checkpoint"
            checkpoint_directory.mkdir()
            for step in range(args.steps):
                result["steps"].append(
                    _step(model, ddp, optimizer, input_ids, position_ids, mask, labels, step)
                )
                if step + 1 == checkpoint_step:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng": _rng_state(),
                        },
                        checkpoint_directory / "model_optimizer_rng.pt",
                    )
                    save_fp4_sidecar(model, optimizer, checkpoint_directory, checkpoint_step)
            result["execution_counts"] = execution_counts()
            for kind in ("dense", "grouped"):
                for direction in ("fwd", "dgrad", "wgrad"):
                    key = f"{args.recipe}.{kind}.{direction}"
                    if result["execution_counts"].get(key, 0) <= 0:
                        raise AssertionError(f"Quantized path was not exercised: {key}")
            changed = []
            for name, param in model.named_parameters():
                if id(param) != parameter_ids[name]:
                    raise AssertionError(f"Parameter identity changed: {name}")
                if not torch.equal(initial[name], param):
                    changed.append(name)
            if not changed or not any(".experts." in name for name in changed):
                raise AssertionError("Expected model and expert weights to change")
            result["changed_parameters"] = changed
            result["deosc"] = [
                {"step_count": runner.step_count, "stats": dict(runner.stats)} for runner in runners
            ]
            if runners and any(runner.stats.get("periods", 0) < 1 for runner in runners):
                raise AssertionError("DeOsc smoke did not reach a completed period")
            final_fp4_state = fp4_state_dict(model, optimizer)
            final_rng = _rng_state()
            fresh_model, fresh_ddp, fresh_optimizer, _ = _build_training_objects(args, config)
            # This is a locally generated bundle. TE may store BytesIO extra_state,
            # so the full model bundle uses Torch's ordinary trusted-file loader.
            checkpoint = torch.load(checkpoint_directory / "model_optimizer_rng.pt", weights_only=False)
            fresh_model.load_state_dict(checkpoint["model"], strict=True)
            fresh_optimizer.load_state_dict(checkpoint["optimizer"])
            load_fp4_sidecar(fresh_model, fresh_optimizer, checkpoint_directory, checkpoint_step)
            _load_rng_state(checkpoint["rng"])
            fresh_model.train()
            replay = [
                _step(fresh_model, fresh_ddp, fresh_optimizer, input_ids, position_ids, mask, labels, step)
                for step in range(checkpoint_step, args.steps)
            ]
            _assert_equal(result["steps"][checkpoint_step:], replay, "losses_and_gradients")
            _assert_equal(model.state_dict(), fresh_model.state_dict(), "model")
            _assert_equal(optimizer.state_dict(), fresh_optimizer.state_dict(), "optimizer_and_masters")
            _assert_equal(final_fp4_state, fp4_state_dict(fresh_model, fresh_optimizer), "private_fp4_state")
            _assert_equal(final_rng, _rng_state(), "global_rng")
            _assert_equal(
                {name: param.main_grad for name, param in model.named_parameters() if param.requires_grad},
                {
                    name: param.main_grad
                    for name, param in fresh_model.named_parameters()
                    if param.requires_grad
                },
                "ddp_main_grad",
            )
            result["checkpoint_replay"] = {
                "status": "passed",
                "checkpoint_after_step": checkpoint_step,
                "replayed_steps": replay,
                "loss_and_gradients_exact": True,
                "model_parameters_exact": True,
                "optimizer_and_fp32_masters_exact": True,
                "ddp_main_grad_exact": True,
                "private_rng_and_deosc_exact": True,
                "global_rng_exact": True,
                "scope": "Fresh model/DDP/optimizer; Torch bundle plus Primus FP4 sidecar",
            }
            result["status"] = "passed"
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def main():
    args = parse_args()
    result = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": args.recipe,
        "seed": args.seed,
        "model": {
            "implementation": "Megatron GPTModel",
            "layers": 2,
            "layer_pattern": ["dense", "moe"],
            "hidden_size": 128,
            "attention_heads": 2,
            "kv_channels": 64,
            "num_experts": 4,
            "topk": 2,
            "ffn_hidden_size": 128,
            "batch_size": 2,
            "seq_length": args.seq_length,
            "vocab_size": 128,
            "activation": "SiLU GLU",
            "glu_clamp": args.glu_clamp,
            "glu_offset": args.glu_offset,
            "attention": "TE unfused BF16; no FlyDSL performance comparison",
            "data": "fixed synthetic tokens; no convergence claim",
            "activation_recompute": "full/uniform/1" if args.recompute else None,
        },
    }
    try:
        run(args, result)
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"FP4 smoke {result['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    # Permit direct invocation from the repository without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
