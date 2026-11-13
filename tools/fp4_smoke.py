# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""One/two-GPU Megatron GPTModel smoke for two-level FP4 adapters.

This exercises real model/optimizer/DDP paths on synthetic tokens. It is not a
convergence or throughput benchmark, and does not use FlyDSL attention. The
gpt-oss profile retains the original hidden/head/expert geometry; only depth,
sequence, batch and vocabulary are reduced. Use torchrun for two replicated DP
ranks, one visible GPU per local rank.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def model_geometry(profile, num_layers=None):
    """Pinned Primus gpt_oss_20B.yaml geometry, independent of GPU imports."""
    if profile == "gpt-oss":
        layers = 1 if num_layers is None else num_layers
        return dict(
            layers=layers,
            hidden_size=2880,
            attention_heads=64,
            query_groups=8,
            kv_channels=64,
            num_experts=32,
            topk=4,
            ffn_hidden_size=2880,
            layer_pattern=[1] * layers,
        )
    if profile != "tiny":
        raise ValueError(f"Unknown model profile: {profile}")
    layers = 2 if num_layers is None else num_layers
    return dict(
        layers=layers,
        hidden_size=128,
        attention_heads=2,
        query_groups=1,
        kv_channels=64,
        num_experts=4,
        topk=2,
        ffn_hidden_size=128,
        layer_pattern=[0] * (layers - 1) + [1],
    )


def _source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    files = [
        "tools/fp4_smoke.py",
        *[
            f"primus/backends/megatron/core/{name}.py"
            for name in (
                "fp4_contract",
                "fp4_dispatch",
                "fp4_reference",
                "fp4_backend",
                "fp4_triton",
                "fp4_autograd",
                "fp4_state",
                "fp4_checkpoint",
                "optimizer/two_level_fp4_deosc",
                "extensions/two_level_fp4",
            )
        ],
        "primus/backends/megatron/patches/two_level_fp4_patches.py",
        "third_party/Megatron-LM/megatron/core/transformer/moe/moe_utils.py",
    ]
    files.extend(
        str(path.relative_to(root))
        for path in sorted((root / "primus/backends/megatron/core/optimizer").glob("*fp4*shard*.py"))
        if str(path.relative_to(root)) not in files
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("nvfp4", "amdfp4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-profile", choices=("tiny", "gpt-oss"), default="tiny")
    parser.add_argument("--num-layers", type=int, choices=(1, 2), default=None)
    parser.add_argument("--seq-length", type=int, choices=(16, 32, 64, 128), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--quantizer", choices=("torch", "triton"), default="torch")
    parser.add_argument("--checkpoint-format", choices=("bundle", "torch_dist"), default="bundle")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--glu-clamp", type=float, default=None)
    parser.add_argument("--glu-offset", type=float, default=0.0)
    parser.add_argument("--deosc", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--deosc-accumulator-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic Torch/Megatron operations for exact checkpoint replay",
    )
    parser.add_argument("--recompute", action="store_true", help="Exercise full-layer activation recompute")
    args = parser.parse_args(argv)
    args.seq_length = args.seq_length or (32 if args.model_profile == "gpt-oss" else 64)
    args.batch_size = (
        args.batch_size if args.batch_size is not None else (1 if args.model_profile == "gpt-oss" else 2)
    )
    args.lr = args.lr if args.lr is not None else (5.7e-4 if args.model_profile == "gpt-oss" else 0.002)
    if args.steps < 2 or (args.deosc and args.steps < 4):
        parser.error("checkpoint smoke needs at least two updates; DeOsc needs at least four")
    if args.glu_clamp is not None and args.glu_clamp <= 0:
        parser.error("glu-clamp must be positive")
    if args.batch_size <= 0 or args.vocab_size < 8 or args.vocab_size % 8 or args.lr <= 0:
        parser.error("batch-size/lr must be positive and vocab-size must be a positive multiple of eight")
    if args.gradient_accumulation_steps < 1:
        parser.error("gradient-accumulation-steps must be positive")
    if args.distributed_optimizer and args.checkpoint_format != "torch_dist":
        parser.error("distributed optimizer smoke requires --checkpoint-format torch_dist")
    return args


def _make_config(args):
    import torch
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    from primus.backends.megatron.core.enums import Fp4Recipe

    geometry = model_geometry(args.model_profile, args.num_layers)
    config = TransformerConfig(
        num_layers=geometry["layers"],
        hidden_size=geometry["hidden_size"],
        num_attention_heads=geometry["attention_heads"],
        num_query_groups=geometry["query_groups"],
        kv_channels=geometry["kv_channels"],
        ffn_hidden_size=geometry["ffn_hidden_size"],
        num_moe_experts=geometry["num_experts"],
        moe_ffn_hidden_size=geometry["ffn_hidden_size"],
        moe_layer_freq=geometry["layer_pattern"],
        moe_router_topk=geometry["topk"],
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
        deterministic_mode=args.deterministic,
    )
    # Primus options are not dataclass fields in upstream TransformerConfig.
    config.fp4_backend = "reference"
    config.fp4_rng_seed = args.seed
    config.fp4_quantizer = args.quantizer
    config.use_distributed_optimizer = args.distributed_optimizer
    config.weight_deosc = args.deosc
    config.weight_deosc_accumulator_dtype = args.deosc_accumulator_dtype
    config.enable_primus_turbo = False
    config.use_turbo_gemm = False
    config.use_turbo_grouped_gemm = False
    return config


_CHECK_CHUNK_ELEMENTS = 4 * 1024 * 1024


def _tensor_chunks(tensor):
    flat = tensor.detach().reshape(-1)
    for begin in range(0, flat.numel(), _CHECK_CHUNK_ELEMENTS):
        yield flat[begin : begin + _CHECK_CHUNK_ELEMENTS]


def _state_fingerprint(value):
    """Hash every tensor byte and structural metadata using bounded host copies.

    Large geometry replay must not retain two complete GPU models or two sets of
    Adam/DeOsc state. SHA256 lets the first training objects be released before
    fresh construction while checking all values, including zero gradients.
    """
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            for chunk in _tensor_chunks(item):
                digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif isinstance(item, io.BytesIO):
            digest.update(b"bytesio:" + item.getvalue())
        else:
            digest.update(f"{type(item).__name__}:{item!r}:".encode())

    visit(value)
    return digest.hexdigest()


def _finite(tensor):
    import torch

    valid = torch.ones((), device=tensor.device, dtype=torch.bool)
    for chunk in _tensor_chunks(tensor):
        valid.logical_and_(torch.isfinite(chunk).all())
    return bool(valid)


def _squared_norm(tensor):
    import torch

    total = torch.zeros((), device=tensor.device, dtype=torch.float32)
    for chunk in _tensor_chunks(tensor):
        total.add_(chunk.float().square().sum())
    return total


def _assert_replicated(named_tensors, label):
    """Compare every element to rank0, with bounded GPU scratch and no sampling."""
    import torch
    import torch.distributed as dist

    tensors = list(named_tensors)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return {"status": "single_rank", "tensors": len(tensors), "elements": 0}
    if not tensors:
        raise ValueError(f"DP {label} check requires at least one tensor")
    metadata = [(name, tuple(tensor.shape), str(tensor.dtype)) for name, tensor in tensors]
    identities = [None] * world_size
    dist.all_gather_object(identities, _state_fingerprint(metadata))
    if len(set(identities)) != 1:
        raise AssertionError(f"DP {label} tensor metadata differs between ranks")
    mismatch = torch.zeros((), device=tensors[0][1].device, dtype=torch.int32)
    elements = 0
    for _, tensor in tensors:
        for chunk in _tensor_chunks(tensor):
            reference = chunk.clone()
            dist.broadcast(reference, src=0)
            mismatch.bitwise_or_((chunk != reference).any().int())
            elements += chunk.numel()
    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
    if int(mismatch):
        raise AssertionError(f"DP {label} differs after synchronization/update")
    return {"status": "exact", "tensors": len(tensors), "elements": elements}


def _memory_summary():
    import torch

    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


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
        if not _finite(grad):
            raise AssertionError(f"Non-finite main_grad: {name}")
        if param.grad is not None:
            raise AssertionError(f"DDP gradient hook did not consume param.grad for {name}")
        grads.append(_squared_norm(grad))
        if ".router." in name and name.endswith("weight"):
            router_norms[name] = float(_squared_norm(grad).sqrt())
    if not router_norms or any(value <= 0 for value in router_norms.values()):
        raise AssertionError(f"MoE router gradients must be nonzero: {router_norms}")
    return {
        "main_grad_tensors": len(grads),
        "global_grad_norm": float(torch.stack(grads).sum().sqrt()),
        "router_grad_norms": router_norms,
    }


def _initialize_adam_state(optimizer, *unused):
    """MCore calls this before loading sharded optimizer tensors."""
    import torch

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)


def _build_training_objects(args, config):
    import torch
    import torch.distributed as dist
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    from primus.backends.megatron.core.fp4_state import bind_fp4_model

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_length,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=150000,
    ).cuda()
    bind_fp4_model(model, rank=dist.get_rank())
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            use_distributed_optimizer=args.distributed_optimizer,
            overlap_param_gather=False,
            bucket_size=40_000_000,
        ),
        model,
    )
    if args.distributed_optimizer:
        # Use the real MCore construction path, including separate expert and
        # dense leaf optimizers and the actual DDP buffer partition mapping.
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                bf16=True,
                params_dtype=torch.bfloat16,
                lr=args.lr,
                clip_grad=0.0,
                adam_beta1=0.9,
                adam_beta2=0.95,
                weight_decay=0.0,
                use_distributed_optimizer=True,
            ),
            [ddp],
        )
    else:
        optimizer = Float16OptimizerWithFloat16Params(
            torch.optim.Adam(
                model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False
            ),
            OptimizerConfig(bf16=True, params_dtype=torch.bfloat16, lr=args.lr, clip_grad=0.0),
            grad_scaler=None,
            init_state_fn=_initialize_adam_state,
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
                data_parallel_size=dist.get_world_size(),
                use_distributed_optimizer=args.distributed_optimizer,
                weight_deosc_accumulator_dtype=args.deosc_accumulator_dtype,
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
        "torch_cuda": torch.cuda.get_rng_state(),
        "megatron_cuda": get_cuda_rng_tracker().get_states(),
    }


def _load_rng_state(state):
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
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


def _step(
    model,
    ddp,
    optimizer,
    input_ids,
    position_ids,
    mask,
    labels,
    step,
    vocab_size,
    *,
    distributed_optimizer=False,
    gradient_accumulation_steps=1,
):
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad()
    ddp.zero_grad_buffer()
    loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
    for microbatch in range(gradient_accumulation_steps):
        context = ddp.no_sync() if microbatch + 1 < gradient_accumulation_steps else nullcontext()
        with context:
            logits = ddp(input_ids, position_ids, mask)
            micro_loss = (
                torch.nn.functional.cross_entropy(logits.float().reshape(-1, vocab_size), labels.flatten())
                / gradient_accumulation_steps
            )
            if not torch.isfinite(micro_loss):
                raise AssertionError("Non-finite language model cross entropy")
            micro_loss.backward()
            loss.add_(micro_loss.detach())
    ddp.finish_grad_sync()
    average_loss = loss.detach().clone()
    dist.all_reduce(average_loss)
    average_loss.div_(dist.get_world_size())
    gradient_summary = _gradient_summary(model)
    if distributed_optimizer:
        # Only owned intervals contain reduced gradients after ReduceScatter.
        # Do not label the norm of the entire backing buffer as a global norm.
        gradient_summary["buffer_grad_norm"] = gradient_summary.pop("global_grad_norm")
    row = {
        "step": step + 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "loss": float(loss.detach()),
        "dp_average_loss": float(average_loss),
        **gradient_summary,
    }
    if not distributed_optimizer:
        row["gradient_sync"] = _assert_replicated(
            (
                (name, parameter.main_grad)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ),
            "main_grad",
        )
    success, _, _ = optimizer.step()
    if not success:
        raise AssertionError("Optimizer skipped an update")
    if any(not _finite(param) for param in model.parameters()):
        raise AssertionError("Non-finite model parameter after optimizer update")
    row["parameter_sync"] = _assert_replicated(model.named_parameters(), "model parameters")
    if distributed_optimizer:
        masters = list(optimizer.get_parameters())
        local_elements = sum(parameter.numel() for parameter in masters)
        global_elements = torch.tensor(local_elements, dtype=torch.int64, device=input_ids.device)
        dist.all_reduce(global_elements)
        expected_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if int(global_elements) != expected_elements:
            raise AssertionError("Optimizer shards do not cover the complete trainable model exactly")
        gradient_squares = torch.zeros((), dtype=torch.float64, device=input_ids.device)
        for parameter in masters:
            if parameter.dtype != torch.float32 or not _finite(parameter):
                raise AssertionError("Invalid local FP32 optimizer shard")
            if parameter.grad is None or not _finite(parameter.grad):
                raise AssertionError("Missing or non-finite reduced optimizer gradient shard")
            gradient_squares.add_(_squared_norm(parameter.grad))
        dist.all_reduce(gradient_squares)
        row["global_grad_norm"] = float(gradient_squares.sqrt())
        row["gradient_sync"] = {"status": "reduce_scatter", "finite_owned_gradients": True}
        row["master_sync"] = {
            "status": "partitioned",
            "local_elements": local_elements,
            "global_elements": int(global_elements),
            "coverage_exact": True,
        }
    else:
        row["master_sync"] = _assert_replicated(
            ((str(index), parameter) for index, parameter in enumerate(optimizer.get_parameters())),
            "FP32 masters",
        )
    torch.cuda.synchronize()
    row["elapsed_seconds_including_checks"] = time.perf_counter() - started
    return row


def _save_checkpoint(args, model, optimizer, directory, iteration):
    import torch
    import torch.distributed as dist

    if args.checkpoint_format == "torch_dist":
        from primus.backends.megatron.core.fp4_checkpoint import save_fp4_checkpoint

        save_fp4_checkpoint([model], optimizer, directory, iteration, extra_state={"rng": _rng_state()})
    else:
        from primus.backends.megatron.core.fp4_state import save_fp4_sidecar

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state()},
            directory / f"model_optimizer_rng.rank{dist.get_rank()}.pt",
        )
        save_fp4_sidecar(model, optimizer, directory, iteration)
    dist.barrier()


def _load_checkpoint(args, model, optimizer, directory, iteration):
    import torch
    import torch.distributed as dist

    if args.checkpoint_format == "torch_dist":
        from primus.backends.megatron.core.fp4_checkpoint import load_fp4_checkpoint

        extra = load_fp4_checkpoint([model], optimizer, directory, iteration)
        _load_rng_state(extra["rng"])
    else:
        from primus.backends.megatron.core.fp4_state import load_fp4_sidecar

        # The file is generated by this run. mmap avoids keeping a second full
        # model and Adam state resident on host during large-geometry restore.
        checkpoint = torch.load(
            directory / f"model_optimizer_rng.rank{dist.get_rank()}.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        load_fp4_sidecar(model, optimizer, directory, iteration)
        _load_rng_state(checkpoint["rng"])
        del checkpoint
    dist.barrier()


def _optimizer_leaves(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield optimizer
    else:
        for child in children:
            yield from _optimizer_leaves(child)


def _complete_optimizer_state(optimizer):
    # DistributedOptimizer.state_dict intentionally omits the parameter state,
    # which is saved by a separate sharded-checkpoint path. Include real local
    # masters and inner Adam moments here so an exact-replay assertion cannot
    # accidentally succeed by comparing only metadata.
    return {
        "wrapper_state": optimizer.state_dict(),
        "leaves": [
            {
                "wrapper": type(leaf).__name__,
                "inner_state": leaf.optimizer.state_dict() if leaf.optimizer is not None else None,
                "fp32_masters": list(leaf.get_parameters()),
            }
            for leaf in _optimizer_leaves(optimizer)
        ],
    }


def _training_fingerprints(model, optimizer, *, include_gradients=True):
    from primus.backends.megatron.core.fp4_state import fp4_state_dict

    fingerprints = {
        "model": _state_fingerprint(model.state_dict()),
        "optimizer_and_fp32_masters": _state_fingerprint(_complete_optimizer_state(optimizer)),
        "private_rng_and_deosc": _state_fingerprint(fp4_state_dict(model, optimizer)),
        "global_rng": _state_fingerprint(_rng_state()),
    }
    if include_gradients:
        fingerprints["ddp_main_grad"] = _state_fingerprint(
            {
                name: parameter.main_grad
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
    return fingerprints


def _comparable_steps(steps):
    return [
        {key: value for key, value in row.items() if key != "elapsed_seconds_including_checks"}
        for row in steps
    ]


def _changed_parameters(model, initial, identities):
    changed = []
    for name, parameter in model.named_parameters():
        if id(parameter) != identities[name]:
            raise AssertionError(f"Parameter identity changed: {name}")
        if _state_fingerprint(parameter) != initial[name]:
            changed.append(name)
    if not changed or not any(".experts." in name for name in changed):
        raise AssertionError("Expected model and expert weights to change")
    return changed


def run(args, result):
    import torch
    import torch.distributed as dist

    if args.deterministic:
        from megatron.training.determinism import apply_determinism_env

        # Validate/set TE, BLAS and collective determinism before creating GPU
        # contexts or communicators, just as the real MCore training entry does.
        apply_determinism_env(os.environ)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 2, 4, 8):
        raise RuntimeError("The reference smoke supports one, two, four or eight DP ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError("Expose exactly one GPU per local torchrun rank using ROCR_VISIBLE_DEVICES")
    torch.cuda.set_device(local_rank)
    torch.use_deterministic_algorithms(args.deterministic)

    from megatron.core import parallel_state, tensor_parallel

    from primus.backends.megatron.core.fp4_autograd import execution_counts
    from primus.backends.megatron.core.fp4_dispatch import resolve_fp4_plan
    from primus.backends.megatron.patches.turbo.fp4_patches import (
        patch_enums,
        patch_fp4_context,
    )
    from primus.backends.megatron.patches.two_level_fp4_patches import (
        install_two_level_fp4_spec_provider,
    )

    result["environment"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(local_rank),
        "device_total_memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
        "visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvte_allow_nondeterministic_algo": os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO"),
        "nccl_algo": os.environ.get("NCCL_ALGO"),
    }
    with tempfile.TemporaryDirectory(prefix="primus-fp4-smoke-") as directory:
        init_method = "env://" if world_size > 1 else f"file://{directory}/rendezvous"
        dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
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
            torch.cuda.reset_peak_memory_stats()
            model, ddp, optimizer, runners = _build_training_objects(args, config)
            model.train()
            result["memory"] = {"initialization": _memory_summary()}
            result["initial_replica_check"] = _assert_replicated(
                model.named_parameters(), "initial parameters"
            )
            initial = {name: _state_fingerprint(parameter) for name, parameter in model.named_parameters()}
            parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
            masters = optimizer.get_parameters()
            if any(parameter.dtype != torch.float32 for parameter in masters):
                raise AssertionError("Expected FP32 optimizer master parameters")
            result["optimizer"] = {
                "wrapper": type(optimizer).__name__,
                "inner": [type(leaf.optimizer).__name__ for leaf in _optimizer_leaves(optimizer)],
                "master_parameter_count": len(masters),
                "master_element_count": sum(parameter.numel() for parameter in masters),
                "ddp": type(ddp).__name__,
                "overlap_grad_reduce": True,
                "replicated_full_masters": not args.distributed_optimizer,
                "distributed_optimizer": args.distributed_optimizer,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.lr,
            }
            del masters
            data_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1 + rank)
            input_ids = torch.randint(
                0, args.vocab_size, (args.batch_size, args.seq_length), generator=data_rng
            ).cuda()
            data_signatures = [None] * world_size
            dist.all_gather_object(data_signatures, _state_fingerprint(input_ids))
            if world_size > 1 and len(set(data_signatures)) != world_size:
                raise AssertionError("DP smoke requires distinct synthetic inputs on each rank")
            result["data"] = {
                "rank_seed": args.seed + 1 + rank,
                "rank_input_sha256": data_signatures,
                "distinct_between_ranks": len(set(data_signatures)) == world_size,
            }
            labels = input_ids.roll(-1, dims=-1)
            position_ids = torch.arange(args.seq_length, device="cuda").expand_as(input_ids)
            mask = torch.triu(
                torch.ones((1, 1, args.seq_length, args.seq_length), device="cuda", dtype=torch.bool),
                diagonal=1,
            )
            execution_counts(reset=True)
            result["steps"] = []
            checkpoint_step = args.steps // 2
            shared_directory = [
                (
                    str(args.checkpoint_dir.resolve())
                    if args.checkpoint_dir
                    else str(Path(directory) / "checkpoint")
                )
            ]
            dist.broadcast_object_list(shared_directory, src=0)
            checkpoint_directory = Path(shared_directory[0])
            if rank == 0:
                checkpoint_directory.mkdir(parents=True, exist_ok=True)
                if any(checkpoint_directory.iterdir()):
                    raise ValueError(f"Use an empty checkpoint directory: {checkpoint_directory}")
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            for step in range(args.steps):
                result["steps"].append(
                    _step(
                        model,
                        ddp,
                        optimizer,
                        input_ids,
                        position_ids,
                        mask,
                        labels,
                        step,
                        args.vocab_size,
                        distributed_optimizer=args.distributed_optimizer,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                    )
                )
                if step + 1 == checkpoint_step:
                    result["checkpoint_saved_state_sha256"] = _training_fingerprints(
                        model, optimizer, include_gradients=False
                    )
                    _save_checkpoint(args, model, optimizer, checkpoint_directory, checkpoint_step)
                    after_save = _training_fingerprints(model, optimizer, include_gradients=False)
                    _assert_equal(result["checkpoint_saved_state_sha256"], after_save, "save_side_effects")
            result["memory"]["training_and_checkpoint"] = _memory_summary()
            result["execution_counts"] = execution_counts()
            for kind in ("dense", "grouped"):
                for direction in ("fwd", "dgrad", "wgrad"):
                    key = f"{args.recipe}.{kind}.{direction}"
                    if result["execution_counts"].get(key, 0) <= 0:
                        raise AssertionError(f"Quantized path was not exercised: {key}")
            result["changed_parameters"] = _changed_parameters(model, initial, parameter_ids)
            result["deosc"] = [
                {"step_count": runner.step_count, "stats": dict(runner.stats)} for runner in runners
            ]
            if runners and any(runner.stats.get("periods", 0) < 1 for runner in runners):
                raise AssertionError("DeOsc smoke did not reach a completed period")
            expected_fingerprints = _training_fingerprints(model, optimizer)
            # Do not retain two large models, optimizer masters/moments or DeOsc
            # histories while testing the original GPT-OSS expert-bank geometry.
            del model, ddp, optimizer, runners
            gc.collect()
            torch.cuda.empty_cache()
            result["memory"]["after_releasing_original"] = _memory_summary()
            torch.cuda.reset_peak_memory_stats()
            fresh_model, fresh_ddp, fresh_optimizer, fresh_runners = _build_training_objects(args, config)
            _load_checkpoint(args, fresh_model, fresh_optimizer, checkpoint_directory, checkpoint_step)
            fresh_model.train()
            restored_fingerprints = _training_fingerprints(
                fresh_model, fresh_optimizer, include_gradients=False
            )
            result["checkpoint_restored_state_sha256"] = restored_fingerprints
            _assert_equal(
                result["checkpoint_saved_state_sha256"],
                restored_fingerprints,
                "immediately_restored_state_sha256",
            )
            result["checkpoint_initial_state_exact"] = True
            replay = [
                _step(
                    fresh_model,
                    fresh_ddp,
                    fresh_optimizer,
                    input_ids,
                    position_ids,
                    mask,
                    labels,
                    step,
                    args.vocab_size,
                    distributed_optimizer=args.distributed_optimizer,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
                for step in range(checkpoint_step, args.steps)
            ]
            # Preserve evidence before asserting so failed replays can distinguish
            # missing checkpoint state from nondeterministic forward/backward ops.
            actual_fingerprints = _training_fingerprints(fresh_model, fresh_optimizer)
            result["memory"]["fresh_restore_and_replay"] = _memory_summary()
            result["checkpoint_replay"] = {
                "status": "checking",
                "format": args.checkpoint_format,
                "checkpoint_after_step": checkpoint_step,
                "replayed_steps": replay,
                "expected_state_sha256": expected_fingerprints,
                "state_sha256": actual_fingerprints,
            }
            _assert_equal(
                _comparable_steps(result["steps"][checkpoint_step:]),
                _comparable_steps(replay),
                "losses_and_gradients",
            )
            _assert_equal(expected_fingerprints, actual_fingerprints, "complete_training_state_sha256")
            result["checkpoint_replay"] = {
                "status": "passed",
                "format": args.checkpoint_format,
                "checkpoint_after_step": checkpoint_step,
                "replayed_steps": replay,
                "comparison": "Exact scalar checks and SHA256 over every tensor byte plus structural metadata; no sampling",
                "state_sha256": actual_fingerprints,
                "loss_and_gradients_exact": True,
                "model_parameters_exact": True,
                "optimizer_and_fp32_masters_exact": True,
                "ddp_main_grad_exact": True,
                "private_rng_and_deosc_exact": True,
                "global_rng_exact": True,
                "scope": "Fresh model/DDP/optimizer; original training objects released before reconstruction",
            }
            result["status"] = "passed"
            rank_summaries = [None] * world_size
            dist.all_gather_object(
                rank_summaries,
                {
                    "rank": rank,
                    "status": "passed",
                    "losses": [row["loss"] for row in result["steps"]],
                    "dp_average_losses": [row["dp_average_loss"] for row in result["steps"]],
                    "peak_allocated_bytes": result["memory"]["training_and_checkpoint"][
                        "peak_allocated_bytes"
                    ],
                    "checkpoint_replay": "passed",
                    "gradient_sync": result["steps"][-1]["gradient_sync"],
                    "parameter_sync": result["steps"][-1]["parameter_sync"],
                    "master_sync": result["steps"][-1]["master_sync"],
                },
            )
            result["rank_results"] = rank_summaries
            del fresh_model, fresh_ddp, fresh_optimizer, fresh_runners
            gc.collect()
            dist.barrier()  # All ranks must finish restore before rank0 deletes temporary files.
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    output = (
        args.output
        if rank == 0
        else args.output.with_name(f"{args.output.stem}.rank{rank}{args.output.suffix}")
    )
    geometry = model_geometry(args.model_profile, args.num_layers)
    result = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": args.recipe,
        "quantizer": args.quantizer,
        "seed": args.seed,
        "model": {
            "implementation": "Megatron GPTModel",
            "profile": args.model_profile,
            **geometry,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "vocab_size": args.vocab_size,
            "activation": "SiLU GLU",
            "glu_clamp": args.glu_clamp,
            "glu_offset": args.glu_offset,
            "attention": "TE unfused BF16; no FlyDSL performance comparison",
            "data": "fixed synthetic tokens per rank; no convergence claim",
            "activation_recompute": "full/uniform/1" if args.recompute else None,
        },
        "checkpoint_format": args.checkpoint_format,
        "source_sha256": _source_fingerprints(),
    }
    try:
        run(args, result)
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"FP4 smoke rank{rank} {result['status']}: {output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
