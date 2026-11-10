# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Executable two-level FP4 reference GEMMs, with independently quantized gradients.

These operators pack/unpack real FP4 values and multiply the dequantized values.
They do not call native FP4 GEMMs. Parameters retain their identity and ordinary
autograd gradients: Megatron DDP owns main_grad accumulation and readiness hooks.
There is no persistent quantized-weight cache to invalidate after optimizer steps.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from .fp4_contract import FP4RandomState, padded_group_offsets
from .fp4_reference import compute_outer_scale, qdq, rht_pair

_EXECUTIONS = Counter()


def execution_counts(reset=False):
    """Evidence that all three quantized directions ran (not a profiler)."""
    result = dict(_EXECUTIONS)
    if reset:
        _EXECUTIONS.clear()
    return result


def _record(config, kind, direction):
    _EXECUTIONS[f"{config.recipe}.{kind}.{direction}"] += 1


def _check_random(config, random):
    if config.gradient_sr or (config.rht_size and config.rht_randomized):
        if not isinstance(random, FP4RandomState):
            raise ValueError("Bind a private FP4RandomState before reference training")
    if random is not None and random.recipe_fingerprint != config.fingerprint:
        raise ValueError("FP4 RNG recipe does not match GEMM config")


def _q(x, config, direction, role, random, outer=None):
    return qdq(x, config, config.operand(direction, role), outer_scale=outer, random=random, dtype=x.dtype)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, config, random):
        if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
            raise ValueError("FP4 linear expects [...,K] input and [N,K] weight")
        _check_random(config, random)
        ctx.save_for_backward(x, weight)
        ctx.config, ctx.random = config, random
        flat = x.reshape(-1, x.shape[-1])
        xq = _q(flat, config, "fwd", "input", random)
        wq = _q(weight, config, "fwd", "weight", random)
        _record(config, "dense", "fwd")
        return (xq @ wq.t()).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        config, random = ctx.config, ctx.random
        flat = x.reshape(-1, x.shape[-1])
        dy = grad_output.reshape(-1, weight.shape[0]).contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dyq = _q(dy, config, "dgrad", "grad_output", random)
            wq = _q(weight, config, "dgrad", "weight", random)
            dx = (dyq @ wq).reshape_as(x)
            _record(config, "dense", "dgrad")
        if ctx.needs_input_grad[1]:
            xh, dyh = rht_pair(flat, dy, config, random=random)
            xq = _q(xh, config, "wgrad", "input", random)
            dyq = _q(dyh, config, "wgrad", "grad_output", random)
            dw = dyq.t() @ xq
            _record(config, "dense", "wgrad")
        return dx, dw, None, None


def linear(x, weight, config, random=None):
    return _Linear.apply(x, weight, config, random)


def _lengths(lengths, rows, experts):
    if isinstance(lengths, torch.Tensor):
        if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expert lengths require a 1-D integer tensor")
        # The reference backend intentionally permits a host synchronization.
        lengths = lengths.detach().cpu().tolist()
    lengths = tuple(lengths)
    offsets, padded = padded_group_offsets(lengths)
    if len(lengths) != experts or offsets[-1] != rows:
        raise ValueError("Expert lengths must match the weight bank and span the input")
    return lengths, offsets, padded


def _split_pad(x, lengths):
    return [F.pad(part, (0, 0, 0, (-n) % 128)) for part, n in zip(x.split(lengths), lengths)]


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, lengths, config, random):
        if x.ndim != 2 or weights.ndim != 3 or x.shape[1] != weights.shape[2]:
            raise ValueError("FP4 grouped linear expects [M,K] input and [E,N,K] weights")
        lengths, offsets, padded = _lengths(lengths, x.shape[0], weights.shape[0])
        _check_random(config, random)
        ctx.save_for_backward(x, weights)
        ctx.config, ctx.random, ctx.lengths = config, random, lengths
        ctx.group_offsets, ctx.padded_offsets = offsets, padded
        # All local experts share the bank's outer scale. Per-expert quantization
        # must not accidentally replace that statistic with a local maximum.
        outer_x = compute_outer_scale(x, config)
        wq = _q(weights, config, "fwd", "weight", random)
        parts = []
        for i, (xp, n) in enumerate(zip(_split_pad(x, lengths), lengths)):
            if n:
                xq = _q(xp, config, "fwd", "input", random, outer_x)
                parts.append((xq @ wq[i].t())[:n])
            else:
                parts.append(x.new_empty((0, weights.shape[1])))
        _record(config, "grouped", "fwd")
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, weights = ctx.saved_tensors
        config, random, lengths = ctx.config, ctx.random, ctx.lengths
        dy = grad_output.contiguous()
        xp, dyp = _split_pad(x, lengths), _split_pad(dy, lengths)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            outer_dy = compute_outer_scale(dy, config)
            wq = _q(weights, config, "dgrad", "weight", random)
            parts = []
            for i, (part, n) in enumerate(zip(dyp, lengths)):
                if n:
                    dyq = _q(part, config, "dgrad", "grad_output", random, outer_dy)
                    parts.append((dyq @ wq[i])[:n])
                else:
                    parts.append(x.new_empty((0, x.shape[1])))
            dx = torch.cat(parts, dim=0)
            _record(config, "grouped", "dgrad")
        if ctx.needs_input_grad[1]:
            transformed = [rht_pair(a, b, config, random=random) for a, b in zip(xp, dyp)]
            # wgrad's outer scope is the full transformed routed buffer. RHT is
            # independently padded per expert and shared by both operands.
            outer_x = compute_outer_scale(torch.cat([a for a, _ in transformed]), config)
            outer_dy = compute_outer_scale(torch.cat([b for _, b in transformed]), config)
            dw = torch.zeros_like(weights)
            for i, ((a, b), n) in enumerate(zip(transformed, lengths)):
                if n:
                    aq = _q(a, config, "wgrad", "input", random, outer_x)
                    bq = _q(b, config, "wgrad", "grad_output", random, outer_dy)
                    dw[i] = bq.t() @ aq
            _record(config, "grouped", "wgrad")
        return dx, dw, None, None, None


def grouped_linear(x, weights, lengths, config, random=None):
    return _GroupedLinear.apply(x, weights, lengths, config, random)
