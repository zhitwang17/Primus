# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Compare the actual MX and two-level eager activation bodies without TE imports.

This validates the no-bias, non-interleaved SiLU control used by GPT-OSS here.
It does not establish equality of the surrounding quantized GEMMs or training.
"""

import ast
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Union

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    (shape, probability_dtype, clamp, offset)
    for shape in ((35, 34), (128, 5760))
    for probability_dtype in (torch.float32, torch.bfloat16)
    for clamp, offset in ((None, 0.0), (1.5, 0.0), (1.5, 1.0))
]


@lru_cache(maxsize=1)
def _actual_methods():
    namespace = {"torch": torch, "F": F, "Union": Union, "squared_relu": object()}
    for relative, class_name, methods in (
        (
            "primus/backends/megatron/core/transformer/experts.py",
            "PrimusGroupedMLP",
            ("bias_act_func", "bias_act_func_with_mask"),
        ),
        (
            "primus/backends/megatron/core/extensions/two_level_fp4.py",
            "TwoLevelFP4GroupedMLP",
            ("_activate",),
        ),
    ):
        path = ROOT / relative
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        selected = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert len(selected) == len(methods)
        # Compile the unmodified production bodies, including the MX outer route.
        module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def no_fused_routes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The eager activation control reached a fused implementation")

    for name, symbol in (
        ("primus.backends.megatron.core.extensions.primus_turbo", "fused_bias_act_with_probs"),
        ("primus.backends.megatron.core.fusions.fused_bias_swiglu", "weighted_bias_swiglu_impl"),
    ):
        module = types.ModuleType(name)
        setattr(module, symbol, forbidden)
        monkeypatch.setitem(sys.modules, name, module)
    return forbidden


def _assert_bytes_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _check(device, shape, probability_dtype, clamp, offset):
    functions = _actual_methods()
    config = types.SimpleNamespace(
        gated_linear_unit=True,
        moe_mlp_glu_interleave_size=None,
        activation_func_clamp_value=clamp,
        glu_linear_offset=offset,
        activation_func=F.silu,
        use_te_activation_func=False,
        bias_activation_fusion=False,
        use_fused_weighted_squared_relu=False,
    )
    obj = types.SimpleNamespace(config=config, activation_func=F.silu, use_turbo_fused_act_with_probs=False)
    obj.bias_act_func = types.MethodType(functions["bias_act_func"], obj)
    generator = torch.Generator(device="cpu").manual_seed(327)
    x = torch.randn(shape, generator=generator).to(device=device, dtype=torch.bfloat16).requires_grad_()
    probabilities = (
        torch.rand((shape[0], 1), generator=generator)
        .to(device=device, dtype=probability_dtype)
        .requires_grad_()
    )
    x_reference = x.detach().clone().requires_grad_()
    probabilities_reference = probabilities.detach().clone().requires_grad_()
    result = functions["bias_act_func_with_mask"](obj, x, None, probabilities, None)
    reference = functions["_activate"](obj, x_reference, probabilities_reference)
    grad = torch.randn(result.shape, generator=generator).to(device=device, dtype=result.dtype)
    dx, dp = torch.autograd.grad(result, (x, probabilities), grad)
    dx_reference, dp_reference = torch.autograd.grad(reference, (x_reference, probabilities_reference), grad)
    _assert_bytes_equal(result, reference)
    _assert_bytes_equal(dx, dx_reference)
    _assert_bytes_equal(dp, dp_reference)


@pytest.mark.parametrize("shape,probability_dtype,clamp,offset", CASES)
def test_cpu_eager_activation_control(shape, probability_dtype, clamp, offset, no_fused_routes):
    _check("cpu", shape, probability_dtype, clamp, offset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an explicitly allocated GPU")
@pytest.mark.parametrize("shape,probability_dtype,clamp,offset", CASES)
def test_device_kernel_eager_activation_control(shape, probability_dtype, clamp, offset, no_fused_routes):
    _check("cuda", shape, probability_dtype, clamp, offset)
