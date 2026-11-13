# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Pin initialization against TE 2.17.1+130099ce4's parameter protocol.

TE GroupedLinear registers weight{i}/bias{i} in expert order, then its base
reset_parameters initializes each registered weight in params_dtype under its
own expert RNG fork. Turbo subsequently copies those weights into one bank.
TE LayerNormLinear registers constant norm parameters before weight/bias.
The oracle below is that observed TE protocol, without importing native TE or
requiring a GPU. Actual device normal_ equality remains a GPU hash check.
"""

from contextlib import contextmanager

import pytest
import torch
from test_fp4_modules import config, modules  # noqa: F401: pytest fixture


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_cpu_initializes_each_native_expert_in_parameter_dtype(modules, dtype):
    cfg = config(params_dtype=dtype)
    calls = []

    def initialize(tensor):
        calls.append((tuple(tensor.shape), tensor.dtype))
        torch.nn.init.normal_(tensor, std=0.2)

    torch.manual_seed(531)
    expected = torch.stack([torch.empty(17, 19, dtype=dtype).normal_(std=0.2) for _ in range(3)])
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(531)
    layer = modules.M.TwoLevelFP4GroupedLinear(
        3,
        19,
        17,
        config=cfg,
        init_method=initialize,
        bias=True,
    )
    assert calls == [((17, 19), dtype)] * 3
    assert torch.equal(layer.weights, expected)
    assert torch.equal(torch.get_rng_state(), expected_state)
    assert torch.count_nonzero(layer.bias) == 0
    assert [name for name, _ in layer.named_parameters()] == ["weights", "bias"]
    assert layer.weights.is_contiguous()
    # The former whole-bank call differs for these deliberately nonaligned
    # dimensions, so this test detects changes to draw granularity.
    torch.manual_seed(531)
    whole_bank = torch.empty_like(expected).normal_(std=0.2)
    assert not torch.equal(whole_bank, expected)


class Tracker:
    def __init__(self):
        self.states = {
            "model": torch.Generator().manual_seed(39).get_state(),
            "expert": torch.Generator().manual_seed(79).get_state(),
        }
        self.calls = []

    def is_initialized(self):
        return True

    @contextmanager
    def fork(self, name="model"):
        self.calls.append(name)
        outside = torch.get_rng_state()
        torch.set_rng_state(self.states[name])
        try:
            yield
        finally:
            self.states[name] = torch.get_rng_state()
            torch.set_rng_state(outside)


def test_device_init_protocol_forks_each_expert_without_global_rng_consumption(modules, monkeypatch):
    tracker = Tracker()
    cfg = config(params_dtype=torch.bfloat16, use_cpu_initialization=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(modules.M, "get_cuda_rng_tracker", lambda: tracker)
    outside = torch.get_rng_state().clone()
    expected_tracker = Tracker()
    expected = []
    for _ in range(3):
        with expected_tracker.fork("expert"):
            expected.append(torch.empty(17, 19, dtype=torch.bfloat16).normal_(std=0.2))
    layer = modules.M.TwoLevelFP4GroupedLinear(
        3,
        19,
        17,
        config=cfg,
        init_method=cfg.init_method,
        bias=True,
    )
    assert tracker.calls == ["expert"] * 3
    assert torch.equal(layer.weights, torch.stack(expected))
    assert torch.equal(tracker.states["expert"], expected_tracker.states["expert"])
    assert torch.equal(tracker.states["model"], expected_tracker.states["model"])
    assert torch.equal(torch.get_rng_state(), outside)


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_qkv_norm_registers_before_weight_and_preserves_dense_cpu_master_init(
    modules,
    normalization,
    zero_centered,
):
    cfg = config(
        params_dtype=torch.bfloat16, normalization=normalization, layernorm_zero_centered_gamma=zero_centered
    )
    calls = []

    def initialize(tensor):
        calls.append((tuple(tensor.shape), tensor.dtype))
        torch.nn.init.normal_(tensor, std=0.2)

    torch.manual_seed(317)
    expected = torch.empty(48, 16, dtype=torch.float32).normal_(std=0.2).to(torch.bfloat16)
    state = torch.get_rng_state().clone()
    torch.manual_seed(317)
    layer = modules.M.TwoLevelFP4LayerNormColumnParallelLinear(
        16,
        48,
        config=cfg,
        init_method=initialize,
        bias=True,
    )
    names = ["layer_norm_weight"]
    if normalization == "LayerNorm":
        names.append("layer_norm_bias")
        assert torch.count_nonzero(layer.layer_norm_bias) == 0
    assert list(dict(layer.named_parameters(recurse=False))) == names + ["weight", "bias"]
    assert torch.all(layer.layer_norm_weight == (0 if zero_centered else 1))
    assert calls == [((48, 16), torch.float32)]
    assert torch.equal(layer.weight, expected)
    assert torch.equal(torch.get_rng_state(), state)


def test_skip_initialization_does_not_call_weight_initializer_or_tracker(modules, monkeypatch):
    cfg = config(perform_initialization=False, use_cpu_initialization=False)
    tracker = Tracker()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(modules.M, "get_cuda_rng_tracker", lambda: tracker)

    def forbidden(tensor):
        raise AssertionError("weight initialization was disabled")

    modules.M.TwoLevelFP4GroupedLinear(3, 19, 17, config=cfg, init_method=forbidden)
    modules.M.TwoLevelFP4LayerNormColumnParallelLinear(16, 48, config=cfg, init_method=forbidden)
    assert not tracker.calls
