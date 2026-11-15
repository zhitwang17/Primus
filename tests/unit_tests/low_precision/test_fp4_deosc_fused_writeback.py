# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Exercise the production fused-copy callback with real CPU SGD and DeOsc.

Only the HIP Adam launch is substituted: SGD writes its updated FP32 master to
BF16 early, as the fused kernel does. The actual nested callback is compiled
from its source rather than imitating its skip logic. These CPU tests do not
claim to qualify the HIP kernel or BF16 Adam moments; GPU smoke must validate
those separately.
"""

import ast
import copy
from types import SimpleNamespace

import pytest
import torch
from test_fp4_deosc_updates import D, ROOT, Weight, accumulate, assert_tree_equal


@pytest.fixture
def fused_copy():
    path = ROOT / "primus/backends/megatron/patches/te_patches/fused_adam_clip_patches.py"
    installer = next(
        item
        for item in ast.parse(path.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "_install_patch"
    )
    callback = next(
        item
        for item in installer.body
        if isinstance(item, ast.FunctionDef) and item.name == "patched_copy_main_params_to_model_params"
    )

    @torch.no_grad()
    def ordinary_copy(optimizer):
        optimizer.ordinary_copies += 1
        optimizer.model.weight.copy_(optimizer.master.view_as(optimizer.model.weight))

    def copy_fp32_groups(optimizer):
        optimizer.fast_copies += 1

    namespace = {
        "original_copy_main_params": ordinary_copy,
        "_copy_only_fp32_model_groups": copy_fp32_groups,
    }
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[callback.name]


class FusedWritebackAdapter:
    """Explicit FP32 masters with the real production BF16-copy skip callback."""

    def __init__(self, model, callback, distributed):
        self.model = model
        master = model.weight.detach().float().clone()
        self.master = torch.nn.Parameter(master.flatten() if distributed else master)
        self.optimizer = torch.optim.SGD([self.master], lr=1 / 128)
        self.config = SimpleNamespace(
            use_precision_aware_optimizer=True,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        )
        self.successful = True
        self.ordinary_copies = self.fast_copies = self.early_writebacks = 0
        self._primus_bf16_writeback_active = False
        self._copy_main_params_to_model_params = callback.__get__(self)
        if distributed:
            self.model_float16_groups = [[model.weight]]
            self.shard_fp32_from_float16_groups = [[self.master]]
            self.model_param_gbuf_map = {model.weight: (0, (torch.bfloat16, torch.float32), 0)}
            self.buffers = [
                SimpleNamespace(
                    param_index_map={model.weight: (17, 17 + master.numel(), 0)},
                    data_parallel_group=None,
                )
            ]
        else:
            self.float16_groups = [[model.weight]]
            self.fp32_from_float16_groups = [[self.master]]

    def _get_model_param_range_map(self, parameter):
        assert parameter is self.model.weight
        return {"param": SimpleNamespace(start=0, end=parameter.numel())}

    @torch.no_grad()
    def step_with_ready_grads(self):
        if not self.successful:
            return False
        self.master.grad = self.model.weight.grad.float().reshape_as(self.master).clone()
        self.optimizer.step()
        # The HIP fused Adam kernel writes BF16 before the DeOsc copy hook.
        self.model.weight.copy_(self.master.view_as(self.model.weight))
        self.early_writebacks += 1
        self._primus_bf16_writeback_active = True
        self._copy_main_params_to_model_params()
        return True

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("distributed", [False, True])
def test_snap_rewrites_bf16_without_disabling_fused_writeback(recipe, distributed, fused_copy):
    model = Weight(recipe)
    optimizer = FusedWritebackAdapter(model, fused_copy, distributed)
    options = SimpleNamespace(
        weight_deosc=True,
        weight_deosc_period=1,
        weight_deosc_ratio=4,
        use_distributed_optimizer=distributed,
        use_precision_aware_optimizer=True,
    )
    runner = D.install_megatron_deosc(optimizer, [model], options)[0]
    assert hasattr(runner, "shard_bindings") == distributed

    # Constructor/reload copies preserve normal ownership and do not observe.
    optimizer._copy_main_params_to_model_params()
    assert optimizer.ordinary_copies == 1 and runner.step_count == 0
    assert optimizer._primus_bf16_writeback_active is False

    accumulate(model, 1)
    assert optimizer.step_with_ready_grads()
    assert runner.stats["snapped"] == 0
    assert model.weight.flatten()[0].item() == 0.74609375
    assert optimizer.fast_copies == 1 and optimizer.ordinary_copies == 1
    assert optimizer._primus_bf16_writeback_active is False

    # Failed updates never alter either DeOsc or writeback ownership.
    optimizer.zero_grad()
    accumulate(model, -1)
    optimizer.successful = False
    before = copy.deepcopy(runner.state_dict())
    assert optimizer.step_with_ready_grads() is False
    assert_tree_equal(runner.state_dict(), before)
    assert optimizer.early_writebacks == 1 and optimizer.ordinary_copies == 1
    assert optimizer._primus_bf16_writeback_active is False

    optimizer.successful = True
    assert optimizer.step_with_ready_grads()
    assert runner.stats["snapped"] == 1
    assert optimizer.master.flatten()[0].item() == 1.0
    # Without invalidation, the production callback leaves 0.75390625 here.
    assert model.weight.flatten()[0].item() == 1.0
    assert optimizer.ordinary_copies == 2 and optimizer.fast_copies == 1
    assert optimizer._primus_bf16_writeback_active is False

    # A later update with no snap uses the original fused skip fast path again.
    optimizer.zero_grad()
    accumulate(model, 0)
    assert optimizer.step_with_ready_grads()
    assert runner.stats["snapped"] == 1
    assert optimizer.early_writebacks == 3 and optimizer.fast_copies == 2
    assert optimizer.ordinary_copies == 2
    assert optimizer._primus_bf16_writeback_active is False
    assert torch.equal(model.weight, optimizer.master.view_as(model.weight).bfloat16())

    state = copy.deepcopy(runner.state_dict())
    optimizer._copy_main_params_to_model_params()
    assert_tree_equal(runner.state_dict(), state)
    assert optimizer.ordinary_copies == 3
