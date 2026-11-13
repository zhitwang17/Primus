# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Real CPU FP4/optimizer state through the optional MLPerf warmup boundary."""

import copy
import importlib
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_fp4_training_hooks import hooks  # noqa: F401: pytest fixture


@pytest.fixture
def warm(hooks, monkeypatch):
    h = hooks
    h.W = importlib.import_module("primus.backends.megatron.core.fp4_warmup")
    # The hooks fixture reloads the actual contract class for each test.
    monkeypatch.delitem(sys.modules, "primus.backends.megatron.core.fp4_autograd", raising=False)
    h.A = importlib.import_module("primus.backends.megatron.core.fp4_autograd")
    h.patch = importlib.import_module("primus.backends.megatron.patches.two_level_fp4_state_patches")
    h.args.fp4, h.args.fp4_recipe, h.args.fp4_backend = "e2m1", "nvfp4", "reference"
    h.args.consumed_train_samples = 0
    h.args.seq_length, h.args.micro_batch_size, h.args.padded_vocab_size = 16, 1, 32
    h.args.overlap_grad_reduce = False
    h.config = SimpleNamespace(finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None)
    h.logs = []
    h.tracker = SimpleNamespace(states={"model": torch.arange(16, dtype=torch.uint8)})
    h.tracker.is_initialized = lambda: True
    h.tracker.get_states = lambda: h.tracker.states
    h.tracker.set_states = lambda states: setattr(h.tracker, "states", states)

    def package(name, **members):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    package("megatron.core")
    package("megatron.core.tensor_parallel")
    package("megatron.core.tensor_parallel.random", get_cuda_rng_tracker=lambda: h.tracker)
    package("megatron.core.pipeline_parallel", get_forward_backward_func=lambda: None)
    package("megatron.core.distributed", finalize_model_grads=lambda *a: None)
    package(
        "megatron.core.distributed.distributed_data_parallel", DistributedDataParallel=type("DDP", (), {})
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("SYNTH_WARMUP_STEPS", "3")
    monkeypatch.delenv("WARMUP_RECIPE", raising=False)
    monkeypatch.delenv("WARMUP_FP8_RECIPE", raising=False)
    return h


class Optimizer:
    def __init__(self, h, adam=False):
        self.model = h.models[0]
        self.master = torch.nn.Parameter(self.model.weight.detach().float().clone())
        self.optimizer = (torch.optim.AdamW if adam else torch.optim.SGD)([self.master], lr=0.01)
        self.calls, self.successful_updates = 0, 0
        self.runner = h.deosc.TwoLevelDeOsc(
            {"weight": (self.model.weight, self.master, self.model.fp4_config)},
            period=2,
            start_step=0,
        )
        self._primus_two_level_deosc_runners = [self.runner]

    def scale_loss(self, loss):
        return loss

    def step(self):
        self.master.grad = self.model.weight.grad.float()
        self.optimizer.step()
        self.runner.step()
        with torch.no_grad():
            self.model.weight.copy_(self.master)
        self.calls += 1
        if not getattr(self, "_primus_fp4_warmup_active", False):
            self.successful_updates += 1

    def reload_model_params(self):
        with torch.no_grad():
            self.master.copy_(self.model.weight)


def make_training(h, adam=False):
    h.state.bind_fp4_model(h.models)
    optimizer = Optimizer(h, adam)
    scheduler = SimpleNamespace(num_steps=0, step=lambda increment: None)

    def train_step(forward, data, model, opt, sched, config, fwd_bwd, iteration=0):
        model[0].zero_grad(set_to_none=True)
        opt.optimizer.zero_grad(set_to_none=True)
        if data is not None:
            next(data)
        x = (torch.arange(16 * 16).reshape(16, 16).float() / 256).to(torch.bfloat16).requires_grad_()
        layer = model[0]
        layer._primus_fp4_last_config = layer.fp4_config
        loss = h.A.linear(x, layer.weight, layer.fp4_config, layer.fp4_rng).float().square().mean()
        loss.backward()
        opt.step()
        sched.num_steps += 1
        random.random()
        np.random.random()
        h.tracker.states["model"].add_(1)
        return loss.item()

    return optimizer, scheduler, train_step


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


def actual_warmup():
    # Resolve only the package location; importing its __init__ would load a
    # trainer and native dependencies. Execute the installed warmup file as-is.
    spec = importlib.util.find_spec("primus_mllog")
    if spec is None:
        pytest.skip("Installed primus_mllog is needed for the real warmup-body test")
    path = Path(next(iter(spec.submodule_search_locations))) / "warmup.py"
    module_spec = importlib.util.spec_from_file_location("_fp4_test_actual_mllog_warmup", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_real_mllog_warmup_rewinds_fp4_and_matches_first_real_update(warm):
    h = warm
    optimizer, scheduler, train_step = make_training(h)
    private = copy.deepcopy(h.state.fp4_state_dict(h.models, optimizer))
    rng = h.W._capture_rng()
    original_model = copy.deepcopy(h.models[0].state_dict())
    expected_loss = train_step(None, None, h.models, optimizer, scheduler, h.config, None)
    expected = dict(
        model=copy.deepcopy(h.models[0].state_dict()),
        optimizer=copy.deepcopy(optimizer.optimizer.state_dict()),
        private=copy.deepcopy(h.state.fp4_state_dict(h.models, optimizer)),
    )
    h.models[0].load_state_dict(original_model)
    optimizer, scheduler, train_step = make_training(h)
    h.state.load_fp4_state_dict(h.models, private, optimizer)
    h.W._restore_rng(rng)
    wrapped = h.W.wrap_synthetic_warmup(actual_warmup().run_synthetic_warmup, h.logs.append)
    wrapped(train_step, None, h.models, optimizer, scheduler, h.config, h.args)
    assert optimizer.calls == 3 and optimizer.successful_updates == 0
    assert_nested_equal(h.state.fp4_state_dict(h.models, optimizer), private)
    assert scheduler.num_steps == 0
    assert_nested_equal(h.W._capture_rng(), rng)
    assert not hasattr(optimizer, "_primus_fp4_warmup_active")
    assert not hasattr(h.args, "_primus_fp4_warmup_active")
    assert train_step(None, None, h.models, optimizer, scheduler, h.config, None) == expected_loss
    assert optimizer.successful_updates == 1
    assert_nested_equal(h.models[0].state_dict(), expected["model"])
    assert_nested_equal(optimizer.optimizer.state_dict(), expected["optimizer"])
    assert_nested_equal(h.state.fp4_state_dict(h.models, optimizer), expected["private"])


@pytest.mark.parametrize("kind", ["iteration", "consumed", "scheduler", "deosc", "adam", "zero_adam"])
def test_resume_and_existing_optimizer_states_skip_destructive_warmup(warm, kind):
    h = warm
    optimizer, scheduler, train_step = make_training(h, adam=True)
    if kind == "iteration":
        h.args.iteration = 77
    elif kind == "consumed":
        h.args.consumed_train_samples = 32
    elif kind == "scheduler":
        scheduler.num_steps = 32
    elif kind == "deosc":
        optimizer.runner.step()
    else:
        optimizer.master.grad = torch.ones_like(optimizer.master)
        optimizer.optimizer.step()
        if kind == "zero_adam":
            for value in optimizer.optimizer.state[optimizer.master].values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
    before = copy.deepcopy(optimizer.optimizer.state_dict())
    master = optimizer.master.detach().clone()

    def forbidden(*args):
        raise AssertionError("Resumed/preallocated Adam state would have been destroyed")

    h.W.wrap_synthetic_warmup(forbidden, h.logs.append)(
        train_step,
        None,
        h.models,
        optimizer,
        scheduler,
        h.config,
        h.args,
    )
    assert_nested_equal(optimizer.optimizer.state_dict(), before)
    assert torch.equal(optimizer.master, master)
    assert "real training continues normally" in h.logs[0]


def test_disabled_and_native_mx_do_not_snapshot_or_inspect_model(warm, monkeypatch):
    h = warm

    def forbidden(*args):
        raise AssertionError("Unexpected state snapshot")

    monkeypatch.setattr(h.W, "_capture_rng", forbidden)
    wrapped = h.W.wrap_synthetic_warmup(lambda *args: "original", h.logs.append)
    monkeypatch.setenv("SYNTH_WARMUP_STEPS", "0")
    assert wrapped(None, None, object(), object(), None, object(), h.args) == "original"
    monkeypatch.setenv("SYNTH_WARMUP_STEPS", "3")
    h.args.fp4_recipe = "mxfp4"
    assert wrapped(None, None, object(), object(), None, object(), h.args) == "original"


@pytest.mark.parametrize("env", ["WARMUP_RECIPE", "WARMUP_FP8_RECIPE"])
def test_foreign_warmup_autocast_is_rejected_before_mutation(warm, monkeypatch, env):
    h = warm
    optimizer, scheduler, train_step = make_training(h)
    monkeypatch.setenv(env, "fp4_nvfp4")
    with pytest.raises(ValueError, match="leave WARMUP_RECIPE"):
        h.W.wrap_synthetic_warmup(lambda *args: None, h.logs.append)(
            train_step,
            None,
            h.models,
            optimizer,
            scheduler,
            h.config,
            h.args,
        )
    assert not optimizer.calls


def test_failed_warmup_propagates_and_rewinds_private_state_rng_and_flags(warm):
    h = warm
    optimizer, scheduler, train_step = make_training(h)
    private = copy.deepcopy(h.state.fp4_state_dict(h.models, optimizer))
    rng = h.W._capture_rng()

    def fail(step, forward, model, opt, sched, config, args):
        assert opt._primus_fp4_warmup_active and args._primus_fp4_warmup_active
        step(forward, None, model, opt, sched, config, None)
        args.consumed_train_samples = 999
        config.no_sync_func = object()
        raise RuntimeError("injected warmup failure")

    with pytest.raises(RuntimeError, match="injected warmup failure"):
        h.W.wrap_synthetic_warmup(fail, h.logs.append)(
            train_step,
            None,
            h.models,
            optimizer,
            scheduler,
            h.config,
            h.args,
        )
    assert_nested_equal(h.state.fp4_state_dict(h.models, optimizer), private)
    assert_nested_equal(h.W._capture_rng(), rng)
    assert h.args.consumed_train_samples == 0 and h.config.no_sync_func is None
    assert not hasattr(optimizer, "_primus_fp4_warmup_active")


def test_installer_is_idempotent_and_updates_the_locally_imported_entrypoint(warm):
    module = SimpleNamespace(run_synthetic_warmup=lambda *args: None)
    assert warm.patch.install_two_level_fp4_warmup_state(module)
    first = module.run_synthetic_warmup
    assert warm.patch.install_two_level_fp4_warmup_state(module)
    assert module.run_synthetic_warmup is first


def _empty_owner_worker(rank, path, source_root):
    from datetime import timedelta

    # Load the real standalone core through the preflight loader, without
    # Primus/Megatron initialization. No GPU is exposed to this Gloo test.
    spec = importlib.util.spec_from_file_location(
        "_warmup_gloo_preflight", Path(source_root) / "tools/fp4_preflight.py"
    )
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    contract, _ = preflight.load_contracts()
    wrapper = importlib.import_module(contract.__package__ + ".fp4_warmup")
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        args = SimpleNamespace(fp4_recipe="nvfp4", fp4_backend="reference", iteration=0)
        optimizer = SimpleNamespace(state={"owned": {"step": 0}} if rank else {})
        log = []

        def forbidden(*args):
            raise AssertionError("All ranks must skip when any owner has optimizer state")

        wrapper.wrap_synthetic_warmup(forbidden, log.append)(
            None,
            None,
            object(),
            optimizer,
            None,
            object(),
            args,
        )
        assert len(log) == 1 and "Synthetic warmup skipped" in log[0]
        # Prove both processes remain in the same collective sequence.
        count = torch.tensor([1])
        dist.all_reduce(count)
        assert count.item() == 2
    finally:
        dist.destroy_process_group()


def test_empty_owner_rank_collectively_skips_warmup_when_peer_has_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_WARMUP_STEPS", "1")
    source_root = Path(__file__).resolve().parents[3]
    mp.spawn(
        _empty_owner_worker, args=(str(tmp_path / "warmup-rendezvous"), str(source_root)), nprocs=2, join=True
    )
