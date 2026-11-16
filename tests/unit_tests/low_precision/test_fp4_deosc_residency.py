# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""DeOsc CPU history must change storage only, including snap and resume paths."""

import copy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_fp4_deosc_distributed import C, D, S, _delta, _init, _inputs, _tree_equal


def _runner(recipe, residency, *, device="cpu", sharded=False, accumulator="fp32", count=2, side=16):
    cfg = C.TwoLevelFP4Config(recipe, quantizer="triton" if device == "cuda" else "torch")
    bindings = {}
    for index in range(count):
        master = torch.full((side, side), 6.0, dtype=torch.float32, device=device)
        master.flatten()[0] = 0.74609375
        model = master.bfloat16()
        bindings[f"weight{index}"] = (
            S.FP4ShardBinding(model, master.flatten(), cfg, 0, master.numel())
            if sharded
            else (model, master, cfg)
        )
    cls = S.ShardedTwoLevelDeOsc if sharded else D.TwoLevelDeOsc
    return cls(bindings, period=2, ratio=4, accumulator_dtype=accumulator, state_residency=residency)


def _update(runner, index):
    for _, master, _ in runner.bindings.values():
        master.flatten()[0] = 0.75390625 if index % 2 else 0.74609375


def _assert_cpu_history(runner):
    assert runner.state_residency == "cpu"
    assert all(
        value.device.type == "cpu"
        for state in runner._states.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("accumulator", ["fp32", "bf16"])
def test_history_residency_preserves_every_update_and_checkpoint(recipe, sharded, accumulator):
    device = _runner(recipe, "device", sharded=sharded, accumulator=accumulator)
    host = _runner(recipe, "cpu", sharded=sharded, accumulator=accumulator)
    rng = torch.random.get_rng_state().clone()
    for index in range(7):
        for runner in (device, host):
            _update(runner, index)
            runner.step()
        _assert_cpu_history(host)
        _tree_equal(host.state_dict(), device.state_dict())
        for key in device.bindings:
            assert torch.equal(host.bindings[key][1], device.bindings[key][1])
    assert host.stats["snapped"] > 0
    assert torch.equal(rng, torch.random.get_rng_state())

    # Old numerical sidecars intentionally carry no residency field and load
    # into either storage mode, without losing a partially observed window.
    for source, target in ((device, host), (host, device)):
        saved = source.state_dict()
        assert "state_residency" not in saved["settings"]
        target.load_state_dict(saved)
        _tree_equal(source.state_dict(), target.state_dict())
        saved["states"]["weight0"]["prev"].zero_()
        assert target._states["weight0"]["prev"].count_nonzero() > 0
    _assert_cpu_history(host)


@pytest.mark.parametrize("invalid", [None, "gpu", "cuda", True, 0])
def test_invalid_residency_rejected(invalid):
    with pytest.raises(ValueError, match="state_residency"):
        _runner("nvfp4", invalid)


def test_failed_offload_cannot_continue_or_publish_partial_checkpoint(monkeypatch):
    runner = _runner("nvfp4", "cpu")
    runner.step()
    saved = runner.state_dict()
    original = runner._copy_history

    def fail_store(state, device):
        if device == "cpu":
            raise OSError("injected host allocation/copy failure")
        return original(state, device)

    monkeypatch.setattr(runner, "_copy_history", fail_store)
    with pytest.raises(OSError, match="injected"):
        runner.step()
    _assert_cpu_history(runner)
    with pytest.raises(RuntimeError, match="failed update"):
        runner.state_dict()
    with pytest.raises(RuntimeError, match="earlier update"):
        runner.step()
    broken = copy.deepcopy(saved)
    broken["states"]["weight0"]["prev"] = torch.empty(1)
    with pytest.raises(ValueError):
        runner.load_state_dict(broken)
    with pytest.raises(RuntimeError, match="failed update"):
        runner.state_dict()
    monkeypatch.setattr(runner, "_copy_history", original)
    runner.load_state_dict(saved)
    _tree_equal(runner.state_dict(), saved)
    runner.step()
    _assert_cpu_history(runner)


def test_failed_active_validation_also_blocks_sidecar_and_continuation(monkeypatch):
    runner = _runner("nvfp4", "cpu")
    runner.step()
    saved = runner.state_dict()

    def fail_active():
        raise ValueError("injected active/collective failure after Adam update")

    monkeypatch.setattr(runner, "_active_bindings", fail_active)
    with pytest.raises(ValueError, match="injected"):
        runner.step()
    assert runner.step_count == saved["step_count"]
    _assert_cpu_history(runner)
    with pytest.raises(RuntimeError, match="failed update"):
        runner.state_dict()
    with pytest.raises(RuntimeError, match="earlier update"):
        runner.step()
    runner.load_state_dict(saved)
    _tree_equal(runner.state_dict(), saved)


@pytest.mark.parametrize("distributed", [False, True])
def test_installer_propagates_cpu_history_and_keeps_model_copy_contract(distributed):
    from test_fp4_deosc import D as Installer
    from test_fp4_deosc import MockFloat16Optimizer, ReferenceModule, args

    model = ReferenceModule()
    optimizer = MockFloat16Optimizer(model, distributed=distributed)
    (runner,) = Installer.install_megatron_deosc(optimizer, [model], args(weight_deosc_state_residency="cpu"))
    model._primus_fp4_last_config = model.fp4_config
    assert optimizer.step_with_ready_grads() is True
    optimizer.master.flatten()[0] = 0.75390625
    assert optimizer.step_with_ready_grads() is True
    _assert_cpu_history(runner)
    assert runner.stats["snapped"] == 1
    assert model.weight.flatten()[0] == optimizer.master.flatten()[0] == 1


def _distributed_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        for recipe in ("nvfp4", "amdfp4"):
            cfg = C.TwoLevelFP4Config(recipe)
            full, bindings, ranges = _inputs(rank, True, cfg)
            cpu_bindings = {
                key: replace(value, master=value.master.clone()) for key, value in bindings.items()
            }
            device = S.ShardedTwoLevelDeOsc(bindings, process_group=dist.group.WORLD, period=2)
            host = S.ShardedTwoLevelDeOsc(
                cpu_bindings, process_group=dist.group.WORLD, period=2, state_residency="cpu"
            )
            for update in range(5):
                for runner in (device, host):
                    for key, item in runner.shard_bindings.items():
                        start, end = ranges[key]
                        item.master.add_(_delta(key, full[key][1], update).flatten()[start:end])
                    runner.step()
                _tree_equal(host.state_dict(), device.state_dict())
                _assert_cpu_history(host)
            # Configuration disagreement fails collectively before any history
            # work or rank-dependent transfer can leave peers waiting.
            with pytest.raises(ValueError, match="disagree on settings"):
                S.ShardedTwoLevelDeOsc(
                    bindings, process_group=dist.group.WORLD, state_residency="cpu" if rank == 0 else "device"
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_sharded_zero_owners_and_rank_residency_agreement(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "residency-gloo"),), nprocs=3, join=True)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for actual CPU/GPU transfer proof")
@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("sharded", [False, True])
def test_gpu_cpu_history_is_bitwise_equal_and_resumes(recipe, sharded):
    device = _runner(recipe, "device", device="cuda", sharded=sharded)
    host = _runner(recipe, "cpu", device="cuda", sharded=sharded)
    rng = torch.cuda.get_rng_state().clone()
    for index in range(5):
        for runner in (device, host):
            _update(runner, index)
            runner.step()
        _assert_cpu_history(host)
        _tree_equal(host.state_dict(), device.state_dict())
        for key in device.bindings:
            assert torch.equal(host.bindings[key][1], device.bindings[key][1])
        if index == 1:
            cpu_saved, device_saved = host.state_dict(), device.state_dict()
            assert all(state["window_steps"] == 1 for state in cpu_saved["states"].values())
            host.load_state_dict(device_saved)
            device.load_state_dict(cpu_saved)
            _assert_cpu_history(host)
            assert all(
                t.device.type == "cuda"
                for state in device._states.values()
                for t in state.values()
                if isinstance(t, torch.Tensor)
            )
    assert torch.equal(rng, torch.cuda.get_rng_state())
    saved = device.state_dict()
    host.load_state_dict(saved)
    _assert_cpu_history(host)
    _tree_equal(host.state_dict(), saved)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for allocated-memory proof")
def test_gpu_history_does_not_accumulate_across_parameters(monkeypatch):
    runner = _runner("nvfp4", "cpu", device="cuda", count=3, side=1024)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    before_each = []
    original = runner._observe_binding

    def observe(*args):
        torch.cuda.synchronize()
        before_each.append(torch.cuda.memory_allocated())
        return original(*args)

    monkeypatch.setattr(runner, "_observe_binding", observe)
    for _ in range(3):
        runner.step()
        _assert_cpu_history(runner)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - baseline < 2 * 1024**2
    assert max(before_each) - min(before_each) < 2 * 1024**2
    history_bytes = sum(
        t.numel() * t.element_size()
        for state in runner._states.values()
        for t in state.values()
        if isinstance(t, torch.Tensor)
    )
    assert history_bytes == 3 * 1024 * 1024 * 12
    # Restore that exact same sidecar in default mode: GPU history is expected
    # to consume its full size, establishing sensitivity of the memory check.
    runner.state_residency = "device"
    runner.load_state_dict(runner.state_dict())
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - baseline >= history_bytes
