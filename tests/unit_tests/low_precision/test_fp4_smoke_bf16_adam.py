# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU checks for the opt-in GPU qualification and its evidence requirements.

Actual tensor dtypes, full shard comparisons, DeOsc numerics and two-process
Gloo aggregation are exercised here. These tests do not execute HIP Adam;
the default-bf16-adam torchrun smoke is the required GPU qualification.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_fp4_deosc import D, binding
from test_fp4_smoke_harness import SMOKE


def _args(*extra):
    return SMOKE.parse_args(["--recipe", "nvfp4", "--output", "/tmp/qualification.json", *extra])


PROFILE = (
    "--optimizer-profile",
    "default-bf16-adam",
    "--distributed-optimizer",
    "--deosc",
    "--checkpoint-format",
    "torch_dist",
)


def test_profile_is_opt_in_and_keeps_existing_fp32_semantics(monkeypatch):
    for key in SMOKE._DEFAULT_BF16_ADAM_ENV:
        monkeypatch.delenv(key, raising=False)
    original = _args()
    assert original.optimizer_profile == "fp32" and original.lr == 0.002
    assert SMOKE._configure_optimizer_profile(original) == {}
    old = SMOKE._distributed_optimizer_kwargs(original)
    assert old["clip_grad"] == old["weight_decay"] == 0
    assert "use_precision_aware_optimizer" not in old
    assert "exp_avg_dtype" not in old and "exp_avg_sq_dtype" not in old
    profile = _args(*PROFILE)
    assert profile.lr == 5.7e-4
    assert SMOKE._configure_optimizer_profile(profile) == SMOKE._DEFAULT_BF16_ADAM_ENV
    options = SMOKE._distributed_optimizer_kwargs(profile)
    assert options["use_precision_aware_optimizer"]
    assert options["main_params_dtype"] == torch.float32
    assert options["exp_avg_dtype"] == options["exp_avg_sq_dtype"] == torch.bfloat16
    assert (options["clip_grad"], options["weight_decay"], options["adam_eps"]) == (1.0, 0.1, 1e-5)


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--distributed-optimizer", "--checkpoint-format", "torch_dist"),
        ("--deosc", "--checkpoint-format", "torch_dist"),
    ],
)
def test_qualification_rejects_missing_sharding_deosc_or_checkpoint(extra):
    with pytest.raises(SystemExit):
        _args("--optimizer-profile", "default-bf16-adam", *extra)


def _leaf(model=None, master=None, *, start=2):
    if model is None:
        model = torch.zeros(8, dtype=torch.bfloat16)
        master = torch.arange(4, dtype=torch.float32) / 16
        model[start : start + master.numel()].copy_(master)
    else:
        start = 0
        model.detach().view(-1).copy_(master.reshape(-1))
    destination = model.detach().view(-1)[start : start + master.numel()]
    inner = SimpleNamespace(
        param_groups=[{"params": [master]}],
        state={
            master: {
                "exp_avg": torch.zeros_like(master, dtype=torch.bfloat16),
                "exp_avg_sq": torch.zeros_like(master, dtype=torch.bfloat16),
            }
        },
        _primus_bf16_writeback_by_param_ptr={master.data_ptr(): destination},
    )

    def fused_multi_tensor_adam():
        destination.copy_(master.reshape(-1))

    inner.multi_tensor_adam = fused_multi_tensor_adam
    master.decoupled_grad = torch.ones_like(master, dtype=torch.bfloat16)
    leaf = SimpleNamespace(
        optimizer=inner,
        config=SimpleNamespace(
            use_precision_aware_optimizer=True,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        ),
        model_float16_groups=[[model]],
        shard_fp32_from_float16_groups=[[master]],
        _get_model_param_range_map=lambda param: {
            "param": SimpleNamespace(start=start, end=start + master.numel())
        },
    )
    return leaf, master, model


def test_evidence_reads_actual_moments_and_checks_full_owned_shards(monkeypatch):
    monkeypatch.setattr(SMOKE, "_CHECK_CHUNK_ELEMENTS", 2)
    leaf, master, model = _leaf()
    evidence = SMOKE._default_optimizer_evidence(leaf)
    assert evidence["actual_dtypes"] == {
        "master": ["torch.float32"],
        "model": ["torch.bfloat16"],
        "exp_avg": ["torch.bfloat16"],
        "exp_avg_sq": ["torch.bfloat16"],
        "gradient": ["torch.bfloat16"],
    }
    assert evidence["owned_master_elements"] == 4
    assert evidence["all_owned_master_model_shards_exact"]
    # Corrupt the final element: exhaustive checking must not sample it away.
    model[5] += 1
    with pytest.raises(AssertionError, match="differs from its BF16 model shard"):
        SMOKE._default_optimizer_evidence(leaf)


@pytest.mark.parametrize("name", ["exp_avg", "exp_avg_sq"])
def test_recipe_bf16_cannot_hide_real_fp32_adam_state(name):
    leaf, master, _ = _leaf()
    leaf.config.exp_avg_dtype = leaf.config.exp_avg_sq_dtype = torch.bfloat16
    leaf.optimizer.state[master][name] = leaf.optimizer.state[master][name].float()
    with pytest.raises(AssertionError, match=f"Actual Adam {name} must be BF16"):
        SMOKE._default_optimizer_evidence(leaf)


def test_restore_checks_bf16_state_before_next_adam_or_writeback_cache_exists():
    leaf, master, _ = _leaf()
    del master.decoupled_grad
    del leaf.optimizer._primus_bf16_writeback_by_param_ptr
    evidence = SMOKE._default_optimizer_evidence(leaf, require_update=False)
    assert evidence["actual_dtypes"]["exp_avg"] == ["torch.bfloat16"]
    assert not evidence["all_owned_bf16_writeback_buffers_exact"]
    leaf.optimizer.state[master]["exp_avg_sq"] = torch.ones_like(master)
    with pytest.raises(AssertionError, match="exp_avg_sq must be BF16"):
        SMOKE._default_optimizer_evidence(leaf, require_update=False)


def test_complete_fingerprint_detects_actual_dtype_hidden_by_te_style_serialization():
    leaf, master, _ = _leaf()
    # TE's public state_dict upcasts BF16 moments, so raw state must be included.
    leaf.optimizer.state_dict = lambda: {
        name: value.float() for name, value in leaf.optimizer.state[master].items()
    }
    leaf.state_dict = lambda: {"step": 4}
    leaf.get_parameters = lambda: [master]
    before = SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(leaf))
    public_before = SMOKE._state_fingerprint(leaf.optimizer.state_dict())
    leaf.optimizer.state[master]["exp_avg"] = leaf.optimizer.state[master]["exp_avg"].float()
    assert SMOKE._state_fingerprint(leaf.optimizer.state_dict()) == public_before
    assert SMOKE._state_fingerprint(SMOKE._complete_optimizer_state(leaf)) != before


def test_model_match_does_not_hide_wrong_fused_writeback_destination():
    leaf, master, _ = _leaf()
    leaf.optimizer._primus_bf16_writeback_by_param_ptr[master.data_ptr()] = torch.zeros_like(
        master, dtype=torch.bfloat16
    )
    with pytest.raises(AssertionError, match="buffer is missing or stale"):
        SMOKE._default_optimizer_evidence(leaf)


def test_hidden_master_or_missing_parameter_coverage_cannot_qualify():
    leaf, _, _ = _leaf()
    leaf.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
    with pytest.raises(AssertionError, match="master ownership"):
        SMOKE._default_optimizer_evidence(leaf)
    leaf.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False
    leaf.optimizer.param_groups[0]["params"].append(torch.ones(3))
    with pytest.raises(AssertionError, match="every optimizer parameter"):
        SMOKE._default_optimizer_evidence(leaf)


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_observer_proves_real_deosc_changed_model_and_rejects_stale_copy(recipe):
    model, master, config = binding(recipe)
    leaf, master, model = _leaf(model, master)
    runner = D.TwoLevelDeOsc({"weight": (model, master, config)}, period=1)
    observer = SMOKE._DefaultBF16AdamObserver(leaf, [runner])
    before = SMOKE._state_fingerprint(master)
    leaf.optimizer.multi_tensor_adam()
    runner.step()
    first = observer.after_step()
    assert first["fused_adam_calls"] == 1 and first["deosc_snapped_local"] == 0
    assert SMOKE._state_fingerprint(master) == before

    observer.begin_step()
    master.view(-1)[0] = 0.75390625
    leaf.optimizer.multi_tensor_adam()
    runner.step()
    assert master.view(-1)[0].item() == 1
    with pytest.raises(AssertionError, match="differs from its BF16 model shard"):
        observer.after_step()
    model.data.copy_(master)
    second = observer.after_step()
    assert second["deosc_snapped_local"] == second["deosc_changed_bf16_local"] == 1
    assert second["all_owned_bf16_writeback_buffers_exact"]
    observer.begin_step()
    with pytest.raises(AssertionError, match="No actual Primus fused Adam"):
        observer.after_step()


def _gloo_totals_worker(rank, rendezvous, snap):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        rows = [
            {
                "optimizer_qualification": {
                    "fused_adam_calls": 1,
                    "deosc_snapped_local": rank if snap else 0,
                    "deosc_changed_bf16_local": rank if snap else 0,
                }
            }
        ]
        if snap:
            assert SMOKE._qualification_totals(rows, "cpu") == {
                "global_fused_adam_calls": 2,
                "global_deosc_snapped": 1,
                "global_deosc_changed_bf16": 1,
            }
        else:
            with pytest.raises(AssertionError, match="propagated BF16-changing DeOsc snap"):
                SMOKE._qualification_totals(rows, "cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("snap", [False, True])
def test_real_dp2_qualification_requires_global_snap_but_allows_zero_on_one_rank(tmp_path, snap):
    mp.spawn(_gloo_totals_worker, args=(str(tmp_path / "rendezvous"), snap), nprocs=2, join=True)
