# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Real private counter/JSON persistence with isolated external training APIs."""

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "primus/backends/megatron/core"
PATCHES = CORE.parent / "patches"


@pytest.fixture
def shared_hooks(monkeypatch, tmp_path):
    def package(name, path=None, **members):
        module = ModuleType(name)
        module.__path__ = [] if path is None else [str(path)]
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        parent, _, child = name.rpartition(".")
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    for name, path in (
        ("primus", ROOT / "primus"),
        ("primus.backends", CORE.parent.parent),
        ("primus.backends.megatron", CORE.parent),
        ("primus.backends.megatron.core", CORE),
        ("primus.backends.megatron.patches", PATCHES),
        ("primus.backends.megatron.patches.turbo", PATCHES / "turbo"),
        ("primus.core", ROOT / "primus/core"),
        ("primus.core.utils", ROOT / "primus/core/utils"),
    ):
        package(name, path)
    registrations = []

    def register(*args, **kwargs):
        registrations.append((args, kwargs))
        return lambda fn: fn

    package(
        "primus.core.patches",
        PatchContext=SimpleNamespace,
        get_args=lambda ctx: ctx.args,
        register_patch=register,
    )
    package("primus.core.utils.module_utils", log_rank_0=lambda *args: None)
    load("primus.backends.megatron.patches._patch_guard", PATCHES / "_patch_guard.py")
    dense_patch = load(
        "primus.backends.megatron.patches.turbo.dense_mlp_fp4_patches",
        PATCHES / "turbo/dense_mlp_fp4_patches.py",
    )
    prefix = "primus.backends.megatron.core"
    for name in ("fp4_contract", "fp4_dispatch", "fp4_rng", "fp4_reference", "fp4_triton"):
        load(prefix + "." + name, CORE / (name + ".py"))
    runtime = load(prefix + ".fp4_mxfp4_shared_sr", CORE / "fp4_mxfp4_shared_sr.py")
    for name in (
        "primus_turbo",
        "primus_turbo.pytorch",
        "primus_turbo.pytorch.ops",
        "primus_turbo.pytorch.kernels",
        "primus_turbo.pytorch.kernels.quantization",
    ):
        package(name)

    def native_dense(*args, **kwargs):
        raise AssertionError("Checkpoint hook tests must not launch GPU quantization")

    def native_grouped(*args, **kwargs):
        raise AssertionError("Checkpoint hook tests must not launch GPU quantization")

    functions = dict(quantize_mxfp4_impl=native_dense, grouped_quantize_mxfp4_impl=native_grouped)
    package("primus_turbo.pytorch.ops.quantization", **functions)
    package("primus_turbo.pytorch.kernels.quantization.quantization_impl", **functions)
    package("megatron")
    package("megatron.training")
    package("megatron.core")
    package("megatron.core.tensor_parallel")
    tracker = SimpleNamespace(states={"model": torch.arange(16, dtype=torch.uint8)})
    tracker.is_initialized = lambda: True
    tracker.get_states = lambda: tracker.states
    tracker.set_states = lambda states: setattr(tracker, "states", states)
    package("megatron.core.tensor_parallel.random", get_cuda_rng_tracker=lambda: tracker)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("PRIMUS_TURBO_FUSED_DENSE_MLP", raising=False)
    monkeypatch.delenv("PRIMUS_TURBO_FUSED_GEMM", raising=False)
    h = SimpleNamespace(
        args=SimpleNamespace(
            fp4="e2m1",
            fp4_recipe="mxfp4",
            fp4_gradient_rounding_policy="shared_philox",
            enable_primus_turbo=True,
            use_turbo_gemm=True,
            use_turbo_grouped_gemm=True,
            mxfp4_gradient_stochastic_rounding=True,
            transformer_impl="transformer_engine",
            seed=1234,
            fp4_rng_seed=1234,
            num_experts=4,
            mxfp4_scale_rounding_mode=2,
            save=str(tmp_path),
            load=str(tmp_path),
            finetune=False,
            iteration=0,
            disable_last_saving=False,
            train_iters=10,
        ),
        model=[torch.nn.Linear(16, 16, bias=False)],
        runtime=runtime,
        saves=[],
        loads=[],
        load_result=(3, 0.0),
        load_failure=False,
        dense_requested=dense_patch._fused_dense_mlp_requested,
        tracker=tracker,
    )

    def get_checkpoint_name(root, iteration, release=False, return_base_dir=False):
        assert return_base_dir
        return str(Path(root) / ("release" if release else f"iter_{iteration:07d}"))

    def save(iteration, model, optimizer, *args, **kwargs):
        h.saves.append((iteration, model, optimizer, args, kwargs))
        release = bool(args[11])
        directory = Path(get_checkpoint_name(h.args.save, iteration, release, return_base_dir=True))
        h.main_save_saw_sidecar = (directory / "shared_fp4_sr/rank_0.json").is_file()
        (Path(h.args.save) / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        return "main_saved"

    def restore(model, optimizer, scheduler, load_arg="load", *args, **kwargs):
        h.loads.append((model, optimizer, scheduler, load_arg, args, kwargs))
        if h.load_failure:
            runtime._reserve(31)
            raise RuntimeError("native load failed")
        return h.load_result

    h.training = package(
        "megatron.training.training",
        get_model=lambda *a, **k: h.model,
        save_checkpoint=save,
        load_checkpoint=restore,
    )
    h.checkpointing = package(
        "megatron.training.checkpointing",
        get_checkpoint_name=get_checkpoint_name,
        get_checkpoint_tracker_filename=lambda root: str(Path(root) / "latest_checkpointed_iteration.txt"),
    )
    h.patch = load(
        "primus.backends.megatron.patches.shared_fp4_sr_patches", PATCHES / "shared_fp4_sr_patches.py"
    )
    h.registrations = registrations
    h.install = lambda: h.patch.install_shared_mxfp4_training_state(h.args)
    h.save = lambda step=3: h.training.save_checkpoint(step, h.model, object(), object(), 0.0)
    h.load = lambda: h.training.load_checkpoint(h.model, object(), object())
    h.sidecar = (
        lambda step=3: Path(get_checkpoint_name(h.args.save, step, return_base_dir=True))
        / "shared_fp4_sr/rank_0.json"
    )
    yield h
    runtime.uninstall_shared_mxfp4_sr()


def test_opt_in_guard_leaves_recipe_policy_and_two_level_routes_untouched(shared_hooks):
    h = shared_hooks
    original = h.training.get_model, h.training.save_checkpoint, h.training.load_checkpoint
    for recipe, policy in (("mxfp4", "recipe"), ("nvfp4", "shared_philox"), ("amdfp4", "shared_philox")):
        h.args.fp4_recipe, h.args.fp4_gradient_rounding_policy = recipe, policy
        assert h.install() is False
        assert original == (h.training.get_model, h.training.save_checkpoint, h.training.load_checkpoint)
        assert not h.runtime.is_installed()


@pytest.mark.parametrize(
    "field",
    [
        "turbo_fused_gemm",
        "use_turbo_fused_dense_mlp",
        "turbo_fused_grouped_gemm",
        "moe_permute_quant_fusion",
        "moe_backward_permute_quant_fusion",
        "use_turbo_mega_moe",
        "use_turbo_deepep",
        "fp4_use_native_te_autocast",
        "moe_use_legacy_grouped_gemm",
        "use_te_op_fuser",
        "use_transformer_engine_op_fuser",
        "use_grouped_gemm_for_dense_mlp",
    ],
)
def test_unintercepted_routes_are_rejected_before_mutation(shared_hooks, field):
    h = shared_hooks
    original = h.training.get_model
    setattr(h.args, field, True)
    with pytest.raises(ValueError, match=field):
        h.install()
    assert h.training.get_model is original and not h.runtime.is_installed()


@pytest.mark.parametrize(
    "dense_env,gemm_env,args_enabled,expected",
    [
        (None, None, False, False),
        (None, None, True, True),
        ("1", None, False, True),
        (None, "1", False, True),
        ("true", "0", False, True),
        ("0", "1", True, False),
        ("false", "1", True, False),
        ("False", "1", True, False),
        ("", "1", True, False),
        (None, "0", True, False),
        (None, "false", True, False),
        (None, "False", True, False),
        (None, "", True, False),
        ("FALSE", "0", False, True),  # The native producer treats this as true.
    ],
)
def test_dense_fusion_environment_uses_actual_producer_priority(
    shared_hooks, monkeypatch, dense_env, gemm_env, args_enabled, expected
):
    h = shared_hooks
    for name, value in (("PRIMUS_TURBO_FUSED_DENSE_MLP", dense_env), ("PRIMUS_TURBO_FUSED_GEMM", gemm_env)):
        if value is not None:
            monkeypatch.setenv(name, value)
    h.args.turbo_fused_gemm = h.args.use_turbo_fused_dense_mlp = args_enabled
    assert h.dense_requested(h.args) is expected
    original = h.training.get_model
    if expected:
        with pytest.raises(ValueError, match="fused dense MLP"):
            h.install()
        assert h.training.get_model is original and not h.runtime.is_installed()
    else:
        assert h.install()


def test_dense_fusion_environment_is_part_of_checkpoint_identity(shared_hooks, monkeypatch):
    h = shared_hooks
    monkeypatch.setenv("PRIMUS_TURBO_FUSED_DENSE_MLP", "0")
    monkeypatch.setenv("PRIMUS_TURBO_FUSED_GEMM", "1")
    h.install()
    h.training.get_model()
    h.save()
    stored = json.loads(h.sidecar().read_text())["identity"]["environment"]
    assert stored["PRIMUS_TURBO_FUSED_DENSE_MLP"] == "0"
    assert stored["PRIMUS_TURBO_FUSED_GEMM"] == "1"
    before = h.runtime.state_dict()
    monkeypatch.setenv("PRIMUS_TURBO_FUSED_GEMM", "0")
    with pytest.raises(ValueError, match="identity"):
        h.load()
    assert not h.loads and h.runtime.state_dict() == before


def test_late_fused_dense_override_is_rejected_before_model_build(shared_hooks, monkeypatch):
    h = shared_hooks
    h.install()
    monkeypatch.setenv("PRIMUS_TURBO_FUSED_GEMM", "1")
    with pytest.raises(ValueError, match="fused dense MLP"):
        h.training.get_model()
    assert not h.runtime.is_installed()


def test_fused_activation_with_probabilities_does_not_bypass_shared_sr(shared_hooks):
    h = shared_hooks
    h.args.use_turbo_fused_act_with_probs = True
    assert h.install()


@pytest.mark.parametrize(
    "field,value",
    [
        ("turbo_grouped_gemm_without_padding", True),
        ("moe_router_padding_for_quantization", True),
        ("moe_pad_expert_input_to_capacity", True),
        ("moe_expert_capacity_factor", 1.25),
    ],
)
def test_changed_grouped_padding_or_capacity_rejects_continuation(shared_hooks, field, value):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.save()
    before = h.runtime.state_dict()
    setattr(h.args, field, value)
    with pytest.raises(ValueError, match="identity"):
        h.load()
    assert not h.loads and h.runtime.state_dict() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_primus_turbo", False),
        ("use_turbo_gemm", False),
        ("use_turbo_grouped_gemm", False),
        ("mxfp4_gradient_stochastic_rounding", False),
        ("moe_grouped_gemm", False),
        ("transformer_impl", "local"),
        ("warmup_train_steps", 2),
        ("cuda_graph_impl", "local"),
        ("turbo_sync_free_moe_stage", 2),
    ],
)
def test_missing_native_route_or_unsupported_warmup_is_rejected(shared_hooks, field, value):
    h = shared_hooks
    setattr(h.args, field, value)
    with pytest.raises(ValueError):
        h.install()


def test_before_train_registration_and_deferred_install_keep_counter(shared_hooks, monkeypatch):
    h = shared_hooks
    registration = next(
        item for item in h.registrations if item[0][0] == "megatron.shared_fp4_sr.training_state"
    )
    assert registration[1]["phase"] == "before_train"
    assert registration[1]["priority"] > 100
    assert h.install()
    assert not h.runtime.is_installed()
    wrappers = h.training.get_model, h.training.save_checkpoint, h.training.load_checkpoint
    assert h.install() and wrappers == (
        h.training.get_model,
        h.training.save_checkpoint,
        h.training.load_checkpoint,
    )
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert h.training.get_model() is h.model
    assert h.runtime.state_dict()["topology"] == {"rank": 1, "world_size": 2}
    h.runtime._reserve(17)
    saved = h.runtime.state_dict()
    h.training.get_model()
    assert h.runtime.state_dict() == saved


def test_real_sidecar_precedes_main_save_and_resumes_counter_exactly(shared_hooks):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.runtime._reserve(17)
    assert h.save() == "main_saved" and h.main_save_saw_sidecar
    saved = h.runtime.state_dict()
    assert json.loads(h.sidecar().read_text())["runtime"] == saved
    expected = h.runtime._reserve(71)
    h.runtime._reserve(199)
    assert h.load() == (3, 0.0)
    assert h.runtime.state_dict() == saved
    assert h.runtime._reserve(71) == expected


@pytest.mark.parametrize(
    "mutation",
    ["version", "iteration", "policy", "torch_version", "identity", "rank", "world_size", "offset"],
)
def test_bad_checkpoint_is_rejected_before_native_load_and_rng_mutation(shared_hooks, mutation):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.save()
    h.runtime._reserve(89)
    before = h.runtime.state_dict()
    state = json.loads(h.sidecar().read_text())
    if mutation in ("rank", "world_size"):
        state["runtime"]["topology"][mutation] = 9
    elif mutation == "offset":
        state["runtime"]["rng"]["sr_offset"] = -1
    elif mutation == "identity":
        state["identity"]["config"]["mxfp4_scale_rounding_mode"] = 0
    else:
        state[mutation] = "invalid"
    h.sidecar().write_text(json.dumps(state))
    with pytest.raises(ValueError):
        h.load()
    assert not h.loads
    assert h.runtime.state_dict() == before


def test_missing_sidecar_and_changed_model_fail_before_native_load(shared_hooks):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.save()
    saved_file = h.sidecar().read_bytes()
    h.sidecar().unlink()
    with pytest.raises(FileNotFoundError):
        h.load()
    h.sidecar().write_bytes(saved_file)
    h.model = [torch.nn.Linear(16, 32, bias=False)]
    with pytest.raises(ValueError, match="identity"):
        h.load()
    assert not h.loads


def test_failed_native_load_rolls_back_private_counter(shared_hooks):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.save()
    h.runtime._reserve(17)
    before = h.runtime.state_dict()
    h.load_failure = True
    with pytest.raises(RuntimeError, match="native load failed"):
        h.load()
    assert h.runtime.state_dict() == before


def test_named_resume_directory_and_native_load_options_are_preserved(shared_hooks):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.save()
    h.args.resume = h.args.save
    context = object()
    h.training.load_checkpoint(h.model, None, None, "resume", False, context, True)
    assert h.loads[-1][3:] == ("resume", (False, context, True), {})


def test_nonpersistent_save_rejected_and_release_positional_binding_preserved(shared_hooks):
    h = shared_hooks
    h.install()
    h.training.get_model()
    with pytest.raises(ValueError, match="persistent"):
        h.training.save_checkpoint(3, h.model, None, None, 0, None, None, None, None, None, None, True)
    assert not h.saves
    h.training.save_checkpoint(
        3, h.model, None, None, 0, None, None, None, None, None, None, False, None, None, True
    )
    assert (Path(h.args.save) / "release/shared_fp4_sr/rank_0.json").exists()
    assert h.saves[-1][3][11] is True


def test_disabled_last_save_and_finetune_do_not_require_sidecar(shared_hooks):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.args.disable_last_saving = True
    h.save(10)
    assert not h.sidecar(10).exists()
    h.args.finetune = True
    before = h.runtime.state_dict()
    h.load()
    assert h.runtime.state_dict() == before


@pytest.mark.parametrize("failure", [False, True])
def test_synthetic_warmup_rewinds_real_private_reservations_even_on_exception(shared_hooks, failure):
    h = shared_hooks
    h.install()
    h.training.get_model()
    h.runtime._reserve(31)
    before = h.runtime.state_dict()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    tracker_rng = copy.deepcopy(h.tracker.states)

    def original(*args):
        h.runtime._reserve(191)
        random.random()
        np.random.random()
        torch.rand(7)
        h.tracker.states["model"].add_(1)
        if failure:
            raise RuntimeError("warmup failed")
        return "warm"

    module = SimpleNamespace(run_synthetic_warmup=original)
    assert h.patch.install_shared_sr_warmup(module)
    wrapped = module.run_synthetic_warmup
    assert h.patch.install_shared_sr_warmup(module) and module.run_synthetic_warmup is wrapped
    call = (None, None, h.model, None, None, None, h.args)
    if failure:
        with pytest.raises(RuntimeError, match="warmup failed"):
            wrapped(*call)
    else:
        assert wrapped(*call) == "warm"
    assert h.runtime.state_dict() == before
    assert random.getstate() == python_rng
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_rng[0] and current_numpy[2:] == numpy_rng[2:]
    assert np.array_equal(current_numpy[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(h.tracker.states["model"], tracker_rng["model"])
    expected = h.runtime._reserve(29)
    h.runtime.load_state_dict(before)
    assert h.runtime._reserve(29) == expected


@pytest.mark.parametrize("continuation", ["iteration", "consumed", "scheduler", "adam_moments"])
def test_continuation_skips_destructive_warmup_and_preserves_adam(shared_hooks, continuation):
    h = shared_hooks
    h.install()
    h.training.get_model()
    parameter = next(h.model[0].parameters())
    optimizer = torch.optim.AdamW([parameter])
    scheduler = SimpleNamespace(num_steps=0)
    if continuation == "iteration":
        h.args.iteration = 7
    elif continuation == "consumed":
        h.args.consumed_train_samples = 32
    elif continuation == "scheduler":
        scheduler.num_steps = 7
    else:
        optimizer.state[parameter] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.full_like(parameter, 0.125),
            "exp_avg_sq": torch.full_like(parameter, 0.25),
        }
    moments = copy.deepcopy(optimizer.state_dict())
    private = h.runtime.state_dict()
    calls = []

    def destructive(*args):
        calls.append(True)
        for state in optimizer.state.values():
            for value in state.values():
                value.zero_()
        h.runtime._reserve(191)

    wrapped = h.patch.wrap_shared_sr_warmup(destructive)
    assert wrapped(None, None, h.model, optimizer, scheduler, None, h.args) is None
    assert not calls and h.runtime.state_dict() == private
    current = optimizer.state_dict()
    assert current["param_groups"] == moments["param_groups"]
    assert current["state"].keys() == moments["state"].keys()
    for index in moments["state"]:
        for key in moments["state"][index]:
            assert torch.equal(current["state"][index][key], moments["state"][index][key])


def test_continuation_on_another_rank_skips_warmup_locally(shared_hooks, monkeypatch):
    h = shared_hooks
    h.install()
    h.training.get_model()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    reductions = []

    def other_rank_has_state(tensor, op):
        reductions.append((tensor.item(), op))
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", other_rank_has_state)
    calls = []
    wrapped = h.patch.wrap_shared_sr_warmup(lambda *args: calls.append(True))
    assert wrapped(None, None, h.model, None, None, None, h.args) is None
    assert not calls and reductions == [(0, torch.distributed.ReduceOp.MAX)]


def test_warmup_wrapper_is_noop_for_recipe_policy(shared_hooks):
    h = shared_hooks
    h.args.fp4_gradient_rounding_policy = "recipe"
    calls = []
    wrapped = h.patch.wrap_shared_sr_warmup(lambda *args: calls.append(args))
    wrapped(None, None, h.model, None, None, None, h.args)
    assert len(calls) == 1 and not h.runtime.is_installed()
