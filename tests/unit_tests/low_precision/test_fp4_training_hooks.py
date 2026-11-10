# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Actual FP4 training wrappers with CPU state and fake external Megatron entrypoints.

These tests load the implementation files, including real binding, DeOsc and
sidecar serialization. Only framework initialization and the external training
entrypoints are isolated; no implementation bodies are copied through AST.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "primus/backends/megatron/core"
PATCHES = CORE.parent / "patches"


@pytest.fixture
def hooks(monkeypatch, tmp_path):
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
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    for name, path in (
        ("primus", ROOT / "primus"),
        ("primus.backends", CORE.parent.parent),
        ("primus.backends.megatron", CORE.parent),
        ("primus.backends.megatron.core", CORE),
        ("primus.backends.megatron.core.optimizer", CORE / "optimizer"),
        ("primus.backends.megatron.patches", PATCHES),
        ("primus.core", ROOT / "primus/core"),
        ("primus.core.utils", ROOT / "primus/core/utils"),
    ):
        package(name, path)
    package(
        "primus.core.patches",
        PatchContext=SimpleNamespace,
        get_args=lambda ctx: ctx.args,
        register_patch=lambda *args, **kwargs: lambda function: function,
    )
    package("primus.core.utils.module_utils", log_rank_0=lambda *_: None)
    package("megatron")
    package("megatron.training")

    prefix = "primus.backends.megatron.core"
    contract = load(prefix + ".fp4_contract", CORE / "fp4_contract.py")
    load(prefix + ".fp4_reference", CORE / "fp4_reference.py")
    load(prefix + ".fp4_backend", CORE / "fp4_backend.py")
    load(prefix + ".fp4_dispatch", CORE / "fp4_dispatch.py")
    state = load(prefix + ".fp4_state", CORE / "fp4_state.py")
    deosc = load(prefix + ".optimizer.two_level_fp4_deosc", CORE / "optimizer/two_level_fp4_deosc.py")

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(16, 16, dtype=torch.bfloat16))
            self.fp4_config = contract.TwoLevelFP4Config("nvfp4")
            self.fp4_configs = (self.fp4_config,)
            self.fp4_rngs = {}
            self._primus_fp4_last_config = self.fp4_config

    class ExternalOptimizer:
        """The model/master attributes supplied by an external Megatron optimizer."""

        def __init__(self, model_chunks):
            parameters = [parameter for model in model_chunks for parameter in model.parameters()]
            self.float16_groups = [parameters]
            self.fp32_from_float16_groups = [[p.detach().float().clone() for p in parameters]]
            self.config = SimpleNamespace()

        def _copy_main_params_to_model_params(self):
            with torch.no_grad():
                for model, master in zip(self.float16_groups[0], self.fp32_from_float16_groups[0]):
                    model.copy_(master)

        def step_with_ready_grads(self):
            self._copy_main_params_to_model_params()
            return True

    h = SimpleNamespace(
        args=SimpleNamespace(
            save=str(tmp_path / "save"),
            load=str(tmp_path / "save"),
            iteration=0,
            finetune=False,
            weight_deosc=False,
            weight_deosc_period=2,
            weight_deosc_ratio=4.0,
            weight_deosc_start_step=0,
            disable_last_saving=False,
            train_iters=10,
        ),
        events=[],
        models=[Layer()],
        Layer=Layer,
        state=state,
        deosc=deosc,
        load_result=(3, 9.0),
        main_save_saw_sidecar=False,
        last_save=None,
        last_load=None,
    )

    def get_model(model_provider_func, model_type=None, wrap_with_ddp=True, config=None, pg_collection=None):
        h.events.append("get_model")
        h.model_call = model_provider_func, model_type, wrap_with_ddp, config, pg_collection
        return h.models

    def get_optimizer(
        config,
        model_chunks,
        config_overrides=None,
        use_gloo_process_groups=True,
        pg_collection=None,
        dump_param_to_param_group_map=None,
    ):
        h.events.append("get_optimizer")
        assert all(hasattr(model, "fp4_stream_id") for model in model_chunks)
        h.optimizer_call = config, model_chunks, config_overrides, use_gloo_process_groups
        return ExternalOptimizer(model_chunks)

    def get_checkpoint_name(base, iteration, release=False, return_base_dir=False, **kwargs):
        assert return_base_dir
        return str(Path(base) / ("release" if release else f"iter_{iteration:07d}"))

    # Keep the pinned upstream positional order: non_persistent_ckpt is slot11,
    # release is slot14. Test doubles only model external save/load effects.
    def save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context=None,
        pipeline_rank=None,
        expert_rank=None,
        tensor_rank=None,
        pipeline_parallel=None,
        expert_parallel=None,
        non_persistent_ckpt=False,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
        release=False,
        **kwargs,
    ):
        if h.args.disable_last_saving and iteration == h.args.train_iters:
            h.events.append("last_save_skipped")
            return "skipped"
        h.events.append("main_save")
        h.last_save = locals().copy()
        directory = Path(get_checkpoint_name(h.args.save, iteration, release, return_base_dir=True))
        h.main_save_saw_sidecar = (directory / "two_level_fp4/rank_0.pt").is_file()
        Path(h.args.save).mkdir(parents=True, exist_ok=True)
        (Path(h.args.save) / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        return "saved"

    def load_checkpoint(
        ddp_model,
        optimizer,
        opt_param_scheduler,
        load_arg="load",
        strict=True,
        checkpointing_context=None,
        skip_load_to_model_and_opt=False,
        **kwargs,
    ):
        h.events.append("main_load")
        assert all(hasattr(model, "fp4_stream_id") for model in ddp_model)
        h.last_load = locals().copy()
        return h.load_result

    h.checkpoint_name = get_checkpoint_name
    h.training = package(
        "megatron.training.training",
        get_model=get_model,
        get_megatron_optimizer=get_optimizer,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
    )
    package("megatron.training.checkpointing", get_checkpoint_name=get_checkpoint_name)
    load("primus.backends.megatron.patches.two_level_fp4_patches", PATCHES / "two_level_fp4_patches.py")
    impl = load(
        "primus.backends.megatron.patches.two_level_fp4_state_patches",
        PATCHES / "two_level_fp4_state_patches.py",
    )
    h.install = lambda: impl.install_two_level_fp4_training_state(h.args)
    return h


@pytest.mark.parametrize("keywords", [False, True])
def test_binding_follows_model_build_and_precedes_optimizer_without_reset(hooks, keywords):
    h = hooks
    h.install()
    provider, config, groups = object(), object(), object()
    if keywords:
        models = h.training.get_model(model_provider_func=provider, config=config, pg_collection=groups)
    else:
        models = h.training.get_model(provider, None, True, config, groups)
    assert models is h.models
    assert h.model_call == (provider, None, True, config, groups)
    assert models[0].fp4_rng.sr_offset == 0
    models[0].fp4_rng.reserve("sr", 7)
    if keywords:
        optimizer = h.training.get_megatron_optimizer(config=config, model_chunks=models)
    else:
        optimizer = h.training.get_megatron_optimizer(config, models)
    assert optimizer.float16_groups[0][0] is models[0].weight
    assert models[0].fp4_rng.sr_offset == 7
    assert h.events == ["get_model", "get_optimizer"]


def test_real_deosc_installed_after_optimizer_and_installation_is_idempotent(hooks):
    h = hooks
    h.args.weight_deosc = True
    h.install()
    wrappers = h.training.get_model, h.training.get_megatron_optimizer, h.training.save_checkpoint
    h.install()
    assert wrappers == (h.training.get_model, h.training.get_megatron_optimizer, h.training.save_checkpoint)
    optimizer = h.training.get_megatron_optimizer(object(), h.models)
    runners = optimizer._primus_two_level_deosc_runners
    assert len(runners) == 1 and isinstance(runners[0], h.deosc.TwoLevelDeOsc)
    assert runners[0].step_count == 0
    assert optimizer.step_with_ready_grads() is True
    assert runners[0].step_count == 1


def test_binding_failure_prevents_optimizer_creation(hooks):
    h = hooks
    h.install()
    with pytest.raises(ValueError, match="No two-level"):
        h.training.get_megatron_optimizer(object(), [torch.nn.Linear(2, 2)])
    assert h.events == []


@pytest.mark.parametrize("keywords", [False, True])
def test_sidecar_exists_before_main_save_and_tracker_advance(hooks, keywords):
    h = hooks
    h.install()
    h.state.bind_fp4_model(h.models)
    scheduler, context = object(), {"async": False}
    if keywords:
        result = h.training.save_checkpoint(
            iteration=3,
            model=h.models,
            optimizer=None,
            opt_param_scheduler=scheduler,
            num_floating_point_operations_so_far=9.0,
            checkpointing_context=context,
            tp_group="group",
        )
    else:
        result = h.training.save_checkpoint(3, h.models, None, scheduler, 9.0, context, tp_group="group")
    assert result == "saved"
    assert h.main_save_saw_sidecar
    assert h.last_save["checkpointing_context"] is context
    assert h.last_save["opt_param_scheduler"] is scheduler
    assert h.last_save["kwargs"] == {"tp_group": "group"}
    assert (Path(h.args.save) / "latest_checkpointed_iteration.txt").read_text() == "3"


def test_sidecar_failure_prevents_main_save_and_tracker(hooks, monkeypatch):
    h = hooks

    def failed_sidecar(*args, **kwargs):
        raise OSError("injected sidecar write failure")

    monkeypatch.setattr(h.state, "save_fp4_sidecar", failed_sidecar)
    h.install()
    with pytest.raises(OSError, match="sidecar write failure"):
        h.training.save_checkpoint(3, h.models, None, None, 0)
    assert "main_save" not in h.events
    assert not (Path(h.args.save) / "latest_checkpointed_iteration.txt").exists()


@pytest.mark.parametrize("keywords", [False, True])
def test_nonpersistent_checkpoint_rejected_for_both_call_styles(hooks, keywords):
    h = hooks
    h.install()
    h.state.bind_fp4_model(h.models)
    with pytest.raises(ValueError, match="persistent checkpoint"):
        if keywords:
            h.training.save_checkpoint(3, h.models, None, None, 0, non_persistent_ckpt=True)
        else:
            h.training.save_checkpoint(3, h.models, None, None, 0, None, None, None, None, None, None, True)
    assert "main_save" not in h.events
    assert not Path(h.args.save).exists()


@pytest.mark.parametrize("keywords", [False, True])
def test_release_checkpoint_uses_same_directory_as_main_save(hooks, keywords):
    h = hooks
    h.install()
    h.state.bind_fp4_model(h.models)
    if keywords:
        h.training.save_checkpoint(3, h.models, None, None, 0, release=True)
    else:
        h.training.save_checkpoint(
            3, h.models, None, None, 0, None, None, None, None, None, None, False, None, None, True
        )
    assert h.main_save_saw_sidecar
    assert (Path(h.args.save) / "release/two_level_fp4/rank_0.pt").is_file()
    assert not (Path(h.args.save) / "iter_0000003").exists()


def test_disabled_last_save_does_not_write_sidecar(hooks):
    h = hooks
    h.args.disable_last_saving = True
    h.install()
    assert h.training.save_checkpoint(10, h.models, None, None, 0) == "skipped"
    assert h.events == ["last_save_skipped"]
    assert not Path(h.args.save).exists()


def test_missing_explicit_save_path_is_rejected(hooks):
    h = hooks
    h.args.save = None
    h.install()
    with pytest.raises(ValueError, match="explicit save directory"):
        h.training.save_checkpoint(3, h.models, None, None, 0)
    assert "main_save" not in h.events


@pytest.mark.parametrize("keywords", [False, True])
def test_continuation_without_sidecar_fails_after_main_load(hooks, keywords):
    h = hooks
    h.install()
    with pytest.raises(FileNotFoundError, match="Missing FP4 continuation"):
        if keywords:
            h.training.load_checkpoint(ddp_model=h.models, optimizer=None, opt_param_scheduler=None)
        else:
            h.training.load_checkpoint(h.models, None, None)
    assert h.events == ["main_load"]


@pytest.mark.parametrize("keywords", [False, True])
def test_load_restores_real_rng_with_custom_load_arg_and_forwarded_arguments(hooks, keywords):
    h = hooks
    h.args.alternate_load = str(Path(h.args.save).parent / "alternate")
    h.state.bind_fp4_model(h.models)
    h.models[0].fp4_rng.reserve("sr", 13)
    directory = h.checkpoint_name(h.args.alternate_load, 3, return_base_dir=True)
    h.state.save_fp4_sidecar(h.models, None, directory, 3)
    h.models = [h.Layer()]
    h.install()
    context, scheduler = {"testing": True}, object()
    if keywords:
        result = h.training.load_checkpoint(
            ddp_model=h.models,
            optimizer=None,
            opt_param_scheduler=scheduler,
            load_arg="alternate_load",
            strict=False,
            checkpointing_context=context,
            tp_group="group",
        )
    else:
        result = h.training.load_checkpoint(h.models, None, scheduler, "alternate_load", False, context)
    assert result == (3, 9.0)
    assert h.models[0].fp4_rng.sr_offset == 13
    assert h.last_load["load_arg"] == "alternate_load"
    assert h.last_load["strict"] is False
    assert h.last_load["checkpointing_context"] is context
    assert h.last_load["opt_param_scheduler"] is scheduler
    if keywords:
        assert h.last_load["kwargs"] == {"tp_group": "group"}


def test_explicit_finetune_starts_fresh_private_rng_without_loading_old_sidecar(hooks):
    h = hooks
    h.state.bind_fp4_model(h.models)
    h.models[0].fp4_rng.reserve("sr", 19)
    directory = h.checkpoint_name(h.args.load, 3, return_base_dir=True)
    h.state.save_fp4_sidecar(h.models, None, directory, 3)
    h.models = [h.Layer()]
    h.args.finetune = True
    h.load_result = (0, 0)
    h.install()
    assert h.training.load_checkpoint(h.models, None, None) == (0, 0)
    assert h.models[0].fp4_rng.sr_offset == 0


def test_fresh_training_without_checkpoint_does_not_require_sidecar(hooks):
    h = hooks
    h.args.load = None
    h.load_result = (0, 0)
    h.install()
    assert h.training.load_checkpoint(h.models, None, None) == (0, 0)
    assert h.models[0].fp4_rng.sr_offset == 0
