# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU P0 checks. Run with --confcutdir=tests/unit_tests/low_precision.

Actual contract modules and fp4_utils bodies execute. TE/Megatron import
dependencies are stubbed only in hook tests, not represented as GPU coverage.
"""

import ast
import contextlib
import enum
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = load_file("fp4_preflight_test", ROOT / "tools/fp4_preflight.py")
C, D = preflight.load_contracts()


def options(recipe="nvfp4", **updates):
    path = ROOT / f"examples/megatron/low_precision/two_level_fp4/{recipe}.yaml"
    values = yaml.safe_load(path.read_text())
    values.update(updates)
    return values


@pytest.mark.parametrize(
    "recipe,inner,maximum,floor",
    [
        ("nvfp4", "e4m3fn", 448, 2**-6),
        ("amdfp4", "ue5m3", 114688, 2**-17),
    ],
)
def test_recipe_fragment_and_routes(recipe, inner, maximum, floor):
    result = preflight.inspect_options(options(recipe))
    assert [r["module_kind"] for r in result["routes"]] == ["dense", "grouped"]
    cfg = D.build_two_level_config(options(recipe))
    assert (cfg.inner_format, cfg.inner_max, cfg.inner_floor) == (inner, maximum, floor)
    assert C.TwoLevelFP4Config(**json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert all(r["execution_status"] == "QDQ reference consumer" for r in result["routes"])
    assert len(D.validate_fp4_options(options(recipe))) == 2


@pytest.mark.parametrize(
    "update",
    [
        {"fp4_block_size": 32},
        {"fp4_inner_storage": "uint8"},
        {"fp4_outer_scope": "expert"},
        {"fp4_rht_size": 8},
        {"fp4_rng_seed": -1},
        {"fp4_gradient_sr": "invalid"},
        {"fp4_backend": "typo"},
        {"fp4_backend": "turbo"},
        {"fp4_backend": "te", "use_turbo_gemm": True},
        {"turbo_fused_grouped_gemm": True},
        {"weight_deosc": True, "data_parallel_size": 2, "use_distributed_optimizer": True},
        {"fp4_quantizer": "unsupported"},
        {"fp4_backend": "te", "fp4_quantizer": "triton"},
        {"fp4_backend": "auto", "fp4_quantizer": "triton"},
        {"weight_deosc_direct_qdq": True},
        {"moe_permute_quant_fusion": True},
        {"moe_backward_permute_quant_fusion": True},
        {"fp4_param": True},
        {"fp4_param_gather": True},
        {"fp8": "e4m3"},
        {"expert_model_parallel_size": 2},
        {"tensor_model_parallel_size": 2},
        {"transformer_impl": "local"},
    ],
)
def test_invalid_combinations_rejected(update):
    with pytest.raises(ValueError):
        D.validate_fp4_options(options(**update), require_runtime=False)


@pytest.mark.parametrize(
    "key",
    [
        "PRIMUS_FUSED_RMSNORM_MXFP4_QKV",
        "PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT",
        "PRIMUS_TURBO_GRAD_OWNERSHIP",
    ],
)
def test_incompatible_fused_environment(key):
    with pytest.raises(ValueError, match=key):
        D.validate_fp4_options(options(), {key: "1"}, require_runtime=False)


def test_dense_and_grouped_backend_checks_and_legacy_mxfp4():
    assert D.validate_fp4_options({"fp4": "e2m1", "fp4_recipe": "mxfp4"}) == ()
    args = options(fp4_backend="auto", use_turbo_gemm=False, use_turbo_grouped_gemm=True)
    assert D.resolve_fp4_plan(args, "dense").backend == "te"
    with pytest.raises(ValueError, match="Turbo grouped"):
        D.validate_fp4_options(args, require_runtime=False)
    with pytest.raises(ValueError, match="no AMDFP4"):
        D.resolve_fp4_plan(options("amdfp4", fp4_backend="te", use_turbo_gemm=False))
    with pytest.raises(ValueError, match="disagrees"):
        D.validate_fp4_options(options(), {"PRIMUS_LINEAR_PRECISION": "mxfp4"}, require_runtime=False)


@pytest.mark.parametrize("flag", ["weight_deosc", "weight_deosc_direct_qdq", "turbo_fused_grouped_gemm"])
def test_te_nvfp4_also_rejects_mxfp4_specific_hooks(flag):
    args = options(fp4_backend="te", use_turbo_gemm=False, use_turbo_grouped_gemm=False)
    args[flag] = True
    with pytest.raises(ValueError, match=flag):
        D.validate_fp4_options(args, require_runtime=False)


def test_config_fallback_and_operand_directions():
    cfg = D.build_two_level_config(SimpleNamespace(fp4_recipe="amdfp4"), options("amdfp4", fp4_rng_seed=5678))
    assert cfg.rng_seed == 5678
    expected = {
        ("fwd", "input"): (-1, False, False, 0),
        ("fwd", "weight"): (-1, True, False, 0),
        ("dgrad", "grad_output"): (-1, False, True, 0),
        ("dgrad", "weight"): (-2, True, False, 0),
        ("wgrad", "input"): (0, False, False, 32),
        ("wgrad", "grad_output"): (0, False, True, 32),
    }
    for pair, values in expected.items():
        o = cfg.operand(*pair)
        assert (o.axis, o.two_dimensional, o.stochastic_rounding, o.hadamard_size) == values
    with pytest.raises(ValueError):
        cfg.operand("wgrad", "weight")


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_scale_oracle(recipe):
    cfg = C.TwoLevelFP4Config(recipe)
    assert cfg.outer_scale(0) * cfg.inner_floor > torch.finfo(torch.float32).tiny
    assert cfg.inner_scale(0, 1) == cfg.inner_floor
    assert cfg.inner_scale(1e30, 1) == cfg.inner_max
    # Independent finite grid + midpoint ties, with clamp excluded.
    bias, last = (7, 126) if recipe == "nvfp4" else (15, 254)
    grid = [
        m * 2 ** (1 - bias - 3) if e == 0 else (1 + m / 8) * 2 ** (e - bias)
        for code in range(last + 1)
        for e, m in [divmod(code, 8)]
    ]
    for i, value in enumerate(grid):
        if value < cfg.inner_floor:
            continue
        assert cfg.inner_scale(value * 6, 1) == value
        if i < last:
            midpoint = (value + grid[i + 1]) / 2
            assert cfg.inner_scale(midpoint * 6, 1) == grid[i + (i % 2)]
    if recipe == "nvfp4":
        for value in torch.logspace(-6, 2.6, 100).tolist():
            expected = torch.tensor(max(value, cfg.inner_floor)).to(torch.float8_e4m3fn).float().item()
            assert cfg.inner_scale(value * 6, 1) == expected
    for invalid in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            cfg.outer_scale(invalid)
    nv, amd = C.TwoLevelFP4Config("nvfp4"), C.TwoLevelFP4Config("amdfp4")
    assert amd.outer_scale(6) == nv.outer_scale(6) / 256


def make_tensor(recipe="nvfp4", direction="fwd", role="input"):
    cfg = C.TwoLevelFP4Config(recipe)
    layout = C.FP4TensorLayout((32, 48), (32, 48), cfg.operand(direction, role))
    packed, scales = layout.storage_shapes(cfg)
    return C.TwoLevelFP4Tensor(
        torch.zeros(packed, dtype=torch.uint8), torch.ones(scales), torch.ones(1), cfg, layout
    )


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize(
    "direction,role", [("fwd", "input"), ("fwd", "weight"), ("dgrad", "weight"), ("wgrad", "grad_output")]
)
def test_tensor_storage_metadata_roundtrip(recipe, direction, role):
    tensor = make_tensor(recipe, direction, role).validate()
    metadata = json.loads(json.dumps(tensor.metadata()))
    restored = C.TwoLevelFP4Tensor.from_metadata(
        tensor.data, tensor.inner_scale, tensor.outer_scale, metadata
    )
    assert restored.layout == tensor.layout
    assert restored.config == tensor.config
    metadata["inner_format"] = "e8m0"
    with pytest.raises(ValueError, match="metadata mismatch"):
        C.TwoLevelFP4Tensor.from_metadata(tensor.data, tensor.inner_scale, tensor.outer_scale, metadata)


def test_invalid_scale_storage_and_transpose_contract():
    tensor = make_tensor()
    with pytest.raises(ValueError, match="storage requires"):
        C.TwoLevelFP4Tensor(
            tensor.data, tensor.inner_scale.to(torch.uint8), tensor.outer_scale, tensor.config, tensor.layout
        ).validate()
    with pytest.raises(ValueError, match="storage requires"):
        C.TwoLevelFP4Tensor(
            tensor.data, tensor.inner_scale, torch.ones(2), tensor.config, tensor.layout
        ).validate()
    cfg = C.TwoLevelFP4Config("nvfp4")
    forward = C.FP4TensorLayout((2880, 5760), (2880, 5760), cfg.operand("fwd", "weight"))
    backward = C.FP4TensorLayout((2880, 5760), (2880, 5760), cfg.operand("dgrad", "weight"))
    assert forward.storage_shapes(cfg) == ((2880, 2880), (180, 360))
    assert backward.storage_shapes(cfg) == ((5760, 1440), (360, 180))


def test_expert_padding_and_empty_experts():
    lengths = [0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129]
    original, padded = C.padded_group_offsets(lengths)
    cfg = C.TwoLevelFP4Config("amdfp4")
    for direction, role in (("fwd", "input"), ("wgrad", "input")):
        layout = C.FP4TensorLayout(
            (sum(lengths), 2880), (padded[-1], 2880), cfg.operand(direction, role), original, padded
        )
        layout.validate(cfg)
    assert padded[0] == padded[1] == 0
    bad = C.FP4TensorLayout((128, 16), (128, 16), cfg.operand("fwd", "input"), (0, 64, 128), (0, 64, 128))
    with pytest.raises(ValueError, match="independently padded"):
        bad.validate(cfg)


def test_private_rng_streams_and_recipe_resume():
    cfg = C.TwoLevelFP4Config("nvfp4", rng_seed=5678)
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    rng = C.FP4RandomState.for_config(cfg)
    first = rng.reserve("sr", 32)
    shared_rht = rng.reserve("rht", 1)
    assert first["offset"] == shared_rht["offset"] == 0 and first["stream"] != shared_rht["stream"]
    assert first["seed"] != shared_rht["seed"]
    assert C.FP4RandomState.for_config(cfg, "rank0.layer1").reserve("sr", 1)["seed"] != first["seed"]
    restored = C.FP4RandomState.from_state_dict(json.loads(json.dumps(rng.state_dict())), cfg)
    assert restored.reserve("sr", 16) == rng.reserve("sr", 16)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(ValueError, match="mismatch"):
        C.FP4RandomState.from_state_dict(rng.state_dict(), C.TwoLevelFP4Config("amdfp4", rng_seed=5678))


def test_context_nesting_override_and_restore_after_error():
    nv = D.resolve_fp4_plan(options())
    amd = D.resolve_fp4_plan(options("amdfp4"))
    assert D.current_fp4_plan() is None
    with D.fp4_recipe_context(nv):
        with pytest.raises(RuntimeError), D.fp4_recipe_context(amd):
            assert D.current_fp4_plan() is amd
            raise RuntimeError("test")
        assert D.current_fp4_plan() is nv
        bf16 = {"override_quantized_autocast": True, "fp4_quantization_recipe": None}
        with D.module_fp4_context(bf16, options()):
            assert D.current_fp4_plan() is None
        assert D.current_fp4_plan() is nv
        override = {"override_quantized_autocast": True, "fp4_quantization_recipe": "amdfp4"}
        with D.module_fp4_context(override, options()):
            assert D.current_fp4_plan().config.recipe == "amdfp4"
    assert D.current_fp4_plan() is None


def package(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return module


@pytest.fixture
def fp4_utils(monkeypatch):
    # Execute the real full source file with only external framework imports
    # isolated. No AST replacement of the functions under test.
    for name in (
        "primus",
        "primus.backends",
        "primus.backends.megatron",
        "primus.backends.megatron.core",
        "primus.core",
        "primus.core.utils",
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.training",
    ):
        package(monkeypatch, name)
    monkeypatch.setitem(sys.modules, "primus.backends.megatron.core.fp4_dispatch", D)
    monkeypatch.setitem(sys.modules, "primus.backends.megatron.core.fp4_contract", C)
    enums = load_file("_fp4_test_enums", ROOT / "primus/backends/megatron/core/enums.py")
    monkeypatch.setitem(sys.modules, "primus.backends.megatron.core.enums", enums)
    package(monkeypatch, "megatron.core.parallel_state", model_parallel_is_initialized=lambda: False)
    package(monkeypatch, "megatron.core.transformer.transformer_config", TransformerConfig=object)
    package(monkeypatch, "megatron.core.utils", is_te_min_version=lambda version: True)
    package(monkeypatch, "megatron.training.global_vars", get_args=lambda: SimpleNamespace(**options()))
    package(monkeypatch, "primus.core.utils.module_utils", warning_rank_0=lambda *a: None)
    monkeypatch.setitem(sys.modules, "transformer_engine", None)
    monkeypatch.setitem(sys.modules, "primus_turbo", None)
    return load_file("_fp4_test_utils", ROOT / "primus/backends/megatron/core/fp4_utils.py")


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
def test_real_config_builder_and_context_hook_without_gpu_stack(fp4_utils, recipe):
    config = SimpleNamespace(fp4_recipe=recipe, fp4="e2m1")
    quant, error = fp4_utils.get_fp4_quant_config(config)
    assert isinstance(quant, C.TwoLevelFP4Config) and quant.recipe == recipe and error == ""
    with fp4_utils.get_fp4_context(config):
        assert D.current_fp4_plan().config.recipe == recipe
    assert D.current_fp4_plan() is None
    with fp4_utils.get_fp4_context(config, is_init=True):
        pass  # BF16 model parameter initialization remains unquantized.
    config.fp4_backend = "te"
    config.enable_primus_turbo = False
    if recipe == "nvfp4":
        with pytest.raises(RuntimeError, match="Transformer Engine is unavailable"):
            fp4_utils.get_fp4_context(config)
    else:
        with pytest.raises(ValueError, match="no AMDFP4"):
            fp4_utils.get_fp4_context(config)


def test_real_rocm_validation_hook_rejects_unsupported_fusion(fp4_utils, monkeypatch):
    package(monkeypatch, "primus.core.utils.logger", info_with_caller=lambda *a: None)
    rocm = load_file("_fp4_test_rocm", ROOT / "primus/backends/megatron/patches/args/rocm_arg_validation.py")
    args = SimpleNamespace(**options(deterministic_mode=False, weight_deosc_direct_qdq=True))
    with pytest.raises(ValueError, match="weight_deosc_direct_qdq"):
        rocm.validate_args_on_rocm(args)


def test_mxfp4_context_delegates_unchanged(fp4_utils, monkeypatch):
    calls = []
    marker = contextlib.nullcontext()
    monkeypatch.setattr(fp4_utils, "_legacy_get_fp4_context", lambda *a: calls.append(a) or marker)
    config = SimpleNamespace(fp4="e2m1", fp4_recipe="mxfp4")
    assert fp4_utils.get_fp4_context(config, 3, False) is marker
    assert calls == [(config, 3, False)]


def test_te_missing_recipe_never_disables_autocast(fp4_utils, monkeypatch):
    monkeypatch.setattr(fp4_utils, "HAVE_TE", True)
    monkeypatch.setattr(fp4_utils, "get_fp4_recipe", lambda cfg: (None, "missing NVFP4BlockScaling"))
    config = SimpleNamespace(**options(fp4_backend="te", enable_primus_turbo=False))
    with pytest.raises(RuntimeError, match="refusing disabled autocast"):
        fp4_utils.get_fp4_context(config)


def test_te_selection_preserves_external_kernel_responsibility(fp4_utils, monkeypatch):
    monkeypatch.setattr(fp4_utils, "HAVE_TE", True)
    monkeypatch.setattr(fp4_utils, "get_fp4_recipe", lambda cfg: (object(), ""))
    captured = []
    monkeypatch.setattr(
        fp4_utils, "_legacy_get_fp4_context", lambda cfg, *a: captured.append(cfg) or contextlib.nullcontext()
    )
    config = SimpleNamespace(**options(fp4_backend="te", use_turbo_gemm=False, use_turbo_grouped_gemm=False))
    with fp4_utils.get_fp4_context(config):
        pass
    assert captured[0].fp4_use_native_te_autocast is True
    assert not hasattr(config, "fp4_use_native_te_autocast")


def test_enum_patch_rebinds_already_imported_module_aliases(fp4_utils, monkeypatch):
    for name in ("primus.backends.megatron.patches", "primus.backends.megatron.patches.turbo"):
        package(monkeypatch, name)
    package(
        monkeypatch,
        "primus.backends.megatron.patches.turbo.utils",
        is_primus_turbo_can_patch=lambda ctx: True,
        optional_layer_loop_modules=lambda: [],
    )
    package(
        monkeypatch,
        "primus.core.patches",
        PatchContext=object,
        get_args=lambda ctx: ctx,
        register_patch=lambda *a, **k: lambda fn: fn,
    )
    monkeypatch.setattr(
        sys.modules["primus.core.utils.module_utils"], "log_rank_0", lambda *a: None, raising=False
    )
    old_enum = enum.Enum("OldFp4Recipe", {"nvfp4": "nvfp4", "custom": "custom"}, type=str)
    enums = package(monkeypatch, "megatron.core.enums", Fp4Recipe=old_enum)
    alias = package(monkeypatch, "megatron.core.quantization", Fp4Recipe=old_enum)
    patches = load_file("_fp4_test_patches", ROOT / "primus/backends/megatron/patches/turbo/fp4_patches.py")
    patches.patch_enums(None)
    assert alias.Fp4Recipe is enums.Fp4Recipe
    assert alias.Fp4Recipe("amdfp4").value == "amdfp4"
    assert alias.Fp4Recipe("custom").value == "custom"


def test_actual_per_module_hook_gates_new_formats_and_preserves_bf16(fp4_utils):
    path = ROOT / "primus/backends/megatron/core/extensions/primus_turbo.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_fp8_autocast_for_quant_recipe"
    )
    marker = contextlib.nullcontext()
    ns = {
        "TEQuantizationRecipe": object,
        "FP8GlobalStateManager": SimpleNamespace(is_fp8_enabled=lambda: True),
        "_is_fp4_or_fp8_enabled": lambda: False,
        "get_args": lambda: SimpleNamespace(**options()),
        "nullcontext": contextlib.nullcontext,
        "primus_turbo_fp8_autocast": lambda **k: marker,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    fn = ns[node.name]
    q = SimpleNamespace(
        fp4_quantization_recipe=None,
        fp8_quantization_recipe=None,
        override_quantized_autocast=True,
        override_nonquantized_autocast=False,
    )
    assert fn(q) is marker
    q.fp4_quantization_recipe = "amdfp4"
    with pytest.raises(ValueError, match="reference spec provider"):
        fn(q)
    q.override_quantized_autocast = False
    with fn(q):
        pass


def shell_config(**env):
    script = ROOT / "examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1.sh"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", 'source "$1" || exit; env -0', "test", str(script)],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
    )
    if result.returncode:
        return result.returncode, result.stderr.decode()
    return 0, dict(x.split("=", 1) for x in result.stdout.decode().split("\0") if "=" in x)


@pytest.mark.parametrize(
    "precision,lr,warmup",
    [
        ("mxfp4", "5.7e-4", "117"),
        ("nvfp4", "5.7e-4", "117"),
        ("amdfp4", "5.7e-4", "117"),
        ("fp8", "4.0e-4", "128"),
    ],
)
def test_filename_independent_launcher(precision, lr, warmup):
    snapshots = []
    keys = (
        "PRIMUS_LR",
        "PRIMUS_LR_WARMUP_ITERS",
        "PRIMUS_EXP_AVG_DTYPE",
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR",
        "PRIMUS_TURBO_GRAD_OWNERSHIP",
    )
    for filename in ("/config/MXFP4.yaml", "/config/NVFP4.yaml", "/config/renamed.yaml"):
        code, env = shell_config(EXP=filename, PRIMUS_LINEAR_PRECISION=precision)
        assert code == 0
        snapshots.append({key: env.get(key) for key in keys})
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert snapshots[0]["PRIMUS_LR"] == lr and snapshots[0]["PRIMUS_LR_WARMUP_ITERS"] == warmup
    assert snapshots[0]["MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR"] == precision


def test_launcher_defaults_overrides_and_rejections():
    code, env = shell_config()
    assert code == 0 and env["PRIMUS_LINEAR_PRECISION"] == "mxfp4"
    assert env["PRIMUS_TURBO_GRAD_OWNERSHIP"] == "1"
    code, env = shell_config(PRIMUS_LINEAR_PRECISION="nvfp4", PRIMUS_LR="0.001", PRIMUS_LR_WARMUP_ITERS="99")
    assert code == 0 and env["PRIMUS_LR"] == "0.001" and env["PRIMUS_LR_WARMUP_ITERS"] == "99"
    assert shell_config(EXP="/config/custom.yaml")[0] == 2
    assert shell_config(PRIMUS_LINEAR_PRECISION="typo")[0] == 2


def test_docker_config_export_propagates_source_failure(tmp_path):
    # Execute only the actual config-export section; never invoke Docker.
    source = (ROOT / "examples/mlperf/gpt_oss_20b/run_with_docker.sh").read_text()
    section = source[source.index("if ! _config_export_text=$(") : source.index("_config_docker_env=()")]
    bad = tmp_path / "bad.sh"
    bad.write_text("return 9\n")
    result = subprocess.run(
        ["bash", "-c", '_config_file="$1"; _config_selectors=();\n' + section, "test", str(bad)],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
    )
    assert result.returncode == 2 and b"refusing to launch" in result.stderr
