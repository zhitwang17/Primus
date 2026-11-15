# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU-only tests for the production JIT staging functions; no GPU/compiler."""

import ast
import fcntl
import hashlib
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH = ROOT / "primus/backends/megatron/patches/te_patches/fused_adam_clip_patches.py"
KERNEL = PATCH.with_name("fused_adam_clip_kernel.cu")


@pytest.fixture
def staging(tmp_path, monkeypatch):
    # Compile the actual production functions without importing the optimizer
    # stack. Only the PyTorch build-directory resolver/compile call are mocked.
    tree = ast.parse(PATCH.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_cached_extension_source", "_load_extension")
    ]
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    source = readonly / KERNEL.name
    source.write_bytes(KERNEL.read_bytes())
    source.chmod(0o444)
    readonly.chmod(0o555)
    cache = tmp_path / "cache"
    calls = []
    module = ModuleType("torch.utils.cpp_extension")
    module._get_build_directory = lambda name, verbose: str(cache / name)
    module.load = lambda **kwargs: calls.append(kwargs) or SimpleNamespace(compiled=True)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", module)
    namespace = {
        "fcntl": fcntl,
        "hashlib": hashlib,
        "os": os,
        "tempfile": tempfile,
        "Path": Path,
        "_EXTENSION_NAME": "primus_te_fused_adam_clip_v6",
        "_extension": None,
        "__file__": str(readonly / PATCH.name),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(PATCH), "exec"), namespace)
    yield SimpleNamespace(
        source=source,
        cache=cache,
        calls=calls,
        namespace=namespace,
        stage=namespace["_cached_extension_source"],
        load=namespace["_load_extension"],
    )
    readonly.chmod(0o755)


def test_readonly_source_is_copied_byte_exact_into_build_directory(staging):
    original = staging.source.read_bytes()
    before = {p.name: p.stat().st_mtime_ns for p in staging.source.parent.iterdir()}
    name, staged, directory = staging.stage(staging.source)
    assert directory.parent == staging.cache
    assert staged.parent == directory and staged != staging.source
    assert staged.read_bytes() == original
    assert name.endswith(hashlib.sha256(original).hexdigest())
    assert {p.name: p.stat().st_mtime_ns for p in staging.source.parent.iterdir()} == before
    assert not staging.source.with_suffix(".hip").exists()


def test_identical_source_reuses_same_file_without_rewriting(staging):
    first = staging.stage(staging.source)
    before = first[1].stat()
    second = staging.stage(staging.source)
    after = second[1].stat()
    assert first == second
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_changed_source_gets_distinct_module_and_build_cache(staging, tmp_path):
    first = staging.stage(staging.source)
    other = tmp_path / "another_revision" / staging.source.name
    other.parent.mkdir()
    other.write_bytes(staging.source.read_bytes() + b"\n// New source revision\n")
    second = staging.stage(other)
    assert first[0] != second[0] and first[2] != second[2]
    assert first[1].read_bytes() == staging.source.read_bytes()
    assert second[1].read_bytes() == other.read_bytes()


def test_corrupt_content_address_is_rejected(staging):
    _, staged, _ = staging.stage(staging.source)
    staged.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="content address"):
        staging.stage(staging.source)


def test_loader_passes_only_cached_source_and_build_directory(staging):
    extension = staging.load()
    assert extension.compiled is True and staging.load() is extension
    assert len(staging.calls) == 1
    call = staging.calls[0]
    directory = Path(call["build_directory"])
    assert call["sources"] == [str(directory / staging.source.name)]
    assert Path(call["sources"][0]).read_bytes() == KERNEL.read_bytes()
    assert call["extra_cflags"] == call["extra_cuda_cflags"] == ["-O3"]
    assert call["with_cuda"] is True and call["verbose"] is False
    assert call["name"] == directory.name


def test_atomic_publication_failure_removes_only_own_temporary(staging, monkeypatch):
    def fail_replace(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        staging.stage(staging.source)
    assert not list(staging.cache.rglob("*.tmp"))
    assert not list(staging.cache.rglob("*.cu"))
    assert staging.source.read_bytes() == KERNEL.read_bytes()


def test_eight_rank_staging_is_atomic_and_reuses_one_source(staging):
    context = multiprocessing.get_context("fork")
    outputs = context.Queue()

    def worker():
        name, source, directory = staging.stage(staging.source)
        outputs.put((name, str(source), source.read_bytes(), source.stat().st_ino))

    processes = [context.Process(target=worker) for _ in range(8)]
    try:
        for process in processes:
            process.start()
        results = [outputs.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        outputs.close()
    assert all(result == results[0] for result in results)
    assert results[0][2] == KERNEL.read_bytes()
    assert len(list(staging.cache.rglob("*.cu"))) == 1
    assert not list(staging.cache.rglob("*.tmp"))
