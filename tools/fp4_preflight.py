#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU-only preflight for explicit two-level FP4 option fragments, not a trainer.

Load the stdlib-only contract files without Primus.__init__, which initializes
the training framework. This is also the lightweight loader for the CPU tests.
"""

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_contracts():
    name = "_primus_fp4_preflight"
    if name + ".fp4_dispatch" in sys.modules:
        return sys.modules[name + ".fp4_contract"], sys.modules[name + ".fp4_dispatch"]
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "primus/backends/megatron/core")]
    sys.modules[name] = package
    modules = []
    for filename in ("fp4_contract", "fp4_dispatch"):
        spec = importlib.util.spec_from_file_location(
            name + "." + filename, Path(package.__path__[0]) / (filename + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def inspect_options(options, environ=None):
    _, dispatch = load_contracts()
    plans = dispatch.validate_fp4_options(options, environ=environ, require_runtime=False)
    if not plans:
        raise ValueError("Preflight requires an NVFP4 or AMDFP4 option fragment")
    return {
        "scope": "P0 configuration/metadata only; no model, quantizer or GEMM execution",
        "routes": [p.to_dict() for p in plans],
        "operands": {
            p.config.recipe: [
                vars(p.config.operand(direction, role))
                for direction, role in (
                    ("fwd", "input"),
                    ("fwd", "weight"),
                    ("dgrad", "grad_output"),
                    ("dgrad", "weight"),
                    ("wgrad", "input"),
                    ("wgrad", "grad_output"),
                )
            ]
            for p in plans
        },
    }


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Flat option fragment; no inheritance/env expansion"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    options = yaml.safe_load(args.config.read_text())
    if not isinstance(options, dict) or "modules" in options or "extends" in options:
        parser.error("Use a flat FP4 option fragment, not a full Primus experiment config")
    try:
        result = inspect_options(options, environ=os.environ)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
