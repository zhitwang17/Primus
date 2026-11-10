# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Install executable reference FP4 modules at Megatron's spec boundary."""

from primus.backends.megatron.core.fp4_dispatch import is_two_level, option
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext):
    args = get_args(ctx)
    return bool(option(args, "fp4")) and is_two_level(args) and option(args, "fp4_backend") == "reference"


def install_two_level_fp4_spec_provider():
    """Also callable by small-model integration tests before building a GPTModel."""
    import sys

    from megatron.core.extensions import transformer_engine_spec_provider
    from megatron.core.models.gpt import gpt_layer_specs, moe_module_specs
    from megatron.core.transformer import multi_token_prediction

    from primus.backends.megatron.core.extensions.two_level_fp4 import (
        TwoLevelFP4SpecProvider,
    )

    previous = transformer_engine_spec_provider.TESpecProvider
    modules = [transformer_engine_spec_provider, gpt_layer_specs, moe_module_specs, multi_token_prediction]
    for module in modules:
        module.TESpecProvider = TwoLevelFP4SpecProvider
    # Only imported provider aliases; do not alter the Primus provider's own
    # upstream delegate, which preserves attention/router/CE behavior.
    for name, module in list(sys.modules.items()):
        if name.startswith("megatron.") and getattr(module, "TESpecProvider", None) is previous:
            module.TESpecProvider = TwoLevelFP4SpecProvider
    return TwoLevelFP4SpecProvider


@register_patch(
    "megatron.two_level_fp4.spec_provider",
    backend="megatron",
    phase="before_train",
    description="Install TP1/EP1 NVFP4/AMDFP4 QDQ dense and MoE modules",
    condition=_enabled,
    priority=42,  # After the general Primus Turbo provider (41).
)
def patch_two_level_fp4_provider(ctx: PatchContext):
    install_two_level_fp4_spec_provider()
    log_rank_0("[TwoLevelFP4] Installed reference dense/MoE adapters (QDQ + ordinary GEMM)")
