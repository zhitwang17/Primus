#!/bin/bash
# GPT-OSS-20B MXFP4/UOS2 preflight launcher environment.
# The inherited system/runtime tuning is shared with the validated 1x8 recipe;
# all model/training semantics are declared in the target YAML below.

_GPTOSS_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh
source "${_GPTOSS_CONFIG_DIR}/config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh"

export EXP="${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-UOS2-training-preflight.yaml"
export PRIMUS_MIN_LR=4.0e-5
# The inherited shell pins every dense/grouped precision to Triton. Dense FP4
# has no Triton backend in the current Turbo dispatcher, while the UOS
# forward-port covers the supported HIPBLASLt dense and FlyDSL grouped paths.
# Scope these pins to FP4 so other precisions keep their normal defaults.
export PRIMUS_TURBO_GEMM_BACKEND=FP4:HIPBLASLT
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=FP4:FLYDSL
# Keep the TE MXFP4 recipe metadata aligned with the Turbo kernels. Turbo's
# dense/grouped implementations select RHT on the backward colwise activation
# and gradient operands themselves; this flag covers any TE-owned MXFP4 path.
export NVTE_MXFP4_USE_HADAMARD=1

unset _GPTOSS_CONFIG_DIR
