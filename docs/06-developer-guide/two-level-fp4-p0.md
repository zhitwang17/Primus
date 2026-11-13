# NVFP4 / AMDFP4 P0 integration

Historical P0 record. The subsequent [reference adapter implementation](two-level-fp4-reference.md)
now supplies executable dense/MoE QDQ, DeOsc/state handling and reduced-model GPU
smoke. The missing-consumer status below describes the original P0 commit.
The 2026-11-13 implementation also adds exact optimizer-sharded DeOsc with
CPU/Gloo validation; GPU qualification and full training remain separate gates.
See the reference document for current support and checkpoint boundaries;
the original milestone's scope and test results below are retained as history.

Date: 2026-11-10. Branch: `train/nvfp4-amdfp4-gpt-oss`, based on Primus
`865846318716b73136fca49bc7b1b44447cafd57`.

## Delivered scope

This change implements the **configuration/dispatch and numerical/storage contract**
milestones. It does not implement the P1 Megatron dense/grouped QDQ consumers,
GPU quantizers or native GEMMs. `fp4_backend: reference` is inspectable on CPU;
attempting training raises an explicit P1-consumer error before it can silently
execute BF16 or send the new scales to an MXFP4 kernel.

No Primus-Turbo change is required at this stage. `TwoLevelFP4Config` is a distinct
Primus type; it is never passed to Turbo's block32/E8M0 `Float4QuantConfig`.

| P0 item | Implementation |
| --- | --- |
| Recipe identity | `Fp4Recipe.amdfp4`; rebind already-imported Megatron enum aliases so global and per-module parsing agree; retain upstream `custom` identity |
| Quant config | `fp4_contract.py`: immutable `TwoLevelFP4Config`, JSON representation and fingerprint |
| Dispatch | `fp4_dispatch.py`: explicit `auto/reference/te/turbo` selection; dense **and** grouped checks; incompatible fusion, storage and topology guards |
| Autocast | `fp4_utils.py`: reference builder available without TE/Turbo, execution gate before fallback; NVFP4 TE path must have a real recipe object |
| Module override | Turbo's existing per-module boundary consults the new selector; override flags and nested BF16 metadata scopes are respected |
| Tensor payload | `TwoLevelFP4Tensor`: packed data, inner scale, required outer scale, config and layout together; shape/dtype/device/contiguity validation and metadata round-trip |
| Randomness | `FP4RandomState`: private SR/RHT block reservations, deterministic domain-separated numeric keys, stable caller-supplied rank/module `stream_id`, checkpoint identity checks |
| Launcher | Explicit `PRIMUS_LINEAR_PRECISION`; no filename-based precision inference; preserve explicit LR/warmup; propagate config-export failures before launching |

`auto` retains the existing MXFP4 path. For NVFP4 it can delegate to TE if no
Turbo GEMM consumer is selected. TE is responsible for its actual device/version
support; this is **not** a claim that ROCm TE has executable NVFP4 kernels.
AMDFP4 requires explicit `reference`; a TE or Turbo request fails clearly.
The two-level options describe the reference contract, not TE's internal recipe.

## Numerical and layout contract

- Data: E2M1, low nibble first, two elements per byte.
- Activations/gradients: 1×16. Weights: 16×16 by default; 1-D weights are an
  explicit alternative. No substitution of 32-element MX blocks is allowed.
- NVFP4 inner grid: E4M3FN, max448, current ALTO-compatible floor `2^-6`.
  AMDFP4: UE5M3, max114688, floor `2^-17`. Reference storage is **FP32 grid
  values**; native packed scale storage is a later, distinct interface.
- Outer scale: required one-element FP32 tensor; dynamic scalar specification
  `max(amax/(6*inner_max), 1e-30)`. Scope is the complete tensor supplied to the
  quantizer, including a local `[E,N,K]` expert bank. A per-expert loop must share
  this outer value to preserve the same contract. Distributed/sharded scope
  remains P1 work. Scalar CPU helpers reject non-finite amax; they are not the
  ALTO Triton NaN-sanitizing path and do not establish GPU parity.
- Physical reference storage is contiguous **axis-last**: move the specified
  axis to the last dimension, then pack/scale. Metadata retains original and
  padded shapes. Unlike ALTO's public return layout, this is a canonical storage
  contract; the P1 adapter must convert explicitly, not relabel ALTO buffers.
- fwd: X/W axis -1; dgrad: dY axis -1 and W axis -2; wgrad: X/dY token axis 0.
  Reusing transposed packed data does not replace quantization on a new axis.
- RHT: default randomized H32, only on both wgrad operands, with the **same**
  transform reservation. H16 and disabled RHT are explicit different configs.
  SR applies to gradient quantization. The state API reserves counters but does
  not generate H matrices, uniforms or implement Philox; P1 must define/verify
  their mapping and recompute reuse.
- Grouped reference layout: pad each expert independently to128 rows; keep
  original/padded offsets, including zero-length experts. Total-buffer alignment
  alone is insufficient. Structural metadata validation does not inspect GPU
  values, synchronize a device or prove scale-grid membership.

## CPU preflight and tests

Use an isolated CPU environment (`torch==2.6.0+cpu`, `numpy==2.2.6`,
`pytest==8.3.5`, `PyYAML==6.0.2`). The tools load the actual stdlib-only contract
modules by file and bypass `primus.__init__`, which initializes training imports.

```bash
python tools/fp4_preflight.py \
  --config examples/megatron/low_precision/two_level_fp4/nvfp4.yaml
python tools/fp4_preflight.py \
  --config examples/megatron/low_precision/two_level_fp4/amdfp4.yaml
python -m pytest --confcutdir=tests/unit_tests/low_precision \
  tests/unit_tests/low_precision/test_two_level_fp4.py -q
```

The two YAML files are flat **option fragments**, not runnable training recipes.
The preflight reports the selected dense/grouped route, operand policies,
fingerprint and `P1 consumer required` status. It does not expand full Primus
config inheritance or environment substitutions. Running the CPU suite with
`--confcutdir` avoids the repository's GPU warmup/distributed fixtures.

Validation on 2026-11-10: **58 P0 tests passed**, plus **18 existing gather/quant
configuration regression tests passed** (the latter with `--noconftest`). Both
preflights also passed under system Python with no Torch/TE/Turbo/Megatron
imports. Black, isort, autoflake, ShellCheck, YAML parsing and the submodule-lock
consistency check passed. No GPU or complete training-stack test was run.

Coverage includes both recipes; invalid scale/backend/fusion combinations;
metadata/shape/transpose/outer round-trip; per-expert tail cases
0/1/15/16/17/31/32/33/127/128/129; scalar format grids and midpoint rounding;
private RNG replay; context restoration; real `fp4_utils` and ROCm-validation
hooks with external framework imports stubbed; enum alias propagation; and the
per-module function body. Launcher tests execute only config/export code, never
Docker or training. These hooks do not establish a complete installed-stack test.

## Launcher migration

The default experiment still selects MXFP4 with LR5.7e-4/warmup117. For any custom
`EXP`, set `PRIMUS_LINEAR_PRECISION=mxfp4|nvfp4|amdfp4|fp8` explicitly. FP8 retains
LR4e-4/warmup128; all FP4 selections start from the same schedule. Explicit
`PRIMUS_LR`, `PRIMUS_MIN_LR` and `PRIMUS_LR_WARMUP_ITERS` win over these defaults.
The Docker path forwards these selectors and stops on configuration errors.
At argument validation, the precision selector must agree with the FP4 recipe.

Only MXFP4 enables the existing MXFP4-specific shell optimizations. A new-format
reference fragment also disables fused grouped MLP, permute+quant, DeOsc and
direct QDQ. Unset/disable `PRIMUS_FUSED_RMSNORM_MXFP4_QKV`,
`PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT`, and `PRIMUS_TURBO_GRAD_OWNERSHIP`.
The guards reject stale enabled values instead of silently applying them.

## Next milestone

Implement P1 dense/grouped QDQ adapters using these contracts, preserving BF16
model weights/FP32 masters, main_grad, parameter hooks and probs gradients.
Then validate quantizer/three-direction parity, DeOsc and checkpoint handling on
one GPU and a reduced-depth model; two-GPU synchronization follows. No eight-GPU
full training or formal MXFP4 performance comparison is part of P0.
