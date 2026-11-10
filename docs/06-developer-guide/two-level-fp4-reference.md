# NVFP4 / AMDFP4 reference training adapters

The `train/nvfp4-amdfp4-gpt-oss` branch now implements actual two-level FP4
quantization and Megatron dense/MoE training adapters. `fp4_backend: reference`
implements E2M1 quantization with explicit inner and outer scales, dequantizes
to BF16 and executes ordinary GEMMs. Forward, dgrad and wgrad quantize their own operands
on the correct axes. This is a numerical integration backend, not a native
FP4 throughput implementation or a performance claim.

The preceding [P0 record](two-level-fp4-p0.md) describes the original contract
milestone; its missing-consumer limitation is superseded by this implementation.

## Components

| Component | Implementation |
| --- | --- |
| Numerical reference | `core/fp4_reference.py`: real E2M1 packing, block16/16x16 scales, E4M3FN or UE5M3 grid, FP32 outer, axis-last layout, SR and paired RHT |
| GPU quantizer | `core/fp4_backend.py` / `core/fp4_triton.py`: opt-in Triton outer-amax, packed quant/dequant and fused direct QDQ with the same BF16 rounding boundaries |
| Scale consumer prototype | `core/fp4_scaled_gemm.py`: dense forward consumes packed E2M1 plus both scales, decodes to BF16 in registers and calls `tl.dot`; experimental and not used by training |
| Three-direction autograd | `core/fp4_autograd.py`: dense and grouped fwd/dgrad/wgrad; full-bank outer statistics, per-expert padding128 and empty-expert zero gradients |
| Megatron adapters | `core/extensions/two_level_fp4.py`: dense, norm+linear, grouped linears and MoE MLP; standard Parameters, bias/probs/GLU gradients and DDP hooks |
| Dispatch | `patches/two_level_fp4_patches.py`: install the reference spec provider after the general Primus provider; Turbo modules reject new-format metadata |
| Private state | `core/fp4_state.py`: stable rank/module/recipe RNG identity, strict continuation metadata and atomic per-rank sidecars |
| DeOsc | `core/optimizer/two_level_fp4_deosc.py`: successful update, full FP32 master QDQ and observation before master-to-model copy |
| Training hooks | `patches/two_level_fp4_state_patches.py`: bind after model creation, install DeOsc, save/restore state around standard Primus checkpoint calls |
| Native checkpoint harness | `core/fp4_checkpoint.py`: real Megatron `torch_dist` model/optimizer tensors plus rank-local RNG and strict FP4 sidecars |

Paths above are relative to `primus/backends/megatron/`.

## Numerical behavior

Weights remain BF16 model Parameters with ordinary FP32 optimizer masters.
There is no retained quantized-weight cache; every operation recomputes current
scales. Native FP4 storage/all-gather and MXFP4-specific fusions remain disabled.
Existing norm/attention/router/CE choices are retained by the provider, except
the TE combined norm-linear slot executes a separate Torch norm followed by the
reference GEMM. Embeddings and the LM head remain outside transformer FP4 scopes.

A grouped weight `[E,N,K]` has one outer scalar across the local expert bank.
Routed activation/gradient buffers likewise share their outer statistic; each
expert is padded separately before token-axis RHT/quantization. Cropping happens
after the product. Empty experts produce explicit zero weight gradients.
Quantized weight gradients return through ordinary autograd, so Megatron DDP
accumulates `main_grad` and fires readiness hooks. This does not implement
Turbo's direct fused FP32 wgrad accumulation/ownership optimization.

Forward input/weight quantization is deterministic. Backward reserves private
streams for gradient SR and randomized H32 (H16/disabled are explicit options).
Both wgrad operands receive the same transform. Recomputed forwards consume no
FP4 randomness. The reference mapping is
`torch-cpu-reservation-sha256-v1`, which uses isolated CPU generators and copies
random tensors to the device; it is not ALTO/Turbo Philox stream parity. Exact
continuation requires the same Torch version. Value checks and routed lengths
may synchronize the GPU: this backend favors inspectable correctness.

DeOsc uses the same BF16 pre-rounding and whole-bank forward grid as the GEMM.
It retains Primus's BF16 snapshot/subtraction policy, FP32 distance sums and FP32
master snap. After a snap it recomputes the entire grid because dynamic outer
scales can change. BF16 override modules are skipped, unsuccessful optimizer
steps and reload copies do not advance windows, and bias/norm parameters are
excluded. Legacy MXFP4 fused/direct-QDQ code is never used for these recipes.

## Configuration and scope

Use the option fragments in
`examples/megatron/low_precision/two_level_fp4/{nvfp4,amdfp4}.yaml` to inspect
the reference settings. They are not standalone full training recipes. The
reference provider is selected automatically for these recipes/backend.
`use_turbo_gemm` and `use_turbo_grouped_gemm` are unnecessary and default to false
in the fragments. Existing Turbo attention can still be selected separately.

Initial execution scope is TP1/EP1/PP1/CP1, grouped MoE, BF16 model weights.
DeOsc requires complete explicit FP32 masters. Ordinary Megatron Float16Optimizer
supports replicated DP masters; full DP1 DistributedOptimizer mappings are also
implemented. GPU smoke covers the former. DP>1 DistributedOptimizer and partial master shards, TE op
fusers, CUDA graphs, offload, native FP4 parameters and MXFP4 producer/fusion
paths are rejected or remain outside this supported reference path.

Continuation records each module's names, parameter shape/dtype/trainability,
base/train/eval routing and override flags, recipe fingerprints, SR/RHT offsets,
Torch/RNG implementation, rank/world size and DeOsc history. Missing/mismatched
state fails explicitly; `finetune` is an explicit fresh-state operation.
The numerical RNG stream excludes the `fp4_quantizer` implementation choice,
so Torch/Triton comparisons do not change the random sequence. The complete
checkpoint identity still includes that choice and rejects implementation
switches on resume. Older checkpoints without the new quantizer identity also
fail strict continuation; no implicit migration is performed.
DeOsc schema v2 records a complete window-presence manifest, so a missing
active window is rejected instead of silently reseeded. The prior v1 schema
is rejected explicitly.

Sidecars are written before the ordinary checkpoint tracker can advance.
Nonpersistent checkpoints are unsupported. Grouped reference weights use the
`weights[E,N,K]` schema; importing existing TE `weight{i}` checkpoints needs an
explicit converter and is not implemented by this change.

## Validation and reproduction

CPU tests execute the complete reference operators and adapters. External
Megatron initialization dependencies are isolated only in CPU hook/module tests.
The actual GPU test uses pinned Megatron
`4b4acac9a1d28ea6829c8d4f566d75698a21249d` (0.19.2), real TE, real Megatron DDP
with overlap reduction, and `Float16OptimizerWithFloat16Params` with FP32 masters.

On 2026-11-10, NVFP4 and AMDFP4 each passed a four-update single-MI355X smoke,
with and without full-layer activation recompute. The model has two layers
(dense + MoE), hidden128, head_dim64, four experts/top2, batch2, sequence64 and
vocabulary128. Attention is unfused BF16 TE; data is fixed synthetic tokens.
This reduces both depth and width, not only the depth of GPT-OSS-20B.

All six quantized GEMM directions ran, router gradients were finite/nonzero,
20 DDP main-grad tensors were valid, model/expert parameters changed, and eight
DeOsc parameter windows closed. Each run saved after step2, constructed fresh
model/DDP/optimizer objects and reproduced steps3–4 exactly: loss, gradients,
parameters, FP32 masters, global/private RNG and DeOsc state. Recompute also
matched the corresponding non-recompute trajectory. The smoke checkpoint is
a Torch model/optimizer bundle plus the production FP4 sidecar helpers;
production checkpoint wrappers have CPU tests. This describes the historical
P1 validation; the expanded P2 harness below adds native `torch_dist` save/load.

CPU environment: Python3.10, Torch2.6.0+cpu, pytest8.3.5. From the repository:

```bash
python tools/fp4_preflight.py --config examples/megatron/low_precision/two_level_fp4/nvfp4.yaml
python tools/fp4_preflight.py --config examples/megatron/low_precision/two_level_fp4/amdfp4.yaml
python -m pytest --confcutdir=tests/unit_tests/low_precision tests/unit_tests/low_precision/test*fp4*.py -q
python -m pytest --noconftest tests/unit_tests/backends/megatron/test_gather_quant_config.py -q
```

GPU dependencies tested in `zirui3/primus-v26.7-flux:v0.1`, image ID
`sha256:ce1a8f23dd3ee989369ea73c7cfe37fa2a1bc5f35c916fb23069aea87a56452c`:
Python3.12.3, Torch2.12.0+rocm10.0.0, TE2.17.0+rocm10.0.0. Initialize only the
pinned Megatron submodule, expose one available GPU, mount this repo at `/work`
and an output directory at `/results`, then run:

```bash
export PYTHONPATH=/work:/work/third_party/Megatron-LM
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1
python tools/fp4_smoke.py --recipe nvfp4 --steps 4 --deosc --output /results/nvfp4_smoke.json
python tools/fp4_smoke.py --recipe amdfp4 --steps 4 --deosc --output /results/amdfp4_smoke.json
# Add --recompute and use a different output filename to exercise recomputation.
```

## P2 geometry, native checkpoint and quantizer validation

`fp4_quantizer: torch` is the default; `fp4_quantizer: triton` opts into the
fused direct-QDQ implementation for all three GEMM directions and DeOsc.
The latter skips materializing packed bytes during QDQ while preserving exactly
the quantization grid and BF16 output. The public packed quant/dequant API is
also implemented and independently checked. SR keeps the reference private CPU
reservations; paired RHT is unchanged. No quantized weight cache is retained.

The `gpt-oss` smoke profile keeps hidden2880, heads64, query groups8, head_dim64,
32 experts/top4 and expert FFN2880 from the pinned Primus model. It reduces the
model to one MoE layer, batch1, sequence32 and vocabulary128. Attention is TE
unfused BF16, and data is synthetic. Use `--num-layers 2` for two MoE layers.
The harness checks every gradient/parameter/master element across DP ranks,
uses distinct rank inputs and SR/RHT streams, and covers a DeOsc boundary.
At step2 it saves, constructs fresh training objects, restores, then checks
steps3-4 and complete tensor/metadata fingerprints. Peak memory is recorded.
Use `--deterministic` for exact continuation: the default MCore non-fused MoE
unpermute uses atomic scatter accumulation and can differ on replay despite
identical saved/restored state. The strict mode selects MCore/Torch deterministic
algorithms and determinism environment settings; it is a correctness setting,
not the configuration used for component performance measurements.

```bash
# Expose one free GPU, with the same PYTHONPATH and NVTE flags as above.
python tools/fp4_smoke.py --recipe nvfp4 --model-profile gpt-oss \
  --quantizer triton --steps 4 --deosc --deterministic --checkpoint-format torch_dist \
  --output /results/nvfp4_gptoss_dp1.json
# Expose exactly two free GPUs. DP uses replicated FP32 masters, not shards.
torchrun --standalone --nproc_per_node=2 tools/fp4_smoke.py \
  --recipe amdfp4 --model-profile gpt-oss --quantizer triton \
  --steps 4 --deosc --deterministic --checkpoint-format torch_dist \
  --output /results/amdfp4_gptoss_dp2.json
python tools/fp4_triton_validate.py --gemm --output /results/triton_parity.json
python tools/fp4_profile.py --recipe nvfp4 --backend triton \
  --shape-profile gpt-oss --cases qkv o moe_fc1 moe_fc2 --rows 128 \
  --warmup 1 --repeats 3 --output /results/triton_profile.json
```

Change `--recipe` to test the other format, or add `--recompute` to the smoke.
The native checkpoint helper currently supports one PP1 model chunk and the
same parallel topology on restore. It is a direct MCore checkpoint test, not
a full Primus launcher/data-loader checkpoint lifecycle test.

On 2026-11-10 the expanded profile passed NVFP4 Torch/Triton DP1, AMDFP4 Triton
DP1, NVFP4 Triton DP2, and AMDFP4 Triton DP2 with full-layer recompute. All five
runs reproduced steps3-4 and complete final model/master/moment/gradient/RNG/DeOsc
state exactly under strict determinism. Both DP2 jobs checked all 823,642,688
elements of gradients, parameters and masters on both ranks. NVFP4 Torch/Triton
also matched their four-step losses, gradients and final tensor states; private
state metadata intentionally distinguishes the quantizer implementations.
The test DeOsc period is two deltas: step1 seeds, step3 closes the four weight
windows, and the step2 checkpoint therefore exercises an in-progress window.
Final CPU regression: 217 FP4 tests plus 18 existing gather/quant tests passed.
GPU operator validation: 58 exact quantization cases and 12 packed-GEMM cases
within explicit BF16 tolerance (real QKV GEMM is not bit-exact).

Profiling separately measures outer-amax, quant/dequant/QDQ, prepared BF16 GEMM,
and DeOsc observation/period close. It reports synchronized wall medians and
GPU-event spans (which can include host launch gaps); these overlapping rows
must not be summed into an estimated training step or TTT. `--unchecked` adds
finite-input kernel-only rows without changing the checked defaults.
`--packed-gemm` measures the experimental dense consumer and a dynamic
quantization-plus-consumer path separately. It uses BF16 `tl.dot`, not FP4 MFMA.

The new fusion is direct QDQ for these formats, also reused by DeOsc. DeOsc
statistics and period-close operations themselves are not fused. Existing
MXFP4 fused grouped MLP, permute+quant, norm+quant, direct DeOsc and gradient
ownership flags remain rejected. Their block32/E8M0 producer/consumer contract
does not accept block16 E4M3FN/UE5M3 inner scales; the ownership path additionally
needs complete beta=0 main-grad producers, including empty experts. Norm fusion
must retain the BF16 materialization boundary unless changing the recipe.

For a custom full MLPerf experiment, the launcher still requires explicit
`PRIMUS_LINEAR_PRECISION=nvfp4|amdfp4`. Native FP4 performance kernels,
optimizer sharding, imported baseline checkpoints, full convergence and
MLPerf/TTT comparisons remain subsequent work.
