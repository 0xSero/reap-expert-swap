# PROGRESS.md - Qwen dynamic UVA runtime proof and next benchmark lane

Date: 2026-03-17
Repo: /Users/sero/ai/autoresearch
Remote host: ser@192.168.1.70

## Current goal

Prove real live expert-weight residency control in vLLM, then run a 3-arm benchmark:

1. GPUs 0-3: Qwen3.5-35B BF16 baseline
2. GPUs 4-5: plain vLLM offload, no dynamic
3. GPUs 6-7: dynamic offload

Metrics required:
- coherence
- correctness
- TTFT
- prefill speed
- generation speed
- VRAM usage
- swap amount

## What is now proven

### 1) The live runtime patch is real

On the live Qwen server at `127.0.0.1:8012`, the patched runtime now binds loaded MoE tensors back to the UVA registry after vLLM post-load weight rewrites.

Live proof from `/debug_uva_binding`:
- `bound_after_call: 80`
- `offloader_name_count: 80`
- `offloader_entry_count: 160`

The sampled MoE params now have attached runtime metadata:
- `_vllm_reap_cpu_data`
- `_vllm_reap_uva_data`
- `_vllm_reap_device`
- `_vllm_is_uva_offloaded`

### 2) Live swapping is now real

The runtime can now move **bound fused MoE tensors** in and out of GPU memory while the server is live.

Direct proof run:
- targets:
  - `language_model.model.layers.0.mlp.experts.w13_weight`
  - `language_model.model.layers.0.mlp.experts.w2_weight`

Onload result:
- `onloaded_param_count: 2`
- `bytes_onloaded: 805306368`

VRAM before onload:
- GPU0: `18411 MiB`
- GPU1: `18411 MiB`

VRAM after onload:
- GPU0: `19185 MiB`
- GPU1: `19185 MiB`

Restore result:
- `offloaded_param_count: 26`
- `bytes_offloaded: 10468982784`

VRAM after restore:
- GPU0: `18405 MiB`
- GPU1: `18405 MiB`

This is the first non-fake proof that live runtime mutation is actually affecting the vLLM weight path.

### 3) The server survives after live swaps

After the live swap sequence:
- `/v1/models` still responded
- `/v1/chat/completions` still responded

Note: one short post-swap test returned Chinese text instead of literal `OK`, but the important fact is that the model still answered after the residency mutation.

## Important limitation

This is still **fused MoE tensor-level** control, not true per-expert slice-level control.

Qwen in vLLM exposes MoE weights as fused tensors:
- `...mlp.experts.w13_weight`
- `...mlp.experts.w2_weight`

It does **not** expose one runtime parameter per expert ID.

So today’s proven capability is:
- live vLLM runtime mutation works
- live UVA swapping works
- but at fused-tensor granularity

The next real systems step is slice-aware expert residency inside fused MoE tensors.

## Remote patches currently in play

### Latest vLLM env
- `/home/ser/reap-expert-swap-vllm0171`
- vLLM `0.17.1`

### Patched files on remote

1. UVA offloader:
- `/home/ser/reap-expert-swap-vllm0171/lib/python3.12/site-packages/vllm/model_executor/offloader/uva.py`

Current important additions:
- selective registry for REAP-controlled params
- `bind_named_parameters(...)`
- `reap_set_resident_prefixes(...)`
- mixed startup residency
- offload-first / onload-second policy patch in progress for budget-safe swaps

2. REAP multiplex server:
- `/home/ser/reap-expert-swap-reap/scripts/vllm_multiplex_server.py`

Current important additions:
- EP global->local expert mapping for Qwen
- `/debug_uva_binding`
- `/debug_uva_prefix_swap`
- worker debug methods for direct live residency changes

3. vLLM guard removal:
- `/home/ser/reap-expert-swap-vllm0171/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py`

This patch allows input-batch reinit while CPU weight offloading is enabled so the Qwen server can finish startup.

4. Attribute-preservation patches across post-load weight replacement:
- `/home/ser/reap-expert-swap-vllm0171/lib/python3.12/site-packages/vllm/model_executor/utils.py`
- `/home/ser/reap-expert-swap-vllm0171/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/layer_utils.py`

These preserve REAP/UVA metadata through vLLM parameter replacement after weight loading.

## Qwen dynamic plan facts

Plan used:
- `/home/ser/reap-expert-swap-reap/output/qwen35-dynamic-10plus10-20260309/plan.json`

Metadata:
- `mode: dynamic_core_specialist`
- `signalKey: expert_frequency`
- `perLayer`: 40 layers

Observed plan shape:
- each layer has `coreExperts` length `19`
- each layer has a `sliceCatalog`

Examples:
- `layer_0 coreExperts`:
  - `[25, 39, 62, 64, 73, 75, 79, 89, 125, 161, 164, 170, 188, 207, 216, 219, 230, 231, 236]`
- `layer_1 coreExperts`:
  - `[14, 18, 24, 33, 46, 50, 51, 75, 84, 86, 127, 137, 192, 193, 214, 219, 220, 222, 237]`

Most frequently core-marked expert IDs across layers include:
- `89`, `157`, `165`, `203`, `99`, `33`, `220`, `42`, `202`, `206`, `23`, `208`, `72`, `229`

## Current live Qwen state

Last confirmed binding snapshot on the live server:
- `resident_count: 8`
- `offloaded_count: 72`
- resident examples start around layers `24-27`
- early-layer fused MoE tensors are offloaded

This means the runtime is already maintaining a real resident vs offloaded set.

## Immediate next steps

### A. Startup core preload
Implement startup preload from the Qwen plan’s `coreExperts`-derived resident set so the server starts with an explicit REAP-informed core budget rather than an implicit mixed start.

### B. Budget-safe swap order
Finalize and validate:
1. offload deselected residents first
2. free cache
3. onload selected targets second
4. never exceed the pre-swap resident budget

### C. 3-arm benchmark
Stand up:
1. BF16 arm on GPUs `0-3`
2. plain non-dynamic offload arm on GPUs `4-5`
3. dynamic arm on GPUs `6-7`

Then run 10 samples and collect:
- correctness
- coherence
- TTFT
- prefill speed
- generation speed
- VRAM usage
- swap amount

### D. Prefetch backend experiment
After the above checkpoint, test a stock vLLM prefetch lane using:
- `--offload-backend prefetch`
- `--offload-group-size`
- `--offload-num-in-group`
- `--offload-prefetch-step`
- `--offload-params experts`

This should be treated as a separate benchmark against plain UVA and dynamic UVA.

## Bottom line

As of this checkpoint:
- live runtime patching is real
- live UVA swapping is real
- live VRAM movement is proven
- server survival after swap is proven
- true per-expert slice-level control is still not implemented
