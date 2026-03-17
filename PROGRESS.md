# PROGRESS.md - Qwen selective-UVA proof and preload benchmark lane

Date: 2026-03-17
Repo: /Users/sero/ai/autoresearch
Remote host: ser@192.168.1.70

## Current objective

Prove that **upfront loading of the best weights for the right workload** is measurably better than stock vLLM UVA on personal calibration prompts, then carry the winning shape back to Kimi.

## Current truth

### 1. Selective UVA is real again

The live Qwen runtime on the remote host is again binding MoE tensors into the selective-UVA registry.

Confirmed from `/debug_uva_binding` on the corrected run:
- `bound_param_count: 80`
- `offloader_name_count: 80`
- `offloader_entry_count: 160`

This only became true again after fixing the running process environment to include:
- `VLLM_REAP_SELECTIVE_UVA=1`

Without that env, warm/swap endpoints were effective no-ops.

### 2. Live swapping is real

The runtime can move bound fused MoE tensors in and out of GPU memory while the server is live.

Previously-proved direct live swap on Qwen:
- onloaded 2 fused tensors
- `bytes_onloaded: 805306368`
- VRAM rose by ~774 MiB per GPU
- server still answered after the swap

### 3. The current dynamic Qwen plan is not prompt-sensitive

Plan:
- `/home/ser/reap-expert-swap-reap/output/qwen35-dynamic-10plus10-20260309/plan.json`

Important finding:
- the current plan has only one source summary:
  - `personal_live_qwen35_2k_20260309`
- `build_active_set_payload(...)` returns the same selected-slice signature across distinct personal prompts
- tags differ, but signatures do not

So the current "dynamic" planner is effectively a **single personal heatmap**, not a request-varying routing policy.

This means any claimed preload win must come from:
1. a better planner, or
2. an explicit startup hot-layer preload policy

—not from the current constant-signature planner.

## Benchmark artifacts captured

### Invalid/contaminated A/B artifact
- `/home/ser/proof-output/kimi-vllm/qwen-personal-preload-ab-1773758037.json`

Why it is not a valid preload-win proof:
- only one row did real warm activity
- later rows were `no_op_reuse: true`
- the current planner reused the same active-set signature

### Stock-UVA baseline on personal prompts
- `/home/ser/proof-output/kimi-vllm/qwen-stock-chat-bench-1773759550.json`

Summary:
- **TTFT:** `5.26s`
- **Prefill:** `59.23 tok/s`
- **Generation:** `14.22 tok/s`

This is the current honest baseline to beat.

## Runtime patches now in play on remote

### vLLM runtime
- `/home/ser/reap-expert-swap-vllm0171`
- vLLM `0.17.1`

### Patched files
1. UVA offloader:
- `/home/ser/reap-expert-swap-vllm0171/lib/python3.12/site-packages/vllm/model_executor/offloader/uva.py`

Current relevant additions:
- selective registry restore/bind helpers
- `reap_set_resident_prefixes(...)`
- exchange-budget-aware offload-first/onload-second logic
- startup resident-prefix support via `REAP_START_RESIDENT_PREFIXES`

2. REAP multiplex server:
- `/home/ser/reap-expert-swap-reap/scripts/vllm_multiplex_server.py`

Current relevant additions:
- `/debug_uva_binding`
- `/debug_uva_prefix_swap`
- startup preload hook
- startup preload now accepts explicit resident prefixes from env in the worker path

3. Launcher script on remote:
- `/tmp/relaunch_qwen_dynamic.sh`

Current relevant fixes:
- preserves external overrides for:
  - `REAP_STARTUP_PRELOAD_CORE`
  - `REAP_STARTUP_PRELOAD_BLOCKING`
- exports `VLLM_REAP_SELECTIVE_UVA=1`

## What failed and why

### Manual hot-swap after load
Manual hot swap to layers like:
- `38, 39, 6, 5, 1, 3, 2, 4`

did prove the exchange-budget logic works:
- offloaded first
- onloaded second
- changed resident layers successfully

But it pushed the engine too close to the limit and later debug calls OOMed.

### Startup preload attempt #1
The first startup-resident attempt failed because loader-time names were local module names, not fully-qualified names like:
- `language_model.model.layers.38.mlp.experts.`

So matching at initial selective offload time was the wrong control point.

### Current startup preload fix
The new approach is:
1. let vLLM load normally in selective mode
2. bind full names
3. run startup preload from the worker startup hook using explicit resident prefixes

That is the correct control point for a real preload test.

## Current active lane

The current run in progress is the **startup hot-preload lane**.

Target hot layers currently being tested:
- `38, 39, 6, 5, 1, 3, 2, 4`

These were chosen from the plan by highest `coreActivationMass`.

The purpose is simple:
- start with a hot resident set already loaded
- benchmark the same personal prompts
- compare directly against stock UVA

## Next step

Finish the startup hot-preload run, then produce an honest 2-arm comparison:

1. **stock UVA**
2. **startup hot preload**

on the same personal calibration prompts.

The result only counts if:
- the startup preload actually changes `resident_layers`
- the server remains healthy through the benchmark
- TTFT/prefill improvement is measured against the stock artifact above
