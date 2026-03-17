# PROGRESS.md - Qwen selective-UVA closure and Kimi pivot

Date: 2026-03-17
Repo: /Users/sero/ai/autoresearch
Remote host: ser@192.168.1.70

## Current objective

Close Qwen with an honest speed/quality conclusion, then carry the winning resident-budget logic back to Kimi.

## Late update: miss-driven hotset controller is not the winner

### Warmup and benchmark artifacts
- `/home/ser/proof-output/kimi-vllm/qwen-hotset-warm16-1773784456.json`
- `/home/ser/proof-output/kimi-vllm/qwen-hotset-pinned10-chat-bench-1773784658.json`

### What the hotset run proved

The pinned-base miss-driven hotset controller is materially better than the earlier broken hotset run, but it still does **not** beat the valid static preload winner.

Warmup behavior:
- bootstrap resident set stayed at stock `24-39`
- first real hotset refresh added layer `20` while keeping the full stock tail pinned
- second refresh changed the extra layer to `15`
- during the 10-prompt benchmark, a third refresh changed the extra layer to `19`

Observed swap accounting:
- second refresh swap bytes: `3221225472`
- third refresh swap bytes: `3221225472`

Pinned hotset benchmark summary:
- **TTFT:** `1.7778632640838623s`
- **Prefill:** `120.03727298843728 tok/s`
- **Generation:** `10.712105774048581 tok/s`

Comparison:
- better than stock control on TTFT (`2.85s -> 1.78s`)
- only slightly better than stock on prefill (`116.29 -> 120.04 tok/s`)
- still materially worse than the valid static preload winner (`1.59s`, `133.18 tok/s`)
- decode/generation speed regressed badly versus both stock and static preload

Quality on the pinned hotset run:
- no Chinese drift
- no punctuation-collapse failure
- but outputs frequently turned into `Thinking Process` / internal reasoning-trace style responses instead of clean answers

So this is **not** a quality-preserving speed win. It is an exploratory negative result.

### Router-miss observability fix

The hotset controller exposed a real metrics bug:
- `/forensics/{request_id}` contained real `router_miss_summary` / `router_miss_by_layer`
- `/router_misses/{request_id}` was still returning zeroed aggregates in hotset mode

Live fix now applied on the remote multiplex server:
- `/home/ser/reap-expert-swap-reap/scripts/vllm_multiplex_server.py`

Current behavior:
- `/router_misses/{request_id}` falls back to the per-request forensic payload when the worker aggregate is empty in hotset mode
- verified on live request `182220c9099743129ee67b017f6db1ab`
- returned:
  - `inactive_mass_total: 7096.04150875`
  - `observed_mass_total: 7119.99996412`
  - `inactive_ratio: 0.99663505`
  - `inactive_expert_total: 4703`

### Qwen decision

Freeze the Qwen proof on the **static** resident-set result, not the miss-driven hotset path.

Current Qwen winner remains:
- `1,2,3,4,5,6,24,25,26,27,28,29,30,31,38,39`
- artifact: `/home/ser/proof-output/kimi-vllm/qwen-personal16_a_10-chat-bench-1773763213.json`

Interpretation:
- Qwen proved that **better upfront resident-set selection** improves TTFT/prefill
- Qwen did **not** prove that the current miss-driven hotset controller is better than the best static preload
- the hotset controller is now useful as an observability/debugging path, not the production proof path

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

### Fresh no-op control on the corrected 16-layer stock set
- `/home/ser/proof-output/kimi-vllm/qwen-stock16_manual-chat-bench-1773762777.json`

What it proved:
- a **no-op** `debug_uva_prefix_swap` over the full current resident set does **not** corrupt outputs
- the earlier punctuation-collapse artifacts came from using an **invalid 8-layer target** that silently halved the resident budget, not from the swap hook itself

Summary:
- **TTFT:** `5.24s`
- **Prefill:** `58.53 tok/s`
- **Generation:** `14.24 tok/s`

### 10-prompt apples-to-apples stock control
- `/home/ser/proof-output/kimi-vllm/qwen-stock16_10-chat-bench-1773763097.json`

Resident set:
- stock resident layers = `24-39`

Summary on 10 personal-calibration prompts:
- **TTFT:** `2.85s`
- **Prefill:** `116.29 tok/s`
- **Generation:** `14.13 tok/s`

### 10-prompt mixed personal preload winner so far
- `/home/ser/proof-output/kimi-vllm/qwen-personal16_a_10-chat-bench-1773763213.json`

Resident set after swap:
- `1,2,3,4,5,6,24,25,26,27,28,29,30,31,38,39`
- i.e. replace stock layers `32-37` with personal-hot layers `1-6`

Swap accounting:
- **onloaded:** `12` params
- **offloaded:** `12` params
- **bytes_onloaded:** `4831838208`
- **bytes_offloaded:** `4831838208`

Summary on the same 10 personal-calibration prompts:
- **TTFT:** `1.59s`
- **Prefill:** `133.18 tok/s`
- **Generation:** `13.94 tok/s`

Interpretation:
- **TTFT improved ~44%** vs the 10-prompt stock control (`2.85s -> 1.59s`)
- **Prefill improved ~14.5%** (`116.29 -> 133.18 tok/s`)
- **Generation stayed roughly flat / slightly down** (`14.13 -> 13.94 tok/s`)
- outputs remained coherent enough to keep testing, but one prompt drifted into Chinese and another showed a minor truncation/typo, so this is a **promising speed result**, not final parity proof

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

But the more important later finding is this:
- those early tests only targeted **8 layers**
- the real stock resident set on this server is **16 layers** (`24-39`)
- so the bad punctuation-collapse runs were not valid apples-to-apples comparisons; they cut the resident budget in half

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

The current winning lane is now:

1. start from the stock 16-layer resident set
2. preserve the **same resident-layer count**
3. replace only the least-interesting stock slice with the hottest personal slice

Current best mixed set:
- `1,2,3,4,5,6,24,25,26,27,28,29,30,31,38,39`

This is the first preload shape that showed:
- materially better TTFT
- materially better prefill
- no punctuation-collapse failure

## Next step

Move back to `Kimi-K2.5-PRISM-REAP-530B-A32B` and apply the Qwen lesson directly:

1. use the existing Kimi observation outputs as the input signal
2. prefer **static or startup resident-budget selection** first, not prompt-by-prompt hotset churn
3. keep `REAP_SWAP_MASKS_ONLY=1` and `REAP_ENABLE_ROUTER_MASKS=0`
4. benchmark Kimi for TTFT, prefill, generation, swap bytes, resident budget, router misses, and quality

The Kimi result only counts if:
- the server actually loads and serves from vLLM `0.17.0`
- resident/offloaded behavior is measured from the live server
- speed gains are paired with acceptable output quality
