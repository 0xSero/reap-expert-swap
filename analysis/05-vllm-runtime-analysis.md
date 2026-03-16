# 05 — vLLM Runtime Analysis: MoE Features, Custom Server Gap, and Forward Path

**Date:** 2026-03-16  
**vLLM Version:** v0.17.1 (stable), latest dev preview  
**Model Focus:** Qwen3.5-35B-A3B (256 experts, 8 routed + 1 shared)

---

## A. Native vLLM MoE Features (as of March 2026)

### A1. `--cpu-offload-params experts`

**Available since:** v0.15+ (present in v0.17.1 stable)

**What it does:**  
Selectively offloads model parameters whose fully-qualified name contains the segment `"experts"` to CPU pinned memory. The remaining parameters (attention, router, shared experts, embeddings, layer norms) stay on GPU. During each forward pass, offloaded weights are accessed via UVA (Unified Virtual Addressing) — they are streamed from CPU→GPU on demand.

**Usage:**
```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
    --cpu-offload-gb 28 \
    --cpu-offload-params experts
```

**Limitations:**
- **Not routing-aware.** All expert weights in the offloaded set are touched during every forward pass, regardless of which experts the router actually selected. There is no selective caching of hot experts.
- **No on-demand loading.** All experts must be present in memory (GPU or CPU) at startup. JIT or lazy expert loading is not supported.
- **Dumb page-in/page-out** of entire weight tensors. No intelligence about access frequency or expert popularity.
- Confirmed by vLLM forum (Nov 2025) and the RunLLM support bot: "vLLM requires all experts to be loaded at initialization."

### A2. `--enable-return-routed-experts`

**Available since:** v0.14+ (present in v0.17.1 stable)

**What it does:**  
When enabled, vLLM returns which experts were actually selected by the router for each generated token. The data is included in the response metadata.

**Usage:**
```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
    --enable-return-routed-experts
```

**How it maps to router miss tracking:**  
This provides the raw "which experts were routed" data that could be consumed by an external orchestrator to detect routing patterns, build active-set predictions, or feed a REAP-style planner. However, it does **not**:
- Report probability mass allocated to each expert
- Indicate whether an expert was masked or unavailable
- Track cumulative miss statistics across layers
- Integrate with any swap or offload mechanism

It is a **passive telemetry feature**, not an active control mechanism. The custom server's router miss tracking hooks are strictly more powerful — they compute per-layer inactive mass, per-token miss probabilities, and feed back into decode-refresh decisions.

### A3. Expert Parallel + EPLB

**Available since:** v0.8+ (EP), v0.10+ (EPLB); mature in v0.17.x

**What it does:**  
Expert Parallelism (EP) distributes experts across multiple GPUs so each GPU holds a subset. EPLB (Expert Parallel Load Balancer) dynamically rebalances expert placement based on runtime load statistics, replicating hot experts across more ranks.

**Usage:**
```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2}'
```

**Key parameters:**

| Parameter | Description | Default |
|---|---|---|
| `window_size` | Engine steps to track for rebalancing | 1000 |
| `step_interval` | Rebalance frequency (every N steps) | 3000 |
| `num_redundant_experts` | Extra experts per EP rank beyond equal split | 0 |
| `use_async` | Non-blocking EPLB for reduced latency | false |
| `policy` | Load balancing policy type | "default" |

**When to use:**  
Multi-GPU serving where you want to shard experts across GPUs (e.g., 8×H200 for DeepSeek-V3). Requires DeepEP kernel installation for multi-node communication.

**When NOT applicable:**  
Single-GPU memory compression. EP does not reduce the total number of loaded experts — it redistributes them. It cannot replace sparse subset approaches like the REAP server.

### A4. New Developments Since Previous Research

| Feature | Status | Notes |
|---|---|---|
| `fast_moe_cold_start` (compilation config) | Present in v0.17.x | Compilation optimization for MoE startup; not a dynamic expert management feature. Limited documentation. |
| Dual Batch Overlap (`--enable-dbo`) | Stable in v0.17.x | Overlaps all-to-all communication with compute for EP workloads. Not related to expert offloading. |
| Async scheduling (`--async-scheduling`) | Experimental | Overlaps scheduling with model execution. |
| `--performance-mode interactivity` | v0.17.x | Fine-grained CUDA graphs, latency-oriented kernels. May benefit single-GPU MoE. |
| Multiple MoE backends (`--kernel-config.moe_backend`) | v0.17.x | Options: `auto`, `triton`, `deep_gemm`, `cutlass`, `flashinfer_trtllm`, `flashinfer_cutlass`, `marlin`. Allows tuning MoE kernel performance per model. |

---

## B. Custom Server Features — Comparison Table

The REAP multiplex server (`reap_swap/vllm_multiplex_server.py`) provides capabilities that native vLLM does not. Below is a feature-by-feature comparison:

| Feature | Custom Server (REAP) | vLLM v0.17.x Native | Gap |
|---|---|---|---|
| **Per-request active set control** | ✅ `POST /swap_active_set` with explicit expert lists per layer | ❌ No API to specify which experts should be loaded | **Full gap** — no native equivalent |
| **Incremental delta swaps** | ✅ `compute_keep_set_delta()` copies only added experts, zeroes only removed ones | ❌ `--cpu-offload-params` touches all experts every pass | **Full gap** — no delta optimization |
| **Router mask injection** | ✅ Forward hooks add `-inf` to logits for inactive experts before TopK | ❌ No mechanism to mask/bias router logits | **Full gap** — no router control |
| **Router miss tracking** | ✅ Per-layer, per-token, per-request: tracks inactive mass, inactive expert sets, observed mass | ⚠️ Partial: `--enable-return-routed-experts` returns routed expert IDs but no mass/miss tracking | **Partial gap** — native provides raw IDs only |
| **Dense-to-sparse cold shrink** | ✅ First swap zeroes hundreds of experts, converting from full model to sparse specialist | ❌ No concept of dynamically subsetting a loaded model | **Full gap** |
| **CPU base snapshot + restore** | ✅ Full expert weights in CPU pinned memory; any expert restorable to GPU on demand | ❌ `--cpu-offload-params` stores in CPU but no selective restore | **Full gap** — no on-demand expert loading |
| **Signature-based no-op optimization** | ✅ If active set unchanged, swap is skipped entirely (0 bytes, instant return) | ❌ N/A (no swap concept) | **Full gap** |

### Summary

The REAP server fills a gap that does not yet exist in native vLLM: **runtime, per-request, routing-aware expert management for single-GPU memory-constrained deployments.** Native vLLM's MoE features are oriented toward multi-GPU scaling (EP/EPLB) or static parameter offloading (`--cpu-offload-params`), not dynamic subsetting.

---

## C. Forward-Looking

### C1. Status of PR #31938 and Issue #33869

**PR #31938 (`moe-offload-fixed`):**  
- **State:** Closed (not merged)
- **Author:** Digital China Group, based on vLLM v0.14.0 fork
- **What it proposed:** True kernel-level MoE CPU offloading with:
  - GPU Expert Cache (only `cache_expert_num` hot experts per layer)
  - CPU pinned memory holding all experts
  - Routing-aware cache policy (LRU-like based on `topk_ids`)
  - DBO mode: parallel GPU + CPU expert computation
  - Prefetch mode: double-buffered miss expert buffers
  - AVX512/AMX-optimized CPU MoE FFN kernels
  - New C++/CUDA extension (`_offload_C`) with OpenMP threading
- **Why closed:** Maintainers requested it be converted to an RFC/draft given the large scope (28 files changed). Code quality issues flagged by reviewers (division by zero, hardcoded NUMA configs, busy-waiting patterns).

**Issue #33869 (RFC):**  
- **State:** Open (as of Feb 2026)
- **Content:** Repackages the PR #31938 design as a formal RFC
- **No successor PR has been opened.** The RFC remains in discussion.

### C2. What Would Change If True MoE Offloading Lands in vLLM

If PR #31938's approach (or a successor) is merged into mainline vLLM, the following changes to the REAP server would be needed:

1. **CPU expert computation replaces zeroing.** The REAP server currently zeroes inactive experts and relies on router masks to prevent selection. Native offloading would compute cold experts on CPU instead, preserving information. The zeroing + mask approach would become unnecessary for correctness but could still be used for performance (skipping CPU compute for truly unwanted experts).

2. **Cache management moves to the kernel layer.** The per-layer cache policy (`cache_map`, `miss_map`, `policy_sort`) would be managed internally by vLLM's `CpuOffloadInfer` module. The REAP server's `compute_keep_set_delta()` and explicit GPU weight manipulation would need to be replaced with API calls to the native cache manager.

3. **Router masks may become redundant.** If the native system handles cache misses transparently (compute on CPU), there's no need to mask the router. However, the REAP server's **per-request active set control** (using specific experts for specific use cases) would still require some form of router bias.

4. **`/swap_active_set` endpoint would need reimplementation.** Instead of zeroing/restoring GPU weights directly, it would configure the native offload manager's cache preferences (e.g., "pin these experts to GPU cache" or "evict these experts").

5. **Router miss tracking might be simplified.** If the native system tracks cache hits/misses internally, the custom forward hooks could be replaced with queries to the native cache statistics.

### C3. Monkey-Patching Risks Across vLLM Versions

The REAP server relies on monkey-patching internal vLLM classes:

```python
# Patched module paths (fragile across versions):
"vllm.v1.worker.gpu_worker.Worker"
"vllm.v1.worker.cpu_worker.CPUWorker"
"vllm.v1.worker.xpu_worker.XPUWorker"
```

**Known risks:**
- **Worker class relocation:** vLLM has moved worker classes between versions (e.g., `vllm.worker` → `vllm.v1.worker`). Any reorganization breaks the patch.
- **Method signature changes:** The code assumes `self.model_runner.model.named_parameters()` traversal pattern and specific model tree shapes (`model.model.layers`, `model.language_model.model.layers`). Architecture-specific changes break the resolution.
- **Router/gate API changes:** The hook assumes `gate.register_forward_hook()` is available, and that the gate module has a `weight` parameter with shape `[num_experts, ...]`. Changes to the router implementation (e.g., grouped router, shared router) could break this.
- **`engine.collective_rpc()` API:** The server uses `engine.collective_rpc("method_name", args=(...))` to invoke methods on workers. This API could change in calling conventions or timeout handling.
- **Expert map format:** The code handles `_expert_map` tensors with specific semantics (global→local mapping). EP changes could alter this data structure.

**Mitigation:** Pin to specific vLLM versions and test after each upgrade. The server currently targets vLLM v0.17.x; upgrading to v0.18+ will likely require patching adjustments.

---

## D. Qwen3.5-35B-A3B Specific

### D1. Architecture

Qwen3.5-35B-A3B uses a **Gated Delta Networks + MoE hybrid** architecture:

```
Total Parameters: 35B (3B active per token)
Hidden Dimension: 2048
Layers: 40

Hidden Layout (per 10-layer block):
  10 × (
    3 × (Gated DeltaNet → MoE)    ← linear attention + MoE
    1 × (Gated Attention → MoE)    ← full attention + MoE
  )

Total blocks: 10
Layers per block: 4 (3 DeltaNet-MoE + 1 Attention-MoE)
Total layers: 40
```

**MoE Configuration:**
- 256 total experts per MoE layer
- 8 routed experts + 1 shared expert per token
- Expert intermediate dimension: 512
- Token embedding: 248,320 (padded)

**Attention Configuration:**
- DeltaNet: 32 linear attention heads (V), 16 heads (QK), dim 128
- Gated Attention: 16 Q heads, 2 KV heads (GQA), dim 256, RoPE dim 64

**Context:** 262,144 tokens natively; extensible to 1,010,000 via YaRN RoPE scaling.

### D2. `--language-model-only` Flag

When serving Qwen3.5 for text-only workloads, the `--language-model-only` flag skips loading the vision encoder and multimodal preprocessing. This:
- Frees GPU memory for additional KV cache
- Enables Expert Parallelism (required for EP deployment)
- Eliminates multimodal profiling overhead

**Recommended usage:**
```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
    --language-model-only \
    --reasoning-parser qwen3
```

**Note:** Qwen3.5 is a **unified vision-language** model. The `--language-model-only` flag disables vision capabilities entirely. This is appropriate for pure text workloads but incompatible with image/video input.

### D3. Known Inference Quirks

| Issue | Details | Workaround |
|---|---|---|
| **B200 / Blackwell FP8 garbage output** | Qwen3.5-35B-A3B-FP8 produces random/garbage output on B200 and B100 GPUs with vLLM v0.17.0/v0.17.1. Works correctly on A100 and H200. ([#36773](https://github.com/vllm-project/vllm/issues/36773)) | Use BF16 model instead of FP8 on Blackwell; or try `VLLM_USE_FLASHINFER_MOE_FP8=0` or `--kv-cache-dtype fp8 --dtype bfloat16` |
| **CUDA graph capture vs. Mamba cache** | `causal_conv1d_update` assertion failure when CUDA graph capture size exceeds Mamba cache size. | Reduce `--max-cudagraph-capture-size` below default 512. See [PR #34571](https://github.com/vllm-project/vllm/pull/34571). |
| **MTP speculative decoding** | MTP-1 reduces per-token latency but degrades throughput under high concurrency (speculative tokens consume KV cache). | Disable MTP for throughput-critical workloads; use for latency-sensitive, low-concurrency scenarios only. |
| **Thinking mode by default** | Qwen3.5 generates `<think>...</think>` content before final response. No soft `/think`/`/nothink` toggle (unlike Qwen3). | Disable via `chat_template_kwargs: {"enable_thinking": False}` in API calls. |
| **Expert routing sparsity** | With 8/256 experts per token (3.1% density), routing is extremely sparse. Novel prompts may trigger rarely-used experts not in a REAP active set. | Monitor router miss statistics via `/router_misses/{request_id}` and trigger decode refresh swaps when inactive mass exceeds threshold. |
| **DeltaNet-MoE vs. Attention-MoE layers** | 30 of 40 MoE layers follow DeltaNet (linear attention); 10 follow full Attention. Expert usage patterns may differ between these two types. | Consider layer-type-aware REAP planning (separate budgets for DeltaNet-MoE and Attention-MoE layers). |

### D4. vLLM Serving Configurations (from official recipes)

**Throughput-focused (text-only, 8× GPU):**
```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    -dp 8 \
    --enable-expert-parallel \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching
```

**Single-GPU (35B-A3B variant):**
```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only
```

**With MTP speculative decoding:**
```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 8 \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

---

## Sources

### Web Sources
1. [vLLM CLI / serve docs (stable)](https://docs.vllm.ai/en/stable/cli/serve/) — `--enable-return-routed-experts`, `--cpu-offload-params` docs
2. [Expert Parallel Deployment docs (latest)](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) — EP + EPLB configuration
3. [GitHub Issue #33869: RFC MoE Offload](https://github.com/vllm-project/vllm/issues/33869) — Open RFC for true MoE offloading
4. [GitHub PR #31938: moe-offload-fixed](https://github.com/vllm-project/vllm/pull/31938) — Closed PR (Digital China fork, v0.14.0)
5. [GitHub Issue #36773: Qwen3.5 B200 FP8 bug](https://github.com/vllm-project/vllm/issues/36773) — Blackwell FP8 garbage output
6. [vLLM Recipes: Qwen3.5 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) — Official deployment recipes
7. [Qwen3.5-35B-A3B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) — Architecture details, benchmarks
8. [vLLM Forum: Enable Expert Offloading](https://discuss.vllm.ai/t/enable-expert-offloading/1884) — No native expert offloading confirmed
9. [vLLM Blog: Large Scale Serving with Wide-EP](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) — EPLB at scale

### Code Sources (Local)
1. `/Users/sero/ai/autoresearch/reap_swap/vllm_multiplex_server.py` — Custom REAP server (~780 lines)
2. `/Users/sero/ai/autoresearch/reap_swap/dynamic_swap_delta.py` — Delta computation
3. `/Users/sero/ai/autoresearch/reap_swap/dynamic_reap.py` — Validation, plan management
4. `/Users/sero/ai/autoresearch/test-output/vllm_moe_offloading_research_report.md` — Previous research report
