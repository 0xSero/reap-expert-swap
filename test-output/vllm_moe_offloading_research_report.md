# vLLM 0.17.x MoE Expert Offloading & Dynamic Expert Management — Technical Research Report

**Date:** 2026-03-16  
**Scope:** vLLM v0.17.x capabilities for MoE expert offloading, the custom REAP multiplex server, and opportunities for replacement/improvement.

---

## 1. How vLLM's `--cpu-offload-params experts` Works

### Official Documentation (vLLM stable / v0.17.1)

From the [vLLM Engine Arguments docs](https://docs.vllm.ai/en/stable/configuration/engine_args/):

> **`--cpu-offload-params`**: The set of parameter name segments to target for CPU offloading. Unmatched parameters are not offloaded. If this set is empty, parameters are offloaded non-selectively until the memory limit defined by `cpu_offload_gb` is reached.  
> Examples: For parameter name `"mlp.experts.w2_weight"`: `"experts"` or `"experts.w2_weight"` will match. `"expert"` or `"w2"` will NOT match (must be exact segments).

### What It Actually Offloads

When combined with `--cpu-offload-gb 28 --cpu-offload-params experts`:

1. **Selective parameter offloading**: Only model parameters whose fully-qualified name contains the segment `"experts"` are moved to CPU memory. This targets the MoE expert FFN weights (`w1`, `w2`, `w3` / gate, up, down projections) while leaving attention weights, router weights, shared experts, embeddings, and layer norms on GPU.

2. **UVA (Unified Virtual Addressing) / zero-copy access**: The offloaded weights reside in CPU pinned memory and are accessed via UVA. During each forward pass, the needed expert weights are streamed from CPU→GPU on-the-fly. This is a **static, non-intelligent** mechanism — vLLM does NOT selectively load only the routed experts; ALL expert weights in the offloaded set are touched during every forward pass.

3. **Memory budget**: The `--cpu-offload-gb 28` flag reserves 28 GiB of "virtual GPU expansion" per GPU. Effectively, any expert parameters that would exceed GPU VRAM are kept in CPU RAM and paged in during inference.

### Key Limitation

As confirmed by vLLM forum posts (Nov 2025) and the [RunLLM bot](https://discuss.vllm.ai/t/enable-expert-offloading/1884):

> "vLLM requires all experts to be loaded at initialization... there is no support for on-demand loading of only the gated experts; all experts must be present in memory (GPU or CPU) at startup. On-demand or JIT expert loading is not currently supported in vLLM."

The mechanism is a **dumb page-in/page-out** of entire weight tensors — NOT routing-aware expert caching. Every forward pass copies the full expert weights from CPU to GPU regardless of which experts are actually selected by the router.

---

## 2. The Custom `/swap_active_set` Endpoint

### Location
`/Users/sero/ai/autoresearch/reap_swap/vllm_multiplex_server.py` — the `_swap_active_set_impl()` async function, exposed at `POST /swap_active_set`.

### What It Does

The endpoint implements **per-request, routing-aware dynamic expert management**. Before each inference request, the client sends a payload specifying which experts should be "active" (resident with real weights on GPU) for each layer. The server then:

1. **Validates the payload** against the REAP plan (`validate_active_set_payload()` in `dynamic_reap.py`), checking:
   - Phase is `"prefill"` or `"decode_refresh"`
   - Active set respects the plan's budget constraints (byte budget from `plan["budget"]`)
   - Core experts are present in the union
   - Generates a signature for the active set

2. **Computes a delta** (`compute_keep_set_delta()` in `dynamic_swap_delta.py`):
   - Compares the **current** active set on GPU vs. the **desired** active set
   - Identifies which experts need to be **added** (restored from base snapshot → GPU) and which need to be **removed** (zeroed on GPU)
   - Tracks **reused** experts that don't need any swap

3. **Performs the GPU swap** via `engine.collective_rpc("multiplex_swap_active_set", ...)`:
   - On each worker, `MultiplexWorkerExtension.multiplex_swap_active_set()` iterates over all model parameters containing `"experts"` in their name
   - For each parameter tensor indexed by `[local_expert_idx, ...]`:
     - If the expert is in the **removed** set → `param.data[local_expert_idx].zero_()` (zero it out)
     - If the expert is in the **added** set → `param.data[local_expert_idx].copy_(base_snapshot[...])` (restore from CPU pinned base)
   - Then applies **router masks** via `_apply_router_masks_and_hooks()`

4. **Applies router masks** to prevent the router from selecting zeroed-out experts:
   - Registers a forward hook on each `gate` (router) module
   - The hook adds `-inf` to logits for non-active experts, forcing the TopK selection to ignore them
   - Also tracks "router misses" — when the router *would have* selected a zeroed expert absent the mask

5. **Signature-based no-op optimization**: If the same `active_set_signature` is already loaded globally, the swap is skipped entirely (0 bytes copied, instant return).

6. **Serialized single-flight concurrency**: All swaps are serialized via `app.state.dynamic_swap_lock` (asyncio Lock), ensuring only one swap happens at a time.

### Endpoints Summary

| Endpoint | Purpose |
|---|---|
| `POST /swap_active_set` | Full swap with router mask update |
| `POST /warm_active_set` | Same swap but marked as warm-start (e.g., pre-populating before request) |
| `GET /router_misses/{request_id}` | Retrieve router miss statistics for a request (how much probability mass went to masked experts) |
| `GET /forensics/{request_id}` | Retrieve forensic data about a swap operation |
| `POST /swap_cartridge/{cartridge_id}` | Legacy: swap a whole pre-built "cartridge" (full expert weight snapshot) |

---

## 3. Dense-to-Sparse "Shrink" (Cold Start)

### The Mechanism

The cold start is the process of going from a **dense model** (all 256 experts active with real weights) to a **sparse subset** (e.g., 8-32 active experts per layer, rest zeroed).

#### Step 1: Base Snapshot Creation (`_get_base_expert_snapshot()`)

On first call, the server creates a **CPU-pinned clone of ALL expert parameters**:

```python
for name, param in self.model_runner.model.named_parameters():
    if "experts" not in name:
        continue
    host_copy = param.detach().cpu().clone().pin_memory()
    base_snapshot[name] = host_copy
```

This is the "ground truth" — the full dense model weights stored in CPU RAM with pinned memory for fast GPU transfers.

#### Step 2: Dense Keep Sets Initialization (`_get_dense_keep_sets()`)

The server computes the "dense" keep set — all experts for all layers — from the plan:

```python
def build_dense_keep_sets(plan):
    for layer_key, layer_plan in plan.get("perLayer", {}).items():
        num_experts = infer_num_experts(layer_plan)
        dense_keep_sets[layer_idx] = set(range(num_experts))  # e.g., {0, 1, ..., 255}
```

Initially, `_reap_current_keep_sets` equals the dense set (all experts active).

#### Step 3: First Swap Request Triggers the Shrink

When the first `/swap_active_set` request arrives with a sparse active set (e.g., 16 experts per layer out of 256), `compute_keep_set_delta()` computes:

- **removed**: 240 experts per layer (256 - 16 that are in the desired set)
- **added**: 0 (since going from dense → sparse, the desired experts are already loaded)
- The worker then **zeros out** the 240 removed experts' weights on GPU

This is the "shrink" — a massive one-time zeroing operation that converts the dense model to a sparse specialist configuration.

#### Step 4: Subsequent Swaps Are Incremental

After the cold start, subsequent swaps only touch the **delta**:
- If going from active set A to active set B, only experts in `B - A` are restored from the base snapshot, and experts in `A - B` are zeroed.
- Experts in `A ∩ B` are untouched (reused).

---

## 4. Router Masking / Zeroing Mechanism

### Two-Layer Defense

The system uses **two complementary mechanisms** to handle non-active experts:

#### Layer 1: Weight Zeroing (Hard)

Non-active expert weights are literally set to zero on GPU via `param.data[local_expert_idx].zero_()`. This means even if an expert *were* selected by the router, its computation would produce zero output (no contribution to the hidden state).

#### Layer 2: Router Masking (Soft — Forward Hook)

A forward hook is registered on each router/gate module (`gate.register_forward_hook(make_hook())`):

```python
mask = torch.zeros(num_experts, dtype=float32, device=gate_weight.device)
for expert_idx in range(num_experts):
    if expert_idx not in keep_set:
        mask[expert_idx] = float("-inf")

# In the hook:
masked_logits = output[0] + mask.unsqueeze(0)  # -inf for inactive experts
```

The `-inf` mask is added to the router logits **before** TopK selection, ensuring the router NEVER selects a zeroed expert. This is critical because:
- Even with zeroed weights, selecting a masked expert wastes compute (routing tokens to a no-op expert)
- The router's softmax normalization would be distorted by near-zero logits

#### Router Miss Tracking

The hook also tracks **what the router would have done** without masking:
- Before applying the mask, it computes TopK on the raw logits
- Tracks which experts in the TopK are inactive and their probability mass
- This data feeds back to the orchestrator for dynamic refresh decisions (if too much mass goes to inactive experts, trigger a `decode_refresh` swap)

#### Expert Map Awareness

The code handles vLLM's internal expert remapping (`_expert_map`) which maps global expert IDs to local (potentially compressed) indices when using expert parallelism or other optimizations:

```python
expert_map = getattr(expert_layer, "_expert_map", None)
if expert_map is not None:
    for global_idx, mapped_local_idx in enumerate(expert_map.tolist()):
        if mapped_local_idx >= 0:
            mapping[int(mapped_local_idx)] = int(global_idx)
```

---

## 5. Newer vLLM Features (0.17+) That Could Replace This Custom Server

### 5a. `--cpu-offload-params experts` (Available in v0.17.x stable)

**Status**: Already used by this codebase, but it's a **dumb offload** — NOT routing-aware. It cannot replace the custom server's dynamic expert management.

### 5b. `--enable-return-routed-experts` (Available in v0.17.x stable)

From the engine args docs:
> Whether to return routed experts. Default: False

This new flag allows vLLM to return which experts were actually selected by the router for each request. This could **partially replace** the custom router miss tracking hooks, but:
- It only reports *which* experts were selected, not the probability mass to masked experts
- It doesn't provide the infrastructure for dynamic swapping

### 5c. Expert Parallel Deployment + EPLB (Available in v0.17.x)

vLLM now supports [Expert Parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) with:
- `--enable-expert-parallel` for distributing experts across GPUs
- `--enable-eplb` for dynamic Expert Parallel Load Balancing
- The EPLB **dynamically rebalances** expert placement based on runtime load statistics
- Supports `num_redundant_experts` for popular expert replication

**Relevance**: This is designed for **multi-GPU scaling** (distributing experts across GPUs), not for single-GPU memory compression. It does NOT reduce the total number of loaded experts — it just redistributes them. **Cannot replace** the REAP server's sparse subset approach.

### 5d. RFC: True MoE Expert Offloading (PR #31938 — NOT MERGED, Feb 2026)

[GitHub Issue #33869](https://github.com/vllm-project/vllm/issues/33869) / [PR #31938](https://github.com/vllm-project/vllm/pull/31938) proposes a **proper MoE CPU offload module**:

- **GPU Expert Cache**: Only `cache_expert_num` hot experts per layer on GPU
- **CPU maintains all experts** in pinned memory
- **Routing-aware cache management**: After TopK routing, cache policy determines which experts to copy to GPU vs. compute on CPU
- **Dual execution modes**:
  - DBO (Dual Batch Overlap): GPU and CPU compute in parallel
  - Prefetch: Pre-copy miss experts with double-buffered miss buffers
- **AVX512/AMX CPU kernels**: Optimized CPU-side MoE FFN computation
- **Asynchronous GPU↔CPU callback system**

**Status**: This PR was **closed** (not merged) as of the fetch date. It was a fork-based implementation on v0.14.0 by Digital China Group. The maintainers requested it be converted to an RFC/draft. It is the most technically similar to what the REAP server does, but it operates at the **kernel/framework level** (transparent to the user) rather than at the API/endpoint level.

**If merged**, this would be the closest native replacement for the REAP custom server, as it implements:
- ✅ Routing-aware expert caching (vs. REAP's explicit orchestration)
- ✅ CPU-side expert computation (REAP doesn't do this — it only zeroes)
- ✅ Async GPU-CPU overlap (REAP does sync copies)
- ❌ But it's automated — no external API to control which experts are active (REAP allows explicit orchestration per-request)

### 5e. `fast_moe_cold_start` Compilation Config (v0.17.x)

In the compilation config, there's a `fast_moe_cold_start` field. This appears to be a compilation optimization for MoE model startup, not a dynamic expert management feature. Limited documentation available.

### 5f. KTransformers (External Alternative)

For Qwen3.5-35B-A3B specifically, [KTransformers](https://github.com/kvcache-ai/ktransformers) offers CPU-GPU heterogeneous inference with expert offloading optimized for consumer hardware. This is a full framework alternative, not a vLLM feature.

### Summary: Can Native vLLM Replace the Custom Server?

**No, not as of v0.17.1.** The key features the REAP server provides that vLLM lacks natively:

| Feature | REAP Custom Server | vLLM v0.17.x Native |
|---|---|---|
| Routing-aware expert subsetting | ✅ Via `/swap_active_set` | ❌ |
| Per-request expert set control | ✅ Explicit API | ❌ |
| Router mask injection | ✅ Forward hooks with `-inf` | ❌ |
| Router miss tracking | ✅ Per-layer, per-request | Partial (`--enable-return-routed-experts`) |
| Incremental delta swaps | ✅ Only changed experts | ❌ |
| Dense→sparse shrink | ✅ Cold start zeroing | ❌ |
| CPU expert computation | ❌ (just zeroes) | ❌ (PR #31938 proposes this) |
| Cartridge caching (LRU) | ✅ Pinned CPU memory | ❌ |

---

## 6. Known Limitations of the REAP Approach

### 6a. Performance Limitations

1. **Synchronous weight copying**: `param.data.copy_(source, non_blocking=False)` — each swap blocks until the CPU→GPU copy completes. No overlap with computation.

2. **Full parameter tensor granularity**: The swap operates on entire expert weight tensors (e.g., `w1_weight[expert_idx]`), not at a finer granularity. Each swap touches all parameters for each changed expert.

3. **Single-flight serialization**: Only one swap can happen at a time (`dynamic_swap_lock`). Under high concurrency, requests queue behind the lock.

4. **No CPU-side expert computation**: Unlike the proposed PR #31938, the REAP server can only zero out experts — it cannot run cold experts on CPU. Tokens routed to masked experts produce zero output, losing information.

5. **Cold start latency**: The first swap (dense→sparse) must zero out hundreds of expert weight tensors across all layers. For Qwen3.5-35B-A3B with 256 experts × 40 layers, this is a substantial operation.

### 6b. Correctness / Quality Limitations

6. **Information loss from zeroing**: Unlike proper expert offloading (where cold experts run on CPU), zeroed experts contribute nothing. If the router would have sent significant probability mass to a zeroed expert, the model's output quality degrades.

7. **Router mask distortion**: Adding `-inf` to logits changes the softmax distribution. The remaining active experts receive higher weights than they would in the full model, potentially amplifying certain experts' influence.

8. **Monkey-patching fragility**: The server works by monkey-patching vLLM's `build_app` function and injecting methods into worker classes at import time:
   ```python
   _install_worker_extensions()  # Patches Worker, CPUWorker, XPUWorker classes
   api_server.build_app = build_app_with_swap  # Replaces the app builder
   ```
   This is fragile across vLLM version upgrades. The patched class paths (`vllm.v1.worker.gpu_worker.Worker`) change between versions.

9. **Expert map assumption**: The code assumes expert weight tensors have the expert index as the first dimension (`param[local_expert_idx]`), which is architecture-specific.

### 6c. Qwen3.5-35B-A3B Specific Considerations

10. **256 experts, 8 routed + 1 shared**: With only 8/256 experts active per token, the model is extremely sparse. The REAP approach of keeping ~16-32 active experts per layer means most tokens can be served without mask hits, but:
    - The shared expert is always active (not managed by REAP)
    - Novel token patterns may trigger unexpected expert needs

11. **Gated DeltaNet + MoE hybrid**: Qwen3.5 uses a novel architecture with `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` per 10-layer block. The REAP server treats all MoE layers uniformly, which may not be optimal given the different roles of DeltaNet-associated vs. Attention-associated MoE layers.

12. **Vision encoder integration**: Qwen3.5-35B-A3B includes a vision encoder. The vLLM model card recommends `--language-model-only` for text-only workloads to save memory. The REAP server doesn't appear to handle multimodal routing differently.

### 6d. Operational Limitations

13. **No hot-reloading of plans**: The REAP plan is loaded once at startup from `REAP_PLAN_FILE`. Changing the expert routing strategy requires a server restart.

14. **Memory overhead**: The base snapshot stores ALL expert weights in CPU pinned memory. For Qwen3.5-35B-A3B with 256 experts across 40 layers, this is significant (the full model is 35B params with experts being the bulk).

15. **No graceful degradation**: If the swap fails mid-operation (e.g., OOM during copy), the model weights are left in an inconsistent state. The server includes crash classification (`classify_forensic_crash()`) but no recovery mechanism.

---

## Sources & References

### Web Sources
1. [vLLM Engine Arguments (stable)](https://docs.vllm.ai/en/stable/configuration/engine_args/) — `--cpu-offload-params` documentation
2. [vLLM Forum: Enable Expert Offloading](https://discuss.vllm.ai/t/enable-expert-offloading/1884) — Confirmed no native expert offloading
3. [vLLM Forum: Expert Offloading Feature Request](https://discuss.vllm.ai/t/expert-offloading/1880) — Community discussion
4. [GitHub Issue #33869: RFC MoE Offload](https://github.com/vllm-project/vllm/issues/33869) — True MoE offloading RFC
5. [GitHub PR #31938: moe-offload-fixed](https://github.com/vllm-project/vllm/pull/31938) — Closed PR implementing MoE offload (Digital China fork)
6. [vLLM Expert Parallel Deployment docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) — EP + EPLB
7. [Qwen3.5-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) — Architecture details (256 experts, 8 routed + 1 shared)
8. [vLLM Blog: Large Scale Serving with Wide-EP](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) — DeepSeek-scale MoE deployment
9. [MoE-Infinity Paper](https://arxiv.org/html/2401.14361v2) — Academic reference for offloading-efficient MoE serving

### Code Sources (Local)
1. `/Users/sero/ai/autoresearch/reap_swap/vllm_multiplex_server.py` — Main server (780 lines)
2. `/Users/sero/ai/autoresearch/reap_swap/dynamic_swap_delta.py` — Delta computation (85 lines)
3. `/Users/sero/ai/autoresearch/reap_swap/dynamic_reap.py` — Validation, plan management, scoring (1811 lines)
4. `/Users/sero/ai/autoresearch/reap_swap/multiplex_cache.py` — LRU cartridge cache (30 lines)
