# REAP-swap

Use REAP observation data to make smarter CPU offloading decisions for MoE models in vLLM.

## The problem

Large MoE models (100B+ parameters) don't fit in consumer GPU VRAM. vLLM's UVA backend offloads expert parameters to CPU memory and pulls them in on demand, but it picks which experts to keep GPU-resident by default ordering (typically the tail layers of the model). This works, but it's leaving performance on the table -- the default resident set has no relationship to which experts your workload actually needs.

## The idea

REAP (arXiv:2510.13999) was designed to prune MoE experts permanently. Its observation phase captures per-expert activation mass from calibration prompts -- basically a heatmap of which experts the model routes to most for your workload.

REAP-swap repurposes that heatmap. Instead of pruning, it tells vLLM which experts to pre-load into GPU memory. No experts are removed. The full model remains accessible through UVA. You just get fewer expensive CPU-to-GPU transfers during inference because the hot experts are already resident.

## Results

Tested on Qwen3.5-35B-A3B, 8x RTX 3090, 512GB DDR4, EPYC 7443P.

**Speed** (stock UVA vs REAP-swap, same 16-layer resident budget):

| Metric | Stock UVA | REAP-swap | Change |
|--------|-----------|-----------|--------|
| TTFT | 2.85s | 1.59s | **-44%** |
| Prefill | 116.29 tok/s | 133.18 tok/s | **+14.5%** |
| Generation | 14.13 tok/s | 13.94 tok/s | -1.3% |

Stock resident layers: `24-39` (vLLM default, tail of model).
REAP-swap resident layers: `1,2,3,4,5,6,24,25,26,27,28,29,30,31,38,39` (early layers that handle most activation mass for coding workloads, plus stable tail layers).

**Quality** (dynamic active-set swaps, 30% resident budget):

88% overall accuracy, 100% coherence across ARC Challenge, GSM8K, HellaSwag, MMLU, WinoGrande. Zero router misses. Zero bytes copied at swap time. See `example/arm2_dynamic_results.md` for full breakdown.

This is slower than full-VRAM serving. The claim is not free performance -- it's better performance than naive offloading when the model doesn't fit in memory.

## How it works

1. **Observe**: Run REAP's observation phase over a calibration corpus to capture per-expert activation mass across all MoE layers.
2. **Plan**: A planner reads the observations and outputs a JSON plan file specifying core experts (always resident) and specialist slices (swappable groups of co-activated experts) per layer. See `reap_plan.schema.md`.
3. **Serve**: Start vLLM with the multiplex server module. It loads the plan, monkey-patches vLLM's app builder, and adds endpoints for active-set swaps, router miss tracking, and forensic inspection.

The server supports two modes:
- **Static preload**: Pick a fixed resident set at startup based on the plan's core experts. This produced the speed numbers above.
- **Dynamic per-request**: Select specialist slices per request based on prompt characteristics. This produced the quality numbers above. In the working configuration (`REAP_SWAP_MASKS_ONLY=1`, `REAP_ENABLE_ROUTER_MASKS=0`), swaps update internal tracking only -- all experts stay accessible through UVA.

## File layout

```
production/
  README.md                          -- this file
  research.md                        -- full research writeup: hypothesis, experiment, results, failures
  reap_plan.schema.md                -- plan file format specification

  reap_swap/
    __init__.py
    vllm_multiplex_server.py         -- the runtime server (1111 lines)
    dynamic_swap_delta.py            -- delta computation for expert set transitions
    dynamic_reap.py                  -- plan validation, hashing, router miss summarization
    multiplex_cache.py               -- LRU cache for loaded cartridges

  example/
    strict30-v2-plan.json            -- 30% budget plan for Qwen3.5-35B-A3B
    arm2_dynamic_results.md          -- benchmark results from quality evaluation
```

## Running

```bash
# Required environment
export REAP_PLAN_FILE=/path/to/strict30-v2-plan.json
export REAP_SWAP_MASKS_ONLY=1
export REAP_ENABLE_ROUTER_MASKS=0

# Start vLLM with the multiplex server module
python -m reap_swap.vllm_multiplex_server \
  --model /path/to/model \
  --cpu-offload-params experts \
  --tensor-parallel-size 8
```

The server exposes these additional endpoints on top of vLLM's standard OpenAI-compatible API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/swap_cartridge/{id}` | POST | Swap to a pre-loaded cartridge (legacy path) |
| `/swap_active_set` | POST | Dynamic per-request expert set swap |
| `/warm_active_set` | POST | Pre-warm an active set without committing |
| `/router_misses/{request_id}` | GET | Per-request router miss instrumentation |
| `/forensics/{request_id}` | GET | Full forensic payload for a swap request |

## Calibration data

The observation corpus was built from 2 years of personal AI coding sessions extracted via [ai-data-extraction](https://github.com/0xSero/ai-data-extraction). 2,048 samples, 187K tokens, 16K-token multiturn conversations from Claude Code, Cursor, Codex, Opencode, Windsurf, Trae, and others.

The plan is workload-specific. A different calibration corpus will produce a different plan with different resident experts. The speed and quality results here are for a coding-heavy personal workload.

## Related

- [REAP paper](https://arxiv.org/abs/2510.13999) -- the pruning method whose observation phase this builds on
- [Full research writeup](research.md) -- experiment history, failed paths, and detailed analysis
- [ai-data-extraction](https://github.com/0xSero/ai-data-extraction) -- the tool used to build the calibration corpus
- [Kimi-K2.5-PRISM-REAP-530B-A32B](https://huggingface.co/Ex0bit/Kimi-K2.5-PRISM-REAP-530B-A32B) -- 50% REAP-pruned Kimi-K2.5, the second model in this research
