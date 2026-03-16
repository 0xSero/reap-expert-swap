# 01 -- System Architecture

## Diagram (Mermaid)

See `diagrams/system-architecture.mermaid` for the full flowchart.

## Diagram (ASCII)

```
                         OFFLINE (Plan Generation)
 ┌─────────────────────────────────────────────────────────────────┐
 │                                                                 │
 │  Qwen3.5-35B-A3B ──► Observer ──► Summary JSON ──► Plan Builder │
 │  256 exp × 40 layers   2048 samples   reap/ean/freq   build_dynamic_plan()
 │                        "personal"     per expert       ──► strict30-best-plan.json
 │                                       per layer            core:49 + spec:16/layer
 │                                                            26 slices of 8 experts
 └─────────────────────────────────────────────────────────────────┘
                                │
                    plan.json ──┘
                                │
                         RUNTIME (Per-Request)
 ┌──────────────────────────────┼──────────────────────────────────┐
 │                              ▼                                  │
 │  Prompt ──► infer_domain_tags() ──► rank_slice_ids_for_prompt() │
 │             keyword matching         score all 26 slices/layer  │
 │             → [code, math, ...]      base_score + support_score │
 │                                             │                   │
 │                                             ▼                   │
 │                                  assemble_active_set()          │
 │                                  greedy top-2 slices within     │
 │                                  per-layer specialist budget    │
 │                                             │                   │
 │                                             ▼                   │
 │                                  Active Set Payload             │
 │                                  65 experts/layer + sig hash    │
 └─────────────────────────────────────┼───────────────────────────┘
                                       │
                         vLLM MULTIPLEX SERVER (Remote GPU)
 ┌─────────────────────────────────────┼───────────────────────────┐
 │                                     ▼                           │
 │  POST /swap_active_set ──► compute_keep_set_delta()             │
 │                             │                                   │
 │                             ├──► zero removed experts on GPU    │
 │                             ├──► copy added experts from CPU    │
 │                             └──► inject router masks (-inf)     │
 │                                     │                           │
 │                                     ▼                           │
 │  POST /v1/completions ──► vLLM inference (top-8 within active)  │
 │                                     │                           │
 │  GET /router_misses/{id} ──► inactive mass tracking             │
 │                                     │                           │
 │  (decode_refresh path) ◄────────────┘ (not currently wired)     │
 └─────────────────────────────────────────────────────────────────┘
                                       │
                         EVALUATION
 ┌─────────────────────────────────────┼───────────────────────────┐
 │  5 benchmarks × 5 samples = 25 prompts (seed=7)                │
 │  Three arms: A(prewarmed) B(forced-static) C(cold-relaunch)    │
 │  parse_prediction() + coherence_pass()                          │
 │  signature tracking + swap phase classification                 │
 │  ──► summary.json with accuracy, dynamics, latency              │
 └─────────────────────────────────────────────────────────────────┘
```

## Component Inventory

| # | Component | File | Purpose | Current State |
|---|---|---|---|---|
| 1 | Model | Qwen3.5-35B-A3B (remote) | 256 experts/layer, top-8, 40 MoE layers | Working |
| 2 | Observer | (external, not in repo) | Collects reap/ean/freq per expert | Ran once on "personal" data |
| 3 | Summary JSON | (not in repo, consumed by plan builder) | Per-layer per-expert signal arrays | Single source: "personal", 2048 samples |
| 4 | Plan Builder | `reap_swap/dynamic_reap.py:416` `build_dynamic_plan()` | Generates plan JSON from summaries | Works but produces degenerate priors |
| 5 | Plan JSON | `assets/strict30-best-plan.json` | Core experts + specialist slices + budget + priors | 75/25 split, no rotation, empty priors |
| 6 | Tag Inference | `reap_swap/dynamic_reap.py:133` `infer_domain_tags()` | Classify prompt domain by keywords | Works correctly |
| 7 | Slice Ranking | `reap_swap/dynamic_reap.py:1359` `rank_slice_ids_for_prompt()` | Score and rank slices per prompt | Produces identical rankings for all prompts |
| 8 | Active Set Assembly | `reap_swap/dynamic_reap.py:1530` `assemble_active_set()` | Pick top slices within budget | Works but always picks same slices |
| 9 | vLLM Server | `reap_swap/vllm_multiplex_server.py` | Custom swap/mask/miss-tracking endpoints | Working (proven by feasible envelope) |
| 10 | Delta Computation | `reap_swap/dynamic_swap_delta.py` | Diff current vs desired active set | Working |
| 11 | Evaluation Harness | `reap_swap/evaluate_original_vs_multiplex.py` | Run benchmarks, collect metrics | Working but no refresh loop |
| 12 | E2E Runner | `tools/strict30_pair01_e2e.py` | SSH to remote, launch, run 3 arms | Working |
| 13 | Size Estimator | `reap_swap/size_estimator.py` | Model size arithmetic | Working |
| 14 | Research Gate | `reap_swap/research_gate.py` | Quality gate evaluation | Working but baselines mismatched |
| 15 | BF16 Baseline | `assets/bf16-baseline-seed7-s10.json` | Full-model reference results | Matched seed, 10 samples/benchmark |
