# 03 -- Scoring Pipeline Mechanical Trace

## Entry Point

```
build_active_set_payload(plan, prompt_text, request_id, benchmark, phase)
  → rank_slice_ids_for_prompt(plan, prompt_text, benchmark=benchmark)
  → assemble_active_set(plan, ranked, support_ranked, candidate_ranked)
  → validate_active_set_payload(payload, plan)
```

## Step 1: Input Processing (dynamic_reap.py:1359-1396)

```
routing_text = prompt_text                    # if no conversation context
             = f"{context}\n\nlatest_user_prompt: {prompt_text}"  # if context exists

tags = infer_domain_tags(routing_text)        # keyword match against DOMAIN_PATTERNS
     + BENCHMARK_TO_TAGS[benchmark]           # e.g. gsm8k → [math, research]
     deduplicated

prompt_fingerprint = normalize(routing_text).lower()
prompt_tokens = tokenize(routing_text)        # set of lowercase words, stopwords removed
```

### Tag Mapping

| Benchmark | Tags from benchmark | Typical prompt tags | Combined |
|---|---|---|---|
| mmlu | research, writing | varies by subject | research, writing + subject keywords |
| arc_challenge | research | research | research |
| hellaswag | writing | writing | writing |
| winogrande | writing | writing | writing |
| gsm8k | math, research | math | math, research |

## Step 2: Data Extraction from Plan (dynamic_reap.py:1388-1400)

```
priors         = plan.scorerArtifacts.taskFamilySlicePriors   # {"global": {...}, "personal": {...}}
cluster_priors = plan.scorerArtifacts.promptClusterPriors     # {} (empty in current plan)
benchmark_miss = plan.scorerArtifacts.benchmarkMissPriors     # {} (absent)
tag_miss       = plan.scorerArtifacts.tagMissPriors           # {} (absent)
lexical_miss   = plan.scorerArtifacts.lexicalMissExemplars    # [] (absent)
support_config = {mode: "none", ...}                          # defaults
rotation_policy = "none"
```

## Step 3: Per-Layer Scoring Loop (dynamic_reap.py:1404-1523)

For each of 40 layers, for each of 26 slices:

### Formula

```
base_score(slice) =
    activationMass                                               (A) STATIC
  + Σ_{tag} AM × 0.15    if slice ∈ cluster_priors[tag][layer]  (B) TAG BOOST
  + Σ_{tag} taskPriors[tag] × 0.25  if slice ∈ priors[tag][layer] (C) LABEL BOOST
  + 1000 × |slice.experts ∩ miss_experts|                        (D) EMERGENCY

support_score(slice) = base_score
  + bench_prior_mass × layer_activation_scale × 20               (E) BENCH MISS
  + tag_prior_mass × layer_activation_scale × 8                  (F) TAG MISS
  + lexical_prior_mass × layer_activation_scale × lexical_scale  (G) LEXICAL
```

### What-If Analysis

| Component | Data Needed | Present? | Effect When Present | Effect When Absent |
|---|---|---|---|---|
| **(A) activationMass** | Plan slice catalog | YES | Base ranking (static) | N/A |
| **(B) cluster_priors** | `promptClusterPriors[tag][layer]` | NO (empty) | +AM*0.15 per tag, but monotone | Zero boost |
| **(C) label_priors** | `taskFamilySlicePriors[tag][layer]` | MISMATCH (keys don't match tags) | +taskPriors*0.25, correlated with AM | Zero boost |
| **(D) emergency** | `router_misses.by_layer[layer].inactive_experts` | Only at refresh | +1000 per missed expert, strong reorder | No boost |
| **(E) bench_miss** | `benchmarkMissPriors[bench][layer]` + mode != none | NO + NO | Per-benchmark expert weights, strong differentiation | Zero boost |
| **(F) tag_miss** | `tagMissPriors[tag][layer]` + mode != none | NO + NO | Per-tag expert weights, moderate differentiation | Zero boost |
| **(G) lexical** | `lexicalMissExemplars[]` + mode != none | NO + NO | Jaccard similarity to known miss patterns, strongest per-prompt signal | Zero boost |

### Active Status in Current Plan

```
(A) activationMass    → ACTIVE, provides base ranking (prompt-invariant)
(B) cluster_priors    → INACTIVE (empty dict)
(C) label_priors      → INACTIVE (key mismatch)
(D) emergency         → INACTIVE (no router_misses at prefill)
(E) bench_miss        → INACTIVE (no data + mode=none)
(F) tag_miss          → INACTIVE (no data + mode=none)
(G) lexical           → INACTIVE (no data + mode=none)
```

Net effect: `score = activationMass` (constant ranking for all prompts).

## Step 4: Rotation (dynamic_reap.py:1492-1510)

```python
apply_rotation = (rotation_policy == "late_prompt_hash"     # FALSE in current plan
                  and rotation_window > 1
                  and layer_idx >= total_layers // 2)

# If applied:
semantic_offset = Σ TAG_ROTATION_WEIGHTS[tag]               # {general:1, code:2, ops:3, math:5, research:7, writing:11}
                + sha1(benchmark)[:12] % window
                + sha1(conversation_id)[:12] % window
                + turn_index % window

lexical_offset  = sha1(f"{layer_key}|{prompt_fingerprint}")[:12] % window

rotation_offset = (semantic_offset + lexical_offset) % window

# Permutation: circular rotate top-N
scores = scores[offset:window] + scores[:offset] + scores[window:]
```

**Current state**: Disabled. `rotation_policy == "none"` so this block never executes.

## Step 5: Mode Selection (dynamic_reap.py:1552-1554)

```python
mode = support_config["mode"]   # "none"

selected_scores  = base_scores   if mode != "full_prefill"       else support_scores
candidate_scores = base_scores   if mode not in {full_prefill, candidate_only, prefill_reserve} else support_scores
```

| Mode | Selected (for active) | Candidate (for pool) |
|---|---|---|
| **none (current)** | **base_scores** | **base_scores** |
| full_prefill | support_scores | support_scores |
| candidate_only | base_scores | support_scores |
| prefill_reserve | base_scores | support_scores |

## Step 6: Assembly (dynamic_reap.py:1530-1641)

```
For each layer:
  active = set(coreExperts)                    # 49 experts
  for slice_id in selected_order:
    if cumulative_bytes + slice.byteCost <= layer_specialist_budget:
      active |= slice.experts
      cumulative_bytes += slice.byteCost       # stops after 2 slices (budget=2 slices exactly)

  active_set[layer] = sorted(active)           # 49 + 16 = 65 experts
```

## Step 7: Signature (dynamic_reap.py:1706-1718)

```python
canonical = {"active_set": {layer: sorted_experts}, "selected_slice_ids": {layer: sorted_ids}}
signature = sha1(json.dumps(canonical, sort_keys=True))[:16]
```

Same experts on all layers = same canonical form = same hash = `6ebe4366c72f4528` for every prompt.
