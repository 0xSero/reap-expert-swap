# 02 -- Root Cause Analysis

## Diagram

See `diagrams/signature-failure-chain.mermaid` for the causal chain.

## Part A: Why Only 1 Signature (7 Compounding Failures)

### Failure 1: `rotationPolicy: "none"` [PROVED]

- **Location**: `strict30-best-plan.json` top-level field
- **Mechanism**: The `late_prompt_hash` policy circularly permutes the top-N slice window using `sha1(layer|prompt_fingerprint)` (dynamic_reap.py:1492-1510). With `"none"`, this block never executes.
- **Proof**: Direct from plan JSON: `"rotationPolicy": "none"`
- **Impact**: The ONLY mechanism that uses raw prompt content for differentiation (in the absence of miss priors) is disabled.
- **Fix category**: Plan JSON edit (1 field)

### Failure 2: `supportEstimatorConfig.mode: "none"` [PROVED]

- **Location**: `scorerArtifacts.supportEstimatorConfig` absent from plan
- **Mechanism**: `_support_estimator_config()` (dynamic_reap.py:1098-1112) defaults to `{"mode": "none"}`. With mode=none, line ~1552 selects `base_scores` not `support_scores`, bypassing all support boosts.
- **Proof**: Direct from plan JSON: `scorerArtifacts` has no `supportEstimatorConfig` key.
- **Impact**: benchmark_prior, tag_prior, and lexical_similarity boosts never contribute to score.
- **Fix category**: Plan JSON edit + data collection

### Failure 3: Empty miss prior data [PROVED]

- **Location**: `scorerArtifacts.benchmarkMissPriors`, `tagMissPriors`, `lexicalMissExemplars`
- **Mechanism**: These keys don't exist in the plan. Even if support mode were enabled, the scoring loop would find no data.
- **Proof**: Direct from plan JSON: these keys are absent. `build_dynamic_plan()` never populates them.
- **Impact**: Even with support mode enabled, zero contribution from the three most powerful prompt-differentiation signals.
- **Fix category**: Data collection (requires router miss observation runs on benchmark prompts)

### Failure 4: `promptClusterPriors` all identical [PROVED]

- **Location**: `scorerArtifacts.promptClusterPriors`
- **Mechanism**: In `build_dynamic_plan()` (~line 540), cluster_priors are built from `activation_summary["domain_tag_counts"]`. Since `activation_records` was empty when the plan was built, `domain_tag_counts` is `{}`, and the loop executes zero iterations. Even if populated, the only source label "personal" matches neither "coding" nor "communication" (the two hard-coded domain mappings), so every tag falls through to `"global"`.
- **Proof**: Diagnostic output confirmed: "ALL cluster_priors orderings are IDENTICAL to global for every tag and every layer."
- **Impact**: Tag-conditioned boost applies the same ordering for all tags, so different prompts with different tags still get identical slice rankings.
- **Fix category**: Code change in `build_dynamic_plan()` + activation record collection

### Failure 5: `taskFamilySlicePriors` key mismatch [PROVED]

- **Location**: `scorerArtifacts.taskFamilySlicePriors`
- **Mechanism**: Keys are `["global", "personal"]` (from source summary labels). Domain tags from `infer_domain_tags()` are `[code, math, writing, research, ops, general]`. The scoring loop does `priors.get(tag, {})` for each tag -- every lookup returns `{}`.
- **Proof**: Diagnostic output confirmed: "taskFamilySlicePriors keys: ['global', 'personal'] / Domain tags: code, math, ... / Intersection: NONE"
- **Impact**: Zero contribution from the taskFamily prior path for every prompt.
- **Fix category**: Code change to align keys OR multi-domain source summaries with matching labels

### Failure 6: Monotone boost formula [PROVED]

- **Location**: Scoring inner loop, dynamic_reap.py:~1467-1476
- **Mechanism**: Cluster_priors boost adds `AM * 0.15` per matching tag. Since every slice appears in the prior list, every slice gets boosted. The boost is proportional to each slice's own `activationMass`. A higher-mass slice always gets a higher absolute boost. Net formula: `score = AM * (1 + 0.15 * N_tags)` -- a scalar multiple that preserves relative ranking.
- **Proof**: Diagnostic computed scores for MMLU vs GSM8K on layer_0. MMLU top-2: `[slice_00, slice_01]`, GSM top-2: `[slice_00, slice_01]`. Both follow strict activationMass ordering despite different tag counts and different absolute scores.
- **Impact**: Even when tags differ between prompts, the ranking cannot change.
- **Fix category**: Design fix -- priors need to produce different orderings, not same ordering with proportional boosts

### Failure 7: Budget = exactly 2 slices/layer [CONTEXTUAL]

- **Location**: `plan.budget.specialist_budget_bytes` / `perLayer.*.specialistBudgetBytes`
- **Mechanism**: `specialistBudgetBytes = 100,663,296` per layer. Each 8-expert slice = 50,331,648 bytes. Exactly 2 fit: `100,663,296 / 50,331,648 = 2.0`. With zero headroom, even a partial reorder at position 3 would have no effect.
- **Proof**: Budget arithmetic confirmed by diagnostic.
- **Impact**: Amplifies the effect of failures 1-6. Even if a reorder occurred at position 3, it wouldn't change the selected set.
- **Fix category**: Budget split change (75/25 -> 50/50 gives ~4 slices)

### Combined Effect

For every prompt on every benchmark:
1. `base_score` ranking = static `activationMass` ranking (failures 2-6)
2. No rotation applied (failure 1)
3. Top-2 slices are always `slice_00` and `slice_01` (failure 7)
4. 65 experts per layer on all 40 layers = one signature forever

---

## Part B: Why 0.40 Accuracy (5 Contributing Factors)

### Factor 1: Aggressive compression (~75% expert removal) [INFERRED from paper]

- 65/256 active = 25.4% retention = ~74.6% compression
- REAP paper's closest data point: Qwen3-30B at 50% compression = 0.503 MC avg
- strict30's ~75% compression is well beyond tested range
- Paper shows non-linear degradation that accelerates at higher compression

### Factor 2: Single-source calibration mismatch [INFERRED from paper + plan]

- Plan built from "personal" chat workload (2048 samples)
- Evaluated on academic benchmarks (MMLU, ARC, HellaSwag, Winogrande, GSM8K)
- REAP paper: C4 calibration at 50% causes REAP Eval+ to drop from 0.780 to 0.329
- Personal-chat calibration for MCQ is the same category of domain mismatch

### Factor 3: Inverted budget split (75/25 vs default 35/65) [PROVED]

- Default: 35% core / 65% specialist = 23 core + 43 specialist
- Current: 75% core / 25% specialist = 49 core + 16 specialist
- Both have ~65 total active, but current plan has almost no dynamic headroom
- Only 2 variable slices per layer vs ~5 with default split

### Factor 4: No router miss feedback loop [PROVED from code]

- `evaluate_samples()` fetches `router_misses` AFTER completion (for logging)
- Never triggers `build_active_set_payload()` with `router_misses` parameter
- The `decode_refresh` path that could add missing experts mid-request is unused
- `max_refreshes_per_request = 1` but never consumed

### Factor 5: Information loss from zeroing [STRUCTURAL]

- Inactive experts are zeroed, not computed on CPU
- Router mask prevents selection, but softmax distribution is distorted
- Remaining active experts receive inflated weights vs full model
- Unlike true offloading, tokens that would need inactive experts get zero contribution
