# 08 -- strict30-v2 Plan Specification

## Summary of Changes from v1

| Parameter | v1 (Current) | v2 (Proposed) | Rationale |
|---|---|---|---|
| `core_budget_fraction` | 0.75 | 0.50 | More specialist headroom: 4 slices vs 2 |
| `specialist_budget_fraction` | 0.25 | 0.50 | Balanced core/specialist distribution |
| `rotationPolicy` | `"none"` | `"late_prompt_hash"` | Minimum viable fix for signature diversity |
| Source summaries | 1x "personal" | 3x: math, code, general | Domain-specific calibration (per REAP paper) |
| `promptClusterPriors` | Empty `{}` | Per-domain-tag orderings | Enable tag-conditioned slice ranking |
| `taskFamilySlicePriors` keys | `[global, personal]` | `[global, math, code, general]` | Align with domain tag namespace |
| `supportEstimatorConfig.mode` | `"none"` (default) | `"prefill_reserve"` | Enable support scoring pathway |
| `reserve_fraction` | 0.0 | 0.30 | Reserve 30% of specialist budget for support-ranked slices |
| `activation_records` | None | From benchmark runs | Populate domain_tag_counts for cluster priors |

## Budget Arithmetic (50/50 Split)

```
full_bf16_gib              = 63.42
max_resident_ratio         = 0.30
max_resident_gib           = 19.03
always_resident_gib        = 3.42
swappable_budget_gib       = 15.61  (unchanged)
per_expert_bytes           = 6,291,456

core_budget_fraction       = 0.50
core_budget_bytes          = 8,378,940,006
specialist_budget_bytes    = 8,378,940,006

core_experts_per_layer     = 33
specialist_experts_per_layer = 33
candidate_experts_per_layer  = 42  (with multiplier 1.25)

Total active per layer     = 66  (~25.8% of 256)
Per-layer specialist budget = 209,473,500 bytes
Slices fitting per layer   = 4.16 (4 full slices of 8)
```

## Rotation Policy Configuration

```json
{
  "rotationPolicy": "late_prompt_hash"
}
```

**Mechanics**:
- Applies to layers >= `total_layers // 2` = layers 20-39 (20 late layers)
- `rotation_window = min(max(2, candidate_target), 4) = 4`
- Offset formula: `(semantic_offset + sha1(layer|prompt)) % 4`
- Top-4 slices are circularly permuted by the offset
- With 4 slices fitting in budget, rotation can change which slices are selected

**Expected signature diversity**:
- Different benchmarks have different `TAG_ROTATION_WEIGHTS` sums
- Different prompts have different `sha1(prompt_fingerprint)` hashes
- On 20 late layers with window=4, theoretical max = 4^20 ~ 10^12 signatures
- Practical: 5+ distinct signatures across 5 benchmarks

## Source Summary Requirements

### Why Multi-Domain Summaries

The REAP paper proves domain-specific calibration is mandatory at high compression:
- C4 calibration at 50% compression: REAP Eval+ drops from 0.780 to 0.329
- Domain-specific calibration preserves quality even at 50% compression
- At ~75% compression (strict30), domain mismatch would be catastrophic

### Proposed Source Summaries

| Label | Dataset | Samples | Purpose |
|---|---|---|---|
| `math` | tulu-3-sft-personas-math or NuminaMath | 1024 | Math reasoning expert selection |
| `code` | evol-codealpaca-v1 | 1024 | Code generation expert selection |
| `general` | WildChat or C4 (curated) | 1024 | General/conversational coverage |

Each summary must contain per-layer per-expert arrays for:
- `reap` (gate-weighted activation norm average)
- `weighted_ean_sum` (gate-weighted EAN)
- `expert_frequency` (routing frequency)

### How to Collect

Run the existing observer on each dataset independently, producing 3 summary JSON files. Then pass all 3 to `build_dynamic_plan()`:

```python
summaries = [
    ("math", math_summary),
    ("code", code_summary),
    ("general", general_summary),
]
plan = build_dynamic_plan(summaries, signal_key="reap", ...)
```

## Code Change: `build_dynamic_plan` Cluster Priors Fix

### Current Code (dynamic_reap.py ~line 540)

```python
cluster_priors: dict[str, dict[str, list[str]]] = {}
domain_tag_counts = activation_summary.get("domain_tag_counts", {})
for tag in sorted(domain_tag_counts.keys()):
    preferred_label = None
    if tag == "code" and "coding" in labels:
        preferred_label = "coding"
    elif tag in {"writing", "research"} and "communication" in labels:
        preferred_label = "communication"
    cluster_priors[tag] = task_family_priors.get(preferred_label or "global", {})
```

### Problem

1. `activation_records` is empty, so `domain_tag_counts` is `{}`, loop never runs
2. Even if it ran, the label matching is too narrow ("coding", "communication")
3. All tags fall through to `"global"` -- identical orderings

### Proposed Fix

```python
# Always generate cluster priors for all known domain tags
CLUSTER_LABEL_MAP = {
    "code": "code",
    "math": "math",
    "writing": "general",
    "research": "math",      # research prompts use math-like expert patterns
    "ops": "code",            # ops prompts use code-like expert patterns
    "general": "general",
}

cluster_priors: dict[str, dict[str, list[str]]] = {}
for tag in sorted(DOMAIN_PATTERNS.keys()) + ["general"]:
    preferred_label = CLUSTER_LABEL_MAP.get(tag)
    if preferred_label and preferred_label in labels:
        cluster_priors[tag] = task_family_priors.get(preferred_label, {})
    else:
        cluster_priors[tag] = task_family_priors.get("global", {})
```

This ensures:
- Cluster priors are always populated for all domain tags
- Each tag maps to the most relevant source summary label
- With 3 source summaries (math, code, general), different tags get different orderings
- No dependency on `activation_records` for basic functionality

## Code Change: TaskFamily Priors Key Alignment

### Current Behavior

`taskFamilySlicePriors` keys are source summary labels: `["global", "math", "code", "general"]`

Domain tags from `infer_domain_tags()` are: `[code, math, writing, research, ops, general]`

With the proposed 3 source summaries, the intersection is: `{math, code, general}`.
`writing`, `research`, and `ops` would still miss.

### Proposed Fix

Add synthetic label entries that alias domain tags to source summary orderings:

```python
# After building task_family_priors from source labels:
TAG_TO_LABEL_ALIAS = {
    "writing": "general",
    "research": "math",
    "ops": "code",
}
for tag, label in TAG_TO_LABEL_ALIAS.items():
    if label in task_family_priors and tag not in task_family_priors:
        task_family_priors[tag] = task_family_priors[label]
```

## Support Mode Configuration

```json
{
  "scorerArtifacts": {
    "supportEstimatorConfig": {
      "mode": "prefill_reserve",
      "reserve_fraction": 0.30,
      "late_layer_start_frac": 0.0,
      "benchmark_scale": 20.0,
      "tag_scale": 8.0,
      "lexical_scale": 10.0,
      "lexical_k": 5
    }
  }
}
```

**Note**: `prefill_reserve` mode is only useful when `benchmarkMissPriors`, `tagMissPriors`, or `lexicalMissExemplars` are populated. Without these, it falls back to base_scores. This is a forward-looking configuration for when miss prior data is collected.

## Expected Outcomes

### Signature Diversity (Local Validation)

With `late_prompt_hash` rotation and 50/50 budget:
- **Minimum**: 2+ distinct signatures across 25 prompts (math vs non-math)
- **Expected**: 5-10 distinct signatures (per-benchmark differentiation via rotation weights)
- **Maximum**: 25 distinct signatures (one per prompt, if lexical hashes differ enough)

### Accuracy (Requires GPU Run)

Based on REAP paper extrapolation at ~75% compression:
- **MC Avg**: 0.35-0.42 (paper predicts this range; domain calibration may help)
- **Code Avg**: 0.43-0.50 (coding is more resilient to REAP compression)
- **Math Avg**: 0.75-0.82 (math is most resilient)
- **With multi-domain calibration**: possible 5-10% improvement across benchmarks
- **Ceiling**: REAP 50% compression = 0.503 MC. strict30 at 75% is below this.

### Latency (Requires GPU Run)

- Warm swap should remain ~0.06s (signature changes don't increase copy volume much)
- Cold start remains ~60s (unchanged, one-time)
- Per-request overhead: unchanged (scoring pipeline is fast, ~1ms)
