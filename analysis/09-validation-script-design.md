# 09 -- Local Validation Script Design

## Purpose

`tools/strict30_v2_local_validate.py` -- a script that validates a plan JSON produces real signature diversity across the benchmark prompt set, without needing GPUs or a remote server.

## Interface

```bash
cd /Users/sero/ai/autoresearch
uv run --with datasets tools/strict30_v2_local_validate.py \
    --plan-json assets/strict30-v2-plan.json \
    --sample-count 5 \
    --seed 7 \
    --output-json test-output/v2-local-validation.json
```

## What It Does

1. **Load plan** from `--plan-json`
2. **Load benchmarks** using `load_benchmarks(sample_count=N, calibration_count=0, seed=S)`
3. **For each prompt** (N * 5 benchmarks):
   a. Call `build_active_set_payload(plan, prompt, benchmark=benchmark)`
   b. Record: `signature`, `selected_slice_ids`, `tags`, `benchmark`, `prompt_id`
4. **Compute metrics**:
   - `unique_signature_count`
   - `per_benchmark_signature_groups` (which benchmarks share signatures)
   - `per_layer_slice_variation` (how many distinct slice selections per layer)
   - `per_layer_top_slice_frequency` (how often each slice_id is selected)
   - `rotation_applied_count` (how many layers had rotation applied)
   - `tag_distribution` (count of each domain tag across prompts)
5. **Validate acceptance criteria**
6. **Write output JSON**

## Acceptance Criteria

| Criterion | Threshold | Rationale |
|---|---|---|
| `unique_signature_count` | > 1 | Minimum: system is not static |
| `per_benchmark_distinct_signatures` | >= 2 groups | At least 2 benchmarks produce different signatures |
| `layers_with_slice_variation` | > 0 | At least one layer selects different slices for different prompts |
| Budget compliance | All payloads pass `validate_active_set_payload` | No budget violations |
| Core presence | All core experts present in every active set | No missing cores |

## Output Format

```json
{
  "plan_path": "assets/strict30-v2-plan.json",
  "plan_sha256": "...",
  "sample_count": 5,
  "seed": 7,
  "total_prompts": 25,
  "unique_signature_count": 7,
  "signatures": {
    "abc123...": {
      "count": 5,
      "benchmarks": ["mmlu"],
      "prompt_ids": ["mmlu::astronomy::abc...", ...]
    },
    ...
  },
  "per_benchmark_summary": {
    "mmlu": {"unique_signatures": 1, "tags": ["research", "writing"]},
    "gsm8k": {"unique_signatures": 3, "tags": ["math", "research"]},
    ...
  },
  "per_layer_slice_variation": {
    "layer_0": {"distinct_selections": 1, "most_common": ["slice_00", "slice_01"]},
    "layer_20": {"distinct_selections": 3, "most_common": ["slice_00", "slice_02"]},
    ...
  },
  "rotation_stats": {
    "layers_with_rotation": 20,
    "layers_without_rotation": 20,
    "distinct_offsets_per_layer_avg": 3.2
  },
  "acceptance": {
    "signature_diversity": {"pass": true, "value": 7, "threshold": ">1"},
    "benchmark_groups": {"pass": true, "value": 4, "threshold": ">=2"},
    "slice_variation": {"pass": true, "value": 18, "threshold": ">0"},
    "budget_compliance": {"pass": true, "violations": 0},
    "core_presence": {"pass": true, "missing": 0}
  },
  "verdict": "PASS: strict30-v2 produces real signature diversity"
}
```

## Implementation Notes

### Dependencies

- `datasets` (HuggingFace, for benchmark loading)
- All other imports are from vendored `reap_swap/` package

### Key Functions Used

```python
from reap_swap.dynamic_reap import build_active_set_payload, compute_active_set_signature
from reap_swap.evaluate_original_vs_multiplex import (
    load_benchmarks, format_prompt, BenchmarkSpec, stable_id, BENCHMARK_SPECS
)
```

### winogrande Handling

The e2e runner's spec_map uses `winogrande_xl` but BENCHMARK_SPECS uses `winogrande_m`. The validation script should use the BENCHMARK_SPECS version for consistency with the evaluation harness. Winogrande uses `question_style="binary"` not `"mcq_numeric"` -- the e2e runner's `make_first_active_set` has a bug here.

### No Network Required

- Plan JSON loaded locally
- Benchmark datasets cached by HuggingFace after first download
- No server connection needed
- No SSH, no GPU, no tunnel

### Execution Time

- 25 prompts * `build_active_set_payload` (pure Python, ~10ms each) = ~250ms
- Benchmark loading from HF cache: ~5s
- Total: under 10 seconds

## Comparison Mode

The script should also accept `--compare-plan` to compare two plans side by side:

```bash
uv run --with datasets tools/strict30_v2_local_validate.py \
    --plan-json assets/strict30-v2-plan.json \
    --compare-plan assets/strict30-best-plan.json \
    --sample-count 5 --seed 7
```

This would show:
- v1 signature count vs v2 signature count
- Which prompts changed signatures
- Per-layer delta in slice selections
- Budget utilization comparison
