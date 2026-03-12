# Multi-Turn Evaluation Protocol (v0)

## Purpose
Define a minimal, repeatable multi-turn evaluation path that stress-tests conversational state retention and reasoning continuity for dynamic expert selection, while remaining comparable to existing single-turn smoke runs.

## Scope
- **In scope:** 3-turn chains for MCQ benchmarks and GSM8K, scoring schema, aggregation/report fields, matched-baseline rules.
- **Out of scope:** model/runtime code changes, new judge model integration, long-horizon (>3 turn) dialogs.

## Dataset + Sampling Contract
Use the same benchmark families currently used by smoke runs:
- `mmlu`
- `arc_challenge`
- `hellaswag`
- `winogrande`
- `gsm8k`

For each run:
- fixed `sample_count_per_benchmark`
- fixed `seed`
- deterministic ordering policy (documented in report)

For confirmation runs, vary seeds (e.g. pass-5 = 5 distinct seeds), not repeated runs on the same seed.

---

## 3-Turn Chain Templates

## A) MCQ Benchmarks (MMLU / ARC / HellaSwag / WinoGrande)

### Turn 1 (Answer)
**Prompt template:**
- Ask the original benchmark question.
- Instruct model to output only final option token (`A-E` or `1/2` for WinoGrande).

**Scored outputs:**
- `turn1_answer`
- `turn1_correct`
- `turn1_parse_error`

### Turn 2 (Reason)
**Prompt template:**
- “Briefly explain why your previous answer is correct. Keep it concise.”

**Scored outputs:**
- `turn2_coherent`
- `turn2_error`
- optional `turn2_reason_quality` (heuristic flag only in v0)

### Turn 3 (Recommit)
**Prompt template:**
- “Given your explanation, restate only the final answer token.”

**Scored outputs:**
- `turn3_answer`
- `turn3_correct`
- `turn3_parse_error`
- `answer_retention` = (`turn3_answer == turn1_answer`)

---

## B) GSM8K

### Turn 1 (Solve)
**Prompt template:**
- Solve normally, keep response concise, enforce final numeric form (`Final answer: <number>`).

**Scored outputs:**
- `turn1_answer`
- `turn1_correct`
- `turn1_parse_error`

### Turn 2 (Verify)
**Prompt template:**
- “Verify your result using a different method, concise.”

**Scored outputs:**
- `turn2_coherent`
- `turn2_error`

### Turn 3 (Recommit)
**Prompt template:**
- “Restate only the final numeric answer.”

**Scored outputs:**
- `turn3_answer`
- `turn3_correct`
- `turn3_parse_error`
- `answer_retention` = numeric equality between turn1 and turn3 parsed answers

---

## Scoring + Aggregation (Required)

## Per-sample fields
- identifiers: `sample_id`, `benchmark`, `seed`, `conversation_id`
- per-turn: answer, parsed answer, latency, parse error, request error
- derived:
  - `turn1_correct`
  - `turn3_correct`
  - `answer_retention`
  - `conversation_success` = (no request error on turns 1-3) AND `turn2_coherent` AND `turn3_correct`

## Per-benchmark aggregate (required)
- `turn1_accuracy`
- `turn3_accuracy`
- `accuracy_drop_turn1_to_turn3`
- `answer_retention_rate`
- `turn2_coherence_rate`
- `conversation_success_rate`
- `parse_error_rate_by_turn` (t1, t3 minimum)
- latency metrics by turn (`p50`, `p95`, `avg`)

## Overall aggregate (required)
- weighted and macro averages across benchmarks for all metrics above
- worst-benchmark degradation fields:
  - `worst_turn3_accuracy_drop_abs`
  - `worst_conversation_success_drop_abs` (vs baseline)

---

## Matched-Baseline Requirement (Non-negotiable)

Retained metrics are valid **only** when baseline and candidate are matched on:
- same benchmark samples
- same seed
- same sample_count_per_benchmark
- same protocol version/templates
- same turn structure

Disallow retained-metric reporting when any match key differs.

## Required report metadata
Each run report must include:
- `protocol_version` (e.g. `multiturn_v0`)
- `seed`
- `sample_count_per_benchmark`
- benchmark list
- chain templates hash/id
- baseline artifact path + baseline match status

If unmatched, set:
- `retained_metrics_status = "invalid_unmatched_baseline"`
- include explicit mismatch reasons.

---

## Pass-5 Recommendation (Signal Strength)
For stronger coherence-by-category signal:
- run 5 distinct seeds (e.g. `7,17,27,37,47`)
- compute per-seed aggregates, then report mean + variance/confidence band
- always pair each seed with its own matched baseline

---

## Minimal Deliverables Per Run
1. Raw JSON artifact with per-turn/per-sample fields
2. Markdown summary with per-benchmark + overall tables
3. Baseline-match validity block
4. Pass/fail gate block for multi-turn-specific thresholds

