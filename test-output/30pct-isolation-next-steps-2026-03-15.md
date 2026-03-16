# 30pct isolation next steps - 2026-03-15

This is the shortest path to stop lying to ourselves.

## Constraint

I cannot edit `/Users/sero/ai/reap-expert-swap` from the current sandbox. The runtime repo is read-only here.

So this file stages:

1. the exact code changes to make there,
2. the exact one-pair experiments to run after patching,
3. the interpretation rules.

---

## Priority 0: fix the measurements before any more runs

### File: `scripts/evaluate_original_vs_multiplex.py`

#### Change A: add explicit cold-vs-warm and no-op classification to every row

The script already records:

- `swap_time_s`
- `swap_internal_s`
- `swap_phase`
- `swap_signature_changed`
- `swap_reused_active_set`
- `swap_bytes_copied`
- `swap_bytes_zeroed`

That is not enough because the main summaries still hide first-shrink dominance.

Add these per-row fields:

- `is_cold_shrink`
- `is_same_signature`
- `is_zero_copy_swap`
- `swap_control_plane_s = swap_time_s - swap_internal_s`
- `completion_tokens_per_s = completion_tokens / request_latency_s`

Suggested row logic:

```python
is_same_signature = (
    previous_active_signature is not None
    and effective_active_signature is not None
    and effective_active_signature == previous_active_signature
)
is_zero_copy_swap = int(swap_worker_metrics.get("bytes_copied") or 0) == 0
is_cold_shrink = str(swap_phase or "").startswith("cold_")
swap_control_plane_s = max(0.0, swap_time_s - float(swap_payload.get("swap_time_s", 0.0)))
completion_tokens_per_s = (
    float(usage.get("completion_tokens") or 0.0) / request_latency_s
    if request_latency_s > 0
    else 0.0
)
```

#### Change B: add a better overall summary split

Add these summary metrics:

- `avg_cold_request_time_s`
- `avg_cold_sample_time_s`
- `avg_cold_swap_control_plane_s`
- `avg_warm_swap_control_plane_s`
- `avg_warm_change_swap_time_s`
- `avg_warm_reuse_swap_time_s`
- `avg_completion_tokens_per_s`
- `avg_warm_completion_tokens_per_s`
- `rows_with_same_signature`
- `rows_with_zero_copy_swap`

Reason:

- right now `avg_swap_time_s` is being polluted by the first shrink;
- we need to know whether later requests are paying copy cost, control-plane cost, or bad decode cost.

#### Change C: add a `--warm-start-active-set-json` option

Add a new evaluator option:

```bash
--warm-start-active-set-json /path/to/payload.json
```

Behavior:

1. before the measured loop starts, POST that payload once to `/swap_active_set`;
2. do not count it in benchmark results;
3. persist the warm-start response in output metadata.

This removes the dense->sparse first-shrink from measured latency.

#### Change D: add a `--force-static-active-set-json` option

Add:

```bash
--force-static-active-set-json /path/to/payload.json
```

Behavior in dynamic mode:

- skip `build_active_set_payload(...)` inside the loop;
- load the provided payload;
- rewrite only `request_id` per sample;
- keep the same `active_set`, `selected_slice_ids`, and signature.

This is the cleanest way to compare:

- current pseudo-dynamic 30% run
- true fixed sparse set

without inventing a second harness.

---

## Priority 1: fix the runtime no-op semantics

### File: `scripts/vllm_multiplex_server.py`

#### Change A: no-op reuse must compare against global active signature

Current bug:

```python
last_signature = app.state.dynamic_request_signatures.get(request_id)
```

That keys reuse by request ID. That is wrong for benchmark loops because every prompt has a new request ID.

Replace the check with:

```python
global_active_signature = getattr(app.state, "dynamic_active_signature", None)
if active_set_signature and active_set_signature == global_active_signature:
    ...
```

Keep request-local refresh bookkeeping, but no-op reuse should be based on the server's current active state.

#### Change B: expose control-plane timing explicitly

Return:

- `rpc_swap_time_s`
- `endpoint_overhead_s`

where:

```python
rpc_swap_time_s = swap_duration
endpoint_overhead_s = endpoint_elapsed - swap_duration
```

That lets the evaluator separate:

- lock/validation/aggregation cost
- worker swap cost

#### Change C: add a warm-sparse endpoint

Add:

```http
POST /warm_active_set
```

with behavior:

1. validate payload,
2. swap to it,
3. set `dynamic_active_signature`,
4. mark response as `warm_start_only=true`,
5. do not reset benchmark request-local router stats unless requested.

This is just a convenience wrapper so the benchmark can pre-shrink cleanly.

#### Change D: pin the dynamic base snapshot

Current dynamic path:

```python
base_snapshot[name] = param.cpu().clone()
```

Change to:

```python
base_snapshot[name] = param.detach().cpu().pin_memory().clone()
```

or equivalently clone first and then pin, depending on what is legal in your environment.

Point is simple:

- cartridge path uses pinned host memory,
- dynamic path should not use a worse transfer source than cartridge path.

#### Change E: stop doing pointless swap RPC work for same-set requests

Once global signature reuse is fixed, requests with identical signatures should skip:

- worker collective RPC,
- copy loop,
- router mask reapplication.

Return `worker_swap_result.status = "no_op_reuse_global"`.

---

## Priority 2: the decisive one-pair experiment

Run on one GPU pair only. Do not repeat the 4-pair concurrent run yet.

### Arms

#### Arm A - current 30pct path

- same plan
- measured from current dynamic evaluator
- but with warm-start pre-shrink

#### Arm B - forced static first active set

- generate the first sample's active-set payload once
- force it for all samples with `--force-static-active-set-json`

#### Arm C - current 30pct path without warm-start

- same as the current bad run
- included only to quantify first-shrink distortion

### Required outputs

- per-row JSON
- summary JSON
- one markdown verdict

### Interpretation

- If A ~= B on quality and latency:
  - current "dynamic" behavior is fake dynamic;
  - active-set diversity is not contributing.
- If A << C on swap latency but quality is unchanged:
  - first-shrink contamination was the latency distortion;
  - quality loss is active-set quality, not swap overhead.
- If A improves on B:
  - prompt-conditioned set changes are real and helping.

---

## Priority 3: the actual 3-arm system comparison

Use the prepared prompt sets:

- `test-output/three-arm-comparison-prep-2026-03-15/benchmark_slice_200.jsonl`
- `test-output/three-arm-comparison-prep-2026-03-15/personal_prompt_slice_50.jsonl`

### Arms

1. 4-GPU control
2. 2-GPU dense offload / UVA
3. 2-GPU dynamic 30pct sparse

### Why

This isolates the real bottleneck:

- if arm 2 keeps quality but arm 3 loses it -> active-set quality problem
- if arm 2 already loses quality or blows latency -> runtime/offload path problem

---

## Priority 4: true decode-refresh experiment

Do this only on:

- GSM8K
- ARC-Challenge

Those are the failure cases.

### Procedure

For each prompt:

1. prefill with active set S0
2. collect router misses
3. build refresh payload using `router_misses=...`, `phase="decode_refresh"`
4. apply exactly one refresh
5. continue generation

### Required measurements

- pre-refresh inactive ratio
- post-refresh inactive ratio
- answer correctness
- coherence
- extra swap time
- signature change yes/no

### Interpretation

- If one refresh sharply drops inactive ratio and improves GSM8K/ARC:
  - the architecture still has a path.
- If it does not:
  - 30pct is below the useful quality floor for this plan family.

---

## Concrete command skeletons

These are intentionally direct. Adjust ports and model names to match your remote host.

### 1. Warm-start pre-shrink payload generation

Generate a first-request payload from the same first benchmark prompt:

```bash
cd /Users/sero/ai/reap-expert-swap
python3 - <<'PY'
import json
from pathlib import Path
from scripts.evaluate_original_vs_multiplex import load_benchmarks, BenchmarkSpec, format_prompt, stable_id
from scripts.dynamic_reap import build_active_set_payload

plan = json.loads(Path('test-output/support-set-research-20260311-30pct/best-plan.json').read_text())
rows = load_benchmarks(sample_count=1, calibration_count=0, seed=7)
row = rows[0]
spec = {
    "mmlu": BenchmarkSpec("mmlu", "cais/mmlu", "all", "validation", "mcq", 8),
    "arc_challenge": BenchmarkSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation", "mcq", 8),
    "hellaswag": BenchmarkSpec("hellaswag", "Rowan/hellaswag", None, "validation", "mcq", 8),
    "winogrande": BenchmarkSpec("winogrande", "allenai/winogrande", "winogrande_xl", "validation", "mcq_numeric", 8),
    "gsm8k": BenchmarkSpec("gsm8k", "gsm8k", "main", "test", "math", 256),
}[row["benchmark"]]
prompt = format_prompt(spec, row)
payload = build_active_set_payload(
    plan,
    prompt,
    request_id=stable_id("dynamic", row["id"], "7"),
    benchmark=row["benchmark"],
    phase="prefill",
)
Path('/tmp/qwen35-30pct-first-active-set.json').write_text(json.dumps(payload, indent=2) + '\\n')
print('/tmp/qwen35-30pct-first-active-set.json')
PY
```

### 2. Arm A - current dynamic, but pre-shrunk

```bash
python3 scripts/evaluate_original_vs_multiplex.py \
  --mode dynamic \
  --server-url http://127.0.0.1:18361 \
  --model qwen35-dynamic-30pct-pair01 \
  --plan-json test-output/support-set-research-20260311-30pct/best-plan.json \
  --sample-count 5 \
  --calibration-count 0 \
  --seed 7 \
  --warm-start-active-set-json /tmp/qwen35-30pct-first-active-set.json \
  --output-json /tmp/armA.json \
  --output-md /tmp/armA.md
```

### 3. Arm B - forced static sparse set

```bash
python3 scripts/evaluate_original_vs_multiplex.py \
  --mode dynamic \
  --server-url http://127.0.0.1:18361 \
  --model qwen35-dynamic-30pct-pair01 \
  --plan-json test-output/support-set-research-20260311-30pct/best-plan.json \
  --sample-count 5 \
  --calibration-count 0 \
  --seed 7 \
  --force-static-active-set-json /tmp/qwen35-30pct-first-active-set.json \
  --output-json /tmp/armB.json \
  --output-md /tmp/armB.md
```

### 4. Arm C - current path, no warm-start

```bash
python3 scripts/evaluate_original_vs_multiplex.py \
  --mode dynamic \
  --server-url http://127.0.0.1:18361 \
  --model qwen35-dynamic-30pct-pair01 \
  --plan-json test-output/support-set-research-20260311-30pct/best-plan.json \
  --sample-count 5 \
  --calibration-count 0 \
  --seed 7 \
  --output-json /tmp/armC.json \
  --output-md /tmp/armC.md
```

---

## What success looks like next

The next successful step is not "30pct passes."

The next successful step is:

1. we know whether current 30pct eval is static or actually dynamic,
2. we know how much first-shrink polluted latency,
3. we know whether dense 2-GPU offload is the true bottleneck or not,
4. we know whether one refresh can recover GSM8K/ARC.

Until those are answered, any further global optimization work is premature.
