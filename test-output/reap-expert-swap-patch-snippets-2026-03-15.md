# reap-expert-swap patch snippets - 2026-03-15

I tried writing directly into `/Users/sero/ai/reap-expert-swap` after you said full access was enabled.

The sandbox still rejects writes there:

```text
zsh: operation not permitted: /Users/sero/ai/reap-expert-swap/.codex_write_test_...
```

So here are the exact code snippets to paste in once the permission change actually reaches this session.

---

## 1) `scripts/vllm_multiplex_server.py`

### A. Pin the dynamic base snapshot

Replace:

```python
base_snapshot[name] = param.cpu().clone()
```

With:

```python
host_copy = param.detach().cpu().clone()
try:
    host_copy = host_copy.pin_memory()
except Exception:
    pass
base_snapshot[name] = host_copy
```

Why:

- cartridge path already uses pinned memory;
- dynamic path should not take the slower host-memory path.

---

### B. Use global active signature for no-op reuse

Current code:

```python
state = app.state.dynamic_request_state.get(request_id, {"refreshes_used": 0})
last_signature = app.state.dynamic_request_signatures.get(request_id)
```

Change to:

```python
state = app.state.dynamic_request_state.get(request_id, {"refreshes_used": 0})
last_signature = getattr(app.state, "dynamic_active_signature", None)
```

And keep:

```python
app.state.dynamic_active_signature = active_set_signature
```

after a real swap succeeds.

Why:

- request IDs change every prompt;
- same-signature requests should be treated as true no-op reuse.

---

### C. Return explicit timing split from `/swap_active_set`

Around the collective RPC:

```python
swap_start = time.time()
swap_results = await engine.collective_rpc(
    "multiplex_swap_active_set",
    args=(validated, plan),
    timeout=600,
)
swap_duration = time.time() - swap_start
```

Change to:

```python
endpoint_started = time.time()
swap_start = time.time()
swap_results = await engine.collective_rpc(
    "multiplex_swap_active_set",
    args=(validated, plan),
    timeout=600,
)
swap_duration = time.time() - swap_start
endpoint_elapsed = time.time() - endpoint_started
endpoint_overhead_s = max(0.0, endpoint_elapsed - swap_duration)
```

Then return these extra fields in both no-op and real-swap success payloads:

```python
"rpc_swap_time_s": swap_duration,
"endpoint_overhead_s": endpoint_overhead_s,
```

For no-op reuse:

```python
"rpc_swap_time_s": 0.0,
"endpoint_overhead_s": 0.0,
```

And mirror them into `worker_swap_result` / `forensic_payload` if you want downstream collection to stay uniform.

---

## 2) `scripts/evaluate_original_vs_multiplex.py`

### A. Add CLI options

Under the existing parser args:

```python
parser.add_argument("--warm-start-active-set-json")
parser.add_argument("--force-static-active-set-json")
```

---

### B. Add helper loaders near `swap_active_set(...)`

Add:

```python
def load_active_set_payload(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid active-set payload: {path}")
    return payload


def clone_active_set_payload(payload: dict[str, Any], *, request_id: str, phase: str = "prefill") -> dict[str, Any]:
    cloned = json.loads(json.dumps(payload))
    cloned["request_id"] = request_id
    cloned["phase"] = phase
    return cloned
```

---

### C. Extend `evaluate_samples(...)`

Change signature from:

```python
def evaluate_samples(
    rows,
    spec_map,
    server_url,
    model,
    mode,
    plan,
    benchmark_to_cartridge,
    interleaved,
    seed,
    request_timeout_s,
):
```

To:

```python
def evaluate_samples(
    rows,
    spec_map,
    server_url,
    model,
    mode,
    plan,
    benchmark_to_cartridge,
    interleaved,
    seed,
    request_timeout_s,
    *,
    warm_start_active_set_payload: dict[str, Any] | None = None,
    force_static_active_set_payload: dict[str, Any] | None = None,
):
```

---

### D. Warm-start before timed loop

Right after:

```python
results = []
current_cartridge = None
current_active_signature = None
```

Add:

```python
warm_start_result = None
if mode == "dynamic" and warm_start_active_set_payload:
    warm_payload = clone_active_set_payload(
        warm_start_active_set_payload,
        request_id=f"warmstart::{seed}",
        phase="prefill",
    )
    warm_start_result = swap_active_set(server_url, warm_payload)
    if warm_start_result.get("status") != "success":
        raise RuntimeError(warm_start_result.get("error", "warm start failed"))
    current_active_signature = warm_start_result.get("active_set_signature") or warm_payload.get("active_set_signature")
```

And later, write this into top-level payload metadata.

---

### E. Allow forced static active set

Inside the dynamic branch, replace:

```python
active_payload = build_active_set_payload(
    plan,
    prompt,
    request_id=request_id,
    benchmark=row["benchmark"],
    phase="prefill",
)
```

With:

```python
if force_static_active_set_payload:
    active_payload = clone_active_set_payload(
        force_static_active_set_payload,
        request_id=request_id,
        phase="prefill",
    )
else:
    active_payload = build_active_set_payload(
        plan,
        prompt,
        request_id=request_id,
        benchmark=row["benchmark"],
        phase="prefill",
    )
```

---

### F. Add better per-row metrics

After `swap_worker_metrics = extract_swap_worker_metrics(swap_payload)`, add:

```python
swap_internal_s = float(swap_payload.get("swap_time_s", 0.0)) if swap_payload else 0.0
swap_control_plane_s = max(0.0, swap_time_s - swap_internal_s)
is_same_signature = bool(
    previous_active_signature is not None
    and effective_active_signature is not None
    and effective_active_signature == previous_active_signature
)
is_zero_copy_swap = int(swap_worker_metrics.get("bytes_copied") or 0) == 0
is_cold_shrink = str(swap_phase or "").startswith("cold_")
completion_tokens_per_s = (
    float(usage.get("completion_tokens") or 0.0) / request_latency_s
    if request_latency_s > 0
    else 0.0
)
```

Then add these row fields:

```python
"swap_control_plane_s": round(swap_control_plane_s, 6),
"is_same_signature": is_same_signature,
"is_zero_copy_swap": is_zero_copy_swap,
"is_cold_shrink": is_cold_shrink,
"completion_tokens_per_s": round(completion_tokens_per_s, 6),
```

And replace:

```python
"swap_internal_s": round(float(swap_payload.get("swap_time_s", 0.0)), 6)
```

with:

```python
"swap_internal_s": round(swap_internal_s, 6)
```

---

### G. Add better summary metrics

Inside `build_bucket(...)`, add collections for:

```python
swap_control_plane_latencies = [
    float(row.get("swap_control_plane_s") or 0.0)
    for row in rows
    if float(row.get("swap_time_s") or 0.0) > 0
]
warm_swap_control_plane_latencies = [
    float(row.get("swap_control_plane_s") or 0.0)
    for row in warm_swap_rows
]
cold_request_latencies = [
    float(row.get("request_latency_s") or 0.0)
    for row in cold_swap_rows
    if not row.get("error")
]
cold_total_latencies = [
    float(row.get("total_latency_s") or 0.0)
    for row in cold_swap_rows
]
warm_change_swap_latencies = [
    float(row.get("swap_time_s") or 0.0)
    for row in warm_change_rows
]
warm_reuse_swap_latencies = [
    float(row.get("swap_time_s") or 0.0)
    for row in warm_reuse_rows
]
completion_tps = [
    float(row.get("completion_tokens_per_s") or 0.0)
    for row in rows
    if not row.get("error")
]
warm_completion_tps = [
    float(row.get("completion_tokens_per_s") or 0.0)
    for row in rows
    if not row.get("error") and row not in cold_swap_rows
]
rows_with_same_signature = sum(1 for row in rows if row.get("is_same_signature"))
rows_with_zero_copy_swap = sum(1 for row in rows if row.get("is_zero_copy_swap"))
```

Then add summary fields:

```python
"avg_swap_control_plane_s": round(statistics.fmean(swap_control_plane_latencies), 6) if swap_control_plane_latencies else 0.0,
"avg_warm_swap_control_plane_s": round(statistics.fmean(warm_swap_control_plane_latencies), 6) if warm_swap_control_plane_latencies else 0.0,
"avg_cold_request_time_s": round(statistics.fmean(cold_request_latencies), 6) if cold_request_latencies else 0.0,
"avg_cold_sample_time_s": round(statistics.fmean(cold_total_latencies), 6) if cold_total_latencies else 0.0,
"avg_warm_change_swap_time_s": round(statistics.fmean(warm_change_swap_latencies), 6) if warm_change_swap_latencies else 0.0,
"avg_warm_reuse_swap_time_s": round(statistics.fmean(warm_reuse_swap_latencies), 6) if warm_reuse_swap_latencies else 0.0,
"avg_completion_tokens_per_s": round(statistics.fmean(completion_tps), 6) if completion_tps else 0.0,
"avg_warm_completion_tokens_per_s": round(statistics.fmean(warm_completion_tps), 6) if warm_completion_tps else 0.0,
"rows_with_same_signature": rows_with_same_signature,
"rows_with_zero_copy_swap": rows_with_zero_copy_swap,
```

---

### H. Thread the new args through `main()`

After parsing args:

```python
warm_start_active_set_payload = load_active_set_payload(args.warm_start_active_set_json)
force_static_active_set_payload = load_active_set_payload(args.force_static_active_set_json)
```

Then, when building `eval_kwargs`, add:

```python
if args.protocol != "multi_turn":
    eval_kwargs["warm_start_active_set_payload"] = warm_start_active_set_payload
    eval_kwargs["force_static_active_set_payload"] = force_static_active_set_payload
```

And in the final top-level payload metadata, add:

```python
payload.setdefault("artifacts", {})
if args.warm_start_active_set_json:
    payload["artifacts"]["warm_start_active_set_json"] = str(Path(args.warm_start_active_set_json).resolve())
if args.force_static_active_set_json:
    payload["artifacts"]["force_static_active_set_json"] = str(Path(args.force_static_active_set_json).resolve())
```

If you also preserve `warm_start_result`, store that in payload metadata too.

---

## 3) After patching, use this exact first experiment

Use the already-staged runbook:

- `/Users/sero/ai/autoresearch/test-output/run-30pct-isolation-experiments.sh`

That executes:

1. current dynamic, pre-shrunk
2. forced static first active set
3. current cold-included path

on one pair only.
