# One-Request Forensic Replay Summary

```text
Forensic readiness
────────────────────────────────────────
Poison detected             ✅
Crash classification clean  ❌
Expected fields complete    ❌
Ready for full sweep        ❌
```

## Crash Classification
- `missing_completion_payload`

## Expected Field Coverage
- Resolved: **0**
- Missing: **10**

### Missing Fields
- `request_id`
- `swap_request_id`
- `swap_plan_identity`
- `plan_sha256`
- `active_set_signature`
- `union_validation`
- `core_presence_summary`
- `worker_swap_result`
- `router_miss_payload`
- `crash_classification`

## Resolved Forensic Fields

## Expected JSON Interface (server -> collector)
```json
{
  "request_id": "...",
  "swap_request_id": "...",
  "plan_identity": {
    "plan_mode": "dynamic_core_specialist",
    "plan_budget_bytes": 0,
    "plan_sha256": "...",
    "plan_path": "..."
  },
  "active_set": {
    "signature": "...",
    "union_validation": {
      "ok": true,
      "violations": []
    },
    "core_presence_summary": {
      "layers_checked": 40,
      "layers_missing_core": 0
    }
  },
  "worker": {
    "swap_result": {
      "active_expert_bytes": 0,
      "active_expert_count": 0,
      "delta_added": 0,
      "delta_removed": 0,
      "delta_reused": 0
    }
  },
  "router_miss_payload": {},
  "crash": {
    "classification": "none|poisoned_runtime|engine_dead"
  }
}
```
