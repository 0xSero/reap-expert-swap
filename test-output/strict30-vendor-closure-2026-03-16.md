# strict30 vendor closure

## Minimal eval-only set
- `reap_swap/evaluate_original_vs_multiplex.py`
- `reap_swap/dynamic_reap.py`
- `reap_swap/research_gate.py`
- `reap_swap/size_estimator.py`
- `test-output/support-set-research-20260311-30pct/best-plan.json (hosted in /Users/sero/ai/reap-expert-swap)`

## Minimal full runtime+eval set
- `reap_swap/evaluate_original_vs_multiplex.py`
- `reap_swap/vllm_multiplex_server.py`
- `reap_swap/dynamic_reap.py`
- `reap_swap/research_gate.py`
- `reap_swap/size_estimator.py`
- `reap_swap/dynamic_swap_delta.py`
- `reap_swap/multiplex_cache.py`
- `test-output/support-set-research-20260311-30pct/best-plan.json (hosted in /Users/sero/ai/reap-expert-swap)`

## External assumptions
- remote runtime host: ser@192.168.1.70
- remote vllm env: /home/ser/reap-expert-swap-vllm016/.venv
- remote runtime repo: /home/ser/reap-expert-swap-reap
- server must launch through vllm_multiplex_server.py, not plain vllm serve
- server-side deps: torch, vllm, uvloop
- eval-side deps: requests, datasets
- optional plan-side deps when support-router is enabled: joblib, scipy
- REAP_PLAN_FILE must point at a valid dynamic_core_specialist plan
- evaluator expects /v1/completions, /swap_active_set, /warm_active_set, /router_misses/{request_id}, /forensics/{request_id}
