# Dynamic+Saliency Parity Handoff

This repo was intentionally reduced to the minimum artifact set for the proven working path:

- `reap_swap/vllm_multiplex_server.py`
- `assets/strict30-v2-plan.json`
- `test-output/three-arm-20260316-141609/arm2_dynamic.json`
- `test-output/three-arm-20260316-141609/arm2_dynamic.md`

## Proven result

Dynamic+Saliency reached BF16 parity:

- accuracy: `0.88`
- coherence: `1.0`
- avg sample time: `4.12s`
- router miss ratio: `0.0`
- avg swap bytes copied: `0.0`
- dynamic signature count: `25`

## Required runtime settings

Use the multiplex server with:

- `REAP_SWAP_MASKS_ONLY=1`
- `REAP_ENABLE_ROUTER_MASKS=0`
- `REAP_PLAN_FILE=/path/to/strict30-v2-plan.json`
- vLLM `--cpu-offload-params experts`

This is the parity configuration because:

- removed experts are not zeroed
- router logits are not masked to inactive experts
- all experts remain accessible through native vLLM UVA CPU offload
- REAP still drives dynamic active-set signatures and request-specific planning

## Why this worked

Quality loss was caused by two bugs in the old path:

1. zeroing removed experts in VRAM
2. masking router logits with `-inf` for non-active experts

With both disabled, Dynamic+Saliency matched BF16 quality exactly.

## If continuing from here

Focus only on vLLM + UVA + REAP-driven dynamic planning.

Do not reintroduce:

- expert zeroing
- router masking of inactive experts
- any path that prevents vLLM native expert offload from accessing all experts
