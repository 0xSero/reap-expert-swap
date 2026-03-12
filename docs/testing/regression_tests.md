# Regression tests

This repo now includes focused regression coverage for the active Python research/runtime path.

## Why these tests exist

The recent changes introduced risk in core logic that should be covered without needing GPU-backed end-to-end runs:

- byte-aware size targeting
- gate integrity parsing from server logs
- bounded cartridge-cache behavior
- controller target-count selection
- dynamic core/specialist budget splitting
- slice-catalog construction and active-set assembly
- one-refresh enforcement and router-miss refresh policy
- personal activation corpus extraction and dedupe
- dynamic quality gate thresholds

These are pure logic changes, so they should be covered with fast deterministic tests rather than GPU-bound end-to-end runs.

## Current test layout

### Python regression tests

- `tests_py/test_research_scripts.py`
- `tests_py/test_dynamic_reap.py`

Run with:

```bash
uv run python -m unittest discover -s tests_py -p 'test_*.py'
```

## What the Python tests cover

### 1. Bounded cartridge cache logic

`scripts/multiplex_cache.py`

Covered behaviors:

- touching an already-loaded cartridge moves it to the MRU end
- loading a new cartridge evicts the oldest one when the cache is full

### 2. Gate integrity parsing

`scripts/research_gate.py`

Covered behaviors:

- positive zeroed-expert counts are parsed correctly from server logs
- missing cartridge entries are reported as invalid
- non-positive zeroed-expert counts are rejected
- dynamic quality gate rejects >5 percent overall quality loss or >0.10 absolute benchmark drop

### 3. Size estimator and controller targeting

`scripts/size_estimator.py`
`scripts/target_controller.py`

Covered behaviors:

- `5` cartridges is not near a true 20 percent resident target for Qwen1.5-MoE-A2.7B-Chat
- `13` cartridges is near the target
- the controller picks `13` as the target count
- empty-history planning starts with `13, 12, 14`

### 4. Dynamic core/specialist planning

`scripts/dynamic_reap.py`
`scripts/build_partitioned_reap_plan.py`

Covered behaviors:

- swappable expert budget is split into core and specialist pools
- learned slice catalogs are emitted per layer
- candidate pool sizing respects the `3x` multiplier
- active-set assembly stays within budget
- `core ∪ selected slices` is enforced exactly
- refresh policy triggers once and then stops when the budget is exhausted

### 5. Personal activation corpus build

`scripts/personal_activation_corpus.py`
`scripts/build_personal_activation_corpus.py`

Covered behaviors:

- user prompts are extracted from multiple local history shapes
- duplicate prompts are removed by normalized prompt hash
- domain tags are assigned for scorer priors

## What is not covered yet

These tests do **not** currently cover:

- full GPU-backed vLLM multiplex end-to-end behavior
- actual mid-generation refresh inside a live decode stream
- end-to-end expert activation collection over the personal chat corpus on a live model
- live benchmark dynamic-mode wins at the 20 percent VRAM target

## Recommended next tests

Once the live dynamic path is wired on GPU, add tests for:

1. prefill-only router miss capture on real requests
2. decode refresh application inside the same request lifecycle
3. active-set cache reuse and eviction behavior for repeated prompts
4. end-to-end acceptance artifact generation for baseline vs static vs dynamic

## Related code

- `scripts/dynamic_reap.py`
- `scripts/build_personal_activation_corpus.py`
- `scripts/personal_activation_corpus.py`
- `scripts/vllm_multiplex_server.py`
- `scripts/evaluate_original_vs_multiplex.py`
- `scripts/research_gate.py`
- `scripts/size_estimator.py`
- `scripts/target_controller.py`
- `tests_py/test_research_scripts.py`
- `tests_py/test_dynamic_reap.py`
