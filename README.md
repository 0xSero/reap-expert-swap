# reap-expert-swap

Dynamic expert-floor construction and runtime swapping for sparse Mixture-of-Experts models.

This system reduces the resident VRAM footprint of large MoE models (tested on Qwen3.5-35B-A3B) by keeping a profiled expert floor in GPU memory and dynamically materializing prompt-conditioned specialist experts at request time.

## Key results (Qwen3.5-35B-A3B)

| Metric | Value | Notes |
|---|---|---|
| Full BF16 model size | 63.4 GiB | estimated via size_estimator |
| Profiled floor size | 23.5 GiB | 62.9% VRAM reduction |
| Swap latency | 0.33s per request | avg across smoke eval |
| Accuracy retained (smoke, 5-bench) | 98% | vs dense baseline, same seed |
| Parsed answer agreement vs BF16 | 100% | 5-sample matched eval |
| Avg response similarity vs BF16 | 76.75% | verbatim text overlap varies |
| Full95 coverage by floor | 40.1% | known gap -- active research area |

Caveats: the accuracy and fidelity numbers come from small sample evaluations (5-50 prompts). The 40.1% full95 coverage means the floor only covers ~40% of the experts the full model uses at the 95th-percentile activity threshold. This is the main bottleneck under active research. See [activation-diff analysis](docs/system_technical_report_20260312.md) for details.

## How it works

1. **Profile router activity** across a calibration corpus to identify which experts fire most frequently per layer.
2. **Build a resident floor** -- the subset of experts that covers the bulk of activation mass -- and keep it permanently in VRAM.
3. **At request time**, the router identifies which non-resident experts the prompt needs. A delta swap loads those experts from disk/host memory into GPU memory, displacing the least-recently-used non-floor experts.
4. **Evaluate** the dynamic configuration against a dense BF16 baseline using matched prompts, seeds, and benchmark suites.
5. **Gate** each experiment through a research gate that checks accuracy, coherence, swap latency, and parse error rate before accepting.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/0xsero/reap-expert-swap.git
cd reap-expert-swap
uv sync
```

For GPU-dependent scripts (the vLLM server, remote evaluation), you also need:
- [vLLM](https://docs.vllm.ai/) 0.16+ with tensor-parallel support
- PyTorch 2.x with CUDA
- A machine with enough VRAM to hold the resident floor (24+ GiB recommended for Qwen3.5-35B-A3B)

For local-only work (plan building, analysis, tests), no GPU is required.

## Quick start

### Run the tests

```bash
uv run python -m pytest tests_py/ -v
```

### Build a dynamic floor plan

Given a base plan and router activity profile, construct a profiled floor:

```bash
uv run python scripts/build_profiled_floor_plan.py \
  --base-plan path/to/base-plan.json \
  --profile path/to/router-activity-profile.json \
  --active-threshold full95 \
  --inactive-threshold full80 \
  --output path/to/floor-plan.json
```

### Profile router activity

Analyze dynamic evaluation results to produce per-layer activation mass profiles:

```bash
uv run python scripts/profile_router_activity.py \
  --dynamic-payload path/to/dynamic-results.json \
  --plan path/to/plan.json \
  --output path/to/router-profile.json
```

### Build a partitioned plan

Construct expert partitions with core and specialist budgets:

```bash
uv run python scripts/dynamic_reap.py \
  --help  # see all plan-building modes
```

The main plan-building entry points in `dynamic_reap.py`:
- `build_dynamic_plan()` -- general dynamic plan from observation summaries
- `build_dynamic_floor_plan()` -- floor plan from observation data
- `build_active_set_payload()` -- request-time active set construction

### Evaluate baseline vs dynamic

Run a matched evaluation comparing the dense BF16 baseline against a dynamic configuration:

```bash
uv run python scripts/evaluate_original_vs_multiplex.py \
  --baseline-url http://HOST:PORT/v1 \
  --dynamic-url http://HOST:PORT/v1 \
  --plan path/to/plan.json \
  --sample-count 40 \
  --seed 7 \
  --output-dir path/to/results/
```

This produces per-sample results, timing data, router miss analysis, and a gate verdict.

### Start the patched vLLM server

The multiplex server patches vLLM's serving layer to support dynamic expert swaps:

```bash
REAP_MAX_LOADED_CARTRIDGES=4 \
REAP_ENABLE_ROUTER_MASKS=1 \
uv run python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 8 \
  --port 8011
```

The server monkey-patches vLLM at import time (see `scripts/vllm_multiplex_server.py`) to intercept expert loading and apply active-set masks.

### Research gate

Every experiment passes through a gate that checks retention thresholds:

```bash
uv run python -c "
from scripts.research_gate import evaluate_payload_gate
import json, pathlib
payload = json.loads(pathlib.Path('results/dynamic.json').read_text())
verdict = evaluate_payload_gate(payload, gate_profile='budget_static')
print(json.dumps(verdict, indent=2))
"
```

Gate profiles are defined in `scripts/research_gate.py` with thresholds for accuracy, coherence, swap time, and parse error rate.

## About autoresearch

This project uses an autoresearch pattern inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): an autonomous loop that generates candidate plans, materializes them on a remote GPU machine, runs baseline-vs-dynamic evaluation, checks the result against a gate, and mutates toward the next experiment.

**The autoresearch orchestration scripts are not included in this public repo** because they contain infrastructure-specific wiring (SSH paths, remote host addresses, PID management). What IS included are all the building blocks the loop calls:

| Script | Role in the loop |
|---|---|
| `dynamic_reap.py` | Plan generation and active-set construction |
| `evaluate_original_vs_multiplex.py` | Matched baseline-vs-dynamic evaluation |
| `research_gate.py` | Automated pass/fail gating |
| `vllm_multiplex_server.py` | Patched vLLM runtime with expert swapping |
| `profile_router_activity.py` | Post-hoc activation profiling |
| `build_profiled_floor_plan.py` | Floor refinement from profiles |
| `build_support_router_dataset.py` | Training data for learned routing |
| `train_support_router.py` | Lightweight learned support-router |

To build your own autoresearch loop, wire these scripts together:

1. Generate a plan with `dynamic_reap.py`
2. Deploy the plan to a vLLM server running `vllm_multiplex_server.py`
3. Run evaluation with `evaluate_original_vs_multiplex.py`
4. Check the gate with `research_gate.py`
5. If the gate fails, mutate the plan (adjust budgets, swap thresholds, floor composition) and repeat
6. Profile the results with `profile_router_activity.py` to inform the next iteration

Over 22 experiments were run through this loop. All 22 were rejected by the gate, which led to the current profiled-floor approach that achieves 98% retained accuracy.

## Project layout

```
scripts/                  # Core runtime, evaluation, and planning scripts
  dynamic_reap.py         # Plan building, active-set construction (1540 lines)
  evaluate_original_vs_multiplex.py  # Evaluation harness (2071 lines)
  vllm_multiplex_server.py           # Patched vLLM server (879 lines)
  research_gate.py        # Automated experiment gating (564 lines)
  profile_router_activity.py         # Activation profiling (222 lines)
  build_profiled_floor_plan.py       # Floor construction from profiles
  router_activity.py      # Router activity utilities
  support_router.py       # Learned routing path
  train_support_router.py # Support router training
  build_support_router_dataset.py    # Training data construction
  build_partitioned_reap_plan.py     # Partitioned plan builder
  size_estimator.py       # VRAM size estimation
  dynamic_swap_delta.py   # Delta swap computation
  multiplex_cache.py      # LRU cartridge cache
  personal_activation_corpus.py      # Activation corpus builder
  run_budget_oracle_analysis.py      # Budget oracle analysis
tests_py/                 # Regression tests (no GPU required)
docs/                     # Architecture docs, protocols, research history
  system_technical_report_20260312.md
  research_history_20260312.md
  architecture/           # System design documents
  protocol/               # Evaluation and research protocols
  notes/                  # Blog post and informal notes
artifacts/summaries/      # Curated experiment summaries
configs/                  # Configuration templates
dataset/                  # Dataset schema documentation
examples/                 # Example configurations
```

## Documentation

- [System Technical Report](docs/system_technical_report_20260312.md) -- full architecture, runtime design, evaluation harness, and current metrics
- [Research History](docs/research_history_20260312.md) -- chronological record of all experiments and decisions
- [Research Protocol](docs/protocol/research_protocol.md) -- evaluation methodology and scientific controls
- [Core Architecture](docs/architecture/core_specialist_dynamic_architecture.md) -- core-specialist dynamic architecture design
- [Multiplex Loading Strategy](docs/architecture/multiplex_loading_strategy.md) -- how expert loading and swapping works
- [RESEARCH.md](RESEARCH.md) -- high-level research summary

## License

Apache License 2.0. See [LICENSE](LICENSE).
