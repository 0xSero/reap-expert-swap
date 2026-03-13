# reap-expert-swap

This repo is the **target experiment workspace** for VRAM-first research on sparse MoE serving.

Right now, the clean operator entrypoint is **not this repo root**. The clean control surface lives in:

- `~/ai/autoresearch`

From there, the local research model is brought up on:

- `http://localhost:8317/v1`
- model: `gpt-5.4`
- reasoning effort: `xhigh`

and the overnight loop calls into this repo as the target workspace.

## What this repo is actually for

This repo contains the messy research machinery for trying to make:

> **resident floor + on-demand MoE deltas**

work with **real VRAM savings** instead of fake logical sparsity.

Concretely, the research goal is:
- reduce **real physical GPU residency**
- preserve a **resident floor/core** that handles most requests
- keep outputs close to a frozen **BF16 baseline**
- capture enough traces to improve routing over time

## Current state, honestly

This is not a polished package. It is a live research workspace.

### What currently works
- matched BF16-vs-candidate evaluation
- benchmark and workflow response capture
- long-context replay probing
- router-miss / inactive-ratio collection
- sleep-backed coarse cartridge switching that **does** reduce real GPU memory
- cartridge autoresearch over routing-derived request clusters
- core+delta manifest construction

### What does **not** currently work well enough
- the old dynamic path does **not** give real VRAM savings because it still keeps the full expert tensor surface on GPU and mostly zeroes inactive experts in place
- the first compact active-expert offload attempt physically copied only active experts but crashed during inference
- the old zeroed sleep cartridges save VRAM but produce bad outputs
- generalization is not good enough yet on real workflow conversations

## The one thing to understand

There are two different systems here:

### 1. Old dynamic active-set path
This gives:
- logical active expert footprints
- but **not** true physical GPU savings

### 2. Sleep-backed cartridge path
This gives:
- real off-GPU unloading
- real GPU memory drops
- but currently needs better cartridges and better routing-derived partitions

That is why the current lane is:

> **resident floor + delta cartridges + coarse router + sleep-backed offload**

## Clean way to start

Do **not** start here first.

Start from `~/ai/autoresearch`:

```bash
cd ~/ai/autoresearch
bash scripts/start_local_openai_stack.sh
bash scripts/run_reap_vram_first.sh
```

That uses:
- local research model endpoint on `localhost:8317`
- remote execution target from `.env.autoresearch`
- this repo as the experiment target workspace

## The evaluation framework

The current framework is:

> **Frozen BF16 once + fixed slices + full traces + 5 metrics + no touching the evaluator mid-loop**

### The 5 metrics
1. `live_gpu_gib`
2. `benchmark_accuracy_pct`
3. `similarity_pct`
4. `total_s`
5. `invalid_output_pct`

### Required trace visibility
A serious run should emit:
- `responses.csv`
- `responses_with_baseline.csv`
- `resource_samples.jsonl`
- `comparison.json`
- `summary.json`
- `summary.md`
- router misses / inactive ratio
- swap bytes / swap occurrence / swap time

## Current operator mental model

### Operator/control repo
- `~/ai/autoresearch`

### Target workspace
- `~/ai/reap-expert-swap`

### Local model endpoint
- `http://localhost:8317/v1`

### Remote execution host
- configured through env (currently `ser@192.168.1.70`)

## Most useful docs in this repo

- `active_work/docs/five_metric_framework_20260313.md`
- `active_work/docs/vram_first_plan_20260313.md`
- `active_work/analysis/core_delta_cartridges_20260313/RECOMMENDATION.md`
- `active_work/analysis/generalization_check_20260313/report.md`
- `experiments/README.md`

## Most useful scripts in this repo

- `scripts/run_vram_first_overnight.sh`
- `scripts/run_dynamic_cartridge_autoresearch.py`
- `scripts/build_core_delta_cartridges.py`
- `scripts/build_zeroed_cartridge_exports.py`
- `scripts/run_cartridge_vs_bf16_eval.py`
- `scripts/run_long_context_personal_probe.py`
- `scripts/run_sleep_cartridge_real_replay.py`
- `scripts/vllm_multiplex_server.py`

## What a good overnight run should do

A good overnight run should:
1. build conservative routing-derived cartridge candidates
2. keep the floor stable
3. export only bounded artifacts
4. compare against frozen BF16 references
5. record VRAM, latency, validity, and similarity
6. reject candidates that only improve one metric by breaking everything else

## What not to believe

Do **not** trust:
- logical active-expert bytes alone
- benchmark-only wins with no real GPU drop
- runs without full trace artifacts
- results that mutate the evaluation slices or scorer mid-loop

## Bottom line

This repo is the **research target**, not the clean control repo.

If you want the clean start, go to `~/ai/autoresearch`.
If you want the messy experiment internals, they are here.
