# REAP Multiplex Research Protocol

This project now treats multiplexing as a boundary-mapping problem, not a one-off demo.

## North Star

Find the smallest expert cartridge that preserves useful quality while keeping swap overhead low enough for practical deployment.

## Current validated facts

- The multiplex server now truly zeroes non-owned experts under TP8/EP sharding.
- The first "100% retained accuracy" result was invalid because cartridges were clones.
- After fixing zeroing, the 10-cartridge configuration retained only about `9.9%` of baseline accuracy on the current 5-benchmark / 200-sample evaluation.
- The same 10-cartridge run retained only about `24.1%` of baseline coherence and introduced about `0.824s` average swap latency across `152` swaps.

## Scientific controls

- Hold the dense baseline fixed for each seed before any multiplex comparisons.
- Use the same benchmark suite, prompt format, sample counts, and model build for baseline and multiplex.
- Separate plan generation from evaluation and store both as artifacts.
- Record the exact seed, cartridge count, plan path, baseline path, and multiplex path for every run.
- Refuse a multiplex run if startup logs do not show strictly positive zeroed expert counts.

## Core metrics

- `accuracy_retained_pct`
- `coherence_retained_pct`
- `avg_sample_time_s`
- `p95_sample_time_s`
- `avg_swap_time_s`
- `swap_count`
- `cartridge_transition_rate`
- `parse_error_rate`
- raw per-sample outputs for failure review

## Minimum experiment matrix

Run these first:

- cartridge counts: `2`, `4`, `6`, `8`, `10`
- seeds: at least `2`
- benchmarks: current 5-benchmark suite
- traffic mode: current interleaved multiplex evaluation

Expand only after the first frontier is visible.

## Artifact layout

- `full-run-20k/research/runs/<run-tag>/seed-<seed>/baseline/`
- `full-run-20k/research/runs/<run-tag>/seed-<seed>/cartridge-<count>/`
- `full-run-20k/research/plans/`
- `full-run-20k/research/dashboard.json`
- `full-run-20k/research/dashboard.md`
- `full-run-20k/research/status.json`
- `full-run-20k/research/daemon-status.json`

## Continuous research loop

- `scripts/run_boundary_research.py` runs one scientific batch.
- `scripts/run_continuous_research.py` keeps launching new batches with new seeds.
- `scripts/build_research_dashboard.py` aggregates all completed runs into a dashboard.

## Autoresearch adaptation

`karpathy/autoresearch` is useful here as an organizing pattern, not as the literal training harness.

For this project, the equivalent autonomous loop is:

1. choose the next experiment batch,
2. run dense baseline,
3. build partition plans,
4. verify true zeroing,
5. run multiplex evaluation,
6. publish artifacts and dashboard updates,
7. repeat with new seeds or cartridge counts.
