# Multiplex loading strategy

This document explains the current cartridge loading approach for the vLLM multiplex server and why it changed.

## Problem summary

At high cartridge counts such as `13`, the original eager preload path was not stable.

Observed failure modes:

- eager preload attempted to pin nearly every cartridge before evaluation
- bulk `collective_rpc` calls eventually failed during preload with:
  - `collective_rpc should not be called on follower node`
- even after removing eager preload, retaining too many pinned cartridges at once caused shared-memory and broadcast stalls

This made the 20 percent objective effectively untestable, even when the controller and plan selection logic were correct.

## Current strategy

The multiplex server now uses:

1. **lazy cartridge loading**
   - a cartridge is loaded only when first requested by `/swap_cartridge/{cartridge_id}`
2. **bounded pinned-cartridge cache**
   - only a limited number of cartridges remain pinned in CPU memory at once
3. **FIFO eviction**
   - once the cache is full, the oldest pinned cartridge is unloaded before a new one is loaded

The default cache size is controlled by:

```bash
REAP_MAX_LOADED_CARTRIDGES=4
```

## Why this is better

The lazy bounded-cache design avoids two pathologies:

- bulk preload storms across all cartridges
- pinned-memory hoarding that scales linearly with cartridge count

In live testing this changed swap behavior from:

- multi-second preload-driven swaps
- preload failures around `cartridge_11` or `cartridge_12`

to:

- on-demand loads
- repeated successful evictions
- swap times around `0.34s` to `0.38s`
- bandwidth around `60` to `70 GB/s`

## Operational phases

### Calibration phase

Calibration touches many cartridges because the selector compares candidates across benchmarks.

For this phase the bounded cache is important because:

- the server may cycle through most or all cartridges
- keeping every cartridge pinned is wasteful
- a smaller hot set is enough

### Test phase

After benchmark winners are chosen, the number of actually used cartridges is usually much smaller than the full cartridge set.

Future optimization:

- after selection, raise the cache size to match the unique selected cartridges
- optionally preload only the selected winners

That should reduce swap churn further in the real evaluation pass.

## Current limitation

The current multiplex evaluator still performs exhaustive per-benchmark cartridge calibration.

That means the server may still touch nearly every cartridge during calibration even with the improved loading strategy.

So the loading fix solves the systems stability problem, but not the search-cost problem.

## Recommended next improvement

The next performance improvement should be:

### shortlist calibration

Instead of evaluating every cartridge for every benchmark:

1. score cartridges using plan priors or previous winners
2. keep only the top `2` to `4`
3. evaluate only those on calibration examples

This will reduce:

- swap count
- calibration wall-clock
- total system load

## Related code

- `scripts/vllm_multiplex_server.py`
- `scripts/multiplex_cache.py`
- `scripts/evaluate_original_vs_multiplex.py`
- `scripts/run_boundary_research.py`
