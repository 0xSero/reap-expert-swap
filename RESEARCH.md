# REAP Expert-Swap Research Summary

## Objective

Serve the original sparse model weights with a much smaller resident BF16 footprint by keeping a resident expert floor in memory and dynamically materializing a prompt-conditioned tail at runtime.

## Current research direction

The repo currently centers on:

1. observer-driven expert scoring
2. resident-floor construction
3. dynamic specialist assembly
4. baseline-vs-dynamic evaluation
5. router activity profiling
6. iterative floor refinement

## Main findings so far

- expert-granularity delta swaps are viable
- a profile-derived floor is materially better than blind low-budget selection
- multi-turn evaluation must be calibrated carefully or the protocol itself becomes the dominant loss source
- the hardest remaining problem is preserving the true expert core under a smaller resident budget

## What this public export preserves

- core runtime and evaluation logic
- public-safe architecture and protocol documentation
- public-safe regression tests
- a curated technical record of the system shape

## What this export intentionally omits

- personal activation corpora
- private host/infrastructure wiring
- large raw experiment artifacts
- internal operator-specific mission scaffolding


## Full project history

- `docs/research_history_20260312.md`
