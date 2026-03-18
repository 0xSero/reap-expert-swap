# Dynamic evaluation report

- model: `qwen35-dynamic`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 88.00%
- overall coherence: 100.00%
- overall error rate: 0.00%
- average request time: 3.922s
- average sample time: 4.125s
- p95 sample time: 13.826s
- swap count: 25
- cold swap count: 1
- warm swap count: 24
- avg cold swap time: 0.091s
- avg cold control-plane swap time: 0.062s
- avg warm swap time: 0.076s
- avg warm control-plane swap time: 0.050s
- avg warm change swap time: 0.076s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 3.693s
- avg completion tokens/s: 5.118
- avg warm completion tokens/s: 5.311
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 0.00%
- avg router inactive ratio: 0.00%
- dynamic signature count: 25
- dynamic signature transitions: 24
- rows with same signature: 0
- rows with zero-copy swap: 25
- nonzero swap copy rows: 0
- nonzero swap add rows: 24
- unique slices used: 120

## Runtime identity

- concurrency mode: `serialized_single_flight`
- plan mode: `dynamic_core_specialist`

## Plan identity

- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Request-level dynamic evidence

- `hellaswag::45609`: active set `c5a11e501daf30d1`
  - phase `cold_shrink`, copied `0`, zeroed `0`, added `0`, removed `15280`, inactive ratio `0.0000`
- `winogrande::ae689fe59c39d3ae`: active set `920a2c9ac2729eb5`
  - phase `warm_change`, copied `0`, zeroed `0`, added `288`, removed `288`, inactive ratio `0.0000`
- `arc::Mercury_SC_LBS10666`: active set `ae0b8dd0d320bcf6`
  - phase `warm_change`, copied `0`, zeroed `0`, added `272`, removed `272`, inactive ratio `0.0000`
- `gsm8k::ba16b653a431e439`: active set `10612a9da4419aeb`
  - phase `warm_change`, copied `0`, zeroed `0`, added `224`, removed `224`, inactive ratio `0.0000`
- `gsm8k::1ddec3ad4562a490`: active set `5713c275ed8c9db7`
  - phase `warm_change`, copied `0`, zeroed `0`, added `208`, removed `208`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 9.697s
- avg request time: 1.741s
- avg sample time: 1.939s
- p95 sample time: 2.095s
- avg swap time: 0.075s
- avg warm swap time: 0.075s
- avg warm change/reuse swap time: 0.075s / 0.000s
- avg completion tokens/s: 4.036
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 50.668s
- avg request time: 9.935s
- avg sample time: 10.134s
- p95 sample time: 13.826s
- avg swap time: 0.075s
- avg warm swap time: 0.075s
- avg warm change/reuse swap time: 0.075s / 0.000s
- avg completion tokens/s: 10.874
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 23.494s
- avg request time: 4.494s
- avg sample time: 4.699s
- p95 sample time: 14.478s
- avg swap time: 0.082s
- avg warm swap time: 0.079s
- avg warm change/reuse swap time: 0.079s / 0.000s
- avg completion tokens/s: 2.847
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 1/4
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 80.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 11.209s
- avg request time: 2.027s
- avg sample time: 2.242s
- p95 sample time: 2.523s
- avg swap time: 0.076s
- avg warm swap time: 0.076s
- avg warm change/reuse swap time: 0.076s / 0.000s
- avg completion tokens/s: 3.532
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### winogrande
- accuracy: 60.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 8.045s
- avg request time: 1.416s
- avg sample time: 1.609s
- p95 sample time: 1.692s
- avg swap time: 0.073s
- avg warm swap time: 0.073s
- avg warm change/reuse swap time: 0.073s / 0.000s
- avg completion tokens/s: 4.301
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%
