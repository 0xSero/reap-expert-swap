# Baseline evaluation report

- model: `qwen35-bf16-full`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 88.00%
- overall coherence: 100.00%
- overall error rate: 0.00%
- average request time: 9.354s
- average sample time: 9.354s
- p95 sample time: 17.115s
- swap count: 0
- cold swap count: 0
- warm swap count: 0
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.000s
- avg warm control-plane swap time: 0.000s
- avg warm change swap time: 0.000s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 9.354s
- avg completion tokens/s: 3.170
- avg warm completion tokens/s: 3.170
- cartridge transition rate: 0.00%
- avg active expert bytes: 0.0
- avg active expert count: 0.0
- refresh suggested rate: 0.00%
- avg router inactive ratio: 0.00%
- dynamic signature count: 0
- dynamic signature transitions: 0
- rows with same signature: 0
- rows with zero-copy swap: 25
- nonzero swap copy rows: 0
- nonzero swap add rows: 0
- unique slices used: 0

## Runtime identity

- server URL: `http://127.0.0.1:18370`
- host: `127.0.0.1`
- port: `18370`
- concurrency mode: `n/a`
- plan path: `n/a`
- plan mode: `n/a`

## Request-level dynamic evidence

- `mmlu::high_school_us_history::78043c7a88efdb6b`: request `25d57b533722e515`, swap request `None`, active set `None`, plan `n/a`, router miss request `n/a`
  - phase `None`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `mmlu::high_school_biology::da945bc086fd50a8`: request `f6b59c9e9acbc61f`, swap request `None`, active set `None`, plan `n/a`, router miss request `n/a`
  - phase `None`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `mmlu::professional_law::9ecdf69beb5c0909`: request `dcbb1658fb255a6c`, swap request `None`, active set `None`, plan `n/a`, router miss request `n/a`
  - phase `None`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `mmlu::professional_law::b32e24ddf34be094`: request `7524892174a15de6`, swap request `None`, active set `None`, plan `n/a`, router miss request `n/a`
  - phase `None`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `mmlu::high_school_psychology::b7ab86e60e01bde4`: request `4c255ac3640f4d39`, swap request `None`, active set `None`, plan `n/a`, router miss request `n/a`
  - phase `None`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 33.785s
- avg request time: 6.757s
- avg sample time: 6.757s
- p95 sample time: 11.343s
- avg swap time: 0.000s
- avg warm swap time: 0.000s
- avg warm change/reuse swap time: 0.000s / 0.000s
- avg completion tokens/s: 2.667
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/0
- dynamic signatures: 0
- avg active expert bytes: 0.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 67.200s
- avg request time: 13.440s
- avg sample time: 13.440s
- p95 sample time: 16.774s
- avg swap time: 0.000s
- avg warm swap time: 0.000s
- avg warm change/reuse swap time: 0.000s / 0.000s
- avg completion tokens/s: 8.091
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/0
- dynamic signatures: 0
- avg active expert bytes: 0.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 27.766s
- avg request time: 5.553s
- avg sample time: 5.553s
- p95 sample time: 17.115s
- avg swap time: 0.000s
- avg warm swap time: 0.000s
- avg warm change/reuse swap time: 0.000s / 0.000s
- avg completion tokens/s: 2.645
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/0
- dynamic signatures: 0
- avg active expert bytes: 0.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 80.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 67.016s
- avg request time: 13.403s
- avg sample time: 13.403s
- p95 sample time: 50.239s
- avg swap time: 0.000s
- avg warm swap time: 0.000s
- avg warm change/reuse swap time: 0.000s / 0.000s
- avg completion tokens/s: 1.555
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/0
- dynamic signatures: 0
- avg active expert bytes: 0.0
- refresh suggested rate: 0.00%

### winogrande
- accuracy: 60.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 38.076s
- avg request time: 7.615s
- avg sample time: 7.615s
- p95 sample time: 13.763s
- avg swap time: 0.000s
- avg warm swap time: 0.000s
- avg warm change/reuse swap time: 0.000s / 0.000s
- avg completion tokens/s: 0.892
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/0
- dynamic signatures: 0
- avg active expert bytes: 0.0
- refresh suggested rate: 0.00%

