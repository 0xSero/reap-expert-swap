# Dynamic evaluation report

- model: `qwen35-full-expert`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 88.00%
- overall coherence: 100.00%
- overall error rate: 0.00%
- average request time: 4.564s
- average sample time: 4.802s
- p95 sample time: 14.179s
- swap count: 25
- cold swap count: 0
- warm swap count: 25
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.091s
- avg warm control-plane swap time: 0.069s
- avg warm change swap time: 0.091s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 4.802s
- avg completion tokens/s: 4.388
- avg warm completion tokens/s: 4.388
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
- nonzero swap add rows: 25
- unique slices used: 120

## Runtime identity

- server URL: `http://127.0.0.1:18362`
- host: `127.0.0.1`
- port: `18362`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`

## Warm-start metadata

- status: `success`
- request id: `warmstart::7`
- active set signature: `0c914cf2d2e02cdd`
- warm-start only: `True`
- endpoint time: `0.089347`

## Plan identity

- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/strict30-full-expert-20260316-115953/runtime-readiness-identity.json`
- readiness service: `full-expert-test`
- readiness host: `127.0.0.1`
- readiness port: `18362`
- readiness plan file: `assets/strict30-v2-plan.json`

## Request-level dynamic evidence

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `c5a11e501daf30d1`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `2440ab84195997ca`
  - phase `warm_change`, copied `0`, zeroed `0`, added `400`, removed `400`, inactive ratio `0.0000`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `920a2c9ac2729eb5`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_change`, copied `0`, zeroed `0`, added `288`, removed `288`, inactive ratio `0.0000`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `ae0b8dd0d320bcf6`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_change`, copied `0`, zeroed `0`, added `272`, removed `272`, inactive ratio `0.0000`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `10612a9da4419aeb`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_change`, copied `0`, zeroed `0`, added `224`, removed `224`, inactive ratio `0.0000`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `5713c275ed8c9db7`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_change`, copied `0`, zeroed `0`, added `208`, removed `208`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 13.306s
- avg request time: 2.413s
- avg sample time: 2.661s
- p95 sample time: 3.078s
- avg swap time: 0.101s
- avg warm swap time: 0.101s
- avg warm change/reuse swap time: 0.101s / 0.000s
- avg completion tokens/s: 3.028
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
- benchmark time: 53.312s
- avg request time: 10.438s
- avg sample time: 10.662s
- p95 sample time: 14.179s
- avg swap time: 0.088s
- avg warm swap time: 0.088s
- avg warm change/reuse swap time: 0.088s / 0.000s
- avg completion tokens/s: 10.265
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
- benchmark time: 28.739s
- avg request time: 5.480s
- avg sample time: 5.748s
- p95 sample time: 14.695s
- avg swap time: 0.094s
- avg warm swap time: 0.094s
- avg warm change/reuse swap time: 0.094s / 0.000s
- avg completion tokens/s: 1.848
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 80.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 16.368s
- avg request time: 3.056s
- avg sample time: 3.274s
- p95 sample time: 3.932s
- avg swap time: 0.087s
- avg warm swap time: 0.087s
- avg warm change/reuse swap time: 0.087s / 0.000s
- avg completion tokens/s: 2.553
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
- benchmark time: 8.321s
- avg request time: 1.435s
- avg sample time: 1.664s
- p95 sample time: 1.805s
- avg swap time: 0.085s
- avg warm swap time: 0.085s
- avg warm change/reuse swap time: 0.085s / 0.000s
- avg completion tokens/s: 4.244
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

