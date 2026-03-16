# Dynamic evaluation report

- model: `qwen35-no-mask-test`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 24.00%
- overall coherence: 64.00%
- overall error rate: 0.00%
- average request time: 4.059s
- average sample time: 4.352s
- p95 sample time: 20.124s
- swap count: 25
- cold swap count: 0
- warm swap count: 25
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.221s
- avg warm control-plane swap time: 0.055s
- avg warm change swap time: 0.221s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 4.352s
- avg completion tokens/s: 6.842
- avg warm completion tokens/s: 6.842
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 0.00%
- avg router inactive ratio: 0.00%
- dynamic signature count: 25
- dynamic signature transitions: 24
- rows with same signature: 0
- rows with zero-copy swap: 0
- nonzero swap copy rows: 25
- nonzero swap add rows: 25
- unique slices used: 120

## Runtime identity

- server URL: `http://127.0.0.1:18363`
- host: `127.0.0.1`
- port: `18363`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`

## Warm-start metadata

- status: `success`
- request id: `warmstart::7`
- active set signature: `0c914cf2d2e02cdd`
- warm-start only: `True`
- endpoint time: `59.724868`

## Plan identity

- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Request-level dynamic evidence

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `c5a11e501daf30d1`, plan `/tmp/autoresearch-no-mask-manual.plan.json`, router miss request `2440ab84195997ca`
  - phase `warm_change`, copied `1258291200`, zeroed `1258291200`, added `400`, removed `400`, inactive ratio `0.0000`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `920a2c9ac2729eb5`, plan `/tmp/autoresearch-no-mask-manual.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_change`, copied `905969664`, zeroed `905969664`, added `288`, removed `288`, inactive ratio `0.0000`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `ae0b8dd0d320bcf6`, plan `/tmp/autoresearch-no-mask-manual.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_change`, copied `855638016`, zeroed `855638016`, added `272`, removed `272`, inactive ratio `0.0000`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `10612a9da4419aeb`, plan `/tmp/autoresearch-no-mask-manual.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_change`, copied `704643072`, zeroed `704643072`, added `224`, removed `224`, inactive ratio `0.0000`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `5713c275ed8c9db7`, plan `/tmp/autoresearch-no-mask-manual.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_change`, copied `654311424`, zeroed `654311424`, added `208`, removed `208`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 20.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 7.853s
- avg request time: 1.307s
- avg sample time: 1.571s
- p95 sample time: 1.703s
- avg swap time: 0.204s
- avg warm swap time: 0.204s
- avg warm change/reuse swap time: 0.204s / 0.000s
- avg completion tokens/s: 6.227
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 0.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 64.638s
- avg request time: 12.624s
- avg sample time: 12.928s
- p95 sample time: 20.699s
- avg swap time: 0.200s
- avg warm swap time: 0.200s
- avg warm change/reuse swap time: 0.200s / 0.000s
- avg completion tokens/s: 10.698
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 20.00%
- coherence: 60.00%
- errors: 0 (0.00%)
- parse errors: 2 (40.00%)
- benchmark time: 20.593s
- avg request time: 3.787s
- avg sample time: 4.119s
- p95 sample time: 12.897s
- avg swap time: 0.245s
- avg warm swap time: 0.245s
- avg warm change/reuse swap time: 0.245s / 0.000s
- avg completion tokens/s: 3.966
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 20.00%
- coherence: 20.00%
- errors: 0 (0.00%)
- parse errors: 4 (80.00%)
- benchmark time: 9.232s
- avg request time: 1.572s
- avg sample time: 1.846s
- p95 sample time: 2.126s
- avg swap time: 0.220s
- avg warm swap time: 0.220s
- avg warm change/reuse swap time: 0.220s / 0.000s
- avg completion tokens/s: 5.338
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### winogrande
- accuracy: 60.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 6.471s
- avg request time: 1.002s
- avg sample time: 1.294s
- p95 sample time: 1.325s
- avg swap time: 0.235s
- avg warm swap time: 0.235s
- avg warm change/reuse swap time: 0.235s / 0.000s
- avg completion tokens/s: 7.982
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

