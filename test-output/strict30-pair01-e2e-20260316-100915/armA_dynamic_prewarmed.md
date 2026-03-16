# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair01`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 20.00%
- overall coherence: 56.00%
- overall error rate: 0.00%
- average request time: 7.656s
- average sample time: 8.054s
- p95 sample time: 22.815s
- swap count: 25
- cold swap count: 0
- warm swap count: 25
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.303s
- avg warm control-plane swap time: 0.060s
- avg warm change swap time: 0.303s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 8.054s
- avg completion tokens/s: 7.512
- avg warm completion tokens/s: 7.512
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 100.00%
- avg router inactive ratio: 59.78%
- dynamic signature count: 25
- dynamic signature transitions: 24
- rows with same signature: 0
- rows with zero-copy swap: 0
- nonzero swap copy rows: 25
- nonzero swap add rows: 25
- unique slices used: 120

## Runtime identity

- server URL: `http://127.0.0.1:18361`
- host: `127.0.0.1`
- port: `18361`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`

## Warm-start metadata

- status: `success`
- request id: `warmstart::7`
- active set signature: `0c914cf2d2e02cdd`
- warm-start only: `True`
- endpoint time: `59.802045`

## Plan identity

- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/strict30-pair01-e2e-20260316-100915/runtime-readiness-identity.json`
- readiness service: `pair01-isolation`
- readiness host: `127.0.0.1`
- readiness port: `18361`
- readiness plan file: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`

## Request-level dynamic evidence

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `c5a11e501daf30d1`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `2440ab84195997ca`
  - phase `warm_change`, copied `1258291200`, zeroed `1258291200`, added `400`, removed `400`, inactive ratio `0.6147`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `920a2c9ac2729eb5`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_change`, copied `905969664`, zeroed `905969664`, added `288`, removed `288`, inactive ratio `0.5041`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `ae0b8dd0d320bcf6`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_change`, copied `855638016`, zeroed `855638016`, added `272`, removed `272`, inactive ratio `0.6137`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `10612a9da4419aeb`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_change`, copied `704643072`, zeroed `704643072`, added `224`, removed `224`, inactive ratio `0.6762`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `5713c275ed8c9db7`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_change`, copied `654311424`, zeroed `654311424`, added `208`, removed `208`, inactive ratio `0.6460`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 3 (60.00%)
- benchmark time: 8.014s
- avg request time: 1.237s
- avg sample time: 1.603s
- p95 sample time: 2.294s
- avg swap time: 0.290s
- avg warm swap time: 0.290s
- avg warm change/reuse swap time: 0.290s / 0.000s
- avg completion tokens/s: 6.916
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### gsm8k
- accuracy: 0.00%
- coherence: 60.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 103.621s
- avg request time: 20.348s
- avg sample time: 20.724s
- p95 sample time: 22.815s
- avg swap time: 0.282s
- avg warm swap time: 0.282s
- avg warm change/reuse swap time: 0.282s / 0.000s
- avg completion tokens/s: 11.438
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### hellaswag
- accuracy: 40.00%
- coherence: 80.00%
- errors: 0 (0.00%)
- parse errors: 1 (20.00%)
- benchmark time: 74.270s
- avg request time: 14.412s
- avg sample time: 14.854s
- p95 sample time: 67.678s
- avg swap time: 0.319s
- avg warm swap time: 0.319s
- avg warm change/reuse swap time: 0.319s / 0.000s
- avg completion tokens/s: 5.044
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### mmlu
- accuracy: 20.00%
- coherence: 20.00%
- errors: 0 (0.00%)
- parse errors: 4 (80.00%)
- benchmark time: 7.709s
- avg request time: 1.132s
- avg sample time: 1.542s
- p95 sample time: 1.694s
- avg swap time: 0.319s
- avg warm swap time: 0.319s
- avg warm change/reuse swap time: 0.319s / 0.000s
- avg completion tokens/s: 6.950
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### winogrande
- accuracy: 40.00%
- coherence: 80.00%
- errors: 0 (0.00%)
- parse errors: 1 (20.00%)
- benchmark time: 7.730s
- avg request time: 1.150s
- avg sample time: 1.546s
- p95 sample time: 1.900s
- avg swap time: 0.307s
- avg warm swap time: 0.307s
- avg warm change/reuse swap time: 0.307s / 0.000s
- avg completion tokens/s: 7.210
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

