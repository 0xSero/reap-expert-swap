# Dynamic evaluation report

- model: `qwen35-masks-only`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 24.00%
- overall coherence: 56.00%
- overall error rate: 0.00%
- average request time: 6.472s
- average sample time: 6.728s
- p95 sample time: 25.914s
- swap count: 25
- cold swap count: 0
- warm swap count: 25
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.152s
- avg warm control-plane swap time: 0.054s
- avg warm change swap time: 0.152s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 6.728s
- avg completion tokens/s: 5.783
- avg warm completion tokens/s: 5.783
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 100.00%
- avg router inactive ratio: 59.60%
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
- endpoint time: `0.231406`

## Plan identity

- plan path: `/Users/sero/ai/autoresearch/assets/strict30-v2-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/strict30-masks-only-20260316-110053/runtime-readiness-identity.json`
- readiness service: `masks-only-test`
- readiness host: `127.0.0.1`
- readiness port: `18362`
- readiness plan file: `assets/strict30-v2-plan.json`

## Request-level dynamic evidence

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `c5a11e501daf30d1`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `2440ab84195997ca`
  - phase `warm_change`, copied `0`, zeroed `0`, added `400`, removed `400`, inactive ratio `0.6155`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `920a2c9ac2729eb5`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_change`, copied `0`, zeroed `0`, added `288`, removed `288`, inactive ratio `0.5049`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `ae0b8dd0d320bcf6`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_change`, copied `0`, zeroed `0`, added `272`, removed `272`, inactive ratio `0.6134`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `10612a9da4419aeb`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_change`, copied `0`, zeroed `0`, added `224`, removed `224`, inactive ratio `0.6831`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `5713c275ed8c9db7`, plan `/tmp/strict30-masks-only-test.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_change`, copied `0`, zeroed `0`, added `208`, removed `208`, inactive ratio `0.6466`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 3 (60.00%)
- benchmark time: 9.444s
- avg request time: 1.665s
- avg sample time: 1.889s
- p95 sample time: 2.209s
- avg swap time: 0.146s
- avg warm swap time: 0.146s
- avg warm change/reuse swap time: 0.146s / 0.000s
- avg completion tokens/s: 4.908
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### gsm8k
- accuracy: 0.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 119.006s
- avg request time: 23.554s
- avg sample time: 23.801s
- p95 sample time: 26.393s
- avg swap time: 0.156s
- avg warm swap time: 0.156s
- avg warm change/reuse swap time: 0.156s / 0.000s
- avg completion tokens/s: 9.905
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### hellaswag
- accuracy: 40.00%
- coherence: 80.00%
- errors: 0 (0.00%)
- parse errors: 1 (20.00%)
- benchmark time: 22.135s
- avg request time: 4.117s
- avg sample time: 4.427s
- p95 sample time: 13.826s
- avg swap time: 0.162s
- avg warm swap time: 0.162s
- avg warm change/reuse swap time: 0.162s / 0.000s
- avg completion tokens/s: 3.438
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### mmlu
- accuracy: 40.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 3 (60.00%)
- benchmark time: 10.116s
- avg request time: 1.772s
- avg sample time: 2.023s
- p95 sample time: 2.300s
- avg swap time: 0.147s
- avg warm swap time: 0.147s
- avg warm change/reuse swap time: 0.147s / 0.000s
- avg completion tokens/s: 4.438
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### winogrande
- accuracy: 40.00%
- coherence: 80.00%
- errors: 0 (0.00%)
- parse errors: 1 (20.00%)
- benchmark time: 7.491s
- avg request time: 1.252s
- avg sample time: 1.498s
- p95 sample time: 1.583s
- avg swap time: 0.151s
- avg warm swap time: 0.151s
- avg warm change/reuse swap time: 0.151s / 0.000s
- avg completion tokens/s: 6.226
- rows same-signature/zero-copy: 0/5
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

