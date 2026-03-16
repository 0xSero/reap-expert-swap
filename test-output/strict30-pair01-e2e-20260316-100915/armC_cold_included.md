# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair01`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 20.00%
- overall coherence: 56.00%
- overall error rate: 0.00%
- average request time: 4.958s
- average sample time: 5.375s
- p95 sample time: 22.848s
- swap count: 25
- cold swap count: 1
- warm swap count: 24
- avg cold swap time: 0.408s
- avg cold control-plane swap time: 0.068s
- avg warm swap time: 0.307s
- avg warm control-plane swap time: 0.067s
- avg warm change swap time: 0.307s
- avg warm reuse swap time: 0.000s
- avg warm sample time: 5.529s
- avg completion tokens/s: 8.162
- avg warm completion tokens/s: 8.239
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
  - phase `cold_bootstrap`, copied `1258291200`, zeroed `1258291200`, added `400`, removed `400`, inactive ratio `0.6147`
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
- benchmark time: 7.389s
- avg request time: 1.086s
- avg sample time: 1.478s
- p95 sample time: 1.568s
- avg swap time: 0.299s
- avg warm swap time: 0.299s
- avg warm change/reuse swap time: 0.299s / 0.000s
- avg completion tokens/s: 7.409
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
- benchmark time: 104.550s
- avg request time: 20.525s
- avg sample time: 20.910s
- p95 sample time: 22.867s
- avg swap time: 0.283s
- avg warm swap time: 0.283s
- avg warm change/reuse swap time: 0.283s / 0.000s
- avg completion tokens/s: 11.377
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
- benchmark time: 7.929s
- avg request time: 1.124s
- avg sample time: 1.586s
- p95 sample time: 1.684s
- avg swap time: 0.339s
- avg warm swap time: 0.322s
- avg warm change/reuse swap time: 0.322s / 0.000s
- avg completion tokens/s: 6.581
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 1/4
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### mmlu
- accuracy: 20.00%
- coherence: 20.00%
- errors: 0 (0.00%)
- parse errors: 4 (80.00%)
- benchmark time: 7.759s
- avg request time: 1.134s
- avg sample time: 1.552s
- p95 sample time: 1.697s
- avg swap time: 0.313s
- avg warm swap time: 0.313s
- avg warm change/reuse swap time: 0.313s / 0.000s
- avg completion tokens/s: 6.963
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
- benchmark time: 6.759s
- avg request time: 0.920s
- avg sample time: 1.352s
- p95 sample time: 1.412s
- avg swap time: 0.323s
- avg warm swap time: 0.323s
- avg warm change/reuse swap time: 0.323s / 0.000s
- avg completion tokens/s: 8.479
- rows same-signature/zero-copy: 0/0
- cold/warm swaps: 0/5
- dynamic signatures: 5
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

