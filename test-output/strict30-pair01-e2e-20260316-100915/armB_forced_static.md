# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair01`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 24.00%
- overall coherence: 60.00%
- overall error rate: 0.00%
- average request time: 2.791s
- average sample time: 2.938s
- p95 sample time: 22.643s
- swap count: 25
- cold swap count: 1
- warm swap count: 24
- avg cold swap time: 0.354s
- avg cold control-plane swap time: 0.058s
- avg warm swap time: 0.050s
- avg warm control-plane swap time: 0.050s
- avg warm change swap time: 0.000s
- avg warm reuse swap time: 0.050s
- avg warm sample time: 2.991s
- avg completion tokens/s: 7.765
- avg warm completion tokens/s: 7.817
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 4.00%
- avg router inactive ratio: 2.45%
- dynamic signature count: 1
- dynamic signature transitions: 0
- rows with same signature: 24
- rows with zero-copy swap: 24
- nonzero swap copy rows: 1
- nonzero swap add rows: 1
- unique slices used: 80

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

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `0c914cf2d2e02cdd`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `2440ab84195997ca`
  - phase `cold_bootstrap`, copied `1006632960`, zeroed `1006632960`, added `320`, removed `320`, inactive ratio `0.6125`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `0c914cf2d2e02cdd`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `0c914cf2d2e02cdd`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `0c914cf2d2e02cdd`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `0c914cf2d2e02cdd`, plan `/tmp/autoresearch-strict30-pair01.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 3 (60.00%)
- benchmark time: 5.874s
- avg request time: 1.046s
- avg sample time: 1.175s
- p95 sample time: 1.245s
- avg swap time: 0.050s
- avg warm swap time: 0.050s
- avg warm change/reuse swap time: 0.000s / 0.050s
- avg completion tokens/s: 7.502
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 20.00%
- coherence: 80.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 49.436s
- avg request time: 9.735s
- avg sample time: 9.887s
- p95 sample time: 22.696s
- avg swap time: 0.052s
- avg warm swap time: 0.052s
- avg warm change/reuse swap time: 0.000s / 0.052s
- avg completion tokens/s: 9.063
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 20.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 3 (60.00%)
- benchmark time: 6.709s
- avg request time: 1.136s
- avg sample time: 1.342s
- p95 sample time: 1.660s
- avg swap time: 0.111s
- avg warm swap time: 0.050s
- avg warm change/reuse swap time: 0.000s / 0.050s
- avg completion tokens/s: 6.705
- rows same-signature/zero-copy: 4/4
- cold/warm swaps: 1/4
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 20.00%

### mmlu
- accuracy: 20.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 3 (60.00%)
- benchmark time: 6.202s
- avg request time: 1.106s
- avg sample time: 1.240s
- p95 sample time: 1.391s
- avg swap time: 0.049s
- avg warm swap time: 0.049s
- avg warm change/reuse swap time: 0.000s / 0.049s
- avg completion tokens/s: 6.975
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### winogrande
- accuracy: 60.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 5.226s
- avg request time: 0.933s
- avg sample time: 1.045s
- p95 sample time: 1.077s
- avg swap time: 0.051s
- avg warm swap time: 0.051s
- avg warm change/reuse swap time: 0.000s / 0.051s
- avg completion tokens/s: 8.581
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

