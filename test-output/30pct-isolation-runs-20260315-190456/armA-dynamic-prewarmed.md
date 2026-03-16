# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair45`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 32.00%
- overall coherence: 64.00%
- overall error rate: 0.00%
- average request time: 6.701s
- average sample time: 6.873s
- p95 sample time: 26.149s
- swap count: 25
- cold swap count: 0
- warm swap count: 25
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.055s
- avg warm control-plane swap time: 0.055s
- avg warm change swap time: 0.000s
- avg warm reuse swap time: 0.055s
- avg warm sample time: 6.873s
- avg completion tokens/s: 6.359
- avg warm completion tokens/s: 6.359
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 0.00%
- avg router inactive ratio: 0.00%
- dynamic signature count: 1
- dynamic signature transitions: 0
- rows with same signature: 25
- rows with zero-copy swap: 25
- nonzero swap copy rows: 0
- nonzero swap add rows: 0
- unique slices used: 80

## Runtime identity

- server URL: `http://127.0.0.1:18365`
- host: `127.0.0.1`
- port: `18365`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`

## Warm-start metadata

- status: `success`
- request id: `warmstart::7`
- active set signature: `6ebe4366c72f4528`
- warm-start only: `True`
- endpoint time: `63.487460`

## Plan identity

- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/30pct-isolation-runs-20260315-190456/runtime-readiness-identity.json`
- readiness service: `pair45`
- readiness host: `127.0.0.1`
- readiness port: `18365`
- readiness plan file: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`

## Request-level dynamic evidence

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-pair45-20260315-190456.plan.json`, router miss request `2440ab84195997ca`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-pair45-20260315-190456.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-pair45-20260315-190456.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-pair45-20260315-190456.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-pair45-20260315-190456.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 0.00%
- errors: 0 (0.00%)
- parse errors: 5 (100.00%)
- benchmark time: 7.344s
- avg request time: 1.332s
- avg sample time: 1.469s
- p95 sample time: 1.497s
- avg swap time: 0.052s
- avg warm swap time: 0.052s
- avg warm change/reuse swap time: 0.000s / 0.052s
- avg completion tokens/s: 6.009
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 0.00%
- coherence: 40.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 130.487s
- avg request time: 25.950s
- avg sample time: 26.097s
- p95 sample time: 26.175s
- avg swap time: 0.054s
- avg warm swap time: 0.054s
- avg warm change/reuse swap time: 0.000s / 0.054s
- avg completion tokens/s: 9.865
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 40.00%
- coherence: 80.00%
- errors: 0 (0.00%)
- parse errors: 1 (20.00%)
- benchmark time: 19.020s
- avg request time: 3.599s
- avg sample time: 3.804s
- p95 sample time: 12.854s
- avg swap time: 0.053s
- avg warm swap time: 0.053s
- avg warm change/reuse swap time: 0.000s / 0.053s
- avg completion tokens/s: 4.433
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 80.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 7.818s
- avg request time: 1.347s
- avg sample time: 1.564s
- p95 sample time: 1.707s
- avg swap time: 0.055s
- avg warm swap time: 0.055s
- avg warm change/reuse swap time: 0.000s / 0.055s
- avg completion tokens/s: 5.233
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### winogrande
- accuracy: 40.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 7.166s
- avg request time: 1.279s
- avg sample time: 1.433s
- p95 sample time: 1.466s
- avg swap time: 0.059s
- avg warm swap time: 0.059s
- avg warm change/reuse swap time: 0.000s / 0.059s
- avg completion tokens/s: 6.256
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

