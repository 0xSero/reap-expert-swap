# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair01`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 25
- overall accuracy: 40.00%
- overall coherence: 56.00%
- overall error rate: 0.00%
- average request time: 5.959s
- average sample time: 6.123s
- p95 sample time: 23.753s
- swap count: 25
- cold swap count: 0
- warm swap count: 25
- avg cold swap time: 0.000s
- avg cold control-plane swap time: 0.000s
- avg warm swap time: 0.058s
- avg warm control-plane swap time: 0.058s
- avg warm change swap time: 0.000s
- avg warm reuse swap time: 0.058s
- avg warm sample time: 6.123s
- avg completion tokens/s: 7.629
- avg warm completion tokens/s: 7.629
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

- server URL: `http://127.0.0.1:18361`
- host: `127.0.0.1`
- port: `18361`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`

## Warm-start metadata

- status: `success`
- request id: `warmstart::7`
- active set signature: `6ebe4366c72f4528`
- warm-start only: `True`
- endpoint time: `62.953639`

## Plan identity

- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/30pct-isolation-e2e-20260315-190134/runtime-readiness-identity.json`
- readiness service: `pair01-isolation`
- readiness host: `127.0.0.1`
- readiness port: `18361`
- readiness plan file: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`

## Request-level dynamic evidence

- `hellaswag::45609`: request `2440ab84195997ca`, swap request `2440ab84195997ca`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-8361.plan.json`, router miss request `2440ab84195997ca`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `winogrande::ae689fe59c39d3ae`: request `3124d34b90661dca`, swap request `3124d34b90661dca`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-8361.plan.json`, router miss request `3124d34b90661dca`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-8361.plan.json`, router miss request `5d2a4cc08feb491e`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `gsm8k::ba16b653a431e439`: request `f61e659164814c21`, swap request `f61e659164814c21`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-8361.plan.json`, router miss request `f61e659164814c21`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`
- `gsm8k::1ddec3ad4562a490`: request `092d421fb4060cf6`, swap request `092d421fb4060cf6`, active set `6ebe4366c72f4528`, plan `/tmp/30pct-isolation-8361.plan.json`, router miss request `092d421fb4060cf6`
  - phase `warm_reuse`, copied `0`, zeroed `0`, added `0`, removed `0`, inactive ratio `0.0000`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 0.00%
- errors: 0 (0.00%)
- parse errors: 5 (100.00%)
- benchmark time: 6.306s
- avg request time: 1.123s
- avg sample time: 1.261s
- p95 sample time: 1.334s
- avg swap time: 0.055s
- avg warm swap time: 0.055s
- avg warm change/reuse swap time: 0.000s / 0.055s
- avg completion tokens/s: 7.163
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 20.00%
- coherence: 20.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 117.111s
- avg request time: 23.286s
- avg sample time: 23.422s
- p95 sample time: 24.078s
- avg swap time: 0.056s
- avg warm swap time: 0.056s
- avg warm change/reuse swap time: 0.000s / 0.056s
- avg completion tokens/s: 10.998
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 60.00%
- coherence: 60.00%
- errors: 0 (0.00%)
- parse errors: 2 (40.00%)
- benchmark time: 17.618s
- avg request time: 3.326s
- avg sample time: 3.524s
- p95 sample time: 12.183s
- avg swap time: 0.064s
- avg warm swap time: 0.064s
- avg warm change/reuse swap time: 0.000s / 0.064s
- avg completion tokens/s: 5.219
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
- benchmark time: 6.506s
- avg request time: 1.096s
- avg sample time: 1.301s
- p95 sample time: 1.387s
- avg swap time: 0.053s
- avg warm swap time: 0.053s
- avg warm change/reuse swap time: 0.000s / 0.053s
- avg completion tokens/s: 6.447
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
- benchmark time: 5.541s
- avg request time: 0.962s
- avg sample time: 1.108s
- p95 sample time: 1.134s
- avg swap time: 0.062s
- avg warm swap time: 0.062s
- avg warm change/reuse swap time: 0.000s / 0.062s
- avg completion tokens/s: 8.317
- rows same-signature/zero-copy: 5/5
- cold/warm swaps: 0/5
- dynamic signatures: 1
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 0.00%

## Retention

- retained metrics status: `invalid_unmatched_baseline`
- baseline mismatch reasons: sample_count_per_benchmark, result_signatures
- overall accuracy retained: 50.00%
- overall coherence retained: 56.00%
- overall quality loss: 40.00%
- worst benchmark accuracy drop: 1.0000

- arc_challenge: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- gsm8k: accuracy retained 20.00%, coherence retained 20.00%, accuracy drop 0.8000
- hellaswag: accuracy retained 60.00%, coherence retained 60.00%, accuracy drop 0.4000
- mmlu: accuracy retained n/a, coherence retained 100.00%, accuracy drop 0.0000
- winogrande: accuracy retained 40.00%, coherence retained 100.00%, accuracy drop 0.6000
