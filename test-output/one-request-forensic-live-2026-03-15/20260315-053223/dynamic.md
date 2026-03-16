# Dynamic evaluation report

- model: `qwen35-dynamic-2gpu-16k`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 5
- overall accuracy: 0.00%
- overall coherence: 0.00%
- overall error rate: 100.00%
- average request time: 0.000s
- average sample time: 0.008s
- p95 sample time: 0.010s
- swap count: 0
- cartridge transition rate: 0.00%
- avg active expert bytes: 23506137907.2
- avg active expert count: 3736.2
- refresh suggested rate: 0.00%
- avg router inactive ratio: 0.00%
- unique slices used: 81

## Runtime identity

- server URL: `http://127.0.0.1:18111`
- host: `127.0.0.1`
- port: `18111`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/reap-expert-swap/test-output/disagreement-backfill-20260312/combined-backfill-fullrerank.plan.json`
- plan mode: `dynamic_core_specialist`

## Plan identity

- plan path: `/Users/sero/ai/reap-expert-swap/test-output/disagreement-backfill-20260312/combined-backfill-fullrerank.plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `26165525811`

## Request-level dynamic evidence

- `gsm8k::d3d4ef1085aa3a51`: request `6e01691e1d69bbb3`, swap request `None`, active set `368b0bd85009e64c`, plan `n/a`, router miss request `n/a`
- `mmlu::high_school_us_history::78043c7a88efdb6b`: request `727b897e1a08ab48`, swap request `None`, active set `77febee0e197a619`, plan `n/a`, router miss request `n/a`
- `winogrande::378822cb7a4c18b9`: request `0ca99fa03a6de35f`, swap request `None`, active set `febf7cd263a14e1f`, plan `n/a`, router miss request `n/a`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `None`, active set `3a494f78c235b326`, plan `n/a`, router miss request `n/a`
- `hellaswag::24952`: request `2dd64578aea841a1`, swap request `None`, active set `52ea0ac1bd50e911`, plan `n/a`, router miss request `n/a`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 0.00%
- errors: 1 (100.00%)
- parse errors: 0 (0.00%)
- benchmark time: 0.008s
- avg request time: 0.000s
- avg sample time: 0.008s
- p95 sample time: 0.008s
- avg swap time: 0.000s
- avg active expert bytes: 23492296704.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 0.00%
- coherence: 0.00%
- errors: 1 (100.00%)
- parse errors: 0 (0.00%)
- benchmark time: 0.008s
- avg request time: 0.000s
- avg sample time: 0.008s
- p95 sample time: 0.008s
- avg swap time: 0.000s
- avg active expert bytes: 23523753984.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 0.00%
- coherence: 0.00%
- errors: 1 (100.00%)
- parse errors: 0 (0.00%)
- benchmark time: 0.008s
- avg request time: 0.000s
- avg sample time: 0.008s
- p95 sample time: 0.008s
- avg swap time: 0.000s
- avg active expert bytes: 23523753984.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 0.00%
- coherence: 0.00%
- errors: 1 (100.00%)
- parse errors: 0 (0.00%)
- benchmark time: 0.008s
- avg request time: 0.000s
- avg sample time: 0.008s
- p95 sample time: 0.008s
- avg swap time: 0.000s
- avg active expert bytes: 23498588160.0
- refresh suggested rate: 0.00%

### winogrande
- accuracy: 0.00%
- coherence: 0.00%
- errors: 1 (100.00%)
- parse errors: 0 (0.00%)
- benchmark time: 0.010s
- avg request time: 0.000s
- avg sample time: 0.010s
- p95 sample time: 0.010s
- avg swap time: 0.000s
- avg active expert bytes: 23492296704.0
- refresh suggested rate: 0.00%

## Retention

- retained metrics status: `invalid_unmatched_baseline`
- baseline mismatch reasons: sample_count_per_benchmark, calibration_count_per_benchmark, result_signatures
- overall accuracy retained: 0.00%
- overall coherence retained: 0.00%
- overall quality loss: 90.00%
- worst benchmark accuracy drop: 1.0000

- arc_challenge: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- gsm8k: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- hellaswag: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- mmlu: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- winogrande: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 0.5000
