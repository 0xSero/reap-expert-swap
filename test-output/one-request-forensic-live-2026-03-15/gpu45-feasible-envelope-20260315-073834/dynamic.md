# Dynamic evaluation report

- model: `qwen35-multiplex-offload-2gpu-3k`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 5
- overall accuracy: 80.00%
- overall coherence: 100.00%
- overall error rate: 0.00%
- average request time: 4.772s
- average sample time: 13.940s
- p95 sample time: 56.200s
- swap count: 5
- cartridge transition rate: 0.00%
- avg active expert bytes: 23506137907.2
- avg active expert count: 3736.2
- refresh suggested rate: 20.00%
- avg router inactive ratio: 14.52%
- unique slices used: 81

## Runtime identity

- server URL: `http://127.0.0.1:18351`
- host: `127.0.0.1`
- port: `18351`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/reap-expert-swap/test-output/disagreement-backfill-20260312/combined-backfill-fullrerank.plan.json`
- plan mode: `dynamic_core_specialist`

## Plan identity

- plan path: `/Users/sero/ai/reap-expert-swap/test-output/disagreement-backfill-20260312/combined-backfill-fullrerank.plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `26165525811`

## Request-level dynamic evidence

- `gsm8k::d3d4ef1085aa3a51`: request `6e01691e1d69bbb3`, swap request `6e01691e1d69bbb3`, active set `368b0bd85009e64c`, plan `/tmp/forensic-gpu45-gpu45-feasible-envelope-20260315-073834.plan.json`, router miss request `6e01691e1d69bbb3`
- `mmlu::high_school_us_history::78043c7a88efdb6b`: request `727b897e1a08ab48`, swap request `727b897e1a08ab48`, active set `77febee0e197a619`, plan `/tmp/forensic-gpu45-gpu45-feasible-envelope-20260315-073834.plan.json`, router miss request `727b897e1a08ab48`
- `winogrande::378822cb7a4c18b9`: request `0ca99fa03a6de35f`, swap request `0ca99fa03a6de35f`, active set `febf7cd263a14e1f`, plan `/tmp/forensic-gpu45-gpu45-feasible-envelope-20260315-073834.plan.json`, router miss request `0ca99fa03a6de35f`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `3a494f78c235b326`, plan `/tmp/forensic-gpu45-gpu45-feasible-envelope-20260315-073834.plan.json`, router miss request `5d2a4cc08feb491e`
- `hellaswag::24952`: request `2dd64578aea841a1`, swap request `2dd64578aea841a1`, active set `52ea0ac1bd50e911`, plan `/tmp/forensic-gpu45-gpu45-feasible-envelope-20260315-073834.plan.json`, router miss request `2dd64578aea841a1`

## Benchmarks

### arc_challenge
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 3.650s
- avg request time: 1.759s
- avg sample time: 3.650s
- p95 sample time: 3.650s
- avg swap time: 1.397s
- avg active expert bytes: 23492296704.0
- refresh suggested rate: 0.00%

### gsm8k
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 56.200s
- avg request time: 17.740s
- avg sample time: 56.200s
- p95 sample time: 56.200s
- avg swap time: 37.977s
- avg active expert bytes: 23523753984.0
- refresh suggested rate: 0.00%

### hellaswag
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 3.369s
- avg request time: 1.609s
- avg sample time: 3.369s
- p95 sample time: 3.369s
- avg swap time: 1.273s
- avg active expert bytes: 23523753984.0
- refresh suggested rate: 0.00%

### mmlu
- accuracy: 0.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 3.635s
- avg request time: 1.802s
- avg sample time: 3.635s
- p95 sample time: 3.635s
- avg swap time: 1.428s
- avg active expert bytes: 23498588160.0
- refresh suggested rate: 100.00%

### winogrande
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 2.846s
- avg request time: 0.952s
- avg sample time: 2.846s
- p95 sample time: 2.846s
- avg swap time: 1.322s
- avg active expert bytes: 23492296704.0
- refresh suggested rate: 0.00%

## Retention

- retained metrics status: `invalid_unmatched_baseline`
- baseline mismatch reasons: sample_count_per_benchmark, calibration_count_per_benchmark, result_signatures
- overall accuracy retained: 89.00%
- overall coherence retained: 100.00%
- overall quality loss: 10.00%
- worst benchmark accuracy drop: 1.0000

- arc_challenge: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
- gsm8k: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
- hellaswag: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
- mmlu: accuracy retained 0.00%, coherence retained 100.00%, accuracy drop 1.0000
- winogrande: accuracy retained 200.00%, coherence retained 100.00%, accuracy drop 0.0000
