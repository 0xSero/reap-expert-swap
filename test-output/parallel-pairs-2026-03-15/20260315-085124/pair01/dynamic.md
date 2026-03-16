# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair01`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 5
- overall accuracy: 80.00%
- overall coherence: 80.00%
- overall error rate: 0.00%
- average request time: 8.511s
- average sample time: 25.507s
- p95 sample time: 121.101s
- swap count: 5
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 100.00%
- avg router inactive ratio: 60.02%
- unique slices used: 80

## Runtime identity

- server URL: `http://127.0.0.1:18361`
- host: `127.0.0.1`
- port: `18361`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`

## Plan identity

- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/parallel-pairs-2026-03-15/20260315-085124/pair01/runtime-readiness-identity.json`
- readiness service: `pair01`
- readiness host: `127.0.0.1`
- readiness port: `18361`
- readiness plan file: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`

## Request-level dynamic evidence

- `gsm8k::d3d4ef1085aa3a51`: request `6e01691e1d69bbb3`, swap request `6e01691e1d69bbb3`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair01-20260315-085124.plan.json`, router miss request `6e01691e1d69bbb3`
- `mmlu::high_school_us_history::78043c7a88efdb6b`: request `727b897e1a08ab48`, swap request `727b897e1a08ab48`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair01-20260315-085124.plan.json`, router miss request `727b897e1a08ab48`
- `winogrande::378822cb7a4c18b9`: request `0ca99fa03a6de35f`, swap request `0ca99fa03a6de35f`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair01-20260315-085124.plan.json`, router miss request `0ca99fa03a6de35f`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair01-20260315-085124.plan.json`, router miss request `5d2a4cc08feb491e`
- `hellaswag::24952`: request `2dd64578aea841a1`, swap request `2dd64578aea841a1`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair01-20260315-085124.plan.json`, router miss request `2dd64578aea841a1`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 1.680s
- avg request time: 1.353s
- avg sample time: 1.680s
- p95 sample time: 1.680s
- avg swap time: 0.166s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### gsm8k
- accuracy: 100.00%
- coherence: 0.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 121.101s
- avg request time: 37.492s
- avg sample time: 121.101s
- p95 sample time: 121.101s
- avg swap time: 83.381s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### hellaswag
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 1.597s
- avg request time: 1.240s
- avg sample time: 1.597s
- p95 sample time: 1.597s
- avg swap time: 0.185s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### mmlu
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 1.736s
- avg request time: 1.332s
- avg sample time: 1.736s
- p95 sample time: 1.736s
- avg swap time: 0.228s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### winogrande
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 1.422s
- avg request time: 1.138s
- avg sample time: 1.422s
- p95 sample time: 1.422s
- avg swap time: 0.183s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

## Retention

- retained metrics status: `ok`
- overall accuracy retained: 100.00%
- overall coherence retained: 80.00%
- overall quality loss: 0.00%
- worst benchmark accuracy drop: 1.0000

- arc_challenge: accuracy retained 0.00%, coherence retained 100.00%, accuracy drop 1.0000
- gsm8k: accuracy retained 100.00%, coherence retained 0.00%, accuracy drop 0.0000
- hellaswag: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
- mmlu: accuracy retained 0.00%, coherence retained 100.00%, accuracy drop 0.0000
- winogrande: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
