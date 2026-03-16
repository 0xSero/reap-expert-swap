# Dynamic evaluation report

- model: `qwen35-dynamic-30pct-pair67`
- protocol: `single_turn` / `singleturn_v0`
- turns per sample: 1
- total samples: 5
- overall accuracy: 60.00%
- overall coherence: 60.00%
- overall error rate: 0.00%
- average request time: 9.981s
- average sample time: 27.055s
- p95 sample time: 126.246s
- swap count: 5
- cartridge transition rate: 0.00%
- avg active expert bytes: 16357785600.0
- avg active expert count: 2600.0
- refresh suggested rate: 100.00%
- avg router inactive ratio: 59.91%
- unique slices used: 80

## Runtime identity

- server URL: `http://127.0.0.1:18367`
- host: `127.0.0.1`
- port: `18367`
- concurrency mode: `serialized_single_flight`
- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`

## Plan identity

- plan path: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`
- plan mode: `dynamic_core_specialist`
- swappable budget bytes: `16757880012`

## Runtime readiness evidence

- source: `configured-runtime-identity`
- identity path: `/Users/sero/ai/autoresearch/test-output/parallel-pairs-2026-03-15/20260315-085124/pair67/runtime-readiness-identity.json`
- readiness service: `pair67`
- readiness host: `127.0.0.1`
- readiness port: `18367`
- readiness plan file: `/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json`

## Request-level dynamic evidence

- `gsm8k::d3d4ef1085aa3a51`: request `6e01691e1d69bbb3`, swap request `6e01691e1d69bbb3`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair67-20260315-085124.plan.json`, router miss request `6e01691e1d69bbb3`
- `mmlu::high_school_us_history::78043c7a88efdb6b`: request `727b897e1a08ab48`, swap request `727b897e1a08ab48`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair67-20260315-085124.plan.json`, router miss request `727b897e1a08ab48`
- `winogrande::378822cb7a4c18b9`: request `0ca99fa03a6de35f`, swap request `0ca99fa03a6de35f`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair67-20260315-085124.plan.json`, router miss request `0ca99fa03a6de35f`
- `arc::Mercury_SC_LBS10666`: request `5d2a4cc08feb491e`, swap request `5d2a4cc08feb491e`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair67-20260315-085124.plan.json`, router miss request `5d2a4cc08feb491e`
- `hellaswag::24952`: request `2dd64578aea841a1`, swap request `2dd64578aea841a1`, active set `6ebe4366c72f4528`, plan `/tmp/parallel-pair67-20260315-085124.plan.json`, router miss request `2dd64578aea841a1`

## Benchmarks

### arc_challenge
- accuracy: 0.00%
- coherence: 0.00%
- errors: 0 (0.00%)
- parse errors: 1 (100.00%)
- benchmark time: 2.183s
- avg request time: 1.916s
- avg sample time: 2.183s
- p95 sample time: 2.183s
- avg swap time: 0.190s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### gsm8k
- accuracy: 0.00%
- coherence: 0.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 126.246s
- avg request time: 42.117s
- avg sample time: 126.246s
- p95 sample time: 126.246s
- avg swap time: 83.944s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### hellaswag
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 2.451s
- avg request time: 2.107s
- avg sample time: 2.451s
- p95 sample time: 2.451s
- avg swap time: 0.164s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### mmlu
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 2.661s
- avg request time: 2.280s
- avg sample time: 2.661s
- p95 sample time: 2.661s
- avg swap time: 0.200s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

### winogrande
- accuracy: 100.00%
- coherence: 100.00%
- errors: 0 (0.00%)
- parse errors: 0 (0.00%)
- benchmark time: 1.733s
- avg request time: 1.485s
- avg sample time: 1.733s
- p95 sample time: 1.733s
- avg swap time: 0.167s
- avg active expert bytes: 16357785600.0
- refresh suggested rate: 100.00%

## Retention

- retained metrics status: `ok`
- overall accuracy retained: 75.00%
- overall coherence retained: 60.00%
- overall quality loss: 20.00%
- worst benchmark accuracy drop: 1.0000

- arc_challenge: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- gsm8k: accuracy retained 0.00%, coherence retained 0.00%, accuracy drop 1.0000
- hellaswag: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
- mmlu: accuracy retained 0.00%, coherence retained 100.00%, accuracy drop 0.0000
- winogrande: accuracy retained 100.00%, coherence retained 100.00%, accuracy drop 0.0000
