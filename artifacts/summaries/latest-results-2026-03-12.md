# Latest results (2026-03-12)

Current frontier for Qwen3.5-35B-A3B at the profile-derived floor operating point.

## Top runs

| Run | Resident GiB | Raw Acc | Retained Acc | BF16 Answer Agr | BF16 Sim | Exact Match | Parse Err | Avg Swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| backfill + full rerank | 23.885 | 84% | 98% | 86% | 88.55% | 70% | 0% | 0.657s |
| backfill + full rerank (wider candidates) | 23.885 | 84% | 98% | 86% | 90.14% | 70% | 0% | 0.689s |
| backfill + full rerank (2 refresh) | 23.885 | 84% | 98% | 86% | 88.64% | 70% | 0% | 0.614s |
| benchmark-gated rerank (hellaswag + mmlu) | 23.487 | 82% | 95% | 84% | 90.15% | 68% | 0% | 0.651s |
| full rerank, no backfill | 23.487 | 80% | 93% | 86% | 88.56% | 68% | 0% | 0.669s |

## What changed

- **Targeted backfill (+68 experts)** fixed the known GSM8K arithmetic failure.
- **Full disagreement-conditioned reranking** recovered BF16 parsed-answer agreement.
- The combined approach is the current best balanced operating point.

## Benchmark read for the best balanced run

- GSM8K: 100% retained
- HellaSwag: 100% retained
- MMLU: 100% retained
- WinoGrande: 100% retained
- ARC: 89% retained

## Caveats

- These are still small matched slices (50 prompts), not a full large-scale benchmark campaign.
- Multi-turn evaluation infrastructure exists, but the latest best run above is from the single-turn matched harness.
- BF16 parity is not achieved yet.
