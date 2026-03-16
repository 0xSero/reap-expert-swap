# strict30 Deep Analysis

Comprehensive audit of the strict30 Qwen3.5-35B-A3B dynamic MoE lane.

Generated: 2026-03-16

---

## Documents

| # | Document | Summary |
|---|---|---|
| 01 | [System Architecture](01-system-architecture.md) | Every component end-to-end: model, observer, plan builder, scoring pipeline, server, evaluation harness. Mermaid + ASCII diagrams. |
| 02 | [Root Cause Analysis](02-root-cause-analysis.md) | 7 compounding failures causing 1-signature (fake dynamic) + 5 factors causing 0.40 accuracy. Each with exact code location, mechanism, proof, and fix category. |
| 03 | [Scoring Pipeline Trace](03-scoring-pipeline-trace.md) | Line-by-line mechanical trace of `rank_slice_ids_for_prompt`. Per-component formula, active/inactive status, what-if analysis table. |
| 04 | [REAP Paper Comparison](04-reap-paper-comparison.md) | Full extraction of Qwen3-30B-A3B data from REAP paper (arXiv:2510.13999v2). Every table row mapped to strict30 equivalents. Irreducible error theorem. Calibration findings. |
| 05 | [vLLM Runtime Analysis](05-vllm-runtime-analysis.md) | Native vLLM 0.17.x MoE features vs custom server. Feature comparison table. PR #31938 status. Qwen3.5-35B-A3B specific quirks. |
| 06 | [Budget Arithmetic](06-budget-arithmetic.md) | 3 budget scenarios (75/25, 35/65, 50/50) with exact waterfall calculations. Side-by-side comparison tables. Recommendation. |
| 07 | [Dataset Methodology Audit](07-dataset-methodology-audit.md) | Benchmark specs, seed determinism, coherence check analysis, baseline matching verification, sample size power analysis. |
| 08 | [strict30-v2 Spec](08-strict30-v2-spec.md) | Concrete next-plan configuration: 50/50 budget, late_prompt_hash rotation, 3 domain summaries, cluster priors fix, code change specs. |
| 09 | [Validation Script Design](09-validation-script-design.md) | Design for `tools/strict30_v2_local_validate.py`: proves signature diversity without GPUs. Interface, output format, acceptance criteria. |

## Diagrams

| Diagram | File | Description |
|---|---|---|
| System Architecture | [diagrams/system-architecture.mermaid](diagrams/system-architecture.mermaid) | End-to-end data flow from model to evaluation |
| Scoring Pipeline | [diagrams/scoring-pipeline.mermaid](diagrams/scoring-pipeline.mermaid) | Every branch in the scoring formula |
| Signature Failure Chain | [diagrams/signature-failure-chain.mermaid](diagrams/signature-failure-chain.mermaid) | 7 failures as a causal chain leading to 1 signature |
| Budget Waterfall | [diagrams/budget-waterfall.mermaid](diagrams/budget-waterfall.mermaid) | Budget decomposition for 3 scenarios |

## Key Findings

1. **strict30 is provably fake-dynamic**: 7 compounding failures make every prompt select the same 65 experts per layer. Zero signature movement, zero swap-adds, identical to a static sparse model.

2. **0.40 accuracy is consistent with ~75% compression on mismatched calibration**: The REAP paper predicts 0.35-0.42 MC accuracy at this compression tier on Qwen3-30B-A3B. The "personal" calibration source is a domain mismatch for academic benchmarks.

3. **The swap mechanism works**: Proven by the feasible envelope run. The problem is the scoring pipeline, not the runtime.

4. **Minimum viable fix**: Enable `late_prompt_hash` rotation + change budget to 50/50. This alone produces signature diversity.

5. **Quality fix requires multi-domain calibration**: New observer runs on math, code, and general datasets. The REAP paper proves this is non-negotiable at high compression.

## Next Steps

1. Build strict30-v2 plan (doc 08)
2. Validate locally (doc 09)
3. Collect multi-domain observer summaries
4. Rerun three-arm isolation on GPUs
5. Evaluate against acceptance criteria
