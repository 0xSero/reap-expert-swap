# 07 -- Dataset and Methodology Audit

## Benchmark Specifications

| Benchmark | HF Dataset | Config | Split | Style | Max Tokens | Gold Parse |
|---|---|---|---|---|---|---|
| mmlu | `cais/mmlu` | `all` | `validation` | mcq (A-E) | 8 | Letter match |
| arc_challenge | `ai2_arc` | `ARC-Challenge` | `validation` | mcq (A-E) | 8 | Letter match |
| hellaswag | `hellaswag` | `None` | `validation` | mcq (A-D) | 8 | Letter match |
| winogrande | `winogrande` | `winogrande_m` | `validation` | binary (1/2) | 8 | Digit match |
| gsm8k | `gsm8k` | `main` | `test` | math | 256 | Numeric parse |

**Note**: The E2E runner spec_map uses `winogrande_xl` but BENCHMARK_SPECS uses `winogrande_m`. The e2e runner's `make_first_active_set` hardcodes `winogrande_xl`. This is a minor inconsistency that could affect sample selection.

## Sample Selection

- `sample_count=5` per benchmark (in isolation run), `seed=7`
- `calibration_count=0`
- Total: 5 benchmarks x 5 samples = **25 prompts**
- Selection: `dataset.shuffle(seed=7)`, take first 5 rows
- Deterministic: same seed always produces same rows

## Seed=7 Determinism Verification

HuggingFace `datasets.shuffle(seed=7)` uses a deterministic permutation. Combined with:
- `format_prompt()` is a pure function of (spec, row)
- `stable_id()` uses SHA1 of concatenated parts
- `temperature=0` for inference

The full pipeline is deterministic: same seed = same prompts = same expected outputs.

## Coherence Check Analysis

```python
def coherence_pass(spec, text, parsed_answer):
    return (
        bool(text.strip())              # non-empty
        and printable_ratio(text) >= 0.95   # mostly printable
        and not is_repetitive(text)         # no trigram spam
        and parsed_answer is not None       # parseable answer found
    )
```

### Can empty `<think></think>` + single letter pass?

**Yes.** A response like `"\n\n<think>\n\n</think>\n\nC"` passes all four checks:
1. Non-empty: `"\n\n<think>\n\n</think>\n\nC".strip()` = `"<think>\n\n</think>\n\nC"` -- truthy
2. Printable ratio: all characters are printable or `\n` -- passes 0.95 threshold
3. Not repetitive: too short for trigram detection (needs 20+ tokens)
4. Parseable: `parse_prediction` finds `C` via regex `\b([A-E])\b`

The BF16 baseline shows this exact pattern: MCQ responses are `<think></think>X` with empty reasoning blocks. The coherence check does NOT assess reasoning quality.

### Implications

- Coherence rate measures "did the model produce a parseable answer" not "did the model reason correctly"
- A model that always outputs `A` would pass coherence on every MCQ prompt
- The 0.56 coherence rate on strict30 armA means 44% of responses couldn't even be parsed
- This is worse than the quality issue -- the model is producing incoherent output 44% of the time

## BF16 Baseline Matching

| Property | BF16 Baseline | strict30 Isolation |
|---|---|---|
| Seed | 7 | 7 |
| sample_count | 10 | 5 |
| calibration_count | 0 | 0 |
| Benchmarks | Same 5 | Same 5 |
| Protocol | single_turn | single_turn |
| Temperature | 0 | 0 |
| Model weights | Full BF16 | Same weights, sparse subset |
| Server | Port 8010 (full) | Port 8361 (multiplex) |

**Mismatch**: The baseline uses `sample_count=10` (50 total) while the isolation run uses `sample_count=5` (25 total). The gate comparison is flagged as `invalid_unmatched_baseline` because of `mismatched: sample_count_per_benchmark, result_signatures`.

**However**: The first 5 samples at seed=7 are the same in both runs (deterministic selection). So a matched comparison CAN be done by filtering the baseline to the same 25 prompt IDs. The status.md reports this was done: "BF16 baseline: 22/25 = 0.88 accuracy" on the matched 25 prompt IDs.

## Sample Size Statistical Power

With 25 samples (5 per benchmark):
- **Accuracy**: observed 0.40 (10/25). 95% CI using Wilson interval: [0.23, 0.59]
- **BF16 reference**: observed 0.88 (22/25). 95% CI: [0.70, 0.96]
- **Difference**: 0.48. The CIs don't overlap -- the difference is statistically significant even at n=25.
- **Per-benchmark**: 5 samples per benchmark gives only ±30-40% CI per benchmark. Individual benchmark comparisons are unreliable.

**Conclusion**: 25 samples is sufficient to detect the current ~50% accuracy gap. It is NOT sufficient to detect small improvements (e.g., 0.40 -> 0.50 would not be significant). A meaningful improvement test needs at least n=50 (10/benchmark).

## Prompt Format

### MCQ (mmlu, arc_challenge, hellaswag)
```
Answer the multiple choice question. Reply with only the option letter.

Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:
```

### Binary (winogrande)
```
Choose the better option to fill in the blank. Reply with only 1 or 2.

Sentence: {sentence}
1. {option1}
2. {option2}

Answer:
```

### Math (gsm8k)
```
Solve the math problem. Keep the response concise. End with exactly 'Final answer: <number>'.

Question: {question}

Answer:
```

## Known Issues

1. **winogrande config mismatch**: BENCHMARK_SPECS uses `winogrande_m`, e2e runner uses `winogrande_xl`
2. **Baseline sample count mismatch**: 10 vs 5, requiring manual ID matching
3. **Coherence masks quality**: empty think blocks pass coherence
4. **Small sample size**: 5/benchmark is too few for per-benchmark conclusions
5. **No interleaving in isolation run**: the `--interleave` flag state is not documented
