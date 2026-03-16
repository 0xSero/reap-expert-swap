# REAP Paper Full Data Extraction for Qwen3-30B-A3B

**Source:** arXiv:2510.13999v2 — "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression"
**Authors:** Lasby et al. (Cerebras Systems / U. Calgary), March 2026

---

## 1. Model Architecture Reference (Table 1)

| Property | Qwen3-30B-A3B | strict30 (Qwen3.5-35B-A3B) |
|---|---|---|
| Routed Experts/Layer | 128 | 256 |
| Shared Experts | 0 | 0 (assumed) |
| Top-K | 8 | 8 |
| Sparsity | 93.75% | 96.88% |
| Total Parameters | 30.5B | ~35B |
| Active Parameters | 3B | ~3B |
| First Layer Dense | No | No (assumed) |
| Expert Granularity | **High** | **High** |

---

## 2. Expert Count Mapping Across Compression Tiers

### Qwen3-30B-A3B (128 experts/layer, top-8)

| Tier | Compression | Experts Remaining | % of Total | Experts Active/Token |
|---|---|---|---|---|
| Baseline | 0% | 128 | 100.0% | 8 |
| 25% pruned | 25% | 96 | 75.0% | 8 |
| 50% pruned | 50% | 64 | 50.0% | 8 |

### strict30 Equivalent (256 experts/layer, top-8)

| Tier | Compression | Experts Remaining | % of Total | Experts Active/Token |
|---|---|---|---|---|
| Baseline | 0% | 256 | 100.0% | 8 |
| ~25% pruned | 25% | 192 | 75.0% | 8 |
| ~50% pruned | 50% | 128 | 50.0% | 8 |
| **strict30 target** | **~74.6%** | **65** | **25.4%** | **8** |

**Key insight:** strict30's 65/256 = 25.4% retention is significantly more aggressive than anything tested in REAP. The closest REAP data point is 50% compression (64/128 = 50% retention). strict30 retains half as many experts proportionally. This means strict30 is operating in an **untested, extreme compression regime** relative to REAP's experiments.

### Interpolation Framework for strict30

Using REAP's data at 0%, 25%, and 50% compression on Qwen3-30B-A3B, we can attempt to extrapolate to ~75% compression. However, the paper explicitly notes degradation is **non-linear** and accelerates at higher compression. The 50% data point should be treated as an **optimistic upper bound** for strict30's performance.

---

## 3. Table 2 — MC and Generative Results (Qwen3-30B-A3B, Domain-Specific Calibration)

### 3a. Coding Results

| Compression | Method | Eval+ | LiveCode | Code Avg | strict30 Extrapolation |
|---|---|---|---|---|---|
| Baseline | — | 0.814 | 0.302 | 0.558 | — |
| **25%** | M-SMoE | 0.781 | 0.293 | 0.537 | — |
| **25%** | HC-SMoE | 0.752 | 0.258 | 0.505 | — |
| **25%** | Frequency | 0.805 | 0.302 | 0.553 | — |
| **25%** | EAN | 0.797 | 0.311 | 0.554 | — |
| **25%** | **REAP** | **0.797** | **0.304** | **0.551** | — |
| **50%** | M-SMoE | 0.590 | 0.205 | 0.397 | — |
| **50%** | HC-SMoE | 0.543 | 0.185 | 0.364 | — |
| **50%** | Frequency | 0.668 | 0.236 | 0.452 | — |
| **50%** | EAN | 0.753 | 0.306 | 0.530 | — |
| **50%** | **REAP** | **0.780** | **0.302** | **0.541** | — |
| **~75% (strict30)** | **REAP (projected)** | **~0.65–0.72** | **~0.20–0.27** | **~0.43–0.50** | **Extrapolated; high uncertainty** |

**REAP degradation pattern (Coding):**
- 0% → 25%: Eval+ drops 0.814 → 0.797 (Δ = −2.1%)
- 25% → 50%: Eval+ drops 0.797 → 0.780 (Δ = −2.1%)
- Projected 50% → 75%: If degradation accelerates (as paper warns), expect Eval+ ~0.65–0.72

### 3b. Creative Writing (WildBench Normalized Score)

| Compression | Method | WildBench | strict30 Extrapolation |
|---|---|---|---|
| Baseline | — | 0.811 | — |
| 25% | M-SMoE | 0.805 | — |
| 25% | HC-SMoE | 0.497 | — |
| 25% | Frequency | 0.807 | — |
| 25% | EAN | 0.811 | — |
| 25% | **REAP** | **0.804** | — |
| 50% | M-SMoE | 0.725 | — |
| 50% | HC-SMoE | 0.008 | — |
| 50% | Frequency | 0.677 | — |
| 50% | EAN | 0.702 | — |
| 50% | **REAP** | **0.718** | — |
| **~75% (strict30)** | **REAP (projected)** | **~0.45–0.60** | **Accelerating degradation expected** |

**REAP degradation pattern (WildBench):**
- 0% → 25%: 0.811 → 0.804 (Δ = −0.9%)
- 25% → 50%: 0.804 → 0.718 (Δ = −10.7%) — sharp acceleration
- Projected 50% → 75%: Expect significant further drop, ~0.45–0.60

### 3c. Math (GSM8K + MATH-500)

| Compression | Method | GSM8K | MATH-500 | Math Avg | strict30 Extrapolation |
|---|---|---|---|---|---|
| Baseline | — | 0.903 | 0.872 | 0.887 | — |
| 25% | M-SMoE | 0.901 | 0.872 | 0.886 | — |
| 25% | HC-SMoE | 0.864 | 0.834 | 0.849 | — |
| 25% | Frequency | 0.910 | 0.865 | 0.888 | — |
| 25% | EAN | 0.904 | 0.879 | 0.892 | — |
| 25% | **REAP** | **0.896** | **0.881** | **0.888** | — |
| 50% | M-SMoE | 0.824 | 0.838 | 0.831 | — |
| 50% | HC-SMoE | 0.760 | 0.696 | 0.728 | — |
| 50% | Frequency | 0.871 | 0.860 | 0.865 | — |
| 50% | EAN | 0.874 | 0.855 | 0.864 | — |
| 50% | **REAP** | **0.877** | **0.838** | **0.857** | — |
| **~75% (strict30)** | **REAP (projected)** | **~0.78–0.85** | **~0.72–0.80** | **~0.75–0.82** | **Math is most resilient** |

**REAP degradation pattern (Math Avg):**
- 0% → 25%: 0.887 → 0.888 (Δ = +0.1%, essentially lossless)
- 25% → 50%: 0.888 → 0.857 (Δ = −3.5%)
- Math is the **most resilient** benchmark to expert pruning

### 3d. MC Average

| Compression | Method | MC Avg | strict30 Extrapolation |
|---|---|---|---|
| Baseline | — | 0.721 | — |
| 25% | M-SMoE | 0.558 | — |
| 25% | HC-SMoE | 0.674 | — |
| 25% | Frequency | 0.600 | — |
| 25% | EAN | 0.603 | — |
| 25% | **REAP** | **0.665** | — |
| 50% | M-SMoE | 0.451 | — |
| 50% | HC-SMoE | 0.542 | — |
| 50% | Frequency | 0.483 | — |
| 50% | EAN | 0.493 | — |
| 50% | **REAP** | **0.503** | — |
| **~75% (strict30)** | **REAP (projected)** | **~0.35–0.42** | **MC degrades faster than generative** |

**REAP degradation pattern (MC Avg):**
- 0% → 25%: 0.721 → 0.665 (Δ = −7.8%)
- 25% → 50%: 0.665 → 0.503 (Δ = −24.4%) — catastrophic acceleration
- MC benchmarks degrade **much faster** than generative benchmarks under REAP pruning for Qwen3-30B

---

## 4. Table 3 — Large-Scale Pruned SMoEs

Qwen3-30B-A3B does **not** appear in Table 3. Table 3 covers only Qwen3-Coder-480B and Kimi-K2-Instruct (models ≥110B). However, these results provide cross-architecture validation for REAP at scale:

| Model | Compression | Method | Code Avg | SWE-Bench | MC Avg |
|---|---|---|---|---|---|
| Qwen3-Coder-480B | Baseline | — | 0.636 | 0.540 | 0.750 |
| Qwen3-Coder-480B | 25% | REAP | 0.624 | 0.540 | 0.748 |
| Qwen3-Coder-480B | 50% | REAP | 0.619 | 0.522 | 0.692 |
| Kimi-K2 | Baseline | — | 0.631 | 0.554 | 0.780 |
| Kimi-K2 | 25% | REAP | 0.640 | 0.580 | 0.773 |
| Kimi-K2 | 50% | REAP | 0.624 | 0.576 | 0.643 |

**strict30 relevance:** These larger models show much better resilience to pruning than Qwen3-30B. The 480B model at 50% compression (80 experts remaining out of 160) retains near-baseline coding. This suggests **more experts per layer → more graceful degradation**, which is *favorable* for strict30's 256-expert architecture, partially offsetting the extreme compression ratio.

---

## 5. Table A5 — Quantization + Expert Pruning (Qwen3-30B-A3B)

| Quantization | Num. Experts | Approx. Relative Size | Eval+ | strict30 Equivalent |
|---|---|---|---|---|
| W16A16 | 128 | 100.0% | 81.4 | Baseline reference |
| W16A16 | 64 (REAP 50%) | 50.0% | 78.0 | — |
| W4A16-G128 | 128 | 25.0% | 80.5 | — |
| W4A16-G128 | 64 (REAP 50%) | 12.5% | 77.6 | **Closest to strict30's footprint** |
| W2A16-G128 | 64 | 12.5% | 28.6 | Catastrophic at 2-bit |

**Key finding for strict30:** Combining W4 quantization with 50% expert pruning (Eval+ = 77.6) is far superior to W2 quantization alone (Eval+ = 28.6) at the same checkpoint size. This validates the "prune + quantize" strategy.

**strict30 projection:** At ~75% pruning + W4 quantization, the relative checkpoint size would be ~6.25% of original. Based on the non-linear degradation, Eval+ would likely be in the **65–73 range** (below the 77.6 seen at 50% pruning + W4).

---

## 6. Table A6 — Detailed MC Results (Qwen3-30B-A3B)

### Individual MC Benchmark Breakdown

| Compression | Method | ARC-c | ARC-e | BoolQ | HellaSwag | MMLU | OBQA | RTE | WinoG. | MC Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | — | 0.563 | 0.790 | 0.887 | 0.778 | 0.779 | 0.454 | 0.816 | 0.702 | 0.721 |
| 25% | M-SMoE | 0.357±0.006 | 0.519±0.003 | 0.843±0.006 | 0.529±0.002 | 0.536±0.004 | 0.310±0.005 | 0.735±0.027 | 0.635±0.005 | 0.558±0.003 |
| 25% | HC-SMoE | 0.478±0.006 | 0.722±0.006 | 0.863±0.003 | 0.714±0.000 | 0.684±0.002 | 0.417±0.001 | 0.805±0.004 | 0.710±0.004 | 0.674±0.001 |
| 25% | Frequency | 0.401±0.011 | 0.600±0.016 | 0.847±0.003 | 0.593±0.005 | 0.600±0.004 | 0.342±0.012 | 0.781±0.002 | 0.637±0.005 | 0.600±0.005 |
| 25% | EAN | 0.406±0.007 | 0.603±0.014 | 0.847±0.005 | 0.607±0.006 | 0.600±0.002 | 0.337±0.003 | 0.764±0.002 | 0.660±0.009 | 0.603±0.004 |
| 25% | **REAP** | **0.481±0.004** | **0.727±0.002** | **0.855±0.004** | **0.700±0.006** | **0.673±0.001** | **0.399±0.008** | **0.789±0.014** | **0.696±0.003** | **0.665±0.002** |
| 50% | M-SMoE | 0.278±0.003 | 0.402±0.003 | 0.753±0.004 | 0.399±0.002 | 0.366±0.004 | 0.278±0.002 | 0.586±0.014 | 0.546±0.004 | 0.451±0.002 |
| 50% | HC-SMoE | 0.368±0.002 | 0.593±0.003 | 0.740±0.003 | 0.473±0.002 | 0.516±0.003 | 0.301±0.007 | 0.724±0.004 | 0.620±0.005 | 0.542±0.001 |
| 50% | Frequency | 0.285±0.001 | 0.424±0.002 | 0.779±0.003 | 0.458±0.003 | 0.397±0.002 | 0.286±0.004 | 0.659±0.012 | 0.570±0.009 | 0.483±0.001 |
| 50% | EAN | 0.296±0.006 | 0.426±0.009 | 0.759±0.007 | 0.471±0.002 | 0.443±0.001 | 0.291±0.009 | 0.668±0.020 | 0.589±0.009 | 0.493±0.003 |
| 50% | **REAP** | **0.354±0.006** | **0.503±0.008** | **0.737±0.009** | **0.481±0.004** | **0.496±0.003** | **0.309±0.001** | **0.561±0.020** | **0.584±0.004** | **0.503±0.002** |

### strict30 Projected MC Breakdown (~75% compression)

| Benchmark | Baseline | REAP 25% | REAP 50% | strict30 ~75% (projected) | Notes |
|---|---|---|---|---|---|
| ARC-c | 0.563 | 0.481 | 0.354 | ~0.25–0.30 | Steep decline |
| ARC-e | 0.790 | 0.727 | 0.503 | ~0.35–0.40 | Steep decline |
| BoolQ | 0.887 | 0.855 | 0.737 | ~0.60–0.68 | More resilient |
| HellaSwag | 0.778 | 0.700 | 0.481 | ~0.30–0.38 | Steep decline |
| MMLU | 0.779 | 0.673 | 0.496 | ~0.35–0.42 | Significant |
| OBQA | 0.454 | 0.399 | 0.309 | ~0.22–0.27 | Near random |
| RTE | 0.816 | 0.789 | 0.561 | ~0.50–0.55 | Variable |
| WinoG. | 0.702 | 0.696 | 0.584 | ~0.50–0.55 | Moderate |
| **MC Avg** | **0.721** | **0.665** | **0.503** | **~0.35–0.42** | **Heavy degradation** |

---

## 7. Table A7 — Detailed Coding Results (Qwen3-30B-A3B)

| Compression | Method | HE | HE+ | MBPP | MBPP+ | Eval+ | LiveCode | Code Avg |
|---|---|---|---|---|---|---|---|---|
| Baseline | — | 0.927 | 0.884 | 0.881 | 0.743 | 0.814 | 0.302 | 0.558 |
| 25% | M-SMoE | 0.878±0.012 | 0.833±0.007 | 0.849±0.007 | 0.728±0.007 | 0.781±0.007 | 0.293±0.017 | 0.537±0.006 |
| 25% | HC-SMoE | 0.866±0.011 | 0.805±0.016 | 0.832±0.006 | 0.698±0.005 | 0.752±0.006 | 0.258±0.000 | 0.505±0.003 |
| 25% | Frequency | 0.921±0.006 | 0.874±0.007 | 0.868±0.000 | 0.735±0.003 | 0.805±0.005 | 0.302±0.011 | 0.553±0.003 |
| 25% | EAN | 0.909±0.006 | 0.864±0.004 | 0.859±0.009 | 0.729±0.008 | 0.797±0.005 | 0.311±0.018 | 0.554±0.010 |
| 25% | **REAP** | **0.911±0.004** | **0.870±0.004** | **0.847±0.004** | **0.725±0.008** | **0.797±0.004** | **0.304±0.003** | **0.551±0.004** |
| 50% | M-SMoE | 0.687±0.013 | 0.638±0.004 | 0.618±0.004 | 0.541±0.007 | 0.590±0.005 | 0.205±0.019 | 0.397±0.007 |
| 50% | HC-SMoE | 0.577±0.023 | 0.541±0.013 | 0.631±0.010 | 0.546±0.004 | 0.543±0.005 | 0.185±0.018 | 0.364±0.007 |
| 50% | Frequency | 0.787±0.016 | 0.756±0.022 | 0.692±0.016 | 0.579±0.016 | 0.668±0.019 | 0.236±0.025 | 0.452±0.022 |
| 50% | EAN | 0.886±0.025 | 0.837±0.020 | 0.798±0.006 | 0.669±0.008 | 0.753±0.011 | 0.306±0.003 | 0.530±0.004 |
| 50% | **REAP** | **0.917±0.013** | **0.858±0.015** | **0.818±0.008** | **0.703±0.004** | **0.780±0.006** | **0.302±0.000** | **0.541±0.003** |

### strict30 Projected Coding Breakdown (~75% compression)

| Metric | Baseline | REAP 25% | REAP 50% | strict30 ~75% (projected) |
|---|---|---|---|---|
| HE | 0.927 | 0.911 | 0.917 | ~0.85–0.90 |
| HE+ | 0.884 | 0.870 | 0.858 | ~0.75–0.82 |
| MBPP | 0.881 | 0.847 | 0.818 | ~0.70–0.78 |
| MBPP+ | 0.743 | 0.725 | 0.703 | ~0.58–0.66 |
| Eval+ | 0.814 | 0.797 | 0.780 | ~0.65–0.72 |
| LiveCode | 0.302 | 0.304 | 0.302 | ~0.22–0.28 |
| **Code Avg** | **0.558** | **0.551** | **0.541** | **~0.43–0.50** |

**Notable:** REAP on Qwen3-30B shows remarkably stable coding performance — HE actually *increases* from 25% to 50% compression (0.911 → 0.917), suggesting redundancy in expert allocation for code tasks. LiveCode is essentially unchanged at 0.302 across all REAP compression levels.

---

## 8. Table A8 — C4 vs Domain-Specific Calibration (Qwen3-30B-A3B)

| Calibration | Compression | Method | Eval+ | LiveCode | Code Avg | MC Avg |
|---|---|---|---|---|---|---|
| **evol-codealpaca** | Baseline | — | 0.814 | 0.302 | 0.558 | 0.721 |
| **evol-codealpaca** | 25% | REAP | 0.797 | 0.304 | 0.551 | 0.665 |
| **evol-codealpaca** | 50% | REAP | 0.780 | 0.302 | 0.541 | 0.503 |
| **C4** | Baseline | — | 0.814 | 0.302 | 0.558 | 0.721 |
| **C4** | 25% | M-SMoE | 0.000 | 0.000 | 0.000 | 0.708 |
| **C4** | 25% | HC-SMoE | 0.788 | 0.269 | 0.529 | 0.641 |
| **C4** | 25% | Frequency | 0.000 | 0.000 | 0.000 | 0.709 |
| **C4** | 25% | EAN | 0.000 | 0.000 | 0.000 | 0.713 |
| **C4** | 25% | REAP | 0.763 | 0.253 | 0.508 | 0.702 |
| **C4** | 50% | M-SMoE | 0.000 | 0.000 | 0.000 | 0.422 |
| **C4** | 50% | HC-SMoE | 0.688 | 0.209 | 0.449 | 0.465 |
| **C4** | 50% | Frequency | 0.000 | 0.000 | 0.000 | 0.545 |
| **C4** | 50% | EAN | 0.000 | 0.000 | 0.000 | 0.667 |
| **C4** | 50% | REAP | 0.329 | 0.104 | 0.217 | 0.580 |

### strict30 Calibration Implications

**Critical finding:** C4 calibration causes **catastrophic coding failure** for most methods on Qwen3-30B. Frequency, EAN, and M-SMoE all produce 0% coding accuracy at both 25% and 50% compression with C4.

- **REAP with C4 at 25%**: Eval+ = 0.763 (vs 0.797 with domain-specific) — 4.3% degradation
- **REAP with C4 at 50%**: Eval+ = 0.329 (vs 0.780 with domain-specific) — 57.8% degradation
- **C4 helps MC slightly**: MC Avg with C4 is ~0.70 at 25% vs ~0.665 with evol-codealpaca
- Domain-specific calibration becomes **exponentially more important** at higher compression

**For strict30 at ~75% compression:** Domain-specific calibration is **non-negotiable**. Using general pretraining data would likely result in complete coding failure. The calibration dataset must match the target deployment domain.

---

## 9. Domain-Specific Calibration Analysis (Figures A8, A9)

### Figure A8 — Code Generation Accuracy vs. Calibration Dataset
*(Section 5.1 / Appendix F)*

**Key data points (from text descriptions):**
- C4 calibration results in **collapse in accuracy** for fine-grained models (Qwen3-30B, ERNIE)
- Several compressed model instances **fail to produce coherent output** (0% accuracy) when calibrated on C4
- evol-codealpaca calibration maintains high coding quality even at 50% compression
- The effect is **more pronounced for high-granularity models** (many experts/layer)

### Figure A9 — Accuracy vs. Task Type (Domain-Specific vs General Calibration)
*(Section 5.1 / Appendix F)*

**Key findings:**
- "General" calibration = combination of evol-codealpaca + WritingPrompts + tulu-3-sft-personas-math (3× more samples than domain-specific)
- Even with 3× more data, general calibration produces lower accuracy than domain-specific
- **REAP best preserves accuracy** compared to other compression methods in the combined data calibration setting
- Domain-specific calibration yields higher accuracy despite fewer total calibration samples

**strict30 implication:** Since strict30 is targeting code generation, calibration should use coding-specific SFT data (e.g., evol-codealpaca or similar). More calibration data is not better if it's off-domain.

---

## 10. Irreducible Error Theorem (Section 3.1, Equations 4–8)

### Setup (Eq. 1)
Layer output: `h(x) = Σ_{k∈T(x)} g_k(x) · f_k(x)` where T(x) is the top-k set.

### Input-Dependent Mixing Ratio (Eq. 4)
Define: `r(x) = g_i(x) / (g_i(x) + g_j(x)) ∈ [0,1]`

The original contribution of expert pair (i,j) can be written as:
```
g_i(x)·f_i(x) + g_j(x)·f_j(x) = (g_i(x)+g_j(x)) · [r(x)·f_i(x) + (1-r(x))·f_j(x)]
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                        The ideal, input-dependent target expert
```

### Irreducible Error of Merging (Eq. 5)
For merged expert `f̃(x) = α·f_i(x) + (1-α)·f_j(x)` with constant `α ∈ [0,1]`:

```
E_merge = E_x[(g_i+g_j)² · (r-α*)² · ||Δ_ij||²]
           ^^^^^^^^^^^     ^^^^^^^^^   ^^^^^^^^^^
           router scale    policy var   expert gap
```

Where:
- `α* = E[r(x)]` (optimal constant mixing ratio)
- `Δ_ij = f_i(x) - f_j(x)` (expert difference vector)

### Decomposed Error Bound (Eq. 6)
Under weak correlation assumption:
```
E_merge ≈ E_x[(g_i(x)+g_j(x))²] · Var[r(x)] · G_ij
```
Where `G_ij = E_x[||Δ_ij(x)||²]`

### Pruning Error (Eq. 7)
```
E_prune = E_{x|j∈T(x)}[||g_j(x)·f_j(x) - g'_i(x)·f_i(x)    (substitution error)
                          - ((g_j-g'_i)/(1-g_j+g'_i)) · Σ_{k≠i,j} g_k(x)·f_k(x)||²]  (renorm error)
```

### Pruning Error Upper Bound (Eq. 8)
```
||g_j(x)·f_j(x) - g'_i(x)·f_i(x)|| ≤ g_j(x) · (||f_j(x)|| + ||f_i(x)||)
```

### Variable Definitions
| Variable | Definition |
|---|---|
| `K` | Total experts per layer |
| `f_k` | Expert k function, f_k: ℝ^d → ℝ^d |
| `T(x)` | Top-k set of expert indices for input x |
| `g_k(x)` | Router gate value for expert k (normalized, sum to 1 over T(x)) |
| `r(x)` | Input-dependent mixing ratio = g_i/(g_i+g_j) |
| `α*` | Optimal constant merge ratio = E[r(x)] |
| `Δ_ij` | Expert gap = f_i(x) - f_j(x) |
| `G_ij` | Expected squared expert gap = E[||Δ_ij||²] |
| `Var[r(x)]` | Router policy variability |
| `g'_i(x)` | Gate value of promoted expert after pruning |
| `S_j` | REAP saliency score for expert j |

### Key Insight for strict30

**Why `Var[r(x)]` matters:**

The irreducible merging error is proportional to `Var[r(x)]`. High-granularity SMoEs (like Qwen3-30B with 128 experts, or strict30 with 256 experts) have **highly variable routing policies** because they combine many small contributions. This means:

1. **Merging is fundamentally disadvantaged** for high-granularity architectures (both Qwen3-30B and strict30)
2. **Pruning preserves independent control** — the router still modulates each surviving expert independently
3. **Pruning error is proportional to g_j** (the gate value of the pruned expert), not to policy variability
4. With 256 experts, individual gate values are small (~1/8 for top-8), making each pruning decision low-impact
5. **strict30's extreme pruning (75%)** removes many experts, but each removal has small individual impact due to the small per-expert gate values

**Quantitative prediction:** With 256 experts and top-8 routing, the average gate value per active expert is ~0.125. The substitution error per pruning decision is bounded by `g_j · (||f_j|| + ||f_i||) ≈ 0.125 · 2||f||`. This is smaller than for Qwen3-30B's 128-expert architecture where average gate ≈ 0.125 (same top-8), suggesting **similar per-decision impact but more decisions needed** for strict30.

---

## 11. Generation Quality Analysis (Figure 4, Qwen3-30B at 50% compression)

### Figure 4(a) — N-Gram Diversity
*(Section 5.1)*

**Methodology:** 100 questions from evol-codealpaca, measured N-gram diversity across N=1,2,3,4.

**Findings:**
- **Merged models** (HC-SMoE, M-SMoE): Significantly lower N-gram diversity across ALL N-gram sizes
- **REAP pruned model**: Remains similar to base model, slightly less diverse
- **Interpretation**: Merged models produce more repetitive, less varied text — evidence of functional manifold collapse

### Figure 4(b) — Cross-Perplexity
*(Section 5.1)*

**Methodology:** Perplexity of compressed model outputs evaluated by the original baseline model.

**Findings:**
- **Merged models**: Higher mean AND higher variance cross-perplexity
- **REAP pruned model**: Lower mean and lower variance — outputs more closely aligned to original model
- **Interpretation**: REAP preserves the distribution of model outputs much better than merging

### Figure 4(c) — Completion Logit JSD (Jensen-Shannon Divergence)
*(Section 5.1)*

**Methodology:** JSD between compressed and baseline model logits vs. completion token position.

**Findings:**
- **All methods**: Initially share close alignment with baseline (early tokens)
- **Merged models**: Diverge from baseline **more rapidly** as completion token position increases
- **REAP pruned model**: Diverges more slowly, maintaining closer alignment over longer sequences
- **Interpretation**: Error accumulation in auto-regressive generation is much worse for merged models

### Figure 4(d) — Expert Distance Analysis
*(Section 5.1)*

**Methodology:** Mean relative L2-distance and singular-vector alignment between expert weights at 50% compression.

**Findings:**
- Distance between clustered experts within the same layer **greatly exceeds** the distance between expert weights from pretrained to IFT checkpoints
- Even after weight matching permutations (M-SMoE), expert clusters remain far apart in both weight space and singular-vector alignment
- Expert merging faces a **much harder optimization problem** than standard model merging

### strict30 Implications

All four generation quality metrics point in the same direction: **pruning preserves generation quality far better than merging** for high-granularity architectures. For strict30:

1. **N-gram diversity** should remain near-baseline even at 75% pruning (based on the trend)
2. **Cross-perplexity** will increase but should stay within reasonable bounds
3. **JSD divergence** will accumulate faster at 75% pruning — this is the biggest risk for long-form generation
4. **No merging artifacts** — pruning is a clean coordinate subspace operation

---

## 12. PCA Variance Analysis (Table A4 — Qwen3-30B-A3B)

| Layer | Baseline PC1+PC2 Var | Merged PC1+PC2 Var | Pruned PC1+PC2 Var |
|---|---|---|---|
| Layer 0 (early) | 0.2343 | 0.2700 (+15.2%) | 0.1845 (−21.3%) |
| Layer 47 (late) | 0.7195 | 0.7437 (+3.4%) | 0.6860 (−4.7%) |

**Interpretation:**
- Merged models have **higher** explained variance → lost high-dimensional complexity
- Pruned models have **lower** explained variance → preserved outlier experts and high-dimensional structure
- The difference is more dramatic in early layers for Qwen3-30B

---

## 13. Summary: strict30 Predictions

### Architecture Comparison

| Property | Qwen3-30B-A3B (REAP tested) | strict30 (Qwen3.5-35B-A3B) |
|---|---|---|
| Experts/layer | 128 | 256 |
| Top-K | 8 | 8 |
| Target retention | 50% (64 experts) | 25.4% (65 experts) |
| Effective compression | 50% | ~74.6% |
| Expert granularity | High (128:8 = 16:1) | Very High (256:8 = 32:1) |

### Predicted Performance Range (REAP ~75% compression on strict30-like arch)

| Benchmark | Baseline (est.) | REAP 75% (projected) | Confidence |
|---|---|---|---|
| Eval+ (coding) | ~0.82 | 0.65–0.72 | Medium (extrapolated) |
| LiveCode | ~0.30 | 0.20–0.27 | Medium |
| Code Avg | ~0.56 | 0.43–0.50 | Medium |
| WildBench (writing) | ~0.81 | 0.45–0.60 | Low (high variance) |
| Math Avg | ~0.89 | 0.75–0.82 | Medium-High (resilient) |
| MC Avg | ~0.72 | 0.35–0.42 | Low (degrades fast) |

### Favorable Factors for strict30
1. **More experts (256 vs 128)** → more redundancy → more graceful degradation per decision
2. **Higher granularity** → smaller individual expert contributions → lower per-pruning error bound
3. **Evidence from 480B model** → more experts correlates with better pruning resilience

### Unfavorable Factors for strict30
1. **75% compression is far beyond tested range** — REAP only tested up to 50%
2. **Degradation is non-linear** and accelerates at higher compression
3. **MC benchmarks** degrade much faster than generative ones
4. **Domain-specific calibration** becomes exponentially more critical at extreme compression
5. **No shared experts** in Qwen3 architecture — no safety net for pruned functionality

### Critical Calibration Requirement
Domain-specific calibration is **mandatory** for strict30. C4 (general) calibration at 50% compression on Qwen3-30B already causes REAP Eval+ to drop from 0.780 → 0.329 (58% degradation). At 75% compression, general calibration would almost certainly produce **catastrophic failure** on coding tasks.
