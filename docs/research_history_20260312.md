# REAP Expert-Swap Research History and Progress (Start -> March 12, 2026)

This document is the public-safe long-form record of the project so far.

It exists to preserve the actual knowledge from the private research repo without shipping:

- private runtime topology
- personal activation data
- raw large experiment dumps
- machine-specific paths
- operator-specific mission scaffolding

If you want the shortest version, read:

- `RESEARCH.md`
- `docs/system_technical_report_20260312.md`

If you want the full story, definitions, phases, failures, pivots, and current frontier, read this file.

---

## 1. Project thesis

The project is trying to answer a very specific question:

> Can a very large sparse MoE model serve the **original weights** with a much smaller resident BF16 VRAM footprint by keeping a resident expert floor in memory and dynamically materializing a prompt-conditioned tail at runtime?

This is not ordinary quantization work.
This is not distillation.
This is not producing a smaller static model.

The target is:

- keep the **original sparse model reachable**,
- keep only a fraction resident,
- dynamically load what matters,
- and preserve useful behavior.

---

## 2. Definitions

## 2.1 Sparse MoE model

A sparse Mixture-of-Experts model has:

- a shared trunk,
- many experts per MoE layer,
- and a router that activates only a small subset of those experts at each token/layer decision.

The whole model is large, but each token only uses a small fraction of it.

## 2.2 Resident floor

The **resident floor** is the set of experts kept in VRAM for all requests.

This is meant to capture:

- broad shared competence
- frequently reused experts
- stable core model behavior

## 2.3 Specialist tail

The **specialist tail** is the prompt-conditioned set of experts added on top of the floor.

This is what should recover:

- task-specific competence
- rare reasoning specialists
- prompt-specific expert needs

## 2.4 Active set

For a given request and layer:

```text
active_experts(layer) = core_experts(layer) ∪ selected_specialist_slices(layer)
```

That union is the expert set the runtime makes available for the request.

## 2.5 Delta swap

A **delta swap** updates the runtime from one active set to another by touching only the difference:

- copy added experts
- zero removed experts
- reuse unchanged experts

This is the key runtime optimization that made the system practical enough to study.

## 2.6 Retained accuracy / retained coherence

These are **baseline-relative** metrics.

They mean:

- how much of the baseline system’s aggregate performance the experimental system preserved
- on the same benchmark slice, same protocol, same seed, same sampled prompts

This is different from benchmark correctness itself.

## 2.7 Benchmark truth vs baseline truth

Two truths are used in the repo:

### Benchmark truth
The dataset answer key.
This is the correctness target.

### Baseline truth
The full/original runtime behavior on the same prompt set.
This is the fidelity target.

The system is judged against both.

## 2.8 Inactive ratio / router misses

A **router miss** is when important routing mass falls on experts that are not available in the active set.

The **inactive ratio** is a summary of how much observed routing mass lands outside the active set.

High inactive ratio is a strong signal that the floor or tail is missing the wrong experts.

---

## 3. System shape

The project evolved into a four-part system:

1. **plan construction**
2. **dynamic serving runtime**
3. **evaluation harness**
4. **profiling / analysis tooling**

### 3.1 Plan construction
Builds a resident core and specialist slice structure.

### 3.2 Runtime
A patched vLLM-based server that accepts active-set swaps at request time.

### 3.3 Evaluation
Runs baseline and experimental requests on matched benchmark samples with `temperature: 0`.

### 3.4 Profiling and analysis
Measures:

- router misses
- active expert usage
- activity envelopes
- floor coverage
- support-router training targets

---

## 4. The earliest framing

The original ambition was not modest.

The larger dream behind the work is:

> push toward a regime where only around **5% to 10%** of the original BF16 footprint must be resident, while the original sparse model remains reachable at runtime.

For the repo’s practical milestone, that broad question was scoped down to a tractable test subject and a smaller milestone.

### Current model under test
- `Qwen3.5-35B-A3B`

### Full BF16 size
- **63.42 GiB**

### Important milestone budgets
- **20% resident** = **12.68 GiB**
- **30% resident** = **~19.0–21.4 GiB** depending on plan shape
- **50% resident** = **33.42 GiB**

The work started by asking:

> can the model survive a strict low-resident regime at all?

---

## 5. Phase history

## Phase 0 — static compression thinking

Before the current dynamic runtime work, the broader program had already established a crucial lesson:

- domain-matched calibration matters more than just calibration size
- static REAP compression can work surprisingly well
- but static pruning alone does not solve the frontier problem for giant sparse models

That phase changed the core framing from:

> "How do I compress the model enough?"

into:

> "How do I make the right part of the original model available exactly when it is needed?"

That is where the dynamic runtime path started.

---

## Phase 1 — routing concentration bet

The first important empirical bet was that sparse routing is concentrated enough to exploit.

The team built observer workflows and a larger activation-observation pipeline and found:

- only a small fraction of experts carry a large fraction of routing mass
- expert activity is highly skewed, not uniform
- there is a real stable core signal in the routing distribution

This justified the entire floor-plus-tail idea.

### What this phase established

- the active expert set is much smaller than the full expert universe
- expert reuse is real
- a resident floor is not absurd as a systems design

### What it did **not** establish

- which exact experts matter most for preserving quality
- whether prompt-conditioned tails could be predicted well enough
- whether runtime swaps would be cheap enough

---

## Phase 2 — first dynamic runtime attempts

The first dynamic attempts established the basic floor/tail shape but hit a systems wall.

### Problem
The old runtime path rebuilt too much expert state whenever the active set changed.

### Result
Even if the selected set was reasonable, the serving mechanics were too expensive and unstable.

### Lesson
The main blocker was not only selector quality.
The swap path itself was structurally wrong for low-resident serving.

---

## Phase 3 — delta swap breakthrough

This was the first major systems breakthrough.

The runtime was patched so it could do expert-granularity delta swaps.

### Observed behavior

| swap type | data touched | time |
| --- | ---: | ---: |
| dense -> sparse first materialization | 51.80 GiB | 10.11s |
| sparse A -> sparse B delta | 1.875 GiB | 0.151s |
| repeated same set | 0 GiB | 0.000s |

### Why it mattered

This changed the research question from:

> "Can the runtime do this at all?"

to:

> "Can we select the right experts well enough to make the runtime path worth using?"

That was a major shift.

---

## Phase 4 — strict 20% experiments

This was the first heavy search phase.

### Budget
- full BF16: **63.42 GiB**
- resident target: **12.68 GiB**

### What was tried

A large number of 20% family experiments were run, including:

- patched delta runtime
- rotation-based selectors
- support-set variants
- support prior injection
- nonuniform layer budgets
- hybrid prompt priors
- packaging variants
- autoresearch-generated parameter sweeps

### Best live 20% behavior

The best 20% retained-accuracy point reached only about:

- **38% retained accuracy**

Even when swap latency improved dramatically, answer quality remained too weak.

### Key lesson from 20%

The runtime path had become much healthier, but the **selector quality was still bad**.

This was the first strong indication that the core problem had shifted from systems mechanics to support-set estimation.

---

## Phase 5 — floor frontier

The next question was:

> is low quality at 20% because the runtime path is broken, or because the model simply needs a much larger resident floor?

So larger exact floors were tested.

### Key floor frontier results

| floor | resident GiB | dyn acc | retained % | coherence | avg swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40% | 27.42 | 0.400 | 50% | 0.800 | 23.391s |
| 50% | 33.42 | 0.700 | 88% | 1.000 | 22.343s |

### Interpretation

- **50% exact floor** was the first configuration that got close on quality
- but it was far too expensive in resident footprint and still too slow to swap

This was useful because it proved quality could be recovered by making more of the model resident.
But it was not a solution.

---

## Phase 6 — support-set and prompt-prior work

If 20% static-ish floors were too dumb and 50% floors were too big, the obvious next move was better support-set prediction.

### Families explored

- support-set ranking
- miss-history priors
- benchmark/tag priors
- lexical nearest-neighbor priors
- hybrid priors
- conservative reserve policies

### Important result
Support priors could improve **offline coverage** and sometimes improve **swap behavior**, but they did not produce a real live quality breakthrough at 20%.

### Strong negative result
The first strong support-estimator variant improved offline support coverage, but the aggressive live version crashed.

### Important nuance
Conservative support-prior injection could help swap time without helping retained accuracy enough.

So the question became:

> not just "can support priors move the support set?"

but:

> "can they move it in a quality-positive and stable direction?"

---

## Phase 7 — oracle / surrogate analysis

A trace-derived oracle study was run to answer a different question:

> Is 20% fundamentally impossible, or is the selector just far from the best possible support set under that budget?

### Oracle-style availability estimates

| budget | avg availability |
| --- | ---: |
| 20% | ~0.90 |
| 25% | ~0.96 |
| 30% | ~0.98 |

### Interpretation

The oracle analysis suggested:

- 20% is **not obviously impossible**
- 25% may already be near-feasible
- 30% should be much easier if support choice is good enough

That result was important because it shifted blame away from pure budget and toward selection fidelity.

### Another critical finding
The same oracle analysis highlighted early and mid layers as unusually important:

- early: `0, 1, 5, 6, 7`
- mid: `8, 10, 11, 13, 14`
- late: `39`

This undermined simplistic late-layer-only heuristics.

---

## Phase 8 — 30% bridge attempts

The next move was to test whether a moderate increase from 20% to ~30% would stabilize the system enough.

### What happened
It did **not** solve the problem.

Representative 30% observations:

- some plans were still too static and poor on quality
- some hybrid plans achieved real prompt variation but still selected the wrong specialists
- better swap time did not automatically mean better answers

### Central lesson
The system was not failing only because 20% was too low.
It was failing because the planner still did not preserve the right core or pick the right tail.

This was the point where the architecture question sharpened into:

> "What is the right floor to preserve, and how should dynamic behavior sit on top of it?"

---

## Phase 9 — learned support router

A lightweight learned-router path was added.

### What it did
It built a slice-space dataset from historical dynamic artifacts and trained a small prompt-conditioned ranker.

### Important result
On offline same-budget missing-mass coverage, the learned reranker beat the existing heuristic significantly.

Representative summary:

| model | learned coverage | heuristic coverage | delta |
| --- | ---: | ---: | ---: |
| Multinomial NB / full history | 0.0673 | 0.0182 | +0.0491 |

### Interpretation

This was the first strong sign that the selector problem is learnable.

### Limitation
The learned router was still too eager to emit `__none__` as the top class, so it was not yet trustworthy as the whole allocator.

The right near-term use was:

- reranker
- marginal-slot selector
- not full abstain/refresh owner

---

## Phase 10 — profile-derived floor breakthrough

This was the first real bridge win.

### New strategy
Instead of guessing a smaller floor from weak priors, the system:

1. ran a stronger floor,
2. profiled actual expert activity,
3. then built a smaller floor from that measured activity envelope.

### Strong bridge result

| run | resident GiB | active GiB | retained acc | retained coh | parse err | avg swap | avg sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% rerun | 33.419 | 30.000 | 78% | 98% | 0.00 | 0.324s | 4.670s |
| profiled floor active90+inactive80 | 23.487 | 20.068 | 98% | 96% | 0.04 | 0.333s | 3.779s |

### Why this mattered
This was the first time resident size came down sharply while quality stayed strong and swap remained cheap.

It was not yet the final low-budget answer, but it was the first convincing bridge down from the 50% anchor.

---

## Phase 11 — contamination audit and holdout validation

The first profile-derived win had an integrity problem:

- the floor had been profiled from the same prompts used in evaluation

That is contamination.

A holdout rerun was performed.

### Holdout result
The in-sample numbers dropped on holdout, but the method survived.

### Meaning
The precise original in-sample win was contaminated, but the **profile-derived floor method itself was real**.

This is a crucial distinction.

---

## Phase 12 — activation-diff diagnosis

The next question was:

> If profiled floors are better, why are they still failing on some prompts?

An activation-diff analysis compared the profiled floor against full-model observer activity.

### Key global divergence numbers

| metric | value |
| --- | ---: |
| resident experts | 3425 |
| full-model 95% activity envelope | 6024 |
| overlap with floor core | 2417 |
| full95 coverage ratio | 40.12% |
| full95 experts missing from core | 3607 |
| inactive mass landing on full95 experts | 58.79% |
| mean inactive ratio (smoke) | 16.54% |
| mean inactive ratio (holdout) | 20.10% |

### Interpretation
The misses were **not** random tail spillover.
They were landing on experts the full model uses heavily.

That means the main remaining failure mode is:

> **false-negative residency**

The floor is still leaving out too many experts that actually matter.

### Highest-confidence mistake layers

- `layer_1`
- `layer_3`
- `layer_2`
- `layer_34`
- `layer_33`

This gave the project a much clearer structural target.

---

## Phase 13 — multi-turn evaluation and protocol isolation

Single-turn benchmark smoke was never the whole story.
The system also needed to answer:

> what happens when reasoning/state has to persist across turns?

So multi-turn evaluation was added.

### Big surprise
The first multi-turn protocol was itself too lossy.

### Tiny matched-set protocol isolation result

| run | turn3 accuracy | parse error | coherence | conversation success |
| --- | ---: | ---: | ---: | ---: |
| baseline multi-turn original | 20% | 60% | 40% | 20% |
| baseline multi-turn calibrated | 60% | 0% | 100% | 60% |
| dynamic multi-turn original | 20% | 60% | 40% | 20% |
| dynamic multi-turn calibrated | 40% | 0% | 100% | 40% |

### Interpretation
The default protocol had been contaminating the measurement.

This was important because it prevented a bad conclusion:

- it was **not** fair to blame all multi-turn collapse on the plan/router before fixing the protocol itself

### What remained after calibration
After reducing protocol loss, the dynamic system still lagged the calibrated baseline.
So there is still real model/system loss there.
But the protocol was no longer the dominant unknown.

---

## 6. Everything tried, organized by family

This section is the compact map of what the project has actually tried.

## 6.1 Exact floors

### Tried
- ~30% exact floor
- 40% exact floor
- 50% exact floor

### What happened
- 30% improved quality but still failed badly
- 40% was better than 30% but too slow and still too weak
- 50% was the first close-quality point but far too large/slow

### Lesson
Exact floors are useful anchors, not the answer.

## 6.2 20% dynamic patched runtime

### Tried
- patched 20% dynamic baseline with delta-swaps

### What happened
- parse stability improved dramatically
- sample latency improved
- retained accuracy still collapsed

### Lesson
Runtime mechanics improved; selector quality remained the blocker.

## 6.3 Rotation families

### Tried
- prompt-conditioned rotation
- late-layer rotation

### What happened
- sometimes cheaper swaps
- no real quality breakthrough
- often quality collapsed badly

### Lesson
Prompt variation alone is not useful if the chosen specialists are wrong.

## 6.4 Support-set families

### Tried
- support-v1
- support-v2 reserve variants
- mostly-core support floors

### What happened
- strong offline signal in some cases
- unstable aggressive live variants
- conservative variants improved swap more than quality

### Lesson
Support priors are informative but not yet sufficient.

## 6.5 Prompt priors / hybrid priors

### Tried
- benchmark priors
- tag priors
- lexical KNN
- hybrid benchmark+tag+lexical systems

### What happened
- strong offline recall signals
- live swap could improve a lot
- retained accuracy still poor

### Lesson
The heuristic surrogate space was useful, but not enough to recover the right support set under strict budgets.

## 6.6 Nonuniform layer budgets

### Tried
- oracle-inspired nonuniform budget allocations

### What happened
- live quality still collapsed

### Lesson
Budget allocation alone does not fix selector/runtime coupling.

## 6.7 Packaging / slice-composition families

### Tried
- composition-aware packaging
- benchmark-pure grouping
- fixed-width packaging variants

### What happened
- some were fast
- quality collapsed badly
- some had very high parse error rates

### Lesson
The problem is not just how experts are packaged.
It is whether the active set contains the experts that matter.

## 6.8 Learned router

### Tried
- prompt-only slice ranking
- same-budget learned coverage evaluation

### What happened
- clear offline win over current heuristics
- not yet ready to be primary live selector

### Lesson
Selector quality is learnable, but the learned path needs careful integration.

## 6.9 Profile-derived floors

### Tried
- floors built from measured activity envelopes rather than weak priors

### What happened
- first meaningful bridge downward from strong exact floors
- contaminated in-sample win was corrected by holdout validation
- method survived holdout

### Lesson
Measured activity is far more trustworthy than naive low-budget heuristic planning.

## 6.10 Multi-turn protocol variants

### Tried
- original 3-turn protocol
- calibrated replay variants

### What happened
- protocol itself accounted for a large fraction of apparent loss

### Lesson
Evaluation protocol design is part of the research system. Bad protocol design creates fake model regressions.

---

## 7. Current understanding of the bottlenecks

At this point the main bottlenecks are clearer than they were at the beginning.

## 7.1 What is no longer the main blocker

### Not primarily runtime mechanics
The runtime path is now good enough that it can no longer absorb all the blame.

### Not purely budget impossibility
Oracle-style analysis suggests low budgets are hard, but not obviously impossible.

### Not just late-layer support tuning
The important layers are broader than that.

## 7.2 What **is** the main blocker

### The true expert core is still under-covered
The activation-diff work made this explicit.

### Prompt-conditioned support estimation is still not good enough
Heuristics were too blunt.
The learned router is promising but not yet fully integrated.

### Multi-turn fidelity still needs more live calibrated runs
The protocol has been improved, but the live dynamic-vs-baseline gap still needs larger matched evaluations.

---

## 8. What the project now knows that it did not know at the start

1. **Delta swaps are real and worth it.**
2. **The selector problem dominates once the runtime is healthy enough.**
3. **20% failure does not automatically prove impossibility.**
4. **50% exact floors are quality anchors, not product solutions.**
5. **Profile-derived floors are the first real bridge down from the safe regime.**
6. **The remaining loss is concentrated in specific layers, not just random noise.**
7. **Evaluation protocol design can create fake regressions if it is too lossy.**
8. **Lightweight learned routing is promising, but should initially be used as reranking rather than full control.**

---

## 9. Current frontier

As of March 12, 2026, the most honest summary is:

- a **strict 20% win does not exist yet**
- the runtime path is credible
- profile-derived floors are the strongest bridge result so far
- the remaining structural problem is preserving the real expert core
- the next likely gains come from:
  - surgical floor backfill in the worst layers
  - better marginal-slot routing
  - larger matched calibrated multi-turn sweeps

---

## 10. Recommended reading order for new contributors

1. `README.md`
2. `RESEARCH.md`
3. `docs/system_technical_report_20260312.md`
4. `docs/architecture/core_specialist_dynamic_architecture.md`
5. `docs/protocol/multi_turn_eval_protocol.md`
6. this file

That order gives:

- the short statement
- the technical system map
- the runtime design
- the evaluation design
- and then the full project history

---

## 11. Bottom line

The project began as a compression dream and turned into a systems-and-selection research problem.

The most important evolution was this:

- first the runtime looked impossible,
- then the runtime became plausible,
- then the selector became the obvious bottleneck,
- then the floor itself became measurable,
- and now the project is in the stage where the remaining failures are much more structurally legible.

That is real progress.

It is not the final answer.
But it is no longer a blind search.
