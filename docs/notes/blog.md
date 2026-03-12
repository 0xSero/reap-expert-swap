# I want to run Kimi-K2.5 on a shelf in my apartment

I have one goal: run frontier models locally with no meaningful quality drop.

Kimi-K2.5 takes about 2.5 TB of VRAM. That is 16-32 H200s, north of a million dollars. So I went the other direction.

I built a homelab. Eight RTX 3090s, zip-tied to an Ikea shelf with aluminum sheets and two 1600W PSUs that were definitely not rated for this.

```
       [ PSU 1600W ] ======= [ PSU 1600W ]
            ||                    ||
 +----------++--------------------++----------+
 |  [GPU1]  [GPU2]  [GPU3]  [GPU4]  [GPU5]    |
 |   3090    3090    3090    3090    3090 ...  |
 +--------------------------------------------+
       \____ zip-ties and aluminum dreams ____/
```

It worked. Then Z.AI's model went from 353B to 717B parameters. 2025 gave us a 5.2T parameter model. At some point self-hosting stops being a challenge and becomes an expensive coping mechanism.

## Part 1: static compression works at 7x

Before the dynamic runtime work, I did a static compression project on GLM-4.7. The [full case study is here](https://www.sybilsolutions.ai/case-studies/glm-4-7-compression).

The short version:

| | Before | After |
|---|---:|---:|
| Model size | 717 GB | 92 GB |
| Compression | 1x | **7.8x** |
| Hardware cost | $300,000+ (8xH200) | ~$8,000 (8x3090) |
| Pipeline cost | - | ~$300 |
| Pipeline time | - | ~8 hours |

Two-stage pipeline:

1. REAP expert pruning  
2. AutoRound INT4 quantization

That particular pipeline ran on 8xH200 spot instances on Prime Intellect at about $13/hr.

The calibration dataset is the most important choice when using REAP expert pruning. Your calibration dataset should get 3 things right:

1. Activate all the experts
2. Highlight the core of what you want to retain
3. Use high quality samples with dense context

Here was my calibration dataset. It is small, but it is dense and domain-matched:

| Dataset | Samples | Purpose |
|---|---:|---|
| evol-codealpaca-v1 | 700 | Code generation |
| xlam-function-calling-60k | 330 | Function calling |
| SWE-smith-trajectories | 330 | Agentic multi-turn |
| **Total** | **1,360** | Domain-matched calibration |

That was one of the first hard lessons: 1,360 high-quality domain-relevant samples can beat 10,000 generic ones if you are using REAP. Observation files get big fast, so every extra sample costs real time and storage.

The models are on HuggingFace under [0xSero](https://huggingface.co/0xSero). 40% retention is the sweet spot for most use cases. 50% retention works for extreme VRAM constraints but you start losing reasoning.

If I can get a model down to 15% of its size, why not 10%? And if 10%, why not 5%? That is when this stopped being a compression problem and became a systems problem.

## Part 2: Betting on routing concentration

REAP, from Cerebras, prunes experts in MoE models. The original paper gets you to roughly 50% with minimal quality loss. But for large sparse MoE models, the router only activates a small subset of experts per token per layer. The full model is huge. The active slice at any moment is small.

I built a personal activation corpus from my own Claude and Codex usage. 37,304 deduplicated prompts from real coding, research, daily tasks. Not synthetic benchmark data.

Here is the extractor pipeline I used:
- [ai-data-extraction](https://github.com/0xSero/ai-data-extraction)

Then I ran expert observation on Qwen3.5-35B-A3B across the 8x3090 cluster.

The routing distribution came back like this:

| Routing mass covered | Experts needed (per layer) | % of 256-expert pool |
|---:|---:|---:|
| 50% | ~19 experts | 7.6% |
| 80% | ~70 experts | 27.5% |
| 100% | 256 experts | 100% |

7.6% of experts carry half the routing mass. The distribution is not flat. There is real reuse and skew. A signal worth exploiting.

```
    Routing Mass %
100 |                              .-------
 80 |                       .------'
 50 |           .-----------'
 20 |    .------'
  0 +----'----------------------------------
         7.6%      27.5%              100%
         Experts Required per Layer
```

A 1000-token prompt produces around 40,000 token-layer routing decisions. The space of possible routing patterns is enormous, around 10^584,497 possible discrete states. You cannot search that space directly.

But I do not need the exact routing trace. I need the right support set: a small stable core that should almost always be loaded, plus a prompt-conditioned tail of specialists.

That became the real thesis.

## Part 3: delta swaps

I was inspired by vLLM's sleep mode: https://blog.vllm.ai/2025/10/26/sleep-mode.html

Sleep mode offloads weights and KV cache while keeping enough runtime structure alive that restart times drop from minutes to seconds.

That gave me the basic thought:

> What if instead of deleting low-salience active experts, I zero them out and preserve the model shape? Then I can load snapshots of expert state without reconstructing the whole runtime from scratch.

That exact idea was too restrictive. But it was the proof-of-possibility I needed. It showed me that the real thing I wanted was not just pruning. It was **dynamic expert availability**.

So I patched vLLM so that instead of rebuilding the full expert state on every change, it computes added/removed/reused experts and only touches the delta.

| Swap type | Data touched | Time |
|---|---:|---:|
| Dense to sparse (first load) | 51.80 GiB | 10.11s |
| Sparse A to sparse B (delta) | 1.875 GiB | 0.151s |
| Same set repeated (no-op) | 0 GiB | 0.000s |

```
STATE A:  [E1] [E5] [E10] [E42]
           |    |     |      |
         KEEP  KEEP  ZERO   ZERO
           |    |     |      |
         [E1] [E5] [E99] [E112]  <-- swap in new
           |    |     |      |
STATE B:  [E1] [E5] [E99] [E112]
```

First dense-to-sparse materialization is expensive. After that, transitions between budget-sized active sets are fractional-GiB deltas and, in the good case, sub-second operations. That was the first moment where the systems path stopped feeling fake.

## Part 4: the evaluation contract

At that point I needed a real measurement system, not vibes.

Everything in the main harness runs at **temperature 0**.

That matters. I am not trying to compare different samples from the same stochastic process. I am trying to isolate whether the runtime and selector preserve the original model's behavior.

The evaluator compares:

- **benchmark truth**: the actual dataset answer key
- **baseline truth**: what the original model/runtime did on the exact same sampled prompts

The rule is simple:

> If the baseline and dynamic runs are not matched on protocol, seed, sample count, and sample identity, retained metrics are invalid.

The core benchmark suite is:

| Benchmark | Type | Max tokens |
|---|---|---:|
| MMLU | MCQ | 8 |
| ARC Challenge | MCQ | 8 |
| HellaSwag | MCQ | 8 |
| WinoGrande | binary | 8 |
| GSM8K | math | 256 |

So when I say something like "38% retained accuracy," I mean:

> the dynamic system preserved 38% of the baseline system's aggregate accuracy on the same benchmark slice.

That is stricter than just saying "it got some answers right."

## Part 5: 19 experiments and a wall

I adapted the repo into a Karpathy-style autoresearch loop. Generate a candidate plan, materialize it, ship it to the remote machine, restart the patched vLLM runtime, run baseline-vs-dynamic smoke tests, collect swap timings and parse errors and router misses and benchmark scores, compare against a gate, append to a ledger, mutate.

GPT-5.4-Pro and GPT-5.3-Codex-Spark acted as research workers. They did not solve the problem for me. They accelerated the search: proposing candidates, patching scripts, killing dead directions, documenting evidence. The repo became a research machine instead of a pile of notebooks.

The first strict target was Qwen3.5-35B-A3B at a 20% resident VRAM budget.

| Quantity | Value |
|---|---:|
| Full BF16 | 63.42 GiB |
| 20% resident cap | 12.68 GiB |
| Always-resident trunk | 3.42 GiB |
| Expert budget remaining | 9.27 GiB |
| Active experts per layer | ~38 of 256 (14.8%) |

At that stage, 19 major 20%-regime experiments had gone through the loop. Here is the ledger snapshot that mattered most:

| Experiment | Score | Retained % | Swap (s) | Verdict |
|---|---:|---:|---:|---|
| Patched delta runtime, 20% plan | -63.1 | 38% | 10.1 | **keep** |
| Prompt-conditioned rotation | -11.2 | 12% | 1.3 | discard |
| Late-layer-only rotation | 12.6 | 25% | 1.2 | discard |
| Support-estimator v1 (crash) | -101.9 | 0% | 10.2 | crash |
| Support v1 mostly-core | -84.8 | 25% | 10.0 | discard |
| Support v2, reserve 40%, late layers | 12.8 | 38% | 1.5 | **keep** |
| Support v2, reserve 25%, very late | -73.9 | 38% | 10.2 | discard |
| Autoloop r35-l50-b12-t04 | **15.8** | **38%** | **1.2** | **keep** |
| Autoloop r35-l50-b12-t08 | 12.6 | 25% | 1.2 | discard |
| Focus: r35-l50-b20-t00 | -7.1 | 25% | 1.2 | discard |
| Focus: r40-l38-b16-t04 | 15.8 | 38% | 1.2 | discard |
| Nonuniform support v2 overlay | -40.3 | 12% | 1.2 | discard |
| Nonuniform base (no overlay) | -122.9 | 12% | 10.5 | discard |
| Hybrid prompt prior (live) | 17.9 | 25% | 0.5 | discard |
| Hybrid candidate-only | -80.4 | 22% | 10.0 | discard |
| Hybrid late75 | 15.9 | 25% | 0.5 | discard |
| Hybrid no-lexical GSM8K/ARC | 4.4 | 18% | 0.6 | discard |
| Packaging medium6 | -10.6 | 12% | 0.5 | discard |
| Packaging benchmark-pure8 | -76.5 | 2% | 0.5 | discard |

Score formula:

```text
retained_pct - 10 * swap_s - 100 * parse_error - 20 * max(0, resident - 12.68)
```

Higher is better.

Best live 20% result: **38% retained accuracy, ~1.2s to ~10.1s swaps depending on plan family, 12.68 GiB resident**.

Which is to say: bad. Everything at 20% was bad. But the failure pattern was useful.

## Part 6: what the failures taught me

**The selector was picking the same experts for every prompt.** Sampled layer signatures were often effectively identical across MMLU, ARC, GSM8K, everything. The system was not dynamic enough. It was a crippled static submodel with expensive swap overhead.

**Reasoning collapsed first.** GSM8K was usually the first benchmark to die. MMLU and ARC could still sound fluent while being completely wrong. Confidently dumb is worse than just dumb.

**The hybrid prompt prior improved everything except accuracy.** The offline prompt-prior sweep found a winner:

| Strategy | Avg recall | Oracle gap |
|---|---:|---:|
| hybrid_benchmark_tag_knn_5 | 0.813 | 0.139 |
| lexical_knn_5 | 0.807 | 0.145 |
| benchmark_prior | 0.794 | 0.158 |
| tag_prior | 0.752 | 0.200 |
| global_prior | 0.737 | 0.215 |

Live hybrid prior behavior was brutal and interesting at the same time:

| Variant | Retained acc | Coherence retained | Parse err | Avg swap |
|---|---:|---:|---:|---:|
| hybrid late75 | 25% | 88% | 4% | 0.51s |
| hybrid candidate-only | 22% | 98% | 2% | 10.04s |
| hybrid prior | 25% | 94% | 2% | 0.51s |

The prior made the model more stable and sometimes much faster. It did **not** make it smarter. Safer and dumber.

**Packaging hacks failed.** Width-only repacking and composition-aware benchmark grouping did not save the regime.

Representative packaging outcomes:

| Variant | Retained acc | Coherence retained | Parse err | Avg swap |
|---|---:|---:|---:|---:|
| medium6 | 12% | 74% | 30% | 0.53s |
| benchmark-pure8 | 2% | 12% | 74% | 0.45s |

The problem is not slice width or co-use bundling. Full-universe exact dynamic serving at 20% may simply be the wrong regime.

**Early-layer evidence did not help later-layer prediction.** An offline optimistic study across 85 trace rows showed almost no uplift:

| Budget | Prior recall | +Early evidence | Oracle ceiling | Uplift |
|---:|---:|---:|---:|---:|
| 20% | 0.723 | 0.719 | 0.898 | -0.003 |
| 25% | 0.829 | 0.823 | 0.966 | -0.006 |
| 30% | 0.894 | 0.886 | 0.992 | -0.008 |

The cheap version of early evidence, boosting later-layer experts that appeared in early-layer misses, was not the missing ingredient. Prompt priors were already carrying most of the easy predictive signal.

## Part 7: the budget oracle says 20% might be enough

Before giving up on 20%, I built a trace-derived surrogate oracle from 74 traced prompt rows across 13 dynamic histories.

| Budget | Expert capacity | Avg availability | Min availability |
|---:|---:|---:|---:|
| 20% | 1,581 | 90.0% | 68.1% |
| 25% | 2,122 | 95.7% | 79.5% |
| 30% | 2,663 | 98.3% | 89.3% |
| 35% | 3,204 | 99.3% | 96.3% |
| 40% | 3,745 | 99.7% | 98.5% |

At 20%, the oracle says about 90% average expert availability. The 38% retained accuracy I was getting live is way below what near-optimal selection could theoretically achieve. The gap between current selection and theoretically possible selection is the bottleneck. Not just the budget.

The oracle also flagged which layers matter most. Early and mid layers carried a lot more sensitivity than the simplistic late-layer heuristics were assuming.

| High-sensitivity layers |
|---|
| 0, 1, 5, 6, 7, 8, 10, 11, 13, 14, 39 |

That mattered a lot. It meant the current planner was blunt in exactly the wrong way.

## Part 8: floor frontier

At that point I needed to answer a more basic question:

> Is low-budget dynamic failure happening because the selector is bad, or because the model simply needs a much larger resident floor to stay coherent?

So I ran exact floors.

| Floor | Resident GiB | Dyn acc | Retained % | Dyn coh | Parse err | Avg sample | Avg swap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30% exact floor | 21.42 | 0.300 | 38% | 0.800 | 20% | 3.921s | 7.993s |
| 40% exact floor | 27.42 | 0.400 | 50% | 0.800 | 0% | 9.562s | 23.391s |
| 50% exact floor | 33.42 | 0.700 | 88% | 1.000 | 0% | 7.538s | 22.343s |

50% was the first configuration that got close on quality. But it was still rejected and way too big for the actual goal. It was a diagnostic anchor, not a solution.

## Part 9: the profiled floor bridge

I changed tactics. Instead of guessing a support set from weak heuristics at 20%, I profiled a stronger exact-floor regime and used that to build a better floor.

Approach:

1. run a 50% floor
2. profile which experts the router actually activated
3. build a tighter floor from that profile

| Run | Resident GiB | Retained accuracy | Retained coherence | Avg swap (s) | Avg sample (s) |
|---|---:|---:|---:|---:|---:|
| 50% floor (rerun) | 33.42 | 78% | 98% | 0.324 | 4.670 |
| Profiled floor (active90+inactive80) | **23.49** | **98%** | **96%** | **0.333** | **3.779** |

Cut resident size by almost 10 GiB. Accuracy went up. Swaps stayed cheap. Faster on average. First real step downward from the 50% anchor.

But the bridge had a contamination problem. The plan was profiled on the same 50 prompts it was evaluated on. So I ran a holdout with a different seed and only 1 overlapping prompt out of 50.

| Metric | In-sample | Holdout |
|---|---:|---:|
| Resident GiB | 23.49 | 23.49 |
| Accuracy | 98% | 74% |
| Coherence | 96% | 96% |
| Parse errors | 4% | 2% |
| Avg swap (s) | 0.333 | 0.335 |

Accuracy dropped 24 points on holdout. Profile contamination was real. But the method survived: 74% on unseen prompts is still way above the 38% I was getting at 20%. Profile-derived floors genuinely beat blind heuristics.

## Part 10: the activation diff

Digging into why the profiled floor still misses, I compared plan residency against full-model observer activity.

The numbers:

| Metric | Value |
|---|---|
| Plan resident experts | 3,425 (85.6/layer) |
| Full-model 95% activity envelope | 6,024 experts |
| Covered by plan core | 2,417 / 6,024 = 40.1% |
| Missing from core | 3,607 |
| Inactive mass on full95 experts | 58.8% |
| Mean inactive ratio (smoke) | 16.54% |
| Mean inactive ratio (holdout) | 20.10% |

Almost 59% of inactive routing mass hits experts the full model uses heavily. These are not tail experts nobody cares about. The floor is under-covering the important stuff.

Worst layers by router mistakes:

| Layer | Inactive ratio | Inactive mass on full95 experts | Missing full95 count |
|---|---:|---:|---:|
| layer_1 | 15.4% | 84.2% | 97 |
| layer_3 | 14.9% | 81.6% | 107 |
| layer_2 | 16.2% | 73.5% | 84 |
| layer_34 | 12.4% | 76.0% | 82 |
| layer_33 | 11.6% | 75.4% | 82 |
| layer_11 | 19.8% | 64.0% | 95 |

Layers 1-3 are getting hit hardest. The early layers have the most confident router mistakes: experts that the full model would definitely activate, but the floor plan does not include.

That was one of the biggest pivots in the whole project. The failure mode finally had a shape:

> false-negative residency

## Part 11: the learned router

If the handcrafted selector is garbage, train a small router on the traces. I built a dataset from 26,800 rows across 27 attempts and trained prompt-only per-layer slice rankers.

| Model | Learned coverage | Heuristic coverage | Oracle ceiling | Lift |
|---|---:|---:|---:|---|
| Multinomial NB (full history) | 0.0673 | 0.0182 | 0.1301 | 3.7x |
| SGD logreg (full history) | 0.0322 | 0.0182 | 0.1301 | 1.8x |
| Multinomial NB (bridge only) | 0.0398 | 0.0012 | 0.0701 | 33x |

3.7x lift over the existing heuristic from a Naive Bayes model. The selector is learnable. But the model was too eager to predict `__none__` as the top class, so it should be used as a slice reranker, not the whole policy. Progress, not a solution.

More detail from the same experiment:

| Model | Learned top-4 coverage | Top-1 accuracy | None pred rate |
|---|---:|---:|---:|
| Multinomial NB (full history) | 0.1185 | 45.95% | 71.15% |
| SGD logreg (full history) | 0.0667 | 12.77% | 12.50% |
| Multinomial NB (bridge only) | 0.0637 | 80.56% | 100.00% |

The bridge-only model looked cleaner but underperformed on absolute coverage. Broader history beat cleaner-but-smaller history.

## Part 12: multi-turn evaluation

Single-turn MCQ checks catch some regressions but they do not measure what I care about most: can the model hold a train of thought once a conversation becomes stateful?

I added a 3-turn evaluation chain:

- Turn 1: answer
- Turn 2: explain why
- Turn 3: recommit to the final answer

```
Turn 1 (answer) --> Turn 2 (explain) --> Turn 3 (recommit)
     |                   |                    |
  first-pass         coherence            answer
  accuracy           under follow-up      retention
```

Protocol calibration on the baseline showed the recommit scoring needed work. Default parsing only recovered 20% of turn-3 answers. After calibrating with reason-anchor extraction and numeric backstops:

| Parser variant | Turn-3 accuracy | Parse errors | Coherence |
|---|---:|---:|---:|
| Default | 20% | 60% | 40% |
| Turn-1 backfill | 40% | 0% | 100% |
| Reason anchor v2 | **60%** | **0%** | **100%** |

The multi-turn pipeline now runs on the real remote runtime with per-turn swaps and per-turn router-miss evidence.

First small matched dynamic multi-turn smoke after calibration:

| Run | Turn-3 acc | Parse err | Coherence | Conversation success | Avg swap |
|---|---:|---:|---:|---:|---:|
| Baseline calibrated | 60% | 0% | 100% | 60% | 0.000s |
| Dynamic calibrated | 40% | 0% | 100% | 40% | 0.261s |

Not good numbers. But the infrastructure works, and more importantly, the protocol itself is no longer faking most of the loss.

## What actually works, what does not

**Works:**

| Component | Status | Evidence |
|---|---|---|
| Expert routing is concentrated | Proven | 7.6% covers 50% mass |
| Delta swaps | Proven | 1.875 GiB / 0.151s transitions |
| Remote dynamic assembly | Proven | End-to-end on 8x3090 |
| Profile-derived floors | Proven | 74% holdout accuracy at 23.5 GiB |
| Learned prompt reranking | Proven | 3.7x lift over heuristic |
| Multi-turn dynamic eval | Proven | Pipeline runs, per-turn swaps work |

**Broken:**

| Component | Status | Evidence |
|---|---|---|
| 20% quality | Failed | 38% best after 19 experiments |
| Blind heuristic selectors | Failed | Often picks nearly the same experts every time |
| Packaging hacks | Failed | benchmark-pure8 hit 2% accuracy |
| Early-layer evidence | Failed | Neutral-to-negative uplift |
| Full-universe dynamic at 20% | Likely wrong regime | everything points to floor quality first |

## Where this goes

The GLM-4.7 static compression work proved that REAP + quantization gets you 7x on a single model. That is useful and ships today.

The dynamic runtime work is a different bet. The thesis is that you do not need to permanently delete experts if you can bring the right ones into VRAM fast enough. Delta swaps proved the fast-enough part. The 20% experiments proved the right-ones part is where it breaks.

The profiled floor bridge at 23.5 GiB, around 37% of full BF16, with 74% holdout accuracy, is the current best real result. That is not 5%. But it is a data point that says profile-derived selection beats everything else I have tried by a wide margin.

The likely next move is prune-first or floor-first, not blind dynamic-everything at the full 256-expert universe. If the expert universe is smaller and cleaner, the same live budget becomes a much easier selection problem.

I have not solved the 5% dream. I have solved the runtime, failed at 20% selection quality over and over, found a bridge at 37%, built a learned router, proved multi-turn eval works, and generated enough negative results to know which directions are dead.

Every experiment is another step toward finding out whether this is possible or whether I am mapping the exact boundary between stubbornness and insanity. Probably both.

I will update when I get closer.
