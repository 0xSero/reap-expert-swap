# Next Step: Implement strict30-v2 and Validate Locally

## Context

The deep analysis is complete. We know exactly why strict30 is fake-dynamic
(7 root causes documented in `02-root-cause-analysis.md`) and we have a
concrete v2 plan specification (`08-strict30-v2-spec.md`).

A quick local test just confirmed: flipping `rotationPolicy` to
`late_prompt_hash` alone -- no other changes -- produces **20 unique
signatures** across 20 non-winogrande prompts (up from 1). The rotation
mechanism works. The budget and prior fixes would add further differentiation.

## What to do next

### 1. Build the strict30-v2 plan JSON

Modify `assets/strict30-best-plan.json` to create `assets/strict30-v2-plan.json`:

- Change `rotationPolicy` from `"none"` to `"late_prompt_hash"`
- Change `budget.core_budget_fraction` from `0.75` to `0.50`
- Change `budget.specialist_budget_fraction` from `0.25` to `0.50`
- Recalculate all downstream budget fields (per-layer specialist bytes,
  specialist active target, candidate target) to match 50/50 split
- Add `scorerArtifacts.supportEstimatorConfig.mode: "prefill_reserve"`
  with `reserve_fraction: 0.30` (forward-looking for when miss priors exist)

The core expert lists and slice catalogs stay the same -- those come from the
observer summary and can't change without new observer runs. The budget split
changes which slices are selected at runtime, not which slices exist.

### 2. Implement and run the local validation script

Build `tools/strict30_v2_local_validate.py` per the design in
`09-validation-script-design.md`. Run it on both v1 and v2 plans. Prove:

- v1: 1 unique signature (confirmed baseline)
- v2: significantly more than 1 (target: 5+ distinct benchmark groups)
- Budget compliance on all payloads
- Core expert presence on all payloads

### 3. Fix `build_dynamic_plan` cluster priors (code change)

Apply the code change specified in `08-strict30-v2-spec.md` section
"Code Change: build_dynamic_plan Cluster Priors Fix":

- Make cluster priors always populated for all domain tags
- Map each tag to the most relevant source summary label
- Remove dependency on `activation_records` for basic cluster prior population

And the taskFamily priors key alignment:

- Add synthetic label entries that alias domain tags to source summary orderings

These code changes affect plan generation, not the current plan. They're
needed for when new observer summaries are collected.

### 4. Decide the quality question

The analysis shows strict30's 0.40 accuracy is consistent with ~75%
compression on a mismatched calibration set. The REAP paper predicts
0.35-0.42 MC accuracy at this tier -- which is exactly what we see.

This means: **dynamic movement alone probably won't fix quality.** The core
expert set was selected from personal-chat data, not from benchmark-relevant
data. The paper proves domain-specific calibration is mandatory at this
compression level.

The decision is:

- **If the goal is to prove dynamic movement works**: Steps 1-3 are sufficient.
  Run the three-arm isolation again with v2 plan on GPUs. If
  `dynamic_signature_count > 1` and `rows_with_nonzero_swap_adds > 0`,
  the mechanism is validated.

- **If the goal is to also improve quality**: New observer runs are needed on
  domain-specific datasets (math: NuminaMath/tulu-3-sft-personas-math,
  code: evol-codealpaca, general: WildChat). Then rebuild the plan with
  3 source summaries and the fixed `build_dynamic_plan`. This is a larger
  effort but is the only path to quality improvement.

- **If the goal is to decide whether strict30 is worth continuing**: The paper
  says ~75% compression is extreme. Even with perfect calibration and perfect
  dynamic routing, MC accuracy ceiling is probably 0.45-0.55. If that's not
  acceptable, the lane should pivot to a less aggressive compression target
  (50% = 128/256 experts, which the paper shows retains 0.50+ MC).

## Recommended order

```
Step 1: Build v2 plan JSON                    [~30 min, no GPU]
Step 2: Build + run local validation script   [~30 min, no GPU]
Step 3: Fix build_dynamic_plan code           [~30 min, no GPU]
  ---- requires decision on quality goal ----
Step 4a: GPU rerun with v2 plan (proves mechanism)  [~2 hrs, needs GPUs]
Step 4b: New observer runs + rebuilt plan (improves quality)  [~4 hrs, needs GPUs]
```

Steps 1-3 can all be done right now with no GPU access.
