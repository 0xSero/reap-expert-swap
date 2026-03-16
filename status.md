# autoresearch status

Updated: 2026-03-16 EDT

## Deep Analysis Available

Full audit of strict30's fake-dynamic behavior, root causes, REAP paper comparison,
budget arithmetic, and v2 plan specification:

**[analysis/README.md](analysis/README.md)** -- 9 documents + 4 Mermaid diagrams

Key finding: 7 compounding failures make strict30 provably static. See
[analysis/02-root-cause-analysis.md](analysis/02-root-cause-analysis.md) for details
and [analysis/08-strict30-v2-spec.md](analysis/08-strict30-v2-spec.md) for the fix plan.

## What this repo is now

This is now the **minimal local control-plane + proof-store** for the strict30 Qwen3.5-35B-A3B runtime lane.

It is **not** the full runtime repo.

- Local vendored runtime/eval subset lives in `reap_swap/`
- Remote runtime still executes on `ser@192.168.1.70`
- Remote vLLM env is `/home/ser/reap-expert-swap-vllm016/.venv`
- Remote runtime repo is `/home/ser/reap-expert-swap-reap`

## Minimal repo layout

- `status.md` — current state, prior attempts, next moves
- `remote_auth.py` — ssh/scp auth helper
- `forensic_bundle.py` — one-request forensic packaging
- `one_request_forensic_replay.py` — replay/collector helper
- `assets/` — local vendored plan/baseline references
  - `strict30-best-plan.json`
  - `bf16-baseline-seed7-s10.json`
- `reap_swap/` — vendored minimal reap-swap subset
  - `dynamic_reap.py`
  - `dynamic_swap_delta.py`
  - `evaluate_original_vs_multiplex.py`
  - `multiplex_cache.py`
  - `research_gate.py`
  - `size_estimator.py`
  - `vllm_multiplex_server.py`
- `tools/strict30_pair01_e2e.py` — current reproducible pair01 three-arm runner
- `test-output/` — kept proof/handoff artifacts only

## Where we are

### Current strict30 result

Best current arm-to-arm isolation bundle:
- `test-output/30pct-isolation-e2e-20260315-190134/summary.md`
- `test-output/30pct-isolation-e2e-20260315-190134/summary.json`

Key result:
- strict30 is still behaving like a **fixed sparse set**, not real prompt-conditioned dynamic movement
- steady-state swap is **not** the blocker after pre-shrink
- cold dense->sparse shrink is the main one-time latency tax
- quality is still bad even after isolating that cold-start cost

### Exact matched BF16 comparison on the same 25 prompt IDs

From local artifacts:
- BF16 baseline: **22/25 = 0.88 accuracy**, **25/25 = 1.0 coherence**
- strict30 armA: **10/25 = 0.40 accuracy**, **14/25 = 0.56 coherence**

So the current strict30 lane is still far below BF16 on the same prompt slice.

## What we tried

### 1. Feasible-envelope one-request forensic replay
Artifact:
- `test-output/one-request-forensic-live-2026-03-15/`

What it proved:
- readiness works
- `/swap_active_set` works
- `/router_misses/{request_id}` works
- `/forensics/{request_id}` works
- a feasible offload envelope can serve requests successfully

Important caveat:
- that run proved **mechanism viability**, not strict30 parity

### 2. Parallel strict30 pair runs
Artifact:
- `test-output/parallel-pairs-2026-03-15/20260315-085124/summary.md`

What it proved:
- all 4 GPU pairs reached readiness and completed bounded evals at strict30 residency
- strict30 quality remained rejected
- old pair01 smoke result was `0.8`, but that was only **1 sample per benchmark** (5 total rows)

### 3. Three-arm pair01 isolation
Artifact:
- `test-output/30pct-isolation-e2e-20260315-190134/`

Arms:
- A: dynamic prewarmed
- B: forced static first active set
- C: cold included, fresh relaunch

What it proved:
- A and B are effectively the same
- C is worse mainly because of the first shrink
- all three arms stayed at **1 active-set signature**
- all three arms had **0 nonzero swap-copy rows** and **0 nonzero swap-add rows**

Conclusion:
- strict30 is still effectively static during measured prompts

## What the actual problem is

Not this:
- warm swap is too slow

Actually this:
- the strict30 plan/selector is not producing useful dynamic movement
- the sparse set itself is low quality on this benchmark slice
- current measured behavior is closer to a fixed sparse resident set than to a live dynamic policy

## What the solution is

1. **Stop calling strict30 dynamic unless signature movement is real**
   - reject any run with:
     - `dynamic_signature_count == 1`
     - `rows_with_nonzero_swap_adds == 0`

2. **Generate a matched BF16 baseline for the exact sample-count/signature set**
   - current gate outputs can go invalid if the baseline artifact is mismatched

3. **Fix the selector / plan quality first**
   - add real prompt-conditioned differentiation
   - do not spend more time micro-optimizing warm swap until the plan actually moves

4. **Then rerun the three-arm comparison**
   - prewarmed dynamic
   - forced static
   - cold included

## How to rerun the current flow

Preferred invocation:

```bash
cd /Users/sero/ai/autoresearch
uv run --with requests --with datasets tools/strict30_pair01_e2e.py
```

This runner will:
- read the vendored local plan from `assets/strict30-best-plan.json` by default
- sync the vendored `reap_swap/vllm_multiplex_server.py` to the remote runtime repo
- launch the pair01 remote runtime
- generate the first active-set payload locally
- run arms A/B/C
- relaunch before cold arm C so it is actually cold
- write a fresh summary bundle under `test-output/`

## Kept artifacts

Vendor/minimal-closure proof:
- `test-output/strict30-vendor-closure-2026-03-16.md`
- `test-output/strict30-vendor-closure-2026-03-16.json`


These were kept on purpose:
- `test-output/one-request-forensic-live-2026-03-15/`
- `test-output/parallel-pairs-2026-03-15/`
- `test-output/30pct-isolation-e2e-20260315-190134/`
- `test-output/30pct-isolation-runs-20260315-190456/`
- `test-output/30pct-isolation-patch-verify-20260315-185514/`
- `test-output/30pct-isolation-next-steps-2026-03-15.md`
- `test-output/reap-expert-swap-patch-snippets-2026-03-15.md`
- `test-output/run-30pct-isolation-experiments.sh`
- `test-output/runtime-repo-access-request-2026-03-15.md`
- `test-output/strict30-vendor-closure-2026-03-16.md`
- `test-output/strict30-vendor-closure-2026-03-16.json`

## Archived material

Everything else was archive-moved to:

- `/Users/sero/ai/autoresearch-archive-20260316-050602`

That includes:
- old benchmark/control scaffolding
- exploratory helper scripts that are not part of the current strict30 lane
- older test-output runs outside the active proof/handoff set
