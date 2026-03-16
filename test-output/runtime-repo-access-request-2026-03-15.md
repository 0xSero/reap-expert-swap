# Runtime repo access request

I am blocked on the one repo that actually needs edits:

- `/Users/sero/ai/reap-expert-swap`

## What is failing

This session can write in:

- `/Users/sero/ai/autoresearch`

This session cannot write in:

- `/Users/sero/ai/reap-expert-swap`

Observed error when attempting a trivial write test:

```text
operation not permitted: /Users/sero/ai/reap-expert-swap/.codex_write_test_retry_13070
```

## What I need

Give this session write access to:

```text
/Users/sero/ai/reap-expert-swap
```

The cleanest ways to do that are:

1. restart the session with `/Users/sero/ai/reap-expert-swap` in writable roots, or
2. restart the session with cwd set to `/Users/sero/ai/reap-expert-swap`, if that also grants write access there.

## Why I need it

The required code changes are in:

- `scripts/evaluate_original_vs_multiplex.py`
- `scripts/vllm_multiplex_server.py`

Without write access to that repo, I cannot:

1. patch the evaluator,
2. patch the server,
3. run the one-pair isolation experiment end to end,
4. leave behind actual runtime artifacts instead of staged instructions.

## What I will do immediately once access works

1. patch evaluator for:
   - warm-start active-set support
   - forced static active-set support
   - cold vs warm split
   - better swap timing breakdowns

2. patch server for:
   - global active-signature no-op reuse
   - explicit endpoint vs RPC timing fields
   - better no-op handling for same-signature requests

3. run the one-pair isolation experiment:
   - current dynamic, pre-shrunk
   - forced static first active set
   - current cold-included path

4. return with:
   - changed files
   - commands run
   - output paths
   - diagnosis from the actual new results

## Current staged materials

Already written in `autoresearch`:

- `/Users/sero/ai/autoresearch/test-output/30pct-isolation-next-steps-2026-03-15.md`
- `/Users/sero/ai/autoresearch/test-output/reap-expert-swap-patch-snippets-2026-03-15.md`
- `/Users/sero/ai/autoresearch/test-output/run-30pct-isolation-experiments.sh`

Those are ready now. The only missing piece is write access to the runtime repo.
