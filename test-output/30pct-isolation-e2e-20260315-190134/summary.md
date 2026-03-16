# 30pct isolation three-arm summary

out_dir: `/Users/sero/ai/autoresearch/test-output/30pct-isolation-e2e-20260315-190134`

| arm | acc | coh | avg_sample_s | p95_sample_s | avg_swap_s | cold_swap_s | warm_swap_s | inactive_ratio | sigs | same_sig_rows | zero_copy_rows | nonzero_copy_rows | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A dynamic prewarmed | 0.4 | 0.56 | 6.123242 | 23.752683 | 0.058116 | 0.0 | 0.058116 | 0.0 | 1 | 25 | 25 | 0 | invalid |
| B forced static | 0.4 | 0.56 | 5.711052 | 23.792728 | 0.059543 | 0.062227 | 0.059431 | 0.0 | 1 | 24 | 25 | 0 | invalid |
| C cold included (fresh relaunch) | 0.4 | 0.6 | 8.392374 | 22.642617 | 2.516963 | 61.487533 | 0.059856 | 0.024218 | 1 | 24 | 25 | 0 | invalid |

## Verdict
- A and B are functionally the same on accuracy and signature behavior: the measured strict30 path is still acting like a fixed sparse set, not prompt-conditioned dynamic swapping.
- Prewarming removed the first-shrink penalty: arm A avg swap 0.058s vs arm C 2.517s, with arm C cold swap 61.488s and warm swaps 0.060s.
- All three arms stayed zero-copy after the initial shrink path. The current 30pct plan is not triggering added-expert movement during measured prompts.
- Quality stayed equally bad after removing cold-start contamination, so the remaining problem is the static sparse set itself, not steady-state swap overhead.

## Notes
- Gate outputs are invalid because the reused baseline artifact was generated with a different sample-count/signature set; use the arm-to-arm comparison artifacts here instead of the gate verdicts.
- armC was rerun from a fresh server relaunch; the earlier non-fresh armC outputs were moved under pre_reset_armC/.

## Artifacts
- `armA_dynamic_prewarmed.json/.md/.gate.json/.gate.md`
- `armB_forced_static.json/.md/.gate.json/.gate.md`
- `armC_cold_included.json/.md/.gate.json/.gate.md`
- `first-active-set.json`
- `readiness-attempts.json` and `readiness-attempts-armC-fresh.json`
- `remote-post.txt` and `remote-post-armC-fresh.txt`
- `pre_reset_armC/*` contains the discarded non-fresh cold-run outputs
