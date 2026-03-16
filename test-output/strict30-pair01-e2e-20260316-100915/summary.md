# strict30 pair01 three-arm summary

out_dir: `test-output/strict30-pair01-e2e-20260316-100915`

| arm | acc | coh | avg_sample_s | avg_swap_s | cold_swap_s | warm_swap_s | sigs | same_sig_rows | zero_copy_rows | nonzero_copy_rows | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A dynamic prewarmed | 0.2 | 0.56 | 8.053765 | 0.30324 | 0.0 | 0.30324 | 25 | 0 | 0 | 25 | None |
| B forced static | 0.24 | 0.6 | 2.937906 | 0.062375 | 0.354367 | 0.050209 | 1 | 24 | 24 | 1 | None |
| C cold included (fresh relaunch) | 0.2 | 0.56 | 5.375457 | 0.311312 | 0.408084 | 0.307279 | 25 | 0 | 0 | 25 | None |

## Verdict
- Prewarming isolates the first-shrink tax: warm steady-state swaps are cheap, cold startup is the latency distortion.

## Notes
- Gate outputs remain invalid if the supplied BF16 baseline artifact does not match the current sample-count/signature set.
