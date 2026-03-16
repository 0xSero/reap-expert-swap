# One-request forensic replay proof

## Envelope
- `CUDA_VISIBLE_DEVICES=2,3`
- `tensor-parallel-size=2`
- `max-model-len=3072`
- `max-num-seqs=1`
- `reasoning-parser=qwen3`
- `cpu-offload-gb=28`
- `swap-space=32`
- `gpu-memory-utilization=0.90`

## Attempt policy
- single bounded attempt (no relaunch retries)

## Required artifacts
- dynamic.json: present
- gate.json: present
- forensic_bundle.json: present
- forensic_visual.md: present
- proof.md: present
- verdict.json: present
- verdict.md: present
- full-command.log: present

## Outcome
- success: `false`
- failure_mode: `not_ready`
- collector_rc: `3`
- gate_verdict: `invalid`
