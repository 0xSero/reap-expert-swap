# One-request forensic replay proof

## Envelope
- `CUDA_VISIBLE_DEVICES=4,5`
- `tensor-parallel-size=2`
- `max-model-len=3072`
- `max-num-seqs=1`
- `reasoning-parser=qwen3`
- `cpu-offload-gb=28`
- `swap-space=32`
- `gpu-memory-utilization=0.90`

## Attempt policy
- single bounded attempt

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
- success: `true`
- failure_mode: `none`
- collector_rc: `0`
- gate_verdict: `invalid`
