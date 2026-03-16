#!/usr/bin/env bash
set -euo pipefail
REMOTE="ser@192.168.1.70"
PLAN_LOCAL="/Users/sero/ai/reap-expert-swap/test-output/disagreement-backfill-20260312/combined-backfill-fullrerank.plan.json"
PLAN_REMOTE="/home/ser/reap-expert-swap-reap/.factory/runtime/current-dynamic-plan.json"
BASELINE_JSON="/Users/sero/ai/autoresearch/test-output/sequential-tunneled-2026-03-14/20260314-194006/run-6/baseline.json"
OUT_BASE="/Users/sero/ai/autoresearch/test-output/forensic-next-step-2026-03-15"
RUN="$OUT_BASE/next-run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"

ssh "$REMOTE" "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits" > "$RUN/gpu-before.csv"
scp -q "$PLAN_LOCAL" "$REMOTE:$PLAN_REMOTE"

# HARD RESET (broad)
ssh "$REMOTE" "pkill -9 -f 'vllm_multiplex_server.py' || true; sleep 2; pgrep -af 'vllm_multiplex_server.py' || true" > "$RUN/reset.txt" 2>&1

# RELAUNCH (deterministic)
ssh "$REMOTE" "bash -lc 'export REAP_PLAN_FILE=$PLAN_REMOTE; export CUDA_VISIBLE_DEVICES=2,3; cd /home/ser/reap-expert-swap-reap; nohup /home/ser/reap-expert-swap-vllm016/.venv/bin/python scripts/vllm_multiplex_server.py --model /home/ser/models/Qwen_Qwen3.5-35B-A3B --tensor-parallel-size 2 --dtype half --gpu-memory-utilization 0.60 --enforce-eager --disable-custom-all-reduce --max-model-len 16384 --max-num-seqs 1 --reasoning-parser qwen3 --language-model-only --served-model-name qwen35-dynamic-2gpu-16k --port 8111 --host 127.0.0.1 --cpu-offload-gb 16 --cpu-offload-params experts > /tmp/qwen35-dynamic-8111.log 2>&1 & echo \$! > /tmp/qwen35-dynamic-8111.pid; sleep 15; pgrep -af vllm_multiplex_server.py; tail -n 120 /tmp/qwen35-dynamic-8111.log'" > "$RUN/relaunch.txt" 2>&1 || true

# Tunnel + one-request eval
pkill -f 'ssh -f -N -L 18111:127.0.0.1:8111 ser@192.168.1.70' || true
ssh -f -N -L 18111:127.0.0.1:8111 "$REMOTE" || true
sleep 2
curl -sS http://127.0.0.1:18111/v1/models > "$RUN/models.json" 2> "$RUN/models.stderr" || true

~/.local/bin/uv run --with requests --with datasets python /Users/sero/ai/reap-expert-swap/scripts/evaluate_original_vs_multiplex.py \
  --mode dynamic \
  --server-url http://127.0.0.1:18111 \
  --model qwen35-dynamic-2gpu-16k \
  --plan-json "$PLAN_LOCAL" \
  --sample-count 1 \
  --calibration-count 0 \
  --seed 7 \
  --request-timeout-s 30 \
  --baseline-json "$BASELINE_JSON" \
  --gate-profile dynamic_target \
  --gate-output-json "$RUN/gate.json" \
  --gate-output-md "$RUN/gate.md" \
  --output-json "$RUN/dynamic.json" \
  --output-md "$RUN/dynamic.md" \
  > "$RUN/eval.stdout" 2> "$RUN/eval.stderr" || true

python3 - "$RUN" <<'PY'
import json, os, sys
run=sys.argv[1]
out={}
if os.path.exists(run+'/dynamic.json'):
    d=json.load(open(run+'/dynamic.json'))
    out['overall']=d.get('summary',{}).get('overall',{})
    if d.get('results'):
        r=d['results'][0]
        out['first_error']=r.get('error')
        out['request_id']=r.get('request_id')
        out['swap_request_id']=r.get('swap_request_id')
        out['swap_plan_identity']=r.get('swap_plan_identity')
        out['crash_classification']=r.get('crash_classification')
print(json.dumps(out,indent=2))
json.dump(out,open(run+'/summary.json','w'),indent=2)
PY

echo "$RUN"
