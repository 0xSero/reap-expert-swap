#!/usr/bin/env bash
set -euo pipefail
REMOTE="ser@192.168.1.70"
REMOTE_REPO="/home/ser/reap-expert-swap-reap"
REMOTE_VENV_PY="/home/ser/reap-expert-swap-vllm016/.venv/bin/python"
PLAN_LOCAL="/Users/sero/ai/reap-expert-swap/test-output/disagreement-backfill-20260312/combined-backfill-fullrerank.plan.json"
BASELINE_JSON="/Users/sero/ai/autoresearch/test-output/sequential-tunneled-2026-03-14/20260314-194006/run-6/baseline.json"
RUN_DIR="/Users/sero/ai/autoresearch/test-output/one-request-forensic-live-2026-03-15/feasible-envelope-20260315-070839"
TAG="forensic-feasible-$(basename "$RUN_DIR")"
REMOTE_PLAN="/tmp/${TAG}.plan.json"
REMOTE_LOG="/tmp/${TAG}.log"
REMOTE_PID="/tmp/${TAG}.pid"
REMOTE_PORT=8337
LOCAL_PORT=18337
MODEL_NAME="qwen35-multiplex-offload-2gpu-3k"

exec > >(tee "$RUN_DIR/full-command.log") 2>&1
set -x

date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ" | tee "$RUN_DIR/start.txt"
printf '%s\n' "run_dir=$RUN_DIR" "plan_local=$PLAN_LOCAL" "baseline_json=$BASELINE_JSON" "remote_repo=$REMOTE_REPO" > "$RUN_DIR/context.txt"

shasum -a 256 "$PLAN_LOCAL" | tee "$RUN_DIR/plan_sha_local.txt"
scp -q "$PLAN_LOCAL" "$REMOTE:$REMOTE_PLAN"
ssh "$REMOTE" "sha256sum '$REMOTE_PLAN'" | tee "$RUN_DIR/plan_sha_remote.txt"

# Crash-safe targeted cleanup for prior processes on this port only.
ssh "$REMOTE" "bash -lc 'set -euo pipefail; pgrep -af \"vllm_multiplex_server.py.*--port $REMOTE_PORT\" > /tmp/${TAG}.pre_pgrep.txt || true; if [ -s /tmp/${TAG}.pre_pgrep.txt ]; then awk \"{print \\\$1}\" /tmp/${TAG}.pre_pgrep.txt | xargs -r kill -9; sleep 2; fi; pgrep -af \"vllm_multiplex_server.py.*--port $REMOTE_PORT\" || true'" | tee "$RUN_DIR/reset.log"

ssh "$REMOTE" "bash -lc 'set -euo pipefail; cd $REMOTE_REPO; rm -f $REMOTE_LOG $REMOTE_PID; nohup env REAP_PLAN_FILE=$REMOTE_PLAN CUDA_VISIBLE_DEVICES=2,3 $REMOTE_VENV_PY scripts/vllm_multiplex_server.py --model /home/ser/models/Qwen_Qwen3.5-35B-A3B --host 127.0.0.1 --port $REMOTE_PORT --tensor-parallel-size 2 --max-model-len 3072 --max-num-seqs 1 --reasoning-parser qwen3 --cpu-offload-gb 28 --swap-space 32 --gpu-memory-utilization 0.90 --dtype half --enforce-eager --disable-custom-all-reduce --language-model-only --served-model-name $MODEL_NAME --cpu-offload-params experts > $REMOTE_LOG 2>&1 & echo \$! > $REMOTE_PID; echo launch_pid:\$(cat $REMOTE_PID); pgrep -af \"vllm_multiplex_server.py.*--port $REMOTE_PORT\" || true'" | tee "$RUN_DIR/launch.log"

pkill -f "ssh -f -N -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} ${REMOTE}" || true
ssh -f -N -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} "$REMOTE"
printf 'local_tunnel_port=%s\n' "$LOCAL_PORT" > "$RUN_DIR/tunnel.txt"

ready=0
for i in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" > "$RUN_DIR/readiness-response.json" 2>"$RUN_DIR/readiness-curl.stderr"; then
    ready=1
    echo "ready_attempt=$i" > "$RUN_DIR/readiness.txt"
    break
  fi
  echo "attempt=$i not ready" >> "$RUN_DIR/readiness-probe.txt"
  sleep 15
done
if [ "$ready" -ne 1 ]; then
  echo "ready=false" >> "$RUN_DIR/readiness.txt"
  ssh "$REMOTE" "bash -lc 'echo pid_file:; cat $REMOTE_PID 2>/dev/null || true; echo proc:; pgrep -af \"vllm_multiplex_server.py.*--port $REMOTE_PORT\" || true; echo log_tail:; tail -n 240 $REMOTE_LOG 2>/dev/null || true'" > "$RUN_DIR/remote-post.txt" 2>&1 || true
  python3 - <<PY
import json, pathlib
run=pathlib.Path('$RUN_DIR')
verdict={
  'attempt_policy':'single_bounded_attempt',
  'success':False,
  'failure_mode':'not_ready',
  'ready':False,
  'run_dir':str(run),
}
(run/'verdict.json').write_text(json.dumps(verdict, indent=2)+"\n", encoding='utf-8')
(run/'verdict.md').write_text("# Verdict\n- success: false\n- failure_mode: not_ready\n- ready: false\n", encoding='utf-8')
PY
  exit 0
fi

# Deterministic replay using one pinned sample count.
~/.local/bin/uv run --with requests --with datasets python /Users/sero/ai/reap-expert-swap/scripts/evaluate_original_vs_multiplex.py \
  --mode dynamic \
  --server-url "http://127.0.0.1:${LOCAL_PORT}" \
  --model "$MODEL_NAME" \
  --plan-json "$PLAN_LOCAL" \
  --sample-count 1 \
  --calibration-count 0 \
  --seed 7 \
  --request-timeout-s 120 \
  --baseline-json "$BASELINE_JSON" \
  --gate-profile dynamic_target \
  --gate-output-json "$RUN_DIR/gate.json" \
  --gate-output-md "$RUN_DIR/gate.md" \
  --output-json "$RUN_DIR/dynamic.json" \
  --output-md "$RUN_DIR/dynamic.md" \
  > "$RUN_DIR/eval.stdout" 2> "$RUN_DIR/eval.stderr"

# Extract first-request forensic payload for collector normalization.
python3 - <<PY
import json, pathlib
run=pathlib.Path('$RUN_DIR')
d=json.loads((run/'dynamic.json').read_text(encoding='utf-8'))
results=d.get('results') or []
row=results[0] if results else {}
fp=row.get('forensic_payload') or {}
req_id=row.get('request_id')
(run/'request.json').write_text(json.dumps({'request_id': req_id}, indent=2)+"\n", encoding='utf-8')
(run/'swap_response.json').write_text(json.dumps({
  'request_id': row.get('swap_request_id'),
  'plan_identity': row.get('swap_plan_identity'),
}, indent=2)+"\n", encoding='utf-8')
completion={}
resp=row.get('response')
if isinstance(resp, dict):
  completion=resp
elif isinstance(resp, str) and resp:
  completion={'text': resp}
(run/'completion_response.json').write_text(json.dumps(completion, indent=2)+"\n", encoding='utf-8')
(run/'router_misses.json').write_text(json.dumps(row.get('router_misses'), indent=2)+"\n", encoding='utf-8')
forensic={
  'plan_identity': {'plan_sha256': fp.get('plan_sha256')},
  'active_set': {
    'signature': fp.get('active_set_signature'),
    'union_validation': fp.get('union_validation'),
    'core_presence_summary': fp.get('core_presence_summary'),
  },
  'worker': {'swap_result': fp.get('worker_swap_result')},
  'crash': {'classification': fp.get('crash_classification')},
}
(run/'forensic.json').write_text(json.dumps(forensic, indent=2)+"\n", encoding='utf-8')
summary={
  'total_results': len(results),
  'first_request_id': req_id,
  'first_swap_request_id': row.get('swap_request_id'),
  'first_crash_classification': row.get('crash_classification'),
  'first_error': row.get('error'),
}
(run/'dynamic-summary.json').write_text(json.dumps(summary, indent=2)+"\n", encoding='utf-8')
PY

python3 /Users/sero/ai/autoresearch/one_request_forensic_replay.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --strict \
  > "$RUN_DIR/collector.stdout" 2> "$RUN_DIR/collector.stderr" || echo $? > "$RUN_DIR/collector.rc"
[ -f "$RUN_DIR/collector.rc" ] || echo 0 > "$RUN_DIR/collector.rc"

ssh "$REMOTE" "bash -lc 'echo pid_file:; cat $REMOTE_PID 2>/dev/null || true; echo proc_after_eval:; pgrep -af \"vllm_multiplex_server.py.*--port $REMOTE_PORT\" || true; echo models_probe:; curl -sS http://127.0.0.1:$REMOTE_PORT/v1/models || true; echo; echo log_tail:; tail -n 260 $REMOTE_LOG 2>/dev/null || true'" > "$RUN_DIR/remote-post.txt" 2>&1 || true

python3 - <<PY
import json, pathlib
run=pathlib.Path('$RUN_DIR')
collector_rc=int((run/'collector.rc').read_text().strip())
dyn=json.loads((run/'dynamic.json').read_text())
gate=json.loads((run/'gate.json').read_text())
forensic=json.loads((run/'forensic_bundle.json').read_text())
ready=('ready_attempt=' in (run/'readiness.txt').read_text())
results=dyn.get('results') or []
strict_ok = collector_rc == 0
poison=bool((forensic.get('poison') or {}).get('detected'))
missing=len(((forensic.get('expected_fields') or {}).get('missing') or []))
crash=(forensic.get('classification') or {}).get('crash')
success = ready and bool(results) and strict_ok and (not poison) and crash == 'none'
failure_mode = 'none' if success else (
  'collector_strict_missing_fields' if collector_rc == 3 else
  'collector_poison_or_engine_dead' if collector_rc == 2 else
  'no_results' if not results else
  'unknown_failure'
)
verdict={
  'attempt_policy':'single_bounded_attempt',
  'success': success,
  'failure_mode': failure_mode,
  'ready': ready,
  'dynamic_results': len(results),
  'collector_rc': collector_rc,
  'forensic_missing_fields': missing,
  'forensic_crash_classification': crash,
  'forensic_poison_detected': poison,
  'gate_verdict': gate.get('verdict'),
  'plan_path_used': dyn.get('plan_identity',{}).get('plan_path'),
  'plan_sha256_used': dyn.get('plan_identity',{}).get('plan_sha256'),
  'server_url': dyn.get('runtime_identity',{}).get('server_url'),
  'model': dyn.get('model'),
}
(run/'verdict.json').write_text(json.dumps(verdict, indent=2)+"\n", encoding='utf-8')
lines=[
  '# Verdict',
  f"- success: `{str(success).lower()}`",
  f"- failure_mode: `{failure_mode}`",
  f"- ready: `{ready}`",
  f"- dynamic_results: `{len(results)}`",
  f"- collector_rc: `{collector_rc}`",
  f"- forensic_missing_fields: `{missing}`",
  f"- forensic_crash_classification: `{crash}`",
  f"- gate_verdict: `{gate.get('verdict')}`",
  f"- plan_sha256_used: `{verdict['plan_sha256_used']}`",
]
(run/'verdict.md').write_text('\n'.join(lines)+"\n", encoding='utf-8')
proof=[
  '# One-request forensic replay proof',
  '',
  '## Envelope',
  '- `CUDA_VISIBLE_DEVICES=2,3`',
  '- `tensor-parallel-size=2`',
  '- `max-model-len=3072`',
  '- `max-num-seqs=1`',
  '- `reasoning-parser=qwen3`',
  '- `cpu-offload-gb=28`',
  '- `swap-space=32`',
  '- `gpu-memory-utilization=0.90`',
  '',
  '## Attempt policy',
  '- single bounded attempt (no relaunch retries)',
  '',
  '## Required artifacts',
]
for name in ['dynamic.json','gate.json','forensic_bundle.json','forensic_visual.md','proof.md','verdict.json','verdict.md','full-command.log']:
  proof.append(f"- {name}: {'present' if (run/name).exists() else 'missing'}")
proof += [
  '',
  '## Outcome',
  f"- success: `{str(success).lower()}`",
  f"- failure_mode: `{failure_mode}`",
  f"- collector_rc: `{collector_rc}`",
]
(run/'proof.md').write_text('\n'.join(proof)+"\n", encoding='utf-8')
PY

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee "$RUN_DIR/end.txt"
echo "$RUN_DIR"
