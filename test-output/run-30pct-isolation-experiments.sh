#!/usr/bin/env bash
set -euo pipefail

# This script is a staged runbook. It assumes the code changes documented in
# test-output/30pct-isolation-next-steps-2026-03-15.md have already been
# applied inside /Users/sero/ai/reap-expert-swap.

REAP_ROOT="/Users/sero/ai/reap-expert-swap"
PAIR_SERVER_URL="${PAIR_SERVER_URL:-http://127.0.0.1:18361}"
PAIR_MODEL="${PAIR_MODEL:-qwen35-dynamic-30pct-pair01}"
PLAN_JSON="${PLAN_JSON:-$REAP_ROOT/test-output/support-set-research-20260311-30pct/best-plan.json}"
OUT_DIR="${OUT_DIR:-/Users/sero/ai/autoresearch/test-output/30pct-isolation-runs-$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-7}"
SAMPLE_COUNT="${SAMPLE_COUNT:-5}"
CALIBRATION_COUNT="${CALIBRATION_COUNT:-0}"
FIRST_ACTIVE_SET_JSON="${FIRST_ACTIVE_SET_JSON:-/tmp/qwen35-30pct-first-active-set.json}"

mkdir -p "$OUT_DIR"

cat <<EOF
[runbook]
OUT_DIR=$OUT_DIR
PAIR_SERVER_URL=$PAIR_SERVER_URL
PAIR_MODEL=$PAIR_MODEL
PLAN_JSON=$PLAN_JSON
FIRST_ACTIVE_SET_JSON=$FIRST_ACTIVE_SET_JSON
EOF

echo "[1/4] Generate the first active-set payload"
cd "$REAP_ROOT"
python3 - <<'PY'
import json
from pathlib import Path
from scripts.evaluate_original_vs_multiplex import load_benchmarks, format_prompt, BenchmarkSpec, stable_id
from scripts.dynamic_reap import build_active_set_payload

PLAN_JSON = Path("/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json")
OUT = Path("/tmp/qwen35-30pct-first-active-set.json")
plan = json.loads(PLAN_JSON.read_text())
rows = load_benchmarks(sample_count=1, calibration_count=0, seed=7)
row = rows[0]
spec_map = {
    "mmlu": BenchmarkSpec("mmlu", "cais/mmlu", "all", "validation", "mcq", 8),
    "arc_challenge": BenchmarkSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation", "mcq", 8),
    "hellaswag": BenchmarkSpec("hellaswag", "Rowan/hellaswag", None, "validation", "mcq", 8),
    "winogrande": BenchmarkSpec("winogrande", "allenai/winogrande", "winogrande_xl", "validation", "mcq_numeric", 8),
    "gsm8k": BenchmarkSpec("gsm8k", "gsm8k", "main", "test", "math", 256),
}
spec = spec_map[row["benchmark"]]
prompt = format_prompt(spec, row)
payload = build_active_set_payload(
    plan,
    prompt,
    request_id=stable_id("dynamic", row["id"], "7"),
    benchmark=row["benchmark"],
    phase="prefill",
)
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(OUT)
PY

echo "[2/4] Arm A - current dynamic with warm-start pre-shrink"
python3 "$REAP_ROOT/scripts/evaluate_original_vs_multiplex.py" \
  --mode dynamic \
  --server-url "$PAIR_SERVER_URL" \
  --model "$PAIR_MODEL" \
  --plan-json "$PLAN_JSON" \
  --sample-count "$SAMPLE_COUNT" \
  --calibration-count "$CALIBRATION_COUNT" \
  --seed "$SEED" \
  --warm-start-active-set-json "$FIRST_ACTIVE_SET_JSON" \
  --output-json "$OUT_DIR/armA-dynamic-prewarmed.json" \
  --output-md "$OUT_DIR/armA-dynamic-prewarmed.md"

echo "[3/4] Arm B - forced static active set"
python3 "$REAP_ROOT/scripts/evaluate_original_vs_multiplex.py" \
  --mode dynamic \
  --server-url "$PAIR_SERVER_URL" \
  --model "$PAIR_MODEL" \
  --plan-json "$PLAN_JSON" \
  --sample-count "$SAMPLE_COUNT" \
  --calibration-count "$CALIBRATION_COUNT" \
  --seed "$SEED" \
  --force-static-active-set-json "$FIRST_ACTIVE_SET_JSON" \
  --output-json "$OUT_DIR/armB-forced-static.json" \
  --output-md "$OUT_DIR/armB-forced-static.md"

echo "[4/4] Arm C - current path with cold first shrink included"
python3 "$REAP_ROOT/scripts/evaluate_original_vs_multiplex.py" \
  --mode dynamic \
  --server-url "$PAIR_SERVER_URL" \
  --model "$PAIR_MODEL" \
  --plan-json "$PLAN_JSON" \
  --sample-count "$SAMPLE_COUNT" \
  --calibration-count "$CALIBRATION_COUNT" \
  --seed "$SEED" \
  --output-json "$OUT_DIR/armC-cold-included.json" \
  --output-md "$OUT_DIR/armC-cold-included.md"

echo "[done] compare the three JSON outputs with the interpretation rules in:"
echo "  /Users/sero/ai/autoresearch/test-output/30pct-isolation-next-steps-2026-03-15.md"
