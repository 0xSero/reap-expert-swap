#!/usr/bin/env python3
"""Three-arm comparison: BF16 full vs Dynamic+saliency vs Plain UVA.

Arm 1 (BF16 full):       GPUs 0-3, TP=4, no offload, no REAP
Arm 2 (Dynamic+saliency): GPUs 4,5, TP=2, cpu-offload, REAP plan + masks_only + no router mask
Arm 3 (Plain UVA):        GPUs 6,7, TP=2, cpu-offload, no REAP at all

Arms 2+3 can run in parallel (different GPUs).
All arms: same 25 prompts (5 per benchmark, seed=7), same max-model-len=3072.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remote_auth import resolve_auth_config
from reap_swap.dynamic_reap import build_active_set_payload
from reap_swap.evaluate_original_vs_multiplex import (
    BenchmarkSpec, format_prompt, load_benchmarks, stable_id,
)

SPEC_MAP = {
    "mmlu": BenchmarkSpec("mmlu", "cais/mmlu", "all", "validation", "mcq", 8),
    "arc_challenge": BenchmarkSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation", "mcq", 8),
    "hellaswag": BenchmarkSpec("hellaswag", "Rowan/hellaswag", None, "validation", "mcq", 8),
    "winogrande": BenchmarkSpec("winogrande", "allenai/winogrande", "winogrande_xl", "validation", "mcq_numeric", 8),
    "gsm8k": BenchmarkSpec("gsm8k", "gsm8k", "main", "test", "math", 256),
}
MODEL_PATH = "/home/ser/models/Qwen_Qwen3.5-35B-A3B"
REMOTE_REPO = "/home/ser/reap-expert-swap-reap"
REMOTE_VENV_PY = "/home/ser/reap-expert-swap-vllm016/.venv/bin/python"


def run(cmd, *, check=True, capture=False):
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        print(f"STDERR: {proc.stderr[-500:]}" if proc.stderr else "")
        raise RuntimeError(f"rc={proc.returncode}: {' '.join(cmd[:6])}...")
    return proc


def ssh(config, cmd_str):
    return run([*config.ssh_prefix(), cmd_str], check=False)


def ssh_check(config, cmd_str):
    return run([*config.ssh_prefix(), cmd_str])


def scp(config, local, remote_path):
    run(config.scp_prefix() + [str(local), f"{config.host}:{remote_path}"])


def capture_vram(config, out_path, label):
    proc = ssh(config, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits")
    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": int(parts[0]), "used_mib": int(parts[1]), "total_mib": int(parts[2])})
    result = {"label": label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "gpus": gpus}
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def kill_port(config, port):
    ssh(config, f"pids=$(pgrep -f 'vllm_multiplex_server.py.*--port[[:space:]]{port}' 2>/dev/null || true); [ -n \"$pids\" ] && kill -9 $pids 2>/dev/null; sleep 1")
    ssh(config, f"pids=$(pgrep -f 'vllm.*--port[[:space:]]{port}' 2>/dev/null || true); [ -n \"$pids\" ] && kill -9 $pids 2>/dev/null; sleep 1")


def wait_ready(config, port, *, timeout_s=300):
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        proc = ssh(config, f"curl -fsS http://127.0.0.1:{port}/v1/models 2>/dev/null")
        if proc.returncode == 0:
            elapsed = timeout_s - (deadline - time.time())
            print(f"  ready after {elapsed:.0f}s ({attempt} attempts)")
            return True
        remaining = int(deadline - time.time())
        print(f"  attempt {attempt}: not ready, {remaining}s remaining...", flush=True)
        time.sleep(15)
    return False


def launch_server(config, *, gpus, port, tp, model_name, plan_file=None, cpu_offload_gb=0, extra_env="", log_file="/tmp/arm.log", pid_file="/tmp/arm.pid", use_vanilla_vllm=False):
    env_parts = []
    if extra_env:
        env_parts.append(extra_env)
    if plan_file and not use_vanilla_vllm:
        env_parts.append(f"REAP_PLAN_FILE={shlex.quote(plan_file)}")
    env_parts.append(f"CUDA_VISIBLE_DEVICES={gpus}")
    env_str = " ".join(env_parts)

    offload_args = ""
    if cpu_offload_gb > 0:
        offload_args = f"--cpu-offload-gb {cpu_offload_gb} --cpu-offload-params experts"

    if use_vanilla_vllm:
        server_cmd = f"{REMOTE_VENV_PY} -m vllm.entrypoints.openai.api_server"
    else:
        server_cmd = f"{REMOTE_VENV_PY} scripts/vllm_multiplex_server.py"

    cmd = (
        f"cd {REMOTE_REPO} && rm -f {log_file} {pid_file} && "
        f"nohup env {env_str} {server_cmd} "
        f"--model {MODEL_PATH} --host 127.0.0.1 --port {port} "
        f"--tensor-parallel-size {tp} --max-model-len 3072 --max-num-seqs 1 "
        f"--reasoning-parser qwen3 --swap-space 32 --gpu-memory-utilization 0.92 "
        f"--dtype half --enforce-eager --disable-custom-all-reduce --language-model-only "
        f"--served-model-name {shlex.quote(model_name)} {offload_args} "
        f"</dev/null > {log_file} 2>&1 & echo $! > {pid_file}; cat {pid_file}"
    )
    proc = ssh_check(config, cmd)
    pid = proc.stdout.strip().split()[-1]
    print(f"  launched {model_name} on GPUs {gpus} port {port} pid={pid} {'(vanilla vllm)' if use_vanilla_vllm else '(multiplex)'}")
    return pid


def run_eval(config, *, local_port, remote_port, model_name, plan_json, out_dir, arm_slug, warm_start_json=None, mode="dynamic"):
    # Set up tunnel
    tunnel_cmd = [
        "ssh", "-N", "-o", "StrictHostKeyChecking=no", "-o", "ExitOnForwardFailure=yes",
    ]
    if config.ssh_key_path:
        tunnel_cmd.extend(["-i", config.ssh_key_path])
    tunnel_cmd.extend(["-L", f"{local_port}:127.0.0.1:{remote_port}", config.host])
    tunnel_proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        identity = {"service": "three-arm", "host": "127.0.0.1", "port": local_port,
                     "plan_file": str(plan_json) if plan_json else "none", "plan_mode": "dynamic_core_specialist"}
        id_path = out_dir / "runtime-readiness-identity.json"
        id_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

        env = os.environ.copy()
        env.update({
            "REAP_RUNTIME_READINESS_HOST_ALLOWLIST": "127.0.0.1",
            "REAP_RUNTIME_READINESS_PORT": str(local_port),
            "REAP_RUNTIME_READINESS_IDENTITY_PATH": str(id_path),
        })
        cmd = [
            sys.executable, "-m", "reap_swap.evaluate_original_vs_multiplex",
            "--mode", mode,
            "--server-url", f"http://127.0.0.1:{local_port}",
            "--model", model_name,
            "--plan-json", str(plan_json),
            "--sample-count", "5",
            "--calibration-count", "0",
            "--seed", "7",
            "--request-timeout-s", "300",
            "--output-json", str(out_dir / f"{arm_slug}.json"),
            "--output-md", str(out_dir / f"{arm_slug}.md"),
        ]
        if warm_start_json and mode == "dynamic":
            cmd.extend(["--warm-start-active-set-json", str(warm_start_json)])

        stdout_path = out_dir / f"{arm_slug}.stdout"
        stderr_path = out_dir / f"{arm_slug}.stderr"
        with stdout_path.open("w") as fout, stderr_path.open("w") as ferr:
            proc = subprocess.run(cmd, env=env, text=True, stdout=fout, stderr=sys.stderr, check=False)
        if proc.returncode != 0:
            print(f"  WARNING: eval {arm_slug} exited with rc={proc.returncode}")
        return proc.returncode
    finally:
        tunnel_proc.terminate()
        try:
            tunnel_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel_proc.kill()


def extract_metrics(out_dir, arm_slug):
    path = out_dir / f"{arm_slug}.json"
    if not path.exists():
        return {"error": "no output json"}
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d.get("summary", {})
    overall = s.get("overall", {})
    by_bm = s.get("by_benchmark", {})
    return {
        "accuracy": overall.get("accuracy"),
        "coherence_rate": overall.get("coherence_rate"),
        "bench_time_s": overall.get("bench_time_s"),
        "avg_sample_time_s": overall.get("avg_sample_time_s"),
        "avg_swap_time_s": overall.get("avg_swap_time_s"),
        "avg_completion_tokens_per_s": overall.get("avg_completion_tokens_per_s"),
        "avg_router_miss_inactive_ratio": overall.get("avg_router_miss_inactive_ratio"),
        "dynamic_signature_count": overall.get("dynamic_signature_count"),
        "avg_swap_bytes_copied": overall.get("avg_swap_bytes_copied"),
        "by_benchmark": {
            bm: {"accuracy": bd.get("accuracy"), "avg_sample_time_s": bd.get("avg_sample_time_s")}
            for bm, bd in by_bm.items()
        },
    }


def make_first_active_set(plan_path, out_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = load_benchmarks(sample_count=1, calibration_count=0, seed=7)
    row = rows[0]
    spec = SPEC_MAP[row["benchmark"]]
    prompt = format_prompt(spec, row)
    payload = build_active_set_payload(plan, prompt, request_id=stable_id("dynamic", row["id"], "7"),
                                        benchmark=row["benchmark"], phase="prefill")
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_summary(out_dir, arms_config, arms_metrics, vram_data):
    summary = {"arms": {}, "vram": vram_data}
    for slug, cfg in arms_config.items():
        summary["arms"][slug] = {**cfg, **arms_metrics.get(slug, {})}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["# Three-Arm Comparison: BF16 vs Dynamic+Saliency vs Plain UVA", ""]

    # VRAM table
    lines.extend(["## VRAM Usage", ""])
    for label, snap in sorted(vram_data.items()):
        lines.append(f"**{label}:**")
        for g in snap.get("gpus", []):
            lines.append(f"  GPU {g['index']}: {g['used_mib']} / {g['total_mib']} MiB")
        lines.append("")

    # Results table
    lines.extend(["## Results", "", "| Metric | BF16 Full (4x3090) | Dynamic+Saliency (2x3090) | Plain UVA (2x3090) |", "|---|---|---|---|"])
    keys = ["accuracy", "coherence_rate", "bench_time_s", "avg_sample_time_s", "avg_swap_time_s",
            "avg_completion_tokens_per_s", "avg_router_miss_inactive_ratio", "dynamic_signature_count", "avg_swap_bytes_copied"]
    slugs = ["arm1_bf16", "arm2_dynamic", "arm3_uva"]
    for key in keys:
        vals = []
        for slug in slugs:
            v = arms_metrics.get(slug, {}).get(key, "n/a")
            if isinstance(v, float):
                v = f"{v:.4f}" if v < 1 else f"{v:.1f}"
            vals.append(str(v))
        lines.append(f"| {key} | {' | '.join(vals)} |")

    # Per-benchmark
    lines.extend(["", "## Per-Benchmark Accuracy", "", "| Benchmark | BF16 | Dynamic | UVA |", "|---|---|---|---|"])
    for bm in ["mmlu", "arc_challenge", "hellaswag", "winogrande", "gsm8k"]:
        vals = []
        for slug in slugs:
            bd = arms_metrics.get(slug, {}).get("by_benchmark", {}).get(bm, {})
            vals.append(str(bd.get("accuracy", "n/a")))
        lines.append(f"| {bm} | {' | '.join(vals)} |")

    # Per-benchmark latency
    lines.extend(["", "## Per-Benchmark Avg Sample Time (s)", "", "| Benchmark | BF16 | Dynamic | UVA |", "|---|---|---|---|"])
    for bm in ["mmlu", "arc_challenge", "hellaswag", "winogrande", "gsm8k"]:
        vals = []
        for slug in slugs:
            bd = arms_metrics.get(slug, {}).get("by_benchmark", {}).get(bm, {})
            v = bd.get("avg_sample_time_s", "n/a")
            vals.append(f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
        lines.append(f"| {bm} | {' | '.join(vals)} |")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", default="ser@192.168.1.70")
    parser.add_argument("--ssh-key-path", default=os.environ.get("REMOTE_SSH_KEY") or str(Path.home() / ".ssh/linux-ai"))
    parser.add_argument("--plan-json", default=str(REPO_ROOT / "assets" / "strict30-v2-plan.json"))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = str(REPO_ROOT / "test-output" / f"three-arm-{time.strftime('%Y%m%d-%H%M%S')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_auth_config(host=args.remote_host, auth_mode="auto", ssh_key_path=args.ssh_key_path)
    plan_local = Path(args.plan_json).resolve()

    # Sync code + plan to remote
    print("Syncing code and plan to remote...")
    import tempfile
    local_src = REPO_ROOT / "reap_swap" / "vllm_multiplex_server.py"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        src = local_src.read_text()
        src = src.replace("from .dynamic_swap_delta", "from dynamic_swap_delta")
        src = src.replace("from .multiplex_cache", "from multiplex_cache")
        src = src.replace("from .dynamic_reap", "from dynamic_reap")
        tmp.write(src)
        tmp_path = tmp.name
    scp(config, tmp_path, f"{REMOTE_REPO}/scripts/vllm_multiplex_server.py")
    scp(config, tmp_path, f"{REMOTE_REPO}/vllm_multiplex_server.py")
    os.unlink(tmp_path)
    scp(config, plan_local, "/tmp/three-arm-plan.json")
    ssh(config, f"find {REMOTE_REPO} -name '__pycache__' -type d -exec rm -rf {{}} + 2>/dev/null; echo ok")

    # Build first active set for warm start
    first_as = out_dir / "first-active-set.json"
    make_first_active_set(plan_local, first_as)

    # Capture baseline VRAM
    vram_data = {}
    vram_data["baseline"] = capture_vram(config, out_dir / "vram_baseline.json", "baseline")

    arms_config = {
        "arm1_bf16": {"label": "BF16 Full", "gpus": "0,1,2,3", "tp": 4, "port": 8370, "local_port": 18370,
                      "cpu_offload_gb": 0, "plan": None, "extra_env": "", "model_name": "qwen35-bf16-full"},
        "arm2_dynamic": {"label": "Dynamic+Saliency", "gpus": "4,5", "tp": 2, "port": 8371, "local_port": 18371,
                         "cpu_offload_gb": 28, "plan": "/tmp/three-arm-plan.json",
                         "extra_env": "REAP_SWAP_MASKS_ONLY=1 REAP_ENABLE_ROUTER_MASKS=0",
                         "model_name": "qwen35-dynamic-saliency"},
        "arm3_uva": {"label": "Plain UVA", "gpus": "6,7", "tp": 2, "port": 8372, "local_port": 18372,
                     "cpu_offload_gb": 28, "plan": "/tmp/three-arm-plan.json",
                     "extra_env": "REAP_SWAP_MASKS_ONLY=1 REAP_ENABLE_ROUTER_MASKS=0",
                     "model_name": "qwen35-plain-uva"},
    }

    arms_metrics = {}

    # --- ARM 1: BF16 FULL (4x 3090, no offload) ---
    print("\n=== ARM 1: BF16 Full (GPUs 0-3, TP=4, no offload) ===")
    cfg = arms_config["arm1_bf16"]
    kill_port(config, cfg["port"])
    launch_server(config, gpus=cfg["gpus"], port=cfg["port"], tp=cfg["tp"],
                  model_name=cfg["model_name"], cpu_offload_gb=cfg["cpu_offload_gb"],
                  plan_file=cfg["plan"], extra_env=cfg["extra_env"],
                  log_file="/tmp/arm1_bf16.log", pid_file="/tmp/arm1_bf16.pid",
                  use_vanilla_vllm=True)
    print("  waiting for BF16 server...")
    if not wait_ready(config, cfg["port"], timeout_s=420):
        print("  ERROR: BF16 server did not become ready")
        ssh(config, "tail -30 /tmp/arm1_bf16.log")
    else:
        vram_data["arm1_loaded"] = capture_vram(config, out_dir / "vram_arm1_loaded.json", "arm1_loaded")
        print("  running eval...")
        run_eval(config, local_port=cfg["local_port"], remote_port=cfg["port"],
                 model_name=cfg["model_name"], plan_json=plan_local,
                 out_dir=out_dir, arm_slug="arm1_bf16", mode="baseline")
        vram_data["arm1_after_eval"] = capture_vram(config, out_dir / "vram_arm1_after.json", "arm1_after_eval")
        arms_metrics["arm1_bf16"] = extract_metrics(out_dir, "arm1_bf16")
        print(f"  accuracy: {arms_metrics['arm1_bf16'].get('accuracy')}")

    # Kill arm1 to free GPUs
    kill_port(config, cfg["port"])
    ssh(config, "kill -9 $(cat /tmp/arm1_bf16.pid 2>/dev/null) 2>/dev/null; sleep 2")

    # --- ARM 2 + ARM 3: launch in parallel ---
    print("\n=== ARM 2: Dynamic+Saliency (GPUs 4,5, TP=2, cpu-offload=28) ===")
    cfg2 = arms_config["arm2_dynamic"]
    kill_port(config, cfg2["port"])
    launch_server(config, gpus=cfg2["gpus"], port=cfg2["port"], tp=cfg2["tp"],
                  model_name=cfg2["model_name"], cpu_offload_gb=cfg2["cpu_offload_gb"],
                  plan_file=cfg2["plan"], extra_env=cfg2["extra_env"],
                  log_file="/tmp/arm2_dynamic.log", pid_file="/tmp/arm2_dynamic.pid")

    print("\n=== ARM 3: Plain UVA (GPUs 6,7, TP=2, cpu-offload=28) ===")
    cfg3 = arms_config["arm3_uva"]
    kill_port(config, cfg3["port"])
    launch_server(config, gpus=cfg3["gpus"], port=cfg3["port"], tp=cfg3["tp"],
                  model_name=cfg3["model_name"], cpu_offload_gb=cfg3["cpu_offload_gb"],
                  plan_file=cfg3["plan"], extra_env=cfg3["extra_env"],
                  log_file="/tmp/arm3_uva.log", pid_file="/tmp/arm3_uva.pid")

    print("  waiting for both servers...")
    ready2 = wait_ready(config, cfg2["port"], timeout_s=420)
    ready3 = wait_ready(config, cfg3["port"], timeout_s=420)

    if not ready2:
        print("  ERROR: arm2 server did not become ready")
        ssh(config, "tail -30 /tmp/arm2_dynamic.log")
    if not ready3:
        print("  ERROR: arm3 server did not become ready")
        ssh(config, "tail -30 /tmp/arm3_uva.log")

    if ready2 and ready3:
        vram_data["arm2_3_loaded"] = capture_vram(config, out_dir / "vram_arm2_3_loaded.json", "arm2_3_loaded")

        # Run evals sequentially (different tunnels)
        print("  running arm2 eval...")
        run_eval(config, local_port=cfg2["local_port"], remote_port=cfg2["port"],
                 model_name=cfg2["model_name"], plan_json=plan_local,
                 out_dir=out_dir, arm_slug="arm2_dynamic", warm_start_json=first_as)
        arms_metrics["arm2_dynamic"] = extract_metrics(out_dir, "arm2_dynamic")
        print(f"  arm2 accuracy: {arms_metrics['arm2_dynamic'].get('accuracy')}")

        print("  running arm3 eval...")
        run_eval(config, local_port=cfg3["local_port"], remote_port=cfg3["port"],
                 model_name=cfg3["model_name"], plan_json=plan_local,
                 out_dir=out_dir, arm_slug="arm3_uva", warm_start_json=first_as)
        arms_metrics["arm3_uva"] = extract_metrics(out_dir, "arm3_uva")
        print(f"  arm3 accuracy: {arms_metrics['arm3_uva'].get('accuracy')}")

        vram_data["arm2_3_after_eval"] = capture_vram(config, out_dir / "vram_arm2_3_after.json", "arm2_3_after_eval")

    # Cleanup
    for p in [cfg2["port"], cfg3["port"]]:
        kill_port(config, p)
    ssh(config, "kill -9 $(cat /tmp/arm2_dynamic.pid 2>/dev/null) $(cat /tmp/arm3_uva.pid 2>/dev/null) 2>/dev/null")

    # Build summary
    summary = build_summary(out_dir, arms_config, arms_metrics, vram_data)
    print(f"\n{'='*60}")
    print(f"Results: {out_dir}")
    print(f"{'='*60}")
    print((out_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
