#!/usr/bin/env python3
"""Two-arm test for REAP_SWAP_MASKS_ONLY mode.

Arms:
  A: masks_only dynamic (REAP_SWAP_MASKS_ONLY=1) -- the new mode
  B: zero_copy dynamic  (REAP_SWAP_MASKS_ONLY=0) -- the old mode

Both use the same plan, same prompts, same server config.
Server is relaunched between arms so each starts fresh.

Captures:
  - nvidia-smi VRAM snapshot before and after warm-start
  - accuracy, coherence, swap time, router miss ratio, signature count
  - swap_mode field from forensics to prove the right path ran
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
    BenchmarkSpec,
    format_prompt,
    load_benchmarks,
    stable_id,
)


def run(cmd, *, cwd=None, env=None, check=True, stdout=None, stderr=None):
    out_h = stdout.open("w", encoding="utf-8") if stdout else subprocess.PIPE
    err_h = stderr.open("w", encoding="utf-8") if stderr else subprocess.PIPE
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=out_h, stderr=err_h, check=False)
    finally:
        if stdout:
            out_h.close()
        if stderr:
            err_h.close()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return proc


def ssh_cmd(config, remote_command):
    return [*config.ssh_prefix(), remote_command]


def tunnel_cmd(config, *, local_port, remote_port):
    cmd = ["ssh", "-N", "-o", "StrictHostKeyChecking=no", "-o", "ExitOnForwardFailure=yes"]
    if config.ssh_key_path:
        cmd.extend(["-i", config.ssh_key_path])
    cmd.extend(["-L", f"{local_port}:127.0.0.1:{remote_port}", config.host])
    return cmd


def stop_remote(config, *, remote_pid, remote_port):
    command = f"set -e; if [ -f {shlex.quote(remote_pid)} ]; then pid=$(cat {shlex.quote(remote_pid)} || true); if [ -n \"$pid\" ]; then kill \"$pid\" 2>/dev/null || true; sleep 2; kill -9 \"$pid\" 2>/dev/null || true; fi; fi; pids=$(pgrep -f 'vllm_multiplex_server.py.*--port[[:space:]]{remote_port}' || true); if [ -n \"$pids\" ]; then kill $pids 2>/dev/null || true; sleep 2; kill -9 $pids 2>/dev/null || true; fi"
    run(ssh_cmd(config, command), check=False)


def start_remote(config, *, remote_repo, remote_venv_python, remote_plan, remote_pid, remote_log, gpus, remote_port, model_name, masks_only: bool):
    swap_env = "REAP_SWAP_MASKS_ONLY=1 " if masks_only else ""
    command = (
        f"cd {shlex.quote(remote_repo)} && "
        f"nohup env {swap_env}REAP_PLAN_FILE={shlex.quote(remote_plan)} CUDA_VISIBLE_DEVICES={shlex.quote(gpus)} "
        f"{shlex.quote(remote_venv_python)} scripts/vllm_multiplex_server.py "
        f"--model /home/ser/models/Qwen_Qwen3.5-35B-A3B --host 127.0.0.1 --port {remote_port} "
        f"--tensor-parallel-size 2 --max-model-len 3072 --max-num-seqs 1 --reasoning-parser qwen3 "
        f"--cpu-offload-gb 28 --swap-space 32 --gpu-memory-utilization 0.90 --dtype half "
        f"--enforce-eager --disable-custom-all-reduce --language-model-only "
        f"--served-model-name {shlex.quote(model_name)} --cpu-offload-params experts "
        f"</dev/null > {shlex.quote(remote_log)} 2>&1 & echo $! > {shlex.quote(remote_pid)}"
    )
    run(ssh_cmd(config, command))


def capture_nvidia_smi(config, out_path: Path, label: str):
    """Capture nvidia-smi output from remote host."""
    proc = subprocess.run(
        ssh_cmd(config, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader,nounits"),
        text=True, capture_output=True, check=False,
    )
    result = {"label": label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "raw": proc.stdout.strip(), "gpus": []}
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            result["gpus"].append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
            })
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def capture_remote_log_tail(config, remote_log, out_path: Path, lines=50):
    proc = subprocess.run(
        ssh_cmd(config, f"tail -n {lines} {shlex.quote(remote_log)} 2>/dev/null || true"),
        text=True, capture_output=True, check=False,
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    return proc.stdout


def wait_ready(local_port, *, out_dir, label):
    for attempt in range(1, 31):
        proc = subprocess.run(["curl", "-fsS", f"http://127.0.0.1:{local_port}/v1/models"], text=True, capture_output=True)
        if proc.returncode == 0:
            return
        time.sleep(10)
    raise RuntimeError(f"runtime on local port {local_port} never became ready for {label}")


def make_first_active_set(plan_local: Path, out_path: Path):
    plan = json.loads(plan_local.read_text(encoding="utf-8"))
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
    payload = build_active_set_payload(plan, prompt, request_id=stable_id("dynamic", row["id"], "7"), benchmark=row["benchmark"], phase="prefill")
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def arm_eval_command(*, local_port, model_name, plan_local, out_dir, arm_slug, extra_args):
    env = os.environ.copy()
    env.update({
        "REAP_RUNTIME_READINESS_HOST_ALLOWLIST": "127.0.0.1",
        "REAP_RUNTIME_READINESS_PORT": str(local_port),
        "REAP_RUNTIME_READINESS_IDENTITY_PATH": str(out_dir / "runtime-readiness-identity.json"),
    })
    cmd = [
        sys.executable, "-m", "reap_swap.evaluate_original_vs_multiplex",
        "--mode", "dynamic",
        "--server-url", f"http://127.0.0.1:{local_port}",
        "--model", model_name,
        "--plan-json", str(plan_local),
        "--sample-count", "5",
        "--calibration-count", "0",
        "--seed", "7",
        "--request-timeout-s", "180",
        "--output-json", str(out_dir / f"{arm_slug}.json"),
        "--output-md", str(out_dir / f"{arm_slug}.md"),
        *extra_args,
    ]
    return cmd, env


def extract_arm_metrics(out_dir: Path, arm_slug: str) -> dict[str, Any]:
    payload = json.loads((out_dir / f"{arm_slug}.json").read_text(encoding="utf-8"))
    overall = (payload.get("summary") or {}).get("overall") or {}
    warm = payload.get("warm_start_result") or {}
    swap_modes = set()
    for row in payload.get("results", []):
        fp = row.get("forensic_payload") or {}
        wsr = fp.get("worker_swap_result") or {}
        sm = wsr.get("swap_mode")
        if sm:
            swap_modes.add(str(sm))
    return {
        "accuracy": overall.get("accuracy"),
        "coherence_rate": overall.get("coherence_rate"),
        "avg_sample_time_s": overall.get("avg_sample_time_s"),
        "avg_swap_time_s": overall.get("avg_swap_time_s"),
        "avg_warm_swap_time_s": overall.get("avg_warm_swap_time_s"),
        "avg_router_miss_inactive_ratio": overall.get("avg_router_miss_inactive_ratio"),
        "dynamic_signature_count": overall.get("dynamic_signature_count"),
        "rows_with_nonzero_swap_copies": overall.get("rows_with_nonzero_swap_copies"),
        "rows_with_zero_copy_swap": overall.get("rows_with_zero_copy_swap"),
        "warm_start_swap_time_s": warm.get("swap_time_s"),
        "observed_swap_modes": sorted(swap_modes),
        "total_bytes_copied": sum(r.get("swap_bytes_copied", 0) for r in payload.get("results", [])),
        "total_bytes_zeroed": sum(r.get("swap_bytes_zeroed", 0) for r in payload.get("results", [])),
    }


def build_summary(out_dir: Path, arms: dict[str, dict[str, Any]], vram_snapshots: dict[str, Any]):
    summary = {"out_dir": str(out_dir), "arms": arms, "vram": vram_snapshots, "verdict": []}

    a = arms.get("armA_masks_only", {})
    b = arms.get("armB_zero_copy", {})

    if "masks_only" in a.get("observed_swap_modes", []):
        summary["verdict"].append("Arm A confirmed running in masks_only mode.")
    if "zero_copy" in b.get("observed_swap_modes", []):
        summary["verdict"].append("Arm B confirmed running in zero_copy mode.")

    if a.get("total_bytes_copied", 1) == 0 and a.get("total_bytes_zeroed", 1) == 0:
        summary["verdict"].append("Arm A: zero bytes copied/zeroed -- masks-only path is working.")

    acc_a = a.get("accuracy", 0) or 0
    acc_b = b.get("accuracy", 0) or 0
    if acc_a > acc_b:
        summary["verdict"].append(f"Arm A (masks_only) accuracy {acc_a:.2f} > Arm B (zero_copy) {acc_b:.2f} -- hypothesis confirmed.")
    elif acc_a == acc_b:
        summary["verdict"].append(f"Arm A and B same accuracy {acc_a:.2f} -- no degradation from masks_only.")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# strict30 masks_only vs zero_copy comparison",
        "",
        f"out_dir: `{out_dir}`",
        "",
        "## VRAM Snapshots",
    ]
    for label, snap in sorted(vram_snapshots.items()):
        lines.append(f"### {label}")
        for gpu in snap.get("gpus", []):
            lines.append(f"  - GPU {gpu['index']}: {gpu['memory_used_mib']} MiB used / {gpu['memory_total_mib']} MiB total")
        lines.append("")

    lines.extend([
        "## Results",
        "",
        "| metric | Arm A (masks_only) | Arm B (zero_copy) |",
        "|---|---|---|",
    ])
    for key in ["accuracy", "coherence_rate", "avg_sample_time_s", "avg_swap_time_s", "avg_warm_swap_time_s",
                "avg_router_miss_inactive_ratio", "dynamic_signature_count", "rows_with_nonzero_swap_copies",
                "total_bytes_copied", "total_bytes_zeroed", "observed_swap_modes"]:
        va = a.get(key, "n/a")
        vb = b.get(key, "n/a")
        lines.append(f"| {key} | {va} | {vb} |")

    lines.extend(["", "## Verdict", *[f"- {v}" for v in summary["verdict"]]])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def sync_server_code(config, args):
    """Push the updated vllm_multiplex_server.py to the remote repo."""
    local_module = REPO_ROOT / "reap_swap" / "vllm_multiplex_server.py"
    # The remote repo has the same structure: reap_swap/vllm_multiplex_server.py
    # But it also has a scripts/ entrypoint that does bare imports.
    # We push to reap_swap/ (the package the scripts/ entrypoint imports from).
    remote_dest = f"{args.remote_repo}/reap_swap/vllm_multiplex_server.py"
    run(config.scp_prefix() + [str(local_module), f"{args.remote_host}:{remote_dest}"])
    print(f"  synced vllm_multiplex_server.py -> {remote_dest}")


def run_arm(config, args, out_dir, plan_local, first_active_set, arm_slug, masks_only: bool):
    remote_pid = f"/tmp/autoresearch-masks-test-{arm_slug}.pid"
    remote_plan = f"/tmp/autoresearch-masks-test-{arm_slug}.plan.json"
    remote_log = f"/tmp/autoresearch-masks-test-{arm_slug}.log"

    stop_remote(config, remote_pid=remote_pid, remote_port=args.remote_port)
    run(config.scp_prefix() + [str(plan_local), f"{args.remote_host}:{remote_plan}"])
    start_remote(
        config, remote_repo=args.remote_repo, remote_venv_python=args.remote_venv_python,
        remote_plan=remote_plan, remote_pid=remote_pid, remote_log=remote_log,
        gpus=args.gpus, remote_port=args.remote_port, model_name=args.model_name,
        masks_only=masks_only,
    )

    tunnel_proc = subprocess.Popen(
        tunnel_cmd(config, local_port=args.local_port, remote_port=args.remote_port),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
    )

    try:
        wait_ready(args.local_port, out_dir=out_dir, label=arm_slug)
        vram_before = capture_nvidia_smi(config, out_dir / f"{arm_slug}_vram_before_warm.json", f"{arm_slug}_before_warm")

        cmd, env = arm_eval_command(
            local_port=args.local_port, model_name=args.model_name, plan_local=plan_local,
            out_dir=out_dir, arm_slug=arm_slug,
            extra_args=["--warm-start-active-set-json", str(first_active_set)],
        )
        run(cmd, env=env, stdout=out_dir / f"{arm_slug}.stdout", stderr=out_dir / f"{arm_slug}.stderr")

        vram_after = capture_nvidia_smi(config, out_dir / f"{arm_slug}_vram_after_eval.json", f"{arm_slug}_after_eval")
        capture_remote_log_tail(config, remote_log, out_dir / f"{arm_slug}_server_log_tail.txt", lines=80)

        metrics = extract_arm_metrics(out_dir, arm_slug)
        return metrics, {"before_warm": vram_before, "after_eval": vram_after}
    finally:
        tunnel_proc.terminate()
        try:
            tunnel_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            tunnel_proc.kill()
        stop_remote(config, remote_pid=remote_pid, remote_port=args.remote_port)


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-arm test: masks_only vs zero_copy swap mode")
    parser.add_argument("--remote-host", default="ser@192.168.1.70")
    parser.add_argument("--remote-auth-mode", default="auto")
    parser.add_argument("--ssh-key-path", default=os.environ.get("REMOTE_SSH_KEY") or str(Path.home() / ".ssh/linux-ai"))
    parser.add_argument("--remote-repo", default="/home/ser/reap-expert-swap-reap")
    parser.add_argument("--remote-venv-python", default="/home/ser/reap-expert-swap-vllm016/.venv/bin/python")
    parser.add_argument("--local-port", type=int, default=18362)
    parser.add_argument("--remote-port", type=int, default=8362)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--model-name", default="qwen35-masks-only-test")
    parser.add_argument("--plan-json", default=str(REPO_ROOT / "assets" / "strict30-v2-plan.json"))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = str(REPO_ROOT / "test-output" / f"strict30-masks-only-{time.strftime('%Y%m%d-%H%M%S')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_auth_config(host=args.remote_host, auth_mode=args.remote_auth_mode, ssh_key_path=args.ssh_key_path)
    plan_local = Path(args.plan_json).resolve()

    identity = {
        "service": "masks-only-test",
        "host": "127.0.0.1",
        "port": args.local_port,
        "plan_file": str(plan_local),
        "plan_mode": "dynamic_core_specialist",
    }
    (out_dir / "runtime-readiness-identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    first_active_set = out_dir / "first-active-set.json"
    make_first_active_set(plan_local, first_active_set)

    vram_all = {}
    arms_all = {}

    # Arm A: masks_only=True
    print("=== Arm A: REAP_SWAP_MASKS_ONLY=1 (masks only, no zero/copy) ===")
    print("  syncing server code to remote...")
    sync_server_code(config, args)
    vram_baseline = capture_nvidia_smi(config, out_dir / "vram_baseline.json", "baseline_before_any_launch")
    vram_all["baseline"] = vram_baseline
    metrics_a, vram_a = run_arm(config, args, out_dir, plan_local, first_active_set, "armA_masks_only", masks_only=True)
    arms_all["armA_masks_only"] = metrics_a
    vram_all["armA_before_warm"] = vram_a["before_warm"]
    vram_all["armA_after_eval"] = vram_a["after_eval"]

    # Arm B: masks_only=False (legacy zero/copy)
    print("=== Arm B: REAP_SWAP_MASKS_ONLY=0 (legacy zero/copy) ===")
    metrics_b, vram_b = run_arm(config, args, out_dir, plan_local, first_active_set, "armB_zero_copy", masks_only=False)
    arms_all["armB_zero_copy"] = metrics_b
    vram_all["armB_before_warm"] = vram_b["before_warm"]
    vram_all["armB_after_eval"] = vram_b["after_eval"]

    summary = build_summary(out_dir, arms_all, vram_all)
    print(f"\n{'='*60}")
    print(f"Results: {out_dir}")
    print(f"{'='*60}")
    for v in summary["verdict"]:
        print(f"  {v}")
    print()

    a = arms_all["armA_masks_only"]
    b = arms_all["armB_zero_copy"]
    print(f"  masks_only accuracy: {a.get('accuracy')}  zero_copy accuracy: {b.get('accuracy')}")
    print(f"  masks_only swap_time: {a.get('avg_swap_time_s')}s  zero_copy swap_time: {b.get('avg_swap_time_s')}s")
    print(f"  masks_only bytes_copied: {a.get('total_bytes_copied')}  zero_copy bytes_copied: {b.get('total_bytes_copied')}")
    print(f"  masks_only bytes_zeroed: {a.get('total_bytes_zeroed')}  zero_copy bytes_zeroed: {b.get('total_bytes_zeroed')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
