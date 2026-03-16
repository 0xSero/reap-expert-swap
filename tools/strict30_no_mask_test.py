#!/usr/bin/env python3
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


def start_remote(
    config,
    *,
    remote_repo,
    remote_venv_python,
    remote_plan,
    remote_pid,
    remote_log,
    gpus,
    remote_port,
    model_name,
    masks_only: bool,
    enable_router_masks: bool,
):
    env_parts = []
    if masks_only:
        env_parts.append("REAP_SWAP_MASKS_ONLY=1")
    if not enable_router_masks:
        env_parts.append("REAP_ENABLE_ROUTER_MASKS=0")
    env_parts.append(f"REAP_PLAN_FILE={shlex.quote(remote_plan)}")
    env_parts.append(f"CUDA_VISIBLE_DEVICES={shlex.quote(gpus)}")
    env_prefix = " ".join(env_parts)
    command = (
        f"cd {shlex.quote(remote_repo)} && "
        f"nohup env {env_prefix} "
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


def capture_remote_log_tail(config, remote_log, out_path: Path, lines=80):
    proc = subprocess.run(
        ssh_cmd(config, f"tail -n {lines} {shlex.quote(remote_log)} 2>/dev/null || true"),
        text=True, capture_output=True, check=False,
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    return proc.stdout


def wait_ready(local_port, *, label):
    for _attempt in range(1, 31):
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
    a = arms.get("armA_no_mask", {})
    b = arms.get("armB_masks_only", {})
    if a.get("avg_router_miss_inactive_ratio", 1) == 0:
        summary["verdict"].append("Arm A no_mask reported zero router inactive ratio.")
    if a.get("total_bytes_copied", 1) == 0 and a.get("total_bytes_zeroed", 1) == 0:
        summary["verdict"].append("Arm A no_mask performed zero expert copy/zero operations.")
    if b.get("total_bytes_copied", 1) == 0 and b.get("total_bytes_zeroed", 1) == 0:
        summary["verdict"].append("Arm B masks_only performed zero expert copy/zero operations.")
    acc_a = a.get("accuracy", 0) or 0
    acc_b = b.get("accuracy", 0) or 0
    if acc_a > acc_b:
        summary["verdict"].append(f"Arm A no_mask accuracy {acc_a:.2f} > Arm B masks_only {acc_b:.2f}.")
    elif acc_a == acc_b:
        summary["verdict"].append(f"Arm A and Arm B matched at accuracy {acc_a:.2f}.")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# strict30 no_mask vs masks_only comparison",
        "",
        f"out_dir: `{out_dir}`",
        "",
        "## Results",
        "",
        "| metric | Arm A (no_mask) | Arm B (masks_only) |",
        "|---|---|---|",
    ]
    for key in [
        "accuracy",
        "coherence_rate",
        "avg_sample_time_s",
        "avg_swap_time_s",
        "avg_warm_swap_time_s",
        "avg_router_miss_inactive_ratio",
        "dynamic_signature_count",
        "rows_with_nonzero_swap_copies",
        "total_bytes_copied",
        "total_bytes_zeroed",
        "observed_swap_modes",
    ]:
        lines.append(f"| {key} | {a.get(key, 'n/a')} | {b.get(key, 'n/a')} |")
    lines.extend(["", "## Verdict", *[f"- {v}" for v in summary["verdict"]]])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_arm(config, args, out_dir, plan_local, first_active_set, arm_slug, *, masks_only: bool, enable_router_masks: bool):
    remote_pid = f"/tmp/autoresearch-no-mask-{arm_slug}.pid"
    remote_plan = f"/tmp/autoresearch-no-mask-{arm_slug}.plan.json"
    remote_log = f"/tmp/autoresearch-no-mask-{arm_slug}.log"
    stop_remote(config, remote_pid=remote_pid, remote_port=args.remote_port)
    run(config.scp_prefix() + [str(plan_local), f"{args.remote_host}:{remote_plan}"])
    start_remote(
        config,
        remote_repo=args.remote_repo,
        remote_venv_python=args.remote_venv_python,
        remote_plan=remote_plan,
        remote_pid=remote_pid,
        remote_log=remote_log,
        gpus=args.gpus,
        remote_port=args.remote_port,
        model_name=args.model_name,
        masks_only=masks_only,
        enable_router_masks=enable_router_masks,
    )
    tunnel_proc = subprocess.Popen(
        tunnel_cmd(config, local_port=args.local_port, remote_port=args.remote_port),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        wait_ready(args.local_port, label=arm_slug)
        vram_before = capture_nvidia_smi(config, out_dir / f"{arm_slug}_vram_before_warm.json", f"{arm_slug}_before_warm")
        cmd, env = arm_eval_command(
            local_port=args.local_port,
            model_name=args.model_name,
            plan_local=plan_local,
            out_dir=out_dir,
            arm_slug=arm_slug,
            extra_args=["--warm-start-active-set-json", str(first_active_set)],
        )
        run(cmd, env=env, stdout=out_dir / f"{arm_slug}.stdout", stderr=out_dir / f"{arm_slug}.stderr")
        vram_after = capture_nvidia_smi(config, out_dir / f"{arm_slug}_vram_after_eval.json", f"{arm_slug}_after_eval")
        capture_remote_log_tail(config, remote_log, out_dir / f"{arm_slug}_server_log_tail.txt")
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
    parser = argparse.ArgumentParser(description="Two-arm test: no_mask vs masks_only")
    parser.add_argument("--remote-host", default="ser@192.168.1.70")
    parser.add_argument("--remote-auth-mode", default="auto")
    parser.add_argument("--ssh-key-path", default=os.environ.get("REMOTE_SSH_KEY") or str(Path.home() / ".ssh/linux-ai"))
    parser.add_argument("--remote-repo", default="/home/ser/reap-expert-swap-reap")
    parser.add_argument("--remote-venv-python", default="/home/ser/reap-expert-swap-vllm016/.venv/bin/python")
    parser.add_argument("--local-port", type=int, default=18363)
    parser.add_argument("--remote-port", type=int, default=8363)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--model-name", default="qwen35-no-mask-test")
    parser.add_argument("--plan-json", default=str(REPO_ROOT / "assets" / "strict30-v2-plan.json"))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = str(REPO_ROOT / "test-output" / f"strict30-no-mask-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_auth_config(host=args.remote_host, auth_mode=args.remote_auth_mode, ssh_key_path=args.ssh_key_path)
    plan_local = Path(args.plan_json).resolve()
    identity = {
        "service": "no-mask-test",
        "host": "127.0.0.1",
        "port": args.local_port,
        "plan_file": str(plan_local),
        "plan_mode": "dynamic_core_specialist",
    }
    (out_dir / "runtime-readiness-identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    first_active_set = out_dir / "first-active-set.json"
    make_first_active_set(plan_local, first_active_set)
    vram_all = {"baseline": capture_nvidia_smi(config, out_dir / "vram_baseline.json", "baseline_before_any_launch")}
    arms_all = {}
    metrics_a, vram_a = run_arm(config, args, out_dir, plan_local, first_active_set, "armA_no_mask", masks_only=True, enable_router_masks=False)
    arms_all["armA_no_mask"] = metrics_a
    vram_all["armA_before_warm"] = vram_a["before_warm"]
    vram_all["armA_after_eval"] = vram_a["after_eval"]
    metrics_b, vram_b = run_arm(config, args, out_dir, plan_local, first_active_set, "armB_masks_only", masks_only=True, enable_router_masks=True)
    arms_all["armB_masks_only"] = metrics_b
    vram_all["armB_before_warm"] = vram_b["before_warm"]
    vram_all["armB_after_eval"] = vram_b["after_eval"]
    build_summary(out_dir, arms_all, vram_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
