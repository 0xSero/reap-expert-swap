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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True, stdout: Path | None = None, stderr: Path | None = None) -> subprocess.CompletedProcess[str]:
    out_handle = stdout.open("w", encoding="utf-8") if stdout else subprocess.PIPE
    err_handle = stderr.open("w", encoding="utf-8") if stderr else subprocess.PIPE
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=out_handle,
            stderr=err_handle,
            check=False,
        )
    finally:
        if stdout:
            out_handle.close()
        if stderr:
            err_handle.close()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return proc


def ssh_cmd(config, remote_command: str) -> list[str]:
    return [*config.ssh_prefix(), remote_command]


def tunnel_cmd(config, *, local_port: int, remote_port: int) -> list[str]:
    cmd = ["ssh", "-N", "-o", "StrictHostKeyChecking=no", "-o", "ExitOnForwardFailure=yes"]
    if config.ssh_key_path:
        cmd.extend(["-i", config.ssh_key_path])
    cmd.extend(["-L", f"{local_port}:127.0.0.1:{remote_port}", config.host])
    return cmd


def stop_remote_runtime(config, *, remote_pid: str, remote_port: int) -> None:
    command = f"set -e; if [ -f {shlex.quote(remote_pid)} ]; then pid=$(cat {shlex.quote(remote_pid)} || true); if [ -n \"$pid\" ]; then kill \"$pid\" 2>/dev/null || true; sleep 2; kill -9 \"$pid\" 2>/dev/null || true; fi; fi; pids=$(pgrep -f 'vllm_multiplex_server.py.*--port[[:space:]]{remote_port}' || true); if [ -n \"$pids\" ]; then kill $pids 2>/dev/null || true; sleep 2; kill -9 $pids 2>/dev/null || true; fi"
    run(ssh_cmd(config, command), check=False)


def start_remote_runtime(config, *, remote_repo: str, remote_venv_python: str, remote_plan: str, remote_pid: str, remote_log: str, gpus: str, remote_port: int, model_name: str) -> None:
    command = (
        f"cd {shlex.quote(remote_repo)} && "
        f"nohup env REAP_PLAN_FILE={shlex.quote(remote_plan)} CUDA_VISIBLE_DEVICES={shlex.quote(gpus)} {shlex.quote(remote_venv_python)} scripts/vllm_multiplex_server.py "
        f"--model /home/ser/models/Qwen_Qwen3.5-35B-A3B --host 127.0.0.1 --port {remote_port} --tensor-parallel-size 2 --max-model-len 3072 --max-num-seqs 1 --reasoning-parser qwen3 --cpu-offload-gb 28 --swap-space 32 --gpu-memory-utilization 0.90 --dtype half --enforce-eager --disable-custom-all-reduce --language-model-only --served-model-name {shlex.quote(model_name)} --cpu-offload-params experts "
        f"</dev/null > {shlex.quote(remote_log)} 2>&1 & echo $! > {shlex.quote(remote_pid)}"
    )
    run(ssh_cmd(config, command))


def wait_ready(local_port: int, *, out_dir: Path, label: str) -> None:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 31):
        proc = subprocess.run(["curl", "-fsS", f"http://127.0.0.1:{local_port}/v1/models"], text=True, capture_output=True)
        attempts.append({"attempt": attempt, "rc": proc.returncode})
        if proc.returncode == 0:
            (out_dir / f"readiness-{label}.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
            (out_dir / f"readiness-{label}-response.json").write_text(proc.stdout, encoding="utf-8")
            return
        time.sleep(10)
    (out_dir / f"readiness-{label}.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError(f"runtime on local port {local_port} never became ready for {label}")


def make_first_active_set(plan_local: Path, out_path: Path) -> None:
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
    payload = build_active_set_payload(
        plan,
        prompt,
        request_id=stable_id("dynamic", row["id"], "7"),
        benchmark=row["benchmark"],
        phase="prefill",
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def arm_command(*, local_port: int, model_name: str, plan_local: Path, baseline_json: Path | None, out_dir: Path, arm_slug: str, extra_args: list[str]) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update(
        {
            "REAP_RUNTIME_READINESS_HOST_ALLOWLIST": "127.0.0.1",
            "REAP_RUNTIME_READINESS_PORT": str(local_port),
            "REAP_RUNTIME_READINESS_IDENTITY_PATH": str(out_dir / "runtime-readiness-identity.json"),
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "reap_swap.evaluate_original_vs_multiplex",
        "--mode",
        "dynamic",
        "--server-url",
        f"http://127.0.0.1:{local_port}",
        "--model",
        model_name,
        "--plan-json",
        str(plan_local),
        "--sample-count",
        "5",
        "--calibration-count",
        "0",
        "--seed",
        "7",
        "--request-timeout-s",
        "180",
        "--output-json",
        str(out_dir / f"{arm_slug}.json"),
        "--output-md",
        str(out_dir / f"{arm_slug}.md"),
    ]
    if baseline_json:
        cmd.extend([
            "--baseline-json",
            str(baseline_json),
            "--gate-profile",
            "dynamic_target",
            "--gate-output-json",
            str(out_dir / f"{arm_slug}.gate.json"),
            "--gate-output-md",
            str(out_dir / f"{arm_slug}.gate.md"),
        ])
    cmd.extend(extra_args)
    return cmd, env


def build_summary(out_dir: Path) -> None:
    arms = {
        "armA_dynamic_prewarmed": "A dynamic prewarmed",
        "armB_forced_static": "B forced static",
        "armC_cold_included": "C cold included (fresh relaunch)",
    }
    summary: dict[str, Any] = {"out_dir": str(out_dir), "arms": {}, "verdict": [], "notes": []}
    for slug, label in arms.items():
        payload = json.loads((out_dir / f"{slug}.json").read_text(encoding="utf-8"))
        gate_path = out_dir / f"{slug}.gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
        overall = (payload.get("summary") or {}).get("overall") or {}
        warm = payload.get("warm_start_result") or {}
        summary["arms"][slug] = {
            "label": label,
            "accuracy": overall.get("accuracy"),
            "coherence_rate": overall.get("coherence_rate"),
            "avg_sample_time_s": overall.get("avg_sample_time_s"),
            "p95_sample_time_s": overall.get("p95_sample_time_s"),
            "avg_swap_time_s": overall.get("avg_swap_time_s"),
            "avg_cold_swap_time_s": overall.get("avg_cold_swap_time_s"),
            "avg_warm_swap_time_s": overall.get("avg_warm_swap_time_s"),
            "avg_router_miss_inactive_ratio": overall.get("avg_router_miss_inactive_ratio"),
            "dynamic_signature_count": overall.get("dynamic_signature_count"),
            "rows_with_same_signature": overall.get("rows_with_same_signature"),
            "rows_with_zero_copy_swap": overall.get("rows_with_zero_copy_swap"),
            "rows_with_nonzero_swap_copies": overall.get("rows_with_nonzero_swap_copies"),
            "rows_with_nonzero_swap_adds": overall.get("rows_with_nonzero_swap_adds"),
            "cold_swap_count": overall.get("cold_swap_count"),
            "warm_swap_count": overall.get("warm_swap_count"),
            "gate_verdict": gate.get("verdict"),
            "gate_reasons": gate.get("reasons"),
            "warm_start_status": warm.get("status"),
            "warm_start_swap_time_s": warm.get("swap_time_s"),
        }
    A = summary["arms"]["armA_dynamic_prewarmed"]
    B = summary["arms"]["armB_forced_static"]
    C = summary["arms"]["armC_cold_included"]
    if A["accuracy"] == B["accuracy"] and A["dynamic_signature_count"] == 1 and B["dynamic_signature_count"] == 1:
        summary["verdict"].append("A and B match on accuracy and signature behavior: strict30 is still effectively static during measured prompts.")
    if A["avg_swap_time_s"] < C["avg_swap_time_s"]:
        summary["verdict"].append("Prewarming isolates the first-shrink tax: warm steady-state swaps are cheap, cold startup is the latency distortion.")
    if A["rows_with_nonzero_swap_copies"] == 0 and B["rows_with_nonzero_swap_copies"] == 0 and C["rows_with_nonzero_swap_copies"] == 0:
        summary["verdict"].append("No arm triggered measured expert-copy movement after startup; the current plan is not exercising real dynamic movement.")
    summary["notes"].append("Gate outputs remain invalid if the supplied BF16 baseline artifact does not match the current sample-count/signature set.")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# strict30 pair01 three-arm summary",
        "",
        f"out_dir: `{out_dir}`",
        "",
        "| arm | acc | coh | avg_sample_s | avg_swap_s | cold_swap_s | warm_swap_s | sigs | same_sig_rows | zero_copy_rows | nonzero_copy_rows | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for data in summary["arms"].values():
        lines.append(
            f"| {data['label']} | {data['accuracy']} | {data['coherence_rate']} | {data['avg_sample_time_s']} | {data['avg_swap_time_s']} | {data['avg_cold_swap_time_s']} | {data['avg_warm_swap_time_s']} | {data['dynamic_signature_count']} | {data['rows_with_same_signature']} | {data['rows_with_zero_copy_swap']} | {data['rows_with_nonzero_swap_copies']} | {data['gate_verdict']} |"
        )
    lines += ["", "## Verdict", *[f"- {line}" for line in summary["verdict"]], "", "## Notes", *[f"- {line}" for line in summary["notes"]]]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current strict30 pair01 three-arm isolation flow from autoresearch.")
    parser.add_argument("--remote-host", default="ser@192.168.1.70")
    parser.add_argument("--remote-auth-mode", default="auto")
    parser.add_argument("--ssh-key-path", default=os.environ.get("REMOTE_SSH_KEY") or str(Path.home() / ".ssh/linux-ai"))
    parser.add_argument("--remote-repo", default="/home/ser/reap-expert-swap-reap")
    parser.add_argument("--remote-venv-python", default="/home/ser/reap-expert-swap-vllm016/.venv/bin/python")
    parser.add_argument("--local-port", type=int, default=18361)
    parser.add_argument("--remote-port", type=int, default=8361)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--model-name", default="qwen35-dynamic-30pct-pair01")
    parser.add_argument("--plan-json", default=str(REPO_ROOT / "assets/strict30-best-plan.json"))
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "test-output" / f"strict30-pair01-e2e-{time.strftime('%Y%m%d-%H%M%S')}"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_auth_config(host=args.remote_host, auth_mode=args.remote_auth_mode, ssh_key_path=args.ssh_key_path)
    plan_local = Path(args.plan_json).resolve()
    baseline_json = Path(args.baseline_json).resolve() if args.baseline_json else None
    identity = {
        "service": "pair01-isolation",
        "host": "127.0.0.1",
        "port": args.local_port,
        "plan_file": str(plan_local),
        "plan_mode": "dynamic_core_specialist",
    }
    (out_dir / "runtime-readiness-identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    first_active_set = out_dir / "first-active-set.json"
    make_first_active_set(plan_local, first_active_set)

    tunnel_proc: subprocess.Popen[str] | None = None
    remote_pid = "/tmp/autoresearch-strict30-pair01.pid"
    remote_plan = "/tmp/autoresearch-strict30-pair01.plan.json"
    remote_log = "/tmp/autoresearch-strict30-pair01.log"
    try:
        stop_remote_runtime(config, remote_pid=remote_pid, remote_port=args.remote_port)
        run(config.scp_prefix() + [str(plan_local), f"{args.remote_host}:{remote_plan}"])
        # NOTE: do not SCP local vllm_multiplex_server.py -- the remote uses bare
        # imports (from dynamic_reap import ...) while local uses relative imports
        # (from .dynamic_reap import ...) which breaks standalone execution.
        start_remote_runtime(config, remote_repo=args.remote_repo, remote_venv_python=args.remote_venv_python, remote_plan=remote_plan, remote_pid=remote_pid, remote_log=remote_log, gpus=args.gpus, remote_port=args.remote_port, model_name=args.model_name)
        tunnel_proc = subprocess.Popen(tunnel_cmd(config, local_port=args.local_port, remote_port=args.remote_port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        wait_ready(args.local_port, out_dir=out_dir, label="warm")

        cmd, env = arm_command(local_port=args.local_port, model_name=args.model_name, plan_local=plan_local, baseline_json=baseline_json, out_dir=out_dir, arm_slug="armA_dynamic_prewarmed", extra_args=["--warm-start-active-set-json", str(first_active_set)])
        run(cmd, env=env, stdout=out_dir / "armA_dynamic_prewarmed.stdout", stderr=out_dir / "armA_dynamic_prewarmed.stderr")

        cmd, env = arm_command(local_port=args.local_port, model_name=args.model_name, plan_local=plan_local, baseline_json=baseline_json, out_dir=out_dir, arm_slug="armB_forced_static", extra_args=["--force-static-active-set-json", str(first_active_set)])
        run(cmd, env=env, stdout=out_dir / "armB_forced_static.stdout", stderr=out_dir / "armB_forced_static.stderr")

        if tunnel_proc:
            tunnel_proc.terminate()
            tunnel_proc.wait(timeout=10)
            tunnel_proc = None
        stop_remote_runtime(config, remote_pid=remote_pid, remote_port=args.remote_port)
        run(config.scp_prefix() + [str(plan_local), f"{args.remote_host}:{remote_plan}"])
        start_remote_runtime(config, remote_repo=args.remote_repo, remote_venv_python=args.remote_venv_python, remote_plan=remote_plan, remote_pid=remote_pid, remote_log=remote_log, gpus=args.gpus, remote_port=args.remote_port, model_name=args.model_name)
        tunnel_proc = subprocess.Popen(tunnel_cmd(config, local_port=args.local_port, remote_port=args.remote_port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        wait_ready(args.local_port, out_dir=out_dir, label="cold")

        cmd, env = arm_command(local_port=args.local_port, model_name=args.model_name, plan_local=plan_local, baseline_json=baseline_json, out_dir=out_dir, arm_slug="armC_cold_included", extra_args=[])
        run(cmd, env=env, stdout=out_dir / "armC_cold_included.stdout", stderr=out_dir / "armC_cold_included.stderr")

        build_summary(out_dir)
        print(out_dir)
        return 0
    finally:
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.send_signal(signal.SIGTERM)
            try:
                tunnel_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()
        stop_remote_runtime(config, remote_pid=remote_pid, remote_port=args.remote_port)


if __name__ == "__main__":
    raise SystemExit(main())
