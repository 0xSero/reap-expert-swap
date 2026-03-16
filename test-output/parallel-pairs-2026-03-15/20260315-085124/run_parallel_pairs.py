#!/usr/bin/env python3
import json, os, pathlib, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REMOTE = "ser@192.168.1.70"
REMOTE_REPO = "/home/ser/reap-expert-swap-reap"
REMOTE_VENV_PY = "/home/ser/reap-expert-swap-vllm016/.venv/bin/python"
PLAN_LOCAL = "/Users/sero/ai/reap-expert-swap/test-output/support-set-research-20260311-30pct/best-plan.json"
BASELINE_JSON = "/Users/sero/ai/reap-expert-swap/test-output/multi-turn-eval-20260312/baseline/baseline-singleturn-seed7-s1.json"
RUN_ROOT = pathlib.Path(os.environ["RUN_ROOT"])
PAIRS = [
    {"name": "pair01", "gpus": "0,1", "remote_port": 8361, "local_port": 18361},
    {"name": "pair23", "gpus": "2,3", "remote_port": 8363, "local_port": 18363},
    {"name": "pair45", "gpus": "4,5", "remote_port": 8365, "local_port": 18365},
    {"name": "pair67", "gpus": "6,7", "remote_port": 8367, "local_port": 18367},
]
MODEL_BASE = "qwen35-dynamic-30pct"


def run(cmd, *, input_text=None, env=None, cwd=None, timeout=None, check=True):
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def write(path: pathlib.Path, content: str):
    path.write_text(content, encoding="utf-8")


def ssh_script(script: str, *, check=True):
    return run(["ssh", REMOTE, "bash", "-s"], input_text=script, check=check)


def pair_task(spec):
    name = spec["name"]
    gpus = spec["gpus"]
    remote_port = spec["remote_port"]
    local_port = spec["local_port"]
    model_name = f"{MODEL_BASE}-{name}"
    pair_dir = RUN_ROOT / name
    pair_dir.mkdir(parents=True, exist_ok=True)
    tag = f"parallel-{name}-{RUN_ROOT.name}"
    remote_plan = f"/tmp/{tag}.plan.json"
    remote_log = f"/tmp/{tag}.log"
    remote_pid = f"/tmp/{tag}.pid"
    identity_path = pair_dir / "runtime-readiness-identity.json"
    identity = {
        "service": name,
        "host": "127.0.0.1",
        "port": local_port,
        "plan_file": PLAN_LOCAL,
        "plan_mode": "dynamic_core_specialist",
        "plan_budget_bytes": 16757880012,
    }
    write(identity_path, json.dumps(identity, indent=2) + "\n")
    context = {
        "name": name,
        "gpus": gpus,
        "remote_port": remote_port,
        "local_port": local_port,
        "model_name": model_name,
        "plan_local": PLAN_LOCAL,
        "baseline_json": BASELINE_JSON,
        "identity_path": str(identity_path),
    }
    write(pair_dir / "context.json", json.dumps(context, indent=2) + "\n")

    # reset targeted resources
    subprocess.run(["pkill", "-f", f"ssh -f -N -L {local_port}:127.0.0.1:{remote_port} {REMOTE}"], capture_output=True, text=True)
    reset_script = f"""
set -euo pipefail
if [ -f {shlex.quote(remote_pid)} ]; then
  pid=$(cat {shlex.quote(remote_pid)} || true)
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true; fi
fi
pids=$(pgrep -f 'vllm_multiplex_server.py.*--port[[:space:]]{remote_port}' || true)
if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; sleep 2; kill -9 $pids 2>/dev/null || true; fi
rm -f {shlex.quote(remote_pid)} {shlex.quote(remote_log)} {shlex.quote(remote_plan)}
echo reset_ok
"""
    proc = ssh_script(reset_script, check=False)
    write(pair_dir / "reset.stdout", proc.stdout)
    write(pair_dir / "reset.stderr", proc.stderr)

    plan_sha = run(["shasum", "-a", "256", PLAN_LOCAL])
    write(pair_dir / "plan_sha_local.txt", plan_sha.stdout)
    scp = run(["scp", "-q", PLAN_LOCAL, f"{REMOTE}:{remote_plan}"])
    write(pair_dir / "plan_copy.stderr", scp.stderr)
    remote_sha = run(["ssh", REMOTE, "sha256sum", remote_plan])
    write(pair_dir / "plan_sha_remote.txt", remote_sha.stdout)

    launch_script = f"""
set -euo pipefail
cd {shlex.quote(REMOTE_REPO)}
nohup env REAP_PLAN_FILE={shlex.quote(remote_plan)} CUDA_VISIBLE_DEVICES={shlex.quote(gpus)} {shlex.quote(REMOTE_VENV_PY)} scripts/vllm_multiplex_server.py \
  --model /home/ser/models/Qwen_Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port {remote_port} \
  --tensor-parallel-size 2 \
  --max-model-len 3072 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --cpu-offload-gb 28 \
  --swap-space 32 \
  --gpu-memory-utilization 0.90 \
  --dtype half \
  --enforce-eager \
  --disable-custom-all-reduce \
  --language-model-only \
  --served-model-name {shlex.quote(model_name)} \
  --cpu-offload-params experts \
  > {shlex.quote(remote_log)} 2>&1 &
echo $! > {shlex.quote(remote_pid)}
echo launch_pid:$(cat {shlex.quote(remote_pid)})
"""
    proc = ssh_script(launch_script)
    write(pair_dir / "launch.stdout", proc.stdout)
    write(pair_dir / "launch.stderr", proc.stderr)

    tunnel = run(["ssh", "-f", "-N", "-L", f"{local_port}:127.0.0.1:{remote_port}", REMOTE], check=False)
    write(pair_dir / "tunnel.stderr", tunnel.stderr)

    ready = False
    for attempt in range(1, 26):
        curl = subprocess.run(["curl", "-fsS", f"http://127.0.0.1:{local_port}/v1/models"], capture_output=True, text=True)
        if curl.returncode == 0:
            ready = True
            write(pair_dir / "readiness.txt", f"ready_attempt={attempt}\n")
            write(pair_dir / "readiness-response.json", curl.stdout)
            break
        with (pair_dir / "readiness-probe.txt").open("a", encoding="utf-8") as fh:
            fh.write(f"attempt={attempt} rc={curl.returncode}\n")
        time.sleep(15)
    if not ready:
        write(pair_dir / "readiness.txt", "ready=false\n")

    summary = {
        "name": name,
        "gpus": gpus,
        "ready": ready,
        "model_name": model_name,
        "local_port": local_port,
        "remote_port": remote_port,
        "plan": PLAN_LOCAL,
        "baseline": BASELINE_JSON,
    }

    env = os.environ.copy()
    env.update({
        "REAP_RUNTIME_READINESS_HOST_ALLOWLIST": "127.0.0.1",
        "REAP_RUNTIME_READINESS_PORT": str(local_port),
        "REAP_RUNTIME_READINESS_IDENTITY_PATH": str(identity_path),
    })
    if ready:
        eval_cmd = [
            os.path.expanduser("~/.local/bin/uv"), "run", "--with", "requests", "--with", "datasets", "python",
            "/Users/sero/ai/reap-expert-swap/scripts/evaluate_original_vs_multiplex.py",
            "--mode", "dynamic",
            "--server-url", f"http://127.0.0.1:{local_port}",
            "--model", model_name,
            "--plan-json", PLAN_LOCAL,
            "--sample-count", "1",
            "--calibration-count", "0",
            "--seed", "7",
            "--request-timeout-s", "120",
            "--baseline-json", BASELINE_JSON,
            "--gate-profile", "dynamic_target",
            "--gate-output-json", str(pair_dir / "gate.json"),
            "--gate-output-md", str(pair_dir / "gate.md"),
            "--output-json", str(pair_dir / "dynamic.json"),
            "--output-md", str(pair_dir / "dynamic.md"),
        ]
        proc = run(eval_cmd, env=env, check=False)
        write(pair_dir / "eval.stdout", proc.stdout)
        write(pair_dir / "eval.stderr", proc.stderr)
        write(pair_dir / "eval.rc", str(proc.returncode) + "\n")

    remote_post = ssh_script(f"""
set -euo pipefail
echo pid_file:
cat {shlex.quote(remote_pid)} 2>/dev/null || true
echo proc_after_attempt:
pgrep -af 'vllm_multiplex_server.py.*--port[[:space:]]{remote_port}' || true
echo models_probe:
curl -sS http://127.0.0.1:{remote_port}/v1/models || true
echo
echo log_tail:
tail -n 200 {shlex.quote(remote_log)} 2>/dev/null || true
""", check=False)
    write(pair_dir / "remote-post.txt", remote_post.stdout + remote_post.stderr)

    # cleanup runtime and tunnel
    ssh_script(f"""
set -euo pipefail
if [ -f {shlex.quote(remote_pid)} ]; then
  pid=$(cat {shlex.quote(remote_pid)} || true)
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true; fi
fi
pids=$(pgrep -f 'vllm_multiplex_server.py.*--port[[:space:]]{remote_port}' || true)
if [ -n "$pids" ]; then kill -9 $pids 2>/dev/null || true; fi
""", check=False)
    subprocess.run(["pkill", "-f", f"ssh -f -N -L {local_port}:127.0.0.1:{remote_port} {REMOTE}"], capture_output=True, text=True)

    if (pair_dir / "dynamic.json").exists():
        dyn = json.loads((pair_dir / "dynamic.json").read_text())
        gate = json.loads((pair_dir / "gate.json").read_text()) if (pair_dir / "gate.json").exists() else {}
        summary.update({
            "dynamic_results": len(dyn.get("results") or []),
            "accuracy": ((dyn.get("summary") or {}).get("overall") or {}).get("accuracy"),
            "avg_sample_time_s": ((dyn.get("summary") or {}).get("overall") or {}).get("avg_sample_time_s"),
            "avg_swap_time_s": ((dyn.get("summary") or {}).get("overall") or {}).get("avg_swap_time_s"),
            "p95_sample_time_s": ((dyn.get("summary") or {}).get("overall") or {}).get("p95_sample_time_s"),
            "total_live_footprint_ratio": ((gate.get("metrics") or {}).get("total_live_footprint_ratio")),
            "gate_verdict": gate.get("verdict"),
            "gate_reasons": gate.get("reasons"),
            "plan_sha256": ((dyn.get("plan_identity") or {}).get("plan_sha256")),
        })
    write(pair_dir / "pair-summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    summaries = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(pair_task, spec): spec for spec in PAIRS}
        for fut in as_completed(futs):
            spec = futs[fut]
            try:
                summaries.append(fut.result())
            except Exception as e:
                summaries.append({"name": spec["name"], "gpus": spec["gpus"], "error": str(e)})
    summaries = sorted(summaries, key=lambda x: x["name"])
    (RUN_ROOT / "summary.json").write_text(json.dumps({"run_root": str(RUN_ROOT), "summaries": summaries}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Parallel pair dynamic eval summary", "", f"run_root: `{RUN_ROOT}`", "", "| pair | gpus | ready | results | accuracy | avg_sample_s | avg_swap_s | p95_sample_s | footprint_ratio | gate | notes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for s in summaries:
        notes = "; ".join(s.get("gate_reasons") or []) if s.get("gate_reasons") else (s.get("error") or "")
        notes = notes.replace("|", "/")
        lines.append(f"| {s.get('name')} | {s.get('gpus')} | {s.get('ready')} | {s.get('dynamic_results')} | {s.get('accuracy')} | {s.get('avg_sample_time_s')} | {s.get('avg_swap_time_s')} | {s.get('p95_sample_time_s')} | {s.get('total_live_footprint_ratio')} | {s.get('gate_verdict')} | {notes} |")
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_root": str(RUN_ROOT), "summary_json": str(RUN_ROOT / 'summary.json'), "summary_md": str(RUN_ROOT / 'summary.md')}, indent=2))

if __name__ == "__main__":
    main()
