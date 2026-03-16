from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

POISON_PATTERNS = [
    r"cudaErrorIllegalAddress",
    r"illegal memory access",
    r"EngineDeadError",
    r"Engine core/worker initialization failure",
    r"RuntimeError:\s*cancelled",
    r"Connection reset by peer",
    r"connection refused",
    r"Worker proc .* died unexpectedly",
]

EXPECTED_FORENSIC_FIELDS = {
    "request_id": "request.request_id",
    "swap_request_id": "swap.response.request_id",
    "swap_plan_identity": "swap.response.plan_identity",
    "plan_sha256": "forensic.plan_identity.plan_sha256",
    "active_set_signature": "forensic.active_set.signature",
    "union_validation": "forensic.active_set.union_validation",
    "core_presence_summary": "forensic.active_set.core_presence_summary",
    "worker_swap_result": "forensic.worker.swap_result",
    "router_miss_payload": "router_misses",
    "crash_classification": "forensic.crash.classification",
}


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _contains_poison(text: str) -> list[str]:
    hits: list[str] = []
    for p in POISON_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def _get_path(data: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = data
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collect_bundle(input_dir: Path, forensic_json: Path | None = None) -> dict[str, Any]:
    request = _read_json(input_dir / "request.json") or {}
    swap_response = (
        _read_json(input_dir / "swap_response.json")
        or _read_json(input_dir / "swap.json")
        or {}
    )
    completion = (
        _read_json(input_dir / "completion_response.json")
        or _read_json(input_dir / "response.json")
        or {}
    )
    router_misses = (
        _read_json(input_dir / "router_misses.json")
        or _read_json(input_dir / "router_miss.json")
    )
    forensic = _read_json(forensic_json) if forensic_json else (
        _read_json(input_dir / "forensic.json")
        or _read_json(input_dir / "forensic_bundle.json")
        or {}
    )

    text_blobs: dict[str, str] = {}
    for name in [
        "server.log",
        "launch_step.log",
        "verdict.md",
        "one_request_response.stderr",
        "one_request_response.stdout",
        "remote_status_and_log_after_request.txt",
        "dynamic-phase.stderr",
        "dynamic-phase.stdout",
    ]:
        txt = _read_text(input_dir / name)
        if txt:
            text_blobs[name] = txt

    poison_hits: list[dict[str, Any]] = []
    for name, txt in text_blobs.items():
        patterns = _contains_poison(txt)
        if patterns:
            poison_hits.append({"source": name, "patterns": patterns})

    bundle: dict[str, Any] = {
        "request": request,
        "swap": {"response": swap_response},
        "completion": completion,
        "router_misses": router_misses,
        "forensic": forensic,
        "text_sources": list(text_blobs.keys()),
        "poison": {
            "detected": bool(poison_hits),
            "hits": poison_hits,
        },
    }

    missing: list[str] = []
    resolved: dict[str, Any] = {}
    for key, path in EXPECTED_FORENSIC_FIELDS.items():
        value = _get_path(bundle, path)
        if value is None:
            missing.append(key)
        else:
            resolved[key] = value

    bundle["expected_fields"] = {
        "resolved": resolved,
        "missing": missing,
    }

    crash_classification = "none"
    if bundle["poison"]["detected"]:
        crash_classification = "poisoned_runtime"
    elif completion == {}:
        crash_classification = "missing_completion_payload"

    bundle["classification"] = {
        "crash": crash_classification,
        "ready_for_full_sweep": crash_classification == "none" and not missing,
    }

    return bundle


def render_markdown(bundle: dict[str, Any]) -> str:
    resolved = bundle["expected_fields"]["resolved"]
    missing = bundle["expected_fields"]["missing"]
    poison = bundle["poison"]
    cls = bundle["classification"]

    status = lambda ok: "✅" if ok else "❌"

    lines = [
        "# One-Request Forensic Replay Summary",
        "",
        "```text",
        "Forensic readiness",
        "────────────────────────────────────────",
        f"Poison detected             {status(not poison['detected'])}",
        f"Crash classification clean  {status(cls['crash'] == 'none')}",
        f"Expected fields complete    {status(len(missing) == 0)}",
        f"Ready for full sweep        {status(cls['ready_for_full_sweep'])}",
        "```",
        "",
        "## Crash Classification",
        f"- `{cls['crash']}`",
        "",
        "## Expected Field Coverage",
        f"- Resolved: **{len(resolved)}**",
        f"- Missing: **{len(missing)}**",
    ]

    if missing:
        lines += ["", "### Missing Fields"]
        lines += [f"- `{m}`" for m in missing]

    if poison["detected"]:
        lines += ["", "## Poison Hits"]
        for hit in poison["hits"]:
            pats = ", ".join(hit["patterns"])
            lines.append(f"- `{hit['source']}` -> {pats}")

    lines += [
        "",
        "## Resolved Forensic Fields",
    ]
    for k, v in resolved.items():
        lines.append(f"- `{k}`: `{v}`")

    lines += [
        "",
        "## Expected JSON Interface (server -> collector)",
        "```json",
        json.dumps(
            {
                "request_id": "...",
                "swap_request_id": "...",
                "plan_identity": {
                    "plan_mode": "dynamic_core_specialist",
                    "plan_budget_bytes": 0,
                    "plan_sha256": "...",
                    "plan_path": "...",
                },
                "active_set": {
                    "signature": "...",
                    "union_validation": {
                        "ok": True,
                        "violations": [],
                    },
                    "core_presence_summary": {
                        "layers_checked": 40,
                        "layers_missing_core": 0,
                    },
                },
                "worker": {
                    "swap_result": {
                        "active_expert_bytes": 0,
                        "active_expert_count": 0,
                        "delta_added": 0,
                        "delta_removed": 0,
                        "delta_reused": 0,
                    }
                },
                "router_miss_payload": {},
                "crash": {
                    "classification": "none|poisoned_runtime|engine_dead"
                },
            },
            indent=2,
        ),
        "```",
    ]

    return "\n".join(lines) + "\n"


def write_bundle_and_visual(bundle: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "forensic_bundle.json"
    md_path = output_dir / "forensic_visual.md"
    json_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    return json_path, md_path
