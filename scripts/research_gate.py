#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable


EXPECTED_BENCHMARK_SUITE = (
    "mmlu",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "gsm8k",
)


GATE_PROFILES: dict[str, dict[str, Any]] = {
    "budget_static": {
        "policy": "budget_gate",
        "stage": "static_viability",
        "thresholds": {
            "min_accuracy_retained_pct": 80.0,
            "min_coherence_retained_pct": 90.0,
            "min_benchmark_accuracy_retained_pct": 40.0,
            "min_benchmark_coherence_retained_pct": 70.0,
            "max_parse_error_rate": 0.05,
            "max_error_rate": 0.02,
            "max_p95_sample_time_s": 130.0,
            "max_avg_swap_time_s": None,
            "require_multiple_cartridges": False,
        },
        "pass_actions": {
            "continue_multiplex": True,
            "continue_lower_budgets": True,
            "rerun_confirmation_seed": False,
            "promote_incumbent": False,
        },
        "fail_actions": {
            "continue_multiplex": False,
            "continue_lower_budgets": False,
            "rerun_confirmation_seed": False,
            "promote_incumbent": False,
        },
    },
    "budget_multiplex": {
        "policy": "budget_gate",
        "stage": "multiplex_viability",
        "thresholds": {
            "min_accuracy_retained_pct": 85.0,
            "min_coherence_retained_pct": 92.0,
            "min_benchmark_accuracy_retained_pct": 45.0,
            "min_benchmark_coherence_retained_pct": 75.0,
            "max_parse_error_rate": 0.04,
            "max_error_rate": 0.02,
            "max_p95_sample_time_s": 130.0,
            "max_avg_swap_time_s": 2.5,
            "require_multiple_cartridges": True,
        },
        "pass_actions": {
            "continue_multiplex": True,
            "continue_lower_budgets": True,
            "rerun_confirmation_seed": True,
            "promote_incumbent": False,
        },
        "fail_actions": {
            "continue_multiplex": False,
            "continue_lower_budgets": False,
            "rerun_confirmation_seed": False,
            "promote_incumbent": False,
        },
    },
    "boundary_multiplex": {
        "policy": "boundary_gate",
        "stage": "multiplex_viability",
        "thresholds": {
            "min_accuracy_retained_pct": 80.0,
            "min_coherence_retained_pct": 90.0,
            "min_benchmark_accuracy_retained_pct": 35.0,
            "min_benchmark_coherence_retained_pct": 70.0,
            "max_parse_error_rate": 0.05,
            "max_error_rate": 0.02,
            "max_p95_sample_time_s": 130.0,
            "max_avg_swap_time_s": 2.5,
            "require_multiple_cartridges": True,
        },
        "pass_actions": {
            "continue_multiplex": True,
            "continue_larger_counts": True,
            "rerun_confirmation_seed": True,
            "promote_incumbent": False,
        },
        "fail_actions": {
            "continue_multiplex": False,
            "continue_larger_counts": False,
            "rerun_confirmation_seed": False,
            "promote_incumbent": False,
        },
    },
    "dynamic_target": {
        "policy": "dynamic_quality_gate",
        "stage": "dynamic_targeting",
        "thresholds": {
            "min_accuracy_retained_pct": 95.0,
            "min_coherence_retained_pct": 92.0,
            "min_benchmark_accuracy_retained_pct": 90.0,
            "min_benchmark_coherence_retained_pct": None,
            "max_parse_error_rate": 0.04,
            "max_error_rate": 0.02,
            "max_p95_sample_time_s": 130.0,
            "max_avg_swap_time_s": 2.5,
            "max_quality_loss_pct": 5.0,
            "max_benchmark_accuracy_drop_abs": 0.10,
            "max_total_live_footprint_ratio": 0.30,
            "require_multiple_cartridges": False,
        },
        "pass_actions": {
            "continue_dynamic_tuning": True,
            "promote_incumbent": True,
        },
        "fail_actions": {
            "continue_dynamic_tuning": True,
            "promote_incumbent": False,
        },
    },
}


def parse_zeroed_counts(log_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not log_path.exists():
        return counts
    for line in log_path.read_text(errors="replace").splitlines():
        if "Loaded cartridge_" not in line or "zeroed expert tensors" not in line:
            continue
        fragment = line.split("Loaded ", 1)[1]
        cartridge_id = fragment.split(" across ", 1)[0].strip()
        match = re.search(r"(\d+)\s+zeroed expert tensors", fragment)
        if not match:
            continue
        counts[cartridge_id] = int(match.group(1))
    return counts


def validate_cartridge_integrity(
    log_path: Path, cartridge_ids: Iterable[str]
) -> dict[str, Any]:
    expected = list(cartridge_ids)
    counts = parse_zeroed_counts(log_path)
    missing = [cartridge_id for cartridge_id in expected if cartridge_id not in counts]
    non_positive = {
        cartridge_id: counts[cartridge_id]
        for cartridge_id in expected
        if cartridge_id in counts and counts[cartridge_id] <= 0
    }
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing zeroed expert counts for {', '.join(missing)}")
    if non_positive:
        rendered = ", ".join(
            f"{cartridge_id}={value}" for cartridge_id, value in sorted(non_positive.items())
        )
        reasons.append(f"non-positive zeroed expert counts: {rendered}")
    return {
        "valid": not reasons,
        "counts": counts,
        "expected_cartridges": expected,
        "reasons": reasons,
        "log_path": str(log_path),
    }


def wait_for_cartridge_logs(
    log_path: Path, cartridge_ids: list[str], timeout_s: int = 120
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        integrity = validate_cartridge_integrity(log_path, cartridge_ids)
        if integrity["counts"] and not integrity["reasons"]:
            return integrity
        time.sleep(2)
    final_integrity = validate_cartridge_integrity(log_path, cartridge_ids)
    if final_integrity["valid"]:
        return final_integrity
    raise TimeoutError(
        f"Timed out waiting for cartridge preload in {log_path}; saw {json.dumps(final_integrity['counts'], sort_keys=True)}"
    )


def resolve_gate_profile(
    profile: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    if profile not in GATE_PROFILES:
        raise KeyError(f"Unknown gate profile: {profile}")
    config = json.loads(json.dumps(GATE_PROFILES[profile]))
    overrides = overrides or {}
    thresholds = config["thresholds"]
    for key, value in overrides.items():
        if value is not None and key in thresholds:
            thresholds[key] = value
    return config


def _snapshot_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload.get("summary", {}).get("overall", {})
    comparison = payload.get("comparison", {}).get("overall", {})
    plan_budget = payload.get("plan", {}).get("budget", {})
    return {
        "mode": payload.get("mode"),
        "accuracy": overall.get("accuracy"),
        "coherence_rate": overall.get("coherence_rate"),
        "error_rate": overall.get("error_rate"),
        "parse_error_rate": overall.get("parse_error_rate"),
        "avg_sample_time_s": overall.get("avg_sample_time_s"),
        "p95_sample_time_s": overall.get("p95_sample_time_s"),
        "avg_swap_time_s": overall.get("avg_swap_time_s"),
        "swap_count": overall.get("swap_count"),
        "unique_cartridges_used": overall.get("unique_cartridges_used"),
        "cartridge_transition_rate": overall.get("cartridge_transition_rate"),
        "accuracy_retained_pct": comparison.get("accuracy_retained_pct"),
        "coherence_retained_pct": comparison.get("coherence_retained_pct"),
        "quality_loss_pct": comparison.get("quality_loss_pct"),
        "worst_benchmark_accuracy_drop_abs": comparison.get(
            "worst_benchmark_accuracy_drop_abs"
        ),
        "total_live_footprint_ratio": plan_budget.get("max_resident_ratio"),
        "total_live_footprint_gib": plan_budget.get("max_resident_gib"),
    }


def evaluate_payload_gate(
    payload: dict[str, Any],
    profile: str,
    *,
    threshold_overrides: dict[str, Any] | None = None,
    integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = resolve_gate_profile(profile, threshold_overrides)
    thresholds = config["thresholds"]
    summary = payload.get("summary", {})
    overall = summary.get("overall", {})
    summary_by_benchmark = summary.get("by_benchmark", {})
    comparison = payload.get("comparison", {})
    comp_overall = comparison.get("overall", {})
    comp_by_benchmark = comparison.get("by_benchmark", {})
    reasons: list[str] = []
    checks: list[dict[str, Any]] = []

    structural_errors: list[str] = []
    if not overall:
        structural_errors.append("missing summary.overall")
    if not comp_overall:
        structural_errors.append("missing baseline comparison")
    if not isinstance(summary_by_benchmark, dict) or not summary_by_benchmark:
        structural_errors.append("missing summary.by_benchmark")
    if not isinstance(comp_by_benchmark, dict) or not comp_by_benchmark:
        structural_errors.append("missing comparison.by_benchmark")
    if overall.get("total", 0) <= 0:
        structural_errors.append("no evaluated samples")
    if payload.get("mode") == "dynamic":
        runtime_identity = payload.get("runtime_identity")
        if not isinstance(runtime_identity, dict):
            structural_errors.append("missing runtime identity")
        plan_identity = payload.get("plan_identity")
        if not isinstance(plan_identity, dict):
            structural_errors.append("missing plan identity")
        plan_budget = payload.get("plan", {}).get("budget")
        if not isinstance(plan_budget, dict):
            structural_errors.append("missing plan budget")
        elif plan_budget.get("max_resident_ratio") is None:
            structural_errors.append("missing total live footprint ratio")
        else:
            readiness_evidence = runtime_identity.get("readiness_evidence") if isinstance(runtime_identity, dict) else None
            if not isinstance(readiness_evidence, dict):
                structural_errors.append("missing runtime readiness evidence")
            else:
                readiness_identity = readiness_evidence.get("identity")
                if not isinstance(readiness_identity, dict):
                    structural_errors.append("runtime readiness evidence missing identity payload")
                else:
                    readiness_host = readiness_identity.get("host")
                    readiness_port = readiness_identity.get("port")
                    readiness_plan_file = readiness_identity.get("plan_file")
                    runtime_host = runtime_identity.get("host") if isinstance(runtime_identity, dict) else None
                    runtime_port = runtime_identity.get("port") if isinstance(runtime_identity, dict) else None
                    if readiness_host and runtime_host and readiness_host != runtime_host:
                        structural_errors.append(
                            "runtime readiness evidence host does not match runtime identity"
                        )
                    if readiness_port is not None and runtime_port is not None and int(readiness_port) != int(runtime_port):
                        structural_errors.append(
                            "runtime readiness evidence port does not match runtime identity"
                        )
                    plan_identity_path = plan_identity.get("plan_path") if isinstance(plan_identity, dict) else None
                    if readiness_plan_file and plan_identity_path and str(readiness_plan_file) != str(plan_identity_path):
                        structural_errors.append(
                            "runtime readiness evidence plan file does not match plan identity"
                        )
            runtime_plan_path = runtime_identity.get("plan_path") if isinstance(runtime_identity, dict) else None
            plan_identity_path = plan_identity.get("plan_path") if isinstance(plan_identity, dict) else None
            if runtime_plan_path and plan_identity_path and runtime_plan_path != plan_identity_path:
                structural_errors.append(
                    "runtime identity plan path does not match plan identity"
                )
            runtime_plan_mode = runtime_identity.get("plan_mode") if isinstance(runtime_identity, dict) else None
            plan_identity_mode = plan_identity.get("plan_mode") if isinstance(plan_identity, dict) else None
            if runtime_plan_mode and plan_identity_mode and runtime_plan_mode != plan_identity_mode:
                structural_errors.append(
                    "runtime identity plan mode does not match plan identity"
                )
            runtime_plan_budget = runtime_identity.get("plan_budget_bytes") if isinstance(runtime_identity, dict) else None
            plan_identity_budget = plan_identity.get("plan_budget_bytes") if isinstance(plan_identity, dict) else None
            if (
                runtime_plan_budget is not None
                and plan_identity_budget is not None
                and runtime_plan_budget != plan_identity_budget
            ):
                structural_errors.append(
                    "runtime identity plan budget does not match plan identity"
                )

        request_linkage_errors: list[str] = []
        for row in payload.get("results") or []:
            row_id = str(row.get("id") or row.get("request_id") or "unknown-row")
            request_id = row.get("request_id")
            swap_request_id = row.get("swap_request_id")
            active_set_signature = row.get("active_set_signature")
            swap_plan_identity = row.get("swap_plan_identity")
            router_misses = row.get("router_misses")
            if not request_id:
                request_linkage_errors.append(f"{row_id}: missing request_id")
            if swap_request_id != request_id:
                request_linkage_errors.append(
                    f"{row_id}: swap_request_id does not match request_id"
                )
            if not active_set_signature:
                request_linkage_errors.append(
                    f"{row_id}: missing active_set_signature"
                )
            if not isinstance(swap_plan_identity, dict):
                request_linkage_errors.append(
                    f"{row_id}: missing swap_plan_identity"
                )
            elif isinstance(plan_identity, dict):
                if plan_identity.get("plan_path") and swap_plan_identity.get("plan_path") and swap_plan_identity.get("plan_path") != plan_identity.get("plan_path"):
                    request_linkage_errors.append(
                        f"{row_id}: swap_plan_identity plan_path does not match plan identity"
                    )
                if plan_identity.get("plan_mode") and swap_plan_identity.get("plan_mode") and swap_plan_identity.get("plan_mode") != plan_identity.get("plan_mode"):
                    request_linkage_errors.append(
                        f"{row_id}: swap_plan_identity plan_mode does not match plan identity"
                    )
            if not isinstance(router_misses, dict):
                request_linkage_errors.append(f"{row_id}: missing router_misses")
            elif router_misses.get("request_id") != request_id:
                request_linkage_errors.append(
                    f"{row_id}: router_misses request_id does not match request_id"
                )
        if request_linkage_errors:
            structural_errors.append(
                "request-level evidence linkage failures: " + "; ".join(request_linkage_errors)
            )
    if integrity and not integrity.get("valid", True):
        structural_errors.extend(integrity.get("reasons", []))

    if not structural_errors:
        observed_benchmarks = set(summary_by_benchmark.keys())
        expected_benchmarks = set(payload.get("benchmarks") or EXPECTED_BENCHMARK_SUITE)
        missing_from_summary = sorted(expected_benchmarks - observed_benchmarks)
        if missing_from_summary:
            structural_errors.append(
                f"missing summary.by_benchmark coverage for: {', '.join(missing_from_summary)}"
            )
        missing_from_comparison = sorted(observed_benchmarks - set(comp_by_benchmark.keys()))
        if missing_from_comparison:
            structural_errors.append(
                f"missing comparison.by_benchmark coverage for: {', '.join(missing_from_comparison)}"
            )

    if structural_errors:
        return {
            "policy": config["policy"],
            "profile": profile,
            "stage": config["stage"],
            "verdict": "invalid",
            "accepted": False,
            "valid": False,
            "reasons": structural_errors,
            "checks": [],
            "thresholds": thresholds,
            "metrics": _snapshot_metrics(payload),
            "integrity": integrity,
            "actions": config["fail_actions"],
        }

    def check_min(name: str, value: float | None, minimum: float | None) -> None:
        if minimum is None or value is None:
            return
        passed = value >= minimum
        checks.append(
            {
                "name": name,
                "kind": "min",
                "value": value,
                "threshold": minimum,
                "passed": passed,
            }
        )
        if not passed:
            reasons.append(f"{name} {value:.4f} < {minimum:.4f}")

    def check_max(name: str, value: float | None, maximum: float | None) -> None:
        if maximum is None or value is None:
            return
        passed = value <= maximum
        checks.append(
            {
                "name": name,
                "kind": "max",
                "value": value,
                "threshold": maximum,
                "passed": passed,
            }
        )
        if not passed:
            reasons.append(f"{name} {value:.4f} > {maximum:.4f}")

    check_min(
        "accuracy_retained_pct",
        comp_overall.get("accuracy_retained_pct"),
        thresholds.get("min_accuracy_retained_pct"),
    )
    check_min(
        "coherence_retained_pct",
        comp_overall.get("coherence_retained_pct"),
        thresholds.get("min_coherence_retained_pct"),
    )
    check_max(
        "parse_error_rate",
        overall.get("parse_error_rate"),
        thresholds.get("max_parse_error_rate"),
    )
    check_max(
        "error_rate",
        overall.get("error_rate"),
        thresholds.get("max_error_rate"),
    )
    check_max(
        "p95_sample_time_s",
        overall.get("p95_sample_time_s"),
        thresholds.get("max_p95_sample_time_s"),
    )
    check_max(
        "avg_swap_time_s",
        overall.get("avg_swap_time_s"),
        thresholds.get("max_avg_swap_time_s"),
    )
    check_max(
        "quality_loss_pct",
        comp_overall.get("quality_loss_pct"),
        thresholds.get("max_quality_loss_pct"),
    )
    check_max(
        "worst_benchmark_accuracy_drop_abs",
        comp_overall.get("worst_benchmark_accuracy_drop_abs"),
        thresholds.get("max_benchmark_accuracy_drop_abs"),
    )
    check_max(
        "total_live_footprint_ratio",
        payload.get("plan", {}).get("budget", {}).get("max_resident_ratio"),
        thresholds.get("max_total_live_footprint_ratio"),
    )

    min_benchmark_acc = thresholds.get("min_benchmark_accuracy_retained_pct")
    min_benchmark_coh = thresholds.get("min_benchmark_coherence_retained_pct")
    max_benchmark_drop = thresholds.get("max_benchmark_accuracy_drop_abs")
    for benchmark, benchmark_row in sorted(comp_by_benchmark.items()):
        check_min(
            f"{benchmark}.accuracy_retained_pct",
            benchmark_row.get("accuracy_retained_pct"),
            min_benchmark_acc,
        )
        check_min(
            f"{benchmark}.coherence_retained_pct",
            benchmark_row.get("coherence_retained_pct"),
            min_benchmark_coh,
        )
        check_max(
            f"{benchmark}.accuracy_drop_abs",
            benchmark_row.get("accuracy_drop_abs"),
            max_benchmark_drop,
        )

    if thresholds.get("require_multiple_cartridges"):
        unique_cartridges = overall.get("unique_cartridges_used", 0)
        passed = unique_cartridges >= 2
        checks.append(
            {
                "name": "unique_cartridges_used",
                "kind": "min",
                "value": unique_cartridges,
                "threshold": 2,
                "passed": passed,
            }
        )
        if not passed:
            reasons.append(f"unique_cartridges_used {unique_cartridges} < 2")

    accepted = not reasons
    verdict = "provisional" if accepted else "reject"
    return {
        "policy": config["policy"],
        "profile": profile,
        "stage": config["stage"],
        "verdict": verdict,
        "accepted": accepted,
        "valid": True,
        "reasons": reasons,
        "checks": checks,
        "thresholds": thresholds,
        "metrics": _snapshot_metrics(payload),
        "integrity": integrity,
        "actions": config["pass_actions"] if accepted else config["fail_actions"],
    }


def build_gate_markdown(gate: dict[str, Any]) -> str:
    lines = [
        f"# Gate verdict: {gate['verdict']}",
        "",
        f"- policy: `{gate['policy']}`",
        f"- profile: `{gate['profile']}`",
        f"- stage: `{gate['stage']}`",
        f"- accepted: `{gate['accepted']}`",
        f"- valid: `{gate['valid']}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in gate.get("metrics", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Reasons", ""])
    if gate.get("reasons"):
        lines.extend([f"- {reason}" for reason in gate["reasons"]])
    else:
        lines.append("- none")
    lines.extend(["", "## Thresholds", ""])
    for key, value in gate.get("thresholds", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Actions", ""])
    for key, value in gate.get("actions", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Checks", ""])
    if gate.get("checks"):
        for check in gate["checks"]:
            lines.append(
                f"- {check['name']}: value={check['value']} threshold={check['threshold']} passed={check['passed']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
