#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dynamic_reap import infer_domain_tags, normalize_prompt_text, stable_prompt_id
from run_budget_oracle_analysis import normalized_weights


NONE_LABEL = "__none__"
DATASET_VERSION = "support-router-v0"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def build_activation_lookup(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source_path = Path(path)
    if not source_path.exists():
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(source_path):
        prompt_text = str(record.get("prompt_text") or "")
        if not prompt_text:
            continue
        key = normalize_prompt_text(prompt_text)
        lookup.setdefault(key, record)
    return lookup


def resolve_plan_for_dynamic_artifact(
    dynamic_json_path: str | Path,
    dynamic_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    dynamic_path = Path(dynamic_json_path)
    embedded = dynamic_payload.get("plan")
    if isinstance(embedded, dict):
        plan_identity = dynamic_payload.get("plan_identity") or {}
        plan_path = plan_identity.get("plan_path")
        return embedded, str(plan_path) if plan_path else None

    def _resolve_candidate(path_value: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
        if not path_value:
            return None, None
        raw_candidate = Path(path_value)
        candidates: list[Path] = []
        if raw_candidate.is_absolute():
            candidates.append(raw_candidate)
        else:
            candidates.extend(
                [
                    raw_candidate,
                    (Path.cwd() / raw_candidate),
                ]
            )
            parent_chain = [dynamic_path.parent, *list(dynamic_path.parents)]
            for parent in parent_chain:
                candidates.append(parent / raw_candidate)
        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.exists():
                return load_json(resolved), str(resolved)
        return None, None

    run_summary_path = dynamic_path.parent / "run-summary.json"
    if run_summary_path.exists():
        run_summary = load_json(run_summary_path)
        plan_json = run_summary.get("plan_json")
        resolved_plan, resolved_plan_path = _resolve_candidate(plan_json)
        if resolved_plan:
            return resolved_plan, resolved_plan_path

    smoke_name = dynamic_path.parent.name
    if smoke_name.endswith("-smoke"):
        resolved_plan, resolved_plan_path = _resolve_candidate(
            dynamic_path.parent.parent / smoke_name.replace("-smoke", "-plan") / "plan.json"
        )
        if resolved_plan:
            return resolved_plan, resolved_plan_path
    return None, None


def prompt_feature_text(prompt_text: str, benchmark: str, tags: list[str]) -> str:
    return f"benchmark:{benchmark} tags:{' '.join(tags)} {normalize_prompt_text(prompt_text)}".strip()


def prompt_shape_features(prompt_text: str) -> dict[str, float]:
    normalized = normalize_prompt_text(prompt_text)
    words = [token for token in normalized.split(" ") if token]
    return {
        "prompt_len_chars": float(len(normalized)),
        "prompt_len_words": float(len(words)),
        "newline_count": float(prompt_text.count("\n")),
        "digit_ratio": (sum(1 for ch in normalized if ch.isdigit()) / len(normalized)) if normalized else 0.0,
    }


def _slice_budget(layer: dict[str, Any]) -> int:
    slice_catalog = list(layer.get("sliceCatalog") or [])
    if not slice_catalog:
        return 1
    avg_slice_size = sum(len(row.get("experts", [])) for row in slice_catalog) / len(slice_catalog)
    specialist_target = int(layer.get("specialistActiveExpertTarget", 0) or 0)
    if specialist_target <= 0:
        return 1
    return max(1, int(math.ceil(specialist_target / max(avg_slice_size, 1.0))))


def derive_slice_targets(
    layer: dict[str, Any],
    teacher_weights: dict[int, float],
    *,
    coverage_target: float = 0.80,
) -> dict[str, Any]:
    slice_catalog = list(layer.get("sliceCatalog") or [])
    if not slice_catalog or not teacher_weights:
        return {
            "top_slice_id": NONE_LABEL,
            "positive_slice_ids": [],
            "slice_target_mass": {},
            "covered_target_mass": 0.0,
            "target_mass_total": 0.0,
        }

    scored: list[tuple[float, str]] = []
    slice_target_mass: dict[str, float] = {}
    for row in slice_catalog:
        slice_id = str(row["sliceId"])
        mass = sum(float(teacher_weights.get(int(expert), 0.0)) for expert in row.get("experts", []))
        slice_target_mass[slice_id] = round(mass, 8)
        scored.append((mass, slice_id))
    scored.sort(reverse=True)
    if not scored or scored[0][0] <= 0.0:
        return {
            "top_slice_id": NONE_LABEL,
            "positive_slice_ids": [],
            "slice_target_mass": slice_target_mass,
            "covered_target_mass": 0.0,
            "target_mass_total": 0.0,
        }

    max_slices = _slice_budget(layer)
    target_mass_total = sum(slice_target_mass.values())
    coverage_threshold = coverage_target * target_mass_total
    running = 0.0
    positive: list[str] = []
    covered_experts: set[int] = set()
    slice_rows = {str(row["sliceId"]): row for row in slice_catalog}
    for mass, slice_id in scored:
        if mass <= 0.0:
            continue
        slice_experts = {int(expert) for expert in slice_rows[slice_id].get("experts", [])}
        incremental = sum(float(teacher_weights.get(expert, 0.0)) for expert in slice_experts - covered_experts)
        if incremental <= 0.0:
            continue
        positive.append(slice_id)
        running += incremental
        covered_experts.update(slice_experts)
        if running >= coverage_threshold or len(positive) >= max_slices:
            break
    return {
        "top_slice_id": scored[0][1],
        "positive_slice_ids": positive,
        "slice_target_mass": slice_target_mass,
        "covered_target_mass": round(running, 8),
        "target_mass_total": round(target_mass_total, 8),
    }


def split_bucket(prompt_id: str, *, train_frac: float = 0.8, valid_frac: float = 0.1) -> str:
    bucket = int(hashlib.sha1(prompt_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + valid_frac:
        return "valid"
    return "test"


def dataset_rows_from_dynamic_payload(
    dynamic_json_path: str | Path,
    dynamic_payload: dict[str, Any],
    *,
    observer_summary: dict[str, Any],
    activation_lookup: dict[str, dict[str, Any]] | None = None,
    coverage_target: float = 0.80,
    label_plan: dict[str, Any] | None = None,
    label_plan_path: str | None = None,
) -> list[dict[str, Any]]:
    source_plan, source_plan_path = resolve_plan_for_dynamic_artifact(dynamic_json_path, dynamic_payload)
    plan = label_plan or source_plan
    plan_path = label_plan_path or source_plan_path
    if not plan:
        return []
    if plan.get("mode") != "dynamic_core_specialist":
        return []
    if source_plan is None and label_plan is None:
        return []

    activation_lookup = activation_lookup or {}
    attempt_id = str(
        dynamic_payload.get("attempt_id")
        or dynamic_payload.get("artifacts", {}).get("attempt_id")
        or Path(dynamic_json_path).parent.name
    )
    budget = plan.get("budget") or {}
    source_budget = (source_plan or {}).get("budget") or {}
    source_plan_context = {
        "plan_path": source_plan_path,
        "plan_mode": str((source_plan or {}).get("mode") or ""),
        "selection_strategy": str((source_plan or {}).get("selectionStrategy") or "activation_mass"),
        "core_selection_mode": str((source_plan or {}).get("coreSelectionMode") or "selection_mass"),
        "resident_ratio": float(source_budget.get("max_resident_ratio", 0.0) or 0.0),
        "resident_gib": float(source_budget.get("max_resident_gib", 0.0) or 0.0),
    }
    plan_context = {
        "plan_path": plan_path,
        "plan_mode": str(plan.get("mode") or ""),
        "selection_strategy": str(plan.get("selectionStrategy") or "activation_mass"),
        "core_selection_mode": str(plan.get("coreSelectionMode") or "selection_mass"),
        "resident_ratio": float(budget.get("max_resident_ratio", 0.0) or 0.0),
        "resident_gib": float(budget.get("max_resident_gib", 0.0) or 0.0),
    }

    rows: list[dict[str, Any]] = []
    for result in dynamic_payload.get("results") or []:
        prompt_text = str(result.get("question") or "")
        if not prompt_text:
            continue
        normalized_prompt = normalize_prompt_text(prompt_text)
        prompt_id = stable_prompt_id(prompt_text)
        benchmark = str(result.get("benchmark") or "unknown")
        tags = infer_domain_tags(prompt_text)
        activation_record = activation_lookup.get(normalized_prompt)
        activation_meta = {
            "matched": bool(activation_record),
            "source": str((activation_record or {}).get("source") or "benchmark_eval"),
            "conversation_id": str((activation_record or {}).get("conversation_id") or "unknown"),
        }
        router_misses = result.get("router_misses") or {}
        router_miss_summary = result.get("router_miss_summary") or {}

        global_self_look = {
            "inactive_mass_total": float(router_miss_summary.get("inactive_mass_total", 0.0) or 0.0),
            "observed_mass_total": float(router_miss_summary.get("observed_mass_total", 0.0) or 0.0),
            "inactive_ratio": float(router_miss_summary.get("inactive_ratio", 0.0) or 0.0),
            "inactive_expert_total": int(router_miss_summary.get("inactive_expert_total", 0) or 0),
            "refresh_suggested": bool(result.get("refresh_suggested")),
        }

        for layer_key, layer in (plan.get("perLayer") or {}).items():
            layer_miss = (router_misses.get("by_layer") or {}).get(layer_key) or {}
            inactive_experts = [int(expert) for expert in layer_miss.get("inactive_experts", [])]
            teacher_weights = normalized_weights(observer_summary, layer_key, inactive_experts) if inactive_experts else {}
            target = derive_slice_targets(layer, teacher_weights, coverage_target=coverage_target)
            slice_ids = [str(row.get("sliceId")) for row in layer.get("sliceCatalog") or []]
            row = {
                "dataset_version": DATASET_VERSION,
                "example_id": f"{attempt_id}::{str(result.get('request_id') or prompt_id)}::{layer_key}",
                "split": split_bucket(prompt_id),
                "run_attempt_id": attempt_id,
                "request_id": str(result.get("request_id") or ""),
                "prompt_id": prompt_id,
                "benchmark": benchmark,
                "prompt_text": normalized_prompt,
                "prompt_feature_text": prompt_feature_text(prompt_text, benchmark, tags),
                "domain_tags": tags,
                "activation_meta": activation_meta,
                "prompt_shape": prompt_shape_features(prompt_text),
                "plan_context": {
                    **plan_context,
                    "layer_key": layer_key,
                    "specialist_active_target": int(layer.get("specialistActiveExpertTarget", 0) or 0),
                    "specialist_candidate_target": int(layer.get("specialistCandidateExpertTarget", 0) or 0),
                    "slice_count": len(slice_ids),
                },
                "source_plan_context": source_plan_context,
                "self_look_summary": global_self_look,
                "self_look_layer": {
                    "inactive_mass": float(layer_miss.get("inactive_mass", 0.0) or 0.0),
                    "observed_mass": float(layer_miss.get("observed_mass", 0.0) or 0.0),
                    "inactive_ratio": (
                        float(layer_miss.get("inactive_mass", 0.0) or 0.0)
                        / float(layer_miss.get("observed_mass", 0.0) or 1.0)
                        if float(layer_miss.get("observed_mass", 0.0) or 0.0) > 0
                        else 0.0
                    ),
                    "inactive_expert_total": len(inactive_experts),
                    "inactive_experts": inactive_experts,
                },
                "label": {
                    "candidate_slice_ids": slice_ids,
                    "top_slice_id": target["top_slice_id"],
                    "positive_slice_ids": target["positive_slice_ids"],
                    "slice_target_mass": target["slice_target_mass"],
                    "covered_target_mass": target["covered_target_mass"],
                    "target_mass_total": target["target_mass_total"],
                },
                "outcome": {
                    "correct": bool(result.get("correct")),
                    "coherent": bool(result.get("coherent")),
                    "parse_error": bool(result.get("parse_error")),
                    "error": bool(result.get("error")),
                    "swap_time_s": float(result.get("swap_time_s", 0.0) or 0.0),
                    "request_latency_s": float(result.get("request_latency_s", 0.0) or 0.0),
                    "total_latency_s": float(result.get("total_latency_s", 0.0) or 0.0),
                },
            }
            rows.append(row)
    return rows
