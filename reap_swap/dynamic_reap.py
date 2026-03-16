#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .size_estimator import (
    QWEN15_MOE_A27B_CHAT_CONFIG,
    estimate_qwen2_moe_bf16_bytes,
    normalize_moe_config,
)

DEFAULT_MAX_RESIDENT_RATIO = 0.20
DEFAULT_CORE_BUDGET_FRACTION = 0.35
DEFAULT_SPECIALIST_BUDGET_FRACTION = 0.65
DEFAULT_CANDIDATE_POOL_MULTIPLIER = 3.0
DEFAULT_MAX_REFRESHES_PER_REQUEST = 1
DEFAULT_REFRESH_MISS_MASS_THRESHOLD = 0.18
DEFAULT_SELECTION_STRATEGY = "activation_mass"
DEFAULT_ROTATION_POLICY = "none"
DEFAULT_LEXICAL_K = 5


DEFAULT_FLOOR_EXACT_FRACTION = 0.30
DEFAULT_FLOOR_LAYER_WEIGHT_MODE = "late_boost"


DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "code": (
        "python",
        "typescript",
        "javascript",
        "rust",
        "go",
        "debug",
        "stack trace",
        "function",
        "class",
        "api",
        "sql",
        "bash",
        "shell",
        "docker",
        "kubernetes",
        "regex",
        "json",
        "yaml",
        "git",
    ),
    "math": (
        "equation",
        "integral",
        "probability",
        "algebra",
        "geometry",
        "theorem",
        "proof",
        "solve",
        "gsm8k",
        "arithmetic",
    ),
    "writing": (
        "rewrite",
        "summarize",
        "email",
        "essay",
        "tone",
        "grammar",
        "proposal",
        "blog",
        "copy",
    ),
    "research": (
        "paper",
        "benchmark",
        "compare",
        "evaluate",
        "analyze",
        "study",
        "ablation",
        "experiment",
    ),
    "ops": (
        "deploy",
        "prod",
        "staging",
        "latency",
        "vram",
        "gpu",
        "tailscale",
        "server",
        "log",
        "monitor",
    ),
}


BENCHMARK_TO_TAGS = {
    "mmlu": ["research", "writing"],
    "arc_challenge": ["research"],
    "hellaswag": ["writing"],
    "winogrande": ["writing"],
    "gsm8k": ["math", "research"],
}

TAG_ROTATION_WEIGHTS = {
    "general": 1,
    "code": 2,
    "ops": 3,
    "math": 5,
    "research": 7,
    "writing": 11,
}

SELECTION_STRATEGIES = {
    "activation_mass",
    "support_v1",
}

CORE_SELECTION_MODES = {
    "selection_mass",
    "floor_seeded",
}

ROTATION_POLICIES = {
    "none",
    "late_prompt_hash",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def normalize_prompt_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def stable_prompt_id(text: str) -> str:
    return hashlib.sha1(normalize_prompt_text(text).encode("utf-8")).hexdigest()[:16]


def _stable_bucket(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def tokenize_prompt_text(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "be", "if", "this",
        "that", "from", "by", "as", "it", "at", "into", "their", "your", "you", "we", "they", "them",
    }
    tokens = re.findall(r"[a-z0-9_]+", normalize_prompt_text(text).lower())
    return {token for token in tokens if token not in stopwords and len(token) > 2}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def infer_domain_tags(text: str) -> list[str]:
    normalized = normalize_prompt_text(text).lower()
    tags: list[str] = []
    for tag, patterns in DOMAIN_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            tags.append(tag)
    return tags or ["general"]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type in {"input_text", "output_text", "text"}:
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value)
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    role = str(message.get("role") or "unknown")
    content_text = _message_content_to_text(message.get("content"))
    if not content_text:
        return ""
    return f"{role}: {content_text}"


def _conversation_context_to_text(conversation_context: Any) -> str:
    if isinstance(conversation_context, str):
        return conversation_context
    if isinstance(conversation_context, list):
        lines = [_message_to_text(message) for message in conversation_context]
        return "\n".join(line for line in lines if line)
    if isinstance(conversation_context, dict):
        lines: list[str] = []
        for key in ("summary", "display", "title", "prompt", "text"):
            value = conversation_context.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"{key}: {value}")
        messages = conversation_context.get("messages")
        if isinstance(messages, list):
            lines.extend(_message_to_text(message) for message in messages)
        return "\n".join(line for line in lines if line)
    return ""


def infer_model_config(model_name: str | None, model_config_path: str | None = None) -> dict[str, Any]:
    if model_config_path:
        return normalize_moe_config(load_json(model_config_path))
    if model_name and "Qwen1.5-MoE-A2.7B-Chat" in model_name:
        return dict(QWEN15_MOE_A27B_CHAT_CONFIG)
    if model_name and "Qwen/Qwen1.5-MoE-A2.7B-Chat" in model_name:
        return dict(QWEN15_MOE_A27B_CHAT_CONFIG)
    if model_name and ("Qwen3.5-35B-A3B" in model_name or "Qwen_Qwen3.5-35B-A3B" in model_name):
        raise ValueError(
            "Qwen3.5 config requires --model-config-json because remote observer summaries only store the model path."
        )
    raise ValueError("Unable to infer model config; provide --model-config-json")


def estimate_per_expert_bytes(config: dict[str, Any]) -> int:
    model_bytes = estimate_qwen2_moe_bf16_bytes(config)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    return int(model_bytes["total_expert_params"] * model_bytes["bf16_bytes_per_param"] / (num_layers * num_experts))


def compute_dynamic_budget(
    config: dict[str, Any],
    *,
    max_resident_ratio: float = DEFAULT_MAX_RESIDENT_RATIO,
    max_resident_gib: float | None = None,
    core_budget_fraction: float = DEFAULT_CORE_BUDGET_FRACTION,
    specialist_budget_fraction: float = DEFAULT_SPECIALIST_BUDGET_FRACTION,
    candidate_pool_multiplier: float = DEFAULT_CANDIDATE_POOL_MULTIPLIER,
    max_refreshes_per_request: int = DEFAULT_MAX_REFRESHES_PER_REQUEST,
) -> dict[str, Any]:
    if max_resident_ratio <= 0 or max_resident_ratio > 1:
        raise ValueError("max_resident_ratio must be > 0 and <= 1")
    if core_budget_fraction <= 0 or specialist_budget_fraction <= 0:
        raise ValueError("core and specialist budget fractions must be positive")
    fraction_total = core_budget_fraction + specialist_budget_fraction
    if not math.isclose(fraction_total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("core_budget_fraction + specialist_budget_fraction must equal 1.0")
    if candidate_pool_multiplier < 1.0:
        raise ValueError("candidate_pool_multiplier must be >= 1")
    if max_refreshes_per_request < 0:
        raise ValueError("max_refreshes_per_request must be >= 0")

    model_bytes = estimate_qwen2_moe_bf16_bytes(config)
    full_bytes = int(model_bytes["full_bf16_bytes_estimate"])
    always_resident_bytes = int(model_bytes["always_resident_params"] * model_bytes["bf16_bytes_per_param"])
    cap_from_ratio = int(full_bytes * max_resident_ratio)
    cap_from_gib = int(max_resident_gib * (1024**3)) if max_resident_gib is not None else None
    hard_cap_bytes = min(cap_from_ratio, cap_from_gib) if cap_from_gib is not None else cap_from_ratio
    swappable_expert_budget_bytes = max(0, hard_cap_bytes - always_resident_bytes)
    per_expert_bytes = estimate_per_expert_bytes(config)
    total_active_expert_capacity = swappable_expert_budget_bytes // per_expert_bytes
    num_layers = int(config["num_hidden_layers"])
    core_budget_bytes = int(swappable_expert_budget_bytes * core_budget_fraction)
    specialist_budget_bytes = swappable_expert_budget_bytes - core_budget_bytes
    core_experts_per_layer_target = max(1, core_budget_bytes // max(per_expert_bytes * num_layers, 1))
    specialist_experts_per_layer_target = max(
        1,
        specialist_budget_bytes // max(per_expert_bytes * num_layers, 1),
    )
    candidate_experts_per_layer_target = max(
        specialist_experts_per_layer_target,
        int(math.ceil(specialist_experts_per_layer_target * candidate_pool_multiplier)),
    )
    return {
        "max_resident_gib": round(hard_cap_bytes / (1024**3), 6),
        "max_resident_ratio": round(hard_cap_bytes / full_bytes, 6) if full_bytes else 0.0,
        "requested_max_resident_ratio": max_resident_ratio,
        "requested_max_resident_gib": max_resident_gib,
        "full_bf16_gib": round(full_bytes / (1024**3), 6),
        "always_resident_bytes": always_resident_bytes,
        "always_resident_gib": round(always_resident_bytes / (1024**3), 6),
        "swappable_expert_budget_bytes": swappable_expert_budget_bytes,
        "swappable_expert_budget_gib": round(swappable_expert_budget_bytes / (1024**3), 6),
        "core_budget_bytes": core_budget_bytes,
        "specialist_budget_bytes": specialist_budget_bytes,
        "core_budget_fraction": core_budget_fraction,
        "specialist_budget_fraction": specialist_budget_fraction,
        "candidate_pool_multiplier": candidate_pool_multiplier,
        "max_refreshes_per_request": max_refreshes_per_request,
        "per_expert_bytes": per_expert_bytes,
        "total_active_expert_capacity": int(total_active_expert_capacity),
        "core_experts_per_layer_target": int(core_experts_per_layer_target),
        "specialist_experts_per_layer_target": int(specialist_experts_per_layer_target),
        "candidate_experts_per_layer_target": int(candidate_experts_per_layer_target),
    }


def _normalize_layer_key(layer_key: str) -> str:
    return layer_key if layer_key.startswith("layer_") else f"layer_{layer_key}"


def _vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_max(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return max(materialized) if materialized else 0.0


def _normalized_value(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(value) / float(scale)


def _slice_similarity(experts: list[dict[str, Any]]) -> float:
    if len(experts) <= 1:
        return 1.0
    vectors = [list(expert["signalsBySummary"].values()) for expert in experts]
    scores = []
    for idx in range(len(vectors)):
        for other_idx in range(idx + 1, len(vectors)):
            scores.append(_cosine_similarity(vectors[idx], vectors[other_idx]))
    return round(_mean(scores), 6) if scores else 1.0


def _slice_size_target(specialist_experts_per_layer_target: int) -> int:
    return min(8, max(2, int(math.ceil(specialist_experts_per_layer_target / 2))))


def _task_priors_for_slice(slice_row: dict[str, Any], labels: list[str]) -> dict[str, float]:
    priors = {"global": round(float(slice_row["activationMass"]), 8)}
    for label in labels:
        priors[label] = round(float(slice_row["signalsBySummary"].get(label, 0.0)), 8)
    return priors


def _sorted_layer_scores(layer_blob: dict[str, Any]) -> list[tuple[int, float]]:
    if "scores" in layer_blob:
        scores = list(map(float, layer_blob["scores"]))
        return sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if "expert_scores" in layer_blob:
        items = [(int(k), float(v)) for k, v in dict(layer_blob["expert_scores"]).items()]
        return sorted(items, key=lambda item: item[1], reverse=True)
    raise ValueError("Layer blob must contain either 'scores' or 'expert_scores'.")


def _greedy_allocate_layer_scores(
    layer_scores: dict[str, dict[str, Any]],
    total_exact_budget: int,
    *,
    min_per_layer: int = 1,
    layer_weight_mode: str = DEFAULT_FLOOR_LAYER_WEIGHT_MODE,
) -> dict[str, list[int]]:
    layers = sorted(layer_scores.keys(), key=lambda value: int(str(value).replace("layer_", "")))
    if total_exact_budget < len(layers) * min_per_layer:
        raise ValueError("total_exact_budget is smaller than num_layers * min_per_layer")

    per_layer_sorted = {layer: _sorted_layer_scores(layer_scores[layer]) for layer in layers}
    selected: dict[str, list[int]] = {layer: [] for layer in layers}
    cursor = {layer: 0 for layer in layers}

    def layer_weight(layer: str) -> float:
        if layer_weight_mode == "late_boost":
            idx = layers.index(layer)
            return 1.0 + 0.5 * (idx / max(1, len(layers) - 1))
        if layer_weight_mode == "front_boost":
            idx = layers.index(layer)
            return 1.0 + 0.5 * (1.0 - idx / max(1, len(layers) - 1))
        return 1.0

    for layer in layers:
        for _ in range(min_per_layer):
            if cursor[layer] >= len(per_layer_sorted[layer]):
                raise ValueError(f"{layer}: not enough experts to satisfy min_per_layer")
            expert_id, _score = per_layer_sorted[layer][cursor[layer]]
            selected[layer].append(int(expert_id))
            cursor[layer] += 1

    used = len(layers) * min_per_layer
    while used < total_exact_budget:
        best_layer = None
        best_expert = None
        best_score = None
        for layer in layers:
            if cursor[layer] >= len(per_layer_sorted[layer]):
                continue
            expert_id, score = per_layer_sorted[layer][cursor[layer]]
            weighted = float(score) * layer_weight(layer)
            if best_score is None or weighted > best_score:
                best_score = weighted
                best_layer = layer
                best_expert = int(expert_id)
        if best_layer is None or best_expert is None:
            break
        selected[best_layer].append(best_expert)
        cursor[best_layer] += 1
        used += 1

    return {layer: sorted(experts) for layer, experts in selected.items()}


def _estimate_floor_budget(
    model_config: dict[str, Any],
    *,
    exact_fraction_of_full: float,
    prune_fraction: float = 0.0,
    lowbit_fraction_of_survivors: float = 0.0,
    lowbit_memory_ratio: float = 0.28,
    min_exact_experts_per_layer: int = 1,
) -> dict[str, Any]:
    if exact_fraction_of_full <= 0 or exact_fraction_of_full > 1:
        raise ValueError("exact_fraction_of_full must be > 0 and <= 1")
    if prune_fraction < 0 or prune_fraction >= 1:
        raise ValueError("prune_fraction must be >= 0 and < 1")
    if lowbit_fraction_of_survivors < 0 or lowbit_fraction_of_survivors > 1:
        raise ValueError("lowbit_fraction_of_survivors must be >= 0 and <= 1")
    if lowbit_memory_ratio < 0 or lowbit_memory_ratio > 1:
        raise ValueError("lowbit_memory_ratio must be >= 0 and <= 1")

    model_bytes = estimate_qwen2_moe_bf16_bytes(model_config)
    full_bytes = int(model_bytes["full_bf16_bytes_estimate"])
    always_resident_bytes = int(model_bytes["always_resident_params"] * model_bytes["bf16_bytes_per_param"])
    total_layers = int(model_config["num_hidden_layers"])
    experts_per_layer = int(model_config["num_experts"])
    survivors_fraction = 1.0 - prune_fraction
    routed_pool_bytes = int(model_bytes["total_expert_params"] * model_bytes["bf16_bytes_per_param"])
    exact_budget_bytes = int(round(routed_pool_bytes * survivors_fraction * exact_fraction_of_full))
    lowbit_budget_bytes = int(round(routed_pool_bytes * survivors_fraction * lowbit_fraction_of_survivors * lowbit_memory_ratio))
    per_expert_bytes = estimate_per_expert_bytes(model_config)
    total_exact_experts = max(total_layers * min_exact_experts_per_layer, exact_budget_bytes // max(per_expert_bytes, 1))
    return {
        "mode": "dynamic_exact_floor",
        "full_bf16_gib": round(full_bytes / (1024**3), 6),
        "always_resident_gib": round(always_resident_bytes / (1024**3), 6),
        "always_resident_bytes": always_resident_bytes,
        "per_expert_bytes": per_expert_bytes,
        "prune_fraction": round(prune_fraction, 6),
        "survivors_fraction": round(survivors_fraction, 6),
        "exact_fraction_of_full": round(exact_fraction_of_full, 6),
        "lowbit_fraction_of_survivors": round(lowbit_fraction_of_survivors, 6),
        "lowbit_memory_ratio": round(lowbit_memory_ratio, 6),
        "core_budget_bytes": int(total_exact_experts * per_expert_bytes),
        "specialist_budget_bytes": 0,
        "lowbit_budget_bytes": lowbit_budget_bytes,
        "swappable_expert_budget_bytes": int(total_exact_experts * per_expert_bytes),
        "total_exact_experts": int(total_exact_experts),
        "exact_experts_per_layer_target": int(max(min_exact_experts_per_layer, total_exact_experts // max(total_layers, 1))),
        "min_exact_experts_per_layer": int(min_exact_experts_per_layer),
        "candidate_pool_multiplier": 1.0,
        "max_refreshes_per_request": 0,
        "max_resident_gib": round((always_resident_bytes + total_exact_experts * per_expert_bytes + lowbit_budget_bytes) / (1024**3), 6),
        "max_resident_ratio": round((always_resident_bytes + total_exact_experts * per_expert_bytes + lowbit_budget_bytes) / full_bytes, 6) if full_bytes else 0.0,
    }


def build_dynamic_floor_plan(
    summaries: list[tuple[str, dict[str, Any]]],
    *,
    signal_key: str,
    model_config: dict[str, Any],
    exact_fraction_of_full: float = DEFAULT_FLOOR_EXACT_FRACTION,
    layer_weight_mode: str = DEFAULT_FLOOR_LAYER_WEIGHT_MODE,
    activation_records: list[dict[str, Any]] | None = None,
    prune_fraction: float = 0.0,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one summary is required")
    first_summary = summaries[0][1]
    layer_keys = list(first_summary["layers"].keys())
    labels = [label for label, _summary in summaries]
    activation_records = activation_records or []
    activation_summary = summarize_activation_records(activation_records)
    budget = _estimate_floor_budget(
        model_config,
        exact_fraction_of_full=exact_fraction_of_full,
        prune_fraction=prune_fraction,
    )

    layer_scores: dict[str, dict[str, Any]] = {}
    layer_score_debug: dict[str, dict[int, float]] = {}
    for layer_key in layer_keys:
        normalized_key = _normalize_layer_key(layer_key)
        num_experts = len(first_summary["layers"][layer_key][signal_key])
        combined_scores: list[float] = []
        for expert_idx in range(num_experts):
            values = [float(summary["layers"][layer_key][signal_key][expert_idx]) for _label, summary in summaries]
            peak = max(values) if values else 0.0
            stability = (min(values) / peak) if peak > 0 else 0.0
            combined = sum(values) * 0.85 + stability * 0.15
            combined_scores.append(round(combined, 8))
        layer_scores[normalized_key] = {"scores": combined_scores}
        layer_score_debug[normalized_key] = {idx: score for idx, score in enumerate(combined_scores)}

    selected_exact = _greedy_allocate_layer_scores(
        layer_scores,
        int(budget["total_exact_experts"]),
        min_per_layer=int(budget["min_exact_experts_per_layer"]),
        layer_weight_mode=layer_weight_mode,
    )

    per_layer: dict[str, Any] = {}
    for layer_key in layer_keys:
        normalized_key = _normalize_layer_key(layer_key)
        selected = selected_exact[normalized_key]
        layer_values = []
        for expert_idx in selected:
            signals = {label: float(summary["layers"][layer_key][signal_key][expert_idx]) for label, summary in summaries}
            layer_values.append(sum(signals.values()))
        per_layer[normalized_key] = {
            "rawLayerKey": layer_key,
            "numExperts": len(first_summary["layers"][layer_key][signal_key]),
            "coreExperts": selected,
            "coreActivationMass": round(sum(layer_values), 8),
            "sliceCatalog": [],
            "coreByteCost": len(selected) * budget["per_expert_bytes"],
            "specialistActiveExpertTarget": 0,
            "specialistCandidateExpertTarget": 0,
        }

    total_core = sum(len(layer["coreExperts"]) for layer in per_layer.values())
    full_bytes = int(round(float(budget["full_bf16_gib"]) * (1024**3)))
    actual_exact_bytes = total_core * int(budget["per_expert_bytes"])
    budget["total_exact_experts"] = int(total_core)
    budget["core_budget_bytes"] = int(actual_exact_bytes)
    budget["swappable_expert_budget_bytes"] = int(actual_exact_bytes)
    budget["exact_experts_per_layer_target"] = int(round(total_core / max(1, len(layer_keys))))
    budget["max_resident_gib"] = round((int(budget["always_resident_bytes"]) + actual_exact_bytes + int(budget.get("lowbit_budget_bytes", 0))) / (1024**3), 6)
    budget["max_resident_ratio"] = round((int(budget["always_resident_bytes"]) + actual_exact_bytes + int(budget.get("lowbit_budget_bytes", 0))) / full_bytes, 6) if full_bytes else 0.0
    summary = {
        "layerCount": len(layer_keys),
        "totalCoreExperts": total_core,
        "totalSlices": 0,
        "residentFractionPctMin": round(100.0 * min(len(layer["coreExperts"]) / max(1, layer["numExperts"]) for layer in per_layer.values()), 8),
        "residentFractionPctMax": round(100.0 * max(len(layer["coreExperts"]) / max(1, layer["numExperts"]) for layer in per_layer.values()), 8),
        "residentFractionPctAvg": round(100.0 * total_core / max(1, sum(layer["numExperts"] for layer in per_layer.values())), 8),
    }

    return {
        "mode": "dynamic_core_specialist",
        "selectionMode": "dynamic_exact_floor",
        "optionId": "A",
        "model": first_summary.get("model"),
        "signalKey": signal_key,
        "sourceSummaries": [
            {
                "label": label,
                "workflow": summary_row.get("workflow", label),
                "processedSamples": summary_row.get("processedSamples", 0),
                "totalTokens": summary_row.get("totalTokens", 0),
            }
            for label, summary_row in summaries
        ],
        "budget": budget,
        "perLayer": per_layer,
        "scorerArtifacts": {
            "taskFamilySlicePriors": {"global": {layer_key: [] for layer_key in per_layer.keys()}},
            "promptClusterPriors": {},
            "featureNormalization": activation_summary.get("featureNormalization", {}),
            "activationCorpus": {
                "recordCount": activation_summary.get("record_count", 0),
                "sourceCounts": activation_summary.get("source_counts", {}),
                "domainTagCounts": activation_summary.get("domain_tag_counts", {}),
            },
            "floorSelection": {
                "layerWeightMode": layer_weight_mode,
                "exactFractionOfFull": round(exact_fraction_of_full, 6),
                "selectedExactByLayer": selected_exact,
            },
        },
        "summary": summary,
    }


def summarize_activation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    tag_counter: Counter[str] = Counter()
    lengths: list[int] = []
    sources: Counter[str] = Counter()
    for record in records:
        sources[str(record.get("source") or "unknown")] += 1
        lengths.append(len(normalize_prompt_text(str(record.get("prompt_text") or "")).split()))
        for tag in record.get("domain_tags", []) or ["general"]:
            tag_counter[tag] += 1
    return {
        "record_count": len(records),
        "source_counts": dict(sorted(sources.items())),
        "domain_tag_counts": dict(sorted(tag_counter.items())),
        "featureNormalization": {
            "prompt_length_mean": round(_mean([float(length) for length in lengths]), 6) if lengths else 0.0,
            "prompt_length_p95": sorted(lengths)[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0,
        },
    }


def _load_layer_importance(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = load_json(path)
    layer_importance = payload.get("layer_importance")
    return layer_importance if isinstance(layer_importance, dict) else None


def _layer_importance_weight(row: dict[str, Any] | None) -> float:
    if not row:
        return 1.0
    return max(
        1e-6,
        (
            (0.50 * float(row.get("miss_sensitivity", 0.0) or 0.0))
            + (0.20 * float(row.get("benchmark_dependence", 0.0) or 0.0))
            + (0.20 * float(row.get("expert_entropy", 0.0) or 0.0))
            + (0.10 * (1.0 - float(row.get("prompt_overlap_jaccard", 0.0) or 0.0)))
        ),
    )


def _allocate_per_layer_counts(
    layer_keys: list[str],
    total_capacity: int,
    *,
    layer_importance: dict[str, Any] | None,
    min_per_layer: int,
) -> dict[str, int]:
    if not layer_keys:
        return {}
    if total_capacity < len(layer_keys) * min_per_layer:
        raise ValueError("total_capacity smaller than layer minimum allocation")
    weights = {
        _normalize_layer_key(layer_key): _layer_importance_weight((layer_importance or {}).get(_normalize_layer_key(layer_key)))
        for layer_key in layer_keys
    }
    allocation = {layer_key: int(min_per_layer) for layer_key in weights}
    remaining = total_capacity - (len(layer_keys) * min_per_layer)
    if remaining <= 0:
        return allocation
    total_weight = sum(weights.values())
    raw = {
        layer_key: (weights[layer_key] / total_weight) * remaining
        for layer_key in weights
    }
    for layer_key, value in raw.items():
        allocation[layer_key] += int(math.floor(value))
    used = sum(allocation.values())
    remainders = sorted(
        ((raw[layer_key] - math.floor(raw[layer_key]), layer_key) for layer_key in weights),
        reverse=True,
    )
    for _fraction, layer_key in remainders[: max(0, total_capacity - used)]:
        allocation[layer_key] += 1
    return allocation


def _feature_array(summary: dict[str, Any], layer_key: str, feature_key: str, num_experts: int) -> list[float]:
    layer_blob = summary["layers"][layer_key]
    raw = layer_blob.get(feature_key)
    if isinstance(raw, list) and len(raw) == num_experts:
        return [float(value) for value in raw]
    return [0.0] * num_experts


def _build_layer_expert_rows(
    summaries: list[tuple[str, dict[str, Any]]],
    *,
    layer_key: str,
    signal_key: str,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")
    first_summary = summaries[0][1]
    num_experts = len(first_summary["layers"][layer_key][signal_key])

    reap_arrays = [
        _feature_array(summary, layer_key, "reap", num_experts) for _label, summary in summaries
    ]
    weighted_arrays = [
        _feature_array(summary, layer_key, "weighted_ean_sum", num_experts) for _label, summary in summaries
    ]
    freq_arrays = [
        _feature_array(summary, layer_key, "expert_frequency", num_experts) for _label, summary in summaries
    ]

    reap_scale = _safe_max(value for array in reap_arrays for value in array)
    weighted_scale = _safe_max(value for array in weighted_arrays for value in array)
    freq_scale = _safe_max(value for array in freq_arrays for value in array)

    expert_rows: list[dict[str, Any]] = []
    for expert_idx in range(num_experts):
        signals_by_summary = {
            label: float(summary["layers"][layer_key][signal_key][expert_idx])
            for label, summary in summaries
        }
        combined_signal = sum(signals_by_summary.values())
        peak_signal = max(signals_by_summary.values()) if signals_by_summary else 0.0
        stability = (
            min(signals_by_summary.values()) / peak_signal if peak_signal > 0 else 0.0
        )

        reap_mean = _mean([array[expert_idx] for array in reap_arrays])
        weighted_mean = _mean([array[expert_idx] for array in weighted_arrays])
        freq_mean = _mean([array[expert_idx] for array in freq_arrays])
        reap_norm = _normalized_value(reap_mean, reap_scale)
        weighted_norm = _normalized_value(weighted_mean, weighted_scale)
        freq_norm = _normalized_value(freq_mean, freq_scale)

        if selection_strategy == "support_v1":
            selection_mass = (0.60 * reap_norm) + (0.25 * weighted_norm) + (0.15 * freq_norm)
        else:
            selection_mass = combined_signal

        expert_rows.append(
            {
                "expert": expert_idx,
                "signalsBySummary": signals_by_summary,
                "combinedSignal": round(combined_signal, 8),
                "peakSignal": round(peak_signal, 8),
                "stability": round(stability, 8),
                "reapMean": round(reap_mean, 8),
                "weightedEanMean": round(weighted_mean, 8),
                "expertFrequencyMean": round(freq_mean, 8),
                "selectionMass": round(selection_mass, 8),
                "selectionStrategy": selection_strategy,
            }
        )
    return expert_rows


def _build_floor_seed_scores(
    summaries: list[tuple[str, dict[str, Any]]],
    *,
    layer_keys: list[str],
    signal_key: str,
) -> dict[str, dict[str, Any]]:
    first_summary = summaries[0][1]
    out: dict[str, dict[str, Any]] = {}
    for layer_key in layer_keys:
        normalized_key = _normalize_layer_key(layer_key)
        num_experts = len(first_summary["layers"][layer_key][signal_key])
        combined_scores: list[float] = []
        for expert_idx in range(num_experts):
            values = [
                float(summary["layers"][layer_key][signal_key][expert_idx])
                for _label, summary in summaries
            ]
            peak = max(values) if values else 0.0
            stability = (min(values) / peak) if peak > 0 else 0.0
            combined = sum(values) * 0.85 + stability * 0.15
            combined_scores.append(round(combined, 8))
        out[normalized_key] = {"scores": combined_scores}
    return out


def build_dynamic_plan(
    summaries: list[tuple[str, dict[str, Any]]],
    *,
    signal_key: str,
    model_config: dict[str, Any],
    activation_records: list[dict[str, Any]] | None = None,
    layer_importance: dict[str, Any] | None = None,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    core_selection_mode: str = "selection_mass",
    floor_layer_weight_mode: str = DEFAULT_FLOOR_LAYER_WEIGHT_MODE,
    rotation_policy: str = DEFAULT_ROTATION_POLICY,
    max_resident_ratio: float = DEFAULT_MAX_RESIDENT_RATIO,
    max_resident_gib: float | None = None,
    core_budget_fraction: float = DEFAULT_CORE_BUDGET_FRACTION,
    specialist_budget_fraction: float = DEFAULT_SPECIALIST_BUDGET_FRACTION,
    candidate_pool_multiplier: float = DEFAULT_CANDIDATE_POOL_MULTIPLIER,
    max_refreshes_per_request: int = DEFAULT_MAX_REFRESHES_PER_REQUEST,
    reserve_fraction: float = 0.0,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one summary is required")
    first_summary = summaries[0][1]
    layer_keys = list(first_summary["layers"].keys())
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")
    if core_selection_mode not in CORE_SELECTION_MODES:
        raise ValueError(f"Unknown core_selection_mode: {core_selection_mode}")
    if rotation_policy not in ROTATION_POLICIES:
        raise ValueError(f"Unknown rotation_policy: {rotation_policy}")

    # Clamp reserve_fraction to valid range
    reserve_fraction = max(0.0, min(1.0, float(reserve_fraction)))

    # When reserve_fraction > 0, boost specialist capacity so the base budget
    # portion (1 - reserve_fraction) still matches zero-reserve behavior while
    # the reserve portion provides additional headroom for support slices.
    # Also boost the candidate pool multiplier proportionally.
    reserve_specialist_boost = 1.0 / (1.0 - reserve_fraction) if reserve_fraction < 1.0 else 1.0
    effective_specialist_fraction = min(1.0 - 1e-9, specialist_budget_fraction * reserve_specialist_boost)
    effective_core_fraction = 1.0 - effective_specialist_fraction
    effective_candidate_multiplier = candidate_pool_multiplier * reserve_specialist_boost

    budget = compute_dynamic_budget(
        model_config,
        max_resident_ratio=max_resident_ratio,
        max_resident_gib=max_resident_gib,
        core_budget_fraction=effective_core_fraction if reserve_fraction > 0 else core_budget_fraction,
        specialist_budget_fraction=effective_specialist_fraction if reserve_fraction > 0 else specialist_budget_fraction,
        candidate_pool_multiplier=effective_candidate_multiplier if reserve_fraction > 0 else candidate_pool_multiplier,
        max_refreshes_per_request=max_refreshes_per_request,
    )
    # Record the original requested fractions alongside the effective ones
    budget["requested_core_budget_fraction"] = core_budget_fraction
    budget["requested_specialist_budget_fraction"] = specialist_budget_fraction
    budget["requested_candidate_pool_multiplier"] = candidate_pool_multiplier
    budget["reserve_fraction"] = round(reserve_fraction, 6)
    labels = [label for label, _summary in summaries]
    activation_records = activation_records or []
    activation_summary = summarize_activation_records(activation_records)
    per_layer: dict[str, Any] = {}
    task_family_priors: dict[str, dict[str, list[str]]] = {"global": {}}
    total_layers = len(layer_keys)
    total_core_capacity = max(total_layers, int(budget["core_experts_per_layer_target"]) * total_layers)
    total_specialist_capacity = max(total_layers, int(budget["specialist_experts_per_layer_target"]) * total_layers)
    if core_selection_mode == "floor_seeded":
        protected_core_by_layer = _greedy_allocate_layer_scores(
            _build_floor_seed_scores(
                summaries,
                layer_keys=layer_keys,
                signal_key=signal_key,
            ),
            total_core_capacity,
            min_per_layer=1,
            layer_weight_mode=floor_layer_weight_mode,
        )
        core_targets = {
            layer_key: len(experts)
            for layer_key, experts in protected_core_by_layer.items()
        }
    else:
        protected_core_by_layer = None
        core_targets = _allocate_per_layer_counts(
            layer_keys,
            total_core_capacity,
            layer_importance=layer_importance,
            min_per_layer=1,
        )
    specialist_targets = _allocate_per_layer_counts(
        layer_keys,
        total_specialist_capacity,
        layer_importance=layer_importance,
        min_per_layer=1,
    )
    candidate_targets = {
        _normalize_layer_key(layer_key): max(
            specialist_targets[_normalize_layer_key(layer_key)],
            int(math.ceil(specialist_targets[_normalize_layer_key(layer_key)] * budget["candidate_pool_multiplier"])),
        )
        for layer_key in layer_keys
    }

    for layer_key in layer_keys:
        normalized_key = _normalize_layer_key(layer_key)
        num_experts = len(first_summary["layers"][layer_key][signal_key])
        expert_rows = _build_layer_expert_rows(
            summaries,
            layer_key=layer_key,
            signal_key=signal_key,
            selection_strategy=selection_strategy,
        )

        expert_rows.sort(
            key=lambda row: (
                row["selectionMass"] if selection_strategy == "support_v1" else (row["combinedSignal"] * 0.85 + row["stability"] * 0.15),
                row["reapMean"],
                row["weightedEanMean"],
                -row["expert"],
            ),
            reverse=True,
        )
        if protected_core_by_layer is not None:
            protected_experts = set(
                int(expert) for expert in protected_core_by_layer.get(normalized_key, [])
            )
            core_rows = [row for row in expert_rows if int(row["expert"]) in protected_experts]
            remaining_rows = [row for row in expert_rows if int(row["expert"]) not in protected_experts]
        else:
            core_count = min(num_experts, max(1, int(core_targets[normalized_key])))
            core_rows = expert_rows[:core_count]
            remaining_rows = expert_rows[core_count:]
        layer_specialist_target = min(max(1, int(specialist_targets[normalized_key])), max(1, num_experts - len(core_rows)))
        slice_size = _slice_size_target(layer_specialist_target)
        slice_catalog: list[dict[str, Any]] = []
        for slice_index in range(0, len(remaining_rows), slice_size):
            slice_rows = remaining_rows[slice_index : slice_index + slice_size]
            if not slice_rows:
                continue
            slice_id = f"{normalized_key}_slice_{slice_index // slice_size:02d}"
            signals_by_summary = {
                label: round(sum(float(row["signalsBySummary"][label]) for row in slice_rows), 8)
                for label in labels
            }
            slice_catalog.append(
                {
                    "sliceId": slice_id,
                    "experts": sorted(int(row["expert"]) for row in slice_rows),
                    "byteCost": len(slice_rows) * budget["per_expert_bytes"],
                    "activationMass": round(sum(float(row["selectionMass"]) for row in slice_rows), 8),
                    "rawActivationMass": round(sum(float(row["combinedSignal"]) for row in slice_rows), 8),
                    "reapMass": round(sum(float(row["reapMean"]) for row in slice_rows), 8),
                    "weightedEanMass": round(sum(float(row["weightedEanMean"]) for row in slice_rows), 8),
                    "expertFrequencyMass": round(sum(float(row["expertFrequencyMean"]) for row in slice_rows), 8),
                    "coactivationScore": _slice_similarity(slice_rows),
                    "signalsBySummary": signals_by_summary,
                    "taskPriors": _task_priors_for_slice(
                        {
                            "activationMass": round(sum(float(row["selectionMass"]) for row in slice_rows), 8),
                            "signalsBySummary": signals_by_summary,
                        },
                        labels,
                    ),
                }
            )
        per_layer[normalized_key] = {
            "rawLayerKey": layer_key,
            "numExperts": num_experts,
            "coreExperts": sorted(int(row["expert"]) for row in core_rows),
            "coreActivationMass": round(sum(float(row["selectionMass"]) for row in core_rows), 8),
            "coreRawActivationMass": round(sum(float(row["combinedSignal"]) for row in core_rows), 8),
            "sliceCatalog": slice_catalog,
            "coreByteCost": len(core_rows) * budget["per_expert_bytes"],
            "specialistActiveExpertTarget": layer_specialist_target,
            "specialistCandidateExpertTarget": min(
                num_experts - len(core_rows),
                candidate_targets[normalized_key],
            ),
            "specialistBudgetBytes": int(layer_specialist_target * budget["per_expert_bytes"]),
            "candidateBudgetBytes": int(min(num_experts - len(core_rows), candidate_targets[normalized_key]) * budget["per_expert_bytes"]),
        }
        task_family_priors["global"][normalized_key] = [row["sliceId"] for row in sorted(slice_catalog, key=lambda row: row["activationMass"], reverse=True)]
        for label in labels:
            task_family_priors.setdefault(label, {})[normalized_key] = [
                row["sliceId"]
                for row in sorted(
                    slice_catalog,
                    key=lambda row: (row["signalsBySummary"].get(label, 0.0), row["activationMass"]),
                    reverse=True,
                )
            ]

    CLUSTER_LABEL_MAP = {
        "code": "code",
        "math": "math",
        "writing": "general",
        "research": "math",
        "ops": "code",
        "general": "general",
    }
    TAG_TO_LABEL_ALIAS = {
        "writing": "general",
        "research": "math",
        "ops": "code",
    }
    for alias_tag, alias_label in TAG_TO_LABEL_ALIAS.items():
        if alias_label in task_family_priors and alias_tag not in task_family_priors:
            task_family_priors[alias_tag] = task_family_priors[alias_label]

    all_domain_tags = sorted(set(list(DOMAIN_PATTERNS.keys()) + ["general"]))
    cluster_priors: dict[str, dict[str, list[str]]] = {}
    for tag in all_domain_tags:
        preferred_label = CLUSTER_LABEL_MAP.get(tag)
        if preferred_label and preferred_label in labels:
            cluster_priors[tag] = task_family_priors.get(preferred_label, {})
        else:
            cluster_priors[tag] = task_family_priors.get("global", {})

    return {
        "mode": "dynamic_core_specialist",
        "model": first_summary.get("model"),
        "signalKey": signal_key,
        "selectionStrategy": selection_strategy,
        "coreSelectionMode": core_selection_mode,
        "rotationPolicy": rotation_policy,
        "sourceSummaries": [
            {
                "label": label,
                "workflow": summary.get("workflow", label),
                "processedSamples": summary.get("processedSamples", 0),
                "totalTokens": summary.get("totalTokens", 0),
            }
            for label, summary in summaries
        ],
        "budget": budget,
        "perLayer": per_layer,
        "scorerArtifacts": {
            "selectionStrategy": selection_strategy,
            "coreSelectionMode": core_selection_mode,
            "rotationPolicy": rotation_policy,
            "layerImportance": layer_importance or {},
            "floorSelection": {
                "layerWeightMode": floor_layer_weight_mode,
                "selectedCoreByLayer": protected_core_by_layer or {},
            },
            "layerBudgetTargets": {
                "coreExpertsPerLayer": core_targets,
                "specialistExpertsPerLayer": specialist_targets,
                "candidateExpertsPerLayer": candidate_targets,
            },
            **(
                {"supportEstimatorConfig": {
                    "mode": "prefill_reserve",
                    "reserve_fraction": round(reserve_fraction, 6),
                }}
                if reserve_fraction > 0
                else {}
            ),
            "taskFamilySlicePriors": task_family_priors,
            "promptClusterPriors": cluster_priors,
            "featureNormalization": activation_summary.get("featureNormalization", {}),
            "activationCorpus": {
                "recordCount": activation_summary.get("record_count", 0),
                "sourceCounts": activation_summary.get("source_counts", {}),
                "domainTagCounts": activation_summary.get("domain_tag_counts", {}),
            },
        },
        "summary": {
            "layerCount": len(layer_keys),
            "totalCoreExperts": sum(len(layer["coreExperts"]) for layer in per_layer.values()),
            "totalSlices": sum(len(layer["sliceCatalog"]) for layer in per_layer.values()),
            "residentFractionPctMin": round(100.0 * budget["max_resident_ratio"], 8),
            "residentFractionPctMax": round(100.0 * budget["max_resident_ratio"], 8),
        },
    }


def build_dynamic_markdown(plan: dict[str, Any]) -> str:
    budget = plan["budget"]
    selection_mode = str(plan.get("selectionMode") or plan.get("mode"))
    heading = "# REAP dynamic exact-floor plan" if selection_mode == "dynamic_exact_floor" else "# REAP dynamic core + specialist plan"
    swappable_gib = budget.get("swappable_expert_budget_gib")
    if swappable_gib is None:
        swappable_gib = round(float(budget.get("swappable_expert_budget_bytes", 0)) / (1024**3), 6)
    lines = [
        heading,
        "",
        f"- model: `{plan['model']}`",
        f"- mode: `{plan['mode']}`",
        f"- selection mode: `{selection_mode}`",
        f"- selection strategy: `{plan.get('selectionStrategy', DEFAULT_SELECTION_STRATEGY)}`",
        f"- core selection mode: `{plan.get('coreSelectionMode', 'selection_mass')}`",
        f"- rotation policy: `{plan.get('rotationPolicy', DEFAULT_ROTATION_POLICY)}`",
        f"- signal key: `{plan['signalKey']}`",
        f"- hard resident cap ratio: {float(budget['max_resident_ratio']):.4f}",
        f"- hard resident cap GiB: {budget['max_resident_gib']}",
        f"- swappable expert budget GiB: {swappable_gib}",
        f"- core budget bytes: {budget['core_budget_bytes']}",
        f"- specialist budget bytes: {budget['specialist_budget_bytes']}",
        f"- candidate pool multiplier: {float(budget['candidate_pool_multiplier']):.2f}",
        f"- max refreshes per request: {budget['max_refreshes_per_request']}",
        "",
        "## Activation corpus",
        "",
        f"- records: {plan['scorerArtifacts']['activationCorpus']['recordCount']}",
        f"- domain tags: {plan['scorerArtifacts']['activationCorpus']['domainTagCounts']}",
        "",
    ]
    for layer_key, layer in plan["perLayer"].items():
        lines.extend(
            [
                f"## {layer_key}",
                f"- core experts: {layer['coreExperts']}",
                f"- core bytes: {layer['coreByteCost']}",
                f"- specialist active target: {layer['specialistActiveExpertTarget']}",
                f"- specialist candidate target: {layer['specialistCandidateExpertTarget']}",
            ]
        )
        for slice_row in layer["sliceCatalog"]:
            lines.append(
                f"- {slice_row['sliceId']}: experts={slice_row['experts']} bytes={slice_row['byteCost']} activation_mass={slice_row['activationMass']} coactivation={slice_row['coactivationScore']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _expert_weight_map(blob: dict[str, Any] | None) -> dict[int, float]:
    normalized: dict[int, float] = {}
    if not blob:
        return normalized
    for key, value in blob.items():
        try:
            normalized[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _slice_prior_mass(slice_row: dict[str, Any], expert_weights: dict[int, float]) -> float:
    if not expert_weights:
        return 0.0
    return sum(float(expert_weights.get(int(expert), 0.0)) for expert in slice_row.get("experts", []))


def _layer_activation_scale(layer: dict[str, Any]) -> float:
    slice_catalog = list(layer.get("sliceCatalog", []))
    if not slice_catalog:
        return 0.0
    return max(float(row.get("activationMass", 0.0) or 0.0) for row in slice_catalog)


def _support_estimator_config(plan: dict[str, Any]) -> dict[str, Any]:
    scorer = plan.get("scorerArtifacts", {}) or {}
    config = dict(scorer.get("supportEstimatorConfig") or {})
    if not config and (scorer.get("benchmarkMissPriors") or scorer.get("tagMissPriors")):
        config = {"mode": "full_prefill"}
    config.setdefault("mode", "none")
    config.setdefault("reserve_fraction", 0.0)
    config.setdefault("late_layer_start_frac", 0.0)
    config.setdefault("benchmark_scale", 20.0)
    config.setdefault("tag_scale", 8.0)
    config.setdefault("lexical_scale", 10.0)
    config.setdefault("lexical_k", DEFAULT_LEXICAL_K)
    return config


def _support_router_config(plan: dict[str, Any]) -> dict[str, Any]:
    scorer = plan.get("scorerArtifacts", {}) or {}
    config = dict(scorer.get("supportRouter") or {})
    config.setdefault("enabled", False)
    config.setdefault("manifest_path", "")
    config.setdefault("benchmarks", [])
    config.setdefault("exclude_benchmarks", [])
    config.setdefault("shortlist_size", 8)
    config.setdefault("rerank_support_order", True)
    config.setdefault("rerank_candidate_order", True)
    config.setdefault("min_slice_probability", 0.0)
    return config


def _prompt_shape_features(prompt_text: str) -> dict[str, float]:
    normalized = normalize_prompt_text(prompt_text)
    words = [token for token in normalized.split(" ") if token]
    return {
        "shape::prompt_len_chars": float(len(normalized)),
        "shape::prompt_len_words": float(len(words)),
        "shape::newline_count": float(prompt_text.count("\n")),
        "shape::digit_ratio": (sum(1 for ch in normalized if ch.isdigit()) / len(normalized)) if normalized else 0.0,
    }


def _support_router_prompt_feature_text(prompt_text: str, benchmark: str, tags: list[str]) -> str:
    return f"benchmark:{benchmark} tags:{' '.join(tags)} {normalize_prompt_text(prompt_text)}".strip()


@lru_cache(maxsize=4)
def _load_support_router_bundle(manifest_path: str) -> dict[str, Any] | None:
    resolved = Path(manifest_path).expanduser().resolve()
    if not resolved.exists():
        return None
    try:
        import joblib
        from scipy.sparse import hstack
    except ModuleNotFoundError:
        return None

    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    text_vectorizer_path = manifest.get("text_vectorizer_path")
    meta_vectorizer_path = manifest.get("meta_vectorizer_path")
    if not text_vectorizer_path or not meta_vectorizer_path:
        return None
    layer_models: dict[str, Any] = {}
    for layer_key, layer_info in (manifest.get("trained_layers") or {}).items():
        model_path = layer_info.get("model_path")
        if not model_path:
            continue
        model_file = Path(model_path)
        if not model_file.exists():
            continue
        layer_models[str(layer_key)] = joblib.load(model_file)
    return {
        "manifest": manifest,
        "text_vectorizer": joblib.load(text_vectorizer_path),
        "meta_vectorizer": joblib.load(meta_vectorizer_path),
        "layer_models": layer_models,
        "hstack": hstack,
    }


def _support_router_probabilities(
    manifest_path: str,
    *,
    layer_key: str,
    prompt_text: str,
    benchmark: str,
    tags: list[str],
) -> dict[str, float]:
    bundle = _load_support_router_bundle(manifest_path)
    if not bundle:
        return {}
    estimator = bundle["layer_models"].get(layer_key)
    if estimator is None or not hasattr(estimator, "classes_"):
        return {}
    feature_text = _support_router_prompt_feature_text(prompt_text, benchmark, tags)
    meta = _prompt_shape_features(prompt_text)
    meta[f"benchmark::{benchmark or 'unknown'}"] = 1.0
    for tag in tags:
        meta[f"tag::{tag}"] = 1.0
    X = bundle["hstack"](
        [
            bundle["text_vectorizer"].transform([feature_text]),
            bundle["meta_vectorizer"].transform([meta]),
        ]
    ).tocsr()
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)[0]
    elif hasattr(estimator, "predict_log_proba"):
        probs = [math.exp(value) for value in estimator.predict_log_proba(X)[0]]
    else:
        return {}
    return {
        str(label): float(prob)
        for label, prob in zip(getattr(estimator, "classes_", []), probs)
        if str(label) != "__none__"
    }


def _apply_support_router_rerank(
    ordered_slice_ids: list[str],
    *,
    layer_key: str,
    prompt_text: str,
    benchmark: str,
    tags: list[str],
    config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if not config.get("enabled"):
        return ordered_slice_ids, {"applied": False, "reason": "disabled"}
    manifest_path = str(config.get("manifest_path") or "").strip()
    if not manifest_path:
        return ordered_slice_ids, {"applied": False, "reason": "missing_manifest"}
    router_probs = _support_router_probabilities(
        manifest_path,
        layer_key=layer_key,
        prompt_text=prompt_text,
        benchmark=benchmark or "unknown",
        tags=tags,
    )
    if not router_probs:
        return ordered_slice_ids, {"applied": False, "reason": "no_probabilities"}
    shortlist_size = max(1, int(config.get("shortlist_size", 8) or 8))
    min_slice_probability = float(config.get("min_slice_probability", 0.0) or 0.0)
    shortlist = list(ordered_slice_ids[:shortlist_size])
    remainder = list(ordered_slice_ids[shortlist_size:])
    boosted = [
        slice_id
        for slice_id in shortlist
        if float(router_probs.get(slice_id, 0.0) or 0.0) > min_slice_probability
    ]
    if not boosted:
        return ordered_slice_ids, {
            "applied": False,
            "reason": "no_shortlist_overlap",
            "shortlist_size": shortlist_size,
        }
    reranked_shortlist = sorted(
        shortlist,
        key=lambda slice_id: (-float(router_probs.get(slice_id, 0.0) or 0.0), shortlist.index(slice_id)),
    )
    return reranked_shortlist + remainder, {
        "applied": True,
        "reason": "ok",
        "shortlist_size": shortlist_size,
        "shortlist": shortlist,
        "reranked_shortlist": reranked_shortlist,
        "router_scores": {slice_id: round(float(router_probs.get(slice_id, 0.0) or 0.0), 8) for slice_id in reranked_shortlist[:shortlist_size]},
    }


def _support_router_allowed_for_benchmark(config: dict[str, Any], benchmark: str) -> tuple[bool, str]:
    normalized = str(benchmark or "unknown")
    allowed = {str(item) for item in (config.get("benchmarks") or []) if str(item).strip()}
    denied = {str(item) for item in (config.get("exclude_benchmarks") or []) if str(item).strip()}
    if allowed and normalized not in allowed:
        return False, "benchmark_not_allowed"
    if normalized in denied:
        return False, "benchmark_excluded"
    return True, "ok"


def rank_slice_ids_for_prompt(
    plan: dict[str, Any],
    prompt_text: str,
    *,
    benchmark: str | None = None,
    router_misses: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    messages: list[dict[str, Any]] | list[str] | None = None,
    conversation_context: Any = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    conversation_text_parts = [
        _conversation_context_to_text(conversation_context),
        _conversation_context_to_text(messages),
    ]
    conversation_text = "\n".join(
        part for part in conversation_text_parts if isinstance(part, str) and part.strip()
    )
    routing_text = prompt_text
    if conversation_text:
        routing_text = f"{conversation_text}\n\nlatest_user_prompt: {prompt_text}"

    tags = infer_domain_tags(routing_text)
    if benchmark:
        tags.extend(BENCHMARK_TO_TAGS.get(benchmark, []))
    tags = list(dict.fromkeys(tags))
    prompt_fingerprint = normalize_prompt_text(routing_text).lower()
    prompt_tokens = tokenize_prompt_text(routing_text)
    priors = plan.get("scorerArtifacts", {}).get("taskFamilySlicePriors", {})
    cluster_priors = plan.get("scorerArtifacts", {}).get("promptClusterPriors", {})
    benchmark_miss_priors = plan.get("scorerArtifacts", {}).get("benchmarkMissPriors", {})
    tag_miss_priors = plan.get("scorerArtifacts", {}).get("tagMissPriors", {})
    lexical_miss_exemplars = plan.get("scorerArtifacts", {}).get("lexicalMissExemplars", [])
    support_config = _support_estimator_config(plan)
    support_router_config = _support_router_config(plan)
    rotation_policy = str(
        plan.get("rotationPolicy")
        or plan.get("scorerArtifacts", {}).get("rotationPolicy")
        or DEFAULT_ROTATION_POLICY
    )
    ranked: dict[str, list[str]] = {}
    support_ranked: dict[str, list[str]] = {}
    candidate_ranked: dict[str, list[str]] = {}
    layer_debug: dict[str, Any] = {}
    total_layers = max(1, len(plan.get("perLayer", {})))
    for layer_key, layer in plan.get("perLayer", {}).items():
        slice_rows = {row["sliceId"]: row for row in layer.get("sliceCatalog", [])}
        layer_activation_scale = _layer_activation_scale(layer)
        benchmark_name = str(benchmark or "unknown")
        lexical_scale = float((support_config.get("lexical_scale_by_benchmark") or {}).get(benchmark_name, support_config.get("lexical_scale", 10.0)) or 0.0)
        benchmark_expert_weights = _expert_weight_map(
            benchmark_miss_priors.get(benchmark_name, {}).get(layer_key, {})
        )
        lexical_bucket: dict[int, float] = {}
        lexical_neighbors = []
        if lexical_miss_exemplars and prompt_tokens:
            scored = []
            for exemplar in lexical_miss_exemplars:
                exemplar_tokens = set(str(token) for token in exemplar.get("tokens", []))
                sim = _jaccard_similarity(prompt_tokens, exemplar_tokens)
                if sim <= 0.0:
                    continue
                if benchmark_name and exemplar.get("benchmark") == benchmark_name:
                    sim += 0.05
                if set(exemplar.get("tags", [])) & set(tags):
                    sim += 0.03
                scored.append((sim, exemplar))
            scored.sort(key=lambda item: item[0], reverse=True)
            lexical_neighbors = scored[: int(support_config.get("lexical_k", DEFAULT_LEXICAL_K) or DEFAULT_LEXICAL_K)]
            for sim, exemplar in lexical_neighbors:
                layer_weights = _expert_weight_map((exemplar.get("by_layer") or {}).get(layer_key, {}))
                for expert, value in layer_weights.items():
                    lexical_bucket[expert] = lexical_bucket.get(expert, 0.0) + (sim * value)
        lexical_expert_weights = _expert_weight_map({str(k): v for k, v in lexical_bucket.items()})
        tag_expert_weights = {
            tag: _expert_weight_map(tag_miss_priors.get(tag, {}).get(layer_key, {}))
            for tag in tags
        }
        miss_experts = set((router_misses or {}).get("by_layer", {}).get(layer_key, {}).get("inactive_experts", []))
        try:
            layer_idx = int(str(layer_key).replace("layer_", ""))
        except ValueError:
            layer_idx = 0
        support_allowed = layer_idx >= int(float(support_config.get("late_layer_start_frac", 0.0)) * total_layers)

        base_scores: list[tuple[float, str]] = []
        support_scores: list[tuple[float, str]] = []
        for slice_id, slice_row in slice_rows.items():
            base_score = float(slice_row.get("activationMass", 0.0))
            support_score = base_score
            benchmark_prior_mass = _slice_prior_mass(slice_row, benchmark_expert_weights)
            if support_allowed and benchmark_prior_mass > 0.0 and layer_activation_scale > 0.0:
                support_score += benchmark_prior_mass * layer_activation_scale * float(support_config.get("benchmark_scale", 20.0))
            tag_prior_mass = 0.0
            lexical_prior_mass = _slice_prior_mass(slice_row, lexical_expert_weights)
            for tag in tags:
                for prior_id in cluster_priors.get(tag, {}).get(layer_key, []):
                    if prior_id == slice_id:
                        base_score += float(slice_row.get("activationMass", 0.0)) * 0.15
                        support_score += float(slice_row.get("activationMass", 0.0)) * 0.15
                label_prior = priors.get(tag, {}).get(layer_key, [])
                for prior_id in label_prior:
                    if prior_id == slice_id:
                        base_score += float(slice_row.get("taskPriors", {}).get(tag, 0.0)) * 0.25
                        support_score += float(slice_row.get("taskPriors", {}).get(tag, 0.0)) * 0.25
                tag_prior_mass += _slice_prior_mass(slice_row, tag_expert_weights.get(tag, {}))
            if support_allowed and tag_prior_mass > 0.0 and layer_activation_scale > 0.0:
                support_score += tag_prior_mass * layer_activation_scale * float(support_config.get("tag_scale", 8.0))
            if support_allowed and lexical_prior_mass > 0.0 and layer_activation_scale > 0.0 and lexical_scale > 0.0:
                support_score += lexical_prior_mass * layer_activation_scale * lexical_scale
            for expert in slice_row.get("experts", []):
                if expert in miss_experts:
                    base_score += 1000.0
                    support_score += 1000.0
            base_scores.append((base_score, slice_id))
            support_scores.append((support_score, slice_id))

        base_scores.sort(reverse=True)
        support_scores.sort(reverse=True)
        rotation_window = min(
            max(2, int(layer.get("specialistCandidateExpertTarget", 0) or 0)),
            4,
            len(base_scores),
        )
        rotation_offset = 0
        apply_rotation = (
            rotation_policy == "late_prompt_hash"
            and rotation_window > 1
            and layer_idx >= (total_layers // 2)
        )
        if apply_rotation:
            semantic_offset = sum(TAG_ROTATION_WEIGHTS.get(tag, 0) for tag in tags)
            semantic_offset += _stable_bucket(str(benchmark or "none"), rotation_window)
            semantic_offset += _stable_bucket(str(conversation_id or "none"), rotation_window)
            if turn_index is not None:
                semantic_offset += int(turn_index) % rotation_window
            lexical_offset = _stable_bucket(
                f"{layer_key}|{prompt_fingerprint}",
                rotation_window,
            )
            rotation_offset = (semantic_offset + lexical_offset) % rotation_window
            top_window = base_scores[:rotation_window]
            base_scores = top_window[rotation_offset:] + top_window[:rotation_offset] + base_scores[rotation_window:]
            top_window = support_scores[:rotation_window]
            support_scores = top_window[rotation_offset:] + top_window[:rotation_offset] + support_scores[rotation_window:]

        support_router_debug = {
            "support_order": {"applied": False, "reason": "disabled"},
            "candidate_order": {"applied": False, "reason": "disabled"},
        }
        if support_router_config.get("enabled"):
            router_allowed, router_reason = _support_router_allowed_for_benchmark(
                support_router_config,
                benchmark_name,
            )
            if not router_allowed:
                support_router_debug = {
                    "support_order": {"applied": False, "reason": router_reason},
                    "candidate_order": {"applied": False, "reason": router_reason},
                }
            else:
                support_sorted_ids = [slice_id for _score, slice_id in support_scores]
                base_sorted_ids = [slice_id for _score, slice_id in base_scores]
                if support_router_config.get("rerank_support_order", True):
                    support_sorted_ids, support_router_debug["support_order"] = _apply_support_router_rerank(
                        support_sorted_ids,
                        layer_key=layer_key,
                        prompt_text=prompt_text,
                        benchmark=benchmark_name,
                        tags=tags,
                        config=support_router_config,
                    )
                if support_router_config.get("rerank_candidate_order", True):
                    base_sorted_ids, support_router_debug["candidate_order"] = _apply_support_router_rerank(
                        base_sorted_ids,
                        layer_key=layer_key,
                        prompt_text=prompt_text,
                        benchmark=benchmark_name,
                        tags=tags,
                        config=support_router_config,
                    )
                support_scores = [
                    (next((score for score, candidate_id in support_scores if candidate_id == slice_id), 0.0), slice_id)
                    for slice_id in support_sorted_ids
                ]
                base_scores = [
                    (next((score for score, candidate_id in base_scores if candidate_id == slice_id), 0.0), slice_id)
                    for slice_id in base_sorted_ids
                ]

        mode = str(support_config.get("mode", "none"))
        selected_scores = support_scores if mode == "full_prefill" else base_scores
        candidate_scores = support_scores if mode in {"full_prefill", "candidate_only", "prefill_reserve"} else base_scores
        ranked[layer_key] = [slice_id for _score, slice_id in selected_scores]
        support_ranked[layer_key] = [slice_id for _score, slice_id in support_scores]
        candidate_ranked[layer_key] = [slice_id for _score, slice_id in candidate_scores]
        layer_debug[layer_key] = {
            "tags": tags,
            "apply_rotation": apply_rotation,
            "support_mode": mode,
            "support_allowed": support_allowed,
            "rotation_window": rotation_window,
            "rotation_offset": rotation_offset,
            "benchmark_prior_count": len(benchmark_expert_weights),
            "tag_prior_counts": {tag: len(weights) for tag, weights in tag_expert_weights.items()},
            "lexical_prior_count": len(lexical_expert_weights),
            "lexical_neighbor_count": len(lexical_neighbors),
            "lexical_scale": lexical_scale,
            "base_ranked": [{"slice_id": slice_id, "score": round(score, 8)} for score, slice_id in base_scores[:10]],
            "support_ranked": [{"slice_id": slice_id, "score": round(score, 8)} for score, slice_id in support_scores[:10]],
            "support_router": support_router_debug,
        }
    return ranked, {
        "tags": tags,
        "routing_text_preview": normalize_prompt_text(routing_text)[:512],
        "conversation_context_chars": len(conversation_text),
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "layers": layer_debug,
        "support_ranked_slice_ids": support_ranked,
        "candidate_ranked_slice_ids": candidate_ranked,
        "support_config": support_config,
    }


def assemble_active_set(
    plan: dict[str, Any],
    ranked_slice_ids_by_layer: dict[str, list[str]],
    *,
    support_ranked_slice_ids_by_layer: dict[str, list[str]] | None = None,
    candidate_ranked_slice_ids_by_layer: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    budget = plan["budget"]
    per_layer = plan["perLayer"]
    support_config = _support_estimator_config(plan)
    mode = str(support_config.get("mode", "none"))
    reserve_fraction = max(0.0, min(1.0, float(support_config.get("reserve_fraction", 0.0) or 0.0)))
    specialist_budget_remaining = int(budget["specialist_budget_bytes"])
    candidate_budget_remaining = int(math.ceil(budget["specialist_budget_bytes"] * budget["candidate_pool_multiplier"]))
    selected_slice_ids: dict[str, list[str]] = {}
    candidate_slice_ids: dict[str, list[str]] = {}
    active_set: dict[str, list[int]] = {}
    active_expert_bytes = 0
    active_expert_count = 0

    layers = list(per_layer.keys())
    if not layers:
        return {
            "active_set": {},
            "selected_slice_ids": {},
            "candidate_slice_ids": {},
            "active_expert_bytes": 0,
            "active_expert_count": 0,
        }

    for layer_key, layer in per_layer.items():
        slice_rows = {row["sliceId"]: row for row in layer.get("sliceCatalog", [])}
        base_ranked = list(ranked_slice_ids_by_layer.get(layer_key, []))
        support_ranked = list((support_ranked_slice_ids_by_layer or {}).get(layer_key, base_ranked))
        candidate_ranked = list((candidate_ranked_slice_ids_by_layer or {}).get(layer_key, support_ranked))
        active = set(layer.get("coreExperts", []))
        active_expert_bytes += len(active) * budget["per_expert_bytes"]
        active_expert_count += len(active)
        selected_slice_ids[layer_key] = []
        candidate_slice_ids[layer_key] = []
        layer_specialist_budget = int(layer.get("specialistBudgetBytes", max(1, budget["specialist_budget_bytes"] // len(layers))))
        layer_candidate_budget = int(
            layer.get(
                "candidateBudgetBytes",
                max(1, int(math.ceil((budget["specialist_budget_bytes"] * budget["candidate_pool_multiplier"]) / len(layers)))),
            )
        )
        local_specialist_budget = min(layer_specialist_budget, specialist_budget_remaining)
        local_candidate_budget = min(layer_candidate_budget, candidate_budget_remaining)
        selected_slice_cost = 0
        candidate_slice_cost = 0
        for slice_id in candidate_ranked:
            row = slice_rows.get(slice_id)
            if not row:
                continue
            if candidate_slice_cost + row["byteCost"] <= local_candidate_budget:
                candidate_slice_ids[layer_key].append(slice_id)
                candidate_slice_cost += int(row["byteCost"])

        def maybe_add_selected(slice_id: str, budget_limit: int) -> None:
            nonlocal selected_slice_cost
            if slice_id in selected_slice_ids[layer_key]:
                return
            row = slice_rows.get(slice_id)
            if not row:
                return
            if selected_slice_cost + row["byteCost"] <= budget_limit:
                selected_slice_ids[layer_key].append(slice_id)
                selected_slice_cost += int(row["byteCost"])
                active.update(int(expert) for expert in row.get("experts", []))

        if mode == "prefill_reserve":
            reserve_budget = int(local_specialist_budget * reserve_fraction)
            base_budget = max(0, local_specialist_budget - reserve_budget)
            for slice_id in base_ranked:
                maybe_add_selected(slice_id, base_budget)
            for slice_id in support_ranked:
                maybe_add_selected(slice_id, local_specialist_budget)
        else:
            selected_order = support_ranked if mode == "full_prefill" else base_ranked
            for slice_id in selected_order:
                maybe_add_selected(slice_id, local_specialist_budget)
        specialist_budget_remaining -= selected_slice_cost
        candidate_budget_remaining -= candidate_slice_cost
        active_expert_bytes += selected_slice_cost
        active_expert_count += sum(len(slice_rows[slice_id]["experts"]) for slice_id in selected_slice_ids[layer_key])
        active_set[layer_key] = sorted(active)

    return {
        "active_set": active_set,
        "selected_slice_ids": selected_slice_ids,
        "candidate_slice_ids": candidate_slice_ids,
        "active_expert_bytes": active_expert_bytes,
        "active_expert_count": active_expert_count,
    }


def build_active_set_payload(
    plan: dict[str, Any],
    prompt_text: str,
    *,
    request_id: str | None = None,
    benchmark: str | None = None,
    phase: str = "prefill",
    router_misses: dict[str, Any] | None = None,
    refresh_index: int = 0,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    messages: list[dict[str, Any]] | list[str] | None = None,
    conversation_context: Any = None,
) -> dict[str, Any]:
    ranked_slice_ids, score_debug = rank_slice_ids_for_prompt(
        plan,
        prompt_text,
        benchmark=benchmark,
        router_misses=router_misses,
        conversation_id=conversation_id,
        turn_index=turn_index,
        messages=messages,
        conversation_context=conversation_context,
    )
    assembled = assemble_active_set(
        plan,
        ranked_slice_ids,
        support_ranked_slice_ids_by_layer=score_debug.get("support_ranked_slice_ids"),
        candidate_ranked_slice_ids_by_layer=score_debug.get("candidate_ranked_slice_ids"),
    )
    payload = {
        "request_id": request_id or uuid.uuid4().hex,
        "phase": phase,
        "active_set": assembled["active_set"],
        "selected_slice_ids": assembled["selected_slice_ids"],
        "candidate_slice_ids": assembled["candidate_slice_ids"],
        "budget_bytes": assembled["active_expert_bytes"],
        "refresh_index": refresh_index,
        "rationale": {
            "benchmark": benchmark,
            "prompt_tags": score_debug["tags"],
            "conversation": {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "context_chars": int(score_debug.get("conversation_context_chars", 0) or 0),
            },
            "score_debug": score_debug["layers"],
        },
    }
    return validate_active_set_payload(payload, plan)


def compute_active_set_signature(
    active_set: dict[str, list[int]],
    selected_slice_ids: dict[str, list[str]] | None = None,
) -> str:
    canonical = {
        "active_set": {
            str(layer_key): [int(expert) for expert in experts]
            for layer_key, experts in sorted(active_set.items(), key=lambda item: str(item[0]))
        },
        "selected_slice_ids": {
            str(layer_key): [str(slice_id) for slice_id in slice_ids]
            for layer_key, slice_ids in sorted((selected_slice_ids or {}).items(), key=lambda item: str(item[0]))
        },
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def compute_plan_sha256(plan: dict[str, Any] | None) -> str | None:
    if not isinstance(plan, dict):
        return None
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_active_set_payload(payload: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    phase = payload.get("phase")
    if phase not in {"prefill", "decode_refresh"}:
        raise ValueError("phase must be 'prefill' or 'decode_refresh'")
    if not payload.get("request_id"):
        raise ValueError("request_id is required")
    per_layer = plan.get("perLayer", {})
    per_expert_bytes = int(plan.get("budget", {}).get("per_expert_bytes", 0))
    selected_slice_ids = payload.get("selected_slice_ids") or {}
    active_set = payload.get("active_set") or {}
    computed_bytes = 0
    union_validation: dict[str, Any] = {}
    core_presence_summary: dict[str, Any] = {}
    for layer_key, layer in per_layer.items():
        active = set(int(expert) for expert in active_set.get(layer_key, []))
        core_experts = set(int(expert) for expert in layer.get("coreExperts", []))
        selected_layer_slice_ids = list(selected_slice_ids.get(layer_key, []))
        if not core_experts.issubset(active):
            raise ValueError(f"{layer_key}: active_set is missing core experts")
        slice_rows = {row["sliceId"]: row for row in layer.get("sliceCatalog", [])}
        expected_union = set(core_experts)
        for slice_id in selected_layer_slice_ids:
            if slice_id not in slice_rows:
                raise ValueError(f"{layer_key}: unknown slice_id {slice_id}")
            expected_union.update(int(expert) for expert in slice_rows[slice_id].get("experts", []))
        if active != expected_union:
            raise ValueError(f"{layer_key}: active_set does not match core ∪ selected slices")
        computed_bytes += len(active) * per_expert_bytes
        union_validation[layer_key] = {
            "coreExperts": sorted(core_experts),
            "selectedSliceIds": selected_layer_slice_ids,
            "activeExperts": sorted(active),
        }
        missing_core = core_experts - active
        extras = active - expected_union
        core_presence_summary[layer_key] = {
            "core_count": len(core_experts),
            "selected_slice_count": len(selected_layer_slice_ids),
            "active_count": len(active),
            "missing_core_count": len(missing_core),
            "extra_active_count": len(extras),
            "exact_union_match": active == expected_union,
            "core_hash": hashlib.sha1(
                json.dumps(sorted(core_experts), separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16],
            "selected_slice_hash": hashlib.sha1(
                json.dumps(sorted(selected_layer_slice_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16],
            "active_hash": hashlib.sha1(
                json.dumps(sorted(active), separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16],
        }
    hard_budget = int(plan.get("budget", {}).get("swappable_expert_budget_bytes", 0))
    if computed_bytes > hard_budget:
        raise ValueError(
            f"active expert bytes {computed_bytes} exceed swappable budget {hard_budget}"
        )
    payload = dict(payload)
    payload["budget_bytes"] = computed_bytes
    payload["union_validation"] = union_validation
    payload["core_presence_summary"] = core_presence_summary
    payload["plan_sha256"] = compute_plan_sha256(plan)
    payload["active_set_signature"] = compute_active_set_signature(
        payload.get("active_set") or {},
        payload.get("selected_slice_ids") or {},
    )
    return payload


def summarize_router_misses(router_miss_payload: dict[str, Any] | None) -> dict[str, Any]:
    router_miss_payload = router_miss_payload or {}
    by_layer = router_miss_payload.get("by_layer", {})
    inactive_mass_total = 0.0
    observed_mass_total = 0.0
    inactive_expert_total = 0
    for layer_row in by_layer.values():
        inactive_mass_total += float(layer_row.get("inactive_mass", 0.0))
        observed_mass_total += float(layer_row.get("observed_mass", 0.0))
        inactive_expert_total += len(layer_row.get("inactive_experts", []))
    inactive_ratio = inactive_mass_total / observed_mass_total if observed_mass_total > 0 else 0.0
    return {
        "inactive_mass_total": round(inactive_mass_total, 8),
        "observed_mass_total": round(observed_mass_total, 8),
        "inactive_ratio": round(inactive_ratio, 8),
        "inactive_expert_total": inactive_expert_total,
    }


def should_refresh_request(
    router_miss_payload: dict[str, Any] | None,
    *,
    refreshes_used: int,
    max_refreshes: int = DEFAULT_MAX_REFRESHES_PER_REQUEST,
    miss_mass_threshold: float = DEFAULT_REFRESH_MISS_MASS_THRESHOLD,
) -> dict[str, Any]:
    if refreshes_used >= max_refreshes:
        return {"should_refresh": False, "reason": "refresh_budget_exhausted", "summary": summarize_router_misses(router_miss_payload)}
    summary = summarize_router_misses(router_miss_payload)
    if summary["inactive_ratio"] < miss_mass_threshold:
        return {"should_refresh": False, "reason": "inactive_ratio_below_threshold", "summary": summary}
    if summary["inactive_expert_total"] <= 0:
        return {"should_refresh": False, "reason": "no_inactive_experts_observed", "summary": summary}
    return {"should_refresh": True, "reason": "inactive_mass_threshold_exceeded", "summary": summary}
