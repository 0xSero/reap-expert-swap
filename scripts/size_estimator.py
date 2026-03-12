#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BF16_BYTES = 2


QWEN15_MOE_A27B_CHAT_CONFIG = {
    "model_type": "qwen2_moe",
    "vocab_size": 151936,
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "num_experts": 60,
    "moe_intermediate_size": 1408,
    "shared_expert_intermediate_size": 5632,
    "tie_word_embeddings": False,
}


def normalize_moe_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "text_config" in normalized and isinstance(normalized["text_config"], dict):
        text_config = dict(normalized["text_config"])
        merged = dict(text_config)
        for key, value in normalized.items():
            if key == "text_config":
                continue
            merged.setdefault(key, value)
        normalized = merged
    model_type = str(normalized.get("model_type") or "")
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        normalized["model_type"] = "qwen2_moe"
    return normalized


def load_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return json.loads(candidate.read_text())
    return None


def infer_model_config(record: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = record.get("manifest")
    if manifest_path:
        manifest = load_json_if_exists(manifest_path)
        if manifest:
            for key in ("model_config_json", "model_config_path", "baseline_config_json"):
                config = load_json_if_exists(manifest.get(key))
                if config:
                    return normalize_moe_config(config)

    model = str(record.get("model") or "")
    if "Qwen/Qwen1.5-MoE-A2.7B-Chat" in model or "Qwen1.5-MoE-A2.7B-Chat" in model:
        return dict(QWEN15_MOE_A27B_CHAT_CONFIG)
    return None


def infer_plan(record: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = record.get("manifest")
    if manifest_path:
        manifest = load_json_if_exists(manifest_path)
        if manifest:
            plan = load_json_if_exists(manifest.get("plan_json"))
            if plan:
                return plan
    return None


def estimate_qwen2_moe_bf16_bytes(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_moe_config(config)
    hidden_size = int(config["hidden_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    moe_intermediate_size = int(config["moe_intermediate_size"])
    shared_expert_intermediate_size = int(config["shared_expert_intermediate_size"])
    vocab_size = int(config["vocab_size"])
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    per_expert_params = 3 * hidden_size * moe_intermediate_size
    total_expert_params = num_hidden_layers * num_experts * per_expert_params

    shared_expert_params = num_hidden_layers * (
        3 * hidden_size * shared_expert_intermediate_size
    )
    attention_params = num_hidden_layers * (4 * hidden_size * hidden_size)
    router_params = num_hidden_layers * (hidden_size * num_experts)
    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size
    norm_and_misc_params = num_hidden_layers * (4 * hidden_size) + hidden_size

    always_resident_params = (
        shared_expert_params
        + attention_params
        + router_params
        + embedding_params
        + lm_head_params
        + norm_and_misc_params
    )
    total_params = always_resident_params + total_expert_params
    return {
        "architecture": "qwen2_moe",
        "bf16_bytes_per_param": BF16_BYTES,
        "per_expert_params": per_expert_params,
        "total_expert_params": total_expert_params,
        "always_resident_params": always_resident_params,
        "total_params_estimate": total_params,
        "expert_fraction_estimate": total_expert_params / total_params,
        "always_resident_fraction_estimate": always_resident_params / total_params,
        "full_bf16_bytes_estimate": total_params * BF16_BYTES,
        "full_bf16_gib_estimate": round((total_params * BF16_BYTES) / (1024**3), 6),
    }


def estimate_plan_resident_fraction(plan: dict[str, Any], record: dict[str, Any]) -> float | None:
    summary = plan.get("summary", {})
    resident_min = summary.get("residentFractionPctMin")
    resident_max = summary.get("residentFractionPctMax")
    if resident_min is not None and resident_max is not None:
        return (float(resident_min) + float(resident_max)) / 200.0

    cartridge_count = record.get("cartridge_count")
    if cartridge_count:
        return 1.0 / float(cartridge_count)

    budget_pct = record.get("gpu_budget_pct")
    if budget_pct is not None:
        return float(budget_pct) / 100.0
    return None


def estimate_resident_bf16(record: dict[str, Any]) -> dict[str, Any] | None:
    config = infer_model_config(record)
    plan = infer_plan(record)
    if not config:
        return None

    if config.get("model_type") != "qwen2_moe":
        return None

    model_bytes = estimate_qwen2_moe_bf16_bytes(config)
    resident_expert_fraction = (
        estimate_plan_resident_fraction(plan, record) if plan else None
    )
    if resident_expert_fraction is None:
        resident_expert_fraction = 1.0 / float(record["cartridge_count"])

    resident_ratio = (
        model_bytes["always_resident_fraction_estimate"]
        + model_bytes["expert_fraction_estimate"] * resident_expert_fraction
    )
    resident_bytes = model_bytes["full_bf16_bytes_estimate"] * resident_ratio
    swappable_expert_bytes = (
        model_bytes["full_bf16_bytes_estimate"]
        * model_bytes["expert_fraction_estimate"]
        * resident_expert_fraction
    )
    return {
        "method": "model_estimate_qwen2_moe",
        "resident_expert_fraction_estimate": round(resident_expert_fraction, 6),
        "resident_size_ratio_estimate": round(resident_ratio, 6),
        "resident_bf16_bytes_estimate": int(round(resident_bytes)),
        "resident_bf16_gib_estimate": round(resident_bytes / (1024**3), 6),
        "full_bf16_gib_estimate": model_bytes["full_bf16_gib_estimate"],
        "always_resident_fraction_estimate": round(
            model_bytes["always_resident_fraction_estimate"], 6
        ),
        "expert_fraction_estimate": round(model_bytes["expert_fraction_estimate"], 6),
        "resident_swappable_expert_gib_estimate": round(
            swappable_expert_bytes / (1024**3), 6
        ),
    }
