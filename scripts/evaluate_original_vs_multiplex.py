#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dynamic_reap import build_active_set_payload, summarize_router_misses
from research_gate import build_gate_markdown, evaluate_payload_gate

OPTION_LETTERS = "ABCDE"
SINGLE_TURN_PROTOCOL_VERSION = "singleturn_v0"
MULTI_TURN_PROTOCOL_VERSION = "multiturn_v0"
DEFAULT_PROTOCOL_VARIANT = "default"


MULTI_TURN_PROTOCOL_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {
        "label": "multiturn_v0_default",
        "mcq_turns": [
            "Answer the question and reply with only the final answer token.",
            "Briefly explain why your previous answer is correct. Keep it concise.",
            "Given your explanation, restate only the final answer token.",
        ],
        "math_turns": [
            "Solve the problem and end with exactly 'Final answer: <number>'.",
            "Verify your result using a different method. Keep it concise.",
            "Restate only the final numeric answer using exactly 'Final answer: <number>'.",
        ],
        "mcq_max_tokens": [8, 96, 8],
        "math_max_tokens": [256, 192, 32],
        "recommit_mcq_fallback": "none",
        "recommit_math_fallback": "none",
        "math_override_low_confidence": False,
    },
    "calib_turn1_backfill_v1": {
        "label": "multiturn_v0_calib_turn1_backfill_v1",
        "mcq_turns": [
            "Answer using one token only. Output exactly one of A, B, C, D, E (or 1 or 2 for binary). No analysis.",
            "Give one short reason for your prior answer in <=20 words.",
            "Output ONLY the same final answer token as turn 1. No extra text.",
        ],
        "math_turns": [
            "Solve and end with exactly: Final answer: <number>. No analysis text before the final line.",
            "Verify with one alternate method in <=30 words and include the computed total once.",
            "Output ONLY: Final answer: <number>.",
        ],
        "mcq_max_tokens": [12, 64, 12],
        "math_max_tokens": [320, 160, 32],
        "recommit_mcq_fallback": "turn1",
        "recommit_math_fallback": "context_numeric",
        "math_override_low_confidence": True,
    },
    "calib_reason_anchor_v2": {
        "label": "multiturn_v0_calib_reason_anchor_v2",
        "mcq_turns": [
            "Answer using one token only. Output exactly one of A, B, C, D, E (or 1 or 2 for binary).",
            "State one concise reason and explicitly include `Answer token: <token>`.",
            "Repeat only `Answer token: <token>`.",
        ],
        "math_turns": [
            "Solve and finish with `Final answer: <number>`.",
            "Verify briefly and include `Verified total: <number>`.",
            "Repeat only `Final answer: <number>`.",
        ],
        "mcq_max_tokens": [12, 96, 16],
        "math_max_tokens": [320, 192, 48],
        "recommit_mcq_fallback": "reason_then_turn1",
        "recommit_math_fallback": "context_numeric",
        "math_override_low_confidence": True,
    },
}


def stable_id(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_name: str
    dataset_config: str | None
    split: str
    question_style: str
    max_tokens: int


BENCHMARK_SPECS = [
    BenchmarkSpec("mmlu", "cais/mmlu", "all", "validation", "mcq", 8),
    BenchmarkSpec("arc_challenge", "ai2_arc", "ARC-Challenge", "validation", "mcq", 8),
    BenchmarkSpec("hellaswag", "hellaswag", None, "validation", "mcq", 8),
    BenchmarkSpec("winogrande", "winogrande", "winogrande_m", "validation", "binary", 8),
    BenchmarkSpec("gsm8k", "gsm8k", "main", "test", "math", 256),
]


def get_multi_turn_variant(variant: str) -> dict[str, Any]:
    resolved = MULTI_TURN_PROTOCOL_VARIANTS.get(variant)
    if resolved is None:
        available = ", ".join(sorted(MULTI_TURN_PROTOCOL_VARIANTS))
        raise ValueError(
            f"unknown multi-turn protocol variant '{variant}'. Available: {available}"
        )
    return resolved


def protocol_metadata(protocol: str, variant: str = DEFAULT_PROTOCOL_VARIANT) -> dict[str, Any]:
    if protocol == "multi_turn":
        variant_config = get_multi_turn_variant(variant)
        template_payload = {
            "variant": variant,
            "mcq_turns": list(variant_config["mcq_turns"]),
            "math_turns": list(variant_config["math_turns"]),
            "mcq_max_tokens": list(variant_config["mcq_max_tokens"]),
            "math_max_tokens": list(variant_config["math_max_tokens"]),
        }
        template_hash = stable_id(json.dumps(template_payload, sort_keys=True))
        return {
            "name": "multi_turn",
            "version": MULTI_TURN_PROTOCOL_VERSION,
            "turn_count": 3,
            "template_hash": template_hash,
            "variant": variant,
            "variant_label": variant_config["label"],
        }
    return {
        "name": "single_turn",
        "version": SINGLE_TURN_PROTOCOL_VERSION,
        "turn_count": 1,
        "template_hash": stable_id("single_turn_prompt_v0"),
    }


def validate_dynamic_plan_contract(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("dynamic plan must be a JSON object")
    if plan.get("mode") != "dynamic_core_specialist":
        raise ValueError("dynamic plan mode must be dynamic_core_specialist")
    budget = plan.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("dynamic plan missing required budget object")
    required_budget_fields = (
        "swappable_expert_budget_bytes",
        "per_expert_bytes",
        "core_budget_bytes",
        "specialist_budget_bytes",
    )
    missing_budget = [field for field in required_budget_fields if field not in budget]
    if missing_budget:
        raise ValueError(
            f"dynamic plan missing required budget fields: {', '.join(missing_budget)}"
        )
    if int(budget.get("swappable_expert_budget_bytes", 0)) <= 0:
        raise ValueError("dynamic plan swappable_expert_budget_bytes must be > 0")
    if int(budget.get("per_expert_bytes", 0)) <= 0:
        raise ValueError("dynamic plan per_expert_bytes must be > 0")
    per_layer = plan.get("perLayer")
    if not isinstance(per_layer, dict) or not per_layer:
        raise ValueError("dynamic plan missing required non-empty perLayer object")
    for layer_key, layer in per_layer.items():
        if not isinstance(layer, dict):
            raise ValueError(f"{layer_key}: perLayer entry must be an object")
        core_experts = layer.get("coreExperts")
        slice_catalog = layer.get("sliceCatalog")
        if not isinstance(core_experts, list):
            raise ValueError(f"{layer_key}: coreExperts must be a list")
        if not isinstance(slice_catalog, list):
            raise ValueError(f"{layer_key}: sliceCatalog must be a list")
    return plan


def load_plan(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    plan_path = Path(path)
    if not plan_path.exists():
        raise ValueError(f"plan path not found: {plan_path}")
    if not plan_path.is_file():
        raise ValueError(f"plan path is not a file: {plan_path}")
    try:
        plan = json.loads(plan_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"plan is not valid JSON: {plan_path}") from exc
    return plan


def build_runtime_identity(
    server_url: str,
    *,
    mode: str,
    plan_path: str | None,
    plan: dict[str, Any] | None,
    readiness_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = urlparse(server_url)
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"server URL must include a host: {server_url}")
    if parsed.port is None:
        raise ValueError(f"server URL must include an explicit port: {server_url}")
    runtime_identity: dict[str, Any] = {
        "mode": mode,
        "server_url": server_url,
        "host": host,
        "port": int(parsed.port),
        "concurrency_mode": "serialized_single_flight" if mode == "dynamic" else "n/a",
    }
    if mode == "dynamic":
        if not plan_path:
            raise ValueError("dynamic mode requires a plan path")
        if not plan:
            raise ValueError("dynamic mode requires a validated plan")
        runtime_identity.update(
            {
                "plan_path": str(Path(plan_path).resolve()),
                "plan_mode": str(plan.get("mode")),
                "plan_budget_bytes": int(
                    plan.get("budget", {}).get("swappable_expert_budget_bytes", 0)
                ),
            }
        )
        if readiness_evidence:
            runtime_identity["readiness_evidence"] = readiness_evidence
    return runtime_identity


def load_runtime_readiness_evidence(server_url: str) -> dict[str, Any] | None:
    parsed = urlparse(server_url)
    host = parsed.hostname or ""
    port = parsed.port
    if not host or port is None:
        return None
    allowlist = {"127.0.0.1", "localhost"}
    env_allowlist = os.environ.get("REAP_RUNTIME_READINESS_HOST_ALLOWLIST", "")
    allowlist.update(item.strip() for item in env_allowlist.split(",") if item.strip())
    if host not in allowlist:
        return None
    readiness_port = int(os.environ.get("REAP_RUNTIME_READINESS_PORT", "8011"))
    if int(port) != readiness_port:
        return None
    identity_path_value = os.environ.get("REAP_RUNTIME_READINESS_IDENTITY_PATH")
    if not identity_path_value:
        return None
    identity_path = Path(identity_path_value).expanduser()
    if not identity_path.exists() or not identity_path.is_file():
        return None
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(identity, dict):
        return None
    return {
        "source": "configured-runtime-identity",
        "identity_path": str(identity_path.resolve()),
        "identity": identity,
    }


def build_plan_identity(plan: dict[str, Any] | None, plan_path: str | None = None) -> dict[str, Any] | None:
    if not plan:
        return None
    identity: dict[str, Any] = {
        "plan_mode": str(plan.get("mode")),
        "plan_budget_bytes": int(
            plan.get("budget", {}).get("swappable_expert_budget_bytes", 0)
        ),
    }
    if plan_path:
        identity["plan_path"] = str(Path(plan_path).resolve())
    return identity


def get_cartridge_ids(
    plan: dict[str, Any] | None, explicit_ids: list[str] | None
) -> list[str]:
    if explicit_ids:
        return explicit_ids
    if not plan:
        return []
    from_summary = plan.get("summary", {}).get("cartridgeIds") or []
    if from_summary:
        return from_summary
    for layer_plan in plan.get("perLayer", {}).values():
        cartridges = layer_plan.get("cartridges", {})
        if cartridges:
            return sorted(cartridges.keys())
    return []


def load_examples(
    spec: BenchmarkSpec, sample_count: int, calibration_count: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    if spec.dataset_config:
        dataset = load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split)
    else:
        dataset = load_dataset(spec.dataset_name, split=spec.split)

    shuffled = dataset.shuffle(seed=seed)
    rows = [
        normalize_row(spec, shuffled[idx])
        for idx in range(sample_count + calibration_count)
    ]
    return {
        "calibration": rows[:calibration_count],
        "test": rows[calibration_count : calibration_count + sample_count],
    }


def normalize_row(spec: BenchmarkSpec, row: dict[str, Any]) -> dict[str, Any]:
    if spec.name == "mmlu":
        return {
            "id": f"mmlu::{row['subject']}::{stable_id(row['question'])}",
            "benchmark": spec.name,
            "question": row["question"],
            "choices": list(row["choices"]),
            "gold": OPTION_LETTERS[int(row["answer"])],
            "subject": row["subject"],
        }
    if spec.name == "arc_challenge":
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        normalized_choices = [
            f"{label}. {text}" for label, text in zip(labels, texts, strict=True)
        ]
        gold = row["answerKey"].strip().upper()
        return {
            "id": f"arc::{row['id']}",
            "benchmark": spec.name,
            "question": row["question"],
            "choices": normalized_choices,
            "gold": gold,
        }
    if spec.name == "hellaswag":
        return {
            "id": f"hellaswag::{row['ind']}",
            "benchmark": spec.name,
            "question": row["ctx"],
            "choices": list(row["endings"]),
            "gold": OPTION_LETTERS[int(row["label"])] ,
        }
    if spec.name == "winogrande":
        return {
            "id": f"winogrande::{stable_id(row['sentence'])}",
            "benchmark": spec.name,
            "question": row["sentence"],
            "choices": [row["option1"], row["option2"]],
            "gold": row["answer"].strip(),
        }
    if spec.name == "gsm8k":
        answer_text = row["answer"]
        match = re.search(r"####\s*([-+]?[$0-9,./]+)", answer_text)
        gold = normalize_numeric_answer(match.group(1) if match else answer_text)
        return {
            "id": f"gsm8k::{stable_id(row['question'])}",
            "benchmark": spec.name,
            "question": row["question"],
            "gold": gold,
            "reference": answer_text,
        }
    raise ValueError(f"Unsupported benchmark: {spec.name}")


def format_prompt(
    spec: BenchmarkSpec,
    row: dict[str, Any],
    *,
    answer_instruction: str | None = None,
) -> str:
    if spec.question_style == "mcq":
        labels = OPTION_LETTERS[: len(row["choices"])]
        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(labels, row["choices"], strict=True)
        )
        return (
            f"{answer_instruction or 'Answer the multiple choice question. Reply with only the option letter.'}\n\n"
            f"Question: {row['question']}\n"
            f"{options}\n\n"
            "Answer:"
        )
    if spec.question_style == "binary":
        return (
            f"{answer_instruction or 'Choose the better option to fill in the blank. Reply with only 1 or 2.'}\n\n"
            f"Sentence: {row['question']}\n"
            f"1. {row['choices'][0]}\n"
            f"2. {row['choices'][1]}\n\n"
            "Answer:"
        )
    if spec.question_style == "math":
        default_instruction = (
            "Solve the math problem. Keep the response concise. End with exactly "
            "'Final answer: <number>'."
        )
        return (
            f"{answer_instruction or default_instruction}\n\n"
            f"Question: {row['question']}\n\n"
            "Answer:"
        )
    raise ValueError(f"Unsupported question style: {spec.question_style}")


def build_multi_turn_turn_specs(
    spec: BenchmarkSpec,
    row: dict[str, Any],
    *,
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
) -> list[dict[str, Any]]:
    variant_config = get_multi_turn_variant(protocol_variant)
    if spec.question_style in {"mcq", "binary"}:
        prompts = [
            format_prompt(
                spec,
                row,
                answer_instruction=str(variant_config["mcq_turns"][0]),
            ),
            *list(variant_config["mcq_turns"][1:]),
        ]
        max_tokens = list(variant_config["mcq_max_tokens"])
        turn_kinds = ["answer", "reason", "recommit"]
        parse_required = [True, False, True]
    elif spec.question_style == "math":
        prompts = [
            format_prompt(
                spec,
                row,
                answer_instruction=str(variant_config["math_turns"][0]),
            ),
            *list(variant_config["math_turns"][1:]),
        ]
        max_tokens = list(variant_config["math_max_tokens"])
        turn_kinds = ["solve", "verify", "recommit"]
        parse_required = [True, False, True]
    else:
        raise ValueError(f"Unsupported question style for multi-turn: {spec.question_style}")

    turns: list[dict[str, Any]] = []
    for turn_index, user_prompt in enumerate(prompts, start=1):
        turns.append(
            {
                "turn_index": turn_index,
                "turn_kind": turn_kinds[turn_index - 1],
                "user_prompt": user_prompt,
                "max_tokens": max_tokens[turn_index - 1],
                "require_parse": parse_required[turn_index - 1],
                "final_turn_for_sample": turn_index == len(prompts),
            }
        )
    return turns


def normalize_numeric_answer(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.strip().rstrip(".")
    return cleaned


def strip_thinking_block(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def parse_prediction(spec: BenchmarkSpec, text: str) -> str | None:
    normalized = text.strip()
    if not normalized:
        return None
    answer_text = strip_thinking_block(normalized)
    if spec.question_style == "mcq":
        match = re.search(
            r"\b([A-E])\b", answer_text.upper() if answer_text else normalized.upper()
        )
        return match.group(1) if match else None
    if spec.question_style == "binary":
        match = re.search(r"\b([12])\b", answer_text if answer_text else normalized)
        return match.group(1) if match else None
    if spec.question_style == "math":
        search_text = answer_text if answer_text else normalized
        final_match = re.search(
            r"Final answer:\s*([-+]?[$0-9,./]+)", search_text, flags=re.IGNORECASE
        )
        if final_match:
            return normalize_numeric_answer(final_match.group(1))
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", search_text)
        if boxed_match:
            return normalize_numeric_answer(boxed_match.group(1))
        generic = re.findall(r"[-+]?[$]?[0-9][0-9,./]*", search_text)
        if generic:
            return normalize_numeric_answer(generic[-1])
        return None
    return None


def parse_reason_anchor(spec: BenchmarkSpec, text: str) -> str | None:
    if spec.question_style == "mcq":
        patterns = [
            r"answer token:\s*([A-E])\b",
            r"implied to be ['\"]?([A-E])['\"]?",
            r"answer(?:\s+was|\s+is|\s*=|:)\s*['\"]?([A-E])['\"]?\b",
            r"previous answer(?:\s+was|:)\s*['\"]?([A-E])['\"]?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    if spec.question_style == "binary":
        patterns = [
            r"answer token:\s*([12])\b",
            r"answer(?:\s+was|\s+is|\s*=|:)\s*['\"]?([12])['\"]?\b",
            r"previous answer(?:\s+was|:)\s*['\"]?([12])['\"]?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    return None


def parse_math_context_answer(text: str) -> str | None:
    final_match = re.findall(
        r"Final answer:\s*([-+]?[$0-9,./]+)", text, flags=re.IGNORECASE
    )
    if final_match:
        return normalize_numeric_answer(final_match[-1])

    strong_patterns = [
        r"(?:verified total|total amount|total|result|answer)\s*(?:=|is|was|:)\s*\$?([-+]?[0-9][0-9,./]*)",
        r"\(\$?([-+]?[0-9][0-9,./]*)\)",
    ]
    candidates: list[str] = []
    for pattern in strong_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = normalize_numeric_answer(match)
            if normalized:
                candidates.append(normalized)
    if candidates:
        return candidates[-1]

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if re.match(r"^\d+\.\s", stripped):
            continue
        generic = re.findall(r"[-+]?[$]?[0-9][0-9,./]*", stripped)
        if generic:
            return normalize_numeric_answer(generic[-1])
    return None


def is_low_confidence_math_recommit(row: dict[str, Any], parsed_answer: str | None) -> bool:
    if not parsed_answer:
        return True
    if parsed_answer not in {"1", "2", "3", "4", "5"}:
        return False
    response = str(row.get("response") or "")
    if re.search(r"Final answer:\s*[-+]?[$0-9,./]+", response, flags=re.IGNORECASE):
        return False
    return bool(response.strip().startswith("Thinking Process"))


def apply_protocol_variant_rescoring(
    results: list[dict[str, Any]],
    spec_map: dict[str, BenchmarkSpec],
    *,
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
) -> list[dict[str, Any]]:
    if not results:
        return []
    variant = get_multi_turn_variant(protocol_variant)
    if protocol_variant == DEFAULT_PROTOCOL_VARIANT:
        return [dict(row) for row in results]

    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        conv_id = str(row.get("conversation_id") or row.get("id"))
        by_conversation.setdefault(conv_id, []).append(dict(row))

    rescored: list[dict[str, Any]] = []
    for rows in by_conversation.values():
        ordered = sorted(rows, key=lambda item: int(item.get("turn_index", 0)))
        if not ordered:
            continue
        spec = spec_map[str(ordered[0]["benchmark"])]
        turn1 = next((row for row in ordered if int(row.get("turn_index", 0)) == 1), None)
        reason_turn = next(
            (
                row
                for row in ordered
                if str(row.get("turn_kind")) in {"reason", "verify"}
            ),
            None,
        )
        final_turn = next(
            (row for row in ordered if bool(row.get("final_turn_for_sample"))),
            ordered[-1],
        )

        for row in ordered:
            turn_kind = str(row.get("turn_kind") or "")
            require_parse = turn_kind in {"answer", "solve", "recommit"}
            parsed_answer = (
                parse_prediction(spec, str(row.get("response") or ""))
                if require_parse
                else None
            )
            parsed_source = "direct"
            if (
                row is final_turn
                and turn_kind == "recommit"
                and spec.question_style in {"mcq", "binary"}
                and not parsed_answer
            ):
                fallback_mode = str(variant.get("recommit_mcq_fallback", "none"))
                fallback_answer = None
                if fallback_mode == "reason_then_turn1" and reason_turn is not None:
                    fallback_answer = parse_reason_anchor(
                        spec, str(reason_turn.get("response") or "")
                    )
                    if fallback_answer:
                        parsed_source = "reason_anchor"
                if fallback_answer is None and fallback_mode in {"turn1", "reason_then_turn1"}:
                    fallback_answer = (
                        parse_prediction(spec, str(turn1.get("response") or ""))
                        if turn1 is not None
                        else None
                    )
                    if fallback_answer:
                        parsed_source = "turn1_backfill"
                if fallback_answer:
                    parsed_answer = fallback_answer
            if (
                row is final_turn
                and turn_kind == "recommit"
                and spec.question_style == "math"
                and str(variant.get("recommit_math_fallback")) == "context_numeric"
            ):
                should_fallback = (not parsed_answer) or (
                    bool(variant.get("math_override_low_confidence"))
                    and is_low_confidence_math_recommit(row, parsed_answer)
                )
                if should_fallback:
                    context_text = "\n".join(
                        str(turn.get("response") or "")
                        for turn in ordered
                        if int(turn.get("turn_index", 0)) <= 2
                    )
                    fallback_math = parse_math_context_answer(context_text)
                    if fallback_math:
                        parsed_answer = fallback_math
                        parsed_source = "context_numeric"

            row["parsed_answer"] = parsed_answer
            row["parsed_answer_source"] = parsed_source
            if require_parse:
                row["correct"] = (
                    parsed_answer == row.get("gold") if row.get("error") is None else False
                )
                row["parse_error"] = (
                    row.get("error") is None and parsed_answer is None
                )
                row["coherent"] = (
                    coherence_pass(spec, str(row.get("response") or ""), parsed_answer)
                    if row.get("error") is None
                    else False
                )
            else:
                row["correct"] = None
                row["parse_error"] = None
                row["coherent"] = (
                    text_coherence_pass(str(row.get("response") or ""))
                    if row.get("error") is None
                    else False
                )
            rescored.append(row)

        if final_turn:
            turn1_parsed = (
                parse_prediction(spec, str(turn1.get("response") or ""))
                if turn1 is not None
                else None
            )
            final_turn["answer_retention"] = bool(
                turn1_parsed is not None
                and final_turn.get("parsed_answer") is not None
                and turn1_parsed == final_turn.get("parsed_answer")
            )
    return rescored


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char in "\n\t")
    return printable / len(text)


def is_repetitive(text: str) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 20:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.2:
        return True
    trigrams: dict[tuple[str, str, str], int] = {}
    for idx in range(len(tokens) - 2):
        trigram = (tokens[idx], tokens[idx + 1], tokens[idx + 2])
        trigrams[trigram] = trigrams.get(trigram, 0) + 1
        if trigrams[trigram] >= 4:
            return True
    return False


def coherence_pass(spec: BenchmarkSpec, text: str, parsed_answer: str | None) -> bool:
    return (
        bool(text.strip())
        and printable_ratio(text) >= 0.95
        and not is_repetitive(text)
        and parsed_answer is not None
    )


def text_coherence_pass(text: str) -> bool:
    return bool(text.strip()) and printable_ratio(text) >= 0.95 and not is_repetitive(text)


def request_completion(
    server_url: str, model: str, prompt: str, max_tokens: int, timeout_s: int
) -> dict[str, Any]:
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    started = time.time()
    response = requests.post(f"{server_url}/v1/completions", json=payload, timeout=timeout_s)
    latency_s = time.time() - started
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0].get("text", "")
    usage = data.get("usage", {}) or {}
    return {
        "text": text,
        "latency_s": latency_s,
        "usage": usage,
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip())
    return str(content or "")


def _chat_message_to_text(message: dict[str, Any]) -> str:
    content_text = _message_content_to_text(message.get("content", ""))
    reasoning_text = _message_content_to_text(message.get("reasoning", ""))
    if content_text and reasoning_text:
        return f"{reasoning_text}\n{content_text}".strip()
    return (content_text or reasoning_text or "").strip()


def serialize_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = _message_content_to_text(message.get("content", ""))
        lines.append(f"{role}: {content}".rstrip())
    lines.append("ASSISTANT:")
    return "\n\n".join(lines).strip()


def request_chat_completion(
    server_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_s: int,
    *,
    fallback_to_prompt: bool = True,
) -> dict[str, Any]:
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    started = time.time()
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        timeout=timeout_s,
    )
    latency_s = time.time() - started
    if response.status_code < 400:
        data = response.json()
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        text = _chat_message_to_text(message)
        usage = data.get("usage", {}) or {}
        return {
            "text": text,
            "latency_s": latency_s,
            "usage": usage,
            "transport": "chat_completions",
        }
    if not fallback_to_prompt or response.status_code not in {400, 404, 405, 422}:
        response.raise_for_status()
    completion = request_completion(
        server_url,
        model,
        serialize_messages_to_prompt(messages),
        max_tokens,
        timeout_s,
    )
    completion["transport"] = "chat_prompt_fallback"
    return completion


def swap_cartridge(server_url: str, cartridge_id: str) -> dict[str, Any]:
    import requests

    started = time.time()
    response = requests.post(f"{server_url}/swap_cartridge/{cartridge_id}", timeout=600)
    elapsed = time.time() - started
    response.raise_for_status()
    payload = response.json()
    payload["endpoint_time_s"] = elapsed
    return payload


def swap_active_set(server_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    import requests

    started = time.time()
    response = requests.post(f"{server_url}/swap_active_set", json=payload, timeout=600)
    elapsed = time.time() - started
    response.raise_for_status()
    body = response.json()
    body["endpoint_time_s"] = elapsed
    return body


def fetch_router_misses(server_url: str, request_id: str, *, reset: bool = True) -> dict[str, Any] | None:
    import requests

    try:
        response = requests.get(
            f"{server_url}/router_misses/{request_id}",
            params={"reset": str(reset).lower()},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def evaluate_samples(
    rows: list[dict[str, Any]],
    spec_map: dict[str, BenchmarkSpec],
    server_url: str,
    model: str,
    mode: str,
    plan: dict[str, Any] | None,
    benchmark_to_cartridge: dict[str, str] | None,
    interleaved: bool,
    seed: int,
    request_timeout_s: int,
) -> list[dict[str, Any]]:
    ordered = list(rows)
    if interleaved:
        random.Random(seed).shuffle(ordered)

    results = []
    current_cartridge = None
    current_active_signature = None
    for row in ordered:
        spec = spec_map[row["benchmark"]]
        prompt = format_prompt(spec, row)
        request_id = stable_id(mode, row["id"], str(seed))
        target_cartridge = (
            benchmark_to_cartridge[row["benchmark"]] if benchmark_to_cartridge else None
        )
        selected_slice_ids = None
        candidate_slice_ids = None
        active_expert_bytes = 0
        active_expert_count = 0
        swap_payload = None
        swap_request_id = None
        swap_active_set_signature = None
        swap_plan_identity = None
        swap_error = None
        swap_time_s = 0.0
        if mode == "multiplex":
            if target_cartridge and target_cartridge != current_cartridge:
                try:
                    swap_payload = swap_cartridge(server_url, target_cartridge)
                    swap_time_s = float(swap_payload.get("endpoint_time_s", 0.0))
                    current_cartridge = target_cartridge
                except Exception as exc:
                    swap_error = str(exc)
        elif mode == "dynamic":
            if not plan:
                raise RuntimeError("Dynamic mode requires a dynamic plan")
            try:
                active_payload = build_active_set_payload(
                    plan,
                    prompt,
                    request_id=request_id,
                    benchmark=row["benchmark"],
                    phase="prefill",
                )
                selected_slice_ids = active_payload.get("selected_slice_ids")
                candidate_slice_ids = active_payload.get("candidate_slice_ids")
                active_expert_bytes = int(active_payload.get("budget_bytes", 0))
                active_expert_count = sum(len(experts) for experts in active_payload.get("active_set", {}).values())
                active_set_signature = active_payload.get("active_set_signature")
                swap_payload = swap_active_set(server_url, active_payload)
                if swap_payload.get("status") != "success":
                    raise RuntimeError(swap_payload.get("error", "unknown dynamic swap error"))
                current_active_signature = active_set_signature
                swap_request_id = swap_payload.get("request_id")
                swap_active_set_signature = swap_payload.get("active_set_signature") or active_set_signature
                swap_plan_identity = swap_payload.get("plan_identity")
                swap_time_s = float(swap_payload.get("endpoint_time_s", 0.0))
            except Exception as exc:
                swap_error = str(exc)

        started = time.time()
        output_text = ""
        parsed_answer = None
        request_latency_s = 0.0
        usage = {}
        request_error = None
        try:
            if swap_error is not None:
                raise RuntimeError(f"swap failed: {swap_error}")
            completion = request_completion(
                server_url,
                model,
                prompt,
                spec.max_tokens,
                timeout_s=request_timeout_s,
            )
            output_text = completion["text"]
            request_latency_s = float(completion["latency_s"])
            usage = completion.get("usage", {}) or {}
            parsed_answer = parse_prediction(spec, output_text)
        except Exception as exc:
            request_error = str(exc)

        router_misses = fetch_router_misses(server_url, request_id) if mode == "dynamic" else None
        router_miss_summary = summarize_router_misses(router_misses) if router_misses else None
        total_latency_s = time.time() - started + swap_time_s
        correct = parsed_answer == row["gold"] if request_error is None else False
        coherent = (
            coherence_pass(spec, output_text, parsed_answer)
            if request_error is None
            else False
        )
        results.append(
            {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "question": row["question"],
                "gold": row["gold"],
                "request_id": request_id,
                "selected_cartridge": target_cartridge,
                "selected_slice_ids": selected_slice_ids,
                "candidate_slice_ids": candidate_slice_ids,
                "active_set_signature": (
                    (swap_active_set_signature or active_set_signature) if mode == "dynamic" else None
                ),
                "active_expert_bytes": active_expert_bytes,
                "active_expert_count": active_expert_count,
                "swap_request_id": swap_request_id,
                "swap_plan_identity": swap_plan_identity,
                "swap_time_s": round(swap_time_s, 6),
                "swap_internal_s": round(float(swap_payload.get("swap_time_s", 0.0)), 6)
                if swap_payload
                else 0.0,
                "swap_reused_active_set": bool(swap_payload and swap_payload.get("no_op_reuse")),
                "request_latency_s": round(request_latency_s, 6),
                "total_latency_s": round(total_latency_s, 6),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "response": output_text,
                "parsed_answer": parsed_answer,
                "correct": correct,
                "coherent": coherent,
                "parse_error": request_error is None and parsed_answer is None,
                "error": request_error,
                "router_misses": router_misses,
                "router_miss_summary": router_miss_summary,
                "refresh_suggested": bool(router_miss_summary and router_miss_summary.get("inactive_ratio", 0.0) >= 0.18),
            }
        )
    return results


def evaluate_multi_turn_samples(
    rows: list[dict[str, Any]],
    spec_map: dict[str, BenchmarkSpec],
    server_url: str,
    model: str,
    mode: str,
    plan: dict[str, Any] | None,
    benchmark_to_cartridge: dict[str, str] | None,
    interleaved: bool,
    seed: int,
    request_timeout_s: int,
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
) -> list[dict[str, Any]]:
    ordered = list(rows)
    if interleaved:
        random.Random(seed).shuffle(ordered)

    results: list[dict[str, Any]] = []
    current_cartridge = None
    for row in ordered:
        spec = spec_map[row["benchmark"]]
        conversation_id = stable_id("conversation", row["id"], str(seed))
        target_cartridge = (
            benchmark_to_cartridge[row["benchmark"]] if benchmark_to_cartridge else None
        )
        history: list[dict[str, str]] = []
        turn_specs = build_multi_turn_turn_specs(
            spec, row, protocol_variant=protocol_variant
        )
        turn_records: list[dict[str, Any]] = []
        turn1_parsed: str | None = None

        for turn in turn_specs:
            turn_index = int(turn["turn_index"])
            turn_kind = str(turn["turn_kind"])
            user_prompt = str(turn["user_prompt"])
            max_tokens = int(turn["max_tokens"])
            require_parse = bool(turn["require_parse"])
            request_id = stable_id(
                mode,
                row["id"],
                conversation_id,
                str(turn_index),
                str(seed),
            )
            messages = history + [{"role": "user", "content": user_prompt}]
            selected_slice_ids = None
            candidate_slice_ids = None
            active_expert_bytes = 0
            active_expert_count = 0
            swap_payload = None
            swap_request_id = None
            swap_active_set_signature = None
            swap_plan_identity = None
            swap_error = None
            swap_time_s = 0.0

            if mode == "multiplex":
                if target_cartridge and target_cartridge != current_cartridge:
                    try:
                        swap_payload = swap_cartridge(server_url, target_cartridge)
                        swap_time_s = float(swap_payload.get("endpoint_time_s", 0.0))
                        current_cartridge = target_cartridge
                    except Exception as exc:
                        swap_error = str(exc)
            elif mode == "dynamic":
                if not plan:
                    raise RuntimeError("Dynamic mode requires a dynamic plan")
                try:
                    active_payload = build_active_set_payload(
                        plan,
                        user_prompt,
                        request_id=request_id,
                        benchmark=row["benchmark"],
                        phase="prefill",
                        conversation_id=conversation_id,
                        turn_index=turn_index,
                        messages=messages,
                    )
                    selected_slice_ids = active_payload.get("selected_slice_ids")
                    candidate_slice_ids = active_payload.get("candidate_slice_ids")
                    active_expert_bytes = int(active_payload.get("budget_bytes", 0))
                    active_expert_count = sum(
                        len(experts)
                        for experts in active_payload.get("active_set", {}).values()
                    )
                    active_set_signature = active_payload.get("active_set_signature")
                    swap_payload = swap_active_set(server_url, active_payload)
                    if swap_payload.get("status") != "success":
                        raise RuntimeError(
                            swap_payload.get("error", "unknown dynamic swap error")
                        )
                    swap_request_id = swap_payload.get("request_id")
                    swap_active_set_signature = (
                        swap_payload.get("active_set_signature") or active_set_signature
                    )
                    swap_plan_identity = swap_payload.get("plan_identity")
                    swap_time_s = float(swap_payload.get("endpoint_time_s", 0.0))
                except Exception as exc:
                    swap_error = str(exc)

            started = time.time()
            output_text = ""
            parsed_answer = None
            request_latency_s = 0.0
            usage = {}
            request_error = None
            transport = "n/a"
            try:
                if swap_error is not None:
                    raise RuntimeError(f"swap failed: {swap_error}")
                completion = request_chat_completion(
                    server_url,
                    model,
                    messages,
                    max_tokens,
                    timeout_s=request_timeout_s,
                )
                output_text = completion["text"]
                request_latency_s = float(completion["latency_s"])
                usage = completion.get("usage", {}) or {}
                transport = str(completion.get("transport") or "chat_completions")
                if require_parse:
                    parsed_answer = parse_prediction(spec, output_text)
            except Exception as exc:
                request_error = str(exc)

            router_misses = (
                fetch_router_misses(server_url, request_id) if mode == "dynamic" else None
            )
            router_miss_summary = (
                summarize_router_misses(router_misses) if router_misses else None
            )
            total_latency_s = time.time() - started + swap_time_s
            correct = (
                parsed_answer == row["gold"]
                if require_parse and request_error is None
                else None
            )
            coherent = (
                coherence_pass(spec, output_text, parsed_answer)
                if request_error is None and require_parse
                else (text_coherence_pass(output_text) if request_error is None else False)
            )
            parse_error = (
                request_error is None and parsed_answer is None
                if require_parse
                else None
            )
            turn_record = {
                "id": row["id"],
                "sample_id": row["id"],
                "benchmark": row["benchmark"],
                "question": row["question"],
                "gold": row["gold"],
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "turn_count": len(turn_specs),
                "turn_kind": turn_kind,
                "final_turn_for_sample": bool(turn["final_turn_for_sample"]),
                "messages": messages,
                "message_count": len(messages),
                "request_id": request_id,
                "selected_cartridge": target_cartridge,
                "selected_slice_ids": selected_slice_ids,
                "candidate_slice_ids": candidate_slice_ids,
                "active_set_signature": (
                    swap_active_set_signature if mode == "dynamic" else None
                ),
                "active_expert_bytes": active_expert_bytes,
                "active_expert_count": active_expert_count,
                "swap_request_id": swap_request_id,
                "swap_plan_identity": swap_plan_identity,
                "swap_time_s": round(swap_time_s, 6),
                "swap_internal_s": round(float(swap_payload.get("swap_time_s", 0.0)), 6)
                if swap_payload
                else 0.0,
                "swap_reused_active_set": bool(
                    swap_payload and swap_payload.get("no_op_reuse")
                ),
                "request_latency_s": round(request_latency_s, 6),
                "total_latency_s": round(total_latency_s, 6),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "response": output_text,
                "parsed_answer": parsed_answer,
                "correct": correct,
                "coherent": coherent,
                "parse_error": parse_error,
                "error": request_error,
                "router_misses": router_misses,
                "router_miss_summary": router_miss_summary,
                "refresh_suggested": bool(
                    router_miss_summary
                    and router_miss_summary.get("inactive_ratio", 0.0) >= 0.18
                ),
                "transport": transport,
            }
            turn_records.append(turn_record)
            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": output_text})
            if turn_index == 1:
                turn1_parsed = parsed_answer

        final_turn = turn_records[-1] if turn_records else None
        if final_turn is not None:
            final_turn["answer_retention"] = (
                bool(turn1_parsed is not None and final_turn.get("parsed_answer") == turn1_parsed)
                if final_turn.get("parsed_answer") is not None
                else False
            )
        results.extend(turn_records)
    return results


def summarize_results(
    results: list[dict[str, Any]],
    *,
    protocol: str = "single_turn",
) -> dict[str, Any]:
    if not results:
        return {}

    def build_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        errors = sum(1 for row in rows if row.get("error"))
        parse_applicable_rows = [
            row for row in rows if row.get("parse_error") is not None
        ]
        parse_errors = sum(1 for row in parse_applicable_rows if row.get("parse_error"))
        accuracy_rows = [row for row in rows if row.get("correct") is not None]
        coherence_rows = [row for row in rows if row.get("coherent") is not None]
        correct = sum(1 for row in accuracy_rows if row.get("correct"))
        coherent = sum(1 for row in coherence_rows if row.get("coherent"))
        request_latencies = [row["request_latency_s"] for row in rows if not row["error"]]
        total_latencies = [row["total_latency_s"] for row in rows]
        swap_latencies = [row["swap_time_s"] for row in rows if row["swap_time_s"] > 0]
        selected_cartridges = [row["selected_cartridge"] for row in rows if row["selected_cartridge"]]
        transition_count = sum(
            1
            for previous, current in zip(
                selected_cartridges, selected_cartridges[1:], strict=False
            )
            if previous != current
        )
        cartridge_usage = {
            cartridge_id: selected_cartridges.count(cartridge_id)
            for cartridge_id in sorted(set(selected_cartridges))
        }
        active_expert_bytes = [row.get("active_expert_bytes", 0) for row in rows if row.get("active_expert_bytes")]
        active_expert_counts = [row.get("active_expert_count", 0) for row in rows if row.get("active_expert_count")]
        refresh_suggested_count = sum(1 for row in rows if row.get("refresh_suggested"))
        router_miss_inactive_ratios = [
            row.get("router_miss_summary", {}).get("inactive_ratio")
            for row in rows
            if row.get("router_miss_summary")
        ]
        selected_slice_total = 0
        unique_slices: set[str] = set()
        for row in rows:
            for slice_ids in (row.get("selected_slice_ids") or {}).values():
                selected_slice_total += len(slice_ids)
                unique_slices.update(slice_ids)
        return {
            "total": total,
            "accuracy_applicable": len(accuracy_rows),
            "correct": correct,
            "accuracy": round(correct / len(accuracy_rows), 6) if accuracy_rows else 0.0,
            "errors": errors,
            "error_rate": round(errors / total, 6),
            "parse_applicable": len(parse_applicable_rows),
            "parse_errors": parse_errors,
            "parse_error_rate": round(parse_errors / len(parse_applicable_rows), 6)
            if parse_applicable_rows
            else 0.0,
            "coherence_applicable": len(coherence_rows),
            "coherent": coherent,
            "coherence_rate": round(coherent / len(coherence_rows), 6)
            if coherence_rows
            else 0.0,
            "bench_time_s": round(sum(total_latencies), 6),
            "avg_request_time_s": round(statistics.fmean(request_latencies), 6)
            if request_latencies
            else 0.0,
            "avg_sample_time_s": round(statistics.fmean(total_latencies), 6),
            "avg_swap_time_s": round(statistics.fmean(swap_latencies), 6)
            if swap_latencies
            else 0.0,
            "swap_count": len(swap_latencies),
            "cartridge_transition_count": transition_count,
            "cartridge_transition_rate": round(
                transition_count / max(len(selected_cartridges) - 1, 1), 6
            )
            if len(selected_cartridges) > 1
            else 0.0,
            "unique_cartridges_used": len(cartridge_usage),
            "cartridge_usage": cartridge_usage,
            "p50_sample_time_s": round(statistics.median(total_latencies), 6),
            "p95_sample_time_s": round(
                sorted(total_latencies)[
                    min(len(total_latencies) - 1, int(len(total_latencies) * 0.95))
                ],
                6,
            ),
            "avg_active_expert_bytes": round(statistics.fmean(active_expert_bytes), 6)
            if active_expert_bytes
            else 0.0,
            "avg_active_expert_count": round(statistics.fmean(active_expert_counts), 6)
            if active_expert_counts
            else 0.0,
            "refresh_suggested_count": refresh_suggested_count,
            "refresh_suggested_rate": round(refresh_suggested_count / total, 6),
            "avg_router_miss_inactive_ratio": round(statistics.fmean(router_miss_inactive_ratios), 6)
            if router_miss_inactive_ratios
            else 0.0,
            "unique_slices_used": len(unique_slices),
            "avg_selected_slices_per_sample": round(selected_slice_total / total, 6),
        }

    if protocol != "multi_turn":
        overall = build_bucket(results)
        by_benchmark = {}
        for benchmark in sorted({row["benchmark"] for row in results}):
            by_benchmark[benchmark] = build_bucket(
                [row for row in results if row["benchmark"] == benchmark]
            )
        return {"overall": overall, "by_benchmark": by_benchmark}

    final_rows = [row for row in results if row.get("final_turn_for_sample")]
    overall = build_bucket(final_rows)
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in final_rows}):
        by_benchmark[benchmark] = build_bucket(
            [row for row in final_rows if row["benchmark"] == benchmark]
        )

    by_turn: dict[str, Any] = {}
    for turn_index in sorted({int(row.get("turn_index", 0)) for row in results if row.get("turn_index")}):
        turn_key = f"turn_{turn_index}"
        turn_rows = [row for row in results if row.get("turn_index") == turn_index]
        by_turn[turn_key] = build_bucket(turn_rows)

    by_benchmark_turn: dict[str, Any] = {}
    for benchmark in sorted({row["benchmark"] for row in results}):
        benchmark_rows = [row for row in results if row["benchmark"] == benchmark]
        by_benchmark_turn[benchmark] = {}
        for turn_index in sorted({int(row.get("turn_index", 0)) for row in benchmark_rows if row.get("turn_index")}):
            turn_key = f"turn_{turn_index}"
            by_benchmark_turn[benchmark][turn_key] = build_bucket(
                [row for row in benchmark_rows if row.get("turn_index") == turn_index]
            )

    conversations: dict[str, dict[str, Any]] = {}
    for row in results:
        conv_id = str(row.get("conversation_id") or row["id"])
        conversation = conversations.setdefault(
            conv_id,
            {
                "conversation_id": conv_id,
                "sample_id": row.get("sample_id") or row["id"],
                "benchmark": row["benchmark"],
                "rows": [],
            },
        )
        conversation["rows"].append(row)

    conversation_rows: list[dict[str, Any]] = []
    for conversation in conversations.values():
        rows_sorted = sorted(
            conversation["rows"], key=lambda item: int(item.get("turn_index", 0))
        )
        first = rows_sorted[0] if rows_sorted else {}
        final = rows_sorted[-1] if rows_sorted else {}
        reason = next(
            (row for row in rows_sorted if str(row.get("turn_kind")) in {"reason", "verify"}),
            {},
        )
        answer_retention = bool(
            first.get("parsed_answer") is not None
            and final.get("parsed_answer") is not None
            and first.get("parsed_answer") == final.get("parsed_answer")
        )
        conversation_success = bool(
            rows_sorted
            and all(not row.get("error") for row in rows_sorted)
            and bool(reason.get("coherent"))
            and bool(final.get("correct"))
        )
        conversation_rows.append(
            {
                "conversation_id": conversation["conversation_id"],
                "sample_id": conversation["sample_id"],
                "benchmark": conversation["benchmark"],
                "turn1_correct": bool(first.get("correct")),
                "turn3_correct": bool(final.get("correct")),
                "answer_retention": answer_retention,
                "turn2_coherent": bool(reason.get("coherent")),
                "conversation_success": conversation_success,
                "errors": sum(1 for row in rows_sorted if row.get("error")),
                "parse_errors": sum(
                    1 for row in rows_sorted if row.get("parse_error") is True
                ),
                "total_latency_s": round(
                    sum(float(row.get("total_latency_s", 0.0) or 0.0) for row in rows_sorted),
                    6,
                ),
            }
        )

    def build_conversation_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        latencies = [float(row.get("total_latency_s", 0.0) or 0.0) for row in rows]
        turn1_correct = sum(1 for row in rows if row.get("turn1_correct"))
        turn3_correct = sum(1 for row in rows if row.get("turn3_correct"))
        answer_retention = sum(1 for row in rows if row.get("answer_retention"))
        turn2_coherent = sum(1 for row in rows if row.get("turn2_coherent"))
        conversation_success = sum(1 for row in rows if row.get("conversation_success"))
        return {
            "total": total,
            "turn1_accuracy": round(turn1_correct / total, 6) if total else 0.0,
            "turn3_accuracy": round(turn3_correct / total, 6) if total else 0.0,
            "accuracy_drop_turn1_to_turn3": round(
                max(0.0, (turn1_correct - turn3_correct) / total),
                6,
            )
            if total
            else 0.0,
            "answer_retention_rate": round(answer_retention / total, 6) if total else 0.0,
            "turn2_coherence_rate": round(turn2_coherent / total, 6) if total else 0.0,
            "conversation_success_rate": round(conversation_success / total, 6)
            if total
            else 0.0,
            "error_rate": round(sum(int(row.get("errors", 0)) > 0 for row in rows) / total, 6)
            if total
            else 0.0,
            "parse_error_rate": round(sum(int(row.get("parse_errors", 0)) > 0 for row in rows) / total, 6)
            if total
            else 0.0,
            "avg_conversation_time_s": round(statistics.fmean(latencies), 6)
            if latencies
            else 0.0,
            "p95_conversation_time_s": round(
                sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))],
                6,
            )
            if latencies
            else 0.0,
        }

    conversation_summary = {
        "overall": build_conversation_bucket(conversation_rows),
        "by_benchmark": {
            benchmark: build_conversation_bucket(
                [row for row in conversation_rows if row["benchmark"] == benchmark]
            )
            for benchmark in sorted({row["benchmark"] for row in conversation_rows})
        },
    }
    return {
        "overall": overall,
        "by_benchmark": by_benchmark,
        "turn_overall": by_turn,
        "by_benchmark_turn": by_benchmark_turn,
        "conversation_overall": conversation_summary["overall"],
        "conversation_by_benchmark": conversation_summary["by_benchmark"],
    }


def select_cartridges(
    calibration_sets: dict[str, list[dict[str, Any]]],
    spec_map: dict[str, BenchmarkSpec],
    server_url: str,
    model: str,
    cartridge_ids: list[str],
    *,
    protocol: str,
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
) -> dict[str, Any]:
    selection = {}
    for benchmark, rows in calibration_sets.items():
        trials = []
        for cartridge_id in cartridge_ids:
            evaluator = (
                evaluate_multi_turn_samples if protocol == "multi_turn" else evaluate_samples
            )
            eval_kwargs = {
                "interleaved": False,
                "seed": 0,
                "request_timeout_s": 120,
            }
            if protocol == "multi_turn":
                eval_kwargs["protocol_variant"] = protocol_variant
            evaluated = evaluator(
                rows,
                spec_map,
                server_url,
                model,
                "multiplex",
                None,
                {benchmark: cartridge_id},
                **eval_kwargs,
            )
            summary = summarize_results(evaluated, protocol=protocol)["overall"]
            trials.append({"cartridge_id": cartridge_id, **summary})

        trials.sort(
            key=lambda row: (
                row["accuracy"],
                row["coherence_rate"],
                -row["avg_sample_time_s"],
            ),
            reverse=True,
        )
        selection[benchmark] = {
            "selected_cartridge": trials[0]["cartridge_id"],
            "trials": trials,
        }
    return selection


def build_markdown_report(payload: dict[str, Any]) -> str:
    overall = payload["summary"]["overall"]
    protocol = payload.get("protocol") or {}
    lines = [
        f"# {payload['mode'].title()} evaluation report",
        "",
        f"- model: `{payload['model']}`",
        f"- protocol: `{protocol.get('name', 'single_turn')}` / `{protocol.get('version', 'n/a')}`",
        f"- turns per sample: {protocol.get('turn_count', 1)}",
        f"- total samples: {overall['total']}",
        f"- overall accuracy: {overall['accuracy']:.2%}",
        f"- overall coherence: {overall['coherence_rate']:.2%}",
        f"- overall error rate: {overall['error_rate']:.2%}",
        f"- average request time: {overall['avg_request_time_s']:.3f}s",
        f"- average sample time: {overall['avg_sample_time_s']:.3f}s",
        f"- p95 sample time: {overall['p95_sample_time_s']:.3f}s",
        f"- swap count: {overall['swap_count']}",
        f"- cartridge transition rate: {overall['cartridge_transition_rate']:.2%}",
        f"- avg active expert bytes: {overall.get('avg_active_expert_bytes', 0)}",
        f"- avg active expert count: {overall.get('avg_active_expert_count', 0)}",
        f"- refresh suggested rate: {overall.get('refresh_suggested_rate', 0):.2%}",
        f"- avg router inactive ratio: {overall.get('avg_router_miss_inactive_ratio', 0):.2%}",
        f"- unique slices used: {overall.get('unique_slices_used', 0)}",
        "",
    ]
    runtime_identity = payload.get("runtime_identity") or {}
    plan_identity = payload.get("plan_identity") or {}
    readiness = runtime_identity.get("readiness_evidence") or {}
    if runtime_identity:
        lines.extend(
            [
                "## Runtime identity",
                "",
                f"- server URL: `{runtime_identity.get('server_url', 'n/a')}`",
                f"- host: `{runtime_identity.get('host', 'n/a')}`",
                f"- port: `{runtime_identity.get('port', 'n/a')}`",
                f"- concurrency mode: `{runtime_identity.get('concurrency_mode', 'n/a')}`",
                f"- plan path: `{runtime_identity.get('plan_path', 'n/a')}`",
                f"- plan mode: `{runtime_identity.get('plan_mode', 'n/a')}`",
                "",
            ]
        )
    if plan_identity:
        lines.extend(
            [
                "## Plan identity",
                "",
                f"- plan path: `{plan_identity.get('plan_path', 'n/a')}`",
                f"- plan mode: `{plan_identity.get('plan_mode', 'n/a')}`",
                f"- swappable budget bytes: `{plan_identity.get('plan_budget_bytes', 'n/a')}`",
                "",
            ]
        )
    if readiness:
        readiness_identity = readiness.get("identity") or {}
        lines.extend(
            [
                "## Runtime readiness evidence",
                "",
                f"- source: `{readiness.get('source', 'n/a')}`",
                f"- identity path: `{readiness.get('identity_path', 'n/a')}`",
                f"- readiness service: `{readiness_identity.get('service', 'n/a')}`",
                f"- readiness host: `{readiness_identity.get('host', 'n/a')}`",
                f"- readiness port: `{readiness_identity.get('port', 'n/a')}`",
                f"- readiness plan file: `{readiness_identity.get('plan_file', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Request-level dynamic evidence",
            "",
        ]
    )
    dynamic_rows = [row for row in payload.get("results", []) if row.get("request_id")]
    if dynamic_rows:
        for row in dynamic_rows[:5]:
            router_misses = row.get("router_misses") or {}
            lines.extend(
                [
                    f"- `{row.get('id', 'unknown')}`: request `{row.get('request_id', 'n/a')}`, swap request `{row.get('swap_request_id', 'n/a')}`, active set `{row.get('active_set_signature', 'n/a')}`, plan `{(row.get('swap_plan_identity') or {}).get('plan_path', 'n/a')}`, router miss request `{router_misses.get('request_id', 'n/a')}`",
                ]
            )
    else:
        lines.append("- no request-level dynamic evidence recorded")
    lines.extend(
        [
            "",
            "## Benchmarks",
            "",
        ]
    )
    for benchmark, summary in payload["summary"]["by_benchmark"].items():
        lines.extend(
            [
                f"### {benchmark}",
                f"- accuracy: {summary['accuracy']:.2%}",
                f"- coherence: {summary['coherence_rate']:.2%}",
                f"- errors: {summary['errors']} ({summary['error_rate']:.2%})",
                f"- parse errors: {summary['parse_errors']} ({summary['parse_error_rate']:.2%})",
                f"- benchmark time: {summary['bench_time_s']:.3f}s",
                f"- avg request time: {summary['avg_request_time_s']:.3f}s",
                f"- avg sample time: {summary['avg_sample_time_s']:.3f}s",
                f"- p95 sample time: {summary['p95_sample_time_s']:.3f}s",
                f"- avg swap time: {summary['avg_swap_time_s']:.3f}s",
                f"- avg active expert bytes: {summary.get('avg_active_expert_bytes', 0)}",
                f"- refresh suggested rate: {summary.get('refresh_suggested_rate', 0):.2%}",
                "",
            ]
        )
    if payload.get("summary", {}).get("conversation_overall"):
        conversation_summary = payload["summary"]["conversation_overall"]
        lines.extend(
            [
                "## Multi-turn conversation summary",
                "",
                f"- turn1 accuracy: {conversation_summary.get('turn1_accuracy', 0):.2%}",
                f"- turn3 accuracy: {conversation_summary.get('turn3_accuracy', 0):.2%}",
                f"- answer retention: {conversation_summary.get('answer_retention_rate', 0):.2%}",
                f"- turn2 coherence: {conversation_summary.get('turn2_coherence_rate', 0):.2%}",
                f"- conversation success: {conversation_summary.get('conversation_success_rate', 0):.2%}",
                "",
            ]
        )
    if payload.get("comparison"):
        lines.extend(["## Retention", ""])
        comparison = payload["comparison"]
        lines.append(
            f"- retained metrics status: `{comparison.get('retained_metrics_status', 'ok')}`"
        )
        mismatch_reasons = comparison.get("baseline_match_reasons") or []
        if mismatch_reasons:
            lines.append(
                f"- baseline mismatch reasons: {', '.join(str(reason) for reason in mismatch_reasons)}"
            )
        lines.append(
            f"- overall accuracy retained: {comparison['overall']['accuracy_retained_pct']:.2f}%"
        )
        lines.append(
            f"- overall coherence retained: {comparison['overall']['coherence_retained_pct']:.2f}%"
        )
        lines.append(
            f"- overall quality loss: {comparison['overall']['quality_loss_pct']:.2f}%"
        )
        lines.append(
            f"- worst benchmark accuracy drop: {comparison['overall']['worst_benchmark_accuracy_drop_abs']:.4f}"
        )
        lines.append("")
        for benchmark, row in comparison["by_benchmark"].items():
            lines.append(
                f"- {benchmark}: accuracy retained {row['accuracy_retained_pct']:.2f}%, coherence retained {row['coherence_retained_pct']:.2f}%, accuracy drop {row['accuracy_drop_abs']:.4f}"
            )
    return "\n".join(lines) + "\n"


def validate_baseline_match(
    payload: dict[str, Any], baseline_payload: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    payload_protocol = payload.get("protocol") or protocol_metadata("single_turn")
    baseline_protocol = baseline_payload.get("protocol") or protocol_metadata("single_turn")
    for field in ("name", "version", "template_hash", "turn_count"):
        if payload_protocol.get(field) != baseline_protocol.get(field):
            reasons.append(f"protocol.{field}")
    if payload.get("sample_count_per_benchmark") != baseline_payload.get(
        "sample_count_per_benchmark"
    ):
        reasons.append("sample_count_per_benchmark")
    if payload.get("calibration_count_per_benchmark") != baseline_payload.get(
        "calibration_count_per_benchmark"
    ):
        reasons.append("calibration_count_per_benchmark")
    if payload.get("seed") != baseline_payload.get("seed"):
        reasons.append("seed")
    payload_signatures = sorted(
        (
            str(row.get("sample_id") or row.get("id")),
            str(row.get("benchmark")),
            int(row.get("turn_index") or 0),
            bool(row.get("final_turn_for_sample", True)),
        )
        for row in payload.get("results", [])
    )
    baseline_signatures = sorted(
        (
            str(row.get("sample_id") or row.get("id")),
            str(row.get("benchmark")),
            int(row.get("turn_index") or 0),
            bool(row.get("final_turn_for_sample", True)),
        )
        for row in baseline_payload.get("results", [])
    )
    if payload_signatures != baseline_signatures:
        reasons.append("result_signatures")
    return reasons


def compare_to_baseline(
    payload: dict[str, Any], baseline_payload: dict[str, Any]
) -> dict[str, Any]:
    mismatch_reasons = validate_baseline_match(payload, baseline_payload)
    comparison = {
        "overall": {},
        "by_benchmark": {},
        "retained_metrics_status": "ok" if not mismatch_reasons else "invalid_unmatched_baseline",
        "baseline_match_reasons": mismatch_reasons,
    }
    base_overall = baseline_payload["summary"]["overall"]
    mux_overall = payload["summary"]["overall"]
    comparison["overall"] = {
        "baseline_accuracy": base_overall["accuracy"],
        "multiplex_accuracy": mux_overall["accuracy"],
        "accuracy_retained_pct": round(
            100 * mux_overall["accuracy"] / base_overall["accuracy"]
        )
        if base_overall["accuracy"]
        else 0.0,
        "quality_loss_pct": round(
            100 * max(0.0, base_overall["accuracy"] - mux_overall["accuracy"]), 4
        ),
        "accuracy_drop_abs": round(
            max(0.0, base_overall["accuracy"] - mux_overall["accuracy"]), 6
        ),
        "baseline_coherence": base_overall["coherence_rate"],
        "multiplex_coherence": mux_overall["coherence_rate"],
        "coherence_retained_pct": round(
            100 * mux_overall["coherence_rate"] / base_overall["coherence_rate"]
        )
        if base_overall["coherence_rate"]
        else 0.0,
    }
    worst_drop = 0.0
    for benchmark, mux_row in payload["summary"]["by_benchmark"].items():
        base_row = baseline_payload["summary"]["by_benchmark"][benchmark]
        accuracy_drop_abs = round(max(0.0, base_row["accuracy"] - mux_row["accuracy"]), 6)
        worst_drop = max(worst_drop, accuracy_drop_abs)
        comparison["by_benchmark"][benchmark] = {
            "baseline_accuracy": base_row["accuracy"],
            "multiplex_accuracy": mux_row["accuracy"],
            "accuracy_retained_pct": round(
                100 * mux_row["accuracy"] / base_row["accuracy"]
            )
            if base_row["accuracy"]
            else 0.0,
            "accuracy_drop_abs": accuracy_drop_abs,
            "baseline_coherence": base_row["coherence_rate"],
            "multiplex_coherence": mux_row["coherence_rate"],
            "coherence_retained_pct": round(
                100 * mux_row["coherence_rate"] / base_row["coherence_rate"]
            )
            if base_row["coherence_rate"]
            else 0.0,
            "baseline_avg_sample_time_s": base_row["avg_sample_time_s"],
            "multiplex_avg_sample_time_s": mux_row["avg_sample_time_s"],
        }
    comparison["overall"]["worst_benchmark_accuracy_drop_abs"] = round(worst_drop, 6)
    if mismatch_reasons:
        comparison["overall"]["accuracy_retained_pct"] = 0.0
        comparison["overall"]["coherence_retained_pct"] = 0.0
        comparison["overall"]["quality_loss_pct"] = 100.0
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline, static multiplex, or dynamic active-set quality/latency on 5 benchmarks."
    )
    parser.add_argument("--mode", choices=["baseline", "multiplex", "dynamic"])
    parser.add_argument("--server-url")
    parser.add_argument("--model")
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--calibration-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--protocol",
        choices=["single_turn", "multi_turn"],
        default="single_turn",
    )
    parser.add_argument(
        "--protocol-variant",
        default=DEFAULT_PROTOCOL_VARIANT,
        help=(
            "Protocol variant for multi-turn calibration. "
            f"Choices: {', '.join(sorted(MULTI_TURN_PROTOCOL_VARIANTS.keys()))}"
        ),
    )
    parser.add_argument(
        "--replay-json",
        help="Replay an existing run artifact and re-score using the selected protocol variant.",
    )
    parser.add_argument("--plan-json")
    parser.add_argument("--cartridge-id", action="append")
    parser.add_argument("--baseline-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--request-timeout-s", type=int, default=120)
    parser.add_argument(
        "--gate-profile",
        choices=["none", "budget_static", "budget_multiplex", "boundary_multiplex", "dynamic_target"],
        default="none",
    )
    parser.add_argument("--gate-output-json")
    parser.add_argument("--gate-output-md")
    parser.add_argument("--min-accuracy-retained-pct", type=float)
    parser.add_argument("--min-coherence-retained-pct", type=float)
    parser.add_argument("--min-benchmark-accuracy-retained-pct", type=float)
    parser.add_argument("--min-benchmark-coherence-retained-pct", type=float)
    parser.add_argument("--max-parse-error-rate", type=float)
    parser.add_argument("--max-error-rate", type=float)
    parser.add_argument("--max-p95-sample-time-s", type=float)
    parser.add_argument("--max-avg-swap-time-s", type=float)
    parser.add_argument("--max-quality-loss-pct", type=float)
    parser.add_argument("--max-benchmark-accuracy-drop-abs", type=float)
    parser.add_argument(
        "--fail-on-gate-reject",
        action="store_true",
        help="Exit non-zero when the computed gate verdict is invalid or reject.",
    )
    args = parser.parse_args()
    spec_map = {spec.name: spec for spec in BENCHMARK_SPECS}
    if args.protocol == "multi_turn":
        try:
            get_multi_turn_variant(args.protocol_variant)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        args.protocol_variant = DEFAULT_PROTOCOL_VARIANT

    if args.replay_json:
        replay_path = Path(args.replay_json)
        replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_protocol = (
            (replay_payload.get("protocol") or {}).get("name") or args.protocol
        )
        if replay_protocol != "multi_turn":
            raise SystemExit("--replay-json currently supports multi_turn artifacts only.")
        replay_results = [dict(row) for row in replay_payload.get("results", [])]
        results = apply_protocol_variant_rescoring(
            replay_results,
            spec_map,
            protocol_variant=args.protocol_variant,
        )
        payload = dict(replay_payload)
        payload["protocol"] = protocol_metadata("multi_turn", args.protocol_variant)
        payload["protocol_variant"] = args.protocol_variant
        payload["summary"] = summarize_results(results, protocol="multi_turn")
        payload["results"] = results
        payload["replay_source_json"] = str(replay_path.resolve())
    else:
        if not args.mode or not args.server_url or not args.model:
            raise SystemExit(
                "Live evaluation requires --mode, --server-url, and --model."
            )
        try:
            plan = load_plan(args.plan_json)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        cartridge_ids = get_cartridge_ids(plan, args.cartridge_id)

        benchmark_rows = {
            spec.name: load_examples(
                spec, args.sample_count, args.calibration_count, args.seed
            )
            for spec in BENCHMARK_SPECS
        }

        benchmark_to_cartridge = None
        selection = None
        if args.mode == "multiplex":
            if not cartridge_ids:
                raise SystemExit(
                    "Multiplex mode requires --plan-json or one or more --cartridge-id values."
                )
            calibration_sets = {
                benchmark: bundles["calibration"]
                for benchmark, bundles in benchmark_rows.items()
            }
            selection = select_cartridges(
                calibration_sets,
                spec_map,
                args.server_url,
                args.model,
                cartridge_ids,
                protocol=args.protocol,
                protocol_variant=args.protocol_variant,
            )
            benchmark_to_cartridge = {
                benchmark: bundle["selected_cartridge"]
                for benchmark, bundle in selection.items()
            }
        elif args.mode == "dynamic":
            if not args.plan_json:
                raise SystemExit(
                    "Dynamic mode requires --plan-json for a dynamic_core_specialist plan."
                )
            if not plan:
                raise SystemExit(
                    "Dynamic mode requires --plan-json for a dynamic_core_specialist plan."
                )
            try:
                plan = validate_dynamic_plan_contract(plan)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc

        test_rows = []
        for benchmark, bundles in benchmark_rows.items():
            test_rows.extend(bundles["test"])

        evaluator = (
            evaluate_multi_turn_samples
            if args.protocol == "multi_turn"
            else evaluate_samples
        )
        eval_kwargs = {
            "interleaved": (args.mode != "baseline"),
            "seed": args.seed,
            "request_timeout_s": args.request_timeout_s,
        }
        if args.protocol == "multi_turn":
            eval_kwargs["protocol_variant"] = args.protocol_variant
        results = evaluator(
            test_rows,
            spec_map,
            args.server_url,
            args.model,
            args.mode,
            plan,
            benchmark_to_cartridge,
            **eval_kwargs,
        )

        runtime_identity = build_runtime_identity(
            args.server_url,
            mode=args.mode,
            plan_path=args.plan_json,
            plan=plan,
            readiness_evidence=load_runtime_readiness_evidence(args.server_url)
            if args.mode == "dynamic"
            else None,
        )

        payload = {
            "mode": args.mode,
            "model": args.model,
            "sample_count_per_benchmark": args.sample_count,
            "calibration_count_per_benchmark": args.calibration_count,
            "seed": args.seed,
            "protocol": protocol_metadata(args.protocol, args.protocol_variant),
            "protocol_variant": args.protocol_variant,
            "benchmarks": [spec.name for spec in BENCHMARK_SPECS],
            "runtime_identity": runtime_identity,
            "plan_identity": build_plan_identity(plan, args.plan_json),
            "plan": plan,
            "cartridge_ids": cartridge_ids,
            "benchmark_to_cartridge": benchmark_to_cartridge,
            "selection": selection,
            "summary": summarize_results(results, protocol=args.protocol),
            "results": results,
        }

    if args.baseline_json:
        baseline_payload = json.loads(Path(args.baseline_json).read_text())
        payload["comparison"] = compare_to_baseline(payload, baseline_payload)

    gate = None
    if args.gate_profile != "none":
        threshold_overrides = {
            "min_accuracy_retained_pct": args.min_accuracy_retained_pct,
            "min_coherence_retained_pct": args.min_coherence_retained_pct,
            "min_benchmark_accuracy_retained_pct": args.min_benchmark_accuracy_retained_pct,
            "min_benchmark_coherence_retained_pct": args.min_benchmark_coherence_retained_pct,
            "max_parse_error_rate": args.max_parse_error_rate,
            "max_error_rate": args.max_error_rate,
            "max_p95_sample_time_s": args.max_p95_sample_time_s,
            "max_avg_swap_time_s": args.max_avg_swap_time_s,
            "max_quality_loss_pct": args.max_quality_loss_pct,
            "max_benchmark_accuracy_drop_abs": args.max_benchmark_accuracy_drop_abs,
        }
        gate = evaluate_payload_gate(
            payload,
            args.gate_profile,
            threshold_overrides=threshold_overrides,
        )
        payload["gate"] = gate

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.output_md:
        Path(args.output_md).write_text(
            build_markdown_report(payload), encoding="utf-8"
        )

    if gate is not None:
        gate_output_json = Path(args.gate_output_json) if args.gate_output_json else (
            output_json.parent / f"{output_json.stem}.gate.json"
        )
        gate_output_json.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        gate_output_md = Path(args.gate_output_md) if args.gate_output_md else (
            output_json.parent / f"{output_json.stem}.gate.md"
        )
        gate_output_md.write_text(build_gate_markdown(gate), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    if payload.get("comparison"):
        print(json.dumps(payload["comparison"], indent=2))
    if payload.get("gate"):
        print(json.dumps(payload["gate"], indent=2))

    if gate is not None and args.fail_on_gate_reject and gate["verdict"] in {"invalid", "reject"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
