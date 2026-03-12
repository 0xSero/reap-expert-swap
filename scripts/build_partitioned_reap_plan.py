#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from dynamic_reap import (
    DEFAULT_CANDIDATE_POOL_MULTIPLIER,
    DEFAULT_CORE_BUDGET_FRACTION,
    DEFAULT_MAX_REFRESHES_PER_REQUEST,
    DEFAULT_MAX_RESIDENT_RATIO,
    DEFAULT_ROTATION_POLICY,
    DEFAULT_SELECTION_STRATEGY,
    DEFAULT_SPECIALIST_BUDGET_FRACTION,
    _load_layer_importance,
    build_dynamic_floor_plan,
    build_dynamic_markdown,
    build_dynamic_plan,
    infer_model_config,
)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_summary_arg(raw: str) -> tuple[str, Path]:
    label, sep, path_str = raw.partition("=")
    if not sep or not label.strip() or not path_str.strip():
        raise argparse.ArgumentTypeError(
            "--summary must be provided as label=/path/to/summary.json"
        )
    return label.strip(), Path(path_str.strip())


def normalize_layer_key(layer_key: str) -> str:
    return layer_key if layer_key.startswith("layer_") else f"layer_{layer_key}"


def build_summary_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs = list(args.summary or [])
    if args.coding_summary:
        specs.append(("coding", Path(args.coding_summary)))
    if args.communication_summary:
        specs.append(("communication", Path(args.communication_summary)))
    if not specs:
        raise SystemExit("At least one summary is required")
    return specs


def load_activation_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def assign_experts_to_cartridges(
    expert_rows: list[dict[str, Any]],
    cartridge_ids: list[str],
    max_counts: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    slots = {cartridge_id: [] for cartridge_id in cartridge_ids}
    totals = {cartridge_id: 0.0 for cartridge_id in cartridge_ids}
    if max_counts is None:
        max_counts = [len(expert_rows) // len(cartridge_ids)] * len(cartridge_ids)
        for idx in range(len(expert_rows) % len(cartridge_ids)):
            max_counts[idx] += 1
    if len(max_counts) != len(cartridge_ids):
        raise ValueError("max_counts must match cartridge_ids length")
    max_per_cartridge = dict(zip(cartridge_ids, max_counts, strict=True))

    for expert in sorted(
        expert_rows,
        key=lambda row: (row["combinedSignal"], row["peakSignal"], -row["expert"]),
        reverse=True,
    ):
        eligible = [
            cartridge_id
            for cartridge_id in cartridge_ids
            if len(slots[cartridge_id]) < max_per_cartridge[cartridge_id]
        ]
        best = min(
            eligible,
            key=lambda cartridge_id: (
                totals[cartridge_id],
                len(slots[cartridge_id]),
                cartridge_id,
            ),
        )
        slots[best].append(expert)
        totals[best] += expert["combinedSignal"]

    out: dict[str, dict[str, Any]] = {}
    for cartridge_id in cartridge_ids:
        experts = sorted(row["expert"] for row in slots[cartridge_id])
        expert_set = set(experts)
        out[cartridge_id] = {
            "keep": experts,
            "drop": sorted(
                row["expert"] for row in expert_rows if row["expert"] not in expert_set
            ),
            "expertCount": len(experts),
            "combinedSignal": round(
                sum(row["combinedSignal"] for row in slots[cartridge_id]), 8
            ),
            "signalsBySummary": {
                label: round(
                    sum(float(row["signalsBySummary"][label]) for row in slots[cartridge_id]),
                    8,
                )
                for label in expert_rows[0]["signalsBySummary"].keys()
            }
            if expert_rows
            else {},
        }
    return out


def compute_budget_layout(
    layer_expert_counts: list[int], gpu_budget_pct: float
) -> tuple[list[str], dict[int, dict[str, Any]]]:
    if gpu_budget_pct <= 0 or gpu_budget_pct > 100:
        raise SystemExit("--gpu-budget-pct must be > 0 and <= 100")

    layout_by_expert_count: dict[int, dict[str, Any]] = {}
    max_cartridge_count = 0
    for num_experts in sorted(set(layer_expert_counts)):
        resident_experts = max(1, round(num_experts * gpu_budget_pct / 100.0))
        cartridge_count = max(1, math.ceil(num_experts / resident_experts))
        max_cartridge_count = max(max_cartridge_count, cartridge_count)
        layout_by_expert_count[num_experts] = {
            "targetResidentExperts": resident_experts,
            "cartridgeCount": cartridge_count,
            "actualResidentFractionPct": round(
                100.0 * resident_experts / num_experts, 8
            ),
        }

    cartridge_ids = [f"cartridge_{idx:02d}" for idx in range(max_cartridge_count)]
    for config in layout_by_expert_count.values():
        active_count = config["cartridgeCount"]
        config["activeCartridgeIds"] = cartridge_ids[:active_count]
        config["maxCounts"] = [config["targetResidentExperts"]] * active_count
    return cartridge_ids, layout_by_expert_count


def build_plan(
    summaries: list[tuple[str, dict[str, Any]]],
    signal_key: str,
    cartridge_count: int | None,
    gpu_budget_pct: float | None,
) -> dict[str, Any]:
    first_summary = summaries[0][1]
    layer_keys = list(first_summary["layers"].keys())
    layer_expert_counts = [
        len(first_summary["layers"][raw_layer_key][signal_key])
        for raw_layer_key in layer_keys
    ]

    budget_layout_by_expert_count = None
    if gpu_budget_pct is not None:
        cartridge_ids, budget_layout_by_expert_count = compute_budget_layout(
            layer_expert_counts, gpu_budget_pct
        )
        cartridge_count = len(cartridge_ids)
    else:
        if cartridge_count is None:
            cartridge_count = 10
        cartridge_ids = [f"cartridge_{idx:02d}" for idx in range(cartridge_count)]

    plan: dict[str, Any] = {
        "model": first_summary["model"],
        "signalKey": signal_key,
        "cartridgeCount": cartridge_count,
        "targetGpuBudgetPct": gpu_budget_pct,
        "sourceSummaries": [
            {
                "label": label,
                "workflow": summary.get("workflow", label),
                "processedSamples": summary.get("processedSamples", 0),
                "totalTokens": summary.get("totalTokens", 0),
            }
            for label, summary in summaries
        ],
        "perLayer": {},
        "summary": {
            "cartridgeIds": cartridge_ids,
        },
    }

    total_experts = 0
    combined_mass_total = 0.0
    layer_balance_scores: list[float] = []
    resident_experts_per_layer: list[int] = []
    resident_fraction_pcts: list[float] = []

    for raw_layer_key in layer_keys:
        expert_rows = []
        num_experts = len(first_summary["layers"][raw_layer_key][signal_key])
        normalized_key = normalize_layer_key(raw_layer_key)
        for expert_idx in range(num_experts):
            signals_by_summary = {
                label: float(summary["layers"][raw_layer_key][signal_key][expert_idx])
                for label, summary in summaries
            }
            combined_signal = sum(signals_by_summary.values())
            expert_rows.append(
                {
                    "expert": expert_idx,
                    "signalsBySummary": signals_by_summary,
                    "combinedSignal": combined_signal,
                    "peakSignal": max(signals_by_summary.values()) if signals_by_summary else 0.0,
                }
            )

        if budget_layout_by_expert_count is not None:
            layer_budget = budget_layout_by_expert_count[num_experts]
            active_ids = layer_budget["activeCartridgeIds"]
            cartridges = assign_experts_to_cartridges(
                expert_rows,
                active_ids,
                max_counts=layer_budget["maxCounts"],
            )
            inactive_drop = sorted(row["expert"] for row in expert_rows)
            for inactive_id in cartridge_ids[len(active_ids) :]:
                cartridges[inactive_id] = {
                    "keep": [],
                    "drop": inactive_drop,
                    "expertCount": 0,
                    "combinedSignal": 0.0,
                    "signalsBySummary": {
                        label: 0.0
                        for label in expert_rows[0]["signalsBySummary"].keys()
                    }
                    if expert_rows
                    else {},
                }
            resident_experts = max(
                cartridge["expertCount"] for cartridge in cartridges.values()
            )
            resident_fraction_pct = round(100.0 * resident_experts / num_experts, 8)
        else:
            cartridges = assign_experts_to_cartridges(expert_rows, cartridge_ids)
            resident_experts = max(
                cartridge["expertCount"] for cartridge in cartridges.values()
            )
            resident_fraction_pct = round(100.0 * resident_experts / num_experts, 8)

        assignment = {}
        cartridge_totals = [
            cartridges[cartridge_id]["combinedSignal"] for cartridge_id in cartridge_ids
        ]
        if cartridge_totals:
            layer_balance_scores.append(max(cartridge_totals) - min(cartridge_totals))
        for cartridge_id, cartridge in cartridges.items():
            for expert_idx in cartridge["keep"]:
                assignment[expert_idx] = cartridge_id

        plan["perLayer"][normalized_key] = {
            "rawLayerKey": raw_layer_key,
            "numExperts": num_experts,
            "residentExpertsPerCartridge": resident_experts,
            "residentFractionPct": resident_fraction_pct,
            "cartridges": cartridges,
            "expertToCartridge": [assignment[idx] for idx in range(num_experts)],
            "combinedSignalTotal": round(
                sum(row["combinedSignal"] for row in expert_rows), 8
            ),
        }
        total_experts += num_experts
        combined_mass_total += sum(row["combinedSignal"] for row in expert_rows)
        resident_experts_per_layer.append(resident_experts)
        resident_fraction_pcts.append(resident_fraction_pct)

    plan["summary"].update(
        {
            "layerCount": len(layer_keys),
            "totalExperts": total_experts,
            "averageExpertsPerCartridge": round(total_experts / cartridge_count, 8),
            "combinedSignalTotal": round(combined_mass_total, 8),
            "averageLayerSignalSpread": round(
                sum(layer_balance_scores) / len(layer_balance_scores), 8
            )
            if layer_balance_scores
            else 0.0,
            "residentExpertsPerLayerMin": min(resident_experts_per_layer),
            "residentExpertsPerLayerMax": max(resident_experts_per_layer),
            "residentFractionPctMin": round(min(resident_fraction_pcts), 8),
            "residentFractionPctMax": round(max(resident_fraction_pcts), 8),
        }
    )
    return plan


def build_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# REAP partition report",
        "",
        f"- model: `{plan['model']}`",
        f"- signal key: `{plan['signalKey']}`",
        f"- cartridge count: {plan['cartridgeCount']}",
        f"- target gpu budget pct: {plan['targetGpuBudgetPct']}"
        if plan.get("targetGpuBudgetPct") is not None
        else "- target gpu budget pct: n/a",
        f"- layer count: {plan['summary']['layerCount']}",
        f"- total experts: {plan['summary']['totalExperts']}",
        f"- average experts per cartridge: {plan['summary']['averageExpertsPerCartridge']}",
        f"- average layer signal spread: {plan['summary']['averageLayerSignalSpread']}",
        f"- resident experts per layer: {plan['summary']['residentExpertsPerLayerMin']}..{plan['summary']['residentExpertsPerLayerMax']}",
        f"- resident fraction pct: {plan['summary']['residentFractionPctMin']}..{plan['summary']['residentFractionPctMax']}",
        "",
        "## Source summaries",
        "",
    ]
    for source in plan["sourceSummaries"]:
        lines.append(
            f"- `{source['label']}`: workflow={source['workflow']}, samples={source['processedSamples']}, tokens={source['totalTokens']}"
        )
    lines.append("")
    for layer_key, layer in plan["perLayer"].items():
        lines.extend(
            [
                f"## {layer_key}",
                f"- experts: {layer['numExperts']}",
                f"- resident experts per cartridge: {layer['residentExpertsPerCartridge']}",
                f"- resident fraction pct: {layer['residentFractionPct']}",
                f"- combined signal total: {layer['combinedSignalTotal']}",
            ]
        )
        for cartridge_id, cartridge in layer["cartridges"].items():
            lines.append(
                f"- {cartridge_id}: keep={cartridge['keep']} combined_signal={cartridge['combinedSignal']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build either a static cartridge REAP plan or a dynamic core/specialist plan."
    )
    parser.add_argument("--summary", action="append", type=parse_summary_arg, help="Repeatable summary spec in the form label=/path/to/summary.json")
    parser.add_argument("--coding-summary")
    parser.add_argument("--communication-summary")
    parser.add_argument(
        "--signal-key",
        default="reap",
        choices=["reap", "weighted_ean_sum", "ean_sum", "expert_frequency"],
    )
    parser.add_argument("--mode", choices=["static", "dynamic", "dynamic-floor"], default="static")
    parser.add_argument("--cartridge-count", type=int)
    parser.add_argument("--gpu-budget-pct", type=float)
    parser.add_argument("--activation-corpus-jsonl")
    parser.add_argument("--model-config-json")
    parser.add_argument("--layer-importance-json")
    parser.add_argument("--max-resident-ratio", type=float, default=DEFAULT_MAX_RESIDENT_RATIO)
    parser.add_argument("--max-resident-gib", type=float)
    parser.add_argument("--selection-strategy", choices=["activation_mass", "support_v1"], default=DEFAULT_SELECTION_STRATEGY)
    parser.add_argument("--core-selection-mode", choices=["selection_mass", "floor_seeded"], default="selection_mass")
    parser.add_argument("--rotation-policy", choices=["none", "late_prompt_hash"], default=DEFAULT_ROTATION_POLICY)
    parser.add_argument("--core-budget-fraction", type=float, default=DEFAULT_CORE_BUDGET_FRACTION)
    parser.add_argument("--specialist-budget-fraction", type=float, default=DEFAULT_SPECIALIST_BUDGET_FRACTION)
    parser.add_argument("--candidate-pool-multiplier", type=float, default=DEFAULT_CANDIDATE_POOL_MULTIPLIER)
    parser.add_argument("--max-refreshes-per-request", type=int, default=DEFAULT_MAX_REFRESHES_PER_REQUEST)
    parser.add_argument("--floor-exact-fraction", type=float, default=0.30)
    parser.add_argument("--floor-layer-weight-mode", choices=["none", "late_boost"], default="late_boost")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    if args.cartridge_count is not None and args.cartridge_count <= 0:
        raise SystemExit("--cartridge-count must be positive")
    if args.gpu_budget_pct is not None and not (0 < args.gpu_budget_pct <= 100):
        raise SystemExit("--gpu-budget-pct must be > 0 and <= 100")

    summary_specs = build_summary_specs(args)
    summaries = [(label, load_summary(path)) for label, path in summary_specs]

    if args.mode in {"dynamic", "dynamic-floor"}:
        activation_records = load_activation_records(Path(args.activation_corpus_jsonl)) if args.activation_corpus_jsonl else []
        model_config = infer_model_config(summaries[0][1].get("model"), args.model_config_json)
        layer_importance = _load_layer_importance(args.layer_importance_json)
        if args.mode == "dynamic":
            plan = build_dynamic_plan(
                summaries,
                signal_key=args.signal_key,
                model_config=model_config,
                activation_records=activation_records,
                layer_importance=layer_importance,
                selection_strategy=args.selection_strategy,
                core_selection_mode=args.core_selection_mode,
                floor_layer_weight_mode=args.floor_layer_weight_mode,
                rotation_policy=args.rotation_policy,
                max_resident_ratio=args.max_resident_ratio,
                max_resident_gib=args.max_resident_gib,
                core_budget_fraction=args.core_budget_fraction,
                specialist_budget_fraction=args.specialist_budget_fraction,
                candidate_pool_multiplier=args.candidate_pool_multiplier,
                max_refreshes_per_request=args.max_refreshes_per_request,
            )
        else:
            plan = build_dynamic_floor_plan(
                summaries,
                signal_key=args.signal_key,
                model_config=model_config,
                exact_fraction_of_full=args.floor_exact_fraction,
                layer_weight_mode=args.floor_layer_weight_mode,
                activation_records=activation_records,
            )
        rendered_md = build_dynamic_markdown(plan)
    else:
        plan = build_plan(
            summaries,
            args.signal_key,
            args.cartridge_count,
            args.gpu_budget_pct,
        )
        rendered_md = build_markdown(plan)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(rendered_md, encoding="utf-8")
    print(json.dumps(plan["summary"], indent=2))


if __name__ == "__main__":
    main()
