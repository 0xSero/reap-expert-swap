#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from dynamic_reap import compute_dynamic_budget, infer_model_config


BUDGET_RATIOS = (0.20, 0.25, 0.30, 0.35, 0.40)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def collect_trace_rows(history_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in history_paths:
        payload = load_json(path)
        for row in payload.get("results", []):
            router_misses = row.get("router_misses") or {}
            by_layer = router_misses.get("by_layer") or {}
            if by_layer:
                rows.append(
                    {
                        "history_path": str(path),
                        "benchmark": str(row.get("benchmark") or "unknown"),
                        "id": str(row.get("id") or ""),
                        "question": str(row.get("question") or ""),
                        "by_layer": by_layer,
                    }
                )
    return rows


def normalized_weights(observer_summary: dict[str, Any], layer_key: str, inactive_experts: list[int]) -> dict[int, float]:
    raw_layer_key = str(layer_key).replace("layer_", "")
    layer = observer_summary["layers"][raw_layer_key]
    scores = layer.get("reap") or layer.get("weighted_ean_sum") or []
    weights = {int(expert): float(scores[int(expert)]) for expert in inactive_experts if int(expert) < len(scores)}
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(inactive_experts))
        return {int(expert): uniform for expert in inactive_experts}
    return {expert: max(0.0, value) / total for expert, value in weights.items()}


def concentration_metrics(values: list[float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values)
    if total <= 0:
        return {"top1_share": 0.0, "top8_share": 0.0, "entropy": 0.0}
    probs = [max(0.0, value) / total for value in values if value > 0.0]
    sorted_probs = sorted(probs, reverse=True)
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return {
        "top1_share": sorted_probs[0] if sorted_probs else 0.0,
        "top8_share": sum(sorted_probs[:8]),
        "entropy": entropy,
    }


def average_pairwise_jaccard(sets: list[set[int]]) -> float:
    if len(sets) <= 1:
        return 1.0 if sets else 0.0
    scores = []
    for idx in range(len(sets)):
        for other_idx in range(idx + 1, len(sets)):
            left = sets[idx]
            right = sets[other_idx]
            union = left | right
            scores.append((len(left & right) / len(union)) if union else 1.0)
    return fmean(scores) if scores else 0.0


def per_layer_importance(observer_summary: dict[str, Any], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_sets: dict[str, list[set[int]]] = defaultdict(list)
    by_layer_inactive_ratio: dict[str, list[float]] = defaultdict(list)
    by_layer_benchmark_ratio: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    out: dict[str, Any] = {}

    for row in trace_rows:
        benchmark = row["benchmark"]
        for layer_key, layer_row in row["by_layer"].items():
            inactive = set(int(expert) for expert in layer_row.get("inactive_experts", []))
            inactive_mass = float(layer_row.get("inactive_mass", 0.0) or 0.0)
            observed_mass = float(layer_row.get("observed_mass", 0.0) or 0.0)
            inactive_ratio = inactive_mass / observed_mass if observed_mass > 0 else 0.0
            by_layer_sets[layer_key].append(inactive)
            by_layer_inactive_ratio[layer_key].append(inactive_ratio)
            by_layer_benchmark_ratio[layer_key][benchmark].append(inactive_ratio)

    for layer_key, layer in observer_summary["layers"].items():
        normalized_layer = f"layer_{layer_key}"
        concentration = concentration_metrics([float(value) for value in layer.get("reap", [])])
        inactive_sets = by_layer_sets.get(normalized_layer, [])
        benchmark_means = {
            benchmark: fmean(values) for benchmark, values in by_layer_benchmark_ratio.get(normalized_layer, {}).items()
        }
        benchmark_dependence = (
            (max(benchmark_means.values()) - min(benchmark_means.values())) if len(benchmark_means) > 1 else 0.0
        )
        out[normalized_layer] = {
            "routing_concentration_top1": round(concentration["top1_share"], 8),
            "routing_concentration_top8": round(concentration["top8_share"], 8),
            "expert_entropy": round(concentration["entropy"], 8),
            "miss_sensitivity": round(fmean(by_layer_inactive_ratio.get(normalized_layer, [0.0])), 8),
            "benchmark_dependence": round(benchmark_dependence, 8),
            "prompt_overlap_jaccard": round(average_pairwise_jaccard(inactive_sets), 8),
            "trace_count": len(inactive_sets),
        }
    return out


def uniform_allocation(total_capacity: int, num_layers: int) -> dict[str, int]:
    base = total_capacity // num_layers
    remainder = total_capacity % num_layers
    allocation = {}
    for idx in range(num_layers):
        allocation[f"layer_{idx}"] = base + (1 if idx < remainder else 0)
    return allocation


def nonuniform_allocation(total_capacity: int, layer_stats: dict[str, Any]) -> dict[str, int]:
    layers = sorted(layer_stats.keys(), key=lambda key: int(key.replace("layer_", "")))
    scores = {}
    for layer_key in layers:
        row = layer_stats[layer_key]
        score = (
            (0.50 * float(row["miss_sensitivity"]))
            + (0.20 * float(row["benchmark_dependence"]))
            + (0.20 * float(row["expert_entropy"]))
            + (0.10 * (1.0 - float(row["prompt_overlap_jaccard"])))
        )
        scores[layer_key] = max(score, 1e-6)
    total_score = sum(scores.values())
    raw = {layer_key: (scores[layer_key] / total_score) * total_capacity for layer_key in layers}
    allocation = {layer_key: int(math.floor(value)) for layer_key, value in raw.items()}
    used = sum(allocation.values())
    remainders = sorted(((raw[layer_key] - allocation[layer_key], layer_key) for layer_key in layers), reverse=True)
    for _fraction, layer_key in remainders[: max(0, total_capacity - used)]:
        allocation[layer_key] += 1
    return allocation


def recovered_inactive_mass(
    observer_summary: dict[str, Any],
    trace_row: dict[str, Any],
    allocation: dict[str, int],
) -> dict[str, float]:
    observed_mass_total = 0.0
    inactive_mass_total = 0.0
    recovered_mass_total = 0.0
    for layer_key, layer_row in trace_row["by_layer"].items():
        inactive_experts = [int(expert) for expert in layer_row.get("inactive_experts", [])]
        if not inactive_experts:
            observed_mass_total += float(layer_row.get("observed_mass", 0.0) or 0.0)
            continue
        inactive_mass = float(layer_row.get("inactive_mass", 0.0) or 0.0)
        observed_mass = float(layer_row.get("observed_mass", 0.0) or 0.0)
        observed_mass_total += observed_mass
        inactive_mass_total += inactive_mass
        capacity = min(len(inactive_experts), int(allocation.get(layer_key, 0)))
        if capacity <= 0 or inactive_mass <= 0:
            continue
        weights = normalized_weights(observer_summary, layer_key, inactive_experts)
        recovered_fraction = sum(
            weight for _expert, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:capacity]
        )
        recovered_mass_total += inactive_mass * min(1.0, recovered_fraction)
    active_mass_total = max(0.0, observed_mass_total - inactive_mass_total)
    availability = (active_mass_total + recovered_mass_total) / observed_mass_total if observed_mass_total > 0 else 0.0
    return {
        "observed_mass_total": observed_mass_total,
        "inactive_mass_total": inactive_mass_total,
        "recovered_inactive_mass_total": recovered_mass_total,
        "availability": availability,
    }


def evaluate_budget(
    observer_summary: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    allocation: dict[str, int],
) -> dict[str, Any]:
    per_prompt = []
    per_benchmark: dict[str, list[float]] = defaultdict(list)
    for row in trace_rows:
        metrics = recovered_inactive_mass(observer_summary, row, allocation)
        per_prompt.append(
            {
                "benchmark": row["benchmark"],
                "id": row["id"],
                "availability": round(metrics["availability"], 8),
                "inactive_mass_total": round(metrics["inactive_mass_total"], 8),
                "recovered_inactive_mass_total": round(metrics["recovered_inactive_mass_total"], 8),
            }
        )
        per_benchmark[row["benchmark"]].append(metrics["availability"])
    return {
        "avg_availability": round(fmean([row["availability"] for row in per_prompt]), 8) if per_prompt else 0.0,
        "min_availability": round(min((row["availability"] for row in per_prompt), default=0.0), 8),
        "by_benchmark": {
            benchmark: {
                "avg_availability": round(fmean(values), 8),
                "prompt_count": len(values),
            }
            for benchmark, values in sorted(per_benchmark.items())
        },
        "per_prompt": per_prompt,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Budget oracle analysis",
        "",
        f"- trace rows: {report['trace_rows']}",
        f"- history files: {report['history_files']}",
        f"- note: {report['note']}",
        "",
        "## Budget upper bounds",
        "",
        "| budget | active expert capacity | uniform avg availability | nonuniform avg availability | uniform min | nonuniform min |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["budgets"]:
        lines.append(
            f"| {int(row['budget_ratio'] * 100)}% | {row['total_active_expert_capacity']} | "
            f"{row['uniform']['avg_availability']:.3f} | {row['nonuniform']['avg_availability']:.3f} | "
            f"{row['uniform']['min_availability']:.3f} | {row['nonuniform']['min_availability']:.3f} |"
        )
    lines.extend(["", "## Suggested nonuniform allocations", ""])
    for row in report["budgets"]:
        allocation = row["nonuniform_allocation"]
        top_layers = sorted(allocation.items(), key=lambda item: item[1], reverse=True)[:10]
        lines.append(f"### {int(row['budget_ratio'] * 100)}%")
        lines.append("")
        lines.append(
            "- top allocated layers: "
            + ", ".join(f"`{layer}`={count}" for layer, count in top_layers)
        )
        lines.append("")
    lines.extend(["", "## Layer importance", ""])
    for layer_key, row in sorted(report["layer_importance"].items(), key=lambda item: int(item[0].replace("layer_", ""))):
        lines.append(
            f"- `{layer_key}`: miss_sensitivity={row['miss_sensitivity']:.3f} entropy={row['expert_entropy']:.3f} top8={row['routing_concentration_top8']:.3f} benchmark_dependence={row['benchmark_dependence']:.3f} overlap={row['prompt_overlap_jaccard']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-derived surrogate oracle for budgeted live-set availability.")
    parser.add_argument("--observer-summary-json", required=True)
    parser.add_argument("--model-config-json", required=True)
    parser.add_argument("--history-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    observer_summary = load_json(args.observer_summary_json)
    model_config = infer_model_config(observer_summary.get("model"), args.model_config_json)
    trace_rows = collect_trace_rows([Path(path) for path in args.history_json])
    layer_stats = per_layer_importance(observer_summary, trace_rows)

    budgets = []
    num_layers = int(model_config["num_hidden_layers"])
    for ratio in BUDGET_RATIOS:
        budget = compute_dynamic_budget(
            model_config,
            max_resident_ratio=ratio,
            core_budget_fraction=0.5,
            specialist_budget_fraction=0.5,
            candidate_pool_multiplier=1.0,
            max_refreshes_per_request=0,
        )
        total_capacity = int(budget["total_active_expert_capacity"])
        uniform = evaluate_budget(observer_summary, trace_rows, uniform_allocation(total_capacity, num_layers))
        nonuniform_alloc = nonuniform_allocation(total_capacity, layer_stats)
        nonuniform = evaluate_budget(observer_summary, trace_rows, nonuniform_alloc)
        budgets.append(
            {
                "budget_ratio": ratio,
                "total_active_expert_capacity": total_capacity,
                "uniform_allocation": uniform_allocation(total_capacity, num_layers),
                "uniform": uniform,
                "nonuniform_allocation": nonuniform_alloc,
                "nonuniform": nonuniform,
            }
        )

    report = {
        "observer_summary_json": str(Path(args.observer_summary_json)),
        "history_files": len(args.history_json),
        "trace_rows": len(trace_rows),
        "note": (
            "This is a trace-derived surrogate oracle, not full token-level omniscience. "
            "Per-layer inactive mass is known exactly from live traces; inactive mass is distributed across inactive experts "
            "using observer-summary REAP weights as a proxy because per-inactive-expert masses are not stored in the trace payloads."
        ),
        "layer_importance": layer_stats,
        "budgets": budgets,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "trace_rows": len(trace_rows)}, indent=2))


if __name__ == "__main__":
    main()
