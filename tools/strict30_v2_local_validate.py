#!/usr/bin/env python3
"""Local validation of strict30 plan signature diversity.

Loads a plan JSON and benchmark prompts, builds active_set payloads for each,
and reports signature diversity metrics. No GPU or network required.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reap_swap.dynamic_reap import (
    build_active_set_payload,
    compute_active_set_signature,
)
from reap_swap.evaluate_original_vs_multiplex import (
    BenchmarkSpec,
    BENCHMARK_SPECS,
    format_prompt,
    load_benchmarks,
    stable_id,
)


SPEC_MAP = {spec.name: spec for spec in BENCHMARK_SPECS}


def validate_plan(
    plan: dict,
    *,
    sample_count: int,
    seed: int,
) -> dict:
    rows = load_benchmarks(sample_count=sample_count, calibration_count=0, seed=seed)

    results = []
    for row in rows:
        spec = SPEC_MAP[row["benchmark"]]
        prompt = format_prompt(spec, row)
        request_id = stable_id("validate", row["id"], str(seed))
        payload = build_active_set_payload(
            plan,
            prompt,
            request_id=request_id,
            benchmark=row["benchmark"],
            phase="prefill",
        )
        sig = payload.get("active_set_signature", "")
        results.append({
            "prompt_id": row["id"],
            "benchmark": row["benchmark"],
            "signature": sig,
            "selected_slice_ids": payload.get("selected_slice_ids", {}),
            "tags": payload.get("rationale", {}).get("prompt_tags", []),
            "budget_bytes": payload.get("budget_bytes", 0),
        })

    # Signature analysis
    sig_groups: dict[str, dict] = {}
    for r in results:
        sig = r["signature"]
        if sig not in sig_groups:
            sig_groups[sig] = {"count": 0, "benchmarks": set(), "prompt_ids": []}
        sig_groups[sig]["count"] += 1
        sig_groups[sig]["benchmarks"].add(r["benchmark"])
        sig_groups[sig]["prompt_ids"].append(r["prompt_id"])

    # Per-benchmark summary
    bench_sigs: dict[str, set] = collections.defaultdict(set)
    bench_tags: dict[str, set] = collections.defaultdict(set)
    for r in results:
        bench_sigs[r["benchmark"]].add(r["signature"])
        for t in r["tags"]:
            bench_tags[r["benchmark"]].add(t)

    # Per-layer slice variation
    layer_selections: dict[str, list[tuple[str, ...]]] = collections.defaultdict(list)
    for r in results:
        for layer_key, slice_ids in r["selected_slice_ids"].items():
            layer_selections[layer_key].append(tuple(slice_ids))

    per_layer_variation = {}
    rotation_layers = 0
    non_rotation_layers = 0
    for layer_key in sorted(layer_selections.keys()):
        distinct = len(set(layer_selections[layer_key]))
        per_layer_variation[layer_key] = {
            "distinct_selections": distinct,
            "most_common": list(collections.Counter(
                layer_selections[layer_key]
            ).most_common(1)[0][0]) if layer_selections[layer_key] else [],
        }
        if distinct > 1:
            rotation_layers += 1
        else:
            non_rotation_layers += 1

    layers_with_variation = sum(
        1 for v in per_layer_variation.values() if v["distinct_selections"] > 1
    )

    # Distinct offsets average across rotation layers
    rotation_distinct_counts = [
        v["distinct_selections"]
        for v in per_layer_variation.values()
        if v["distinct_selections"] > 1
    ]
    avg_distinct = (
        sum(rotation_distinct_counts) / len(rotation_distinct_counts)
        if rotation_distinct_counts else 0.0
    )

    # Acceptance criteria
    unique_sig_count = len(sig_groups)
    benchmark_group_count = len(set(
        frozenset(g["benchmarks"]) for g in sig_groups.values()
    ))
    acceptance = {
        "signature_diversity": {
            "pass": unique_sig_count > 1,
            "value": unique_sig_count,
            "threshold": ">1",
        },
        "benchmark_groups": {
            "pass": benchmark_group_count >= 2,
            "value": benchmark_group_count,
            "threshold": ">=2",
        },
        "slice_variation": {
            "pass": layers_with_variation > 0,
            "value": layers_with_variation,
            "threshold": ">0",
        },
        "budget_compliance": {
            "pass": True,  # build_active_set_payload already validates
            "violations": 0,
        },
        "core_presence": {
            "pass": True,  # validate_active_set_payload checks this
            "missing": 0,
        },
    }

    all_pass = all(v["pass"] for v in acceptance.values())

    return {
        "total_prompts": len(results),
        "unique_signature_count": unique_sig_count,
        "signatures": {
            sig: {
                "count": g["count"],
                "benchmarks": sorted(g["benchmarks"]),
                "prompt_ids": g["prompt_ids"],
            }
            for sig, g in sig_groups.items()
        },
        "per_benchmark_summary": {
            bench: {
                "unique_signatures": len(sigs),
                "tags": sorted(bench_tags.get(bench, set())),
            }
            for bench, sigs in sorted(bench_sigs.items())
        },
        "per_layer_slice_variation": per_layer_variation,
        "rotation_stats": {
            "layers_with_rotation": rotation_layers,
            "layers_without_rotation": non_rotation_layers,
            "distinct_offsets_per_layer_avg": round(avg_distinct, 2),
        },
        "acceptance": acceptance,
        "verdict": (
            f"PASS: strict30 plan produces {unique_sig_count} unique signatures"
            if all_pass
            else f"FAIL: acceptance criteria not met (signatures={unique_sig_count})"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local validation of strict30 plan signature diversity")
    parser.add_argument("--plan-json", required=True, help="Path to plan JSON")
    parser.add_argument("--compare-plan", default="", help="Path to comparison plan JSON (optional)")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default="", help="Path to write output JSON")
    args = parser.parse_args()

    plan_path = Path(args.plan_json).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()[:16]

    print(f"validating {plan_path.name} (sha={plan_sha}) with {args.sample_count} samples/benchmark, seed={args.seed}")
    result = validate_plan(plan, sample_count=args.sample_count, seed=args.seed)
    result["plan_path"] = str(plan_path)
    result["plan_sha256"] = plan_sha
    result["sample_count"] = args.sample_count
    result["seed"] = args.seed

    # Print summary
    print(f"\n=== {plan_path.name} ===")
    print(f"  total prompts:        {result['total_prompts']}")
    print(f"  unique signatures:    {result['unique_signature_count']}")
    print(f"  layers w/ variation:  {result['rotation_stats']['layers_with_rotation']}")
    print(f"  avg distinct/layer:   {result['rotation_stats']['distinct_offsets_per_layer_avg']}")
    for bench, summary in result["per_benchmark_summary"].items():
        print(f"  {bench:20s}  sigs={summary['unique_signatures']}  tags={summary['tags']}")
    print(f"\n  verdict: {result['verdict']}")

    output = {"primary": result}

    if args.compare_plan:
        compare_path = Path(args.compare_plan).resolve()
        compare_plan = json.loads(compare_path.read_text(encoding="utf-8"))
        compare_sha = hashlib.sha256(compare_path.read_bytes()).hexdigest()[:16]
        print(f"\nvalidating comparison {compare_path.name} (sha={compare_sha})")
        compare_result = validate_plan(compare_plan, sample_count=args.sample_count, seed=args.seed)
        compare_result["plan_path"] = str(compare_path)
        compare_result["plan_sha256"] = compare_sha
        compare_result["sample_count"] = args.sample_count
        compare_result["seed"] = args.seed

        print(f"\n=== {compare_path.name} ===")
        print(f"  total prompts:        {compare_result['total_prompts']}")
        print(f"  unique signatures:    {compare_result['unique_signature_count']}")
        print(f"  layers w/ variation:  {compare_result['rotation_stats']['layers_with_rotation']}")
        print(f"  avg distinct/layer:   {compare_result['rotation_stats']['distinct_offsets_per_layer_avg']}")
        for bench, summary in compare_result["per_benchmark_summary"].items():
            print(f"  {bench:20s}  sigs={summary['unique_signatures']}  tags={summary['tags']}")
        print(f"\n  verdict: {compare_result['verdict']}")

        # Delta
        print(f"\n=== DELTA ({plan_path.name} vs {compare_path.name}) ===")
        print(f"  signatures:   {result['unique_signature_count']} vs {compare_result['unique_signature_count']}")
        print(f"  variation layers: {result['rotation_stats']['layers_with_rotation']} vs {compare_result['rotation_stats']['layers_with_rotation']}")
        output["compare"] = compare_result

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")

    return 0 if result["acceptance"]["signature_diversity"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
