#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from router_activity import average_pairwise_jaccard, coverage_from_mass_map


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable_prompt_key(row: dict[str, Any]) -> str:
    return str(row.get('request_id') or row.get('id') or row.get('question') or 'unknown')


def summarize_dynamic_router_activity(
    dynamic_payload: dict[str, Any],
    *,
    plan: dict[str, Any] | None = None,
    thresholds: tuple[float, ...] = (0.8, 0.9, 0.95),
) -> dict[str, Any]:
    results = list(dynamic_payload.get('results') or [])
    per_layer_active_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_layer_inactive_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_layer_active_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_layer_inactive_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_layer_prompt_active_sets: dict[str, list[set[int]]] = defaultdict(list)
    per_layer_prompt_inactive_sets: dict[str, list[set[int]]] = defaultdict(list)
    per_layer_inactive_ratios: dict[str, list[float]] = defaultdict(list)
    per_layer_observed_tokens: Counter[str] = Counter()
    per_layer_route_decisions: Counter[str] = Counter()
    benchmark_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_rows: list[dict[str, Any]] = []

    for row in results:
        benchmark_rows[str(row.get('benchmark') or 'unknown')].append(row)
        request_id = str(row.get('request_id') or '')
        prompt_entry = {
            'request_id': request_id,
            'prompt_id': stable_prompt_key(row),
            'benchmark': str(row.get('benchmark') or 'unknown'),
            'correct': bool(row.get('correct')),
            'coherent': bool(row.get('coherent')),
            'parse_error': bool(row.get('parse_error')),
            'inactive_ratio': float((row.get('router_miss_summary') or {}).get('inactive_ratio', 0.0) or 0.0),
            'layers': {},
        }
        router_misses = row.get('router_misses') or {}
        for layer_key, layer_row in (router_misses.get('by_layer') or {}).items():
            active_mass_map = {str(k): float(v) for k, v in (layer_row.get('active_expert_mass') or {}).items()}
            inactive_mass_map = {str(k): float(v) for k, v in (layer_row.get('inactive_expert_mass') or {}).items()}
            active_count_map = {str(k): int(v) for k, v in (layer_row.get('active_expert_counts') or {}).items()}
            inactive_count_map = {str(k): int(v) for k, v in (layer_row.get('inactive_expert_counts') or {}).items()}
            for expert, value in active_mass_map.items():
                per_layer_active_mass[layer_key][expert] += value
            for expert, value in inactive_mass_map.items():
                per_layer_inactive_mass[layer_key][expert] += value
            for expert, value in active_count_map.items():
                per_layer_active_counts[layer_key][expert] += value
            for expert, value in inactive_count_map.items():
                per_layer_inactive_counts[layer_key][expert] += value
            active_set = {int(expert) for expert in layer_row.get('active_experts', [])}
            inactive_set = {int(expert) for expert in layer_row.get('inactive_experts', [])}
            per_layer_prompt_active_sets[layer_key].append(active_set)
            per_layer_prompt_inactive_sets[layer_key].append(inactive_set)
            observed_mass = float(layer_row.get('observed_mass', 0.0) or 0.0)
            inactive_mass = float(layer_row.get('inactive_mass', 0.0) or 0.0)
            inactive_ratio = inactive_mass / observed_mass if observed_mass > 0 else 0.0
            per_layer_inactive_ratios[layer_key].append(inactive_ratio)
            per_layer_observed_tokens[layer_key] += int(layer_row.get('observed_token_count', 0) or 0)
            per_layer_route_decisions[layer_key] += int(layer_row.get('route_decision_count', 0) or 0)
            prompt_entry['layers'][layer_key] = {
                'inactive_ratio': round(inactive_ratio, 8),
                'active_top_experts': coverage_from_mass_map(active_mass_map, thresholds).get('top_experts_by_mass', [])[:8],
                'inactive_top_experts': coverage_from_mass_map(inactive_mass_map, thresholds).get('top_experts_by_mass', [])[:8],
            }
        prompt_rows.append(prompt_entry)

    per_layer_summary: dict[str, Any] = {}
    total_resident_experts = 0
    total_active95_experts = 0
    total_inactive95_experts = 0
    for layer_key in sorted(set(per_layer_active_mass) | set(per_layer_inactive_mass), key=lambda item: int(item.replace('layer_', ''))):
        resident_count = None
        total_experts = None
        if plan:
            layer_plan = (plan.get('perLayer') or {}).get(layer_key) or {}
            resident_count = len(layer_plan.get('coreExperts') or [])
            total_experts = int(layer_plan.get('numExperts', 0) or 0) or None
        active_cov = coverage_from_mass_map(dict(per_layer_active_mass[layer_key]), thresholds)
        inactive_cov = coverage_from_mass_map(dict(per_layer_inactive_mass[layer_key]), thresholds)
        active95 = active_cov.get('95pct', {}).get('expert_count', 0)
        inactive95 = inactive_cov.get('95pct', {}).get('expert_count', 0)
        total_resident_experts += int(resident_count or 0)
        total_active95_experts += int(active95 or 0)
        total_inactive95_experts += int(inactive95 or 0)
        per_layer_summary[layer_key] = {
            'resident_expert_count': resident_count,
            'total_expert_count': total_experts,
            'observed_token_count': int(per_layer_observed_tokens[layer_key]),
            'route_decision_count': int(per_layer_route_decisions[layer_key]),
            'avg_inactive_ratio': round(sum(per_layer_inactive_ratios[layer_key]) / max(1, len(per_layer_inactive_ratios[layer_key])), 8),
            'active_prompt_overlap_jaccard': average_pairwise_jaccard(per_layer_prompt_active_sets[layer_key]),
            'inactive_prompt_overlap_jaccard': average_pairwise_jaccard(per_layer_prompt_inactive_sets[layer_key]),
            'active_mass': active_cov,
            'inactive_mass': inactive_cov,
            'active_top_experts_by_count': [
                {'expert': int(expert), 'count': int(value)}
                for expert, value in sorted(per_layer_active_counts[layer_key].items(), key=lambda item: item[1], reverse=True)[:16]
            ],
            'inactive_top_experts_by_count': [
                {'expert': int(expert), 'count': int(value)}
                for expert, value in sorted(per_layer_inactive_counts[layer_key].items(), key=lambda item: item[1], reverse=True)[:16]
            ],
        }
        if resident_count:
            per_layer_summary[layer_key]['active_95pct_fraction_of_resident'] = round(active95 / resident_count, 8)
            per_layer_summary[layer_key]['inactive_95pct_fraction_of_resident'] = round(inactive95 / resident_count, 8)

    benchmark_summary: dict[str, Any] = {}
    for benchmark, rows in sorted(benchmark_rows.items()):
        ratios = [float((row.get('router_miss_summary') or {}).get('inactive_ratio', 0.0) or 0.0) for row in rows]
        benchmark_summary[benchmark] = {
            'rows': len(rows),
            'accuracy': round(sum(1 for row in rows if row.get('correct')) / max(1, len(rows)), 8),
            'coherence': round(sum(1 for row in rows if row.get('coherent')) / max(1, len(rows)), 8),
            'parse_error_rate': round(sum(1 for row in rows if row.get('parse_error')) / max(1, len(rows)), 8),
            'avg_inactive_ratio': round(sum(ratios) / max(1, len(ratios)), 8),
        }

    overall_inactive_ratios = [float((row.get('router_miss_summary') or {}).get('inactive_ratio', 0.0) or 0.0) for row in results]
    output = {
        'result_count': len(results),
        'overall': {
            'avg_inactive_ratio': round(sum(overall_inactive_ratios) / max(1, len(overall_inactive_ratios)), 8),
            'accuracy': round(sum(1 for row in results if row.get('correct')) / max(1, len(results)), 8),
            'coherence': round(sum(1 for row in results if row.get('coherent')) / max(1, len(results)), 8),
            'parse_error_rate': round(sum(1 for row in results if row.get('parse_error')) / max(1, len(results)), 8),
            'resident_expert_count_total': total_resident_experts or None,
            'active_95pct_expert_count_total': total_active95_experts,
            'inactive_95pct_expert_count_total': total_inactive95_experts,
        },
        'by_benchmark': benchmark_summary,
        'by_layer': per_layer_summary,
        'prompts': prompt_rows,
    }
    if total_resident_experts:
        output['overall']['active_95pct_fraction_of_resident'] = round(total_active95_experts / total_resident_experts, 8)
        output['overall']['inactive_95pct_fraction_of_resident'] = round(total_inactive95_experts / total_resident_experts, 8)
    return output


def build_markdown(profile: dict[str, Any], *, plan_path: str | None, dynamic_path: str) -> str:
    overall = profile['overall']
    lines = [
        '# Router Activity Profile',
        '',
        f'- dynamic artifact: `{dynamic_path}`',
        f'- plan artifact: `{plan_path or "n/a"}`',
        f"- prompts: **{profile['result_count']}**",
        f"- accuracy: **{overall['accuracy']:.2%}**",
        f"- coherence: **{overall['coherence']:.2%}**",
        f"- parse error rate: **{overall['parse_error_rate']:.2%}**",
        f"- avg inactive ratio: **{overall['avg_inactive_ratio']:.4f}**",
    ]
    if overall.get('resident_expert_count_total'):
        lines.extend([
            f"- resident experts total: **{overall['resident_expert_count_total']}**",
            f"- active 95pct expert total: **{overall['active_95pct_expert_count_total']}**",
            f"- active 95pct fraction of resident: **{overall.get('active_95pct_fraction_of_resident', 0.0):.2%}**",
            f"- inactive 95pct expert total: **{overall['inactive_95pct_expert_count_total']}**",
        ])
    lines.extend(['', '## Benchmarks', ''])
    for benchmark, row in profile['by_benchmark'].items():
        lines.append(
            f"- `{benchmark}`: acc={row['accuracy']:.2%}, coh={row['coherence']:.2%}, parse={row['parse_error_rate']:.2%}, inactive={row['avg_inactive_ratio']:.4f}"
        )
    lines.extend(['', '## Layers', ''])
    for layer_key, row in sorted(profile['by_layer'].items(), key=lambda item: int(item[0].replace('layer_', ''))):
        active95 = row['active_mass'].get('95pct', {})
        inactive95 = row['inactive_mass'].get('95pct', {})
        lines.append(
            f"- `{layer_key}`: resident={row.get('resident_expert_count', 'n/a')}, avg_inactive={row['avg_inactive_ratio']:.4f}, "
            f"active95={active95.get('expert_count', 0)}, inactive95={inactive95.get('expert_count', 0)}, "
            f"active_overlap={row['active_prompt_overlap_jaccard']:.4f}, inactive_overlap={row['inactive_prompt_overlap_jaccard']:.4f}"
        )
        active_top = ', '.join(f"{entry['expert']}:{entry['mass']:.3f}" for entry in row['active_mass'].get('top_experts_by_mass', [])[:8])
        inactive_top = ', '.join(f"{entry['expert']}:{entry['mass']:.3f}" for entry in row['inactive_mass'].get('top_experts_by_mass', [])[:8])
        lines.append(f"  - active top mass experts: {active_top or 'none'}")
        lines.append(f"  - inactive top mass experts: {inactive_top or 'none'}")
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile per-layer expert utilization from dynamic.json router activity payloads.')
    parser.add_argument('--dynamic-json', required=True)
    parser.add_argument('--plan-json')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    dynamic = load_json(args.dynamic_json)
    plan = load_json(args.plan_json) if args.plan_json else None
    profile = summarize_dynamic_router_activity(dynamic, plan=plan)
    Path(args.output_json).write_text(json.dumps(profile, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    Path(args.output_md).write_text(build_markdown(profile, plan_path=args.plan_json, dynamic_path=args.dynamic_json), encoding='utf-8')
    print(json.dumps({
        'output_json': str(Path(args.output_json).resolve()),
        'output_md': str(Path(args.output_md).resolve()),
        'result_count': profile['result_count'],
        'overall': profile['overall'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
