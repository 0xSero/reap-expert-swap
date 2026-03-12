#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_profiled_floor_plan(
    base_plan: dict[str, Any],
    profile: dict[str, Any],
    *,
    active_threshold: str,
    inactive_threshold: str,
    profile_path: str | None = None,
) -> dict[str, Any]:
    if str(base_plan.get('selectionMode') or base_plan.get('mode')) != 'dynamic_exact_floor':
        raise ValueError('base plan must be an exact-floor plan')
    plan = json.loads(json.dumps(base_plan))
    per_layer = plan.get('perLayer') or {}
    for layer_key, layer in per_layer.items():
        profile_layer = (profile.get('by_layer') or {}).get(layer_key)
        if not profile_layer:
            continue
        active_set = set((profile_layer.get('active_mass') or {}).get(active_threshold, {}).get('experts', []))
        inactive_set = set((profile_layer.get('inactive_mass') or {}).get(inactive_threshold, {}).get('experts', []))
        merged = sorted(int(expert) for expert in (active_set | inactive_set))
        layer['coreExperts'] = merged
        layer['coreActivationMass'] = None
        layer['coreByteCost'] = len(merged) * int(plan['budget']['per_expert_bytes'])
    budget = plan.get('budget') or {}
    total_core = sum(len(layer.get('coreExperts') or []) for layer in per_layer.values())
    per_expert_bytes = int(budget.get('per_expert_bytes', 0) or 0)
    always_resident_bytes = int(budget.get('always_resident_bytes', 0) or 0)
    lowbit_budget_bytes = int(budget.get('lowbit_budget_bytes', 0) or 0)
    full_bf16_gib = float(budget.get('full_bf16_gib', 0.0) or 0.0)
    full_bytes = int(round(full_bf16_gib * (1024 ** 3)))
    actual_exact_bytes = total_core * per_expert_bytes
    budget['total_exact_experts'] = int(total_core)
    budget['core_budget_bytes'] = int(actual_exact_bytes)
    budget['swappable_expert_budget_bytes'] = int(actual_exact_bytes)
    budget['exact_experts_per_layer_target'] = int(round(total_core / max(1, len(per_layer))))
    budget['max_resident_gib'] = round((always_resident_bytes + actual_exact_bytes + lowbit_budget_bytes) / (1024 ** 3), 6)
    budget['max_resident_ratio'] = round((always_resident_bytes + actual_exact_bytes + lowbit_budget_bytes) / full_bytes, 6) if full_bytes else 0.0
    plan['budget'] = budget
    plan['summary'] = {
        'layerCount': len(per_layer),
        'totalCoreExperts': total_core,
        'totalSlices': 0,
        'residentFractionPctMin': round(100.0 * min(len(layer['coreExperts']) / max(1, layer['numExperts']) for layer in per_layer.values()), 8),
        'residentFractionPctMax': round(100.0 * max(len(layer['coreExperts']) / max(1, layer['numExperts']) for layer in per_layer.values()), 8),
        'residentFractionPctAvg': round(100.0 * total_core / max(1, sum(layer['numExperts'] for layer in per_layer.values())), 8),
    }
    scorer = plan.setdefault('scorerArtifacts', {})
    scorer['profiledFloorSelection'] = {
        'profilePath': profile_path,
        'activeThreshold': active_threshold,
        'inactiveThreshold': inactive_threshold,
    }
    return plan


def build_markdown(plan: dict[str, Any], *, active_threshold: str, inactive_threshold: str, base_plan_path: str, profile_path: str) -> str:
    budget = plan['budget']
    summary = plan['summary']
    return '\n'.join([
        '# Profiled exact-floor plan',
        '',
        f'- base plan: `{base_plan_path}`',
        f'- profile: `{profile_path}`',
        f'- active threshold: `{active_threshold}`',
        f'- inactive threshold: `{inactive_threshold}`',
        f"- total exact experts: **{budget['total_exact_experts']}**",
        f"- resident GiB: **{budget['max_resident_gib']:.3f}**",
        f"- resident ratio: **{budget['max_resident_ratio']:.2%}**",
        f"- avg per-layer resident fraction: **{summary['residentFractionPctAvg']:.2f}%**",
        f"- min per-layer resident fraction: **{summary['residentFractionPctMin']:.2f}%**",
        f"- max per-layer resident fraction: **{summary['residentFractionPctMax']:.2f}%**",
        '',
    ]) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a profile-derived exact-floor plan from router activity.')
    parser.add_argument('--base-plan-json', required=True)
    parser.add_argument('--profile-json', required=True)
    parser.add_argument('--active-threshold', required=True, choices=('80pct', '90pct', '95pct'))
    parser.add_argument('--inactive-threshold', required=True, choices=('80pct', '90pct', '95pct'))
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    base_plan = load_json(args.base_plan_json)
    profile = load_json(args.profile_json)
    plan = build_profiled_floor_plan(
        base_plan,
        profile,
        active_threshold=args.active_threshold,
        inactive_threshold=args.inactive_threshold,
        profile_path=args.profile_json,
    )
    Path(args.output_json).write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    Path(args.output_md).write_text(build_markdown(plan, active_threshold=args.active_threshold, inactive_threshold=args.inactive_threshold, base_plan_path=args.base_plan_json, profile_path=args.profile_json), encoding='utf-8')
    print(json.dumps({
        'output_json': str(Path(args.output_json).resolve()),
        'output_md': str(Path(args.output_md).resolve()),
        'resident_gib': plan['budget']['max_resident_gib'],
        'resident_ratio': plan['budget']['max_resident_ratio'],
        'total_exact_experts': plan['budget']['total_exact_experts'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
