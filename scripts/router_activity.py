#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any


ROUTER_ACTIVITY_MASS_PRECISION = 8


def empty_layer_activity() -> dict[str, Any]:
    return {
        'inactive_mass': 0.0,
        'observed_mass': 0.0,
        'inactive_experts': [],
        'active_mass': 0.0,
        'active_experts': [],
        'observed_token_count': 0,
        'route_decision_count': 0,
        'active_expert_counts': {},
        'active_expert_mass': {},
        'inactive_expert_counts': {},
        'inactive_expert_mass': {},
    }


def _merge_metric_map(target: dict[str, Any], key: str, source_values: dict[str, Any], *, cast=float) -> None:
    bucket = target.setdefault(key, {})
    for expert, value in (source_values or {}).items():
        expert_key = str(int(expert))
        bucket[expert_key] = cast(bucket.get(expert_key, 0)) + cast(value)


def merge_layer_activity(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    target['inactive_mass'] = float(target.get('inactive_mass', 0.0) or 0.0) + float(source.get('inactive_mass', 0.0) or 0.0)
    target['observed_mass'] = float(target.get('observed_mass', 0.0) or 0.0) + float(source.get('observed_mass', 0.0) or 0.0)
    target['active_mass'] = float(target.get('active_mass', 0.0) or 0.0) + float(source.get('active_mass', 0.0) or 0.0)
    target['observed_token_count'] = int(target.get('observed_token_count', 0) or 0) + int(source.get('observed_token_count', 0) or 0)
    target['route_decision_count'] = int(target.get('route_decision_count', 0) or 0) + int(source.get('route_decision_count', 0) or 0)
    target['inactive_experts'] = sorted(set(target.get('inactive_experts', [])) | {int(expert) for expert in source.get('inactive_experts', [])})
    target['active_experts'] = sorted(set(target.get('active_experts', [])) | {int(expert) for expert in source.get('active_experts', [])})
    _merge_metric_map(target, 'active_expert_counts', source.get('active_expert_counts', {}), cast=int)
    _merge_metric_map(target, 'inactive_expert_counts', source.get('inactive_expert_counts', {}), cast=int)
    _merge_metric_map(target, 'active_expert_mass', source.get('active_expert_mass', {}), cast=float)
    _merge_metric_map(target, 'inactive_expert_mass', source.get('inactive_expert_mass', {}), cast=float)
    return target


def collapse_topk_router_activity(
    top_indices_batches: list[list[int]],
    top_probs_batches: list[list[float]],
    *,
    keep_set: set[int],
    local_to_global: dict[int, int] | None = None,
) -> dict[str, Any]:
    local_to_global = local_to_global or {}
    payload = empty_layer_activity()
    active_counts: dict[str, int] = defaultdict(int)
    active_mass: dict[str, float] = defaultdict(float)
    inactive_counts: dict[str, int] = defaultdict(int)
    inactive_mass: dict[str, float] = defaultdict(float)
    active_experts: set[int] = set()
    inactive_experts: set[int] = set()

    for token_indices, token_probs in zip(top_indices_batches, top_probs_batches, strict=False):
        payload['observed_token_count'] += 1
        for expert_idx, prob in zip(token_indices, token_probs, strict=False):
            global_expert = int(local_to_global.get(int(expert_idx), int(expert_idx)))
            weight = float(prob)
            payload['observed_mass'] += weight
            payload['route_decision_count'] += 1
            if global_expert in keep_set:
                payload['active_mass'] += weight
                active_experts.add(global_expert)
                active_counts[str(global_expert)] += 1
                active_mass[str(global_expert)] += weight
            else:
                payload['inactive_mass'] += weight
                inactive_experts.add(global_expert)
                inactive_counts[str(global_expert)] += 1
                inactive_mass[str(global_expert)] += weight

    payload['active_experts'] = sorted(active_experts)
    payload['inactive_experts'] = sorted(inactive_experts)
    payload['active_expert_counts'] = dict(sorted(active_counts.items(), key=lambda item: int(item[0])))
    payload['inactive_expert_counts'] = dict(sorted(inactive_counts.items(), key=lambda item: int(item[0])))
    payload['active_expert_mass'] = {
        key: round(value, ROUTER_ACTIVITY_MASS_PRECISION)
        for key, value in sorted(active_mass.items(), key=lambda item: int(item[0]))
    }
    payload['inactive_expert_mass'] = {
        key: round(value, ROUTER_ACTIVITY_MASS_PRECISION)
        for key, value in sorted(inactive_mass.items(), key=lambda item: int(item[0]))
    }
    payload['inactive_mass'] = round(payload['inactive_mass'], ROUTER_ACTIVITY_MASS_PRECISION)
    payload['active_mass'] = round(payload['active_mass'], ROUTER_ACTIVITY_MASS_PRECISION)
    payload['observed_mass'] = round(payload['observed_mass'], ROUTER_ACTIVITY_MASS_PRECISION)
    return payload


def aggregate_router_activity_results(worker_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_layer: dict[str, dict[str, Any]] = {}
    for result in worker_results:
        payload = result.get('payload', {}) or {}
        for layer_key, layer_row in (payload.get('by_layer') or {}).items():
            target = by_layer.setdefault(layer_key, empty_layer_activity())
            merge_layer_activity(target, layer_row)
    for layer_row in by_layer.values():
        layer_row['inactive_mass'] = round(float(layer_row.get('inactive_mass', 0.0) or 0.0), ROUTER_ACTIVITY_MASS_PRECISION)
        layer_row['active_mass'] = round(float(layer_row.get('active_mass', 0.0) or 0.0), ROUTER_ACTIVITY_MASS_PRECISION)
        layer_row['observed_mass'] = round(float(layer_row.get('observed_mass', 0.0) or 0.0), ROUTER_ACTIVITY_MASS_PRECISION)
        layer_row['active_expert_mass'] = {
            key: round(float(value), ROUTER_ACTIVITY_MASS_PRECISION)
            for key, value in sorted((layer_row.get('active_expert_mass') or {}).items(), key=lambda item: int(item[0]))
        }
        layer_row['inactive_expert_mass'] = {
            key: round(float(value), ROUTER_ACTIVITY_MASS_PRECISION)
            for key, value in sorted((layer_row.get('inactive_expert_mass') or {}).items(), key=lambda item: int(item[0]))
        }
        layer_row['active_expert_counts'] = {
            key: int(value)
            for key, value in sorted((layer_row.get('active_expert_counts') or {}).items(), key=lambda item: int(item[0]))
        }
        layer_row['inactive_expert_counts'] = {
            key: int(value)
            for key, value in sorted((layer_row.get('inactive_expert_counts') or {}).items(), key=lambda item: int(item[0]))
        }
        layer_row['active_experts'] = sorted(int(expert) for expert in layer_row.get('active_experts', []))
        layer_row['inactive_experts'] = sorted(int(expert) for expert in layer_row.get('inactive_experts', []))
    return by_layer


def coverage_from_mass_map(expert_mass: dict[str, float], thresholds: tuple[float, ...] = (0.8, 0.9, 0.95)) -> dict[str, Any]:
    masses = [(str(expert), float(value)) for expert, value in (expert_mass or {}).items() if float(value) > 0.0]
    masses.sort(key=lambda item: item[1], reverse=True)
    total = sum(value for _expert, value in masses)
    out: dict[str, Any] = {'total_mass': round(total, ROUTER_ACTIVITY_MASS_PRECISION)}
    running = 0.0
    cursor = 0
    for threshold in thresholds:
        while cursor < len(masses) and total > 0 and running / total < threshold:
            running += masses[cursor][1]
            cursor += 1
        out[f'{int(threshold * 100)}pct'] = {
            'expert_count': cursor,
            'experts': [int(expert) for expert, _value in masses[:cursor]],
            'covered_mass': round(running, ROUTER_ACTIVITY_MASS_PRECISION),
            'covered_fraction': round((running / total) if total > 0 else 0.0, ROUTER_ACTIVITY_MASS_PRECISION),
        }
    out['top_experts_by_mass'] = [
        {'expert': int(expert), 'mass': round(value, ROUTER_ACTIVITY_MASS_PRECISION)}
        for expert, value in masses[:16]
    ]
    return out


def average_pairwise_jaccard(sets: list[set[int]]) -> float:
    if len(sets) <= 1:
        return 1.0 if sets else 0.0
    scores: list[float] = []
    for idx in range(len(sets)):
        for other_idx in range(idx + 1, len(sets)):
            left = sets[idx]
            right = sets[other_idx]
            union = left | right
            scores.append((len(left & right) / len(union)) if union else 1.0)
    return round(fmean(scores) if scores else 0.0, ROUTER_ACTIVITY_MASS_PRECISION)
