from __future__ import annotations

from typing import Any


def parse_layer_idx(layer_key: str | int) -> int:
    text = str(layer_key)
    if text.startswith("layer_"):
        text = text.split("layer_", 1)[1]
    return int(text)


def infer_num_experts(layer_plan: dict[str, Any]) -> int:
    explicit = layer_plan.get("numExperts")
    if explicit is not None:
        return int(explicit)
    experts = set(int(expert) for expert in layer_plan.get("coreExperts", []))
    for slice_row in layer_plan.get("sliceCatalog", []):
        experts.update(int(expert) for expert in slice_row.get("experts", []))
    return (max(experts) + 1) if experts else 0


def build_dense_keep_sets(plan: dict[str, Any]) -> dict[int, set[int]]:
    dense_keep_sets: dict[int, set[int]] = {}
    for layer_key, layer_plan in (plan.get("perLayer") or {}).items():
        layer_idx = parse_layer_idx(layer_key)
        num_experts = infer_num_experts(layer_plan)
        dense_keep_sets[layer_idx] = set(range(max(0, num_experts)))
    return dense_keep_sets


def compute_keep_set_delta(
    current_keep_sets: dict[int, set[int]] | None,
    desired_keep_sets: dict[int, set[int]],
    dense_keep_sets: dict[int, set[int]],
) -> dict[str, Any]:
    current_keep_sets = current_keep_sets or {}
    all_layers = sorted(set(dense_keep_sets) | set(current_keep_sets) | set(desired_keep_sets))
    by_layer: dict[str, dict[str, Any]] = {}
    added_total = 0
    removed_total = 0
    reused_total = 0
    desired_total = 0
    current_total = 0
    for layer_idx in all_layers:
        baseline = set(dense_keep_sets.get(layer_idx, set()))
        current = set(current_keep_sets.get(layer_idx, baseline))
        desired = set(desired_keep_sets.get(layer_idx, set()))
        added = sorted(desired - current)
        removed = sorted(current - desired)
        reused = sorted(current & desired)
        key = f"layer_{layer_idx}"
        by_layer[key] = {
            "current": sorted(current),
            "desired": sorted(desired),
            "added": added,
            "removed": removed,
            "reused": reused,
        }
        added_total += len(added)
        removed_total += len(removed)
        reused_total += len(reused)
        desired_total += len(desired)
        current_total += len(current)
    return {
        "by_layer": by_layer,
        "summary": {
            "layer_count": len(all_layers),
            "current_expert_total": current_total,
            "desired_expert_total": desired_total,
            "added_expert_total": added_total,
            "removed_expert_total": removed_total,
            "reused_expert_total": reused_total,
        },
    }
