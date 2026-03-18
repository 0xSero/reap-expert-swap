"""Delta computation for REAP-swap expert keep-set transitions.

Computes the minimal set of expert additions and removals needed to move
from the current GPU-resident expert set to a new desired set, given the
dense (all-experts) baseline as reference.
"""

from typing import Any


def build_dense_keep_sets(plan: dict[str, Any]) -> dict[int, set[int]]:
    """Build per-layer keep sets containing ALL experts (the dense baseline).

    This is the "no pruning" reference: every expert is kept. Used as the
    starting point before any active-set selection narrows the resident set.

    Args:
        plan: The REAP plan dict with a ``perLayer`` key mapping layer keys
              (e.g. ``"layer_0"``) to layer objects. Each layer object must
              contain a ``coreExperts`` list (the always-resident experts)
              and a ``sliceCatalog`` list of specialist slices, each with
              an ``experts`` list.

    Returns:
        Dict mapping integer layer indices to sets of all known expert
        indices for that layer (core + every specialist slice expert).
    """
    per_layer = plan.get("perLayer", {})
    dense: dict[int, set[int]] = {}
    for layer_key, layer_obj in per_layer.items():
        try:
            layer_idx = int(str(layer_key).replace("layer_", ""))
        except ValueError:
            continue
        experts: set[int] = set()
        for idx in layer_obj.get("coreExperts", []):
            experts.add(int(idx))
        for slice_entry in layer_obj.get("sliceCatalog", []):
            for idx in slice_entry.get("experts", []):
                experts.add(int(idx))
        dense[layer_idx] = experts
    return dense


def compute_keep_set_delta(
    *,
    current_keep_sets: dict[int, set[int]],
    desired_keep_sets: dict[int, set[int]],
    dense_keep_sets: dict[int, set[int]],
) -> dict[str, Any]:
    """Compute the per-layer delta between current and desired keep sets.

    For each layer, determines which experts need to be added (present in
    desired but not current) and which need to be removed (present in
    current but not desired). Experts present in both are reused.

    Args:
        current_keep_sets: Per-layer sets of currently GPU-resident experts.
        desired_keep_sets: Per-layer sets of experts the new active set wants.
        dense_keep_sets: Per-layer sets of ALL experts (dense baseline).

    Returns:
        Dict with ``by_layer`` (per-layer added/removed/reused lists) and
        ``summary`` (aggregate counts across all layers).
    """
    all_layers = sorted(
        set(current_keep_sets.keys())
        | set(desired_keep_sets.keys())
        | set(dense_keep_sets.keys())
    )
    by_layer: dict[str, dict[str, Any]] = {}
    total_added = 0
    total_removed = 0
    total_reused = 0

    for layer_idx in all_layers:
        current = current_keep_sets.get(layer_idx, set())
        desired = desired_keep_sets.get(layer_idx, set())
        added = sorted(desired - current)
        removed = sorted(current - desired)
        reused = sorted(current & desired)
        layer_key = f"layer_{layer_idx}"
        by_layer[layer_key] = {
            "added": added,
            "removed": removed,
            "reused": reused,
            "added_count": len(added),
            "removed_count": len(removed),
            "reused_count": len(reused),
        }
        total_added += len(added)
        total_removed += len(removed)
        total_reused += len(reused)

    return {
        "by_layer": by_layer,
        "summary": {
            "added_expert_total": total_added,
            "removed_expert_total": total_removed,
            "reused_expert_total": total_reused,
            "layers_touched": sum(
                1
                for v in by_layer.values()
                if v["added_count"] > 0 or v["removed_count"] > 0
            ),
        },
    }
