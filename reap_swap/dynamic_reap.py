"""REAP-swap plan validation, hashing, and router-miss summarization.

These utilities are used by the multiplex server to validate incoming
active-set payloads against the loaded REAP plan, compute deterministic
plan hashes for identity tracking, and summarize per-request router miss
instrumentation data.
"""

import hashlib
import json
from typing import Any


def compute_plan_sha256(plan: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of the REAP plan.

    Serializes the plan to JSON with sorted keys and returns the hex
    digest. Used for plan identity verification -- ensures the server
    and client agree on which plan is loaded.

    Args:
        plan: The full REAP plan dict.

    Returns:
        Hex string of the SHA-256 hash.
    """
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def summarize_router_misses(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-layer router miss data into a single summary.

    Router miss tracking counts how much activation mass flows to experts
    outside the current keep set. This tells you whether the active-set
    selection is actually covering the experts the model wants to use.

    Args:
        payload: Dict with a ``by_layer`` key mapping layer keys to dicts
                 containing ``inactive_mass``, ``observed_mass``, and
                 ``inactive_experts``.

    Returns:
        Dict with ``inactive_mass_total``, ``observed_mass_total``,
        ``inactive_ratio``, and ``inactive_expert_total``.
    """
    by_layer = payload.get("by_layer", {})
    inactive_mass = 0.0
    observed_mass = 0.0
    all_inactive: set[int] = set()

    for layer_data in by_layer.values():
        inactive_mass += float(layer_data.get("inactive_mass", 0.0))
        observed_mass += float(layer_data.get("observed_mass", 0.0))
        for expert_idx in layer_data.get("inactive_experts", []):
            all_inactive.add(int(expert_idx))

    inactive_ratio = inactive_mass / observed_mass if observed_mass > 0 else 0.0

    return {
        "inactive_mass_total": inactive_mass,
        "observed_mass_total": observed_mass,
        "inactive_ratio": inactive_ratio,
        "inactive_expert_total": len(all_inactive),
    }


def validate_active_set_payload(
    payload: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize an active-set swap request payload.

    Checks that the payload contains the required fields (request_id,
    phase, active_set, budget_bytes), that layer keys in the active set
    match layers in the plan, and that requested experts exist in the
    plan's core + specialist catalog for each layer.

    Also computes a deterministic ``active_set_signature`` from the
    requested expert sets and attaches ``union_validation`` and
    ``core_presence_summary`` metadata.

    Args:
        payload: The incoming swap request body.
        plan: The loaded REAP plan dict.

    Returns:
        The validated payload dict, augmented with computed fields:
        ``active_set_signature``, ``union_validation``,
        ``core_presence_summary``, ``selected_slice_ids``.

    Raises:
        ValueError: If required fields are missing or layer/expert
                    references don't match the plan.
    """
    request_id = payload.get("request_id")
    if not request_id:
        raise ValueError("active set payload must include request_id")

    phase = payload.get("phase")
    if not phase:
        raise ValueError("active set payload must include phase")

    active_set = payload.get("active_set")
    if not isinstance(active_set, dict) or not active_set:
        raise ValueError("active set payload must include non-empty active_set dict")

    budget_bytes = payload.get("budget_bytes", 0)

    per_layer = plan.get("perLayer", {})
    validated_set: dict[str, list[int]] = {}
    core_present = 0
    core_total = 0
    selected_slices: dict[str, list[str]] = {}

    for layer_key, experts in active_set.items():
        normalized_key = (
            layer_key if layer_key.startswith("layer_") else f"layer_{layer_key}"
        )
        layer_plan = per_layer.get(normalized_key)
        if layer_plan is None:
            raise ValueError(f"active set references unknown layer: {layer_key}")
        core_experts = set(int(i) for i in layer_plan.get("coreExperts", []))
        all_known = set(core_experts)
        slice_catalog = layer_plan.get("sliceCatalog", [])
        for s in slice_catalog:
            for idx in s.get("experts", []):
                all_known.add(int(idx))

        expert_list = [int(e) for e in experts]
        unknown = set(expert_list) - all_known
        if unknown:
            raise ValueError(
                f"layer {layer_key} references unknown experts: {sorted(unknown)}"
            )
        validated_set[normalized_key] = expert_list

        present = sum(1 for e in expert_list if e in core_experts)
        core_present += present
        core_total += len(core_experts)

        layer_slices = []
        for s in slice_catalog:
            slice_experts = set(int(i) for i in s.get("experts", []))
            if slice_experts and slice_experts.issubset(set(expert_list)):
                slice_id = s.get("sliceId", s.get("id", "unknown"))
                layer_slices.append(str(slice_id))
        if layer_slices:
            selected_slices[normalized_key] = layer_slices

    # Compute deterministic signature from the validated active set
    sig_parts = []
    for lk in sorted(validated_set.keys()):
        sig_parts.append(f"{lk}:{','.join(str(e) for e in sorted(validated_set[lk]))}")
    sig_str = "|".join(sig_parts)
    signature = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:16]

    return {
        "request_id": str(request_id),
        "phase": str(phase),
        "active_set": validated_set,
        "budget_bytes": int(budget_bytes),
        "active_set_signature": signature,
        "selected_slice_ids": selected_slices,
        "union_validation": {
            "layers_validated": len(validated_set),
            "total_experts_requested": sum(len(v) for v in validated_set.values()),
        },
        "core_presence_summary": {
            "core_present": core_present,
            "core_total": core_total,
            "core_coverage": core_present / core_total if core_total > 0 else 0.0,
        },
    }
