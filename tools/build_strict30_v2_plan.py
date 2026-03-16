#!/usr/bin/env python3
"""Transform strict30-best-plan.json (v1) into strict30-v2-plan.json.

Changes applied:
  - rotationPolicy: "none" -> "late_prompt_hash"
  - budget split: 75/25 core/specialist -> 50/50
  - per-layer specialist/candidate targets recalculated
  - supportEstimatorConfig added with prefill_reserve mode
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def transform(plan: dict) -> dict:
    v2 = copy.deepcopy(plan)

    # Primary fix: enable rotation (root cause #1)
    v2["rotationPolicy"] = "late_prompt_hash"

    sa = v2.setdefault("scorerArtifacts", {})
    sa["rotationPolicy"] = "late_prompt_hash"

    # Forward-looking: enable support estimator pathway
    sa["supportEstimatorConfig"] = {
        "mode": "prefill_reserve",
        "reserve_fraction": 0.30,
    }

    # Budget stays the same -- core expert lists are baked at 49/layer from the
    # observer and can't shrink without new observer runs. The rotation mechanism
    # works with the existing budget: rotation_window = min(max(2, candidate_target), 4)
    # = min(max(2, 20), 4) = 4. With window=4, rotation can pick different top-2
    # slices from the top-4 candidates per late layer.

    return v2


def verify(v2: dict) -> None:
    budget = v2["budget"]
    per_expert = int(budget["per_expert_bytes"])
    swappable = int(budget["swappable_expert_budget_bytes"])
    total_active_bytes = 0
    for layer_key, layer in v2["perLayer"].items():
        core_bytes = len(layer["coreExperts"]) * per_expert
        specialist_bytes = layer["specialistBudgetBytes"]
        total_active_bytes += core_bytes + specialist_bytes
        candidate_target = layer.get("specialistCandidateExpertTarget", 0)
        rotation_window = min(max(2, candidate_target), 4, len(layer.get("sliceCatalog", [])))
        assert rotation_window >= 2, f"{layer_key}: rotation_window={rotation_window} too small"
    assert total_active_bytes <= swappable, \
        f"total active {total_active_bytes} exceeds swappable budget {swappable}"
    assert v2["rotationPolicy"] == "late_prompt_hash"
    sa = v2.get("scorerArtifacts", {})
    sec = sa.get("supportEstimatorConfig", {})
    assert sec.get("mode") == "prefill_reserve"
    assert abs(sec.get("reserve_fraction", 0) - 0.30) < 0.001


def main() -> int:
    v1_path = REPO_ROOT / "assets" / "strict30-best-plan.json"
    v2_path = REPO_ROOT / "assets" / "strict30-v2-plan.json"

    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = transform(v1)
    verify(v2)

    v2_text = json.dumps(v2, indent=2) + "\n"
    v2_path.write_text(v2_text, encoding="utf-8")

    sha = hashlib.sha256(v2_text.encode("utf-8")).hexdigest()[:16]
    budget = v2["budget"]
    print(f"wrote {v2_path}")
    print(f"  sha256: {sha}")
    print(f"  rotationPolicy: {v2['rotationPolicy']}")
    print(f"  core/specialist fractions: {budget['core_budget_fraction']}/{budget['specialist_budget_fraction']} (unchanged)")
    layer0 = v2["perLayer"]["layer_0"]
    candidate_target = layer0.get("specialistCandidateExpertTarget", 0)
    rotation_window = min(max(2, candidate_target), 4, len(layer0.get("sliceCatalog", [])))
    print(f"  specialist_per_layer: {layer0['specialistActiveExpertTarget']}")
    print(f"  candidate_per_layer: {layer0['specialistCandidateExpertTarget']}")
    print(f"  rotation_window: {rotation_window}")
    n_slices = layer0['specialistBudgetBytes'] / (8 * budget['per_expert_bytes'])
    print(f"  slices selected per layer: {n_slices:.2f}")
    print(f"  supportEstimatorConfig.mode: {v2['scorerArtifacts']['supportEstimatorConfig']['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
