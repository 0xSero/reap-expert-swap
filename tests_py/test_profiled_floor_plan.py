from __future__ import annotations

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_profiled_floor_plan import build_profiled_floor_plan


class ProfiledFloorPlanTests(unittest.TestCase):
    def test_build_profiled_floor_plan_replaces_core_experts_and_recomputes_budget(self) -> None:
        base_plan = {
            'mode': 'dynamic_core_specialist',
            'selectionMode': 'dynamic_exact_floor',
            'budget': {
                'per_expert_bytes': 2,
                'always_resident_bytes': 10,
                'lowbit_budget_bytes': 0,
                'full_bf16_gib': 1.0,
                'total_exact_experts': 4,
                'core_budget_bytes': 8,
                'swappable_expert_budget_bytes': 8,
                'max_resident_gib': 0.0,
                'max_resident_ratio': 0.0,
            },
            'perLayer': {
                'layer_0': {'numExperts': 8, 'coreExperts': [0, 1], 'sliceCatalog': []},
                'layer_1': {'numExperts': 8, 'coreExperts': [0, 1], 'sliceCatalog': []},
            },
            'summary': {},
            'scorerArtifacts': {},
        }
        profile = {
            'by_layer': {
                'layer_0': {
                    'active_mass': {'90pct': {'experts': [1, 2]}},
                    'inactive_mass': {'80pct': {'experts': [6]}},
                },
                'layer_1': {
                    'active_mass': {'90pct': {'experts': [3]}},
                    'inactive_mass': {'80pct': {'experts': [4, 5]}},
                },
            }
        }
        plan = build_profiled_floor_plan(
            base_plan,
            profile,
            active_threshold='90pct',
            inactive_threshold='80pct',
            profile_path='profile.json',
        )
        self.assertEqual(plan['perLayer']['layer_0']['coreExperts'], [1, 2, 6])
        self.assertEqual(plan['perLayer']['layer_1']['coreExperts'], [3, 4, 5])
        self.assertEqual(plan['budget']['total_exact_experts'], 6)
        self.assertEqual(plan['budget']['core_budget_bytes'], 12)
        self.assertEqual(plan['scorerArtifacts']['profiledFloorSelection']['activeThreshold'], '90pct')
        self.assertEqual(plan['summary']['residentFractionPctAvg'], 37.5)


if __name__ == '__main__':
    unittest.main()
