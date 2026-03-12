from __future__ import annotations

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from router_activity import aggregate_router_activity_results, collapse_topk_router_activity, coverage_from_mass_map
from profile_router_activity import summarize_dynamic_router_activity


class RouterActivityTests(unittest.TestCase):
    def test_collapse_topk_router_activity_tracks_active_and_inactive_metrics(self) -> None:
        payload = collapse_topk_router_activity(
            [[0, 2], [1, 3]],
            [[0.7, 0.3], [0.6, 0.4]],
            keep_set={0, 1},
        )
        self.assertAlmostEqual(payload['observed_mass'], 2.0)
        self.assertAlmostEqual(payload['active_mass'], 1.3)
        self.assertAlmostEqual(payload['inactive_mass'], 0.7)
        self.assertEqual(payload['active_experts'], [0, 1])
        self.assertEqual(payload['inactive_experts'], [2, 3])
        self.assertEqual(payload['active_expert_counts']['0'], 1)
        self.assertEqual(payload['inactive_expert_counts']['3'], 1)

    def test_aggregate_router_activity_results_merges_worker_maps(self) -> None:
        worker_results = [
            {
                'payload': {
                    'by_layer': {
                        'layer_0': {
                            'inactive_mass': 0.2,
                            'observed_mass': 1.0,
                            'inactive_experts': [2],
                            'active_mass': 0.8,
                            'active_experts': [0],
                            'observed_token_count': 1,
                            'route_decision_count': 2,
                            'active_expert_counts': {'0': 1},
                            'active_expert_mass': {'0': 0.8},
                            'inactive_expert_counts': {'2': 1},
                            'inactive_expert_mass': {'2': 0.2},
                        }
                    }
                }
            },
            {
                'payload': {
                    'by_layer': {
                        'layer_0': {
                            'inactive_mass': 0.1,
                            'observed_mass': 1.0,
                            'inactive_experts': [3],
                            'active_mass': 0.9,
                            'active_experts': [1],
                            'observed_token_count': 1,
                            'route_decision_count': 2,
                            'active_expert_counts': {'1': 1},
                            'active_expert_mass': {'1': 0.9},
                            'inactive_expert_counts': {'3': 1},
                            'inactive_expert_mass': {'3': 0.1},
                        }
                    }
                }
            },
        ]
        merged = aggregate_router_activity_results(worker_results)
        layer = merged['layer_0']
        self.assertAlmostEqual(layer['inactive_mass'], 0.3)
        self.assertAlmostEqual(layer['active_mass'], 1.7)
        self.assertEqual(layer['active_experts'], [0, 1])
        self.assertEqual(layer['inactive_experts'], [2, 3])
        self.assertEqual(layer['active_expert_counts']['1'], 1)
        self.assertEqual(layer['inactive_expert_counts']['3'], 1)

    def test_coverage_from_mass_map_returns_minimal_prefix(self) -> None:
        summary = coverage_from_mass_map({'7': 0.6, '2': 0.25, '9': 0.15})
        self.assertEqual(summary['80pct']['expert_count'], 2)
        self.assertEqual(summary['95pct']['experts'], [7, 2, 9])

    def test_profile_router_activity_summarizes_layer_coverage(self) -> None:
        dynamic_payload = {
            'results': [
                {
                    'id': 'row-1',
                    'request_id': 'req-1',
                    'benchmark': 'gsm8k',
                    'question': 'Solve 2+2',
                    'correct': True,
                    'coherent': True,
                    'parse_error': False,
                    'router_miss_summary': {'inactive_ratio': 0.2},
                    'router_misses': {
                        'by_layer': {
                            'layer_0': {
                                'inactive_mass': 0.2,
                                'observed_mass': 1.0,
                                'inactive_experts': [4],
                                'active_mass': 0.8,
                                'active_experts': [0, 1],
                                'observed_token_count': 1,
                                'route_decision_count': 2,
                                'active_expert_counts': {'0': 1, '1': 1},
                                'active_expert_mass': {'0': 0.5, '1': 0.3},
                                'inactive_expert_counts': {'4': 1},
                                'inactive_expert_mass': {'4': 0.2},
                            }
                        }
                    },
                },
                {
                    'id': 'row-2',
                    'request_id': 'req-2',
                    'benchmark': 'arc_challenge',
                    'question': 'Reason about this',
                    'correct': False,
                    'coherent': True,
                    'parse_error': False,
                    'router_miss_summary': {'inactive_ratio': 0.4},
                    'router_misses': {
                        'by_layer': {
                            'layer_0': {
                                'inactive_mass': 0.4,
                                'observed_mass': 1.0,
                                'inactive_experts': [4, 5],
                                'active_mass': 0.6,
                                'active_experts': [1],
                                'observed_token_count': 1,
                                'route_decision_count': 2,
                                'active_expert_counts': {'1': 1},
                                'active_expert_mass': {'1': 0.6},
                                'inactive_expert_counts': {'4': 1, '5': 1},
                                'inactive_expert_mass': {'4': 0.25, '5': 0.15},
                            }
                        }
                    },
                },
            ]
        }
        plan = {
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0, 1, 2, 3],
                    'numExperts': 8,
                }
            }
        }
        profile = summarize_dynamic_router_activity(dynamic_payload, plan=plan)
        layer = profile['by_layer']['layer_0']
        self.assertEqual(profile['result_count'], 2)
        self.assertEqual(layer['resident_expert_count'], 4)
        self.assertEqual(layer['active_mass']['95pct']['expert_count'], 2)
        self.assertEqual(layer['inactive_mass']['95pct']['experts'], [4, 5])
        self.assertAlmostEqual(profile['overall']['active_95pct_fraction_of_resident'], 0.5)


if __name__ == '__main__':
    unittest.main()
