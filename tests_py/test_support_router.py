from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from support_router import derive_slice_targets, dataset_rows_from_dynamic_payload, resolve_plan_for_dynamic_artifact


class SupportRouterTests(unittest.TestCase):
    def test_resolve_plan_for_dynamic_artifact_handles_repo_relative_plan_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            smoke_dir = repo_root / 'test-output' / 'autoresearch' / 'demo-smoke'
            smoke_dir.mkdir(parents=True)
            plan_path = repo_root / 'test-output' / 'plans' / 'demo-plan.json'
            plan_path.parent.mkdir(parents=True)
            plan_payload = {'mode': 'dynamic_core_specialist', 'perLayer': {}}
            plan_path.write_text(json.dumps(plan_payload), encoding='utf-8')
            (smoke_dir / 'run-summary.json').write_text(
                json.dumps({'plan_json': 'test-output/plans/demo-plan.json'}),
                encoding='utf-8',
            )
            dynamic_path = smoke_dir / 'dynamic.json'
            dynamic_path.write_text(json.dumps({'results': []}), encoding='utf-8')

            previous_cwd = Path.cwd()
            try:
                os.chdir(repo_root)
                resolved_plan, resolved_path = resolve_plan_for_dynamic_artifact(dynamic_path, {'results': []})
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(resolved_plan, plan_payload)
            self.assertEqual(Path(resolved_path).resolve(), plan_path.resolve())

    def test_derive_slice_targets_uses_representable_mass_threshold(self) -> None:
        layer = {
            'sliceCatalog': [
                {'sliceId': 'slice_a', 'experts': [0, 1]},
                {'sliceId': 'slice_b', 'experts': [2, 3]},
            ],
            'specialistActiveExpertTarget': 4,
        }
        teacher_weights = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
        target = derive_slice_targets(layer, teacher_weights, coverage_target=0.80)

        self.assertEqual(target['top_slice_id'], 'slice_a')
        self.assertEqual(target['positive_slice_ids'], ['slice_a', 'slice_b'])
        self.assertAlmostEqual(target['target_mass_total'], 1.0)
        self.assertAlmostEqual(target['covered_target_mass'], 1.0)

    def test_dataset_rows_can_label_against_override_plan(self) -> None:
        source_plan = {
            'mode': 'dynamic_core_specialist',
            'selectionStrategy': 'support_v1',
            'budget': {'max_resident_ratio': 0.2, 'max_resident_gib': 1.0},
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0],
                    'sliceCatalog': [{'sliceId': 'src_slice', 'experts': [2]}],
                    'specialistActiveExpertTarget': 1,
                    'specialistCandidateExpertTarget': 1,
                }
            },
        }
        label_plan = {
            'mode': 'dynamic_core_specialist',
            'selectionStrategy': 'support_v1',
            'coreSelectionMode': 'floor_seeded',
            'budget': {'max_resident_ratio': 0.3, 'max_resident_gib': 2.0},
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0],
                    'sliceCatalog': [
                        {'sliceId': 'target_slice_best', 'experts': [2]},
                        {'sliceId': 'target_slice_other', 'experts': [5]},
                    ],
                    'specialistActiveExpertTarget': 2,
                    'specialistCandidateExpertTarget': 2,
                }
            },
        }
        dynamic_payload = {
            'attempt_id': 'demo-smoke',
            'plan': source_plan,
            'results': [
                {
                    'benchmark': 'gsm8k',
                    'request_id': 'req-1',
                    'question': 'Solve 2+2 step by step',
                    'correct': False,
                    'coherent': True,
                    'parse_error': False,
                    'error': False,
                    'swap_time_s': 0.5,
                    'request_latency_s': 1.2,
                    'total_latency_s': 1.4,
                    'router_miss_summary': {
                        'inactive_mass_total': 10.0,
                        'observed_mass_total': 20.0,
                        'inactive_ratio': 0.5,
                        'inactive_expert_total': 2,
                    },
                    'router_misses': {
                        'by_layer': {
                            'layer_0': {
                                'inactive_mass': 10.0,
                                'observed_mass': 20.0,
                                'inactive_experts': [2, 5],
                            }
                        }
                    },
                }
            ],
        }
        observer_summary = {
            'layers': {
                '0': {
                    'reap': [0.0, 0.0, 0.9, 0.0, 0.0, 0.1],
                }
            }
        }

        rows = dataset_rows_from_dynamic_payload(
            'test-output/autoresearch/demo-smoke/dynamic.json',
            dynamic_payload,
            observer_summary=observer_summary,
            label_plan=label_plan,
            label_plan_path='test-output/autoresearch/target-plan/plan.json',
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['label']['top_slice_id'], 'target_slice_best')
        self.assertEqual(row['plan_context']['plan_path'], 'test-output/autoresearch/target-plan/plan.json')
        self.assertEqual(row['source_plan_context']['resident_ratio'], 0.2)
        self.assertEqual(row['plan_context']['resident_ratio'], 0.3)
        self.assertEqual(row['label']['positive_slice_ids'], ['target_slice_best'])
        self.assertAlmostEqual(row['label']['target_mass_total'], 1.0)


if __name__ == '__main__':
    unittest.main()
