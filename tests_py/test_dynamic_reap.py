from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dynamic_reap import (
    build_active_set_payload,
    build_dynamic_floor_plan,
    build_dynamic_plan,
    compute_active_set_signature,
    compute_dynamic_budget,
    should_refresh_request,
    validate_active_set_payload,
)
import evaluate_original_vs_multiplex as eval_mux
from personal_activation_corpus import build_activation_corpus, extract_user_prompts
from research_gate import evaluate_payload_gate
from size_estimator import QWEN15_MOE_A27B_CHAT_CONFIG, estimate_qwen2_moe_bf16_bytes, normalize_moe_config


def make_summary(label: str) -> dict:
    return {
        'workflow': label,
        'model': 'Qwen/Qwen1.5-MoE-A2.7B-Chat',
        'processedSamples': 10,
        'totalTokens': 100,
        'layers': {
            '0': {
                'reap': [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
                'expert_frequency': [9, 8, 7, 3, 2, 1],
            },
            '1': {
                'reap': [0.2, 0.3, 0.95, 0.85, 0.15, 0.05],
                'expert_frequency': [2, 3, 9, 8, 1, 1],
            },
        },
    }


class DynamicBudgetTests(unittest.TestCase):
    def test_budget_splits_swappable_bytes_between_core_and_specialists(self) -> None:
        budget = compute_dynamic_budget(
            dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            max_resident_ratio=0.20,
            core_budget_fraction=0.35,
            specialist_budget_fraction=0.65,
            candidate_pool_multiplier=3.0,
        )
        self.assertGreater(budget['swappable_expert_budget_bytes'], 0)
        self.assertEqual(
            budget['core_budget_bytes'] + budget['specialist_budget_bytes'],
            budget['swappable_expert_budget_bytes'],
        )
        self.assertEqual(budget['candidate_pool_multiplier'], 3.0)
        self.assertGreaterEqual(budget['candidate_experts_per_layer_target'], budget['specialist_experts_per_layer_target'])

    def test_qwen35_text_config_is_normalized_for_budget_estimation(self) -> None:
        config = {
            'model_type': 'qwen3_5_moe',
            'tie_word_embeddings': False,
            'text_config': {
                'model_type': 'qwen3_5_moe_text',
                'vocab_size': 248320,
                'hidden_size': 2048,
                'num_hidden_layers': 40,
                'num_experts': 256,
                'moe_intermediate_size': 512,
                'shared_expert_intermediate_size': 512,
            },
        }
        normalized = normalize_moe_config(config)
        self.assertEqual(normalized['model_type'], 'qwen2_moe')
        estimate = estimate_qwen2_moe_bf16_bytes(config)
        self.assertGreater(estimate['full_bf16_bytes_estimate'], 0)


class DynamicFloorPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = [
            ('coding', make_summary('coding')),
            ('communication', make_summary('communication')),
        ]
        self.activation_records = [
            {
                'prompt_id': 'abc123',
                'source': 'codex',
                'timestamp': '1',
                'prompt_text': 'debug this python api latency issue',
                'conversation_id': 'conv1',
                'domain_tags': ['code', 'ops'],
            }
        ]
        self.plan = build_dynamic_floor_plan(
            self.summaries,
            signal_key='expert_frequency',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            exact_fraction_of_full=0.30,
            activation_records=self.activation_records,
        )

    def test_floor_plan_emits_core_only_plan_under_budget(self) -> None:
        self.assertEqual(self.plan['mode'], 'dynamic_core_specialist')
        self.assertEqual(self.plan['selectionMode'], 'dynamic_exact_floor')
        total_core = sum(len(layer['coreExperts']) for layer in self.plan['perLayer'].values())
        self.assertEqual(total_core, self.plan['budget']['total_exact_experts'])
        self.assertTrue(all(not layer['sliceCatalog'] for layer in self.plan['perLayer'].values()))
        computed_bytes = total_core * self.plan['budget']['per_expert_bytes']
        self.assertEqual(computed_bytes, self.plan['budget']['swappable_expert_budget_bytes'])

    def test_floor_payload_is_constant_and_contract_valid(self) -> None:
        payload_a = build_active_set_payload(
            self.plan,
            'debug this python api latency issue',
            request_id='floor-1',
            benchmark='mmlu',
        )
        payload_b = build_active_set_payload(
            self.plan,
            'summarize this benchmark result',
            request_id='floor-2',
            benchmark='hellaswag',
        )
        self.assertEqual(payload_a['active_set_signature'], payload_b['active_set_signature'])
        self.assertTrue(all(not ids for ids in payload_a['selected_slice_ids'].values()))
        validated = validate_active_set_payload(payload_a, self.plan)
        self.assertEqual(validated['budget_bytes'], self.plan['budget']['swappable_expert_budget_bytes'])


class DynamicPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = [
            ('coding', make_summary('coding')),
            ('communication', make_summary('communication')),
        ]
        self.activation_records = [
            {
                'prompt_id': 'abc123',
                'source': 'codex',
                'timestamp': '1',
                'prompt_text': 'debug this python api latency issue',
                'conversation_id': 'conv1',
                'domain_tags': ['code', 'ops'],
            },
            {
                'prompt_id': 'def456',
                'source': 'claude',
                'timestamp': '2',
                'prompt_text': 'rewrite this benchmark report into a concise email',
                'conversation_id': 'conv2',
                'domain_tags': ['writing', 'research'],
            },
        ]
        self.plan = build_dynamic_plan(
            self.summaries,
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            max_resident_ratio=0.20,
        )

    def test_dynamic_plan_emits_core_and_slice_catalog(self) -> None:
        self.assertEqual(self.plan['mode'], 'dynamic_core_specialist')
        self.assertIn('budget', self.plan)
        self.assertEqual(self.plan['budget']['candidate_pool_multiplier'], 3.0)
        layer0 = self.plan['perLayer']['layer_0']
        self.assertTrue(layer0['coreExperts'])
        self.assertTrue(layer0['sliceCatalog'])
        self.assertIn('promptClusterPriors', self.plan['scorerArtifacts'])
        self.assertEqual(self.plan['scorerArtifacts']['activationCorpus']['recordCount'], 2)
        self.assertEqual(self.plan['selectionStrategy'], 'activation_mass')
        self.assertEqual(self.plan['rotationPolicy'], 'none')

    def test_build_active_set_payload_respects_budget_and_preserves_core_union(self) -> None:
        payload = build_active_set_payload(
            self.plan,
            'please debug this python benchmark issue',
            request_id='req-1',
            benchmark='mmlu',
        )
        self.assertEqual(payload['request_id'], 'req-1')
        self.assertEqual(payload['phase'], 'prefill')
        self.assertGreater(payload['budget_bytes'], 0)
        self.assertLessEqual(
            payload['budget_bytes'], self.plan['budget']['swappable_expert_budget_bytes']
        )
        validated = validate_active_set_payload(payload, self.plan)
        self.assertIn('union_validation', validated)
        self.assertEqual(
            sorted(validated['active_set']['layer_0']),
            sorted(validated['union_validation']['layer_0']['activeExperts']),
        )

    def test_validate_payload_rejects_over_budget_or_non_union_active_set(self) -> None:
        payload = build_active_set_payload(
            self.plan,
            'please debug this python benchmark issue',
            request_id='req-2',
            benchmark='mmlu',
        )
        payload['active_set']['layer_0'] = payload['active_set']['layer_0'] + [99]
        with self.assertRaises(ValueError):
            validate_active_set_payload(payload, self.plan)

    def test_active_set_signature_is_stable_for_same_union(self) -> None:
        payload = build_active_set_payload(
            self.plan,
            'please debug this python benchmark issue',
            request_id='req-3',
            benchmark='mmlu',
        )
        signature_a = payload['active_set_signature']
        signature_b = compute_active_set_signature(
            payload['active_set'],
            payload['selected_slice_ids'],
        )
        self.assertEqual(signature_a, signature_b)

    def test_refresh_policy_triggers_once_and_then_exhausts(self) -> None:
        miss_payload = {
            'by_layer': {
                'layer_0': {
                    'inactive_mass': 0.5,
                    'observed_mass': 1.0,
                    'inactive_experts': [4, 5],
                }
            }
        }
        first = should_refresh_request(miss_payload, refreshes_used=0, max_refreshes=1)
        self.assertTrue(first['should_refresh'])
        second = should_refresh_request(miss_payload, refreshes_used=1, max_refreshes=1)
        self.assertFalse(second['should_refresh'])
        self.assertEqual(second['reason'], 'refresh_budget_exhausted')

    def test_validate_payload_rejects_unknown_phase_with_explicit_error(self) -> None:
        payload = build_active_set_payload(
            self.plan,
            'please debug this python benchmark issue',
            request_id='req-bad-phase',
            benchmark='mmlu',
        )
        payload['phase'] = 'decode'
        with self.assertRaisesRegex(ValueError, "phase must be 'prefill' or 'decode_refresh'"):
            validate_active_set_payload(payload, self.plan)

    def test_single_summary_plan_is_stable_without_selector_priors(self) -> None:
        single_summary_plan = build_dynamic_plan(
            [('personal', make_summary('personal'))],
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            max_resident_ratio=0.20,
        )
        payload_code = build_active_set_payload(
            single_summary_plan,
            'debug this python api timeout and trace the bad call path',
            request_id='code-req',
            benchmark='gsm8k',
        )
        payload_write = build_active_set_payload(
            single_summary_plan,
            'rewrite this result into a concise executive summary email',
            request_id='write-req',
            benchmark='hellaswag',
        )
        self.assertEqual(
            payload_code['active_set_signature'],
            payload_write['active_set_signature'],
        )

    def test_explicit_rotation_policy_varies_late_layers(self) -> None:
        rotating_plan = build_dynamic_plan(
            [('personal', make_summary('personal'))],
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            rotation_policy='late_prompt_hash',
            max_resident_ratio=0.20,
        )
        payload_code = build_active_set_payload(
            rotating_plan,
            'debug this python api timeout and trace the bad call path',
            request_id='code-rotate',
            benchmark='gsm8k',
        )
        payload_write = build_active_set_payload(
            rotating_plan,
            'rewrite this result into a concise executive summary email',
            request_id='write-rotate',
            benchmark='hellaswag',
        )
        self.assertNotEqual(payload_code['active_set_signature'], payload_write['active_set_signature'])

    def test_support_v1_plan_exposes_support_mass_metadata(self) -> None:
        support_plan = build_dynamic_plan(
            [('personal', make_summary('personal'))],
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            selection_strategy='support_v1',
            core_budget_fraction=0.75,
            specialist_budget_fraction=0.25,
            candidate_pool_multiplier=1.25,
            max_resident_ratio=0.20,
        )
        self.assertEqual(support_plan['selectionStrategy'], 'support_v1')
        layer0 = support_plan['perLayer']['layer_0']
        self.assertIn('coreRawActivationMass', layer0)
        self.assertTrue(all('reapMass' in row for row in layer0['sliceCatalog']))

    def test_layer_importance_drives_nonuniform_targets(self) -> None:
        weighted_plan = build_dynamic_plan(
            [('personal', make_summary('personal'))],
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            layer_importance={
                'layer_0': {
                    'miss_sensitivity': 1.0,
                    'benchmark_dependence': 0.5,
                    'expert_entropy': 0.5,
                    'prompt_overlap_jaccard': 0.1,
                },
                'layer_1': {
                    'miss_sensitivity': 0.1,
                    'benchmark_dependence': 0.0,
                    'expert_entropy': 0.1,
                    'prompt_overlap_jaccard': 0.9,
                },
            },
            max_resident_ratio=0.20,
        )
        targets = weighted_plan['scorerArtifacts']['layerBudgetTargets']
        self.assertGreater(
            targets['specialistExpertsPerLayer']['layer_0'],
            targets['specialistExpertsPerLayer']['layer_1'],
        )
        self.assertGreater(
            weighted_plan['perLayer']['layer_0']['specialistActiveExpertTarget'],
            weighted_plan['perLayer']['layer_1']['specialistActiveExpertTarget'],
        )

    def test_floor_seeded_core_selection_uses_protected_core_and_excludes_it_from_slices(self) -> None:
        floor_seeded_plan = build_dynamic_plan(
            [('personal', make_summary('personal'))],
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            selection_strategy='support_v1',
            core_selection_mode='floor_seeded',
            core_budget_fraction=0.75,
            specialist_budget_fraction=0.25,
            candidate_pool_multiplier=1.25,
            max_resident_ratio=0.20,
        )
        self.assertEqual(floor_seeded_plan['coreSelectionMode'], 'floor_seeded')
        for layer_key, layer in floor_seeded_plan['perLayer'].items():
            protected = sorted(
                floor_seeded_plan['scorerArtifacts']['floorSelection']['selectedCoreByLayer'][layer_key]
            )
            self.assertEqual(sorted(layer['coreExperts']), protected)
            protected_set = set(layer['coreExperts'])
            for slice_row in layer['sliceCatalog']:
                self.assertTrue(protected_set.isdisjoint(set(slice_row['experts'])))

    def test_floor_seeded_core_selection_records_floor_metadata(self) -> None:
        floor_seeded_plan = build_dynamic_plan(
            [('personal', make_summary('personal'))],
            signal_key='reap',
            model_config=dict(QWEN15_MOE_A27B_CHAT_CONFIG),
            activation_records=self.activation_records,
            selection_strategy='support_v1',
            core_selection_mode='floor_seeded',
            core_budget_fraction=0.75,
            specialist_budget_fraction=0.25,
            candidate_pool_multiplier=1.25,
            max_resident_ratio=0.20,
        )
        self.assertEqual(
            floor_seeded_plan['scorerArtifacts']['floorSelection']['layerWeightMode'],
            'late_boost',
        )
        self.assertTrue(
            floor_seeded_plan['scorerArtifacts']['floorSelection']['selectedCoreByLayer']
        )


class PersonalActivationCorpusTests(unittest.TestCase):
    def test_extract_user_prompts_handles_display_and_tab_bubbles(self) -> None:
        record = {
            'source': 'cursor',
            'workspace_id': 'ws1',
            'conversation': {
                'display': 'Help me debug this API timeout',
                'tabs': [
                    {
                        'bubbles': [
                            {'role': 'user', 'text': 'Write a python script to parse logs'},
                            {'role': 'assistant', 'text': 'Sure'},
                        ]
                    }
                ],
            },
        }
        prompts = extract_user_prompts(record)
        self.assertIn('Help me debug this API timeout', prompts)
        self.assertIn('Write a python script to parse logs', prompts)
        self.assertNotIn('Sure', prompts)

    def test_build_activation_corpus_deduplicates_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'history.jsonl'
            path.write_text(
                '\n'.join(
                    [
                        '{"source":"codex","conversation":{"display":"Debug this python function"}}',
                        '{"source":"claude","conversation":{"display":"Debug this python function"}}',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            records, summary = build_activation_corpus([path])
            self.assertEqual(len(records), 1)
            self.assertEqual(summary['duplicate_prompts_removed'], 1)
            self.assertEqual(records[0]['domain_tags'], ['code'])


class DynamicGateTests(unittest.TestCase):
    def test_dynamic_target_gate_enforces_quality_loss_and_benchmark_floor(self) -> None:
        payload = {
            'mode': 'dynamic',
            'runtime_identity': {
                'mode': 'dynamic',
                'server_url': 'http://127.0.0.1:8011',
                'host': '127.0.0.1',
                'port': 8011,
                'concurrency_mode': 'serialized_single_flight',
                'readiness_evidence': {
                    'identity_path': '/tmp/dynamic-8011.identity.json',
                    'source': 'mission-runtime',
                    'identity': {
                        'service': 'remote-dynamic-8011',
                        'host': '127.0.0.1',
                        'port': 8011,
                        'plan_file': '/tmp/plan.json',
                    },
                },
            },
            'plan_identity': {
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'plan_path': '/tmp/plan.json',
            },
            'plan': {
                'budget': {
                    'max_resident_ratio': 0.30,
                    'max_resident_gib': 12.0,
                }
            },
            'summary': {
                'overall': {
                    'total': 10,
                    'parse_error_rate': 0.0,
                    'error_rate': 0.0,
                    'p95_sample_time_s': 1.0,
                    'avg_swap_time_s': 0.2,
                    'accuracy': 0.91,
                    'coherence_rate': 0.95,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy': 0.88, 'coherence_rate': 0.95},
                    'arc_challenge': {'accuracy': 0.90, 'coherence_rate': 0.94},
                    'hellaswag': {'accuracy': 0.92, 'coherence_rate': 0.97},
                    'winogrande': {'accuracy': 0.91, 'coherence_rate': 0.95},
                    'gsm8k': {'accuracy': 0.93, 'coherence_rate': 0.96},
                },
            },
            'comparison': {
                'overall': {
                    'accuracy_retained_pct': 95.5,
                    'coherence_retained_pct': 97.0,
                    'quality_loss_pct': 4.5,
                    'worst_benchmark_accuracy_drop_abs': 0.08,
                },
                'by_benchmark': {
                    'mmlu': {
                        'accuracy_retained_pct': 94.0,
                        'coherence_retained_pct': 97.0,
                        'accuracy_drop_abs': 0.08,
                    },
                    'arc_challenge': {
                        'accuracy_retained_pct': 95.0,
                        'coherence_retained_pct': 96.0,
                        'accuracy_drop_abs': 0.05,
                    },
                    'hellaswag': {
                        'accuracy_retained_pct': 96.0,
                        'coherence_retained_pct': 98.0,
                        'accuracy_drop_abs': 0.04,
                    },
                    'winogrande': {
                        'accuracy_retained_pct': 95.0,
                        'coherence_retained_pct': 97.0,
                        'accuracy_drop_abs': 0.05,
                    },
                    'gsm8k': {
                        'accuracy_retained_pct': 97.0,
                        'coherence_retained_pct': 98.0,
                        'accuracy_drop_abs': 0.04,
                    },
                },
            },
        }
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertTrue(gate['accepted'])
        payload['comparison']['overall']['quality_loss_pct'] = 6.0
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertFalse(gate['accepted'])
        self.assertIn('quality_loss_pct', '\n'.join(gate['reasons']))

    def test_dynamic_gate_invalidates_missing_runtime_readiness_evidence(self) -> None:
        """VAL-FOUND-005: Missing runtime readiness evidence invalidates the checkpoint."""
        payload = {
            'mode': 'dynamic',
            'runtime_identity': {
                'mode': 'dynamic',
                'server_url': 'http://127.0.0.1:8011',
                'host': '127.0.0.1',
                'port': 8011,
                'concurrency_mode': 'serialized_single_flight',
                # Missing readiness_evidence
            },
            'plan_identity': {
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'plan_path': '/tmp/plan.json',
            },
            'plan': {
                'budget': {
                    'max_resident_ratio': 0.30,
                    'max_resident_gib': 12.0,
                }
            },
            'summary': {
                'overall': {
                    'total': 10,
                    'parse_error_rate': 0.0,
                    'error_rate': 0.0,
                    'p95_sample_time_s': 1.0,
                    'avg_swap_time_s': 0.2,
                    'accuracy': 0.91,
                    'coherence_rate': 0.95,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy': 0.88, 'coherence_rate': 0.95},
                    'arc_challenge': {'accuracy': 0.90, 'coherence_rate': 0.94},
                    'hellaswag': {'accuracy': 0.92, 'coherence_rate': 0.97},
                    'winogrande': {'accuracy': 0.91, 'coherence_rate': 0.95},
                    'gsm8k': {'accuracy': 0.93, 'coherence_rate': 0.96},
                },
            },
            'comparison': {
                'overall': {
                    'accuracy_retained_pct': 95.5,
                    'coherence_retained_pct': 97.0,
                    'quality_loss_pct': 4.5,
                    'worst_benchmark_accuracy_drop_abs': 0.08,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy_retained_pct': 94.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.08},
                    'arc_challenge': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 96.0, 'accuracy_drop_abs': 0.05},
                    'hellaswag': {'accuracy_retained_pct': 96.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                    'winogrande': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.05},
                    'gsm8k': {'accuracy_retained_pct': 97.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                },
            },
        }
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertEqual(gate['verdict'], 'invalid')
        self.assertFalse(gate['valid'])
        self.assertIn('missing runtime readiness evidence', '\n'.join(gate['reasons']))

    def test_dynamic_gate_invalidates_runtime_plan_identity_mismatch(self) -> None:
        """VAL-CROSS-002: Mismatched plan identity between runtime and plan invalidates checkpoint."""
        payload = {
            'mode': 'dynamic',
            'runtime_identity': {
                'mode': 'dynamic',
                'server_url': 'http://127.0.0.1:8011',
                'host': '127.0.0.1',
                'port': 8011,
                'concurrency_mode': 'serialized_single_flight',
                'plan_path': '/tmp/runtime-plan.json',  # Different from plan_identity
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'readiness_evidence': {
                    'identity_path': '/tmp/dynamic-8011.identity.json',
                    'source': 'mission-runtime',
                    'identity': {
                        'service': 'remote-dynamic-8011',
                        'host': '127.0.0.1',
                        'port': 8011,
                        'plan_file': '/tmp/runtime-plan.json',
                    },
                },
            },
            'plan_identity': {
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'plan_path': '/tmp/plan-identity.json',  # Different from runtime_identity
            },
            'plan': {
                'budget': {
                    'max_resident_ratio': 0.30,
                    'max_resident_gib': 12.0,
                }
            },
            'summary': {
                'overall': {
                    'total': 10,
                    'parse_error_rate': 0.0,
                    'error_rate': 0.0,
                    'p95_sample_time_s': 1.0,
                    'avg_swap_time_s': 0.2,
                    'accuracy': 0.91,
                    'coherence_rate': 0.95,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy': 0.88, 'coherence_rate': 0.95},
                    'arc_challenge': {'accuracy': 0.90, 'coherence_rate': 0.94},
                    'hellaswag': {'accuracy': 0.92, 'coherence_rate': 0.97},
                    'winogrande': {'accuracy': 0.91, 'coherence_rate': 0.95},
                    'gsm8k': {'accuracy': 0.93, 'coherence_rate': 0.96},
                },
            },
            'comparison': {
                'overall': {
                    'accuracy_retained_pct': 95.5,
                    'coherence_retained_pct': 97.0,
                    'quality_loss_pct': 4.5,
                    'worst_benchmark_accuracy_drop_abs': 0.08,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy_retained_pct': 94.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.08},
                    'arc_challenge': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 96.0, 'accuracy_drop_abs': 0.05},
                    'hellaswag': {'accuracy_retained_pct': 96.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                    'winogrande': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.05},
                    'gsm8k': {'accuracy_retained_pct': 97.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                },
            },
        }
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertEqual(gate['verdict'], 'invalid')
        self.assertFalse(gate['valid'])
        rendered = '\n'.join(gate['reasons'])
        self.assertIn('runtime identity plan path does not match plan identity', rendered)

    def test_dynamic_gate_invalidates_broken_request_id_linkage(self) -> None:
        """VAL-CROSS-003: Broken request_id-to-evidence linkage invalidates checkpoint."""
        plan_identity = {
            'plan_mode': 'dynamic_core_specialist',
            'plan_budget_bytes': 123,
            'plan_path': '/tmp/plan.json',
        }
        payload = {
            'mode': 'dynamic',
            'runtime_identity': {
                'mode': 'dynamic',
                'server_url': 'http://127.0.0.1:8011',
                'host': '127.0.0.1',
                'port': 8011,
                'concurrency_mode': 'serialized_single_flight',
                'plan_path': '/tmp/plan.json',
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'readiness_evidence': {
                    'identity_path': '/tmp/dynamic-8011.identity.json',
                    'source': 'mission-runtime',
                    'identity': {
                        'service': 'remote-dynamic-8011',
                        'host': '127.0.0.1',
                        'port': 8011,
                        'plan_file': '/tmp/plan.json',
                    },
                },
            },
            'plan_identity': plan_identity,
            'plan': {
                'budget': {
                    'max_resident_ratio': 0.30,
                    'max_resident_gib': 12.0,
                }
            },
            'summary': {
                'overall': {
                    'total': 10,
                    'parse_error_rate': 0.0,
                    'error_rate': 0.0,
                    'p95_sample_time_s': 1.0,
                    'avg_swap_time_s': 0.2,
                    'accuracy': 0.91,
                    'coherence_rate': 0.95,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy': 0.88, 'coherence_rate': 0.95},
                    'arc_challenge': {'accuracy': 0.90, 'coherence_rate': 0.94},
                    'hellaswag': {'accuracy': 0.92, 'coherence_rate': 0.97},
                    'winogrande': {'accuracy': 0.91, 'coherence_rate': 0.95},
                    'gsm8k': {'accuracy': 0.93, 'coherence_rate': 0.96},
                },
            },
            'comparison': {
                'overall': {
                    'accuracy_retained_pct': 95.5,
                    'coherence_retained_pct': 97.0,
                    'quality_loss_pct': 4.5,
                    'worst_benchmark_accuracy_drop_abs': 0.08,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy_retained_pct': 94.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.08},
                    'arc_challenge': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 96.0, 'accuracy_drop_abs': 0.05},
                    'hellaswag': {'accuracy_retained_pct': 96.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                    'winogrande': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.05},
                    'gsm8k': {'accuracy_retained_pct': 97.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                },
            },
            'results': [
                {
                    'id': 'row-1',
                    'benchmark': 'mmlu',
                    'request_id': 'req-1',
                    'swap_request_id': 'req-2',  # Mismatched - should be req-1
                    'active_set_signature': 'sig-1',
                    'swap_plan_identity': plan_identity,
                    'router_misses': {'request_id': 'req-1', 'by_layer': {}},
                }
            ],
        }
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertEqual(gate['verdict'], 'invalid')
        self.assertFalse(gate['valid'])
        rendered = '\n'.join(gate['reasons'])
        self.assertIn('request-level evidence linkage failures', rendered)
        self.assertIn('row-1', rendered)

    def test_dynamic_gate_invalidates_missing_router_misses(self) -> None:
        """VAL-CROSS-003: Missing router_misses breaks request-level evidence linkage."""
        plan_identity = {
            'plan_mode': 'dynamic_core_specialist',
            'plan_budget_bytes': 123,
            'plan_path': '/tmp/plan.json',
        }
        payload = {
            'mode': 'dynamic',
            'runtime_identity': {
                'mode': 'dynamic',
                'server_url': 'http://127.0.0.1:8011',
                'host': '127.0.0.1',
                'port': 8011,
                'concurrency_mode': 'serialized_single_flight',
                'plan_path': '/tmp/plan.json',
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'readiness_evidence': {
                    'identity_path': '/tmp/dynamic-8011.identity.json',
                    'source': 'mission-runtime',
                    'identity': {
                        'service': 'remote-dynamic-8011',
                        'host': '127.0.0.1',
                        'port': 8011,
                        'plan_file': '/tmp/plan.json',
                    },
                },
            },
            'plan_identity': plan_identity,
            'plan': {
                'budget': {
                    'max_resident_ratio': 0.30,
                    'max_resident_gib': 12.0,
                }
            },
            'summary': {
                'overall': {
                    'total': 10,
                    'parse_error_rate': 0.0,
                    'error_rate': 0.0,
                    'p95_sample_time_s': 1.0,
                    'avg_swap_time_s': 0.2,
                    'accuracy': 0.91,
                    'coherence_rate': 0.95,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy': 0.88, 'coherence_rate': 0.95},
                    'arc_challenge': {'accuracy': 0.90, 'coherence_rate': 0.94},
                    'hellaswag': {'accuracy': 0.92, 'coherence_rate': 0.97},
                    'winogrande': {'accuracy': 0.91, 'coherence_rate': 0.95},
                    'gsm8k': {'accuracy': 0.93, 'coherence_rate': 0.96},
                },
            },
            'comparison': {
                'overall': {
                    'accuracy_retained_pct': 95.5,
                    'coherence_retained_pct': 97.0,
                    'quality_loss_pct': 4.5,
                    'worst_benchmark_accuracy_drop_abs': 0.08,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy_retained_pct': 94.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.08},
                    'arc_challenge': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 96.0, 'accuracy_drop_abs': 0.05},
                    'hellaswag': {'accuracy_retained_pct': 96.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                    'winogrande': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.05},
                    'gsm8k': {'accuracy_retained_pct': 97.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.04},
                },
            },
            'results': [
                {
                    'id': 'row-1',
                    'benchmark': 'mmlu',
                    'request_id': 'req-1',
                    'swap_request_id': 'req-1',
                    'active_set_signature': 'sig-1',
                    'swap_plan_identity': plan_identity,
                    # Missing router_misses entirely
                }
            ],
        }
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertEqual(gate['verdict'], 'invalid')
        self.assertFalse(gate['valid'])
        rendered = '\n'.join(gate['reasons'])
        self.assertIn('request-level evidence linkage failures', rendered)
        self.assertIn('missing router_misses', rendered)

    def test_dynamic_gate_accepts_complete_artifact_chain(self) -> None:
        """VAL-FOUND-005: Complete artifact chain with all linkages is accepted."""
        plan_identity = {
            'plan_mode': 'dynamic_core_specialist',
            'plan_budget_bytes': 123,
            'plan_path': '/tmp/plan.json',
        }
        payload = {
            'mode': 'dynamic',
            'runtime_identity': {
                'mode': 'dynamic',
                'server_url': 'http://127.0.0.1:8011',
                'host': '127.0.0.1',
                'port': 8011,
                'concurrency_mode': 'serialized_single_flight',
                'plan_path': '/tmp/plan.json',
                'plan_mode': 'dynamic_core_specialist',
                'plan_budget_bytes': 123,
                'readiness_evidence': {
                    'identity_path': '/tmp/dynamic-8011.identity.json',
                    'source': 'mission-runtime',
                    'identity': {
                        'service': 'remote-dynamic-8011',
                        'host': '127.0.0.1',
                        'port': 8011,
                        'plan_file': '/tmp/plan.json',
                    },
                },
            },
            'plan_identity': plan_identity,
            'plan': {
                'budget': {
                    'max_resident_ratio': 0.30,
                    'max_resident_gib': 12.0,
                }
            },
            'summary': {
                'overall': {
                    'total': 10,
                    'parse_error_rate': 0.0,
                    'error_rate': 0.0,
                    'p95_sample_time_s': 1.0,
                    'avg_swap_time_s': 0.2,
                    'accuracy': 0.95,
                    'coherence_rate': 0.95,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy': 0.94, 'coherence_rate': 0.95},
                    'arc_challenge': {'accuracy': 0.95, 'coherence_rate': 0.94},
                    'hellaswag': {'accuracy': 0.96, 'coherence_rate': 0.97},
                    'winogrande': {'accuracy': 0.95, 'coherence_rate': 0.95},
                    'gsm8k': {'accuracy': 0.96, 'coherence_rate': 0.96},
                },
            },
            'comparison': {
                'overall': {
                    'accuracy_retained_pct': 96.0,
                    'coherence_retained_pct': 97.0,
                    'quality_loss_pct': 4.0,
                    'worst_benchmark_accuracy_drop_abs': 0.05,
                },
                'by_benchmark': {
                    'mmlu': {'accuracy_retained_pct': 94.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.05},
                    'arc_challenge': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 96.0, 'accuracy_drop_abs': 0.04},
                    'hellaswag': {'accuracy_retained_pct': 96.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.03},
                    'winogrande': {'accuracy_retained_pct': 95.0, 'coherence_retained_pct': 97.0, 'accuracy_drop_abs': 0.04},
                    'gsm8k': {'accuracy_retained_pct': 96.0, 'coherence_retained_pct': 98.0, 'accuracy_drop_abs': 0.03},
                },
            },
            'results': [
                {
                    'id': 'row-1',
                    'benchmark': 'mmlu',
                    'request_id': 'req-1',
                    'swap_request_id': 'req-1',  # Matches request_id
                    'active_set_signature': 'sig-1',
                    'swap_plan_identity': plan_identity,  # Matches plan_identity
                    'router_misses': {'request_id': 'req-1', 'by_layer': {}},  # Matches request_id
                }
            ],
        }
        gate = evaluate_payload_gate(payload, 'dynamic_target')
        self.assertTrue(gate['valid'])
        self.assertEqual(gate['verdict'], 'provisional')
        self.assertTrue(gate['accepted'])


class DynamicEvaluatorReuseTests(unittest.TestCase):
    def test_dynamic_mode_skips_duplicate_active_set_swaps(self) -> None:
        plan = {
            'mode': 'dynamic_core_specialist',
            'budget': {'swappable_expert_budget_bytes': 100, 'per_expert_bytes': 1},
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0],
                    'sliceCatalog': [],
                }
            },
        }
        spec = eval_mux.BenchmarkSpec('mmlu', 'unused', 'unused', 'unused', 'mcq', 8)
        spec_map = {'mmlu': spec}
        rows = [
            {'id': 'row-1', 'benchmark': 'mmlu', 'question': 'q1', 'choices': ['x', 'y'], 'gold': 'A'},
            {'id': 'row-2', 'benchmark': 'mmlu', 'question': 'q2', 'choices': ['x', 'y'], 'gold': 'A'},
        ]
        payload = {
            'request_id': 'req',
            'phase': 'prefill',
            'active_set': {'layer_0': [0]},
            'selected_slice_ids': {'layer_0': []},
            'candidate_slice_ids': {'layer_0': []},
            'budget_bytes': 1,
            'active_set_signature': 'same-signature',
        }
        with (
            patch.object(eval_mux, 'build_active_set_payload', return_value=payload),
            patch.object(
                eval_mux,
                'swap_active_set',
                side_effect=[
                    {'status': 'success', 'endpoint_time_s': 0.5, 'swap_time_s': 0.5, 'no_op_reuse': False},
                    {'status': 'success', 'endpoint_time_s': 0.0, 'swap_time_s': 0.0, 'no_op_reuse': True},
                ],
            ),
            patch.object(eval_mux, 'request_completion', return_value={'text': 'A', 'latency_s': 0.1, 'usage': {}}),
            patch.object(eval_mux, 'fetch_router_misses', return_value=None),
        ):
            results = eval_mux.evaluate_samples(
                rows,
                spec_map,
                server_url='http://example.com',
                model='dummy',
                mode='dynamic',
                plan=plan,
                benchmark_to_cartridge=None,
                interleaved=False,
                seed=7,
                request_timeout_s=10,
            )
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]['swap_reused_active_set'])
        self.assertTrue(results[1]['swap_reused_active_set'])

    def test_dynamic_mode_records_request_scoped_swap_and_router_identity(self) -> None:
        plan = {
            'mode': 'dynamic_core_specialist',
            'budget': {
                'swappable_expert_budget_bytes': 100,
                'per_expert_bytes': 1,
                'max_refreshes_per_request': 1,
            },
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0],
                    'sliceCatalog': [],
                }
            },
        }
        spec = eval_mux.BenchmarkSpec('mmlu', 'unused', 'unused', 'unused', 'mcq', 8)
        spec_map = {'mmlu': spec}
        rows = [
            {'id': 'row-1', 'benchmark': 'mmlu', 'question': 'q1', 'choices': ['x', 'y'], 'gold': 'A'},
        ]
        payload = {
            'request_id': 'req-123',
            'phase': 'prefill',
            'active_set': {'layer_0': [0]},
            'selected_slice_ids': {'layer_0': []},
            'candidate_slice_ids': {'layer_0': []},
            'budget_bytes': 1,
            'active_set_signature': 'sig-123',
        }
        swap_response = {
            'status': 'success',
            'request_id': 'req-123',
            'phase': 'prefill',
            'endpoint_time_s': 0.5,
            'swap_time_s': 0.25,
            'no_op_reuse': False,
            'active_set_signature': 'sig-123',
            'concurrency_mode': 'serialized_single_flight',
            'plan_identity': {'plan_mode': 'dynamic_core_specialist'},
        }
        router_misses = {
            'status': 'success',
            'request_id': 'req-123',
            'refreshes_used': 0,
            'by_layer': {
                'layer_0': {
                    'inactive_mass': 0.5,
                    'observed_mass': 1.0,
                    'inactive_experts': [2],
                }
            },
        }
        with (
            patch.object(eval_mux, 'stable_id', return_value='req-123'),
            patch.object(eval_mux, 'build_active_set_payload', return_value=payload),
            patch.object(eval_mux, 'swap_active_set', return_value=swap_response),
            patch.object(eval_mux, 'request_completion', return_value={'text': 'A', 'latency_s': 0.1, 'usage': {}}),
            patch.object(eval_mux, 'fetch_router_misses', return_value=router_misses),
        ):
            results = eval_mux.evaluate_samples(
                rows,
                spec_map,
                server_url='http://example.com:8011',
                model='dummy',
                mode='dynamic',
                plan=plan,
                benchmark_to_cartridge=None,
                interleaved=False,
                seed=7,
                request_timeout_s=10,
            )
        self.assertEqual(results[0]['request_id'], 'req-123')
        self.assertEqual(results[0]['swap_request_id'], 'req-123')
        self.assertEqual(results[0]['active_set_signature'], 'sig-123')
        self.assertEqual(results[0]['router_misses']['request_id'], 'req-123')
        self.assertEqual(results[0]['swap_plan_identity']['plan_mode'], 'dynamic_core_specialist')
        self.assertEqual(results[0]['router_miss_summary']['inactive_ratio'], 0.5)

    def test_dynamic_mode_reuse_preserves_request_correlation_and_explicit_no_op_evidence(self) -> None:
        plan = {
            'mode': 'dynamic_core_specialist',
            'budget': {
                'swappable_expert_budget_bytes': 100,
                'per_expert_bytes': 1,
                'max_refreshes_per_request': 1,
            },
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0],
                    'sliceCatalog': [],
                }
            },
        }
        spec = eval_mux.BenchmarkSpec('mmlu', 'unused', 'unused', 'unused', 'mcq', 8)
        spec_map = {'mmlu': spec}
        rows = [
            {'id': 'row-1', 'benchmark': 'mmlu', 'question': 'q1', 'choices': ['x', 'y'], 'gold': 'A'},
            {'id': 'row-2', 'benchmark': 'mmlu', 'question': 'q2', 'choices': ['x', 'y'], 'gold': 'A'},
        ]
        payloads = [
            {
                'request_id': 'req-1',
                'phase': 'prefill',
                'active_set': {'layer_0': [0]},
                'selected_slice_ids': {'layer_0': []},
                'candidate_slice_ids': {'layer_0': []},
                'budget_bytes': 1,
                'active_set_signature': 'same-signature',
            },
            {
                'request_id': 'req-2',
                'phase': 'prefill',
                'active_set': {'layer_0': [0]},
                'selected_slice_ids': {'layer_0': []},
                'candidate_slice_ids': {'layer_0': []},
                'budget_bytes': 1,
                'active_set_signature': 'same-signature',
            },
        ]
        router_payloads = [
            {'status': 'success', 'request_id': 'req-1', 'refreshes_used': 0, 'by_layer': {}},
            {'status': 'success', 'request_id': 'req-2', 'refreshes_used': 0, 'by_layer': {}},
        ]
        with (
            patch.object(eval_mux, 'stable_id', side_effect=['req-1', 'req-2']),
            patch.object(eval_mux, 'build_active_set_payload', side_effect=payloads),
            patch.object(
                eval_mux,
                'swap_active_set',
                side_effect=[
                    {'status': 'success', 'request_id': 'req-1', 'endpoint_time_s': 0.5, 'swap_time_s': 0.5, 'no_op_reuse': False},
                    {'status': 'success', 'request_id': 'req-2', 'endpoint_time_s': 0.0, 'swap_time_s': 0.0, 'no_op_reuse': True},
                ],
            ),
            patch.object(eval_mux, 'request_completion', return_value={'text': 'A', 'latency_s': 0.1, 'usage': {}}),
            patch.object(eval_mux, 'fetch_router_misses', side_effect=router_payloads),
        ):
            results = eval_mux.evaluate_samples(
                rows,
                spec_map,
                server_url='http://example.com:8011',
                model='dummy',
                mode='dynamic',
                plan=plan,
                benchmark_to_cartridge=None,
                interleaved=False,
                seed=7,
                request_timeout_s=10,
            )
        self.assertFalse(results[0]['swap_reused_active_set'])
        self.assertTrue(results[1]['swap_reused_active_set'])
        self.assertEqual(results[1]['swap_request_id'], 'req-2')
        self.assertEqual(results[1]['active_set_signature'], 'same-signature')
        self.assertEqual(results[1]['router_misses']['request_id'], 'req-2')


class SupportEstimatorPolicyTests(unittest.TestCase):
    def make_support_policy_plan(self, mode: str, reserve_fraction: float = 0.0) -> dict:
        return {
            'mode': 'dynamic_core_specialist',
            'model': 'dummy',
            'signalKey': 'reap',
            'budget': {
                'specialist_budget_bytes': 2,
                'candidate_pool_multiplier': 2.0,
                'per_expert_bytes': 1,
                'swappable_expert_budget_bytes': 3,
            },
            'perLayer': {
                'layer_0': {
                    'numExperts': 4,
                    'coreExperts': [0],
                    'sliceCatalog': [
                        {'sliceId': 'slice_a', 'experts': [1], 'byteCost': 1, 'activationMass': 10.0, 'taskPriors': {}, 'signalsBySummary': {}},
                        {'sliceId': 'slice_b', 'experts': [2], 'byteCost': 1, 'activationMass': 5.0, 'taskPriors': {}, 'signalsBySummary': {}},
                        {'sliceId': 'slice_c', 'experts': [3], 'byteCost': 1, 'activationMass': 1.0, 'taskPriors': {}, 'signalsBySummary': {}},
                    ],
                    'specialistActiveExpertTarget': 2,
                    'specialistCandidateExpertTarget': 3,
                }
            },
            'scorerArtifacts': {
                'taskFamilySlicePriors': {'global': {'layer_0': ['slice_a', 'slice_b', 'slice_c']}},
                'promptClusterPriors': {},
                'benchmarkMissPriors': {'gsm8k': {'layer_0': {'3': 1.0}}},
                'tagMissPriors': {'math': {'layer_0': {'3': 1.0}}},
                'supportEstimatorConfig': {
                    'mode': mode,
                    'reserve_fraction': reserve_fraction,
                    'late_layer_start_frac': 0.0,
                    'benchmark_scale': 20.0,
                    'tag_scale': 8.0,
                },
            },
        }

    def test_candidate_only_bias_changes_candidates_not_prefill_union(self) -> None:
        plan = self.make_support_policy_plan('candidate_only')
        payload = build_active_set_payload(
            plan,
            'solve this probability puzzle carefully',
            request_id='support-candidate',
            benchmark='gsm8k',
        )
        self.assertEqual(payload['selected_slice_ids']['layer_0'], ['slice_a', 'slice_b'])
        self.assertEqual(payload['candidate_slice_ids']['layer_0'][0], 'slice_c')
        self.assertEqual(payload['active_set']['layer_0'], [0, 1, 2])

    def test_prefill_reserve_keeps_base_slice_and_uses_small_support_reserve(self) -> None:
        plan = self.make_support_policy_plan('prefill_reserve', reserve_fraction=0.5)
        payload = build_active_set_payload(
            plan,
            'solve this probability puzzle carefully',
            request_id='support-reserve',
            benchmark='gsm8k',
        )
        self.assertEqual(payload['selected_slice_ids']['layer_0'], ['slice_a', 'slice_c'])
        self.assertEqual(payload['active_set']['layer_0'], [0, 1, 3])

    def test_conversation_context_can_influence_support_selection(self) -> None:
        plan = self.make_support_policy_plan('full_prefill')
        payload_no_context = build_active_set_payload(
            plan,
            'help me think about this problem',
            request_id='support-no-context',
            benchmark=None,
        )
        payload_with_context = build_active_set_payload(
            plan,
            'help me think about this problem',
            request_id='support-with-context',
            benchmark=None,
            conversation_id='conv-math',
            turn_index=2,
            messages=[
                {'role': 'user', 'content': 'we are solving a probability equation'},
                {'role': 'assistant', 'content': 'let us reason through the math carefully'},
            ],
        )
        self.assertEqual(payload_no_context['selected_slice_ids']['layer_0'], ['slice_a', 'slice_b'])
        self.assertEqual(payload_with_context['selected_slice_ids']['layer_0'], ['slice_c', 'slice_a'])
        self.assertNotEqual(
            payload_no_context['active_set_signature'],
            payload_with_context['active_set_signature'],
        )

    def test_conversation_metadata_is_preserved_in_rationale(self) -> None:
        plan = self.make_support_policy_plan('candidate_only')
        payload = build_active_set_payload(
            plan,
            'help me think about this problem',
            request_id='support-conversation-metadata',
            benchmark='gsm8k',
            conversation_id='conv-123',
            turn_index=5,
            messages=[
                {'role': 'user', 'content': 'initial question'},
                {'role': 'assistant', 'content': 'initial answer'},
            ],
            conversation_context={'summary': 'user is asking about a probability benchmark'},
        )
        conversation = payload['rationale']['conversation']
        self.assertEqual(conversation['conversation_id'], 'conv-123')
        self.assertEqual(conversation['turn_index'], 5)
        self.assertEqual(conversation['message_count'], 2)
        self.assertGreater(conversation['context_chars'], 0)


if __name__ == '__main__':
    unittest.main()
