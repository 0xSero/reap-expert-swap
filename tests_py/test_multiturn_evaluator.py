from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_original_vs_multiplex as eval_mux


class MultiTurnEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec_map = {spec.name: spec for spec in eval_mux.BENCHMARK_SPECS}
        self.row = {
            'id': 'mmlu::sample',
            'sample_id': 'mmlu::sample',
            'benchmark': 'mmlu',
            'question': 'What is 2 + 2?',
            'choices': ['3', '4', '5', '6'],
            'gold': 'B',
        }
        self.plan = {
            'mode': 'dynamic_core_specialist',
            'budget': {
                'swappable_expert_budget_bytes': 1024,
                'per_expert_bytes': 128,
                'core_budget_bytes': 512,
                'specialist_budget_bytes': 512,
            },
            'perLayer': {
                'layer_0': {
                    'coreExperts': [0],
                    'sliceCatalog': [],
                }
            },
        }

    def test_build_multi_turn_turn_specs_mcq(self) -> None:
        spec = self.spec_map['mmlu']
        turns = eval_mux.build_multi_turn_turn_specs(spec, self.row)
        self.assertEqual([turn['turn_kind'] for turn in turns], ['answer', 'reason', 'recommit'])
        self.assertEqual([turn['require_parse'] for turn in turns], [True, False, True])
        self.assertTrue(turns[-1]['final_turn_for_sample'])

    def test_summarize_results_multiturn_uses_final_turn_and_conversation_summary(self) -> None:
        rows = [
            {
                'id': 'sample-1', 'sample_id': 'sample-1', 'benchmark': 'mmlu', 'conversation_id': 'conv-1',
                'turn_index': 1, 'turn_kind': 'answer', 'final_turn_for_sample': False,
                'correct': True, 'coherent': True, 'parse_error': False, 'error': None,
                'request_latency_s': 0.1, 'total_latency_s': 0.2, 'swap_time_s': 0.0,
                'selected_cartridge': None, 'active_expert_bytes': 0, 'active_expert_count': 0,
                'selected_slice_ids': {},
                'parsed_answer': 'B',
            },
            {
                'id': 'sample-1', 'sample_id': 'sample-1', 'benchmark': 'mmlu', 'conversation_id': 'conv-1',
                'turn_index': 2, 'turn_kind': 'reason', 'final_turn_for_sample': False,
                'correct': None, 'coherent': True, 'parse_error': None, 'error': None,
                'request_latency_s': 0.1, 'total_latency_s': 0.2, 'swap_time_s': 0.0,
                'selected_cartridge': None, 'active_expert_bytes': 0, 'active_expert_count': 0,
                'selected_slice_ids': {},
                'parsed_answer': None,
            },
            {
                'id': 'sample-1', 'sample_id': 'sample-1', 'benchmark': 'mmlu', 'conversation_id': 'conv-1',
                'turn_index': 3, 'turn_kind': 'recommit', 'final_turn_for_sample': True,
                'correct': True, 'coherent': True, 'parse_error': False, 'error': None,
                'request_latency_s': 0.1, 'total_latency_s': 0.2, 'swap_time_s': 0.0,
                'selected_cartridge': None, 'active_expert_bytes': 0, 'active_expert_count': 0,
                'selected_slice_ids': {},
                'parsed_answer': 'B',
            },
        ]
        summary = eval_mux.summarize_results(rows, protocol='multi_turn')
        self.assertEqual(summary['overall']['total'], 1)
        self.assertEqual(summary['overall']['accuracy'], 1.0)
        self.assertEqual(summary['turn_overall']['turn_2']['coherence_rate'], 1.0)
        self.assertEqual(summary['conversation_overall']['answer_retention_rate'], 1.0)
        self.assertEqual(summary['conversation_overall']['conversation_success_rate'], 1.0)

    def test_compare_to_baseline_marks_mismatch_invalid(self) -> None:
        payload = {
            'protocol': eval_mux.protocol_metadata('multi_turn'),
            'sample_count_per_benchmark': 1,
            'calibration_count_per_benchmark': 0,
            'seed': 7,
            'results': [{'id': 'a', 'sample_id': 'a', 'benchmark': 'mmlu', 'turn_index': 1, 'final_turn_for_sample': False}],
            'summary': {'overall': {'accuracy': 1.0, 'coherence_rate': 1.0}, 'by_benchmark': {'mmlu': {'accuracy': 1.0, 'coherence_rate': 1.0, 'avg_sample_time_s': 1.0}}},
        }
        baseline = {
            'protocol': eval_mux.protocol_metadata('single_turn'),
            'sample_count_per_benchmark': 1,
            'calibration_count_per_benchmark': 0,
            'seed': 7,
            'results': [{'id': 'a', 'sample_id': 'a', 'benchmark': 'mmlu', 'turn_index': 0, 'final_turn_for_sample': True}],
            'summary': {'overall': {'accuracy': 1.0, 'coherence_rate': 1.0}, 'by_benchmark': {'mmlu': {'accuracy': 1.0, 'coherence_rate': 1.0, 'avg_sample_time_s': 1.0}}},
        }
        comparison = eval_mux.compare_to_baseline(payload, baseline)
        self.assertEqual(comparison['retained_metrics_status'], 'invalid_unmatched_baseline')
        self.assertIn('protocol.name', comparison['baseline_match_reasons'])

    def test_chat_message_to_text_uses_reasoning_when_content_missing(self) -> None:
        message = {'role': 'assistant', 'content': None, 'reasoning': 'Final answer: 4'}
        self.assertEqual(eval_mux._chat_message_to_text(message), 'Final answer: 4')

    def test_evaluate_multi_turn_samples_passes_conversation_context_to_selector(self) -> None:
        responses = [
            {'text': 'B', 'latency_s': 0.1, 'usage': {}, 'transport': 'chat_completions'},
            {'text': 'Because 2 + 2 = 4.', 'latency_s': 0.1, 'usage': {}, 'transport': 'chat_completions'},
            {'text': 'B', 'latency_s': 0.1, 'usage': {}, 'transport': 'chat_completions'},
        ]
        payload = {
            'request_id': 'req',
            'phase': 'prefill',
            'active_set': {'layer_0': [0]},
            'selected_slice_ids': {'layer_0': []},
            'candidate_slice_ids': {'layer_0': []},
            'budget_bytes': 128,
            'active_set_signature': 'sig',
        }
        selector_calls = []

        def fake_build_active_set_payload(*args, **kwargs):
            selector_calls.append(kwargs)
            return payload | {'request_id': kwargs['request_id']}

        with (
            patch.object(eval_mux, 'build_active_set_payload', side_effect=fake_build_active_set_payload),
            patch.object(eval_mux, 'swap_active_set', return_value={'status': 'success', 'request_id': 'req', 'endpoint_time_s': 0.01, 'swap_time_s': 0.01, 'active_set_signature': 'sig', 'plan_identity': {}}),
            patch.object(eval_mux, 'request_chat_completion', side_effect=responses),
            patch.object(eval_mux, 'fetch_router_misses', return_value=None),
        ):
            rows = eval_mux.evaluate_multi_turn_samples(
                [self.row],
                self.spec_map,
                'http://example.test',
                'qwen35-dynamic',
                'dynamic',
                self.plan,
                None,
                interleaved=False,
                seed=7,
                request_timeout_s=30,
            )
        self.assertEqual(len(rows), 3)
        self.assertEqual([row['turn_index'] for row in rows], [1, 2, 3])
        self.assertEqual(selector_calls[0]['turn_index'], 1)
        self.assertEqual(selector_calls[1]['turn_index'], 2)
        self.assertEqual(selector_calls[2]['turn_index'], 3)
        self.assertEqual(len(selector_calls[0]['messages']), 1)
        self.assertEqual(len(selector_calls[1]['messages']), 3)
        self.assertEqual(len(selector_calls[2]['messages']), 5)
        self.assertTrue(rows[-1]['answer_retention'])

    def test_apply_protocol_variant_rescoring_recovers_reason_anchor_and_math_context(self) -> None:
        rows = [
            # arc conversation (gold B) with missing recommit parse + reason anchor
            {
                'id': 'arc::sample', 'sample_id': 'arc::sample', 'benchmark': 'arc_challenge', 'gold': 'B',
                'conversation_id': 'conv-arc', 'turn_index': 1, 'turn_kind': 'answer', 'final_turn_for_sample': False,
                'response': 'The user wants me to answer a multiple', 'error': None, 'request_latency_s': 0.1,
                'total_latency_s': 0.1, 'swap_time_s': 0.0, 'selected_cartridge': None, 'active_expert_bytes': 0,
                'active_expert_count': 0, 'selected_slice_ids': {},
            },
            {
                'id': 'arc::sample', 'sample_id': 'arc::sample', 'benchmark': 'arc_challenge', 'gold': 'B',
                'conversation_id': 'conv-arc', 'turn_index': 2, 'turn_kind': 'reason', 'final_turn_for_sample': False,
                'response': "Reason: previous answer was 'B' because plants make food.", 'error': None,
                'request_latency_s': 0.1, 'total_latency_s': 0.1, 'swap_time_s': 0.0, 'selected_cartridge': None,
                'active_expert_bytes': 0, 'active_expert_count': 0, 'selected_slice_ids': {},
            },
            {
                'id': 'arc::sample', 'sample_id': 'arc::sample', 'benchmark': 'arc_challenge', 'gold': 'B',
                'conversation_id': 'conv-arc', 'turn_index': 3, 'turn_kind': 'recommit', 'final_turn_for_sample': True,
                'response': 'Thinking Process:\\n\\n1.  **', 'error': None, 'request_latency_s': 0.1,
                'total_latency_s': 0.1, 'swap_time_s': 0.0, 'selected_cartridge': None, 'active_expert_bytes': 0,
                'active_expert_count': 0, 'selected_slice_ids': {},
            },
            # gsm8k conversation (gold 18) with low-confidence recommit parse
            {
                'id': 'gsm8k::sample', 'sample_id': 'gsm8k::sample', 'benchmark': 'gsm8k', 'gold': '18',
                'conversation_id': 'conv-gsm', 'turn_index': 1, 'turn_kind': 'solve', 'final_turn_for_sample': False,
                'response': 'Tip is $3. Total amount = $18.', 'error': None, 'request_latency_s': 0.1,
                'total_latency_s': 0.1, 'swap_time_s': 0.0, 'selected_cartridge': None, 'active_expert_bytes': 0,
                'active_expert_count': 0, 'selected_slice_ids': {},
            },
            {
                'id': 'gsm8k::sample', 'sample_id': 'gsm8k::sample', 'benchmark': 'gsm8k', 'gold': '18',
                'conversation_id': 'conv-gsm', 'turn_index': 2, 'turn_kind': 'verify', 'final_turn_for_sample': False,
                'response': 'Verified total: 18', 'error': None, 'request_latency_s': 0.1,
                'total_latency_s': 0.1, 'swap_time_s': 0.0, 'selected_cartridge': None, 'active_expert_bytes': 0,
                'active_expert_count': 0, 'selected_slice_ids': {},
            },
            {
                'id': 'gsm8k::sample', 'sample_id': 'gsm8k::sample', 'benchmark': 'gsm8k', 'gold': '18',
                'conversation_id': 'conv-gsm', 'turn_index': 3, 'turn_kind': 'recommit', 'final_turn_for_sample': True,
                'response': 'Thinking Process:\\n\\n1. Analyze the request.', 'error': None, 'request_latency_s': 0.1,
                'total_latency_s': 0.1, 'swap_time_s': 0.0, 'selected_cartridge': None, 'active_expert_bytes': 0,
                'active_expert_count': 0, 'selected_slice_ids': {},
            },
        ]
        rescored = eval_mux.apply_protocol_variant_rescoring(
            rows,
            self.spec_map,
            protocol_variant='calib_reason_anchor_v2',
        )
        arc_final = next(
            row for row in rescored
            if row['conversation_id'] == 'conv-arc' and row['final_turn_for_sample']
        )
        gsm_final = next(
            row for row in rescored
            if row['conversation_id'] == 'conv-gsm' and row['final_turn_for_sample']
        )
        self.assertEqual(arc_final['parsed_answer'], 'B')
        self.assertTrue(arc_final['correct'])
        self.assertEqual(arc_final['parsed_answer_source'], 'reason_anchor')
        self.assertEqual(gsm_final['parsed_answer'], '18')
        self.assertTrue(gsm_final['correct'])
        self.assertEqual(gsm_final['parsed_answer_source'], 'context_numeric')


if __name__ == '__main__':
    unittest.main()
