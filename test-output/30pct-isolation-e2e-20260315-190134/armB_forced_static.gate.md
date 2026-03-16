# Gate verdict: invalid

- policy: `dynamic_quality_gate`
- profile: `dynamic_target`
- stage: `dynamic_targeting`
- accepted: `False`
- valid: `False`

## Metrics

- mode: dynamic
- accuracy: 0.4
- coherence_rate: 0.56
- error_rate: 0.0
- parse_error_rate: 0.28
- avg_sample_time_s: 5.711052
- p95_sample_time_s: 23.792728
- avg_swap_time_s: 0.059543
- avg_warm_swap_time_s: 0.059431
- avg_cold_swap_time_s: 0.062227
- swap_count: 25
- cold_swap_count: 1
- warm_swap_count: 24
- dynamic_signature_count: 1
- dynamic_signature_transition_count: 0
- unique_cartridges_used: 0
- cartridge_transition_rate: 0.0
- accuracy_retained_pct: 50
- coherence_retained_pct: 56
- quality_loss_pct: 40.0
- worst_benchmark_accuracy_drop_abs: 1.0
- total_live_footprint_ratio: 0.3
- total_live_footprint_gib: 19.025575

## Reasons

- baseline comparison is invalid_unmatched_baseline (mismatched: sample_count_per_benchmark, result_signatures)

## Thresholds

- min_accuracy_retained_pct: 95.0
- min_coherence_retained_pct: 92.0
- min_benchmark_accuracy_retained_pct: 90.0
- min_benchmark_coherence_retained_pct: None
- max_parse_error_rate: 0.04
- max_error_rate: 0.02
- max_p95_sample_time_s: 130.0
- max_avg_swap_time_s: 2.5
- max_quality_loss_pct: 5.0
- max_benchmark_accuracy_drop_abs: 0.1
- max_total_live_footprint_ratio: 0.3
- require_multiple_cartridges: False

## Actions

- continue_dynamic_tuning: True
- promote_incumbent: False

## Checks

- none
