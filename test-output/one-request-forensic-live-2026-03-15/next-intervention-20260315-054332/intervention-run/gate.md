# Gate verdict: invalid

- policy: `dynamic_quality_gate`
- profile: `dynamic_target`
- stage: `dynamic_targeting`
- accepted: `False`
- valid: `False`

## Metrics

- mode: dynamic
- accuracy: 0.0
- coherence_rate: 0.0
- error_rate: 1.0
- parse_error_rate: 0.0
- avg_sample_time_s: 0.008223
- p95_sample_time_s: 0.009802
- avg_swap_time_s: 0.0
- swap_count: 0
- unique_cartridges_used: 0
- cartridge_transition_rate: 0.0
- accuracy_retained_pct: 0
- coherence_retained_pct: 0
- quality_loss_pct: 90.0
- worst_benchmark_accuracy_drop_abs: 1.0
- total_live_footprint_ratio: 0.376631
- total_live_footprint_gib: 23.88538

## Reasons

- baseline comparison is invalid_unmatched_baseline (mismatched: sample_count_per_benchmark, calibration_count_per_benchmark, result_signatures)
- missing runtime readiness evidence
- request-level evidence linkage failures: gsm8k::d3d4ef1085aa3a51: swap_request_id does not match request_id; gsm8k::d3d4ef1085aa3a51: missing swap_plan_identity; gsm8k::d3d4ef1085aa3a51: missing router_misses; mmlu::high_school_us_history::78043c7a88efdb6b: swap_request_id does not match request_id; mmlu::high_school_us_history::78043c7a88efdb6b: missing swap_plan_identity; mmlu::high_school_us_history::78043c7a88efdb6b: missing router_misses; winogrande::378822cb7a4c18b9: swap_request_id does not match request_id; winogrande::378822cb7a4c18b9: missing swap_plan_identity; winogrande::378822cb7a4c18b9: missing router_misses; arc::Mercury_SC_LBS10666: swap_request_id does not match request_id; arc::Mercury_SC_LBS10666: missing swap_plan_identity; arc::Mercury_SC_LBS10666: missing router_misses; hellaswag::24952: swap_request_id does not match request_id; hellaswag::24952: missing swap_plan_identity; hellaswag::24952: missing router_misses

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
