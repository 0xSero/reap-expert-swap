# Gate verdict: reject

- policy: `dynamic_quality_gate`
- profile: `dynamic_target`
- stage: `dynamic_targeting`
- accepted: `False`
- valid: `True`

## Metrics

- mode: dynamic
- accuracy: 0.6
- coherence_rate: 0.8
- error_rate: 0.0
- parse_error_rate: 0.0
- avg_sample_time_s: 27.995419
- p95_sample_time_s: 130.623265
- avg_swap_time_s: 17.202866
- swap_count: 5
- unique_cartridges_used: 0
- cartridge_transition_rate: 0.0
- accuracy_retained_pct: 75
- coherence_retained_pct: 80
- quality_loss_pct: 20.0
- worst_benchmark_accuracy_drop_abs: 1.0
- total_live_footprint_ratio: 0.3
- total_live_footprint_gib: 19.025575

## Reasons

- accuracy_retained_pct 75.0000 < 95.0000
- coherence_retained_pct 80.0000 < 92.0000
- p95_sample_time_s 130.6233 > 130.0000
- avg_swap_time_s 17.2029 > 2.5000
- quality_loss_pct 20.0000 > 5.0000
- worst_benchmark_accuracy_drop_abs 1.0000 > 0.1000
- arc_challenge.accuracy_retained_pct 0.0000 < 90.0000
- arc_challenge.accuracy_drop_abs 1.0000 > 0.1000
- gsm8k.accuracy_retained_pct 0.0000 < 90.0000
- gsm8k.accuracy_drop_abs 1.0000 > 0.1000
- mmlu.accuracy_retained_pct 0.0000 < 90.0000

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

- accuracy_retained_pct: value=75 threshold=95.0 passed=False
- coherence_retained_pct: value=80 threshold=92.0 passed=False
- parse_error_rate: value=0.0 threshold=0.04 passed=True
- error_rate: value=0.0 threshold=0.02 passed=True
- p95_sample_time_s: value=130.623265 threshold=130.0 passed=False
- avg_swap_time_s: value=17.202866 threshold=2.5 passed=False
- quality_loss_pct: value=20.0 threshold=5.0 passed=False
- worst_benchmark_accuracy_drop_abs: value=1.0 threshold=0.1 passed=False
- total_live_footprint_ratio: value=0.3 threshold=0.3 passed=True
- arc_challenge.accuracy_retained_pct: value=0 threshold=90.0 passed=False
- arc_challenge.accuracy_drop_abs: value=1.0 threshold=0.1 passed=False
- gsm8k.accuracy_retained_pct: value=0 threshold=90.0 passed=False
- gsm8k.accuracy_drop_abs: value=1.0 threshold=0.1 passed=False
- hellaswag.accuracy_retained_pct: value=100 threshold=90.0 passed=True
- hellaswag.accuracy_drop_abs: value=0.0 threshold=0.1 passed=True
- mmlu.accuracy_retained_pct: value=0.0 threshold=90.0 passed=False
- mmlu.accuracy_drop_abs: value=0.0 threshold=0.1 passed=True
- winogrande.accuracy_retained_pct: value=100 threshold=90.0 passed=True
- winogrande.accuracy_drop_abs: value=0.0 threshold=0.1 passed=True
