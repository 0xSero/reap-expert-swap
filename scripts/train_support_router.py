#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from support_router import NONE_LABEL, iter_jsonl, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a prompt-only support router from dataset shards.')
    parser.add_argument('--dataset-jsonl', required=True, help='Combined dataset JSONL (all rows).')
    parser.add_argument('--target-plan-json', help='Optional target plan for heuristic baseline replay.')
    parser.add_argument('--output-dir', required=True, help='Directory for model artifacts and reports.')
    parser.add_argument('--max-features', type=int, default=8192, help='TF-IDF feature cap.')
    parser.add_argument('--top-k', type=int, default=4, help='Report learned top-k coverage at this cutoff.')
    parser.add_argument('--min-train-examples', type=int, default=8, help='Minimum per-layer train rows to fit a model.')
    parser.add_argument(
        '--model-family',
        choices=('sgd_logreg', 'multinomial_nb'),
        default='multinomial_nb',
        help='Classifier family for per-layer top-slice prediction.',
    )
    parser.add_argument(
        '--sample-weight-mode',
        choices=('none', 'inactive_ratio', 'inactive_mass_sqrt'),
        default='inactive_ratio',
        help='How to weight examples during training.',
    )
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def prompt_meta_features(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key, value in (row.get('prompt_shape') or {}).items():
        features[f'shape::{key}'] = float(value or 0.0)
    features[f"benchmark::{row.get('benchmark') or 'unknown'}"] = 1.0
    for tag in row.get('domain_tags') or []:
        features[f'tag::{tag}'] = 1.0
    return features


def training_weight(row: dict[str, Any], mode: str) -> float:
    if mode == 'none':
        return 1.0
    if mode == 'inactive_mass_sqrt':
        return max(1.0, float(row['self_look_layer'].get('inactive_mass', 0.0) or 0.0) ** 0.5)
    return 1.0 + float(row['self_look_layer'].get('inactive_ratio', 0.0) or 0.0)


def benchmark_is_reasoning(name: str) -> bool:
    normalized = str(name or '').lower()
    return 'gsm' in normalized or 'arc' in normalized or 'mmlu' in normalized


def compute_coverage(slice_target_mass: dict[str, float], selected_slice_ids: list[str]) -> float:
    return round(sum(float(slice_target_mass.get(slice_id, 0.0) or 0.0) for slice_id in selected_slice_ids), 8)


def build_heuristic_cache(rows: list[dict[str, Any]], plan: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, list[str]]]:
    if plan is None:
        return {}
    from dynamic_reap import build_active_set_payload

    cache: dict[tuple[str, str], dict[str, list[str]]] = {}
    prompts: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        key = (str(row['prompt_id']), str(row.get('benchmark') or 'unknown'))
        prompts.setdefault(key, (str(row['prompt_text']), str(row.get('benchmark') or 'unknown')))
    for key, (prompt_text, benchmark) in prompts.items():
        payload = build_active_set_payload(
            plan,
            prompt_text,
            request_id=f'offline-{key[0]}',
            benchmark=benchmark,
        )
        cache[key] = {layer_key: list(slice_ids) for layer_key, slice_ids in (payload.get('selected_slice_ids') or {}).items()}
    return cache


def summarize_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return {
            'rows': 0,
            'top1_accuracy': 0.0,
            'positive_hit_rate': 0.0,
            'learned_budget_coverage': 0.0,
            'heuristic_budget_coverage': 0.0,
            'oracle_budget_coverage': 0.0,
            'learned_topk_coverage': 0.0,
            'none_prediction_rate': 0.0,
            'coverage_delta_vs_heuristic': 0.0,
        }
    count = len(metric_rows)
    return {
        'rows': count,
        'top1_accuracy': round(sum(row['top1_correct'] for row in metric_rows) / count, 6),
        'positive_hit_rate': round(sum(row['positive_hit'] for row in metric_rows) / count, 6),
        'learned_budget_coverage': round(sum(row['learned_budget_coverage'] for row in metric_rows) / count, 6),
        'heuristic_budget_coverage': round(sum(row['heuristic_budget_coverage'] for row in metric_rows) / count, 6),
        'oracle_budget_coverage': round(sum(row['oracle_budget_coverage'] for row in metric_rows) / count, 6),
        'learned_topk_coverage': round(sum(row['learned_topk_coverage'] for row in metric_rows) / count, 6),
        'none_prediction_rate': round(sum(row['predicted_top1'] == NONE_LABEL for row in metric_rows) / count, 6),
        'coverage_delta_vs_heuristic': round(
            (sum(row['learned_budget_coverage'] for row in metric_rows) - sum(row['heuristic_budget_coverage'] for row in metric_rows))
            / count,
            6,
        ),
    }


def build_markdown(report: dict[str, Any]) -> str:
    valid = report['splits'].get('valid', {})
    test = report['splits'].get('test', {})
    lines = [
        '# Support Router Prompt-Only Training Report',
        '',
        f"- dataset: `{report['dataset_jsonl']}`",
        f"- target plan: `{report.get('target_plan_path') or 'none'}`",
        f"- rows: **{report['row_count']}**",
        f"- trained layers: **{report['trained_layer_count']}** / {report['dataset_layer_count']}",
        f"- max features: **{report['max_features']}**",
        f"- model family: **{report['model_family']}**",
        f"- sample weight mode: **{report['sample_weight_mode']}**",
        '',
        '## Valid split',
        '',
        f"- top1 accuracy: **{valid.get('top1_accuracy', 0.0):.2%}**",
        f"- positive hit rate: **{valid.get('positive_hit_rate', 0.0):.2%}**",
        f"- learned budget coverage: **{valid.get('learned_budget_coverage', 0.0):.4f}**",
        f"- heuristic budget coverage: **{valid.get('heuristic_budget_coverage', 0.0):.4f}**",
        f"- delta vs heuristic: **{valid.get('coverage_delta_vs_heuristic', 0.0):+.4f}**",
        '',
        '## Test split',
        '',
        f"- top1 accuracy: **{test.get('top1_accuracy', 0.0):.2%}**",
        f"- positive hit rate: **{test.get('positive_hit_rate', 0.0):.2%}**",
        f"- learned budget coverage: **{test.get('learned_budget_coverage', 0.0):.4f}**",
        f"- heuristic budget coverage: **{test.get('heuristic_budget_coverage', 0.0):.4f}**",
        f"- oracle budget coverage: **{test.get('oracle_budget_coverage', 0.0):.4f}**",
        f"- learned top-{report['top_k']} coverage: **{test.get('learned_topk_coverage', 0.0):.4f}**",
        f"- delta vs heuristic: **{test.get('coverage_delta_vs_heuristic', 0.0):+.4f}**",
        '',
        '## Reasoning-only test split',
        '',
        f"- top1 accuracy: **{report['reasoning_test'].get('top1_accuracy', 0.0):.2%}**",
        f"- learned budget coverage: **{report['reasoning_test'].get('learned_budget_coverage', 0.0):.4f}**",
        f"- heuristic budget coverage: **{report['reasoning_test'].get('heuristic_budget_coverage', 0.0):.4f}**",
        f"- delta vs heuristic: **{report['reasoning_test'].get('coverage_delta_vs_heuristic', 0.0):+.4f}**",
        '',
        '## Layers',
        '',
    ]
    for layer_key, layer_report in sorted(report['per_layer_test'].items(), key=lambda item: int(item[0].replace('layer_', ''))):
        lines.append(
            f"- `{layer_key}`: rows={layer_report['rows']}, top1={layer_report['top1_accuracy']:.2%}, "
            f"learned_cov={layer_report['learned_budget_coverage']:.4f}, "
            f"heuristic_cov={layer_report['heuristic_budget_coverage']:.4f}, "
            f"delta={layer_report['coverage_delta_vs_heuristic']:+.4f}"
        )
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        import numpy as np
        from scipy.sparse import hstack
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.naive_bayes import MultinomialNB
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f'Missing dependency: {exc}. Run with `uv run --with scikit-learn --with scipy --with joblib python ...`.'
        ) from exc

    rows = iter_jsonl(args.dataset_jsonl)
    if not rows:
        raise SystemExit('Dataset is empty')

    unique_plan_paths = {str(row['plan_context'].get('plan_path') or 'unknown') for row in rows}
    if len(unique_plan_paths) > 1 and not args.target_plan_json:
        raise SystemExit(
            f'Dataset contains multiple target plans ({sorted(unique_plan_paths)}). Rebuild with --target-plan-json or pass one here.'
        )

    target_plan_path = str(Path(args.target_plan_json).resolve()) if args.target_plan_json else next(iter(unique_plan_paths))
    target_plan = load_json(target_plan_path) if target_plan_path and target_plan_path != 'unknown' and Path(target_plan_path).exists() else None

    train_rows = [row for row in rows if row['split'] == 'train']
    valid_rows = [row for row in rows if row['split'] == 'valid']
    test_rows = [row for row in rows if row['split'] == 'test']
    if not train_rows:
        raise SystemExit('No train rows in dataset')

    text_vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), sublinear_tf=True)
    meta_vectorizer = DictVectorizer(sparse=True)

    train_texts = [row['prompt_feature_text'] for row in train_rows]
    train_meta = [prompt_meta_features(row) for row in train_rows]
    X_train = hstack([
        text_vectorizer.fit_transform(train_texts),
        meta_vectorizer.fit_transform(train_meta),
    ]).tocsr()

    eval_rows = valid_rows + test_rows
    X_eval = hstack([
        text_vectorizer.transform([row['prompt_feature_text'] for row in eval_rows]),
        meta_vectorizer.transform([prompt_meta_features(row) for row in eval_rows]),
    ]).tocsr() if eval_rows else None

    layer_train_indices: dict[str, list[int]] = defaultdict(list)
    layer_eval_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        layer_train_indices[str(row['plan_context']['layer_key'])].append(idx)
    for idx, row in enumerate(eval_rows):
        layer_eval_indices[str(row['plan_context']['layer_key'])].append(idx)

    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    layer_manifests: dict[str, Any] = {}
    metric_rows_by_split: dict[str, list[dict[str, Any]]] = {'valid': [], 'test': []}
    metric_rows_by_layer_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_rows: list[dict[str, Any]] = []

    heuristic_cache = build_heuristic_cache(eval_rows, target_plan)

    for layer_key, train_indices in sorted(layer_train_indices.items(), key=lambda item: int(item[0].replace('layer_', ''))):
        y_train = [str(train_rows[idx]['label']['top_slice_id']) for idx in train_indices]
        X_layer_train = X_train[train_indices]
        sample_weights = np.asarray([training_weight(train_rows[idx], args.sample_weight_mode) for idx in train_indices], dtype=float)
        class_counts = Counter(y_train)
        if len(train_indices) < args.min_train_examples:
            continue
        if len(class_counts) == 1:
            estimator = DummyClassifier(strategy='constant', constant=y_train[0])
            estimator.fit(X_layer_train, y_train)
        else:
            if args.model_family == 'sgd_logreg':
                estimator = SGDClassifier(
                    loss='log_loss',
                    penalty='l2',
                    alpha=1e-5,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=args.seed,
                    class_weight='balanced',
                )
            else:
                estimator = MultinomialNB(alpha=0.3)
            estimator.fit(X_layer_train, y_train, sample_weight=sample_weights)
        model_path = models_dir / f'{layer_key}.joblib'
        joblib.dump(estimator, model_path)
        layer_manifests[layer_key] = {
            'model_path': str(model_path.resolve()),
            'train_rows': len(train_indices),
            'class_counts': dict(sorted(class_counts.items())),
        }

        eval_indices = layer_eval_indices.get(layer_key, [])
        if not eval_indices:
            continue
        X_layer_eval = X_eval[eval_indices]
        eval_subset = [eval_rows[idx] for idx in eval_indices]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='invalid value encountered in divide',
                category=RuntimeWarning,
            )
            probabilities = estimator.predict_proba(X_layer_eval)
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze(axis=1) <= 0.0
        if zero_rows.any():
            probabilities[zero_rows, :] = 1.0 / probabilities.shape[1]
            row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        classes = [str(label) for label in estimator.classes_]
        class_lookup = {idx: label for idx, label in enumerate(classes)}
        ranked_indices = np.argsort(probabilities, axis=1)[:, ::-1]

        for local_idx, row in enumerate(eval_subset):
            ranked_labels_full = [class_lookup[int(class_idx)] for class_idx in ranked_indices[local_idx]]
            ranked_slice_labels = [label for label in ranked_labels_full if label != NONE_LABEL]
            prompt_key = (str(row['prompt_id']), str(row.get('benchmark') or 'unknown'))
            heuristic_selected = list(heuristic_cache.get(prompt_key, {}).get(layer_key, []))
            budget_k = max(1, len(heuristic_selected)) if target_plan is not None else max(1, len(row['label']['positive_slice_ids']) or 1)
            learned_budget_selected = ranked_slice_labels[:budget_k]
            learned_topk_selected = ranked_slice_labels[: args.top_k]
            predicted_top1 = ranked_labels_full[0] if ranked_labels_full else NONE_LABEL
            metric_row = {
                'split': row['split'],
                'prompt_id': row['prompt_id'],
                'benchmark': row['benchmark'],
                'layer_key': layer_key,
                'oracle_top_slice_id': row['label']['top_slice_id'],
                'predicted_top1': predicted_top1,
                'predicted_budget_selected': learned_budget_selected,
                'predicted_topk_selected': learned_topk_selected,
                'heuristic_selected': heuristic_selected,
                'positive_slice_ids': list(row['label']['positive_slice_ids']),
                'top1_correct': float(predicted_top1 == row['label']['top_slice_id']),
                'positive_hit': float(predicted_top1 in set(row['label']['positive_slice_ids'])),
                'learned_budget_coverage': compute_coverage(row['label']['slice_target_mass'], learned_budget_selected),
                'heuristic_budget_coverage': compute_coverage(row['label']['slice_target_mass'], heuristic_selected),
                'oracle_budget_coverage': float(row['label'].get('covered_target_mass', 0.0) or 0.0),
                'learned_topk_coverage': compute_coverage(row['label']['slice_target_mass'], learned_topk_selected),
                'inactive_ratio': float(row['self_look_layer'].get('inactive_ratio', 0.0) or 0.0),
                'inactive_mass': float(row['self_look_layer'].get('inactive_mass', 0.0) or 0.0),
                'reasoning': benchmark_is_reasoning(row['benchmark']),
            }
            metric_rows_by_split[row['split']].append(metric_row)
            if row['split'] == 'test':
                metric_rows_by_layer_test[layer_key].append(metric_row)
            prediction_rows.append(metric_row)

    manifest = {
        'version': 'support-router-v0',
        'dataset_jsonl': str(Path(args.dataset_jsonl).resolve()),
        'target_plan_path': target_plan_path,
        'top_k': args.top_k,
        'max_features': args.max_features,
        'model_family': args.model_family,
        'sample_weight_mode': args.sample_weight_mode,
        'trained_layers': layer_manifests,
        'text_vectorizer_path': str((output_dir / 'text_vectorizer.joblib').resolve()),
        'meta_vectorizer_path': str((output_dir / 'meta_vectorizer.joblib').resolve()),
    }
    joblib.dump(text_vectorizer, output_dir / 'text_vectorizer.joblib')
    joblib.dump(meta_vectorizer, output_dir / 'meta_vectorizer.joblib')
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    report = {
        'dataset_jsonl': str(Path(args.dataset_jsonl).resolve()),
        'target_plan_path': target_plan_path,
        'row_count': len(rows),
        'dataset_layer_count': len({row['plan_context']['layer_key'] for row in rows}),
        'trained_layer_count': len(layer_manifests),
        'max_features': args.max_features,
        'model_family': args.model_family,
        'sample_weight_mode': args.sample_weight_mode,
        'top_k': args.top_k,
        'splits': {
            'valid': summarize_metrics(metric_rows_by_split['valid']),
            'test': summarize_metrics(metric_rows_by_split['test']),
        },
        'reasoning_test': summarize_metrics([row for row in metric_rows_by_split['test'] if row['reasoning']]),
        'per_layer_test': {
            layer_key: summarize_metrics(layer_rows)
            for layer_key, layer_rows in sorted(metric_rows_by_layer_test.items(), key=lambda item: int(item[0].replace('layer_', '')))
        },
    }
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (output_dir / 'report.md').write_text(build_markdown(report), encoding='utf-8')
    with (output_dir / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')

    print(json.dumps({
        'output_dir': str(output_dir.resolve()),
        'trained_layers': len(layer_manifests),
        'valid': report['splits']['valid'],
        'test': report['splits']['test'],
        'reasoning_test': report['reasoning_test'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
