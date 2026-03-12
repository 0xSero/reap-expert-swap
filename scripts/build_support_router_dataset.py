#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from support_router import (
    build_activation_lookup,
    dataset_rows_from_dynamic_payload,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build support-router training dataset from dynamic run history.')
    parser.add_argument('--history-glob', action='append', default=[], help='Glob for dynamic.json artifacts.')
    parser.add_argument('--history-json', action='append', default=[], help='Explicit dynamic.json path.')
    parser.add_argument('--observer-summary-json', required=True, help='Observer summary JSON with REAP weights.')
    parser.add_argument('--activation-corpus-jsonl', help='Optional activation corpus for prompt metadata enrichment.')
    parser.add_argument('--target-plan-json', help='Optional target plan to project labels into a single slice space.')
    parser.add_argument('--coverage-target', type=float, default=0.80, help='Fraction of representable target mass to cover.')
    parser.add_argument('--output-dir', required=True, help='Destination directory for JSONL shards + summaries.')
    return parser.parse_args()


def collect_history_paths(history_globs: list[str], history_jsons: list[str]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for raw_path in history_jsons:
        candidate = Path(raw_path)
        resolved = candidate.resolve()
        if resolved.is_file() and str(resolved) not in seen:
            seen.add(str(resolved))
            paths.append(resolved)
    for pattern in history_globs:
        for candidate in sorted(Path.cwd().glob(pattern)):
            resolved = candidate.resolve()
            if resolved.is_file() and str(resolved) not in seen:
                seen.add(str(resolved))
                paths.append(resolved)
    return paths


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def build_summary(
    rows: list[dict[str, Any]],
    *,
    history_paths: list[Path],
    skipped_paths: list[str],
    target_plan_path: str | None,
) -> dict[str, Any]:
    split_counts = Counter(row['split'] for row in rows)
    benchmark_counts = Counter(row['benchmark'] for row in rows)
    layer_counts = Counter(row['plan_context']['layer_key'] for row in rows)
    none_counts = Counter(row['label']['top_slice_id'] == '__none__' for row in rows)
    source_attempts = Counter(row['run_attempt_id'] for row in rows)
    target_plan_counts = Counter(str(row['plan_context'].get('plan_path') or 'unknown') for row in rows)
    source_plan_counts = Counter(str(row['source_plan_context'].get('plan_path') or 'unknown') for row in rows)

    positive_lengths = [len(row['label']['positive_slice_ids']) for row in rows]
    target_mass = [float(row['label'].get('target_mass_total', 0.0) or 0.0) for row in rows]
    inactive_ratio = [float(row['self_look_layer'].get('inactive_ratio', 0.0) or 0.0) for row in rows]
    inactive_mass = [float(row['self_look_layer'].get('inactive_mass', 0.0) or 0.0) for row in rows]

    by_benchmark = {}
    for benchmark, count in sorted(benchmark_counts.items()):
        benchmark_rows = [row for row in rows if row['benchmark'] == benchmark]
        by_benchmark[benchmark] = {
            'rows': count,
            'unique_prompts': len({row['prompt_id'] for row in benchmark_rows}),
            'none_rate': round(
                sum(1 for row in benchmark_rows if row['label']['top_slice_id'] == '__none__') / max(1, count),
                6,
            ),
            'avg_inactive_ratio': round(
                sum(float(row['self_look_layer'].get('inactive_ratio', 0.0) or 0.0) for row in benchmark_rows) / max(1, count),
                6,
            ),
        }

    summary = {
        'dataset_version': rows[0]['dataset_version'] if rows else 'support-router-v0',
        'row_count': len(rows),
        'unique_prompts': len({row['prompt_id'] for row in rows}),
        'unique_attempts': len(source_attempts),
        'history_file_count': len(history_paths),
        'history_files': [str(path) for path in history_paths],
        'skipped_history_files': skipped_paths,
        'target_plan_path': target_plan_path,
        'split_counts': dict(sorted(split_counts.items())),
        'benchmark_counts': dict(sorted(benchmark_counts.items())),
        'layer_count': len(layer_counts),
        'none_label_rate': round(none_counts[True] / max(1, len(rows)), 6),
        'avg_positive_slice_count': round(sum(positive_lengths) / max(1, len(positive_lengths)), 6),
        'avg_target_mass_total': round(sum(target_mass) / max(1, len(target_mass)), 6),
        'avg_inactive_ratio': round(sum(inactive_ratio) / max(1, len(inactive_ratio)), 6),
        'avg_inactive_mass': round(sum(inactive_mass) / max(1, len(inactive_mass)), 6),
        'source_attempt_counts': dict(sorted(source_attempts.items())),
        'target_plan_counts': dict(sorted(target_plan_counts.items())),
        'source_plan_counts': dict(sorted(source_plan_counts.items())),
        'by_benchmark': by_benchmark,
    }
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# Support Router Dataset v0',
        '',
        f"- rows: **{summary['row_count']}**",
        f"- unique prompts: **{summary['unique_prompts']}**",
        f"- unique attempts: **{summary['unique_attempts']}**",
        f"- history files scanned: **{summary['history_file_count']}**",
        f"- target plan: `{summary.get('target_plan_path') or 'source-plan-per-run'}`",
        f"- none label rate: **{summary['none_label_rate']:.2%}**",
        f"- avg positive slice count: **{summary['avg_positive_slice_count']:.2f}**",
        f"- avg target mass total: **{summary['avg_target_mass_total']:.4f}**",
        f"- avg inactive ratio: **{summary['avg_inactive_ratio']:.4f}**",
        '',
        '## Split counts',
        '',
    ]
    for split, count in summary['split_counts'].items():
        lines.append(f'- `{split}`: {count}')
    lines.extend(['', '## Benchmarks', ''])
    for benchmark, row in summary['by_benchmark'].items():
        lines.append(
            f"- `{benchmark}`: rows={row['rows']}, unique_prompts={row['unique_prompts']}, "
            f"none_rate={row['none_rate']:.2%}, avg_inactive_ratio={row['avg_inactive_ratio']:.4f}"
        )
    if summary.get('skipped_history_files'):
        lines.extend(['', '## Skipped history files', ''])
        for path in summary['skipped_history_files']:
            lines.append(f'- `{path}`')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    history_paths = collect_history_paths(args.history_glob, args.history_json)
    if not history_paths:
        raise SystemExit('No dynamic history files matched --history-glob/--history-json')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observer_summary = load_json(args.observer_summary_json)
    activation_lookup = build_activation_lookup(args.activation_corpus_jsonl)
    target_plan = load_json(args.target_plan_json) if args.target_plan_json else None
    target_plan_path = str(Path(args.target_plan_json).resolve()) if args.target_plan_json else None

    rows: list[dict[str, Any]] = []
    skipped_paths: list[str] = []
    for history_path in history_paths:
        payload = load_json(history_path)
        artifact_rows = dataset_rows_from_dynamic_payload(
            history_path,
            payload,
            observer_summary=observer_summary,
            activation_lookup=activation_lookup,
            coverage_target=args.coverage_target,
            label_plan=target_plan,
            label_plan_path=target_plan_path,
        )
        if not artifact_rows:
            skipped_paths.append(str(history_path))
            continue
        rows.extend(artifact_rows)

    all_path = output_dir / 'all.jsonl'
    train_path = output_dir / 'train.jsonl'
    valid_path = output_dir / 'valid.jsonl'
    test_path = output_dir / 'test.jsonl'
    write_jsonl(all_path, rows)
    write_jsonl(train_path, [row for row in rows if row['split'] == 'train'])
    write_jsonl(valid_path, [row for row in rows if row['split'] == 'valid'])
    write_jsonl(test_path, [row for row in rows if row['split'] == 'test'])

    summary = build_summary(
        rows,
        history_paths=history_paths,
        skipped_paths=skipped_paths,
        target_plan_path=target_plan_path,
    )
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (output_dir / 'summary.md').write_text(build_markdown(summary), encoding='utf-8')

    print(json.dumps({
        'output_dir': str(output_dir.resolve()),
        'rows': len(rows),
        'train_rows': summary['split_counts'].get('train', 0),
        'valid_rows': summary['split_counts'].get('valid', 0),
        'test_rows': summary['split_counts'].get('test', 0),
        'skipped_history_files': len(skipped_paths),
        'target_plan_path': target_plan_path,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
