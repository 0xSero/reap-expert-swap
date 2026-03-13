from __future__ import annotations

from typing import Iterable


def normalize_dense_local_to_global(
    dense_local_to_global: dict[int, int] | None,
    dense_local_count: int,
) -> dict[int, int]:
    if dense_local_to_global:
        return {
            int(local_idx): int(global_idx)
            for local_idx, global_idx in dense_local_to_global.items()
        }
    return {idx: idx for idx in range(int(dense_local_count))}


def resolve_compact_indices(
    dense_local_to_global: dict[int, int] | None,
    dense_local_count: int,
    keep_set: Iterable[int],
) -> tuple[list[int], list[int]]:
    normalized = normalize_dense_local_to_global(dense_local_to_global, dense_local_count)
    global_to_local = {int(global_idx): int(local_idx) for local_idx, global_idx in normalized.items()}
    selected_globals = sorted(int(global_idx) for global_idx in keep_set if int(global_idx) in global_to_local)
    dense_local_indices = [global_to_local[global_idx] for global_idx in selected_globals]
    return selected_globals, dense_local_indices


def build_compact_expert_map(global_num_experts: int, selected_globals: list[int]) -> list[int]:
    expert_map = [-1] * int(global_num_experts)
    for compact_local_idx, global_idx in enumerate(selected_globals):
        expert_map[int(global_idx)] = int(compact_local_idx)
    return expert_map


def summarize_compact_delta(
    previous_globals: Iterable[int] | None,
    selected_globals: Iterable[int],
) -> dict[str, int]:
    previous = {int(value) for value in (previous_globals or [])}
    current = {int(value) for value in selected_globals}
    return {
        "added_expert_total": len(current - previous),
        "removed_expert_total": len(previous - current),
        "reused_expert_total": len(previous & current),
    }
