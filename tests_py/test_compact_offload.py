import unittest

from scripts.compact_offload import (
    build_compact_expert_map,
    normalize_dense_local_to_global,
    resolve_compact_indices,
    summarize_compact_delta,
)


class CompactOffloadTests(unittest.TestCase):
    def test_normalize_dense_mapping_defaults_to_identity(self):
        mapping = normalize_dense_local_to_global(None, 4)
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2, 3: 3})

    def test_resolve_compact_indices_filters_and_orders_keep_set(self):
        selected_globals, dense_local_indices = resolve_compact_indices(
            {0: 3, 1: 7, 2: 9},
            3,
            {9, 3},
        )
        self.assertEqual(selected_globals, [3, 9])
        self.assertEqual(dense_local_indices, [0, 2])

    def test_build_compact_expert_map_marks_missing_as_negative_one(self):
        expert_map = build_compact_expert_map(6, [1, 4, 5])
        self.assertEqual(expert_map, [-1, 0, -1, -1, 1, 2])

    def test_summarize_compact_delta(self):
        delta = summarize_compact_delta([1, 2, 3], [2, 3, 4, 5])
        self.assertEqual(
            delta,
            {
                "added_expert_total": 2,
                "removed_expert_total": 1,
                "reused_expert_total": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
