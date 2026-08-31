from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.policy.candidate_statistics import CandidateFacetStore


class CandidateStatisticsTests(unittest.TestCase):
    def test_statistics_measure_partition_and_missing_metadata(self) -> None:
        rows = [
            {
                "parent_asin": "A",
                "categories": ["Shoes"],
                "store": "One",
                "price": 30,
                "details": {"Color": "red"},
            },
            {
                "parent_asin": "B",
                "categories": ["Shoes"],
                "store": "Two",
                "price": 80,
                "details": {"Color": "blue"},
            },
            {
                "parent_asin": "C",
                "categories": ["Shoes"],
                "store": "Two",
                "price": None,
                "details": {},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            statistics = CandidateFacetStore(path).summarize(["A", "B", "C"], limit=3)
        color = statistics.facets["color"]
        self.assertEqual(color.candidate_count, 3)
        self.assertEqual(color.covered_count, 2)
        self.assertGreater(color.partition_gain, 0.0)
        self.assertAlmostEqual(color.no_preference_probability, 1 / 3)

    def test_depth_is_bounded_and_duplicate_ids_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                json.dumps({"parent_asin": "A", "categories": ["Shoes"]}) + "\n",
                encoding="utf-8",
            )
            statistics = CandidateFacetStore(path).summarize(["A", "A"], limit=1)
        self.assertEqual(statistics.candidate_count, 1)


if __name__ == "__main__":
    unittest.main()
