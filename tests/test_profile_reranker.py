from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.retrieval.profile import ProfilePriorReranker


class ProfilePriorRerankerTests(unittest.TestCase):
    def test_matching_profile_tag_is_a_soft_boost(self) -> None:
        products = [
            {"parent_asin": "plain", "title": "Plain shoe"},
            {"parent_asin": "match", "title": "Durability hiking shoe"},
            {"parent_asin": "last", "title": "Formal shoe"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in products))
            reranker = ProfilePriorReranker(path)
            reranker.reset("s", {"preference_tags": ["durability"]})
            result = reranker.rerank(
                "s", ["plain", "match", "last"], weight=1.0, rerank_k=3
            )
        self.assertEqual(result, ["match", "plain", "last"])

    def test_missing_profile_preserves_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps({"parent_asin": "a", "title": "A"}) + "\n")
            reranker = ProfilePriorReranker(path)
            reranker.reset("s", {})
            self.assertEqual(reranker.rerank("s", ["a"], weight=0.5), ["a"])


if __name__ == "__main__":
    unittest.main()
