from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.retrieval.rerank import LinearLexicalReranker


class RerankerTest(unittest.TestCase):
    def test_reranker_is_unique_and_deterministic(self) -> None:
        products = [
            {"parent_asin": "a", "title": "blue cotton shirt"},
            {"parent_asin": "b", "title": "red wool coat"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in products))
            reranker = LinearLexicalReranker(path)
            first = reranker.rerank("blue cotton", ["b", "a"], rerank_k=2)
            second = reranker.rerank("blue cotton", ["b", "a"], rerank_k=2)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
