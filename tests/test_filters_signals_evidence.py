from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.optimization.evidence import (
    EvidenceRecord,
    EvidenceStore,
    family_ucb_allocation,
)
from ghostlab.policy.signals import retrieval_signals
from ghostlab.retrieval.filters import CoverageAwareFilter


class FilterSignalEvidenceTest(unittest.TestCase):
    def test_filter_is_fail_open_for_missing_metadata(self) -> None:
        products = [
            {"parent_asin": "known", "price": 120.0, "details": {"Color": "Red"}},
            {"parent_asin": "unknown", "price": None, "details": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in products))
            catalog_filter = CoverageAwareFilter(path)
            result = catalog_filter.apply(
                ["known", "unknown"], {"budget": ["under $50"]}, minimum_results=1
            )
        self.assertEqual(result, ["unknown"])

    def test_filter_falls_back_if_too_few_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps({"parent_asin": "a", "details": {}}) + "\n")
            result = CoverageAwareFilter(path).apply(
                ["a"], {"color": ["blue"]}, minimum_results=10
            )
        self.assertEqual(result, ["a"])

    def test_signals_have_explicit_missing_semantics(self) -> None:
        singleton = retrieval_signals([1.0])
        self.assertIsNone(singleton.top1_margin)
        self.assertIsNone(singleton.normalized_entropy)
        compared = retrieval_signals([1.0, 0.5], sparse_ids=["a"], dense_ids=["b"])
        self.assertEqual(compared.sparse_dense_top10_overlap, 0.0)

    def test_evidence_round_trip_and_allocation_floor(self) -> None:
        record = EvidenceRecord(
            evidence_id="e1",
            policy_id="p1",
            kind="observation",
            claim="test",
            conditions={"split": "adaptive"},
            session_ids=("s1",),
            mutation_family="question",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "evidence.jsonl")
            store.append(record)
            self.assertEqual(store.read(), [record])
        allocation = family_ucb_allocation({"good": [0.2], "untried": []})
        self.assertAlmostEqual(sum(allocation.values()), 1.0)
        self.assertGreater(allocation["untried"], 0.0)


if __name__ == "__main__":
    unittest.main()
