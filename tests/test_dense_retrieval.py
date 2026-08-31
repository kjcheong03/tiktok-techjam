from __future__ import annotations

import json
import unittest
from pathlib import Path

from ghostlab.retrieval.dense import (
    E5_SMALL_V2,
    MINILM_CONTROL,
    rank_biased_overlap,
    sha256_file,
)


class DenseRetrievalTest(unittest.TestCase):
    def test_asset_manifest_is_pinned_and_within_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "configs/assets/e5_small_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["model_name"], E5_SMALL_V2.model_name)
        self.assertEqual(manifest["revision"], E5_SMALL_V2.revision)
        self.assertLess(manifest["total_bytes"], 500 * 1024 * 1024)

    def test_committed_report_is_current_and_keeps_holdout_sealed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "artifacts/reports/dense_retrieval_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["holdout_accessed"])
        self.assertEqual(report["sample_count"], 150)
        self.assertEqual(report["query_record_count"], 1350)
        self.assertEqual(report["decision"], "PARK_STANDALONE")
        self.assertFalse(report["first_turn_gate_comparison"]["passed"])
        expected_paths = {
            "asset_manifest_sha256": "configs/assets/e5_small_v2.json",
            "manifest_sha256": "configs/experiments/dense_retrieval_v1.json",
            "dense_code_sha256": "ghostlab/retrieval/dense.py",
            "runner_code_sha256": "scripts/run_dense_retrieval.py",
        }
        for key, relative in expected_paths.items():
            self.assertEqual(report["hashes"][key], sha256_file(root / relative))

    def test_e5_applies_required_asymmetric_prefixes(self) -> None:
        product = {
            "title": "Trail Shoe",
            "categories": ["Shoes"],
            "features": ["Water resistant"],
            "details": {"Color": "Blue"},
            "description": "For wet trails",
            "store": "Example Brand",
        }
        self.assertEqual(
            E5_SMALL_V2.format_query("  blue trail shoe "), "query: blue trail shoe"
        )
        document = E5_SMALL_V2.format_document(product)
        self.assertTrue(document.startswith("passage: Title: Trail Shoe"))
        for label in ("Category:", "Features:", "Details:", "Description:", "Brand:"):
            self.assertIn(label, document)

    def test_minilm_control_keeps_unprefixed_current_format(self) -> None:
        self.assertEqual(MINILM_CONTROL.format_query("  hiking shoes "), "hiking shoes")
        self.assertTrue(
            MINILM_CONTROL.format_document({"title": "Hiking Shoes"}).startswith(
                "Title: Hiking Shoes"
            )
        )

    def test_prefixes_change_content_addressed_cache_identity(self) -> None:
        self.assertNotEqual(
            E5_SMALL_V2.canonical_hash(), MINILM_CONTROL.canonical_hash()
        )

    def test_rank_biased_overlap_rewards_identical_order(self) -> None:
        identical = rank_biased_overlap(["a", "b", "c"], ["a", "b", "c"], limit=3)
        reversed_order = rank_biased_overlap(["a", "b", "c"], ["c", "b", "a"], limit=3)
        disjoint = rank_biased_overlap(["a", "b"], ["c", "d"], limit=2)
        self.assertAlmostEqual(identical, 1.0)
        self.assertGreater(identical, reversed_order)
        self.assertEqual(disjoint, 0.0)

    def test_rank_biased_overlap_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            rank_biased_overlap(["a"], ["a"], limit=0)


if __name__ == "__main__":
    unittest.main()
