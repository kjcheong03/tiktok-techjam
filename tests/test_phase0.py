from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_phase0 import (
    PROJECT_ROOT,
    expected_metrics,
    load_object,
    sha256_file,
    string_mapping,
    verify_hashes,
)

MANIFEST_PATH = PROJECT_ROOT / "configs/integrity/official_v1.json"


class Phase0IntegrityTest(unittest.TestCase):
    def test_protected_official_files_match_manifest(self) -> None:
        manifest = load_object(MANIFEST_PATH)
        protected = string_mapping(manifest.get("protected_files"), "protected_files")
        result = verify_hashes(PROJECT_ROOT, protected)
        self.assertTrue(result["passed"], result["mismatches"])

    def test_frozen_starter_matches_recorded_reference(self) -> None:
        manifest = load_object(MANIFEST_PATH)
        reference = manifest.get("starter_reference")
        if not isinstance(reference, dict):
            self.fail("starter_reference must be an object")
        expected = reference.get("sha256")
        frozen = reference.get("frozen_path")
        if not isinstance(expected, str) or not isinstance(frozen, str):
            self.fail("starter_reference path and hash must be strings")
        frozen_path = PROJECT_ROOT / frozen
        self.assertEqual(sha256_file(frozen_path), expected)

    def test_hash_verification_reports_a_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.txt"
            path.write_text("changed", encoding="utf-8")
            result = verify_hashes(root, {"sample.txt": "not-the-real-hash"})
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatches"][0]["path"], "sample.txt")

    def test_expected_metrics_match_official_reference_document(self) -> None:
        manifest = load_object(MANIFEST_PATH)
        published = json.loads(
            (PROJECT_ROOT / "docs/baseline_results.json").read_text(encoding="utf-8")
        )
        expected = expected_metrics(manifest)
        for key, value in expected.items():
            document_key = (
                "technical_score" if key == "recommended_technical_score" else key
            )
            self.assertEqual(value, published[document_key])


if __name__ == "__main__":
    unittest.main()
