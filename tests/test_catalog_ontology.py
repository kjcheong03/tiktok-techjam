from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.runtime.unified_experimental import ExperimentalAgent
from ghostlab.state.catalog_ontology import build_catalog_ontology, normalize_text
from ghostlab.state.memory import ConversationState
from ghostlab.state.normalization import (
    CatalogStateNormalizer,
    NormalizedConversationState,
)


class CatalogOntologyTests(unittest.TestCase):
    def catalog(self, directory: str) -> Path:
        rows = [
            {
                "parent_asin": "A",
                "categories": ["Shoes"],
                "store": "Acme",
                "details": {"Color": "Gray", "Material": "Leather"},
            },
            {
                "parent_asin": "B",
                "categories": ["Shoes"],
                "store": "Acme",
                "details": {"Colour": "Gray", "Material": "Leather"},
            },
        ]
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_build_is_deterministic_and_aliases_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.catalog(directory)
            first = build_catalog_ontology(path)
            second = build_catalog_ontology(path)
        self.assertEqual(first.to_payload(), second.to_payload())
        resolved = first.resolve("grey", attribute_hint="color", category="shoes")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.canonical, "gray")
        self.assertGreaterEqual(resolved.confidence, 0.9)

    def test_normalizer_is_opt_in_and_preserves_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ontology = build_catalog_ontology(self.catalog(directory))
        baseline = ConversationState("baseline", {})
        baseline._add("color", "grey", 1, "grey", "explicit")
        enabled = NormalizedConversationState(
            "enabled", {}, catalog_normalizer=CatalogStateNormalizer(ontology)
        )
        enabled._add("color", "grey", 1, "grey", "explicit")
        self.assertEqual(baseline.active_values()[0].normalized, "grey")
        self.assertEqual(enabled.active_values()[0].value, "grey")
        self.assertEqual(enabled.active_values()[0].normalized, "gray")
        self.assertEqual(enabled.normalization_trace[0]["source"], "catalog_exact")

    def test_text_normalization_preserves_numeric_units(self) -> None:
        self.assertEqual(normalize_text("  10–12 in. "), "10 12 in")

    def test_runtime_normalizer_is_an_explicit_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.catalog(directory)
            normalizer = CatalogStateNormalizer(build_catalog_ontology(path))
            with self.assertRaisesRegex(ValueError, "explicit switch"):
                ExperimentalAgent(path, catalog_normalizer=normalizer)
            agent = ExperimentalAgent(
                path, normalizer="catalog_v1", catalog_normalizer=normalizer
            )
            agent.reset("s", {})
        self.assertIsInstance(agent.sessions["s"], NormalizedConversationState)


if __name__ == "__main__":
    unittest.main()
