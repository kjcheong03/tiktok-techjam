from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.state.baseline_v2 import StateBaselineV2
from ghostlab.state.catalog_ontology import CatalogOntology, OntologyEntry
from ghostlab.state.normalization import CatalogStateNormalizer

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_normalizer_preserves_constraint_metadata() -> None:
    normalizer = CatalogStateNormalizer(
        CatalogOntology(
            "catalog-hash",
            (
                OntologyEntry(
                    attribute="color",
                    canonical="charcoal",
                    aliases=("charcoal", "color black"),
                    frequency=5,
                ),
            ),
        ),
        confidence_threshold=0.9,
    )
    state = StateBaselineV2("session", {})

    reason = AdaptiveHybridAgent._observe_state(
        state, "A key requirement is: color: black.", 2, normalizer
    )

    constraint = next(
        item for item in state.active_constraints if item.attribute == "color"
    )
    assert constraint.values == ["charcoal"]
    assert constraint.source_turn == 2
    assert constraint.source_text == "A key requirement is: color: black."
    assert constraint.polarity == "include"
    assert constraint.provenance == "explicit"
    assert reason == "optional:state.catalog_normalizer.v1:normalized_1"


def test_minilm_view_reorders_but_preserves_every_e5_identifier() -> None:
    identifiers = ["E5-A", "E5-B", "E5-C"]
    scores = {"E5-A": 0.9, "E5-B": 0.8, "E5-C": 0.7}
    auxiliary = SimpleNamespace(
        items=(
            SimpleNamespace(parent_asin="E5-C", normalized_score=1.0),
            SimpleNamespace(parent_asin="E5-B", normalized_score=0.5),
            SimpleNamespace(parent_asin="MINILM-ONLY", normalized_score=1.0),
        )
    )

    ordered, blended, overlap = AdaptiveHybridAgent._blend_minilm_dense_view(
        identifiers, scores, auxiliary, weight=0.35
    )

    assert len(ordered) == len(identifiers)
    assert set(ordered) == set(identifiers)
    assert "MINILM-ONLY" not in ordered
    assert overlap == 2
    assert blended["E5-C"] > scores["E5-C"]


def test_real_catalog_ontology_loads_hashable_entries_and_resolves() -> None:
    ontology = CatalogOntology.from_path(
        ROOT / "artifacts/assets/catalog_ontology_v1.json"
    )

    assert ontology.entries
    assert all(isinstance(item.aliases, tuple) for item in ontology.entries)
    first = ontology.entries[0]
    resolved = ontology.resolve(first.aliases[0], attribute_hint=first.attribute)
    assert resolved is not None
    assert resolved.canonical == first.canonical
