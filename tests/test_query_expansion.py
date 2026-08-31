from __future__ import annotations

import json
from pathlib import Path

from ghostlab.retrieval.pseudo_relevance import CatalogPseudoRelevanceFeedback
from ghostlab.state.query_expansion import ExpansionGuard, ExpansionTerm


def write_catalog(path: Path) -> None:
    products = [
        {
            "parent_asin": "a",
            "title": "waterproof trail running shoe",
            "categories": ["footwear"],
            "features": ["grippy sole"],
        },
        {
            "parent_asin": "b",
            "title": "waterproof hiking shoe",
            "categories": ["footwear"],
            "features": ["grippy sole"],
        },
        {
            "parent_asin": "c",
            "title": "silk formal shirt",
            "categories": ["clothing"],
        },
    ]
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products), encoding="utf-8"
    )


def test_prf_adds_only_grounded_supported_terms(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    expander = CatalogPseudoRelevanceFeedback(
        catalog, feedback_k=2, minimum_support=1.0, max_terms=3, max_added_ratio=1.0
    )
    result = expander.expand("shoe", ["a", "b", "c"])
    values = {item.value for item in result.terms}
    assert values <= {"waterproof", "footwear", "grippy", "sole"}
    assert values
    assert result.expanded_query.startswith("shoe ")


def test_prf_refuses_insufficient_feedback(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    result = CatalogPseudoRelevanceFeedback(catalog).expand("shoe", ["missing"])
    assert result.terms == ()
    assert result.reason == "insufficient_feedback"


def test_expansion_guard_preserves_explicit_terms_and_bounds_additions() -> None:
    guard = ExpansionGuard(max_terms=2, max_added_ratio=0.5)
    accepted = guard.apply(
        "blue trail shoe waterproof",
        [
            ExpansionTerm("trail", 1.0, "test"),
            ExpansionTerm("hiking", 0.9, "test"),
            ExpansionTerm("grippy", 0.8, "test"),
            ExpansionTerm("sole", 0.7, "test"),
        ],
    )
    assert [item.value for item in accepted] == ["hiking", "grippy"]
