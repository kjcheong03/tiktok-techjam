from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import ghostlab.research.technique_suite as suite
from ghostlab.research.technique_suite import UnifiedTechniqueConfig, build_suite_agent
from ghostlab.state.catalog_ontology import build_catalog_ontology
from scripts.run_wave2_combination_campaign import _config, _factor_sets


def _catalog(path: Path) -> None:
    rows = [
        {
            "parent_asin": "a",
            "title": "blue trail shoe",
            "categories": ["shoe"],
            "details": {"color": "blue"},
        },
        {
            "parent_asin": "b",
            "title": "red hiking shoe",
            "categories": ["shoe"],
            "details": {"color": "red"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _base(**updates: object) -> UnifiedTechniqueConfig:
    value: dict[str, object] = {
        "experiment_id": "wave2_factory",
        "state_variant": "raw_history",
        "question_variant": "none",
        "retrieval_route": "keyword",
        "dense_backend": "off",
        "quality_prior_weight": 0.0,
    }
    value.update(updates)
    return UnifiedTechniqueConfig.model_validate(value)


def test_wave2_policy_assets_are_explicit() -> None:
    with pytest.raises(ValidationError, match="joint question policies"):
        _base(question_variant="joint_observable")
    with pytest.raises(ValidationError, match="ontology asset"):
        _base(normalizer="catalog_v1")
    with pytest.raises(ValidationError, match="router asset"):
        _base(routing_variant="calibrated")


def test_normalizer_and_eig_are_composable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite, "PROJECT_ROOT", tmp_path)
    catalog = tmp_path / "catalog.jsonl"
    ontology_path = tmp_path / "ontology.json"
    _catalog(catalog)
    ontology_path.write_text(
        json.dumps(build_catalog_ontology(catalog, minimum_frequency=1).to_payload())
    )
    agent = build_suite_agent(
        _base(
            normalizer="catalog_v1",
            normalizer_asset="ontology.json",
            question_variant="candidate_eig",
        ),
        catalog,
    )
    agent.reset("session", {})
    response = agent.respond("session", "I want a blue shoe", 1, 2)
    assert response["recommendations"]
    assert response["ask_attribute"] == "other"


def test_joint_policy_and_retrieval_switches_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite, "PROJECT_ROOT", tmp_path)
    catalog = tmp_path / "catalog.jsonl"
    asset = tmp_path / "joint.json"
    _catalog(catalog)
    asset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowed_routes": ["keyword"],
                "allowed_depths": [100],
                "policy": {
                    "rules": [],
                    "default_action": {
                        "ask_attribute": "other",
                        "retrieval_route": "keyword",
                        "retrieval_k": 100,
                        "sparse_weight": 0.75,
                        "dense_weight": 0.25,
                    },
                },
            }
        )
    )
    agent = build_suite_agent(
        _base(
            question_variant="joint_observable",
            joint_policy_asset="joint.json",
            query_expansion="prf",
            diversification="facet_mmr",
        ),
        catalog,
    )
    agent.reset("session", {})
    response = agent.respond("session", "trail shoe", 1, 2)
    assert response["ask_attribute"] == "other"
    assert agent.retrieval_trace[-1]["expansion"] is not None
    assert agent.retrieval_trace[-1]["diversification"] is not None


def test_campaign_covers_all_legal_default_factor_combinations() -> None:
    base = _base()
    core = _factor_sets(ranking=False)
    screen = _factor_sets(ranking=True)
    assert len(core) == 24
    assert len(screen) == 72
    assert all(not ({"Q", "J"} <= set(item)) for item in screen)
    assert all(not ({"R", "E"} <= set(item)) for item in screen)
    for factors in screen:
        _config(base, factors)
