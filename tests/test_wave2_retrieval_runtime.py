from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ghostlab.research.technique_suite import UnifiedTechniqueConfig, build_suite_agent
from ghostlab.runtime.unified_experimental import ExperimentalAgent


def write_catalog(path: Path) -> None:
    products = [
        {
            "parent_asin": "a",
            "title": "waterproof trail shoe",
            "categories": ["shoe"],
            "features": ["grippy sole"],
            "details": {"color": "blue"},
        },
        {
            "parent_asin": "b",
            "title": "waterproof hiking shoe",
            "categories": ["shoe"],
            "features": ["grippy sole"],
            "details": {"color": "red"},
        },
        {
            "parent_asin": "c",
            "title": "silk shirt",
            "categories": ["clothing"],
            "features": ["formal"],
            "details": {"color": "blue"},
        },
    ]
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products), encoding="utf-8"
    )


def config(**updates: object) -> UnifiedTechniqueConfig:
    value: dict[str, object] = {
        "experiment_id": "runtime",
        "state_variant": "raw_history",
        "question_variant": "none",
        "retrieval_route": "keyword",
        "dense_backend": "off",
        "quality_prior_weight": 0.0,
    }
    value.update(updates)
    return UnifiedTechniqueConfig.model_validate(
        value,
    )


def test_off_switches_preserve_exact_response(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    control = build_suite_agent(config(), catalog)
    explicit_off = build_suite_agent(
        config(
            query_expansion="off",
            diversification="off",
            question_max_turn=10,
            retrieval_k=200,
            rrf_constant=60,
            dense_activation="always",
            query_expansion_activation="always",
            profile_prior_max_turn=10,
            cross_encoder_activation="always",
            cross_encoder_min_turn=1,
        ),
        catalog,
    )
    profile: dict[str, object] = {}
    for agent in (control, explicit_off):
        agent.reset("session", profile)
    assert control.respond("session", "blue trail shoe", 1, 3) == explicit_off.respond(
        "session", "blue trail shoe", 1, 3
    )


def test_core_switches_do_not_import_heavy_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in {"torch", "transformers", "sentence_transformers"}:
            raise AssertionError(f"heavy optional dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    agent = build_suite_agent(
        config(query_expansion="prf", diversification="facet_mmr"), catalog
    )
    agent.reset("session", {})
    response = agent.respond("session", "trail shoe", 1, 3)
    assert response["recommendations"]
    assert agent.retrieval_trace[-1]["expansion"] is not None
    assert agent.retrieval_trace[-1]["diversification"] is not None


def test_rescue_routes_require_explicit_assets() -> None:
    for route in ("learned_sparse_union", "late_interaction_union"):
        with pytest.raises(ValidationError, match="requires"):
            config(retrieval_route=route)


def test_observable_uncertainty_gate_skips_query_expansion(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    agent = build_suite_agent(
        config(
            query_expansion="prf",
            query_expansion_activation="uncertain",
            query_expansion_min_entropy=0.99,
        ),
        catalog,
    )
    agent.reset("session", {})
    agent.respond("session", "trail shoe", 1, 3)
    trace = agent.retrieval_trace[-1]
    assert trace["activation"]["query_expansion"] is False
    assert trace["expansion"]["reason"] == "uncertainty_gate"


def test_question_horizon_changes_actual_response(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    agent = build_suite_agent(
        config(
            question_variant="sequence",
            question_order=("other", "size"),
            question_max_turn=1,
        ),
        catalog,
    )
    agent.reset("session", {})
    assert agent.respond("session", "trail shoe", 1, 3)["ask_attribute"] == "other"
    assert agent.respond("session", "blue", 2, 3)["ask_attribute"] is None
    assert agent.question_trace[-1]["reason"] == "question_horizon"


def test_dense_uncertainty_gate_changes_actual_route(tmp_path: Path) -> None:
    class DenseStub:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, limit: int) -> list[str]:
            self.calls += 1
            return ["c", "b", "a"]

    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    dense = DenseStub()
    agent = ExperimentalAgent(
        catalog,
        state_variant="raw_history",
        question_variant="none",
        retrieval_route="rrf",
        dense_retriever=dense,
        dense_activation="uncertain",
        dense_activation_min_entropy=0.99,
    )
    agent.reset("session", {})
    agent.respond("session", "trail shoe", 1, 3)
    assert dense.calls == 0
    assert agent.retrieval_trace[-1]["route"] == "keyword"
    assert agent.retrieval_trace[-1]["activation"]["dense"] is False


def test_unavailable_rescue_asset_fails_before_heavy_import(
    tmp_path: Path, monkeypatch
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in {"torch", "transformers"}:
            raise AssertionError(f"heavy optional dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="unavailable"):
        build_suite_agent(
            config(
                retrieval_route="learned_sparse_union",
                learned_sparse_asset="configs/assets/splade_rescue_v1.json",
            ),
            catalog,
        )
