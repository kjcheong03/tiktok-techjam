from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ghostlab.research.technique_suite import UnifiedTechniqueConfig, build_suite_agent


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
        config(query_expansion="off", diversification="off"), catalog
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
