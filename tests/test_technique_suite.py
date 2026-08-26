from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ghostlab.research.technique_suite import (
    PROJECT_ROOT,
    UnifiedTechniqueConfig,
    load_suite_config,
    valid_combinations,
)


def test_all_unified_presets_validate_without_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in {"sentence_transformers", "torch"}:
            raise AssertionError(f"optional dependency imported while parsing: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    paths = sorted((PROJECT_ROOT / "configs/suites").glob("*.json"))
    assert paths
    for path in paths:
        assert load_suite_config(path).experiment_id


def test_catalog_sources_and_dependency_extras_are_declared() -> None:
    catalog = json.loads(
        (PROJECT_ROOT / "configs/techniques/catalog_v1.json").read_text(
            encoding="utf-8"
        )
    )
    identifiers = [item["id"] for item in catalog["techniques"]]
    assert len(identifiers) == len(set(identifiers))
    for item in catalog["techniques"]:
        assert item["extra"] in {"core", "gbdt", "dense", "neural", "all"}
        assert (PROJECT_ROOT / item["source"]).is_file(), item


def test_invalid_dense_and_learned_combinations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="dense backend"):
        UnifiedTechniqueConfig(
            experiment_id="bad_dense",
            retrieval_route="dense",
            dense_backend="off",
            question_variant="none",
        )
    with pytest.raises(ValidationError, match="learned questions"):
        UnifiedTechniqueConfig(
            experiment_id="bad_question",
            retrieval_route="keyword",
            dense_backend="off",
            question_variant="learned",
        )


def test_combination_planner_is_deterministic_and_filters_invalid() -> None:
    base = UnifiedTechniqueConfig(
        experiment_id="planner",
        retrieval_route="keyword",
        dense_backend="off",
        question_variant="none",
        quality_prior_weight=0.0,
    )
    dimensions = {
        "retrieval_route": ["keyword", "dense"],
        "dense_backend": ["off", "e5_small_v2"],
        "dense_model_path": [None, "artifacts/cache/models/e5-small-v2"],
    }
    first, rejected = valid_combinations(base, dimensions)
    second, _ = valid_combinations(base, dimensions)
    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]
    assert len(first) == 2
    assert rejected


def test_asset_paths_cannot_escape_repository() -> None:
    with pytest.raises(ValidationError, match="inside the project"):
        UnifiedTechniqueConfig(
            experiment_id="escape",
            retrieval_route="dense",
            dense_backend="e5_small_v2",
            dense_model_path="../outside",
            question_variant="none",
        )


def test_machine_catalog_is_json_and_small_assets_are_tracked() -> None:
    for path in (
        PROJECT_ROOT / "configs/assets/e5_small_v2.json",
        PROJECT_ROOT / "configs/assets/minilm_control.json",
        PROJECT_ROOT / "configs/assets/cross_encoder_minilm.json",
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["model_name"]
        assert len(value["revision"]) == 40
        assert not Path(value["destination"]).is_absolute()


def test_consolidation_manifest_covers_sources_tests_and_retest_triggers() -> None:
    value = json.loads(
        (PROJECT_ROOT / "configs/integrity/unified_consolidation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    identifiers = [item["id"] for item in value["components"]]
    assert len(identifiers) == len(set(identifiers))
    assert {
        "query_construction",
        "learned_questions",
        "dense_retrieval",
        "metadata_gbdt",
        "cross_encoder",
        "constraint_gbdt_and_guard",
        "deep_dense_gbdt",
        "learned_question_gbdt",
        "neural_gbdt",
        "compiled_guarded_runtime",
        "unified_composer",
    } <= set(identifiers)
    for component in value["components"]:
        assert component["retest_trigger"]
        assert component["source_paths"]
        assert component["test_paths"]
        for field in (
            "source_paths",
            "test_paths",
            "manifest_paths",
            "report_paths",
            "runtime_assets",
        ):
            for relative in component.get(field, []):
                assert (PROJECT_ROOT / relative).is_file(), (component["id"], relative)
