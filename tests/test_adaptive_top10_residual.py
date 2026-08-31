from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from joblib import dump  # type: ignore[import-untyped]

from ghostlab.retrieval.adaptive_residual import (
    ADAPTIVE_RESIDUAL_FEATURES,
    ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
    ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
    TECHNIQUE_ID,
    AdaptiveResidualPolicy,
    AdaptiveTop10ResidualReranker,
    parent_config_sha256,
    sha256_file,
)
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.training.adaptive_residual import (
    AdaptiveResidualTurn,
    config_with_adaptive_residual_asset,
    fit_adaptive_residual_asset,
)


@dataclass(frozen=True)
class ReverseRankModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = features[:, 0] / 10.0
        return np.column_stack((1.0 - positive, positive))


def _catalog(path: Path) -> None:
    rows = [
        {"parent_asin": f"P{index}", "title": f"product {index}"}
        for index in range(1, 11)
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def _verified_reranker(tmp_path: Path) -> AdaptiveTop10ResidualReranker:
    catalog = tmp_path / "catalog.jsonl"
    _catalog(catalog)
    asset = tmp_path / "residual.joblib"
    dump(
        {
            "schema_version": ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
            "technique_id": TECHNIQUE_ID,
            "runtime_family": ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
            "feature_names": ADAPTIVE_RESIDUAL_FEATURES,
            "model": ReverseRankModel(),
        },
        asset,
    )
    asset_hash = sha256_file(asset)
    receipt = tmp_path / "residual.fit_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
                "technique_id": TECHNIQUE_ID,
                "runtime_family": ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
                "model_sha256": asset_hash,
                "parent_config_sha256": parent_config_sha256(AdaptiveHybridConfig()),
                "selected_by_oof": True,
                "holdout_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return AdaptiveTop10ResidualReranker.from_verified_asset(
        catalog,
        asset,
        receipt,
        expected_asset_sha256=asset_hash,
        expected_receipt_sha256=sha256_file(receipt),
        expected_parent_config_sha256=parent_config_sha256(AdaptiveHybridConfig()),
        policy=AdaptiveResidualPolicy(),
    )


def test_adaptive_residual_can_only_permute_exact_top10(tmp_path: Path) -> None:
    reranker = _verified_reranker(tmp_path)
    original = tuple(f"P{index}" for index in range(1, 11))

    decision = reranker.rerank(
        "product",
        original,
        turn=2,
        route="browsing",
        candidate_pool_size=200,
        confirmed_match_count={},
        unknown_constraint_count={},
        soft_preference_count={},
    )

    assert decision.activated
    assert decision.ranking != original
    assert len(decision.ranking) == 10
    assert set(decision.ranking) == set(original)
    with pytest.raises(ValueError, match="at most the final Top-10"):
        reranker.rerank(
            "product",
            (*original, "P11"),
            turn=2,
            route="browsing",
            candidate_pool_size=200,
            confirmed_match_count={},
            unknown_constraint_count={},
            soft_preference_count={},
        )


def test_parent_hash_ignores_only_fitted_d_fields() -> None:
    parent = AdaptiveHybridConfig()
    fitted = parent.model_copy(
        update={
            "extensions": parent.extensions.model_copy(
                update={
                    "top10_residual_enabled": True,
                    "top10_residual_model_path": "artifacts/models/new.joblib",
                    "top10_residual_model_sha256": "a" * 64,
                    "top10_residual_fit_receipt_path": "artifacts/models/new.fit_receipt.json",
                    "top10_residual_fit_receipt_sha256": "b" * 64,
                }
            )
        }
    )

    assert parent_config_sha256(parent) == parent_config_sha256(fitted)


def test_fresh_fit_receipt_is_lineage_safe_and_runtime_verified(tmp_path: Path) -> None:
    config = AdaptiveHybridConfig()
    turns = tuple(
        AdaptiveResidualTurn(
            sample_id=sample_id,
            turn=1,
            features=np.tile(np.arange(1, 10, dtype=np.float64), (10, 1)),
            labels=np.asarray([1, *([0] * 9)], dtype=np.int64),
        )
        for sample_id in ("A", "B", "C")
    )
    artifact = fit_adaptive_residual_asset(
        turns,
        config=config,
        training_ids={"A", "B"},
        validation_ids={"C"},
        group_by_sample={"A": "GA", "B": "GB", "C": "GC"},
        outer_fold=0,
        seed=7,
        project_root=tmp_path,
        output_prefix="artifacts/residual/test",
    )
    receipt = json.loads((tmp_path / artifact.receipt_path).read_text(encoding="utf-8"))

    assert receipt["selected_by_oof"] is True
    assert receipt["holdout_accessed"] is False
    assert receipt["fresh_fit"] is True
    assert sha256_file(tmp_path / artifact.asset_path) == artifact.asset_sha256
    assert sha256_file(tmp_path / artifact.receipt_path) == artifact.receipt_sha256
    fitted = config_with_adaptive_residual_asset(config, artifact)
    assert fitted.extensions.top10_residual_enabled


def test_fit_rejects_lineage_group_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="leaks a lineage group"):
        fit_adaptive_residual_asset(
            (),
            config=AdaptiveHybridConfig(),
            training_ids={"A"},
            validation_ids={"B"},
            group_by_sample={"A": "SAME", "B": "SAME"},
            outer_fold=0,
            seed=1,
            project_root=tmp_path,
            output_prefix="artifacts/residual/leak",
        )
