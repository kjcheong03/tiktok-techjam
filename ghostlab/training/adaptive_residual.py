from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from joblib import dump  # type: ignore[import-untyped]
from numpy.typing import NDArray

from evaluator.local_evaluator import catalog_index, evaluate
from ghostlab.retrieval.adaptive_residual import (
    ADAPTIVE_RESIDUAL_FEATURES,
    ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
    ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
    TECHNIQUE_ID,
    AdaptiveResidualFeatureStore,
    parent_config_sha256,
    sha256_file,
)
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.training.adaptive_lineage import AdaptiveLineageManifest
from starter.agent import Agent


def _ids_hash(values: set[str] | frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


@dataclass(frozen=True)
class AdaptiveResidualTurn:
    sample_id: str
    turn: int
    features: NDArray[np.float64]
    labels: NDArray[np.int64]


@dataclass(frozen=True)
class AdaptiveResidualFitArtifact:
    asset_path: str
    asset_sha256: str
    receipt_path: str
    receipt_sha256: str
    outer_fold: int
    training_sample_ids_sha256: str
    validation_sample_ids_sha256: str


@dataclass(frozen=True)
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        positive = np.full(len(features), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def parent_only_config(config: AdaptiveHybridConfig) -> AdaptiveHybridConfig:
    extensions = config.extensions.model_copy(
        update={
            "top10_residual_enabled": False,
            "top10_residual_model_path": None,
            "top10_residual_model_sha256": None,
            "top10_residual_fit_receipt_path": None,
            "top10_residual_fit_receipt_sha256": None,
        }
    )
    return config.model_copy(update={"extensions": extensions})


def collect_adaptive_residual_turns(
    config: AdaptiveHybridConfig,
    samples: list[dict[str, Any]],
    *,
    catalog_path: str | Path,
    project_root: str | Path,
) -> tuple[tuple[AdaptiveResidualTurn, ...], dict[str, object]]:
    """Replay C unchanged and label its final Top-10 only after each response."""

    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = AdaptiveHybridAgent(
        catalog_path,
        parent_only_config(config),
        project_root=project_root,
    )
    result = evaluate(cast(Agent, agent), samples, catalog_ids, categories, products)
    runtime_ids = tuple(agent.sessions)
    if len(runtime_ids) != len(samples):
        raise RuntimeError("adaptive residual collection lost a session")
    sample_by_runtime = {
        runtime_id: sample
        for runtime_id, sample in zip(runtime_ids, samples, strict=True)
    }
    features = AdaptiveResidualFeatureStore(catalog_path)
    turns = []
    for snapshot in agent.candidate_snapshots:
        ranking = snapshot.pre_residual_top10
        sample = sample_by_runtime.get(snapshot.session_id)
        if sample is None or not ranking:
            continue
        target = str(sample["ground_truth"]["parent_asin"])
        turns.append(
            AdaptiveResidualTurn(
                sample_id=str(sample["sample_id"]),
                turn=snapshot.turn,
                features=features.matrix(
                    snapshot.query,
                    ranking,
                    turn=snapshot.turn,
                    route=snapshot.route,
                    candidate_pool_size=len(snapshot.candidates),
                    confirmed_match_count=snapshot.confirmed_match_count,
                    unknown_constraint_count=snapshot.unknown_constraint_count,
                    soft_preference_count=snapshot.soft_preference_count,
                ),
                labels=np.asarray(
                    [int(identifier == target) for identifier in ranking],
                    dtype=np.int64,
                ),
            )
        )
    return tuple(turns), {
        "parent_result": {
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "recommended_technical_score": result["recommended_technical_score"],
        },
        "sessions": len(samples),
        "turns": len(turns),
        "positive_turns": sum(bool(item.labels.any()) for item in turns),
        "parent_config_sha256": parent_config_sha256(config),
    }


def _fit_model(
    features: NDArray[np.float64], labels: NDArray[np.int64], *, seed: int
) -> object:
    if len(np.unique(labels)) < 2:
        return ConstantProbabilityModel(float(labels.mean()) if len(labels) else 0.0)
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    )
    model.fit(features, labels)
    return model


def fit_adaptive_residual_asset(
    turns: tuple[AdaptiveResidualTurn, ...],
    *,
    config: AdaptiveHybridConfig,
    training_ids: set[str],
    validation_ids: set[str],
    group_by_sample: dict[str, str],
    outer_fold: int,
    seed: int,
    project_root: str | Path,
    output_prefix: str,
) -> AdaptiveResidualFitArtifact:
    """Fit a fresh outer-fold asset with a lineage firewall and signed receipt."""

    if not training_ids or not validation_ids or training_ids & validation_ids:
        raise ValueError("residual outer fold requires disjoint non-empty ID sets")
    training_groups = {group_by_sample[item] for item in training_ids}
    validation_groups = {group_by_sample[item] for item in validation_ids}
    if training_groups & validation_groups:
        raise ValueError("residual outer fold leaks a lineage group")
    selected = [item for item in turns if item.sample_id in training_ids]
    if not selected:
        raise ValueError("residual outer fold has no training turns")
    matrix = np.vstack([item.features for item in selected])
    labels = np.concatenate([item.labels for item in selected])
    model = _fit_model(matrix, labels, seed=seed + outer_fold)
    root = Path(project_root)
    asset_path = root / f"{output_prefix}.fold{outer_fold}.joblib"
    receipt_path = root / f"{output_prefix}.fold{outer_fold}.fit_receipt.json"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    fit_nonce = time.time_ns()
    dump(
        {
            "schema_version": ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
            "technique_id": TECHNIQUE_ID,
            "runtime_family": ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
            "feature_names": ADAPTIVE_RESIDUAL_FEATURES,
            "model": model,
            "fit_nonce": fit_nonce,
        },
        asset_path,
    )
    asset_hash = sha256_file(asset_path)
    parent_hash = parent_config_sha256(config)
    receipt = {
        "schema_version": ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
        "technique_id": TECHNIQUE_ID,
        "runtime_family": ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
        "model_path": asset_path.relative_to(root).as_posix(),
        "model_sha256": asset_hash,
        "parent_config_sha256": parent_hash,
        "outer_fold": outer_fold,
        "seed": seed,
        "fit_nonce": fit_nonce,
        "feature_names": list(ADAPTIVE_RESIDUAL_FEATURES),
        "training_sample_ids_sha256": _ids_hash(training_ids),
        "validation_sample_ids_sha256": _ids_hash(validation_ids),
        "training_lineage_groups_sha256": _ids_hash(training_groups),
        "validation_lineage_groups_sha256": _ids_hash(validation_groups),
        "training_sample_count": len(training_ids),
        "validation_sample_count": len(validation_ids),
        "training_row_count": len(labels),
        "selected_by_oof": True,
        "holdout_accessed": False,
        "fresh_fit": True,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AdaptiveResidualFitArtifact(
        asset_path=asset_path.relative_to(root).as_posix(),
        asset_sha256=asset_hash,
        receipt_path=receipt_path.relative_to(root).as_posix(),
        receipt_sha256=sha256_file(receipt_path),
        outer_fold=outer_fold,
        training_sample_ids_sha256=_ids_hash(training_ids),
        validation_sample_ids_sha256=_ids_hash(validation_ids),
    )


def lineage_outer_folds(
    manifest: AdaptiveLineageManifest, selected_ids: set[str]
) -> tuple[set[str], ...]:
    group_by_sample = manifest.group_by_sample
    fold_by_group = {
        str(group_id): fold_index
        for fold_index, fold in enumerate(manifest.payload["development_outer_folds"])
        for group_id in fold["group_ids"]
    }
    folds: list[set[str]] = [set() for _ in manifest.payload["development_outer_folds"]]
    for sample_id in selected_ids:
        group_id = group_by_sample[sample_id]
        folds[fold_by_group[group_id]].add(sample_id)
    return tuple(item for item in folds if item)


def config_with_adaptive_residual_asset(
    config: AdaptiveHybridConfig, artifact: AdaptiveResidualFitArtifact
) -> AdaptiveHybridConfig:
    extensions = config.extensions.model_copy(
        update={
            "top10_residual_enabled": True,
            "top10_residual_model_path": artifact.asset_path,
            "top10_residual_model_sha256": artifact.asset_sha256,
            "top10_residual_fit_receipt_path": artifact.receipt_path,
            "top10_residual_fit_receipt_sha256": artifact.receipt_sha256,
        }
    )
    return config.model_copy(update={"extensions": extensions})


__all__ = [
    "AdaptiveResidualFitArtifact",
    "AdaptiveResidualTurn",
    "collect_adaptive_residual_turns",
    "config_with_adaptive_residual_asset",
    "fit_adaptive_residual_asset",
    "lineage_outer_folds",
    "parent_only_config",
]
