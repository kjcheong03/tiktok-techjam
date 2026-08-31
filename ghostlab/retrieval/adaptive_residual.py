from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.cross_encoder import product_passage

ADAPTIVE_RESIDUAL_SCHEMA_VERSION = 2
ADAPTIVE_RESIDUAL_RUNTIME_FAMILY = "adaptive_hybrid_top10_v1"
TECHNIQUE_ID = "ranking.top10_residual_reranker.v2"
ADAPTIVE_RESIDUAL_FEATURES = (
    "original_rank",
    "reciprocal_rank",
    "turn_fraction",
    "route_is_browsing",
    "query_document_overlap",
    "confirmed_match_count",
    "unknown_constraint_count",
    "soft_preference_count",
    "candidate_pool_size_fraction",
)


class ProbabilityModel(Protocol):
    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class AdaptiveResidualPolicy:
    rerank_depth: int = 10
    model_weight: float = 1.0
    minimum_expected_gain: float = 0.0
    minimum_probability_margin: float = 0.0
    maximum_moved_ids: int = 10


@dataclass(frozen=True)
class AdaptiveResidualDecision:
    ranking: tuple[str, ...]
    activated: bool
    reason: str
    predicted_rr_gain: float
    moved_ids: int


def membership_preserving_reorder(
    ranking: Sequence[str],
    probabilities: Sequence[float],
    policy: AdaptiveResidualPolicy,
) -> AdaptiveResidualDecision:
    original = tuple(ranking)
    observed = np.asarray(probabilities, dtype=np.float64)
    if len(original) != len(observed):
        raise ValueError("ranking and probability lengths must match")
    if len(set(original)) != len(original):
        raise ValueError("membership-preserving reranking requires unique IDs")
    if len(original) < 2:
        return AdaptiveResidualDecision(original, False, "too_few_candidates", 0.0, 0)
    if not np.all(np.isfinite(observed)):
        return AdaptiveResidualDecision(
            original, False, "non_finite_probability", 0.0, 0
        )
    depth = min(policy.rerank_depth, len(original))
    head_probabilities = np.clip(observed[:depth], 0.0, 1.0)
    descending = np.sort(head_probabilities)[::-1]
    margin = float(descending[0] - descending[1])
    if margin + 1e-15 < policy.minimum_probability_margin:
        return AdaptiveResidualDecision(original, False, "probability_margin", 0.0, 0)
    reciprocal = 1.0 / np.arange(1, depth + 1, dtype=np.float64)
    combined = (
        policy.model_weight * head_probabilities
        + (1.0 - policy.model_weight) * reciprocal
    )
    proposed_order = np.argsort(-combined, kind="stable")
    original_order = np.arange(depth, dtype=np.int64)
    predicted_gain = float(
        sum(
            float(head_probabilities[index]) / rank
            for rank, index in enumerate(proposed_order, 1)
        )
        - sum(
            float(head_probabilities[index]) / rank
            for rank, index in enumerate(original_order, 1)
        )
    )
    if predicted_gain + 1e-15 < policy.minimum_expected_gain:
        return AdaptiveResidualDecision(
            original, False, "expected_gain", predicted_gain, 0
        )
    proposed_head = tuple(original[index] for index in proposed_order)
    moved = sum(left != right for left, right in zip(original[:depth], proposed_head))
    if moved == 0:
        return AdaptiveResidualDecision(original, False, "unchanged", predicted_gain, 0)
    if moved > policy.maximum_moved_ids:
        return AdaptiveResidualDecision(
            original, False, "movement_limit", predicted_gain, moved
        )
    result = (*proposed_head, *original[depth:])
    if len(result) != len(original) or set(result) != set(original):
        raise RuntimeError("adaptive residual membership invariant violated")
    return AdaptiveResidualDecision(result, True, "activated", predicted_gain, moved)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


class AdaptiveResidualFeatureStore:
    """Runtime-observable features for a final Top-10 permutation."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.document_tokens: dict[str, frozenset[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                self.document_tokens[str(product["parent_asin"])] = _tokens(
                    product_passage(product)
                )

    def matrix(
        self,
        query: str,
        ranking: Sequence[str],
        *,
        turn: int,
        route: str,
        candidate_pool_size: int,
        confirmed_match_count: Mapping[str, int],
        unknown_constraint_count: Mapping[str, int],
        soft_preference_count: Mapping[str, int],
    ) -> NDArray[np.float64]:
        query_tokens = _tokens(query)
        rows = []
        for index, identifier in enumerate(ranking):
            document = self.document_tokens.get(identifier, frozenset())
            overlap = len(query_tokens & document) / max(1, len(query_tokens))
            rank = index + 1
            rows.append(
                (
                    float(rank),
                    1.0 / rank,
                    min(1.0, max(0.0, turn / 10.0)),
                    float(route == "browsing"),
                    overlap,
                    float(confirmed_match_count.get(identifier, 0)),
                    float(unknown_constraint_count.get(identifier, 0)),
                    float(soft_preference_count.get(identifier, 0)),
                    min(1.0, max(0.0, candidate_pool_size / 400.0)),
                )
            )
        return np.asarray(rows, dtype=np.float64)


class AdaptiveTop10ResidualReranker:
    """Receipt-verified model that can only permute the supplied Top-10."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        model: ProbabilityModel,
        feature_names: tuple[str, ...],
        policy: AdaptiveResidualPolicy,
    ) -> None:
        if feature_names != ADAPTIVE_RESIDUAL_FEATURES:
            raise ValueError("adaptive residual feature schema mismatch")
        self.features = AdaptiveResidualFeatureStore(catalog_path)
        self.model = model
        self.feature_names = feature_names
        self.policy = policy

    @classmethod
    def from_verified_asset(
        cls,
        catalog_path: str | Path,
        asset_path: str | Path,
        receipt_path: str | Path,
        *,
        expected_asset_sha256: str,
        expected_receipt_sha256: str,
        expected_parent_config_sha256: str,
        policy: AdaptiveResidualPolicy,
    ) -> AdaptiveTop10ResidualReranker:
        from joblib import load  # type: ignore[import-untyped]

        asset = Path(asset_path)
        receipt_file = Path(receipt_path)
        if sha256_file(asset) != expected_asset_sha256:
            raise ValueError("adaptive residual asset hash mismatch")
        if sha256_file(receipt_file) != expected_receipt_sha256:
            raise ValueError("adaptive residual fit receipt hash mismatch")
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        required_receipt = {
            "schema_version": ADAPTIVE_RESIDUAL_SCHEMA_VERSION,
            "technique_id": TECHNIQUE_ID,
            "runtime_family": ADAPTIVE_RESIDUAL_RUNTIME_FAMILY,
            "model_sha256": expected_asset_sha256,
            "parent_config_sha256": expected_parent_config_sha256,
            "selected_by_oof": True,
            "holdout_accessed": False,
        }
        for name, expected in required_receipt.items():
            if receipt.get(name) != expected:
                raise ValueError(f"adaptive residual receipt mismatch: {name}")
        payload = load(asset)
        if not isinstance(payload, dict):
            raise TypeError("adaptive residual asset must contain a dictionary")
        if payload.get("schema_version") != ADAPTIVE_RESIDUAL_SCHEMA_VERSION:
            raise ValueError("old or unsupported residual asset schema")
        if payload.get("technique_id") != TECHNIQUE_ID:
            raise ValueError("residual asset belongs to another technique")
        if payload.get("runtime_family") != ADAPTIVE_RESIDUAL_RUNTIME_FAMILY:
            raise ValueError("residual asset belongs to another runtime family")
        raw_features = payload.get("feature_names")
        model = payload.get("model")
        if not isinstance(raw_features, (list, tuple)) or model is None:
            raise TypeError("adaptive residual asset is incomplete")
        return cls(
            catalog_path,
            model=cast(ProbabilityModel, model),
            feature_names=tuple(str(item) for item in raw_features),
            policy=policy,
        )

    def rerank(
        self,
        query: str,
        ranking: Sequence[str],
        *,
        turn: int,
        route: str,
        candidate_pool_size: int,
        confirmed_match_count: Mapping[str, int],
        unknown_constraint_count: Mapping[str, int],
        soft_preference_count: Mapping[str, int],
    ) -> AdaptiveResidualDecision:
        original = tuple(ranking)
        if len(original) > 10:
            raise ValueError("adaptive residual accepts at most the final Top-10")
        matrix = self.features.matrix(
            query,
            original,
            turn=turn,
            route=route,
            candidate_pool_size=candidate_pool_size,
            confirmed_match_count=confirmed_match_count,
            unknown_constraint_count=unknown_constraint_count,
            soft_preference_count=soft_preference_count,
        )
        probabilities = np.asarray(self.model.predict_proba(matrix), dtype=np.float64)
        if probabilities.shape != (len(original), 2):
            raise ValueError("adaptive residual probability contract mismatch")
        decision = membership_preserving_reorder(
            original, probabilities[:, 1].tolist(), self.policy
        )
        if len(decision.ranking) != len(original) or set(decision.ranking) != set(
            original
        ):
            raise RuntimeError("adaptive residual changed Top-10 membership")
        return decision


def parent_config_sha256(config: Any) -> str:
    """Hash C with every D-only field reset, independent of fitted fold assets."""

    value = config.model_dump(mode="python")
    extensions = dict(value["extensions"])
    defaults = {
        "top10_residual_enabled": False,
        "top10_residual_model_path": None,
        "top10_residual_model_sha256": None,
        "top10_residual_fit_receipt_path": None,
        "top10_residual_fit_receipt_sha256": None,
    }
    for name, replacement in defaults.items():
        if name in extensions:
            extensions[name] = replacement
    value["extensions"] = extensions
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ADAPTIVE_RESIDUAL_FEATURES",
    "ADAPTIVE_RESIDUAL_RUNTIME_FAMILY",
    "ADAPTIVE_RESIDUAL_SCHEMA_VERSION",
    "TECHNIQUE_ID",
    "AdaptiveResidualDecision",
    "AdaptiveResidualFeatureStore",
    "AdaptiveResidualPolicy",
    "AdaptiveTop10ResidualReranker",
    "parent_config_sha256",
    "sha256_file",
]
