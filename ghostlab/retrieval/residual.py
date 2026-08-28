from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.constraint_gbdt import (
    CONSTRAINT_METADATA_FEATURES,
    ConstraintContext,
    ConstraintGBDTFeatureStore,
)
from ghostlab.retrieval.gbdt import METADATA_FEATURES
from ghostlab.state.memory import ConversationState

if TYPE_CHECKING:
    from ghostlab.runtime.unified_experimental import ExperimentalAgent

DERIVED_FEATURES = (
    "parent_model_score",
    "parent_model_score_margin_to_top",
    "route_is_fallback",
)
RESIDUAL_FEATURES = (*CONSTRAINT_METADATA_FEATURES, *DERIVED_FEATURES)
RANK_FEATURES = (
    "original_rank",
    "rank_percentile",
    "reciprocal_rank",
    "turn",
    "retrieval_top1_margin",
    "retrieval_normalized_entropy",
    *DERIVED_FEATURES,
)
FEATURE_SETS = {
    "rank": RANK_FEATURES,
    "metadata": (*METADATA_FEATURES, *DERIVED_FEATURES),
    "full_context": RESIDUAL_FEATURES,
}
TECHNIQUE_ID = "ranking.top10_residual_reranker.v2"


class ProbabilityModel(Protocol):
    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ResidualPolicy:
    """Runtime-safe controls for a membership-preserving Top-K permutation."""

    rerank_depth: int = 10
    model_weight: float = 1.0
    minimum_expected_gain: float = 0.0
    minimum_probability_margin: float = 0.0
    maximum_moved_ids: int = 10

    def __post_init__(self) -> None:
        if self.rerank_depth < 2:
            raise ValueError("rerank_depth must be at least two")
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError("model_weight must be within [0, 1]")
        if self.minimum_expected_gain < 0.0:
            raise ValueError("minimum_expected_gain cannot be negative")
        if self.minimum_probability_margin < 0.0:
            raise ValueError("minimum_probability_margin cannot be negative")
        if self.maximum_moved_ids < 2:
            raise ValueError("maximum_moved_ids must be at least two")


@dataclass(frozen=True)
class ResidualDecision:
    ranking: tuple[str, ...]
    activated: bool
    reason: str
    predicted_rr_gain: float
    moved_ids: int


def _expected_reciprocal_rank(
    probabilities: NDArray[np.float64], order: NDArray[np.int64]
) -> float:
    return float(
        sum(
            float(probabilities[candidate_index]) / rank
            for rank, candidate_index in enumerate(order, 1)
        )
    )


def membership_preserving_reorder(
    ranking: Sequence[str],
    probabilities: Sequence[float],
    policy: ResidualPolicy,
) -> ResidualDecision:
    """Reorder a prefix while enforcing exact identifier multiset preservation.

    The function never inserts, removes, or duplicates an identifier. Any failed
    confidence or movement gate returns the original order byte-for-byte.
    """

    original = tuple(ranking)
    observed = np.asarray(probabilities, dtype=np.float64)
    if len(original) != len(observed):
        raise ValueError("ranking and probability lengths must match")
    if len(set(original)) != len(original):
        raise ValueError("membership-preserving reranking requires unique IDs")
    if len(original) < 2:
        return ResidualDecision(original, False, "too_few_candidates", 0.0, 0)
    if not np.all(np.isfinite(observed)):
        return ResidualDecision(original, False, "non_finite_probability", 0.0, 0)

    depth = min(policy.rerank_depth, len(original))
    head_probabilities = np.clip(observed[:depth], 0.0, 1.0)
    if len(head_probabilities) < 2:
        return ResidualDecision(original, False, "too_few_candidates", 0.0, 0)
    descending = np.sort(head_probabilities)[::-1]
    margin = float(descending[0] - descending[1])
    if margin + 1e-15 < policy.minimum_probability_margin:
        return ResidualDecision(original, False, "probability_margin", 0.0, 0)

    reciprocal = 1.0 / np.arange(1, depth + 1, dtype=np.float64)
    combined = (
        policy.model_weight * head_probabilities
        + (1.0 - policy.model_weight) * reciprocal
    )
    proposed_order = np.argsort(-combined, kind="stable")
    original_order = np.arange(depth, dtype=np.int64)
    predicted_gain = _expected_reciprocal_rank(
        head_probabilities, proposed_order
    ) - _expected_reciprocal_rank(head_probabilities, original_order)
    if predicted_gain + 1e-15 < policy.minimum_expected_gain:
        return ResidualDecision(original, False, "expected_gain", predicted_gain, 0)

    proposed_head = tuple(original[index] for index in proposed_order)
    moved = sum(left != right for left, right in zip(original[:depth], proposed_head))
    if moved == 0:
        return ResidualDecision(original, False, "unchanged", predicted_gain, 0)
    if moved > policy.maximum_moved_ids:
        return ResidualDecision(
            original, False, "movement_limit", predicted_gain, moved
        )

    result = (*proposed_head, *original[depth:])
    if len(result) != len(original) or set(result) != set(original):
        raise RuntimeError("membership-preserving invariant violated")
    return ResidualDecision(result, True, "activated", predicted_gain, moved)


class MembershipPreservingResidualReranker:
    """Fold-fitted residual model with runtime-observable features only."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        model: ProbabilityModel,
        feature_names: tuple[str, ...],
        policy: ResidualPolicy,
    ) -> None:
        unknown = set(feature_names) - set(RESIDUAL_FEATURES)
        if unknown or not feature_names:
            raise ValueError(f"invalid residual feature schema: {sorted(unknown)}")
        self.features = ConstraintGBDTFeatureStore(catalog_path)
        self.model = model
        self.feature_names = feature_names
        self.feature_indices = tuple(
            RESIDUAL_FEATURES.index(name) for name in feature_names
        )
        self.policy = policy

    @classmethod
    def from_asset(
        cls,
        catalog_path: str | Path,
        asset_path: str | Path,
        *,
        policy: ResidualPolicy | None = None,
    ) -> MembershipPreservingResidualReranker:
        from joblib import load  # type: ignore[import-untyped]

        payload = load(asset_path)
        if not isinstance(payload, dict):
            raise TypeError("residual asset must contain a dictionary")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported residual asset schema")
        if payload.get("technique_id") != TECHNIQUE_ID:
            raise ValueError("residual asset belongs to another technique")
        raw_features = payload.get("feature_names")
        raw_policy = payload.get("policy")
        model = payload.get("model")
        if not isinstance(raw_features, (list, tuple)) or not all(
            isinstance(item, str) for item in raw_features
        ):
            raise TypeError("residual asset has invalid feature names")
        if not isinstance(raw_policy, dict) or model is None:
            raise TypeError("residual asset is incomplete")
        return cls(
            catalog_path,
            model=cast(ProbabilityModel, model),
            feature_names=tuple(raw_features),
            policy=policy or ResidualPolicy(**raw_policy),
        )

    def _matrix(
        self,
        query: str,
        ranking: tuple[str, ...],
        *,
        state: ConversationState,
        turn: int,
        retrieval_scores: Sequence[float],
    ) -> NDArray[np.float64]:
        context = ConstraintContext.from_runtime(
            state, turn=turn, retrieval_scores=retrieval_scores
        )
        base = self.features.contextual_matrix(
            query, ranking, context, CONSTRAINT_METADATA_FEATURES
        )
        reciprocal = 1.0 / np.arange(1, len(ranking) + 1, dtype=np.float64)
        derived = np.column_stack(
            (
                reciprocal,
                reciprocal - 1.0,
                np.zeros(len(ranking), dtype=np.float64),
            )
        )
        complete = np.hstack((base, derived))
        return complete[:, self.feature_indices]

    def rerank(
        self,
        query: str,
        ranking: Sequence[str],
        *,
        state: ConversationState,
        turn: int,
        retrieval_scores: Sequence[float],
    ) -> ResidualDecision:
        original = tuple(ranking)
        if len(original) < 2:
            return ResidualDecision(original, False, "too_few_candidates", 0.0, 0)
        matrix = self._matrix(
            query,
            original,
            state=state,
            turn=turn,
            retrieval_scores=retrieval_scores,
        )
        probabilities = np.asarray(self.model.predict_proba(matrix), dtype=np.float64)[
            :, 1
        ]
        return membership_preserving_reorder(
            original, probabilities.tolist(), self.policy
        )


class ResidualAgentAdapter:
    """Apply a fitted residual permutation after an experimental parent response."""

    def __init__(
        self,
        parent: ExperimentalAgent,
        reranker: MembershipPreservingResidualReranker,
    ) -> None:
        self.parent = parent
        self.reranker = reranker

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.parent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        response = self.parent.respond(session_id, user_message, turn, top_k)
        original = tuple(str(item) for item in response.get("recommendations", ()))
        state = self.parent.sessions.get(session_id)
        runtime = self.parent.last_runtime_inputs.get(session_id)
        if not isinstance(state, ConversationState) or runtime is None:
            return response
        query, retrieval_scores = runtime
        decision = self.reranker.rerank(
            query,
            original,
            state=state,
            turn=turn,
            retrieval_scores=retrieval_scores,
        )
        if self.parent.retrieval_trace:
            self.parent.retrieval_trace[-1]["residual"] = {
                "activated": decision.activated,
                "reason": decision.reason,
                "predicted_rr_gain": decision.predicted_rr_gain,
                "moved_ids": decision.moved_ids,
            }
            self.parent.retrieval_trace[-1]["ranked"] = list(decision.ranking)
        return {**response, "recommendations": list(decision.ranking)}
