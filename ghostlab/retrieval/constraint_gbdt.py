from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import local

import numpy as np
from numpy.typing import NDArray

from ghostlab.policy.signals import retrieval_signals
from ghostlab.retrieval.gbdt import (
    METADATA_FEATURES,
    GBDTFeatureStore,
    LambdaMARTModel,
)
from ghostlab.retrieval.sparse import SparseIndex, query_terms
from ghostlab.runtime.experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState, MemoryValue

CONSTRAINT_FEATURES = (
    "active_constraint_coverage_count",
    "active_constraint_coverage_ratio",
    "negative_contradiction_count",
    "negative_contradiction_ratio",
    "explicit_constraint_coverage_ratio",
    "simulator_constraint_coverage_ratio",
    "no_preference_count",
    "no_preference_ratio",
    "invalidated_constraint_count",
    "override_invalidation_present",
    "turn",
    "retrieval_top1_margin",
    "retrieval_normalized_entropy",
)
CONSTRAINT_METADATA_FEATURES = (*METADATA_FEATURES, *CONSTRAINT_FEATURES)


@dataclass(frozen=True)
class ConstraintEvidence:
    terms: frozenset[str]
    provenance: str


@dataclass(frozen=True)
class ConstraintContext:
    positive: tuple[ConstraintEvidence, ...]
    negative: tuple[ConstraintEvidence, ...]
    no_preference_count: int
    asked_count: int
    invalidated_count: int
    turn: int
    retrieval_top1_margin: float
    retrieval_normalized_entropy: float

    @classmethod
    def from_runtime(
        cls,
        state: ConversationState,
        *,
        turn: int,
        retrieval_scores: Sequence[float],
    ) -> ConstraintContext:
        signals = retrieval_signals(retrieval_scores)
        return cls(
            positive=_evidence(state.active_values()),
            negative=_evidence(state.active_values("negative")),
            no_preference_count=len(state.no_preference_attributes),
            asked_count=len(state.asked_attributes),
            invalidated_count=sum(not item.active for item in state.values),
            turn=turn,
            retrieval_top1_margin=float(signals.top1_margin or 0.0),
            retrieval_normalized_entropy=float(signals.normalized_entropy or 0.0),
        )


def _evidence(values: list[MemoryValue]) -> tuple[ConstraintEvidence, ...]:
    return tuple(
        ConstraintEvidence(
            terms=frozenset(query_terms(item.value, 80)),
            provenance=item.provenance,
        )
        for item in values
        if item.active
    )


def _coverage(
    evidence: tuple[ConstraintEvidence, ...], product_terms: frozenset[str]
) -> tuple[float, float]:
    if not evidence:
        return 0.0, 0.0
    matched_constraints = sum(bool(item.terms & product_terms) for item in evidence)
    term_count = sum(len(item.terms) for item in evidence)
    matched_terms = sum(len(item.terms & product_terms) for item in evidence)
    return float(matched_constraints), matched_terms / max(1, term_count)


class ConstraintGBDTFeatureStore(GBDTFeatureStore):
    """Metadata features plus runtime-only structured conversation evidence."""

    def constraint_features(
        self, identifier: str, context: ConstraintContext
    ) -> dict[str, float]:
        product = self.products.get(identifier)
        product_terms = (
            frozenset().union(*product.field_terms)
            if product is not None
            else frozenset()
        )
        positive_count, positive_ratio = _coverage(context.positive, product_terms)
        negative_count, negative_ratio = _coverage(context.negative, product_terms)
        explicit = tuple(
            item for item in context.positive if item.provenance == "explicit"
        )
        simulator = tuple(
            item for item in context.positive if item.provenance == "simulator_answer"
        )
        _, explicit_ratio = _coverage(explicit, product_terms)
        _, simulator_ratio = _coverage(simulator, product_terms)
        return {
            "active_constraint_coverage_count": positive_count,
            "active_constraint_coverage_ratio": positive_ratio,
            "negative_contradiction_count": negative_count,
            "negative_contradiction_ratio": negative_ratio,
            "explicit_constraint_coverage_ratio": explicit_ratio,
            "simulator_constraint_coverage_ratio": simulator_ratio,
            # No-preference is deliberately neutral, never candidate-negative.
            "no_preference_count": float(context.no_preference_count),
            "no_preference_ratio": context.no_preference_count
            / max(1, context.asked_count),
            # Inactive values are counted as context but never used for coverage.
            "invalidated_constraint_count": float(context.invalidated_count),
            "override_invalidation_present": float(context.invalidated_count > 0),
            "turn": float(context.turn),
            "retrieval_top1_margin": context.retrieval_top1_margin,
            "retrieval_normalized_entropy": context.retrieval_normalized_entropy,
        }

    def contextual_matrix(
        self,
        query: str,
        ranking: list[str] | tuple[str, ...],
        context: ConstraintContext,
        feature_names: tuple[str, ...] = CONSTRAINT_METADATA_FEATURES,
    ) -> NDArray[np.float64]:
        unknown = set(feature_names) - set(CONSTRAINT_METADATA_FEATURES)
        if unknown:
            raise ValueError(f"unknown constraint GBDT features: {sorted(unknown)}")
        base = self.matrix(query, ranking, METADATA_FEATURES)
        if not ranking:
            return np.empty((0, len(feature_names)), dtype=np.float64)
        rows = []
        for index, identifier in enumerate(ranking):
            values = {
                **dict(zip(METADATA_FEATURES, base[index], strict=True)),
                **self.constraint_features(identifier, context),
            }
            rows.append([values[name] for name in feature_names])
        return np.asarray(rows, dtype=np.float64)


class ConstraintAwareLambdaMARTReranker:
    def __init__(
        self, features: ConstraintGBDTFeatureStore, model: LambdaMARTModel
    ) -> None:
        self.features = features
        self.model = model

    def rerank_with_context(
        self,
        query: str,
        ranking: list[str],
        *,
        state: ConversationState,
        turn: int,
        retrieval_scores: Sequence[float],
        rerank_k: int = 50,
    ) -> list[str]:
        head = ranking[:rerank_k]
        if len(head) < 2:
            return list(ranking)
        context = ConstraintContext.from_runtime(
            state, turn=turn, retrieval_scores=retrieval_scores
        )
        matrix = self.features.contextual_matrix(
            query, head, context, self.model.feature_names
        )
        predictions = self.model.predict(matrix)
        original_ranks = {identifier: rank for rank, identifier in enumerate(head)}
        scores = {
            identifier: float(score)
            for identifier, score in zip(head, predictions, strict=True)
        }
        ordered = sorted(
            head,
            key=lambda identifier: (
                -scores[identifier],
                original_ranks[identifier],
                identifier,
            ),
        )
        return [*ordered, *ranking[rerank_k:]]


class RuntimeConstraintReranker:
    """Bind public runtime state to the ordinary candidate-reranker interface."""

    def __init__(
        self,
        catalog_path: str,
        sparse_weights: tuple[float, float, float, float, float, float],
        features: ConstraintGBDTFeatureStore,
        model: LambdaMARTModel,
    ) -> None:
        self.catalog_path = catalog_path
        self._sparse_local = local()
        self._sparse_local.index = SparseIndex(catalog_path)
        self.sparse_weights = sparse_weights
        self.reranker = ConstraintAwareLambdaMARTReranker(features, model)
        self._invocation: ContextVar[tuple[ConversationState, int] | None] = ContextVar(
            "constraint_reranker_invocation", default=None
        )

    @contextmanager
    def invocation(self, state: ConversationState, turn: int) -> Iterator[None]:
        token = self._invocation.set((state, turn))
        try:
            yield
        finally:
            self._invocation.reset(token)

    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]:
        invocation = self._invocation.get()
        if invocation is None:
            raise RuntimeError("runtime constraint reranker was not bound")
        state, turn = invocation
        if turn <= 0:
            raise ValueError("runtime turn must be positive")
        sparse = getattr(self._sparse_local, "index", None)
        if sparse is None:
            sparse = SparseIndex(self.catalog_path)
            self._sparse_local.index = sparse
        scored = sparse.search(query, 200, self.sparse_weights)
        retrieval_scores = [
            float(item.raw_score) for item in scored.items if item.raw_score is not None
        ]
        return self.reranker.rerank_with_context(
            query,
            ranking,
            state=state,
            turn=turn,
            retrieval_scores=retrieval_scores,
            rerank_k=rerank_k,
        )


class ConstraintAgentAdapter:
    """Preserve the audited ExperimentalAgent while adding bound state context."""

    def __init__(
        self, wrapped: ExperimentalAgent, reranker: RuntimeConstraintReranker
    ) -> None:
        self.wrapped = wrapped
        self.reranker = reranker

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.wrapped.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        state = self.wrapped.sessions[session_id]
        if not isinstance(state, ConversationState):
            raise TypeError("constraint agent requires ConversationState")
        # ExperimentalAgent mutates this same object before invoking its normal
        # learned-reranker hook, so the adapter sees the current observation.
        with self.reranker.invocation(state, turn):
            return self.wrapped.respond(session_id, user_message, turn, top_k)
