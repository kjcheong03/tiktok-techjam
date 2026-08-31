from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from baseline.retrieval import DenseRetriever, KeywordRetriever, reciprocal_rank_fusion
from baseline.state import SessionState, fixed_question_for_turn
from ghostlab.policy.adaptive_questions import (
    AdaptiveQuestionPolicy,
    QuestionContext,
)
from ghostlab.policy.learned_questions import (
    LinearActionValueModel,
    QuestionAction,
    legal_question_actions,
    observable_question_features,
)
from ghostlab.policy.signals import retrieval_signals
from ghostlab.retrieval.filters import CoverageAwareFilter
from ghostlab.retrieval.fusion import weighted_fuse_ids
from ghostlab.retrieval.profile import ProfilePriorReranker
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.rerank import LinearLexicalReranker
from ghostlab.retrieval.sparse import OFFICIAL_WEIGHTS, SparseIndex
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.memory import ConversationState

StateVariant = Literal["current", "raw_history", "single", "multi", "compressed"]
QuestionVariant = Literal[
    "none",
    "fixed",
    "sequence",
    "missing_priority",
    "feature_first",
    "uncertainty",
    "other_always",
    "adaptive",
    "learned",
]
FEATURE_FIRST = ("feature", "use_case", "material", "style", "color", "budget", "size")


class CandidateReranker(Protocol):
    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]: ...


class ExperimentalAgent:
    """Research agent whose dimensions are explicit constructor switches."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        state_variant: StateVariant = "multi",
        question_variant: QuestionVariant = "missing_priority",
        retrieval_route: Literal["keyword", "dense", "rrf", "weighted"] = "keyword",
        negative_evidence: bool = True,
        provenance: bool = True,
        override_invalidation: bool = True,
        sparse_weight: float = 0.75,
        dense_weight: float = 0.25,
        reranker: Literal["none", "linear"] = "none",
        question_order: tuple[str, ...] | None = None,
        repeat_last_question: bool = False,
        adaptive_policy: AdaptiveQuestionPolicy | None = None,
        learned_question_model: LinearActionValueModel | None = None,
        structured_filter: bool = False,
        profile_prior_weight: float = 0.0,
        quality_prior_weight: float = 0.0,
        sparse_weights: tuple[float, float, float, float, float, float] | None = None,
        learned_reranker: CandidateReranker | None = None,
        quality_prior: CatalogQualityReranker | None = None,
    ) -> None:
        self.state_variant = state_variant
        self.question_variant = question_variant
        self.retrieval_route = retrieval_route
        self.keyword = KeywordRetriever(catalog_path)
        self.dense = (
            DenseRetriever(catalog_path)
            if retrieval_route in {"dense", "rrf", "weighted"}
            else None
        )
        self.catalog_ids = self._read_ids(catalog_path)
        self.negative_evidence = negative_evidence
        self.provenance = provenance
        self.override_invalidation = override_invalidation
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.reranker = (
            LinearLexicalReranker(catalog_path) if reranker == "linear" else None
        )
        self.question_order = question_order
        self.repeat_last_question = repeat_last_question
        self.adaptive_policy = adaptive_policy or AdaptiveQuestionPolicy()
        self.learned_question_model = learned_question_model
        if question_variant == "learned" and learned_question_model is None:
            raise ValueError("learned question policy requires a fitted model")
        if sparse_weights is not None and any(
            weight < 0.0 for weight in sparse_weights
        ):
            raise ValueError("sparse field weights cannot be negative")
        self.sparse_weights = sparse_weights
        self.field_sparse = (
            SparseIndex(catalog_path) if sparse_weights is not None else None
        )
        self.signal_sparse = (
            self.field_sparse or SparseIndex(catalog_path)
            if question_variant in {"adaptive", "learned"}
            else None
        )
        self.structured_filter = (
            CoverageAwareFilter(catalog_path) if structured_filter else None
        )
        self.profile_prior_weight = profile_prior_weight
        self.profile_prior = (
            ProfilePriorReranker(catalog_path) if profile_prior_weight > 0.0 else None
        )
        self.quality_prior_weight = quality_prior_weight
        self.quality_prior = (
            quality_prior or CatalogQualityReranker(catalog_path)
            if quality_prior_weight > 0.0
            else None
        )
        self.learned_reranker = learned_reranker
        self.sessions: dict[str, ConversationState | SessionState] = {}
        self.stopped_sessions: set[str] = set()
        self.last_runtime_inputs: dict[str, tuple[str, list[float]]] = {}
        self.question_trace: list[dict[str, object]] = []

    @staticmethod
    def _read_ids(path: str | Path) -> set[str]:
        import json

        with Path(path).open(encoding="utf-8") as handle:
            return {
                str(json.loads(line)["parent_asin"]) for line in handle if line.strip()
            }

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.keyword.reset(session_id, user_profile)
        if self.profile_prior is not None:
            self.profile_prior.reset(session_id, user_profile)
        if self.state_variant == "single":
            self.sessions[session_id] = SessionState(session_id, user_profile)
        else:
            self.sessions[session_id] = ConversationState(
                session_id,
                user_profile,
                multi_value=self.state_variant in {"multi", "compressed"},
                negative_evidence=self.negative_evidence,
                provenance_enabled=self.provenance,
                override_invalidation=self.override_invalidation,
            )
        self.stopped_sessions.discard(session_id)
        self.last_runtime_inputs.pop(session_id, None)

    def _query_and_question(
        self, state: ConversationState | SessionState, message: str, turn: int
    ) -> tuple[str, str | None]:
        if self.state_variant == "current":
            query = message
        elif self.state_variant == "raw_history":
            assert isinstance(state, ConversationState)
            state.observe(message, turn)
            query = ". ".join(state.messages)
        else:
            state.observe(message, turn)
            query = (
                state.build_query(compressed=self.state_variant == "compressed")
                if isinstance(state, ConversationState)
                else state.build_query()
            )
        if self.question_variant == "none":
            question = None
        elif self.question_variant in {"fixed", "sequence"}:
            order = self.question_order
            if order is None:
                question = fixed_question_for_turn(turn)
            elif turn <= len(order):
                question = order[turn - 1]
            elif self.repeat_last_question and order:
                question = order[-1]
            else:
                question = None
        elif self.question_variant == "feature_first":
            question = (
                state.choose_question(FEATURE_FIRST)
                if isinstance(state, ConversationState)
                else state.choose_question()
            )
        elif self.question_variant == "uncertainty":
            question = (
                state.choose_question()
                if isinstance(state, ConversationState) and turn <= 4
                else None
            )
        elif self.question_variant == "other_always":
            question = "other"
        elif self.question_variant in {"adaptive", "learned"}:
            question = None
        else:
            question = state.choose_question()
        if self.question_variant not in {"adaptive", "learned"} and isinstance(
            state, ConversationState
        ):
            if question is not None and (
                not state.asked_attributes or state.asked_attributes[-1] != question
            ):
                state.asked_attributes.append(question)
            state.last_asked_attribute = question
        return query, question

    def _adaptive_question(
        self,
        state: ConversationState | SessionState,
        message: str,
        turn: int,
        query: str,
    ) -> tuple[str | None, str]:
        if not isinstance(state, ConversationState):
            return state.choose_question(), "single_state_fallback"
        signals = None
        if self.signal_sparse is not None:
            scored = self.signal_sparse.search(query, 200, OFFICIAL_WEIGHTS)
            signals = retrieval_signals(
                [item.raw_score for item in scored.items if item.raw_score is not None]
            )
        decision = self.adaptive_policy.decide(
            QuestionContext(
                turn=turn,
                message=message,
                active_attributes=frozenset(
                    item.attribute for item in state.active_values()
                ),
                asked_attributes=frozenset(state.asked_attributes),
                no_preference_attributes=frozenset(state.no_preference_attributes),
                last_asked_attribute=state.last_asked_attribute,
                retrieval=signals,
            )
        )
        question = decision.ask_attribute
        if question is not None:
            state.asked_attributes.append(question)
        state.last_asked_attribute = question
        return question, decision.reason

    @staticmethod
    def _positive_constraints(
        state: ConversationState | SessionState,
    ) -> dict[str, list[str]]:
        if not isinstance(state, ConversationState):
            return {}
        result: dict[str, list[str]] = {}
        for item in state.active_values():
            result.setdefault(item.attribute, []).append(item.value)
        return result

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        state = self.sessions[session_id]
        query, question = self._query_and_question(state, user_message, turn)
        retrieval_scores: list[float] = []
        if self.retrieval_route == "keyword":
            if self.field_sparse is None or self.sparse_weights is None:
                ranked = self.keyword.search(session_id, query, turn, 200)
                retrieval_scores = [1.0 / rank for rank in range(1, len(ranked) + 1)]
            else:
                sparse_result = self.field_sparse.search(
                    query, 200, self.sparse_weights
                )
                ranked = [item.parent_asin for item in sparse_result.items]
                retrieval_scores = [
                    float(item.raw_score)
                    for item in sparse_result.items
                    if item.raw_score is not None
                ]
        elif self.retrieval_route == "dense":
            assert self.dense is not None
            ranked = self.dense.search(query, 200)
        elif self.retrieval_route == "rrf":
            assert self.dense is not None
            ranked = reciprocal_rank_fusion(
                [
                    self.keyword.search(session_id, query, turn, 200),
                    self.dense.search(query, 200),
                ],
                limit=200,
            )
        else:
            assert self.dense is not None
            ranked = weighted_fuse_ids(
                self.keyword.search(session_id, query, turn, 200),
                self.dense.search(query, 200),
                sparse_weight=self.sparse_weight,
                dense_weight=self.dense_weight,
                limit=200,
            )
        if self.structured_filter is not None:
            ranked = self.structured_filter.apply(
                ranked,
                self._positive_constraints(state),
                minimum_results=max(10, top_k),
            )
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked)
        if self.profile_prior is not None:
            ranked = self.profile_prior.rerank(
                session_id,
                ranked,
                weight=self.profile_prior_weight,
            )
        if self.quality_prior is not None:
            ranked = self.quality_prior.rerank(
                ranked,
                weight=self.quality_prior_weight,
            )
        if self.learned_reranker is not None:
            ranked = self.learned_reranker.rerank(query, ranked)
        self.last_runtime_inputs[session_id] = (query, retrieval_scores)
        question_reason: str = self.question_variant
        if self.question_variant == "adaptive":
            question, question_reason = self._adaptive_question(
                state, user_message, turn, query
            )
        action_values: dict[QuestionAction, float] | None = None
        feature_values: dict[str, float] | None = None
        legal_actions: tuple[QuestionAction, ...] | None = None
        if self.question_variant == "learned":
            if not isinstance(state, ConversationState):
                raise TypeError("learned question policy requires conversation state")
            feature_values = observable_question_features(
                state,
                message=user_message,
                query=query,
                turn=turn,
                retrieval_scores=retrieval_scores,
            )
            legal_actions = legal_question_actions(state)
            if session_id in self.stopped_sessions:
                question = None
                action_values = {None: 0.0}
                question_reason = "absorbing_stop"
            else:
                assert self.learned_question_model is not None
                question, action_values = self.learned_question_model.decide(
                    feature_values, legal_actions
                )
                question_reason = "linear_action_value"
                if question is None:
                    self.stopped_sessions.add(session_id)
            if question is not None:
                state.asked_attributes.append(question)
            state.last_asked_attribute = question
        self.question_trace.append(
            {
                "session_id": session_id,
                "turn": turn,
                "ask_attribute": question,
                "reason": question_reason,
                "features": feature_values,
                "legal_actions": legal_actions,
                "action_values": action_values,
            }
        )
        message = (
            "Here are the closest matches based on what you have shared."
            if question is None
            else f"Do you have a preference for {question.replace('_', ' ')}?"
        )
        return normalize_response(
            {
                "message": message,
                "ask_attribute": question,
                "recommendations": ranked,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            },
            self.catalog_ids,
            top_k,
        )

    def override_last_question(self, session_id: str, attribute: str | None) -> None:
        """Research hook used immediately after a first-action counterfactual branch."""
        state = self.sessions[session_id]
        previous = state.last_asked_attribute
        if (
            isinstance(state, ConversationState)
            and previous is not None
            and state.asked_attributes
            and state.asked_attributes[-1] == previous
        ):
            state.asked_attributes.pop()
        if isinstance(state, ConversationState) and attribute is not None:
            state.asked_attributes.append(attribute)
        state.last_asked_attribute = attribute
