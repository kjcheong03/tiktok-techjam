from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

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
from ghostlab.retrieval.cross_encoder import CrossEncoderReranker
from ghostlab.retrieval.diversify import (
    DiversificationContext,
    DiversificationDecision,
)
from ghostlab.retrieval.filters import CoverageAwareFilter
from ghostlab.retrieval.fusion import sparse_first_union_ids, weighted_fuse_ids
from ghostlab.retrieval.profile import ProfilePriorReranker
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.rerank import LinearLexicalReranker
from ghostlab.retrieval.sparse import OFFICIAL_WEIGHTS, SparseIndex
from ghostlab.retrieval.sparse_semantic_fusion import sparse_semantic_union_ids
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.memory import ConversationState
from ghostlab.state.query import QueryVariant, build_query
from ghostlab.state.query_expansion import QueryExpansion
from ghostlab.state.v2_view import V2SessionController, V2StateView

if TYPE_CHECKING:
    from ghostlab.policy.calibrated_router import CalibratedRouteModel
    from ghostlab.policy.candidate_statistics import CandidateFacetStore
    from ghostlab.policy.eig_questions import CandidateEIGPolicy
    from ghostlab.policy.joint_policy import JointPolicyDecision
    from ghostlab.runtime.component_fallback import ComponentFallback
    from ghostlab.state.normalization import CatalogStateNormalizer

StateVariant = Literal[
    "current",
    "raw_history",
    "single",
    "multi",
    "compressed",
    "baseline_v2",
]
RecommendationHistory = Literal["off", "correction_scoped"]
ActivationMode = Literal["always", "uncertain"]
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
    "candidate_eig",
    "reward_voi",
    "joint_observable",
    "distilled_joint",
]
FEATURE_FIRST = ("feature", "use_case", "material", "style", "color", "budget", "size")


class CandidateReranker(Protocol):
    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]: ...


class DenseCandidateRetriever(Protocol):
    def search(self, query: str, limit: int) -> object: ...


class QueryExpander(Protocol):
    def expand(self, query: str, ranking: list[str]) -> QueryExpansion: ...


class CandidateDiversifier(Protocol):
    def rerank(
        self, ranking: list[str], context: DiversificationContext
    ) -> DiversificationDecision: ...


class JointCandidatePolicy(Protocol):
    @property
    def possible_routes(self) -> frozenset[str]: ...

    def decide(
        self, state: ConversationState, features: dict[str, float]
    ) -> JointPolicyDecision: ...


class ExperimentalAgent:
    """Research agent whose dimensions are explicit constructor switches."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        state_variant: StateVariant = "multi",
        question_variant: QuestionVariant = "missing_priority",
        retrieval_route: Literal[
            "keyword",
            "dense",
            "rrf",
            "weighted",
            "sparse_first_union",
            "learned_sparse_union",
            "late_interaction_union",
        ] = "keyword",
        negative_evidence: bool = True,
        provenance: bool = True,
        override_invalidation: bool = True,
        sparse_weight: float = 0.75,
        dense_weight: float = 0.25,
        retrieval_k: int = 200,
        rrf_constant: int = 60,
        dense_activation: ActivationMode = "always",
        dense_activation_min_entropy: float = 0.0,
        reranker: Literal["none", "linear"] = "none",
        question_order: tuple[str, ...] | None = None,
        repeat_last_question: bool = False,
        adaptive_policy: AdaptiveQuestionPolicy | None = None,
        learned_question_model: LinearActionValueModel | None = None,
        eig_policy: CandidateEIGPolicy | None = None,
        eig_candidate_k: int = 100,
        question_max_turn: int = 10,
        joint_policy: JointCandidatePolicy | None = None,
        routing_variant: Literal["off", "calibrated"] = "off",
        calibrated_router: CalibratedRouteModel | None = None,
        component_fallback: ComponentFallback | None = None,
        normalizer: Literal["off", "catalog_v1"] = "off",
        catalog_normalizer: CatalogStateNormalizer | None = None,
        query_variant: QueryVariant | None = None,
        structured_filter: bool = False,
        profile_prior_weight: float = 0.0,
        profile_prior_max_turn: int = 10,
        quality_prior_weight: float = 0.0,
        sparse_weights: tuple[float, float, float, float, float, float] | None = None,
        learned_reranker: CandidateReranker | None = None,
        learned_rerank_k: int = 50,
        quality_prior: CatalogQualityReranker | None = None,
        dense_retriever: DenseCandidateRetriever | None = None,
        semantic_rescue_retriever: DenseCandidateRetriever | None = None,
        semantic_rescue_weight: float = 0.25,
        cross_encoder_reranker: CrossEncoderReranker | None = None,
        cross_encoder_weight: float = 0.0,
        cross_encoder_rerank_k: int = 20,
        cross_encoder_activation: ActivationMode = "always",
        cross_encoder_min_entropy: float = 0.0,
        cross_encoder_min_turn: int = 1,
        query_expander: QueryExpander | None = None,
        query_expansion_activation: ActivationMode = "always",
        query_expansion_min_entropy: float = 0.0,
        diversifier: CandidateDiversifier | None = None,
        recommendation_history: RecommendationHistory = "off",
    ) -> None:
        self.state_variant = state_variant
        self.question_variant = question_variant
        self.retrieval_route = retrieval_route
        self.keyword = KeywordRetriever(catalog_path)
        self.joint_policy = joint_policy
        if question_variant in {"joint_observable", "distilled_joint"} and (
            joint_policy is None
        ):
            raise ValueError("joint question policy requires a compiled policy")
        if routing_variant == "calibrated" and calibrated_router is None:
            raise ValueError("calibrated routing requires a fold-fitted model")
        if routing_variant == "off" and calibrated_router is not None:
            raise ValueError("calibrated router asset requires its explicit switch")
        self.routing_variant = routing_variant
        self.calibrated_router = calibrated_router
        self.component_fallback = component_fallback
        if normalizer == "catalog_v1" and catalog_normalizer is None:
            raise ValueError("catalog normalization requires a local ontology")
        if normalizer == "off" and catalog_normalizer is not None:
            raise ValueError("catalog normalizer asset requires its explicit switch")
        self.normalizer = normalizer
        self.catalog_normalizer = catalog_normalizer
        needs_dense = (
            retrieval_route
            in {
                "dense",
                "rrf",
                "weighted",
                "sparse_first_union",
            }
            or bool(
                joint_policy is not None
                and joint_policy.possible_routes & {"dense", "rrf", "weighted_fusion"}
            )
            or bool(
                calibrated_router is not None
                and calibrated_router.possible_routes
                & {"dense", "rrf", "weighted_fusion"}
            )
        )
        self.dense: DenseCandidateRetriever | None = dense_retriever
        if needs_dense and self.dense is None:
            self.dense = DenseRetriever(catalog_path)
        needs_rescue = retrieval_route in {
            "learned_sparse_union",
            "late_interaction_union",
        }
        if needs_rescue and semantic_rescue_retriever is None:
            raise ValueError(f"{retrieval_route} requires a semantic rescue retriever")
        if not 0.0 <= semantic_rescue_weight <= 1.0:
            raise ValueError("semantic rescue weight must be between zero and one")
        self.semantic_rescue = semantic_rescue_retriever
        self.semantic_rescue_weight = semantic_rescue_weight
        self.catalog_ids = self._read_ids(catalog_path)
        self.negative_evidence = negative_evidence
        self.provenance = provenance
        self.override_invalidation = override_invalidation
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.retrieval_k = retrieval_k
        self.rrf_constant = rrf_constant
        self.dense_activation = dense_activation
        self.dense_activation_min_entropy = dense_activation_min_entropy
        self.reranker = (
            LinearLexicalReranker(catalog_path) if reranker == "linear" else None
        )
        self.question_order = question_order
        self.repeat_last_question = repeat_last_question
        self.adaptive_policy = adaptive_policy or AdaptiveQuestionPolicy()
        self.learned_question_model = learned_question_model
        if question_variant == "learned" and learned_question_model is None:
            raise ValueError("learned question policy requires a fitted model")
        if eig_candidate_k <= 0 or eig_candidate_k > 400:
            raise ValueError("EIG candidate depth must be between 1 and 400")
        self.eig_candidate_k = eig_candidate_k
        self.question_max_turn = question_max_turn
        self.candidate_facets: CandidateFacetStore | None = None
        self.eig_policy: CandidateEIGPolicy | None = None
        if question_variant in {"candidate_eig", "reward_voi"}:
            from ghostlab.policy.candidate_statistics import CandidateFacetStore
            from ghostlab.policy.eig_questions import CandidateEIGPolicy

            self.candidate_facets = CandidateFacetStore(catalog_path)
            self.eig_policy = eig_policy or CandidateEIGPolicy()
            if question_variant == "reward_voi" and self.eig_policy.calibration is None:
                raise ValueError("reward VOI requires a fold-fitted calibration")
        self.query_variant = query_variant
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
        self.profile_prior_max_turn = profile_prior_max_turn
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
        if learned_rerank_k <= 0:
            raise ValueError("learned rerank depth must be positive")
        self.learned_rerank_k = learned_rerank_k
        if not 0.0 <= cross_encoder_weight <= 1.0:
            raise ValueError("cross-encoder weight must be between zero and one")
        if cross_encoder_rerank_k <= 0:
            raise ValueError("cross-encoder rerank depth must be positive")
        self.cross_encoder_reranker = cross_encoder_reranker
        self.cross_encoder_weight = cross_encoder_weight
        self.cross_encoder_rerank_k = cross_encoder_rerank_k
        self.cross_encoder_activation = cross_encoder_activation
        self.cross_encoder_min_entropy = cross_encoder_min_entropy
        self.cross_encoder_min_turn = cross_encoder_min_turn
        self.query_expander = query_expander
        self.query_expansion_activation = query_expansion_activation
        self.query_expansion_min_entropy = query_expansion_min_entropy
        self.diversifier = diversifier
        if recommendation_history not in {"off", "correction_scoped"}:
            raise ValueError("unknown recommendation-history mode")
        self.recommendation_history = recommendation_history
        self.sessions: dict[str, ConversationState | SessionState] = {}
        self.v2_controllers: dict[str, V2SessionController] = {}
        self.last_v2_views: dict[str, V2StateView] = {}
        self.shown_recommendations: dict[str, set[str]] = {}
        self.recommendation_epochs: dict[str, int] = {}
        self.stopped_sessions: set[str] = set()
        self.last_runtime_inputs: dict[str, tuple[str, list[float]]] = {}
        self.question_trace: list[dict[str, object]] = []
        self.retrieval_trace: list[dict[str, object]] = []

        if retrieval_k < 10 or retrieval_k > 400:
            raise ValueError("retrieval depth must be between 10 and 400")
        if rrf_constant <= 0:
            raise ValueError("RRF constant must be positive")
        for name, value in {
            "dense_activation_min_entropy": dense_activation_min_entropy,
            "query_expansion_min_entropy": query_expansion_min_entropy,
            "cross_encoder_min_entropy": cross_encoder_min_entropy,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    @staticmethod
    def _activate(
        mode: ActivationMode, *, entropy: float | None, minimum_entropy: float
    ) -> bool:
        """Use only observable retrieval uncertainty; missing signals fail open."""

        return mode == "always" or entropy is None or entropy >= minimum_entropy

    @staticmethod
    def _remove_recorded_question(
        state: ConversationState | SessionState, question: str | None
    ) -> None:
        if (
            question is not None
            and isinstance(state, ConversationState)
            and state.asked_attributes
            and state.asked_attributes[-1] == question
        ):
            state.asked_attributes.pop()
        state.last_asked_attribute = None

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
        elif self.state_variant == "baseline_v2":
            from ghostlab.state.baseline_v2 import StateBaselineV2

            state = StateBaselineV2(
                session_id,
                user_profile,
                multi_value=True,
                negative_evidence=self.negative_evidence,
                provenance_enabled=self.provenance,
                override_invalidation=self.override_invalidation,
            )
            self.sessions[session_id] = state
            self.v2_controllers[session_id] = V2SessionController(state)
        else:
            if self.normalizer == "catalog_v1":
                from ghostlab.state.normalization import NormalizedConversationState

                self.sessions[session_id] = NormalizedConversationState(
                    session_id,
                    user_profile,
                    multi_value=self.state_variant in {"multi", "compressed"},
                    negative_evidence=self.negative_evidence,
                    provenance_enabled=self.provenance,
                    override_invalidation=self.override_invalidation,
                    catalog_normalizer=self.catalog_normalizer,
                )
            else:
                self.sessions[session_id] = ConversationState(
                    session_id,
                    user_profile,
                    multi_value=self.state_variant in {"multi", "compressed"},
                    negative_evidence=self.negative_evidence,
                    provenance_enabled=self.provenance,
                    override_invalidation=self.override_invalidation,
                )
        if self.state_variant != "baseline_v2":
            self.v2_controllers.pop(session_id, None)
        self.last_v2_views.pop(session_id, None)
        self.stopped_sessions.discard(session_id)
        self.last_runtime_inputs.pop(session_id, None)
        self.shown_recommendations[session_id] = set()
        self.recommendation_epochs[session_id] = 0

    def _filter_recommendation_history(
        self,
        session_id: str,
        state: ConversationState | SessionState,
        ranking: list[str],
    ) -> list[str]:
        if self.recommendation_history == "off":
            return ranking
        controller = self.v2_controllers.get(session_id)
        if controller is not None:
            return controller.filter_ranking(ranking)
        epoch = int(getattr(state, "intent_epoch", 0))
        if epoch != self.recommendation_epochs[session_id]:
            self.shown_recommendations[session_id].clear()
            self.recommendation_epochs[session_id] = epoch
        shown = self.shown_recommendations[session_id]
        return [identifier for identifier in ranking if identifier not in shown]

    def _record_recommendation_history(
        self,
        session_id: str,
        response: dict,
    ) -> None:
        if self.recommendation_history == "off":
            return
        identifiers = [
            str(item["parent_asin"])
            for item in response["recommendations"]
            if isinstance(item, dict) and item.get("parent_asin")
        ]
        controller = self.v2_controllers.get(session_id)
        if controller is not None:
            controller.record_shown(identifiers)
            return
        self.shown_recommendations[session_id].update(identifiers)

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
                build_query(state, self.query_variant)
                if isinstance(state, ConversationState)
                and self.query_variant is not None
                else state.build_query(compressed=self.state_variant == "compressed")
                if isinstance(state, ConversationState)
                else state.build_query()
            )
        if (
            self.state_variant == "raw_history"
            and isinstance(state, ConversationState)
            and self.query_variant is not None
        ):
            query = build_query(state, self.query_variant)
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
        elif self.question_variant in {
            "adaptive",
            "learned",
            "candidate_eig",
            "reward_voi",
            "joint_observable",
            "distilled_joint",
        }:
            question = None
        else:
            question = state.choose_question()
        if self.question_variant not in {
            "adaptive",
            "learned",
            "candidate_eig",
            "reward_voi",
            "joint_observable",
            "distilled_joint",
        } and isinstance(state, ConversationState):
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
        state: ConversationState | SessionState | V2StateView,
    ) -> dict[str, list[str]]:
        if isinstance(state, V2StateView):
            return state.positive_constraints()
        if not isinstance(state, ConversationState):
            return {}
        result: dict[str, list[str]] = {}
        for item in state.active_values():
            result.setdefault(item.attribute, []).append(item.value)
        return result

    def _sparse_ids(
        self, session_id: str, query: str, turn: int, limit: int = 200
    ) -> tuple[list[str], list[float]]:
        if self.field_sparse is None or self.sparse_weights is None:
            identifiers = self.keyword.search(session_id, query, turn, limit)
            return identifiers, [1.0 / rank for rank in range(1, len(identifiers) + 1)]
        result = self.field_sparse.search(query, limit, self.sparse_weights)
        return (
            [item.parent_asin for item in result.items],
            [float(item.raw_score or 0.0) for item in result.items],
        )

    def _dense_ids(self, query: str, limit: int = 200) -> list[str]:
        if self.dense is None:
            raise RuntimeError("dense retrieval is not configured")
        result = self.dense.search(query, limit)
        if isinstance(result, list):
            return [str(item) for item in result]
        items = getattr(result, "items", None)
        if items is None:
            raise TypeError("dense retriever returned an unsupported result")
        return [str(item.parent_asin) for item in items]

    def _rescue_ids(self, query: str) -> list[str]:
        if self.semantic_rescue is None:
            raise RuntimeError("semantic rescue retrieval is not configured")
        result = self.semantic_rescue.search(query, 200)
        if isinstance(result, list):
            return [str(item) for item in result]
        items = getattr(result, "items", None)
        if items is None:
            raise TypeError("semantic rescue retriever returned an unsupported result")
        return [str(item.parent_asin) for item in items]

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        state = self.sessions[session_id]
        query, question = self._query_and_question(state, user_message, turn)
        ranking_state: ConversationState | SessionState | V2StateView = state
        controller = self.v2_controllers.get(session_id)
        if controller is not None:
            state_view = controller.snapshot(query_text=query, turn=turn)
            self.last_v2_views[session_id] = state_view
            ranking_state = state_view
            query = state_view.query_text
        active_route: str = self.retrieval_route
        retrieval_limit = self.retrieval_k
        active_sparse_weight = self.sparse_weight
        active_dense_weight = self.dense_weight
        joint_features: dict[str, float] | None = None
        joint_values: dict[str, float] | None = None
        joint_reason: str | None = None
        route_decision: dict[str, object] | None = None
        if self.question_variant in {"joint_observable", "distilled_joint"}:
            if not isinstance(state, ConversationState):
                raise TypeError("joint policy requires conversation state")
            assert self.joint_policy is not None
            from ghostlab.policy.joint_actions import observable_joint_features

            previous = self.last_runtime_inputs.get(session_id)
            joint_features = observable_joint_features(
                state,
                turn=turn,
                previous_scores=None if previous is None else previous[1],
            )
            joint_decision = self.joint_policy.decide(state, joint_features)
            question = joint_decision.action.ask_attribute
            active_route = (
                "weighted"
                if joint_decision.action.retrieval_route == "weighted_fusion"
                else joint_decision.action.retrieval_route
            )
            retrieval_limit = joint_decision.action.retrieval_k
            active_sparse_weight = joint_decision.action.sparse_weight
            active_dense_weight = joint_decision.action.dense_weight
            joint_reason = joint_decision.reason
            joint_values = {
                "retrieval_k": float(retrieval_limit),
                "sparse_weight": active_sparse_weight,
                "dense_weight": active_dense_weight,
            }
            if question is not None:
                state.asked_attributes.append(question)
            state.last_asked_attribute = question
        if self.routing_variant == "calibrated":
            assert self.calibrated_router is not None
            if not isinstance(state, ConversationState):
                raise TypeError("calibrated routing requires conversation state")
            if joint_features is None:
                from ghostlab.policy.joint_actions import observable_joint_features

                previous = self.last_runtime_inputs.get(session_id)
                joint_features = observable_joint_features(
                    state,
                    turn=turn,
                    previous_scores=None if previous is None else previous[1],
                )
            selected_route = self.calibrated_router.decide(joint_features)
            active_route = (
                "weighted"
                if selected_route.route == "weighted_fusion"
                else selected_route.route
            )
            route_decision = {
                "route": selected_route.route,
                "predicted_advantage": selected_route.predicted_advantage,
                "confidence": selected_route.confidence,
                "reason": selected_route.reason,
            }
        original_query = query
        sparse_ids, sparse_scores = self._sparse_ids(
            session_id, query, turn, retrieval_limit
        )
        sparse_signal = retrieval_signals(sparse_scores)
        sparse_entropy = sparse_signal.normalized_entropy
        expansion_active = self.query_expander is not None and self._activate(
            self.query_expansion_activation,
            entropy=sparse_entropy,
            minimum_entropy=self.query_expansion_min_entropy,
        )
        expansion_trace: dict[str, object] | None = None
        if self.query_expander is not None and expansion_active:
            expansion = self.query_expander.expand(query, sparse_ids)
            expanded_query = expansion.expanded_query
            expansion_trace = {
                "reason": expansion.reason,
                "terms": [term.value for term in expansion.terms],
            }
            if expanded_query != query:
                query = expanded_query
                sparse_ids, sparse_scores = self._sparse_ids(
                    session_id, query, turn, retrieval_limit
                )
                sparse_signal = retrieval_signals(sparse_scores)
                sparse_entropy = sparse_signal.normalized_entropy
        elif self.query_expander is not None:
            expansion_trace = {
                "reason": "uncertainty_gate",
                "terms": [],
                "activated": False,
            }
        dense_route = active_route in {
            "dense",
            "rrf",
            "weighted",
            "sparse_first_union",
        }
        dense_active = dense_route and self._activate(
            self.dense_activation,
            entropy=sparse_entropy,
            minimum_entropy=self.dense_activation_min_entropy,
        )
        if dense_route and not dense_active:
            active_route = "keyword"
        retrieval_scores: list[float] = sparse_scores
        if active_route == "keyword":
            ranked = sparse_ids
        elif active_route == "dense":
            ranked = self._dense_ids(query, retrieval_limit)
            retrieval_scores = [1.0 / rank for rank in range(1, len(ranked) + 1)]
        elif active_route == "rrf":
            ranked = reciprocal_rank_fusion(
                [sparse_ids, self._dense_ids(query, retrieval_limit)],
                rank_constant=self.rrf_constant,
                limit=retrieval_limit,
            )
        elif active_route == "weighted":
            ranked = weighted_fuse_ids(
                sparse_ids,
                self._dense_ids(query, retrieval_limit),
                sparse_weight=active_sparse_weight,
                dense_weight=active_dense_weight,
                limit=retrieval_limit,
            )
        elif active_route == "sparse_first_union":
            ranked = sparse_first_union_ids(
                sparse_ids,
                self._dense_ids(query, retrieval_limit),
                limit=min(400, retrieval_limit * 2),
            )
        else:
            ranked = sparse_semantic_union_ids(
                sparse_ids,
                self._rescue_ids(query),
                sparse_weight=1.0 - self.semantic_rescue_weight,
                semantic_weight=self.semantic_rescue_weight,
                limit=retrieval_limit,
            )
        fallback_reason: str | None = None
        if self.component_fallback is not None and active_route != "keyword":
            fallback = self.component_fallback.choose(ranked, sparse_ids)
            ranked = list(fallback.ranking)
            fallback_reason = fallback.reason
        retrieved = list(ranked)
        if self.structured_filter is not None:
            ranked = self.structured_filter.apply(
                ranked,
                self._positive_constraints(ranking_state),
                minimum_results=max(10, top_k),
            )
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked)
        profile_prior_active = (
            self.profile_prior is not None and turn <= self.profile_prior_max_turn
        )
        if profile_prior_active:
            assert self.profile_prior is not None
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
            ranked = self.learned_reranker.rerank(
                query, ranked, rerank_k=self.learned_rerank_k
            )
        cross_encoder_active = (
            self.cross_encoder_reranker is not None
            and turn >= self.cross_encoder_min_turn
            and self._activate(
                self.cross_encoder_activation,
                entropy=sparse_entropy,
                minimum_entropy=self.cross_encoder_min_entropy,
            )
        )
        if self.cross_encoder_reranker is not None and cross_encoder_active:
            ranked = self.cross_encoder_reranker.rerank(
                query,
                ranked,
                rerank_k=self.cross_encoder_rerank_k,
                weight=self.cross_encoder_weight,
            )
        diversification_trace: dict[str, object] | None = None
        if self.diversifier is not None:
            diversification_decision = self.diversifier.rerank(
                ranked,
                DiversificationContext(
                    turn=turn,
                    active_constraint_count=sum(
                        len(values)
                        for values in self._positive_constraints(ranking_state).values()
                    ),
                ),
            )
            ranked = list(diversification_decision.ranking)
            diversification_trace = {
                "activated": diversification_decision.activated,
                "reason": diversification_decision.reason,
            }
        ranked = self._filter_recommendation_history(session_id, state, list(ranked))
        self.last_runtime_inputs[session_id] = (query, retrieval_scores)
        self.retrieval_trace.append(
            {
                "session_id": session_id,
                "turn": turn,
                "query": query,
                "original_query": original_query,
                "expansion": expansion_trace,
                "diversification": diversification_trace,
                "route": active_route,
                "route_decision": route_decision,
                "fallback_reason": fallback_reason,
                "activation": {
                    "sparse_entropy": sparse_entropy,
                    "dense": dense_active,
                    "query_expansion": expansion_active,
                    "profile_prior": profile_prior_active,
                    "cross_encoder": cross_encoder_active,
                },
                "retrieved": retrieved,
                "ranked": list(ranked),
            }
        )
        question_reason: str = joint_reason or self.question_variant
        if self.question_variant == "adaptive":
            question, question_reason = self._adaptive_question(
                state, user_message, turn, query
            )
        action_values: dict[QuestionAction, float] | dict[str, float] | None = (
            joint_values
        )
        feature_values: dict[str, float] | None = joint_features
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
                absorbing_values: dict[QuestionAction, float] = {None: 0.0}
                action_values = absorbing_values
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
        if self.question_variant in {"candidate_eig", "reward_voi"}:
            if not isinstance(state, ConversationState):
                raise TypeError("EIG question policy requires conversation state")
            assert self.candidate_facets is not None
            assert self.eig_policy is not None
            statistics = self.candidate_facets.summarize(
                ranked, limit=self.eig_candidate_k
            )
            eig_decision = self.eig_policy.decide(
                state, statistics, turn=turn, message=user_message
            )
            question = eig_decision.ask_attribute
            question_reason = eig_decision.reason
            action_values = dict(eig_decision.values)
            if question is not None:
                state.asked_attributes.append(question)
            state.last_asked_attribute = question
        if turn > self.question_max_turn:
            self._remove_recorded_question(state, question)
            question = None
            question_reason = "question_horizon"
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
        response = normalize_response(
            {
                "message": message,
                "ask_attribute": question,
                "recommendations": ranked,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            },
            self.catalog_ids,
            top_k,
        )
        self._record_recommendation_history(session_id, response)
        return response

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
