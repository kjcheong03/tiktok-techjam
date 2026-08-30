from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from baseline.state import classify_constraint
from ghostlab.competition.contract import AskAttribute
from ghostlab.retrieval.category import CategoryCandidateIndex, CategoryHit
from ghostlab.retrieval.diversify import DiversificationContext, FacetMMRDiversifier
from ghostlab.retrieval.multi_route import (
    CandidateEvidence,
    MergedCandidatePool,
    merge_candidate_routes,
)
from ghostlab.retrieval.profile import ProfilePriorReranker
from ghostlab.retrieval.pseudo_relevance import CatalogPseudoRelevanceFeedback
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.adaptive_components import (
    BoundedLocalLLMSemanticRanker,
    ConflictSafeContextAdapter,
    DiverseDenseTrack,
    DualTrackRouter,
    GuidanceDecision,
    OverGeneralityGuidance,
    ProfileContext,
    ProfileUpdate,
    RouteDecision,
    SemanticActivationPolicy,
    SemanticRankingResult,
    UnionAwareRanker,
    apply_profile_context,
)
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.baseline_v2 import (
    LegacyConstraintAdapter,
    StateBaselineV2,
    StructuredConstraint,
    normalize_value,
)
from ghostlab.state.v2_view import V2SessionController, V2StateView


@dataclass(frozen=True)
class AdaptiveActionRecord:
    turn: int
    route: str
    shown_products: tuple[str, ...]
    asked_attribute: str | None
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveTurnTrace:
    session_id: str
    turn: int
    policy_id: str
    config_sha256: str
    query_sha256: str
    route: str
    route_confidence: float
    route_reason: str
    overloaded: bool
    contribution_counts: dict[str, int]
    query_views: tuple[str, ...]
    union_candidate_count: int
    semantic_backend: str
    semantic_activation_reason: str
    semantic_changed: bool
    profile_active: bool
    profile_reason: str
    profile_update_values: tuple[str, ...]
    profile_update_confidence: float
    profile_update_provenance: str
    profile_update_conflicts: tuple[str, ...]
    ask_attribute: str | None
    fallback_reason: str | None
    top_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveCandidateSnapshot:
    """Runtime-observable merged pool used by fold-safe offline training."""

    session_id: str
    turn: int
    query: str
    route: str
    candidates: tuple[str, ...]
    overloaded: bool
    evidence: tuple[CandidateEvidence, ...] = ()


@dataclass
class _AdaptiveSession:
    state: StateBaselineV2
    controller: V2SessionController
    lock: Lock = field(default_factory=Lock)
    action_history: list[AdaptiveActionRecord] = field(default_factory=list)
    profile_update: ProfileUpdate | None = None


class AdaptiveHybridAgent:
    """Fixed 1A-3B workflow with GhostLab-optimizable component slots."""

    def __init__(
        self,
        catalog_path: str | Path,
        config: AdaptiveHybridConfig,
        *,
        project_root: str | Path,
        dense_track: DiverseDenseTrack | None = None,
        semantic_ranker: BoundedLocalLLMSemanticRanker | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.config_sha256 = config.canonical_hash()
        self.catalog_ids = self._read_catalog_ids(self.catalog_path)
        self.sparse = SparseIndex(self.catalog_path)
        self.category = CategoryCandidateIndex(self.catalog_path)
        self.router = DualTrackRouter(config.router)
        self.context_adapter = ConflictSafeContextAdapter(config.runtime_adaptation)
        self.dense = dense_track or DiverseDenseTrack(
            self.catalog_path,
            config.browsing,
            project_root=self.project_root,
        )
        self.union_ranker = UnionAwareRanker(
            self.catalog_path,
            config.union_ranker,
            project_root=self.project_root,
        )
        self.semantic = semantic_ranker or BoundedLocalLLMSemanticRanker(
            self.catalog_path,
            config.semantic_ranker,
            project_root=self.project_root,
        )
        self.semantic_activation = SemanticActivationPolicy(config.semantic_ranker)
        self.guidance = OverGeneralityGuidance(self.catalog_path, config.guidance)
        self.profile = ProfilePriorReranker(self.catalog_path)
        self.quality = (
            CatalogQualityReranker(self.catalog_path)
            if config.extensions.quality_prior_weight > 0.0
            else None
        )
        self.query_expander = (
            CatalogPseudoRelevanceFeedback(
                self.catalog_path,
                feedback_k=config.extensions.query_prf_feedback_k,
                minimum_support=config.extensions.query_prf_minimum_support,
                max_terms=config.extensions.query_prf_max_terms,
                max_added_ratio=config.extensions.query_prf_max_added_ratio,
            )
            if config.extensions.query_prf_enabled
            else None
        )
        self.diversifier = (
            FacetMMRDiversifier(
                self.catalog_path,
                relevance_weight=config.extensions.facet_relevance_weight,
                rerank_k=config.extensions.facet_rerank_k,
                output_k=config.extensions.facet_output_k,
                max_turn=config.extensions.facet_max_turn,
                max_active_constraints=config.extensions.facet_max_constraints,
            )
            if config.extensions.facet_diversity_enabled
            else None
        )
        self.sessions: dict[str, _AdaptiveSession] = {}
        self._sessions_lock = Lock()
        self.traces: list[AdaptiveTurnTrace] = []
        self.candidate_snapshots: list[AdaptiveCandidateSnapshot] = []
        self._trace_lock = Lock()

    @staticmethod
    def _read_catalog_ids(path: Path) -> set[str]:
        import json

        result: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    result.add(str(json.loads(line)["parent_asin"]))
        return result

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(user_profile, dict):
            raise TypeError("user_profile must be an object")
        state = StateBaselineV2(session_id, dict(user_profile))
        session = _AdaptiveSession(state=state, controller=V2SessionController(state))
        self.profile.reset(session_id, user_profile)
        with self._sessions_lock:
            self.sessions[session_id] = session

    def _session(self, session_id: str) -> _AdaptiveSession:
        with self._sessions_lock:
            if session_id not in self.sessions:
                raise KeyError(f"unknown adaptive session: {session_id}")
            return self.sessions[session_id]

    def _precision_candidates(self, query: str) -> tuple[list[str], list[float]]:
        result = self.sparse.search(
            query,
            self.config.buying.retrieval_k,
            self.config.buying.field_weights,
        )
        return (
            [item.parent_asin for item in result.items],
            [float(item.raw_score or 0.0) for item in result.items],
        )

    def _category_candidates(
        self, query: str, categories: list[str]
    ) -> tuple[CategoryHit, ...]:
        return self.category.search(
            query,
            limit=self.config.merger.category_k,
            preferred_categories=categories,
        )

    def _merge(
        self,
        decision: RouteDecision,
        *,
        keyword_ids: list[str],
        keyword_scores: dict[str, float],
        category_hits: tuple[CategoryHit, ...],
        vector_ids: list[str],
        vector_scores: dict[str, float],
    ) -> MergedCandidatePool:
        if decision.route == "buying":
            return merge_candidate_routes(
                route="buying",
                keyword_ids=keyword_ids,
                category_hits=category_hits,
                vector_ids=vector_ids[: self.config.merger.buying_vector_support_k],
                limit=self.config.merger.merged_k,
                keyword_weight=self.config.merger.buying_keyword_weight,
                category_weight=self.config.merger.buying_category_weight,
                vector_weight=self.config.merger.buying_vector_weight,
                keyword_scores=keyword_scores,
                vector_scores=vector_scores,
                strategy=self.config.merger.strategy,
                rrf_constant=self.config.merger.rrf_constant,
            )
        return merge_candidate_routes(
            route="browsing",
            keyword_ids=keyword_ids[: self.config.merger.browsing_keyword_support_k],
            category_hits=category_hits,
            vector_ids=vector_ids,
            limit=self.config.merger.merged_k,
            keyword_weight=self.config.merger.browsing_keyword_weight,
            category_weight=self.config.merger.browsing_category_weight,
            vector_weight=self.config.merger.browsing_vector_weight,
            keyword_scores=keyword_scores,
            vector_scores=vector_scores,
            strategy=self.config.merger.strategy,
            rrf_constant=self.config.merger.rrf_constant,
        )

    def _apply_optional_rankers(
        self, ranking: list[str], view: V2StateView
    ) -> tuple[list[str], tuple[str, ...]]:
        """Apply legal additions without changing the fixed workflow topology."""

        reasons: list[str] = []
        if self.quality is not None:
            ranking = self.quality.rerank(
                ranking,
                weight=self.config.extensions.quality_prior_weight,
                rerank_k=min(self.config.extensions.quality_rerank_k, len(ranking)),
            )
            reasons.append("optional:prior.quality")
        if self.diversifier is not None:
            decision = self.diversifier.rerank(
                ranking,
                DiversificationContext(
                    turn=view.turn,
                    active_constraint_count=len(view.active_constraints),
                ),
            )
            ranking = decision.ranking
            reasons.append(
                "optional:ranking.facet_diversity.v1:"
                f"{'active' if decision.activated else decision.reason}"
            )
        return ranking, tuple(reasons)

    def _safe_precision_ranking(
        self,
        query: str,
        keyword_ids: list[str],
        category_hits: tuple[CategoryHit, ...],
        positive_constraints: dict[str, list[str]],
        negative_constraints: dict[str, list[str]],
    ) -> list[str]:
        try:
            pool = merge_candidate_routes(
                route="buying",
                keyword_ids=keyword_ids,
                category_hits=category_hits,
                vector_ids=(),
                limit=self.config.merger.merged_k,
                keyword_weight=self.config.merger.buying_keyword_weight,
                category_weight=self.config.merger.buying_category_weight,
                vector_weight=0.0,
            )
            return self.union_ranker.rank(
                query,
                pool,
                positive_constraints=positive_constraints,
                negative_constraints=negative_constraints,
            )
        except Exception:  # noqa: BLE001 - component boundary must fail closed
            return self.union_ranker.filter.apply_strict(
                list(keyword_ids),
                positive_constraints,
                negative_constraints,
            )

    def profile_update(self, session_id: str) -> ProfileUpdate | None:
        """Return the conflict-safe, caller-persistable 3A update for a session."""

        return self._session(session_id).profile_update

    @staticmethod
    def _question_message(question: AskAttribute | None) -> str:
        if question is None:
            return "Here are the closest matches based on what you have shared."
        return f"Do you have a preference for {question.replace('_', ' ')}?"

    def _semantic_rank(
        self,
        query: str,
        ranking: list[str],
        route: RouteDecision,
        view: V2StateView,
        *,
        overloaded: bool,
    ) -> SemanticRankingResult:
        activation = self.semantic_activation.decide(route, view, overloaded=overloaded)
        if activation.active:
            ranked = self.semantic.rank(query, ranking)
            return SemanticRankingResult(
                ranking=ranked.ranking,
                changed=ranked.changed,
                elapsed_ms=ranked.elapsed_ms,
                backend=ranked.backend,
                activation_reason=activation.reason,
            )
        return SemanticRankingResult(
            ranking=tuple(ranking),
            changed=False,
            elapsed_ms=0.0,
            backend=f"skipped:{activation.reason}",
            activation_reason=activation.reason,
        )

    @staticmethod
    def _observe_state(state: StateBaselineV2, message: str, turn: int) -> None:
        parsed = LegacyConstraintAdapter().parse_result(
            message, turn, last_asked_attribute=state.last_asked_attribute
        )
        exclusions = [
            StructuredConstraint(
                attribute=classify_constraint(value),  # type: ignore[arg-type]
                values=[value],
                polarity="exclude",
                strength="hard",
                source_turn=turn,
                source_text=message,
                provenance="explicit",
            )
            for value in re.findall(
                r"\b(?:not|avoid|without|exclude)\s+([^.;,]+)",
                message,
                flags=re.IGNORECASE,
            )
            if normalize_value(value)
        ]
        state.observe(
            message,
            turn,
            parsed_constraints=(*parsed.constraints, *exclusions),
            no_preference_attributes=parsed.no_preference_attributes,
        )

    def _commit(
        self,
        session: _AdaptiveSession,
        response: dict[str, Any],
        *,
        turn: int,
        route: str,
        reason_codes: tuple[str, ...],
    ) -> tuple[str, ...]:
        identifiers = tuple(
            str(item["parent_asin"]) for item in response["recommendations"]
        )
        question = response.get("ask_attribute")
        if question is not None and question not in session.state.asked_attributes:
            session.state.asked_attributes.append(str(question))
        session.state.last_asked_attribute = (
            str(question) if question is not None else None
        )
        session.controller.record_shown(list(identifiers))
        session.action_history.append(
            AdaptiveActionRecord(
                turn=turn,
                route=route,
                shown_products=identifiers,
                asked_attribute=(str(question) if question is not None else None),
                reason_codes=reason_codes,
            )
        )
        session.profile_update = self.context_adapter.update(session.state)
        return identifiers

    def _record_trace(self, trace: AdaptiveTurnTrace) -> None:
        serialized = repr(trace).casefold()
        forbidden = (
            "ground_truth",
            "intent_card",
            "scenario_type",
            "target_id",
            "future_answer",
            "hidden_reward",
        )
        if any(item in serialized for item in forbidden):
            raise ValueError("research-only label detected in adaptive trace")
        with self._trace_lock:
            self.traces.append(trace)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        if not 1 <= turn <= 10:
            raise ValueError("turn must be between one and ten")
        if not 1 <= top_k <= self.config.orchestration.max_recommendations:
            raise ValueError("top_k exceeds the adaptive response contract")

        session = self._session(session_id)
        with session.lock:
            state = session.state
            self._observe_state(state, user_message, turn)
            query = state.build_coverage_adaptive_query() or user_message
            view = session.controller.snapshot(query_text=query, turn=turn)
            try:
                profile_context: ProfileContext = self.context_adapter.distil(
                    state, session.profile_update
                )
            except Exception as error:  # noqa: BLE001 - context must fail closed
                profile_context = ProfileContext(
                    frozenset(),
                    0.0,
                    "runtime_profile",
                    False,
                    f"context_failure:{type(error).__name__}",
                )
            fallback_reason: str | None = None
            try:
                route = self.router.decide(view, user_message)
            except Exception as error:  # noqa: BLE001 - precision abstention
                route = RouteDecision(
                    "buying",
                    1.0,
                    f"router_failure:{type(error).__name__}",
                    True,
                )
                fallback_reason = f"router:{type(error).__name__}"
            reason_codes = [f"route:{route.route}", route.reason]
            query_views: tuple[str, ...] = ()
            semantic = SemanticRankingResult(
                ranking=(), changed=False, elapsed_ms=0.0, backend="not_run"
            )
            contribution_counts = {"keyword": 0, "category": 0, "vector": 0}
            union_count = 0
            candidate_snapshot: AdaptiveCandidateSnapshot | None = None

            try:
                keyword_ids, keyword_scores_raw = self._precision_candidates(query)
            except Exception as error:  # noqa: BLE001 - valid empty response boundary
                keyword_ids, keyword_scores_raw = [], []
                fallback_reason = fallback_reason or f"sparse:{type(error).__name__}"
            if self.query_expander is not None and fallback_reason is None:
                try:
                    preview = self.guidance.decide(
                        state,
                        keyword_ids,
                        turn=turn,
                        message=user_message,
                    )
                    if preview.overloaded:
                        reason_codes.append(
                            "optional:query.catalog_prf.v1:overload_skip"
                        )
                    else:
                        expansion = self.query_expander.expand(query, keyword_ids)
                        reason_codes.append(
                            f"optional:query.catalog_prf.v1:{expansion.reason}"
                        )
                        if expansion.expanded_query != query:
                            query = expansion.expanded_query
                            view = session.controller.snapshot(
                                query_text=query, turn=turn
                            )
                            keyword_ids, keyword_scores_raw = (
                                self._precision_candidates(query)
                            )
                except Exception as error:  # noqa: BLE001 - optional fail-open hook
                    reason_codes.append(
                        f"optional:query.catalog_prf.v1:failure:{type(error).__name__}"
                    )
            positive = view.positive_constraints()
            negative = view.negative_constraints()
            maximum_keyword_score = max(keyword_scores_raw, default=1.0)
            keyword_scores = {
                identifier: score / maximum_keyword_score
                for identifier, score in zip(
                    keyword_ids, keyword_scores_raw, strict=True
                )
            }
            categories = positive.get("category", [])
            try:
                category_hits = self._category_candidates(query, categories)
            except Exception as error:  # noqa: BLE001 - required safe fallback
                category_hits = ()
                fallback_reason = f"category:{type(error).__name__}"

            vector_ids: list[str] = []
            vector_scores: dict[str, float] = {}
            if route.route == "browsing" and fallback_reason is None:
                try:
                    dense = self.dense.search(view)
                    vector_ids = list(dense.identifiers)
                    vector_scores = dict(dense.relevance_scores)
                    query_views = dense.query_views
                except Exception as error:  # noqa: BLE001 - required safe fallback
                    fallback_reason = f"dense:{type(error).__name__}"

            if route.route == "buying" and fallback_reason is None:
                try:
                    dense = self.dense.search(view)
                    vector_ids = list(dense.identifiers)
                    vector_scores = dict(dense.relevance_scores)
                    query_views = dense.query_views
                except Exception as error:  # noqa: BLE001 - precision fallback
                    fallback_reason = f"dense_support:{type(error).__name__}"

            primary = keyword_ids if route.route == "buying" else vector_ids
            guidance_candidates = list(
                dict.fromkeys(
                    [
                        *primary,
                        *(item.parent_asin for item in category_hits),
                        *vector_ids,
                        *keyword_ids,
                    ]
                )
            )
            try:
                guidance = self.guidance.decide(
                    state,
                    guidance_candidates or keyword_ids,
                    turn=turn,
                    message=user_message,
                )
            except Exception as error:  # noqa: BLE001 - precision fallback
                guidance = GuidanceDecision(
                    overloaded=False,
                    ask_attribute=None,
                    reason=f"guidance_failure:{type(error).__name__}",
                    values={},
                )
                fallback_reason = fallback_reason or (
                    f"guidance:{type(error).__name__}"
                )

            if fallback_reason is not None:
                ranking = self._safe_precision_ranking(
                    query, keyword_ids, category_hits, positive, negative
                )
                reason_codes.extend(("fallback:complete_precision", fallback_reason))
            elif guidance.overloaded:
                try:
                    pool = self._merge(
                        route,
                        keyword_ids=keyword_ids,
                        keyword_scores=keyword_scores,
                        category_hits=category_hits,
                        vector_ids=vector_ids,
                        vector_scores=vector_scores,
                    )
                    contribution_counts = pool.contribution_counts()
                    union_count = len(pool.candidates)
                    candidate_snapshot = AdaptiveCandidateSnapshot(
                        session_id=session_id,
                        turn=turn,
                        query=query,
                        route=route.route,
                        candidates=tuple(pool.identifiers),
                        overloaded=True,
                        evidence=pool.candidates,
                    )
                    ranking = self.union_ranker.rank(
                        query,
                        pool,
                        positive_constraints=positive,
                        negative_constraints=negative,
                    )
                    ranking, optional_reasons = self._apply_optional_rankers(
                        ranking, view
                    )
                    semantic = self._semantic_rank(
                        query, ranking, route, view, overloaded=True
                    )
                    ranking = list(semantic.ranking)
                    active_sources = "_".join(
                        source
                        for source in ("keyword", "category", "vector")
                        if contribution_counts[source] > 0
                    )
                    reason_codes.extend(
                        (
                            "overload:cutoff",
                            guidance.reason,
                            f"merge:{active_sources}",
                            "rank:union_aware",
                            *optional_reasons,
                            f"semantic:{semantic.backend}",
                        )
                    )
                except Exception as error:  # noqa: BLE001 - required fallback
                    fallback_reason = f"adaptive:{type(error).__name__}"
                    ranking = self._safe_precision_ranking(
                        query, keyword_ids, category_hits, positive, negative
                    )
                    reason_codes.extend(
                        ("fallback:complete_precision", fallback_reason)
                    )
            else:
                try:
                    pool = self._merge(
                        route,
                        keyword_ids=keyword_ids,
                        keyword_scores=keyword_scores,
                        category_hits=category_hits,
                        vector_ids=vector_ids,
                        vector_scores=vector_scores,
                    )
                    contribution_counts = pool.contribution_counts()
                    union_count = len(pool.candidates)
                    candidate_snapshot = AdaptiveCandidateSnapshot(
                        session_id=session_id,
                        turn=turn,
                        query=query,
                        route=route.route,
                        candidates=tuple(pool.identifiers),
                        overloaded=False,
                        evidence=pool.candidates,
                    )
                    ranking = self.union_ranker.rank(
                        query,
                        pool,
                        positive_constraints=positive,
                        negative_constraints=negative,
                    )
                    ranking, optional_reasons = self._apply_optional_rankers(
                        ranking, view
                    )
                    semantic = self._semantic_rank(
                        query, ranking, route, view, overloaded=False
                    )
                    ranking = list(semantic.ranking)
                    active_sources = "_".join(
                        source
                        for source in ("keyword", "category", "vector")
                        if contribution_counts[source] > 0
                    )
                    reason_codes.extend(
                        (
                            f"merge:{active_sources}",
                            "rank:union_aware",
                            *optional_reasons,
                            f"semantic:{semantic.backend}",
                        )
                    )
                except Exception as error:  # noqa: BLE001 - required safe fallback
                    fallback_reason = f"adaptive:{type(error).__name__}"
                    ranking = self._safe_precision_ranking(
                        query, keyword_ids, category_hits, positive, negative
                    )
                    reason_codes.extend(
                        ("fallback:complete_precision", fallback_reason)
                    )

            ranking = apply_profile_context(
                self.profile,
                session_id,
                ranking,
                profile_context,
                self.config.runtime_adaptation,
            )
            unfiltered = list(ranking)
            ranking = session.controller.filter_ranking(unfiltered)
            response = normalize_response(
                {
                    "message": self._question_message(guidance.ask_attribute),
                    "ask_attribute": guidance.ask_attribute,
                    "recommendations": ranking,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                self.catalog_ids,
                top_k,
            )
            identifiers = self._commit(
                session,
                response,
                turn=turn,
                route=route.route,
                reason_codes=tuple(reason_codes),
            )
            profile_update = session.profile_update or ProfileUpdate(
                values=(),
                confidence=0.0,
                provenance="no_session_evidence",
                intent_epoch=state.intent_epoch,
            )
            self._record_trace(
                AdaptiveTurnTrace(
                    session_id=session_id,
                    turn=turn,
                    policy_id=self.config.policy_id,
                    config_sha256=self.config_sha256,
                    query_sha256=hashlib.sha256(query.encode()).hexdigest(),
                    route=route.route,
                    route_confidence=route.confidence,
                    route_reason=route.reason,
                    overloaded=guidance.overloaded,
                    contribution_counts=contribution_counts,
                    query_views=query_views,
                    union_candidate_count=union_count,
                    semantic_backend=semantic.backend,
                    semantic_activation_reason=semantic.activation_reason,
                    semantic_changed=semantic.changed,
                    profile_active=profile_context.active,
                    profile_reason=profile_context.reason,
                    profile_update_values=profile_update.values,
                    profile_update_confidence=profile_update.confidence,
                    profile_update_provenance=profile_update.provenance,
                    profile_update_conflicts=profile_update.conflicts,
                    ask_attribute=response.get("ask_attribute"),
                    fallback_reason=fallback_reason,
                    top_ids=identifiers,
                    reason_codes=tuple(reason_codes),
                )
            )
            if candidate_snapshot is not None:
                self.candidate_snapshots.append(candidate_snapshot)
            return response


__all__ = [
    "AdaptiveActionRecord",
    "AdaptiveCandidateSnapshot",
    "AdaptiveHybridAgent",
    "AdaptiveTurnTrace",
]
