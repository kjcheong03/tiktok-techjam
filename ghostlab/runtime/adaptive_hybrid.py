from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any

from baseline.state import classify_constraint
from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.signals import RetrievalSignals, retrieval_signals
from ghostlab.retrieval.adaptive_residual import (
    AdaptiveResidualPolicy,
    AdaptiveTop10ResidualReranker,
    parent_config_sha256,
    sha256_file,
)
from ghostlab.retrieval.category import CategoryCandidateIndex, CategoryHit
from ghostlab.retrieval.dense import MINILM_CONTROL, DenseIndex
from ghostlab.retrieval.diversify import DiversificationContext, FacetMMRDiversifier
from ghostlab.retrieval.multi_route import (
    CandidateEvidence,
    MergedCandidatePool,
    merge_candidate_routes,
)
from ghostlab.retrieval.profile import ProfilePriorReranker
from ghostlab.retrieval.pseudo_relevance import CatalogPseudoRelevanceFeedback
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex, query_terms
from ghostlab.runtime.adaptive_components import (
    BoundedLocalLLMSemanticRanker,
    BrowsingSafeRanker,
    ConflictSafeContextAdapter,
    DiverseDenseTrack,
    DualTrackRouter,
    GuidanceDecision,
    OverGeneralityGuidance,
    ProfileContext,
    ProfileUpdate,
    RetrievalPreviewDecision,
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
from ghostlab.state.catalog_ontology import CatalogOntology
from ghostlab.state.normalization import CatalogStateNormalizer
from ghostlab.state.v2_view import AdaptiveTurnContext, V2SessionController, V2StateView

_DIRECT_EXCLUSION_RE = re.compile(
    r"\b(?:avoid|without|exclude)\s+([^.;,]+)", re.IGNORECASE
)
_NOT_EXCLUSION_RE = re.compile(
    r"\bnot\s+(?!(?:including|quite|sure|necessarily|only|just|yet)\b)([^.;,]+)",
    re.IGNORECASE,
)


def _explicit_exclusion_values(message: str) -> tuple[str, ...]:
    """Extract user exclusions while ignoring descriptive/discourse negation."""

    values = [
        *(_DIRECT_EXCLUSION_RE.findall(message)),
        *(_NOT_EXCLUSION_RE.findall(message)),
    ]
    return tuple(dict.fromkeys(value for value in values if normalize_value(value)))


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
    preview_reason: str
    preview_candidate_count: int
    preview_score_flatness: float
    dense_requested_per_view: int
    dense_output_k: int
    dense_selection: str
    constraint_counts: dict[str, int]
    output_constraint_violations: int
    contribution_counts: dict[str, int]
    query_views: tuple[str, ...]
    union_candidate_count: int
    semantic_backend: str
    semantic_activation_reason: str
    semantic_changed: bool
    semantic_elapsed_ms: float
    semantic_failure_reason: str | None
    semantic_candidate_margin: float | None
    semantic_candidate_entropy: float | None
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
    preview_executed: bool
    safe_merge_executed: bool
    safe_ranker_executed: bool
    normal_union_executed: bool
    semantic_decision_reached: bool
    semantic_executed: bool


@dataclass(frozen=True)
class AdaptiveCandidateSnapshot:
    """Runtime-observable merged pool used by fold-safe offline training."""

    session_id: str
    turn: int
    query: str
    route: str
    candidates: tuple[str, ...]
    overloaded: bool
    pre_semantic_candidates: tuple[str, ...] = ()
    post_semantic_candidates: tuple[str, ...] = ()
    pre_authority_candidates: tuple[str, ...] = ()
    authority_removed_ids: tuple[str, ...] = ()
    evidence: tuple[CandidateEvidence, ...] = ()
    confirmed_match_count: dict[str, int] = field(default_factory=dict)
    unknown_constraint_count: dict[str, int] = field(default_factory=dict)
    soft_preference_count: dict[str, int] = field(default_factory=dict)
    profile_terms: frozenset[str] = frozenset()
    pre_residual_top10: tuple[str, ...] = ()
    post_residual_top10: tuple[str, ...] = ()


@dataclass
class _AdaptiveSession:
    state: StateBaselineV2
    controller: V2SessionController
    lock: Lock = field(default_factory=Lock)
    action_history: list[AdaptiveActionRecord] = field(default_factory=list)
    profile_update: ProfileUpdate | None = None
    exploratory_intent_epoch: int | None = None


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
        minilm_dense_index: DenseIndex | None = None,
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
        self.browsing_safe = BrowsingSafeRanker(
            self.catalog_path,
            config.browsing,
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
        self.catalog_normalizer: CatalogStateNormalizer | None = None
        if config.state.catalog_normalizer_enabled:
            ontology_path = self.project_root / str(config.state.catalog_ontology_path)
            expected_ontology_hash = str(config.state.catalog_ontology_sha256)
            if sha256_file(ontology_path) != expected_ontology_hash:
                raise ValueError("catalog ontology asset hash mismatch")
            ontology = CatalogOntology.from_path(ontology_path)
            if ontology.catalog_sha256 != sha256_file(self.catalog_path):
                raise ValueError("catalog ontology was built for another catalog")
            self.catalog_normalizer = CatalogStateNormalizer(
                ontology,
                confidence_threshold=(config.state.constraint_normalization_confidence),
            )
        self.minilm_dense: DenseIndex | None = None
        if config.extensions.minilm_dense_view_enabled:
            if config.extensions.minilm_dense_model_revision != MINILM_CONTROL.revision:
                raise ValueError("auxiliary MiniLM revision is not the pinned control")
            self.minilm_dense = minilm_dense_index or DenseIndex(
                self.catalog_path,
                MINILM_CONTROL,
                cache_dir=self.project_root / config.extensions.minilm_dense_cache_dir,
                model_path=self.project_root
                / config.extensions.minilm_dense_model_path,
                local_files_only=True,
            )
        residual_config = config.extensions
        self.top10_residual: AdaptiveTop10ResidualReranker | None = None
        if residual_config.top10_residual_enabled:
            required = (
                residual_config.top10_residual_model_path,
                residual_config.top10_residual_model_sha256,
                residual_config.top10_residual_fit_receipt_path,
                residual_config.top10_residual_fit_receipt_sha256,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "enabled Top-10 residual requires a freshly fitted asset and receipt"
                )
            self.top10_residual = AdaptiveTop10ResidualReranker.from_verified_asset(
                self.catalog_path,
                self.project_root / str(required[0]),
                self.project_root / str(required[2]),
                expected_asset_sha256=str(required[1]),
                expected_receipt_sha256=str(required[3]),
                expected_parent_config_sha256=parent_config_sha256(config),
                policy=AdaptiveResidualPolicy(
                    rerank_depth=residual_config.top10_residual_rerank_depth,
                    model_weight=residual_config.top10_residual_model_weight,
                    minimum_expected_gain=(
                        residual_config.top10_residual_minimum_expected_gain
                    ),
                    minimum_probability_margin=(
                        residual_config.top10_residual_minimum_probability_margin
                    ),
                    maximum_moved_ids=(
                        residual_config.top10_residual_maximum_moved_ids
                    ),
                ),
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

    def _dense_candidates(
        self, context: AdaptiveTurnContext, *, overloaded: bool
    ) -> Any:
        """Call the budget-aware dense interface with legacy test-double support."""

        try:
            return self.dense.search(context, overloaded=overloaded)
        except TypeError as error:
            if "overloaded" not in str(error):
                raise
            return self.dense.search(context)

    @staticmethod
    def _blend_minilm_dense_view(
        identifiers: list[str],
        scores: dict[str, float],
        auxiliary: Any,
        *,
        weight: float,
    ) -> tuple[list[str], dict[str, float], int]:
        """Blend scores over E5 membership; MiniLM can never add/remove an ID."""

        original = tuple(identifiers)
        auxiliary_scores = {
            item.parent_asin: float(item.normalized_score or 0.0)
            for item in auxiliary.items
        }
        blended = {
            identifier: (
                (1.0 - weight) * float(scores.get(identifier, 0.0))
                + weight
                * auxiliary_scores.get(identifier, float(scores.get(identifier, 0.0)))
            )
            for identifier in original
        }
        positions = {identifier: index for index, identifier in enumerate(original)}
        ordered = sorted(
            original,
            key=lambda identifier: (
                -blended[identifier],
                positions[identifier],
                identifier,
            ),
        )
        if len(ordered) != len(original) or set(ordered) != set(original):
            raise RuntimeError("auxiliary MiniLM changed compulsory E5 membership")
        overlap = sum(identifier in auxiliary_scores for identifier in original)
        return ordered, blended, overlap

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
        context: AdaptiveTurnContext | None = None,
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
                context=context,
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

    def _turn_context(
        self,
        session: _AdaptiveSession,
        *,
        query: str,
        message: str,
        turn: int,
    ) -> AdaptiveTurnContext:
        raw_tags = session.state.user_profile.get("preference_tags")
        tags = raw_tags if isinstance(raw_tags, (list, tuple, set)) else ()
        raw_preferences = session.state.user_profile.get("preferences")
        preferences = raw_preferences if isinstance(raw_preferences, dict) else {}
        supplied_terms = frozenset(
            query_terms(
                " ".join(
                    [
                        *(str(item) for item in tags),
                        *(str(item) for item in preferences.values()),
                    ]
                ),
                40,
            )
        )
        overlay = session.profile_update
        return session.controller.snapshot(
            query_text=query,
            turn=turn,
            current_message=message,
            supplied_profile_terms=supplied_terms,
            profile_overlay_values=(overlay.values if overlay is not None else ()),
            profile_overlay_attributes=(
                frozenset(
                    [
                        *(overlay.attributes if overlay is not None else ()),
                        *(str(item) for item in preferences),
                    ]
                )
            ),
            profile_overlay_confidence=(
                overlay.confidence if overlay is not None else 0.0
            ),
            profile_overlay_epoch=(
                overlay.intent_epoch if overlay is not None else None
            ),
            exploratory_intent=(
                session.exploratory_intent_epoch == session.state.intent_epoch
            ),
        )

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
        pool: MergedCandidatePool | None = None,
    ) -> SemanticRankingResult:
        signals = self._semantic_candidate_signals(ranking, pool)
        activation = self.semantic_activation.decide(
            route,
            view,
            overloaded=overloaded,
            signals=signals,
        )
        if activation.active:
            ranked = self.semantic.rank(query, ranking)
            return replace(
                ranked,
                activation_reason=activation.reason,
                candidate_margin=(signals.top1_margin if signals else None),
                candidate_entropy=(signals.normalized_entropy if signals else None),
            )
        return SemanticRankingResult(
            ranking=tuple(ranking),
            changed=False,
            elapsed_ms=0.0,
            backend=f"skipped:{activation.reason}",
            activation_reason=activation.reason,
            candidate_margin=(signals.top1_margin if signals else None),
            candidate_entropy=(signals.normalized_entropy if signals else None),
        )

    def _semantic_candidate_signals(
        self,
        ranking: list[str],
        pool: MergedCandidatePool | None,
    ) -> RetrievalSignals | None:
        """Expose label-free merge ambiguity at the semantic decision point."""

        if pool is None:
            return None
        by_id = {item.parent_asin: item for item in pool.candidates}
        head = [
            by_id[identifier]
            for identifier in ranking[: self.config.semantic_ranker.rerank_k]
            if identifier in by_id
        ]
        if not head:
            return None
        scores = sorted((float(item.aggregate_score) for item in head), reverse=True)
        sparse_ids = [
            item.parent_asin
            for item in sorted(
                (item for item in pool.candidates if item.keyword_rank is not None),
                key=lambda item: (item.keyword_rank or 0, item.parent_asin),
            )
        ]
        dense_ids = [
            item.parent_asin
            for item in sorted(
                (item for item in pool.candidates if item.vector_rank is not None),
                key=lambda item: (item.vector_rank or 0, item.parent_asin),
            )
        ]
        return retrieval_signals(
            scores,
            sparse_ids=sparse_ids,
            dense_ids=dense_ids,
        )

    @staticmethod
    def _observe_state(
        state: StateBaselineV2,
        message: str,
        turn: int,
        catalog_normalizer: CatalogStateNormalizer | None = None,
    ) -> str | None:
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
            for value in _explicit_exclusion_values(message)
        ]
        incoming = [*parsed.constraints, *exclusions]
        normalized_count = 0
        if catalog_normalizer is not None:
            normalized: list[StructuredConstraint] = []
            try:
                for constraint in incoming:
                    resolutions = [
                        catalog_normalizer.normalize(
                            constraint.attribute, value, state.active_category
                        )
                        for value in constraint.values
                    ]
                    resolved_attributes = {
                        item.attribute for item in resolutions if item is not None
                    }
                    all_resolved = all(item is not None for item in resolutions)
                    target_attribute = (
                        next(iter(resolved_attributes))
                        if all_resolved and len(resolved_attributes) == 1
                        else constraint.attribute
                    )
                    values = [
                        (
                            resolution.canonical
                            if resolution is not None
                            and resolution.attribute == target_attribute
                            else value
                        )
                        for value, resolution in zip(
                            constraint.values, resolutions, strict=True
                        )
                    ]
                    normalized_count += target_attribute != constraint.attribute
                    normalized_count += sum(
                        left != right
                        for left, right in zip(constraint.values, values, strict=True)
                    )
                    normalized.append(
                        replace(
                            constraint,
                            attribute=target_attribute,  # type: ignore[arg-type]
                            values=values,
                        )
                    )
                incoming = normalized
            except Exception as error:  # noqa: BLE001 - optional state hook fails open
                incoming = [*parsed.constraints, *exclusions]
                normalization_reason = (
                    "optional:state.catalog_normalizer.v1:failure:"
                    f"{type(error).__name__}"
                )
            else:
                normalization_reason = (
                    "optional:state.catalog_normalizer.v1:"
                    f"normalized_{normalized_count}"
                )
        else:
            normalization_reason = None
        state.observe(
            message,
            turn,
            parsed_constraints=incoming,
            no_preference_attributes=parsed.no_preference_attributes,
        )
        return normalization_reason

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
            normalization_reason = self._observe_state(
                state, user_message, turn, self.catalog_normalizer
            )
            if self.router.browsing_marker(user_message) is not None:
                session.exploratory_intent_epoch = state.intent_epoch
            elif session.exploratory_intent_epoch != state.intent_epoch:
                session.exploratory_intent_epoch = None
            query = state.build_coverage_adaptive_query() or user_message
            view = self._turn_context(
                session, query=query, message=user_message, turn=turn
            )
            try:
                profile_context: ProfileContext = self.context_adapter.distil(view)
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
            if normalization_reason is not None:
                reason_codes.append(normalization_reason)
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
                    prf_preview = self.guidance.preview(
                        view,
                        keyword_ids,
                        keyword_scores_raw,
                        (),
                    )
                    if prf_preview.overloaded:
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
                            view = replace(view, query_text=query)
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

            try:
                preview = self.guidance.preview(
                    view,
                    keyword_ids,
                    keyword_scores_raw,
                    tuple(item.parent_asin for item in category_hits),
                )
            except Exception as error:  # noqa: BLE001 - precision fallback
                preview = RetrievalPreviewDecision(
                    overloaded=False,
                    reason=f"preview_failure:{type(error).__name__}",
                    candidate_count=0,
                    specific_constraint_count=0,
                    score_flatness=0.0,
                )
                fallback_reason = fallback_reason or f"preview:{type(error).__name__}"

            vector_ids: list[str] = []
            vector_scores: dict[str, float] = {}
            dense_requested_per_view = 0
            dense_output_k = 0
            dense_selection = self.config.browsing.selection
            if route.route == "browsing" and fallback_reason is None:
                try:
                    dense = self._dense_candidates(view, overloaded=preview.overloaded)
                    vector_ids = list(dense.identifiers)
                    vector_scores = dict(dense.relevance_scores)
                    query_views = dense.query_views
                    dense_requested_per_view = dense.requested_per_view
                    dense_output_k = dense.output_k
                    dense_selection = dense.selection
                    if self.minilm_dense is not None and not preview.overloaded:
                        try:
                            auxiliary = self.minilm_dense.search(
                                query,
                                self.config.extensions.minilm_dense_retrieval_k,
                            )
                            vector_ids, vector_scores, overlap = (
                                self._blend_minilm_dense_view(
                                    vector_ids,
                                    vector_scores,
                                    auxiliary,
                                    weight=(self.config.extensions.minilm_dense_weight),
                                )
                            )
                            reason_codes.append(
                                "optional:retrieval.minilm_dense_view.v1:"
                                f"scored_{overlap}"
                            )
                        except Exception as error:  # noqa: BLE001 - additive fail-open
                            reason_codes.append(
                                "optional:retrieval.minilm_dense_view.v1:failure:"
                                f"{type(error).__name__}"
                            )
                except Exception as error:  # noqa: BLE001 - required safe fallback
                    fallback_reason = f"dense:{type(error).__name__}"

            if route.route == "buying" and fallback_reason is None:
                try:
                    dense = self._dense_candidates(view, overloaded=preview.overloaded)
                    vector_ids = list(dense.identifiers)
                    vector_scores = dict(dense.relevance_scores)
                    query_views = dense.query_views
                    dense_requested_per_view = dense.requested_per_view
                    dense_output_k = dense.output_k
                    dense_selection = dense.selection
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
                    view,
                    guidance_candidates or keyword_ids,
                    turn=turn,
                    message=user_message,
                    overloaded=preview.overloaded,
                    profile_known_attributes=(
                        view.profile_overlay_attributes
                        if self.config.runtime_adaptation.profile_question_suppression_enabled
                        else frozenset()
                    ),
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

            constraint_counts = {
                "confirmed_matches": 0,
                "confirmed_violations": 0,
                "unknown_metadata": 0,
                "soft_preferences": 0,
            }
            safe_merge_executed = False
            safe_ranker_executed = False
            normal_union_executed = False
            semantic_decision_reached = False
            semantic_executed = False
            if fallback_reason is not None:
                ranking = self._safe_precision_ranking(
                    query, keyword_ids, category_hits, positive, negative, view
                )
                reason_codes.extend(("fallback:complete_precision", fallback_reason))
            elif guidance.overloaded:
                try:
                    pool = merge_candidate_routes(
                        route=route.route,
                        keyword_ids=keyword_ids[
                            : self.config.guidance.preview_keyword_k
                        ],
                        category_hits=category_hits[
                            : self.config.guidance.preview_category_k
                        ],
                        vector_ids=vector_ids,
                        limit=self.config.browsing.overload_output_k,
                        keyword_weight=(
                            self.config.merger.buying_keyword_weight
                            if route.route == "buying"
                            else self.config.merger.browsing_keyword_weight
                        ),
                        category_weight=(
                            self.config.merger.buying_category_weight
                            if route.route == "buying"
                            else self.config.merger.browsing_category_weight
                        ),
                        vector_weight=(
                            self.config.merger.buying_vector_weight
                            if route.route == "buying"
                            else self.config.merger.browsing_vector_weight
                        ),
                        keyword_scores=keyword_scores,
                        vector_scores=vector_scores,
                        strategy=self.config.merger.strategy,
                        rrf_constant=self.config.merger.rrf_constant,
                    )
                    safe_merge_executed = True
                    pre_authority = tuple(pool.identifiers)
                    authority = self.union_ranker.filter.enforce(pool.identifiers, view)
                    constraint_counts = authority.counts()
                    pool = pool.retain(authority.ranking)
                    contribution_counts = pool.contribution_counts()
                    union_count = 0
                    candidate_snapshot = AdaptiveCandidateSnapshot(
                        session_id=session_id,
                        turn=turn,
                        query=query,
                        route=route.route,
                        candidates=tuple(pool.identifiers),
                        pre_authority_candidates=pre_authority,
                        authority_removed_ids=tuple(
                            identifier
                            for identifier in pre_authority
                            if identifier not in set(authority.ranking)
                        ),
                        overloaded=True,
                        evidence=pool.candidates,
                        confirmed_match_count=authority.confirmed_match_count,
                        unknown_constraint_count=authority.unknown_count,
                        soft_preference_count=authority.soft_preference_count,
                        profile_terms=profile_context.terms,
                    )
                    ranking = self.browsing_safe.rank(query, list(authority.ranking))
                    safe_ranker_executed = True
                    semantic_decision_reached = True
                    semantic = SemanticRankingResult(
                        ranking=tuple(ranking),
                        changed=False,
                        elapsed_ms=0.0,
                        backend="skipped:overload_cutoff",
                        activation_reason="overload_cutoff",
                    )
                    active_sources = "_".join(
                        source
                        for source in ("keyword", "category", "vector")
                        if contribution_counts[source] > 0
                    )
                    reason_codes.extend(
                        (
                            "overload:cutoff",
                            guidance.reason,
                            f"safe_merge:{active_sources}",
                            "rank:browsing_safe",
                            "union:skipped_overload_cutoff",
                            "semantic:skipped_overload_cutoff",
                        )
                    )
                except Exception as error:  # noqa: BLE001 - required fallback
                    fallback_reason = f"adaptive:{type(error).__name__}"
                    ranking = self._safe_precision_ranking(
                        query, keyword_ids, category_hits, positive, negative, view
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
                    pre_authority = tuple(pool.identifiers)
                    authority = self.union_ranker.filter.enforce(pool.identifiers, view)
                    constraint_counts = authority.counts()
                    pool = pool.retain(authority.ranking)
                    contribution_counts = pool.contribution_counts()
                    union_count = len(pool.candidates)
                    candidate_snapshot = AdaptiveCandidateSnapshot(
                        session_id=session_id,
                        turn=turn,
                        query=query,
                        route=route.route,
                        candidates=tuple(pool.identifiers),
                        pre_authority_candidates=pre_authority,
                        authority_removed_ids=tuple(
                            identifier
                            for identifier in pre_authority
                            if identifier not in set(authority.ranking)
                        ),
                        overloaded=False,
                        evidence=pool.candidates,
                        confirmed_match_count=authority.confirmed_match_count,
                        unknown_constraint_count=authority.unknown_count,
                        soft_preference_count=authority.soft_preference_count,
                        profile_terms=profile_context.terms,
                    )
                    ranking = self.union_ranker.rank(
                        query,
                        pool,
                        positive_constraints=positive,
                        negative_constraints=negative,
                        context=view,
                        authority=authority,
                        profile_terms=(
                            profile_context.terms
                            if self.config.runtime_adaptation.union_profile_feature_enabled
                            else frozenset()
                        ),
                    )
                    normal_union_executed = True
                    ranking, optional_reasons = self._apply_optional_rankers(
                        ranking, view
                    )
                    candidate_snapshot = replace(
                        candidate_snapshot,
                        pre_semantic_candidates=tuple(ranking),
                    )
                    semantic = self._semantic_rank(
                        query,
                        ranking,
                        route,
                        view,
                        overloaded=False,
                        pool=pool,
                    )
                    candidate_snapshot = replace(
                        candidate_snapshot,
                        post_semantic_candidates=semantic.ranking,
                    )
                    semantic_decision_reached = True
                    semantic_executed = not semantic.backend.startswith("skipped:")
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
                            *(
                                (
                                    f"optional:{self.config.union_ranker.auxiliary_technique_id}:bounded_union_auxiliary",
                                )
                                if self.config.union_ranker.auxiliary_technique_id
                                else ()
                            ),
                            *optional_reasons,
                            f"semantic:{semantic.backend}",
                        )
                    )
                except Exception as error:  # noqa: BLE001 - required safe fallback
                    fallback_reason = f"adaptive:{type(error).__name__}"
                    ranking = self._safe_precision_ranking(
                        query, keyword_ids, category_hits, positive, negative, view
                    )
                    reason_codes.extend(
                        ("fallback:complete_precision", fallback_reason)
                    )

            if not guidance.overloaded:
                ranking = apply_profile_context(
                    self.profile,
                    session_id,
                    ranking,
                    profile_context,
                    self.config.runtime_adaptation,
                )
            final_authority = self.union_ranker.filter.enforce(list(ranking), view)
            ranking = session.controller.filter_ranking(list(final_authority.ranking))
            reason_codes.append(
                "constraints:route_independent:"
                f"removed_{final_authority.violation_count}"
            )
            before_residual = tuple(ranking[:10])
            after_residual = before_residual
            if self.top10_residual is not None:
                residual = self.top10_residual.rerank(
                    query,
                    before_residual,
                    turn=turn,
                    route=route.route,
                    candidate_pool_size=len(ranking),
                    confirmed_match_count=(
                        candidate_snapshot.confirmed_match_count
                        if candidate_snapshot is not None
                        else {}
                    ),
                    unknown_constraint_count=(
                        candidate_snapshot.unknown_constraint_count
                        if candidate_snapshot is not None
                        else {}
                    ),
                    soft_preference_count=(
                        candidate_snapshot.soft_preference_count
                        if candidate_snapshot is not None
                        else {}
                    ),
                )
                after_residual = residual.ranking
                if set(after_residual) != set(before_residual):
                    raise RuntimeError("Top-10 residual changed C membership")
                ranking = [*after_residual, *ranking[10:]]
                reason_codes.append(
                    f"optional:ranking.top10_residual_reranker.v2:{residual.reason}"
                )
            if candidate_snapshot is not None:
                candidate_snapshot = replace(
                    candidate_snapshot,
                    pre_residual_top10=before_residual,
                    post_residual_top10=after_residual,
                )
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
            validated_ids = tuple(
                str(item["parent_asin"]) for item in response["recommendations"]
            )
            output_authority = self.union_ranker.filter.enforce(
                list(validated_ids), view
            )
            output_constraint_violations = len(validated_ids) - len(
                output_authority.ranking
            )
            if output_constraint_violations:
                raise ValueError("validated response contains a confirmed violation")
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
                    preview_reason=preview.reason,
                    preview_candidate_count=preview.candidate_count,
                    preview_score_flatness=preview.score_flatness,
                    dense_requested_per_view=dense_requested_per_view,
                    dense_output_k=dense_output_k,
                    dense_selection=dense_selection,
                    constraint_counts=constraint_counts,
                    output_constraint_violations=output_constraint_violations,
                    contribution_counts=contribution_counts,
                    query_views=query_views,
                    union_candidate_count=union_count,
                    semantic_backend=semantic.backend,
                    semantic_activation_reason=semantic.activation_reason,
                    semantic_changed=semantic.changed,
                    semantic_elapsed_ms=semantic.elapsed_ms,
                    semantic_failure_reason=semantic.failure_reason,
                    semantic_candidate_margin=semantic.candidate_margin,
                    semantic_candidate_entropy=semantic.candidate_entropy,
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
                    preview_executed=True,
                    safe_merge_executed=safe_merge_executed,
                    safe_ranker_executed=safe_ranker_executed,
                    normal_union_executed=normal_union_executed,
                    semantic_decision_reached=semantic_decision_reached,
                    semantic_executed=semantic_executed,
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
