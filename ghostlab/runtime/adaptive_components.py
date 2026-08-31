from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.adaptive_questions import AdaptiveQuestionPolicy
from ghostlab.policy.candidate_statistics import (
    CandidateFacetStore,
    CandidateStatistics,
)
from ghostlab.policy.eig_questions import CandidateEIGPolicy
from ghostlab.policy.signals import RetrievalSignals
from ghostlab.retrieval.cross_encoder import (
    CrossEncoderReranker,
    blend_ranking,
    product_passage,
)
from ghostlab.retrieval.dense import E5_SMALL_V2, DenseIndex
from ghostlab.retrieval.dense_diversity import (
    embedding_mmr_select,
    max_relevance_select,
    view_balanced_select,
)
from ghostlab.retrieval.dense_query_views import DenseQueryView, build_dense_query_views
from ghostlab.retrieval.ensemble import ModelRankEnsembleReranker, RankEnsembleAsset
from ghostlab.retrieval.filters import ConstraintAuthorityResult, CoverageAwareFilter
from ghostlab.retrieval.gbdt import (
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
)
from ghostlab.retrieval.multi_route import IntentRoute, MergedCandidatePool
from ghostlab.retrieval.profile import ProfilePriorReranker
from ghostlab.retrieval.sparse import query_terms
from ghostlab.retrieval.union_features import (
    SOURCE_AWARE_FEATURES,
    UNION_FEATURES,
    UnionFeatureStore,
)
from ghostlab.runtime.adaptive_config import (
    DiverseDenseTrackConfig,
    DualTrackRouterConfig,
    LocalLLMSemanticRankerConfig,
    ProactiveGuidanceConfig,
    RuntimeAdaptationConfig,
    UnionRankerConfig,
)
from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint
from ghostlab.state.v2_view import AdaptiveTurnContext, ConstraintView, V2StateView


@dataclass(frozen=True)
class RouteDecision:
    route: IntentRoute
    confidence: float
    reason: str
    abstained_to_precision: bool = False


class DualTrackRouter:
    """Observable, conservative Buying/Browsing router."""

    def __init__(self, config: DualTrackRouterConfig) -> None:
        self.config = config

    def browsing_marker(self, message: str) -> str | None:
        """Return observable evidence that the user is still exploring."""

        lowered = message.casefold()
        return next(
            (item for item in self.config.browsing_markers if item in lowered), None
        )

    def _constraint_specificity(
        self, constraints: Sequence[ConstraintView]
    ) -> tuple[float, int, int, int, int]:
        positive = [
            item
            for item in constraints
            if item.attribute != "category" and item.polarity == "include"
        ]
        exclusions = sum(item.polarity == "exclude" for item in constraints)
        hard = sum(
            item.strength == "hard" or item.attribute == "budget" for item in positive
        )
        explicit = sum(
            item.provenance in {"explicit", "simulator_answer"} for item in positive
        )
        score = (
            len(positive)
            + self.config.exclusion_specificity_weight * exclusions
            + self.config.hard_specificity_weight * hard
            + self.config.explicit_specificity_weight * explicit
        )
        return score, len(positive), exclusions, hard, explicit

    def decide(self, view: V2StateView, current_message: str) -> RouteDecision:
        lowered = current_message.casefold()
        query_tokens = re.findall(r"[a-z0-9]+", lowered)
        query_token_set = frozenset(query_tokens)
        marker = self.browsing_marker(current_message)

        def mentioned_now(item: ConstraintView) -> bool:
            if item.source_turn == view.turn:
                return True
            return any(
                (value_tokens := frozenset(re.findall(r"[a-z0-9]+", value)))
                and value_tokens <= query_token_set
                for value in item.values
            )

        current_constraints = tuple(
            item for item in view.active_constraints if mentioned_now(item)
        )
        historical_constraints = tuple(
            item for item in view.active_constraints if not mentioned_now(item)
        )
        current_score, current_positive, current_exclusions, _, _ = (
            self._constraint_specificity(current_constraints)
        )
        historical_score, historical_positive, historical_exclusions, _, _ = (
            self._constraint_specificity(historical_constraints)
        )
        current_attributes = {
            item.attribute
            for item in current_constraints
            if item.attribute != "category" and item.polarity == "include"
        }
        query_length_evidence = 0.0
        if current_positive or current_exclusions:
            query_length_evidence = self.config.current_query_length_weight * min(
                len(query_tokens) / self.config.current_query_length_cap,
                1.0,
            )
        specificity = (
            current_score
            + self.config.historical_specificity_weight * historical_score
            + self.config.current_attribute_coverage_weight * len(current_attributes)
            + query_length_evidence
        )
        current_has_category = any(
            item.attribute == "category" and item.polarity == "include"
            for item in current_constraints
        )
        category_only = (
            current_has_category and current_positive == 0 and current_exclusions == 0
        )
        browsing_evidence = (self.config.browsing_marker_weight if marker else 0.0) + (
            self.config.category_only_browsing_weight if category_only else 0.0
        )
        evidence = (
            f"current={current_score:.2f}:history={historical_score:.2f}:"
            f"attrs={len(current_attributes)}:tokens={len(query_tokens)}:"
            f"category_only={str(category_only).lower()}"
        )
        threshold = max(
            float(self.config.buying_min_specific_constraints),
            self.config.buying_specificity_threshold,
        )
        if view.exploratory_intent and view.asked_attributes:
            return RouteDecision(
                "browsing",
                0.9,
                (
                    "clarified_exploratory_intent:"
                    f"asked={view.asked_attributes[-1]}:{evidence}"
                ),
            )
        if browsing_evidence > 0.0 and browsing_evidence >= specificity:
            confidence = min(0.95, 0.6 + 0.1 * (browsing_evidence - specificity + 1.0))
            return RouteDecision(
                "browsing",
                confidence,
                (
                    f"observable_evidence:browsing={browsing_evidence:.2f}:"
                    f"buying={specificity:.2f}:marker={marker}:{evidence}"
                ),
            )
        if specificity >= threshold:
            confidence = min(0.95, 0.65 + 0.1 * specificity)
            if view.intent_epoch:
                confidence = max(
                    self.config.abstain_confidence,
                    confidence - self.config.correction_confidence_penalty,
                )
            return RouteDecision(
                "buying",
                confidence,
                (
                    f"observable_evidence:buying={specificity:.2f}:"
                    f"browsing={browsing_evidence:.2f}:{evidence}"
                ),
            )
        if (
            current_positive + historical_positive == 0
            and current_exclusions + historical_exclusions == 0
        ):
            return RouteDecision(
                "browsing", 0.75, f"open_ended_category_request:{evidence}"
            )
        confidence = min(0.8, 0.5 + 0.1 * specificity)
        if confidence < self.config.abstain_confidence:
            return RouteDecision(
                "buying", 1.0 - confidence, "low_confidence_precision_abstention", True
            )
        return RouteDecision("buying", confidence, f"specificity_threshold:{evidence}")


@dataclass(frozen=True)
class ProfileContext:
    terms: frozenset[str]
    confidence: float
    provenance: str
    active: bool
    reason: str


@dataclass(frozen=True)
class ProfileUpdate:
    values: tuple[str, ...]
    confidence: float
    provenance: str
    intent_epoch: int
    conflicts: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()


class ConflictSafeContextAdapter:
    def __init__(self, config: RuntimeAdaptationConfig) -> None:
        self.config = config

    def distil(
        self,
        state: StateBaselineV2 | AdaptiveTurnContext,
        overlay: ProfileUpdate | None = None,
    ) -> ProfileContext:
        constraints: Sequence[StructuredConstraint | ConstraintView]
        if isinstance(state, AdaptiveTurnContext):
            supplied_terms = state.supplied_profile_terms
            overlay_terms = frozenset(
                query_terms(" ".join(state.profile_overlay_values), 40)
                if state.profile_overlay_epoch == state.intent_epoch
                else ()
            )
            negative_values = [
                value
                for item in state.active_constraints
                if item.polarity == "exclude"
                for value in item.values
            ]
            constraints = state.active_constraints
        else:
            raw_tags = state.user_profile.get("preference_tags")
            tags = raw_tags if isinstance(raw_tags, (list, tuple, set)) else ()
            supplied_terms = frozenset(
                query_terms(" ".join(str(item) for item in tags), 40)
            )
            overlay_terms = frozenset(
                query_terms(" ".join(overlay.values), 40)
                if overlay is not None and overlay.intent_epoch == state.intent_epoch
                else ()
            )
            negative_values = state.constraint_values(polarity="exclude")
            constraints = tuple(state.active_constraints)
        terms = supplied_terms | overlay_terms
        negative = frozenset(query_terms(" ".join(negative_values), 80))
        if not terms:
            return ProfileContext(terms, 0.0, "supplied_profile", False, "no_tags")
        if terms & negative:
            return ProfileContext(
                terms,
                self.config.profile_confidence,
                "supplied_profile",
                False,
                "explicit_conflict",
            )
        explicit = sum(
            item.attribute != "category" and item.polarity == "include"
            for item in constraints
        )
        if explicit > self.config.maximum_explicit_constraints_for_profile:
            return ProfileContext(
                terms,
                self.config.profile_confidence,
                "supplied_profile",
                False,
                "explicit_intent_sufficient",
            )
        return ProfileContext(
            terms,
            self.config.profile_confidence,
            "supplied_profile",
            True,
            "ambiguous_request_profile_context",
        )

    @staticmethod
    def update(state: StateBaselineV2) -> ProfileUpdate:
        supplied = frozenset(
            query_terms(
                " ".join(
                    str(item)
                    for item in (state.user_profile.get("preference_tags") or ())
                ),
                40,
            )
        )
        excluded = frozenset(
            query_terms(
                " ".join(state.constraint_values(polarity="exclude")),
                80,
            )
        )
        values = tuple(
            dict.fromkeys(
                value
                for item in state.active_constraints
                if item.polarity == "include" and item.attribute != "category"
                for value in item.values
            )
        )
        attributes = tuple(
            dict.fromkeys(
                item.attribute
                for item in state.active_constraints
                if item.polarity == "include" and item.attribute != "category"
            )
        )
        return ProfileUpdate(
            values=values,
            confidence=1.0 if values else 0.0,
            provenance="explicit_session_evidence",
            intent_epoch=state.intent_epoch,
            conflicts=tuple(sorted(supplied & excluded)),
            attributes=attributes,
        )


@dataclass(frozen=True)
class DiverseDenseResult:
    identifiers: tuple[str, ...]
    relevance_scores: Mapping[str, float]
    query_views: tuple[str, ...]
    elapsed_ms: float
    per_view_ranks: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    per_view_scores: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    requested_per_view: int = 0
    output_k: int = 0
    selection: str = "multiview_max_relevance"


class DiverseDenseTrack:
    def __init__(
        self,
        catalog_path: str | Path,
        config: DiverseDenseTrackConfig,
        *,
        project_root: Path,
        index: DenseIndex | None = None,
    ) -> None:
        self.config = config
        self.index = index or DenseIndex(
            catalog_path,
            E5_SMALL_V2,
            cache_dir=project_root / config.cache_dir,
            model_path=project_root / config.model_path,
            local_files_only=True,
        )

    def search(
        self, view: V2StateView, *, overloaded: bool = False
    ) -> DiverseDenseResult:
        started = time.perf_counter()
        views = list(build_dense_query_views(view))
        if (
            self.config.profile_query_view_enabled
            and isinstance(view, AdaptiveTurnContext)
            and view.supplied_profile_terms
        ):
            profile_query = " ".join(sorted(view.supplied_profile_terms))
            if profile_query.casefold() not in {
                item.query_text.casefold() for item in views
            }:
                views.append(DenseQueryView("profile_context", profile_query))
        retrieval_per_view = (
            self.config.overload_retrieval_per_view
            if overloaded
            else self.config.retrieval_per_view
        )
        output_k = self.config.overload_output_k if overloaded else self.config.output_k
        candidates: list[str] = []
        relevance: dict[str, float] = {}
        per_view_ranks: dict[str, dict[str, int]] = {}
        per_view_scores: dict[str, dict[str, float]] = {}
        view_rankings: dict[str, list[str]] = {}
        for query_view in views:
            result = self.index.search(query_view.query_text, retrieval_per_view)
            view_rankings[query_view.name] = []
            per_view_ranks[query_view.name] = {}
            per_view_scores[query_view.name] = {}
            for item in result.items:
                candidates.append(item.parent_asin)
                view_rankings[query_view.name].append(item.parent_asin)
                score = float(item.normalized_score or 0.0)
                per_view_ranks[query_view.name][item.parent_asin] = item.rank
                per_view_scores[query_view.name][item.parent_asin] = score
                relevance[item.parent_asin] = max(
                    relevance.get(item.parent_asin, -math.inf), score
                )
        if self.config.selection == "view_balanced":
            selected = view_balanced_select(view_rankings, relevance, output_k=output_k)
        elif self.config.selection == "embedding_mmr":
            embeddings = {
                identifier: self.index.embeddings[index]
                for index, identifier in enumerate(self.index.identifiers)
                if identifier in relevance
            }
            selected = embedding_mmr_select(
                candidates,
                relevance,
                embeddings,
                output_k=output_k,
                relevance_weight=self.config.mmr_relevance_weight,
            )
        else:
            selected = max_relevance_select(candidates, relevance, output_k=output_k)
        return DiverseDenseResult(
            identifiers=tuple(selected),
            relevance_scores={item: relevance[item] for item in selected},
            query_views=tuple(item.name for item in views),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            per_view_ranks=per_view_ranks,
            per_view_scores=per_view_scores,
            requested_per_view=retrieval_per_view,
            output_k=output_k,
            selection=self.config.selection,
        )


class BrowsingSafeRanker:
    """Ranks an overloaded dense pool without invoking union or semantic stages."""

    def __init__(
        self,
        catalog_path: str | Path,
        config: DiverseDenseTrackConfig,
        *,
        project_root: Path,
    ) -> None:
        self.config = config
        self.reranker: LambdaMARTReranker | None = None
        if config.safe_ranker_backend == "gbdt":
            assert config.safe_ranker_model_path is not None
            assert config.safe_ranker_model_sha256 is not None
            path = project_root / config.safe_ranker_model_path
            if not path.is_file():
                raise FileNotFoundError(f"missing Browsing safe ranker asset: {path}")
            if _sha256(path) != config.safe_ranker_model_sha256:
                raise ValueError("Browsing safe ranker asset hash mismatch")
            self.reranker = LambdaMARTReranker(
                GBDTFeatureStore(catalog_path), LambdaMARTModel.load(path)
            )

    def rank(self, query: str, ranking: list[str]) -> list[str]:
        if self.reranker is None or self.config.safe_ranker_weight == 0.0:
            return list(ranking)
        count = min(self.config.safe_rerank_k, len(ranking))
        head = ranking[:count]
        matrix = self.reranker.features.matrix(
            query, head, self.reranker.model.feature_names
        )
        predictions = self.reranker.model.predict(matrix)
        low = float(predictions.min())
        high = float(predictions.max())
        learned = (
            np.zeros(len(predictions), dtype=np.float64)
            if math.isclose(low, high)
            else (predictions - low) / (high - low)
        )
        weight = self.config.safe_ranker_weight
        original = {identifier: rank for rank, identifier in enumerate(head)}
        scores = {
            identifier: (1.0 - weight)
            * (1.0 if count == 1 else 1.0 - rank / (count - 1))
            + weight * float(learned[rank])
            for rank, identifier in enumerate(head)
        }
        ordered = sorted(
            head,
            key=lambda identifier: (
                -scores[identifier],
                original[identifier],
                identifier,
            ),
        )
        result = [*ordered, *ranking[count:]]
        if len(result) != len(ranking) or set(result) != set(ranking):
            raise ValueError("Browsing safe ranker changed candidate membership")
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.parts
    )
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class UnionAwareRanker:
    def __init__(
        self,
        catalog_path: str | Path,
        config: UnionRankerConfig,
        *,
        project_root: Path,
    ) -> None:
        self.config = config
        self.filter = CoverageAwareFilter(catalog_path)
        self.union_features = UnionFeatureStore(GBDTFeatureStore(catalog_path))
        self.reranker: LambdaMARTReranker | ModelRankEnsembleReranker | None = None
        self.auxiliary_reranker: (
            LambdaMARTReranker | ModelRankEnsembleReranker | None
        ) = None
        if config.backend == "gbdt":
            assert config.model_path is not None
            assert config.model_sha256 is not None
            path = project_root / config.model_path
            if not path.is_file():
                raise FileNotFoundError(f"missing union ranker asset: {path}")
            if _sha256(path) != config.model_sha256:
                raise ValueError("union ranker asset hash mismatch")
            model = LambdaMARTModel.load(path)
            if model.candidate_id == "adaptive_union_gbdt_1650_final_v1" and tuple(
                model.feature_names
            ) != tuple(UNION_FEATURES):
                raise ValueError(
                    "source-aware union model feature schema does not match runtime"
                )
            self.reranker = LambdaMARTReranker(self.union_features.base, model)
        elif config.backend == "rank_ensemble":
            assert config.model_path is not None
            assert config.model_sha256 is not None
            path = project_root / config.model_path
            if not path.is_file():
                raise FileNotFoundError(f"missing union ensemble asset: {path}")
            if _sha256(path) != config.model_sha256:
                raise ValueError("union ensemble asset hash mismatch")
            self.reranker = ModelRankEnsembleReranker.from_asset(
                GBDTFeatureStore(catalog_path),
                RankEnsembleAsset.load(path),
                project_root=project_root,
            )
        if config.auxiliary_backend in {"gbdt", "rank_ensemble"}:
            assert config.auxiliary_model_path is not None
            assert config.auxiliary_model_sha256 is not None
            auxiliary_path = project_root / config.auxiliary_model_path
            if not auxiliary_path.is_file():
                raise FileNotFoundError(
                    f"missing auxiliary ranker asset: {auxiliary_path}"
                )
            if _sha256(auxiliary_path) != config.auxiliary_model_sha256:
                raise ValueError("auxiliary ranker asset hash mismatch")
            if config.auxiliary_backend == "gbdt":
                self.auxiliary_reranker = LambdaMARTReranker(
                    GBDTFeatureStore(catalog_path),
                    LambdaMARTModel.load(auxiliary_path),
                )
            else:
                self.auxiliary_reranker = ModelRankEnsembleReranker.from_asset(
                    GBDTFeatureStore(catalog_path),
                    RankEnsembleAsset.load(auxiliary_path),
                    project_root=project_root,
                )

    def _apply_auxiliary(
        self,
        query: str,
        ranking: list[str],
        pool: MergedCandidatePool,
    ) -> list[str]:
        """Blend one historical ranker as a bounded secondary signal.

        The source-aware union GBDT has already produced ``ranking``.  The
        auxiliary implementation can only influence its bounded head and
        cannot alter membership, hard-constraint filtering, or the compulsory
        primary model.
        """
        if self.config.auxiliary_backend == "none" or not ranking:
            return ranking
        head_size = min(self.config.auxiliary_rerank_k, len(ranking))
        head = ranking[:head_size]
        if self.config.auxiliary_backend == "fixed_lexical":
            evidence = {item.parent_asin: item for item in pool.retain(head).candidates}
            positions = {identifier: index for index, identifier in enumerate(head)}
            auxiliary_order = sorted(
                head,
                key=lambda identifier: (
                    evidence[identifier].keyword_rank is None,
                    evidence[identifier].keyword_rank
                    if evidence[identifier].keyword_rank is not None
                    else len(head) + 1,
                    -float(evidence[identifier].keyword_score or 0.0),
                    positions[identifier],
                    identifier,
                ),
            )
        else:
            assert self.auxiliary_reranker is not None
            auxiliary_order = self.auxiliary_reranker.rerank(
                query,
                head,
                rerank_k=head_size,
            )
        if set(auxiliary_order) != set(head) or len(auxiliary_order) != len(head):
            raise ValueError("auxiliary ranker changed candidate membership")
        auxiliary_positions = {
            identifier: index for index, identifier in enumerate(auxiliary_order)
        }
        auxiliary_scores = tuple(
            1.0
            if head_size == 1
            else 1.0 - auxiliary_positions[identifier] / max(1, head_size - 1)
            for identifier in head
        )
        blended = blend_ranking(
            head,
            auxiliary_scores,
            weight=self.config.auxiliary_weight,
        )
        return [*blended, *ranking[head_size:]]

    def rank(
        self,
        query: str,
        pool: MergedCandidatePool,
        *,
        positive_constraints: dict[str, list[str]],
        negative_constraints: dict[str, list[str]] | None = None,
        context: V2StateView | None = None,
        authority: ConstraintAuthorityResult | None = None,
        profile_terms: frozenset[str] = frozenset(),
    ) -> list[str]:
        ranking = pool.identifiers
        if context is not None:
            authority = authority or self.filter.enforce(ranking, context)
            ranking = list(authority.ranking)
            pool = pool.retain(ranking)
        elif pool.route == "buying":
            ranking = self.filter.apply_strict(
                ranking,
                positive_constraints,
                negative_constraints,
            )
        if isinstance(self.reranker, LambdaMARTReranker):
            head_size = min(self.config.rerank_k, len(ranking))
            head_pool = pool.retain(ranking[:head_size])
            if set(self.reranker.model.feature_names) & set(SOURCE_AWARE_FEATURES):
                matrix = self.union_features.matrix(
                    query,
                    head_pool,
                    self.reranker.model.feature_names,
                    authority=authority,
                    profile_terms=profile_terms,
                )
            else:
                matrix = self.reranker.features.matrix(
                    query, ranking[:head_size], self.reranker.model.feature_names
                )
            predictions = self.reranker.model.predict(matrix)
            low = float(predictions.min()) if len(predictions) else 0.0
            high = float(predictions.max()) if len(predictions) else 0.0
            learned = (
                np.zeros(len(predictions), dtype=np.float64)
                if math.isclose(low, high)
                else (predictions - low) / (high - low)
            )
            positions = {
                identifier: index
                for index, identifier in enumerate(ranking[:head_size])
            }
            if (
                pool.route == "buying"
                and self.config.buying_mode == "sparse_dominant_residual"
            ):
                weight = self.config.buying_residual_weight
                scores = {
                    identifier: (1.0 - weight)
                    * (1.0 - positions[identifier] / max(1, head_size - 1))
                    + weight * float(learned[index])
                    for index, identifier in enumerate(ranking[:head_size])
                }
            else:
                scores = {
                    identifier: float(learned[index])
                    for index, identifier in enumerate(ranking[:head_size])
                }
            ordered = sorted(
                ranking[:head_size],
                key=lambda identifier: (
                    -scores[identifier],
                    positions[identifier],
                    identifier,
                ),
            )
            ranking = [*ordered, *ranking[head_size:]]
        elif self.reranker is not None:
            ranking = self.reranker.rerank(
                query, ranking, rerank_k=min(self.config.rerank_k, len(ranking))
            )
        return self._apply_auxiliary(query, ranking, pool)


@dataclass(frozen=True)
class SemanticRankingResult:
    ranking: tuple[str, ...]
    changed: bool
    elapsed_ms: float
    backend: str
    activation_reason: str = "semantic_ranking_required"
    failure_reason: str | None = None
    candidate_margin: float | None = None
    candidate_entropy: float | None = None


@dataclass(frozen=True)
class SemanticActivationDecision:
    active: bool
    reason: str


class SemanticActivationPolicy:
    """Keep the LLM slot mandatory while invoking it only for semantic need."""

    def __init__(self, config: LocalLLMSemanticRankerConfig) -> None:
        self.config = config

    def decide(
        self,
        route: RouteDecision,
        view: V2StateView,
        *,
        overloaded: bool,
        signals: RetrievalSignals | None = None,
    ) -> SemanticActivationDecision:
        del signals
        if route.abstained_to_precision:
            return SemanticActivationDecision(False, "precision_abstention")
        if route.route == "browsing":
            reason = (
                "overloaded_browsing_semantic_retrieval"
                if overloaded
                else "browsing_semantic_retrieval"
            )
            return SemanticActivationDecision(True, reason)
        del view
        return SemanticActivationDecision(False, "high_confidence_buying")


def causal_chat_template_options(model_id: str) -> dict[str, Any]:
    """Return model-family options required by direct next-token scoring."""

    normalized = model_id.lower().replace("-", "")
    return {"enable_thinking": False} if "qwen3" in normalized else {}


class CausalRelevanceScorer:
    """Batched local causal-LLM yes/no relevance scoring."""

    PROMPT_MEANING = (
        "shopping-relevance-v1: decide whether the product matches the request "
        "and constraints; answer yes or no"
    )

    def __init__(
        self,
        documents: Mapping[str, str],
        config: LocalLLMSemanticRankerConfig,
        *,
        project_root: Path,
    ) -> None:
        self.documents = documents
        self.config = config
        model_path = project_root / config.model_path
        if not model_path.is_dir():
            raise FileNotFoundError(f"missing causal LLM asset: {model_path}")
        if _sha256_directory(model_path) != config.model_sha256:
            raise ValueError("causal LLM asset hash mismatch")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        use_mps = config.device == "mps" or (
            config.device == "auto" and torch.backends.mps.is_available()
        )
        self.device = torch.device("mps" if use_mps else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=config.model_revision,
            local_files_only=True,
        )
        model_options: dict[str, object] = {
            "revision": config.model_revision,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if use_mps:
            model_options["dtype"] = torch.float16
        model: Any = AutoModelForCausalLM.from_pretrained(model_path, **model_options)
        self.model = model.to(self.device)
        self.model.eval()
        self.chat_template_options = causal_chat_template_options(config.model_id)
        self.thinking_mode = (
            "disabled" if self.chat_template_options else "model_default"
        )
        self.yes_token = self._single_token_label("yes")
        self.no_token = self._single_token_label("no")
        if self.yes_token == self.no_token:
            raise ValueError("causal LLM yes/no labels resolve to the same token")

    def _single_token_label(self, label: str) -> int:
        for candidate in (f" {label}", label, f" {label.title()}", label.title()):
            encoded = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                return int(encoded[0])
        raise ValueError(f"causal LLM label-token incompatibility: {label}")

    def _prompt(self, query: str, document: str) -> str:
        content = (
            "You are a shopping relevance judge. Decide whether the product "
            "matches the request and its constraints. Answer yes or no.\n"
            f"Request: {query}\n"
            f"Product: {document}"
        )
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_options,
            )
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("causal LLM chat template returned an invalid prompt")
            return rendered
        return f"{content}\nRelevant:"

    def scores(self, query: str, ranking: Sequence[str]) -> tuple[float, ...]:
        prompts = [
            self._prompt(query, self.documents.get(identifier, ""))
            for identifier in ranking
        ]
        values: list[float] = []
        for start in range(0, len(prompts), self.config.batch_size):
            batch = self.tokenizer(
                prompts[start : start + self.config.batch_size],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                logits = self.model(**batch).logits
            positions = batch["attention_mask"].sum(dim=1) - 1
            rows = self.torch.arange(len(positions), device=self.device)
            yes_logits = logits[rows, positions, self.yes_token]
            no_logits = logits[rows, positions, self.no_token]
            probabilities = self.torch.sigmoid(yes_logits - no_logits)
            values.extend(float(item) for item in probabilities.cpu())
        if len(values) != len(ranking) or not all(
            math.isfinite(item) for item in values
        ):
            raise ValueError("causal LLM returned invalid relevance scores")
        return tuple(values)


class BoundedLocalLLMSemanticRanker:
    """Bounded local Transformer language-model stage for semantic ranking."""

    def __init__(
        self,
        catalog_path: str | Path,
        config: LocalLLMSemanticRankerConfig,
        *,
        project_root: Path,
        reranker: CrossEncoderReranker | None = None,
        llm_scorer: object | None = None,
    ) -> None:
        self.catalog_path = catalog_path
        self.config = config
        self.project_root = project_root
        self._reranker = reranker
        self._llm_scorer = llm_scorer
        self._lock = Lock()
        self.primary_attempts = 0
        self.primary_successes = 0
        self.fallback_count = 0
        self.failure_counts: dict[str, int] = {}
        self.documents: dict[str, str] = {}
        catalog_file = Path(catalog_path)
        if catalog_file.is_file():
            with catalog_file.open(encoding="utf-8") as handle:
                import json

                for line in handle:
                    product = json.loads(line)
                    self.documents[str(product["parent_asin"])] = product_passage(
                        product
                    )

    def _model(self) -> CrossEncoderReranker:
        if self._reranker is not None:
            return self._reranker
        with self._lock:
            if self._reranker is None:
                model_path = self.project_root / self.config.fallback_model_path
                if not model_path.is_dir():
                    raise FileNotFoundError(
                        f"missing semantic model asset: {model_path}"
                    )
                self._reranker = CrossEncoderReranker(
                    self.catalog_path,
                    model_name=str(model_path),
                    revision=self.config.fallback_model_revision,
                    cache_folder=self.project_root / self.config.fallback_cache_dir,
                    local_files_only=True,
                )
        return self._reranker

    def _llm(self) -> object:
        if self._llm_scorer is not None:
            return self._llm_scorer
        with self._lock:
            if self._llm_scorer is None:
                self._llm_scorer = CausalRelevanceScorer(
                    self.documents,
                    self.config,
                    project_root=self.project_root,
                )
        return self._llm_scorer

    def rank(self, query: str, ranking: list[str]) -> SemanticRankingResult:
        started = time.perf_counter()
        bounded = list(ranking)
        head_size = min(self.config.rerank_k, len(bounded))
        head = bounded[:head_size]
        backend: str = self.config.model_id
        failure_reason: str | None = None
        try:
            if self.config.backend == "minilm_cross_encoder_control":
                result = self._model().rerank(
                    query,
                    bounded,
                    rerank_k=head_size,
                    weight=self.config.fallback_weight,
                )
                backend = "minilm_cross_encoder_control"
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if len(result) != len(bounded) or set(result) != set(bounded):
                    raise ValueError("semantic control changed candidate membership")
                return SemanticRankingResult(
                    ranking=tuple(result),
                    changed=result != bounded,
                    elapsed_ms=elapsed_ms,
                    backend=backend,
                )
            self.primary_attempts += 1
            primary_started = time.perf_counter()
            scorer = self._llm()
            scores = scorer.scores(query, head)  # type: ignore[attr-defined]
            if (
                time.perf_counter() - primary_started
            ) * 1000.0 > self.config.timeout_ms:
                raise TimeoutError("causal LLM exceeded its deadline")
            ordered = blend_ranking(head, scores, weight=self.config.weight)
            result = [*ordered, *bounded[head_size:]]
            self.primary_successes += 1
        except Exception as error:  # noqa: BLE001 - declared MiniLM fallback
            failure_reason = type(error).__name__
            self.failure_counts[failure_reason] = (
                self.failure_counts.get(failure_reason, 0) + 1
            )
            self.fallback_count += 1
            result = self._model().rerank(
                query,
                bounded,
                rerank_k=head_size,
                weight=self.config.fallback_weight,
            )
            backend = "fallback_minilm_cross_encoder"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > self.config.timeout_ms:
            raise TimeoutError("local semantic ranking exceeded its deadline")
        if len(result) != len(bounded) or set(result) != set(bounded):
            raise ValueError("semantic ranker changed candidate membership")
        return SemanticRankingResult(
            ranking=tuple(result),
            changed=result != bounded,
            elapsed_ms=elapsed_ms,
            backend=backend,
            failure_reason=failure_reason,
        )

    def diagnostics(self) -> dict[str, object]:
        scorer = self._llm_scorer
        chat_template = (
            getattr(getattr(scorer, "tokenizer", None), "chat_template", None)
            if scorer is not None
            else None
        )
        return {
            "primary_attempts": self.primary_attempts,
            "primary_successes": self.primary_successes,
            "fallback_count": self.fallback_count,
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "prompt_meaning": CausalRelevanceScorer.PROMPT_MEANING,
            "prompt_meaning_sha256": hashlib.sha256(
                CausalRelevanceScorer.PROMPT_MEANING.encode()
            ).hexdigest(),
            "chat_template_used": bool(chat_template),
            "chat_template_sha256": (
                hashlib.sha256(str(chat_template).encode()).hexdigest()
                if chat_template
                else None
            ),
            "thinking_mode": getattr(scorer, "thinking_mode", "not_loaded"),
            "chat_template_options": dict(getattr(scorer, "chat_template_options", {})),
        }


@dataclass(frozen=True)
class GuidanceDecision:
    overloaded: bool
    ask_attribute: AskAttribute | None
    reason: str
    values: Mapping[AskAttribute | None, float]


@dataclass(frozen=True)
class RetrievalPreviewDecision:
    overloaded: bool
    reason: str
    candidate_count: int
    specific_constraint_count: int
    score_flatness: float


class OverGeneralityGuidance:
    def __init__(
        self,
        catalog_path: str | Path,
        config: ProactiveGuidanceConfig,
    ) -> None:
        self.config = config
        self.facets = CandidateFacetStore(catalog_path)
        self.policy = CandidateEIGPolicy(
            question_value_margin=config.question_value_margin,
            max_question_turn=config.max_question_turn,
            broad_discovery_turns=config.broad_discovery_turns,
            fallback=AdaptiveQuestionPolicy(
                initial_other_turns=0,
                other_refresh_interval=0,
                max_question_turn=config.max_question_turn,
            ),
        )

    def preview(
        self,
        context: AdaptiveTurnContext,
        keyword_ids: Sequence[str],
        keyword_scores: Sequence[float],
        category_ids: Sequence[str],
    ) -> RetrievalPreviewDecision:
        """Cheap deterministic overload decision made before dense expansion."""

        candidates = list(
            dict.fromkeys(
                [
                    *keyword_ids[: self.config.preview_keyword_k],
                    *category_ids[: self.config.preview_category_k],
                ]
            )
        )
        specific = sum(
            item.attribute != "category" and item.polarity == "include"
            for item in context.active_constraints
        )
        bounded_scores = list(keyword_scores[: self.config.preview_keyword_k])
        if len(bounded_scores) < 2 or max(bounded_scores) <= 0.0:
            flatness = 1.0 if candidates else 0.0
        else:
            high = max(bounded_scores)
            low = min(bounded_scores)
            flatness = max(0.0, min(1.0, 1.0 - (high - low) / high))
        threshold = min(
            self.config.overload_min_candidates, self.config.preview_min_candidates
        )
        overloaded = (
            len(candidates) >= threshold
            and specific <= self.config.overload_max_specific_constraints
        )
        return RetrievalPreviewDecision(
            overloaded=overloaded,
            reason=(
                "preview_overloaded" if overloaded else "preview_specific_or_bounded"
            ),
            candidate_count=len(candidates),
            specific_constraint_count=specific,
            score_flatness=flatness,
        )

    def decide(
        self,
        state: StateBaselineV2 | AdaptiveTurnContext,
        candidate_ids: list[str],
        *,
        turn: int,
        message: str,
        overloaded: bool | None = None,
        profile_known_attributes: frozenset[str] = frozenset(),
    ) -> GuidanceDecision:
        specific = sum(
            item.attribute != "category" and item.polarity == "include"
            for item in state.active_constraints
        )
        if overloaded is None:
            overloaded = (
                len(candidate_ids) >= self.config.overload_min_candidates
                and specific <= self.config.overload_max_specific_constraints
            )
        statistics: CandidateStatistics = self.facets.summarize(
            candidate_ids, limit=self.config.question_candidate_k
        )
        question = self.policy.decide(
            state,
            statistics,
            turn=turn,
            message=message,
            unavailable_attributes=profile_known_attributes,
        )
        reason = f"overloaded:{question.reason}" if overloaded else question.reason
        return GuidanceDecision(
            overloaded=overloaded,
            ask_attribute=question.ask_attribute,
            reason=reason,
            values=question.values,
        )


def apply_profile_context(
    reranker: ProfilePriorReranker,
    session_id: str,
    ranking: list[str],
    context: ProfileContext,
    config: RuntimeAdaptationConfig,
) -> list[str]:
    if not context.active:
        return ranking
    del session_id
    return reranker.rerank_terms(
        ranking,
        context.terms,
        weight=config.profile_weight * context.confidence,
        rerank_k=min(50, len(ranking)),
    )


__all__ = [
    "BoundedLocalLLMSemanticRanker",
    "ConflictSafeContextAdapter",
    "DiverseDenseResult",
    "DiverseDenseTrack",
    "DualTrackRouter",
    "GuidanceDecision",
    "OverGeneralityGuidance",
    "ProfileContext",
    "ProfileUpdate",
    "RetrievalPreviewDecision",
    "RouteDecision",
    "SemanticRankingResult",
    "UnionAwareRanker",
    "apply_profile_context",
]
