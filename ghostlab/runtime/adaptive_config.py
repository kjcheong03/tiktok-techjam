from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _RequiredConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StateV2Config(_RequiredConfig):
    component: Literal["state_v2"] = "state_v2"
    catalog_normalizer_enabled: bool = False
    catalog_ontology_path: str | None = None
    catalog_ontology_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    constraint_normalization_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def normalizer_asset_is_pinned(self) -> StateV2Config:
        declared = (self.catalog_ontology_path, self.catalog_ontology_sha256)
        if any(item is not None for item in declared) and not all(
            item is not None for item in declared
        ):
            raise ValueError(
                "catalog ontology path and SHA256 must be declared together"
            )
        if self.catalog_normalizer_enabled and not all(
            item is not None for item in declared
        ):
            raise ValueError("catalog normalization requires a pinned ontology asset")
        return self


class DualTrackRouterConfig(_RequiredConfig):
    component: Literal["buying_browsing_router"] = "buying_browsing_router"
    buying_min_specific_constraints: int = Field(default=1, ge=1, le=8)
    browsing_markers: tuple[str, ...] = (
        "still exploring",
        "browsing",
        "ideas",
        "not sure",
        "use your judgment",
        "recommend something",
    )
    abstain_confidence: float = Field(default=0.6, ge=0.5, le=1.0)
    buying_specificity_threshold: float = Field(default=1.0, ge=0.0, le=8.0)
    correction_confidence_penalty: float = Field(default=0.1, ge=0.0, le=0.5)
    exclusion_specificity_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    hard_specificity_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    explicit_specificity_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    browsing_marker_weight: float = Field(default=2.5, ge=0.0, le=8.0)
    historical_specificity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    current_attribute_coverage_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    current_query_length_weight: float = Field(default=0.1, ge=0.0, le=0.5)
    current_query_length_cap: int = Field(default=12, ge=1, le=50)
    category_only_browsing_weight: float = Field(default=0.75, ge=0.0, le=4.0)


class PrecisionTrackConfig(_RequiredConfig):
    component: Literal["field_bm25_precision"] = "field_bm25_precision"
    retrieval_k: int = Field(default=200, ge=10, le=1000)
    field_weights: tuple[float, float, float, float, float, float] = (
        2.0,
        8.0,
        4.0,
        2.5,
        1.5,
        1.0,
    )

    @field_validator("field_weights")
    @classmethod
    def valid_weights(
        cls, value: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        if not all(math.isfinite(item) and item >= 0.0 for item in value):
            raise ValueError("BM25 field weights must be finite and non-negative")
        return value


class DiverseDenseTrackConfig(_RequiredConfig):
    component: Literal["diverse_e5_multiview"] = "diverse_e5_multiview"
    model_path: str = "artifacts/cache/models/e5-small-v2"
    cache_dir: str = "artifacts/cache/dense"
    retrieval_per_view: int = Field(default=400, ge=10, le=1000)
    output_k: int = Field(default=200, ge=10, le=1000)
    selection: Literal["multiview_max_relevance", "view_balanced", "embedding_mmr"] = (
        "multiview_max_relevance"
    )
    mmr_relevance_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    overload_retrieval_per_view: int = Field(default=80, ge=5, le=500)
    overload_output_k: int = Field(default=80, ge=5, le=500)
    profile_query_view_enabled: bool = False
    safe_ranker_backend: Literal["deterministic", "gbdt"] = "deterministic"
    safe_ranker_model_path: str | None = None
    safe_ranker_model_sha256: str | None = None
    safe_rerank_k: int = Field(default=200, ge=10, le=1000)
    safe_ranker_weight: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def safe_ranker_asset_is_declared(self) -> DiverseDenseTrackConfig:
        if self.safe_ranker_backend == "gbdt" and (
            not self.safe_ranker_model_path or not self.safe_ranker_model_sha256
        ):
            raise ValueError("GBDT Browsing safe ranking requires a pinned local model")
        if self.overload_retrieval_per_view > self.retrieval_per_view:
            raise ValueError("overload dense depth cannot exceed normal dense depth")
        if self.overload_output_k > self.output_k:
            raise ValueError("overload output depth cannot exceed normal output depth")
        return self


class MultiRouteMergeConfig(_RequiredConfig):
    component: Literal["keyword_category_vector_merge"] = (
        "keyword_category_vector_merge"
    )
    strategy: Literal["weighted", "rrf", "sparse_first_union"] = "weighted"
    rrf_constant: int = Field(default=60, ge=1, le=200)
    category_k: int = Field(default=120, ge=10, le=1000)
    merged_k: int = Field(default=320, ge=20, le=1000)
    buying_vector_support_k: int = Field(default=40, ge=1, le=200)
    browsing_keyword_support_k: int = Field(default=40, ge=1, le=200)
    buying_keyword_weight: float = Field(default=0.9, ge=0.0, le=1.0)
    buying_category_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    buying_vector_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    browsing_vector_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    browsing_category_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    browsing_keyword_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> MultiRouteMergeConfig:
        if not math.isclose(
            self.buying_keyword_weight
            + self.buying_category_weight
            + self.buying_vector_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Buying merge weights must sum to one")
        if not math.isclose(
            self.browsing_vector_weight
            + self.browsing_category_weight
            + self.browsing_keyword_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Browsing merge weights must sum to one")
        if not (
            self.buying_keyword_weight > self.buying_category_weight
            and self.buying_keyword_weight > self.buying_vector_weight
        ):
            raise ValueError("Buying must remain keyword-primary")
        if not (
            self.browsing_vector_weight > self.browsing_category_weight
            and self.browsing_vector_weight > self.browsing_keyword_weight
        ):
            raise ValueError("Browsing must remain vector-primary")
        if any(
            weight <= 0.0
            for weight in (
                self.buying_keyword_weight,
                self.buying_category_weight,
                self.buying_vector_weight,
                self.browsing_vector_weight,
                self.browsing_category_weight,
                self.browsing_keyword_weight,
            )
        ):
            raise ValueError("all three evidence sources must remain active")
        return self


class UnionRankerConfig(_RequiredConfig):
    component: Literal["union_aware_ranker"] = "union_aware_ranker"
    backend: Literal["deterministic", "gbdt", "rank_ensemble"] = "gbdt"
    model_path: str | None = "artifacts/models/adaptive_union_gbdt_v1.json"
    model_sha256: str | None = (
        "d1c336a3b7fd0fa13d7d5bd5ef87c97503fd339ff0682723a6485baded35f59c"
    )
    rerank_k: int = Field(default=320, ge=10, le=1000)
    buying_mode: Literal["direct", "sparse_dominant_residual"] = (
        "sparse_dominant_residual"
    )
    buying_residual_weight: float = Field(default=0.25, ge=0.0, le=0.49)
    auxiliary_technique_id: str | None = None
    auxiliary_backend: Literal["none", "fixed_lexical", "gbdt", "rank_ensemble"] = (
        "none"
    )
    auxiliary_model_path: str | None = None
    auxiliary_model_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    auxiliary_weight: float = Field(default=0.1, ge=0.0, le=0.25)
    auxiliary_rerank_k: int = Field(default=50, ge=10, le=200)

    @model_validator(mode="after")
    def gbdt_asset_is_declared(self) -> UnionRankerConfig:
        if self.backend in {"gbdt", "rank_ensemble"} and (
            not self.model_path or not self.model_sha256
        ):
            raise ValueError("learned union ranking requires a pinned local model")
        learned_auxiliary = self.auxiliary_backend in {"gbdt", "rank_ensemble"}
        if learned_auxiliary and (
            not self.auxiliary_model_path or not self.auxiliary_model_sha256
        ):
            raise ValueError("learned auxiliary ranking requires a pinned local model")
        if self.auxiliary_backend in {"none", "fixed_lexical"} and (
            self.auxiliary_model_path is not None
            or self.auxiliary_model_sha256 is not None
        ):
            raise ValueError("non-learned auxiliary ranking cannot declare a model")
        if self.auxiliary_backend == "none" and self.auxiliary_technique_id is not None:
            raise ValueError("disabled auxiliary ranking cannot declare a technique")
        if self.auxiliary_backend != "none" and not self.auxiliary_technique_id:
            raise ValueError("enabled auxiliary ranking must declare its technique")
        return self


class AdaptiveExtensionsConfig(_RequiredConfig):
    """Optional additions at declared hooks around the fixed 1A-3B workflow."""

    quality_prior_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_rerank_k: int = Field(default=50, ge=10, le=500)
    query_prf_enabled: bool = False
    query_prf_feedback_k: int = Field(default=5, ge=2, le=50)
    query_prf_minimum_support: float = Field(default=0.4, gt=0.0, le=1.0)
    query_prf_max_terms: int = Field(default=4, ge=0, le=20)
    query_prf_max_added_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    facet_diversity_enabled: bool = False
    facet_relevance_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    facet_rerank_k: int = Field(default=30, ge=10, le=200)
    facet_output_k: int = Field(default=10, ge=2, le=50)
    facet_max_turn: int = Field(default=2, ge=1, le=9)
    facet_max_constraints: int = Field(default=1, ge=0, le=10)
    minilm_dense_view_enabled: bool = False
    minilm_dense_model_path: str = "artifacts/cache/models/all-MiniLM-L6-v2"
    minilm_dense_model_revision: str = Field(
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", min_length=1
    )
    minilm_dense_cache_dir: str = "artifacts/cache/dense"
    minilm_dense_retrieval_k: int = Field(default=80, ge=10, le=400)
    minilm_dense_weight: float = Field(default=0.15, gt=0.0, le=0.35)
    top10_residual_enabled: bool = False
    top10_residual_model_path: str | None = None
    top10_residual_model_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    top10_residual_fit_receipt_path: str | None = None
    top10_residual_fit_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    top10_residual_rerank_depth: int = Field(default=10, ge=2, le=10)
    top10_residual_model_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    top10_residual_minimum_expected_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    top10_residual_minimum_probability_margin: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    top10_residual_maximum_moved_ids: int = Field(default=10, ge=2, le=10)

    @model_validator(mode="after")
    def valid_depths(self) -> AdaptiveExtensionsConfig:
        if self.facet_output_k > self.facet_rerank_k:
            raise ValueError("facet output depth cannot exceed its rerank depth")
        pairs = (
            (
                self.top10_residual_model_path,
                self.top10_residual_model_sha256,
                "residual model",
            ),
            (
                self.top10_residual_fit_receipt_path,
                self.top10_residual_fit_receipt_sha256,
                "residual fit receipt",
            ),
        )
        for path, digest, label in pairs:
            if (path is None) != (digest is None):
                raise ValueError(f"{label} path and SHA256 must be declared together")
        return self


class LocalLLMSemanticRankerConfig(_RequiredConfig):
    component: Literal["bounded_local_llm_semantic_ranker"] = (
        "bounded_local_llm_semantic_ranker"
    )
    backend: Literal[
        "qwen_causal_relevance",
        "local_causal_relevance",
        "minilm_cross_encoder_control",
    ] = "qwen_causal_relevance"
    model_id: str = Field(default="qwen_causal_relevance", min_length=1)
    model_path: str = "artifacts/cache/models/qwen2.5-0.5b-instruct"
    model_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    model_sha256: str = (
        "31b07963d699962dbbc9fdcb9d4cfaa496f5e56abc29c8e597e18195b87ebe77"
    )
    fallback_model_path: str = "artifacts/cache/models/ms-marco-MiniLM-L6-v2"
    fallback_model_revision: str = "c5ee24cb16019beea089334287d17c69e06eb577"
    fallback_cache_dir: str = "artifacts/cache/cross_encoder"
    rerank_k: int = Field(default=10, ge=2, le=50)
    weight: float = Field(default=0.35, gt=0.0, le=1.0)
    fallback_weight: float = Field(default=0.5, gt=0.0, le=1.0)
    batch_size: int = Field(default=8, ge=1, le=32)
    max_length: int = Field(default=256, ge=128, le=1024)
    timeout_ms: int = Field(default=30000, ge=1, le=120000)
    device: Literal["auto", "cpu", "mps"] = "auto"
    activation_policy: Literal["browsing_only"] = "browsing_only"
    activate_for_browsing: Literal[True] = True


class ProactiveGuidanceConfig(_RequiredConfig):
    component: Literal["over_generality_guidance"] = "over_generality_guidance"
    overload_min_candidates: int = Field(default=180, ge=2, le=1000)
    overload_max_specific_constraints: int = Field(default=0, ge=0, le=8)
    question_candidate_k: int = Field(default=100, ge=10, le=500)
    question_value_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    broad_discovery_turns: int = Field(default=2, ge=0, le=4)
    max_question_turn: int = Field(default=8, ge=1, le=9)
    preview_keyword_k: int = Field(default=40, ge=5, le=200)
    preview_category_k: int = Field(default=30, ge=5, le=200)
    preview_min_candidates: int = Field(default=30, ge=2, le=400)
    preview_score_flatness: float = Field(default=0.65, ge=0.0, le=1.0)


class RuntimeAdaptationConfig(_RequiredConfig):
    component: Literal["conflict_safe_context_distillation"] = (
        "conflict_safe_context_distillation"
    )
    profile_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    profile_weight: float = Field(default=0.08, ge=0.0, le=0.5)
    maximum_explicit_constraints_for_profile: int = Field(default=0, ge=0, le=8)
    profile_question_suppression_enabled: bool = False
    union_profile_feature_enabled: bool = False


class AdaptiveOrchestrationConfig(_RequiredConfig):
    component: Literal["fixed_adaptive_coordinator"] = "fixed_adaptive_coordinator"
    max_recommendations: Literal[10] = 10
    atomic_commit: Literal[True] = True
    emit_reason_codes: Literal[True] = True
    fallback: Literal["complete_precision_path"] = "complete_precision_path"


class AdaptiveHybridConfig(BaseModel):
    """Submission-eligible configuration for the immutable 1A-3B workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    architecture: Literal["adaptive_hybrid_1a_3b_v1"] = "adaptive_hybrid_1a_3b_v1"
    policy_id: str = Field(default="adaptive_hybrid_1a_3b_v1", min_length=1)
    submission_eligible: Literal[True] = True
    state: StateV2Config = Field(default_factory=StateV2Config)
    router: DualTrackRouterConfig = Field(default_factory=DualTrackRouterConfig)
    buying: PrecisionTrackConfig = Field(default_factory=PrecisionTrackConfig)
    browsing: DiverseDenseTrackConfig = Field(default_factory=DiverseDenseTrackConfig)
    merger: MultiRouteMergeConfig = Field(default_factory=MultiRouteMergeConfig)
    union_ranker: UnionRankerConfig = Field(default_factory=UnionRankerConfig)
    semantic_ranker: LocalLLMSemanticRankerConfig = Field(
        default_factory=LocalLLMSemanticRankerConfig
    )
    guidance: ProactiveGuidanceConfig = Field(default_factory=ProactiveGuidanceConfig)
    runtime_adaptation: RuntimeAdaptationConfig = Field(
        default_factory=RuntimeAdaptationConfig
    )
    orchestration: AdaptiveOrchestrationConfig = Field(
        default_factory=AdaptiveOrchestrationConfig
    )
    extensions: AdaptiveExtensionsConfig = Field(
        default_factory=AdaptiveExtensionsConfig
    )

    @field_validator(
        "state",
        "router",
        "buying",
        "browsing",
        "merger",
        "union_ranker",
        "semantic_ranker",
        "guidance",
        "runtime_adaptation",
        "orchestration",
        "extensions",
    )
    @classmethod
    def paths_are_confined(cls, value: BaseModel) -> BaseModel:
        for key, raw in value.model_dump().items():
            if not key.endswith("_path") or raw is None:
                continue
            path = PurePath(str(raw))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{key} must stay inside the project root")
        return value

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "AdaptiveExtensionsConfig",
    "AdaptiveHybridConfig",
    "AdaptiveOrchestrationConfig",
    "DiverseDenseTrackConfig",
    "DualTrackRouterConfig",
    "LocalLLMSemanticRankerConfig",
    "MultiRouteMergeConfig",
    "PrecisionTrackConfig",
    "ProactiveGuidanceConfig",
    "RuntimeAdaptationConfig",
    "StateV2Config",
    "UnionRankerConfig",
]
