from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig


class AdaptiveHybridTrial(BaseModel):
    """GhostLab-searchable values that cannot alter the required topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    buying_min_specific_constraints: int = Field(default=1, ge=1, le=4)
    router_abstain_confidence: float = Field(default=0.6, ge=0.5, le=0.9)
    router_specificity_threshold: float = Field(default=1.0, ge=0.0, le=4.0)
    buying_retrieval_k: int = Field(default=200, ge=50, le=500)
    dense_retrieval_per_view: int = Field(default=400, ge=100, le=800)
    dense_output_k: int = Field(default=200, ge=50, le=400)
    dense_mmr_relevance_weight: float = Field(default=0.85, ge=0.5, le=1.0)
    overload_dense_retrieval_per_view: int = Field(default=80, ge=5, le=400)
    overload_dense_output_k: int = Field(default=80, ge=5, le=400)
    category_k: int = Field(default=120, ge=20, le=300)
    merged_k: int = Field(default=320, ge=100, le=600)
    buying_vector_support_k: int = Field(default=40, ge=5, le=100)
    browsing_keyword_support_k: int = Field(default=40, ge=5, le=100)
    buying_keyword_share: float = Field(default=0.9, ge=0.5, le=0.95)
    browsing_vector_share: float = Field(default=0.8, ge=0.5, le=0.95)
    merger_rrf_constant: int = Field(default=60, ge=1, le=200)
    union_rerank_k: int = Field(default=320, ge=10, le=1000)
    buying_residual_weight: float = Field(default=0.25, ge=0.0, le=0.49)
    semantic_weight: float = Field(default=0.5, gt=0.0, le=0.75)
    semantic_rerank_k: int = Field(default=10, ge=5, le=50)
    semantic_fallback_weight: float = Field(default=0.5, gt=0.0, le=0.75)
    overload_min_candidates: int = Field(default=180, ge=50, le=400)
    preview_min_candidates: int = Field(default=30, ge=2, le=200)
    question_value_margin: float = Field(default=0.0, ge=0.0, le=0.25)
    broad_discovery_turns: int = Field(default=2, ge=0, le=4)
    profile_weight: float = Field(default=0.02, gt=0.0, le=0.25)
    profile_max_explicit_constraints: int = Field(default=2, ge=0, le=4)
    quality_prior_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_rerank_k: int = Field(default=50, ge=10, le=500)
    query_prf_feedback_k: int = Field(default=5, ge=2, le=50)
    query_prf_minimum_support: float = Field(default=0.4, gt=0.0, le=1.0)
    query_prf_max_terms: int = Field(default=4, ge=0, le=20)
    query_prf_max_added_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    facet_relevance_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    facet_rerank_k: int = Field(default=30, ge=10, le=200)
    facet_output_k: int = Field(default=10, ge=2, le=50)
    facet_max_turn: int = Field(default=2, ge=1, le=9)
    facet_max_constraints: int = Field(default=1, ge=0, le=10)

    @classmethod
    def from_config(cls, config: AdaptiveHybridConfig) -> AdaptiveHybridTrial:
        return cls(
            buying_min_specific_constraints=(
                config.router.buying_min_specific_constraints
            ),
            router_abstain_confidence=config.router.abstain_confidence,
            router_specificity_threshold=(
                config.router.buying_specificity_threshold
            ),
            buying_retrieval_k=config.buying.retrieval_k,
            dense_retrieval_per_view=config.browsing.retrieval_per_view,
            dense_output_k=config.browsing.output_k,
            dense_mmr_relevance_weight=config.browsing.mmr_relevance_weight,
            overload_dense_retrieval_per_view=(
                config.browsing.overload_retrieval_per_view
            ),
            overload_dense_output_k=config.browsing.overload_output_k,
            category_k=config.merger.category_k,
            merged_k=config.merger.merged_k,
            buying_vector_support_k=config.merger.buying_vector_support_k,
            browsing_keyword_support_k=(config.merger.browsing_keyword_support_k),
            buying_keyword_share=config.merger.buying_keyword_weight,
            browsing_vector_share=config.merger.browsing_vector_weight,
            merger_rrf_constant=config.merger.rrf_constant,
            union_rerank_k=config.union_ranker.rerank_k,
            buying_residual_weight=config.union_ranker.buying_residual_weight,
            semantic_weight=config.semantic_ranker.weight,
            semantic_rerank_k=config.semantic_ranker.rerank_k,
            semantic_fallback_weight=config.semantic_ranker.fallback_weight,
            overload_min_candidates=config.guidance.overload_min_candidates,
            preview_min_candidates=config.guidance.preview_min_candidates,
            question_value_margin=config.guidance.question_value_margin,
            broad_discovery_turns=config.guidance.broad_discovery_turns,
            profile_weight=config.runtime_adaptation.profile_weight,
            profile_max_explicit_constraints=(
                config.runtime_adaptation.maximum_explicit_constraints_for_profile
            ),
            quality_prior_weight=config.extensions.quality_prior_weight,
            quality_rerank_k=config.extensions.quality_rerank_k,
            query_prf_feedback_k=config.extensions.query_prf_feedback_k,
            query_prf_minimum_support=(config.extensions.query_prf_minimum_support),
            query_prf_max_terms=config.extensions.query_prf_max_terms,
            query_prf_max_added_ratio=(config.extensions.query_prf_max_added_ratio),
            facet_relevance_weight=config.extensions.facet_relevance_weight,
            facet_rerank_k=config.extensions.facet_rerank_k,
            facet_output_k=config.extensions.facet_output_k,
            facet_max_turn=config.extensions.facet_max_turn,
            facet_max_constraints=config.extensions.facet_max_constraints,
        )


class AdaptiveArchitectureAudit:
    """Reject trials that make any mandatory 1A-3B capability decorative."""

    REQUIRED_COMPONENTS = (
        "state_v2",
        "buying_browsing_router",
        "field_bm25_precision",
        "diverse_e5_multiview",
        "keyword_category_vector_merge",
        "union_aware_ranker",
        "bounded_local_llm_semantic_ranker",
        "over_generality_guidance",
        "conflict_safe_context_distillation",
        "fixed_adaptive_coordinator",
    )

    @classmethod
    def validate(cls, config: AdaptiveHybridConfig) -> AdaptiveHybridConfig:
        actual = (
            config.state.component,
            config.router.component,
            config.buying.component,
            config.browsing.component,
            config.merger.component,
            config.union_ranker.component,
            config.semantic_ranker.component,
            config.guidance.component,
            config.runtime_adaptation.component,
            config.orchestration.component,
        )
        if actual != cls.REQUIRED_COMPONENTS:
            raise ValueError("adaptive trial changed the required 1A-3B topology")
        if config.semantic_ranker.backend not in {
            "qwen_causal_relevance",
            "local_causal_relevance",
        }:
            raise ValueError("submission trial requires a literal local LLM")
        if not config.semantic_ranker.activate_for_browsing:
            raise ValueError("adaptive trial disabled the semantic LLM capability")
        if (
            min(
                config.merger.buying_keyword_weight,
                config.merger.buying_category_weight,
                config.merger.buying_vector_weight,
                config.merger.browsing_keyword_weight,
                config.merger.browsing_category_weight,
                config.merger.browsing_vector_weight,
            )
            <= 0.0
        ):
            raise ValueError("all three retrieval evidence sources must remain active")
        if not config.orchestration.atomic_commit:
            raise ValueError("adaptive trial disabled atomic commit")
        if config.runtime_adaptation.profile_weight <= 0.0:
            raise ValueError("adaptive trial made profile context decorative")
        return config


def generate_adaptive_trials(
    baseline: AdaptiveHybridConfig, max_trials: int
) -> tuple[AdaptiveHybridTrial, ...]:
    """Generate a small deterministic search around the compliant baseline."""

    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    anchor = AdaptiveHybridTrial.from_config(baseline)
    variations: tuple[dict[str, object], ...] = (
        {},
        {"semantic_weight": 0.35, "semantic_rerank_k": 20},
        {
            "buying_vector_support_k": 20,
            "browsing_keyword_support_k": 20,
        },
        {
            "buying_vector_support_k": 60,
            "browsing_keyword_support_k": 60,
        },
        {"overload_min_candidates": 140, "question_value_margin": 0.02},
        {
            "router_abstain_confidence": 0.7,
            "buying_min_specific_constraints": 2,
        },
        {"profile_weight": 0.05, "profile_max_explicit_constraints": 1},
    )
    return tuple(
        anchor.model_copy(update=updates) for updates in variations[:max_trials]
    )


class AdaptiveHybridBinding:
    """Materialize GhostLab trials only inside mandatory 1A-3B slots."""

    @staticmethod
    def materialize(
        baseline: AdaptiveHybridConfig,
        parameters: AdaptiveHybridTrial | Mapping[str, object],
        *,
        policy_id: str,
    ) -> AdaptiveHybridConfig:
        trial = (
            parameters
            if isinstance(parameters, AdaptiveHybridTrial)
            else AdaptiveHybridTrial.model_validate(parameters)
        )
        router = baseline.router.model_copy(
            update={
                "buying_min_specific_constraints": (
                    trial.buying_min_specific_constraints
                ),
                "abstain_confidence": trial.router_abstain_confidence,
                "buying_specificity_threshold": (
                    trial.router_specificity_threshold
                ),
            }
        )
        buying = baseline.buying.model_copy(
            update={"retrieval_k": trial.buying_retrieval_k}
        )
        browsing = baseline.browsing.model_copy(
            update={
                "retrieval_per_view": trial.dense_retrieval_per_view,
                "output_k": trial.dense_output_k,
                "mmr_relevance_weight": trial.dense_mmr_relevance_weight,
                "overload_retrieval_per_view": min(
                    trial.overload_dense_retrieval_per_view,
                    trial.dense_retrieval_per_view,
                ),
                "overload_output_k": min(
                    trial.overload_dense_output_k, trial.dense_output_k
                ),
            }
        )
        buying_support_share = (1.0 - trial.buying_keyword_share) / 2.0
        browsing_support_share = (1.0 - trial.browsing_vector_share) / 2.0
        merger = baseline.merger.model_copy(
            update={
                "category_k": trial.category_k,
                "merged_k": trial.merged_k,
                "buying_vector_support_k": trial.buying_vector_support_k,
                "browsing_keyword_support_k": trial.browsing_keyword_support_k,
                "buying_keyword_weight": trial.buying_keyword_share,
                "buying_category_weight": buying_support_share,
                "buying_vector_weight": buying_support_share,
                "browsing_vector_weight": trial.browsing_vector_share,
                "browsing_category_weight": browsing_support_share,
                "browsing_keyword_weight": browsing_support_share,
                "rrf_constant": trial.merger_rrf_constant,
            }
        )
        union_ranker = baseline.union_ranker.model_copy(
            update={
                "rerank_k": trial.union_rerank_k,
                "buying_residual_weight": trial.buying_residual_weight,
            }
        )
        semantic = baseline.semantic_ranker.model_copy(
            update={
                "weight": trial.semantic_weight,
                "rerank_k": trial.semantic_rerank_k,
                "fallback_weight": trial.semantic_fallback_weight,
            }
        )
        guidance = baseline.guidance.model_copy(
            update={
                "overload_min_candidates": trial.overload_min_candidates,
                "preview_min_candidates": trial.preview_min_candidates,
                "question_value_margin": trial.question_value_margin,
                "broad_discovery_turns": trial.broad_discovery_turns,
            }
        )
        adaptation = baseline.runtime_adaptation.model_copy(
            update={
                "profile_weight": trial.profile_weight,
                "maximum_explicit_constraints_for_profile": (
                    trial.profile_max_explicit_constraints
                ),
            }
        )
        extensions = baseline.extensions.model_copy(
            update={
                "quality_prior_weight": trial.quality_prior_weight,
                "quality_rerank_k": trial.quality_rerank_k,
                "query_prf_feedback_k": trial.query_prf_feedback_k,
                "query_prf_minimum_support": trial.query_prf_minimum_support,
                "query_prf_max_terms": trial.query_prf_max_terms,
                "query_prf_max_added_ratio": trial.query_prf_max_added_ratio,
                "facet_relevance_weight": trial.facet_relevance_weight,
                "facet_rerank_k": trial.facet_rerank_k,
                "facet_output_k": trial.facet_output_k,
                "facet_max_turn": trial.facet_max_turn,
                "facet_max_constraints": trial.facet_max_constraints,
            }
        )
        value = baseline.model_copy(
            update={
                "policy_id": policy_id,
                "router": router,
                "buying": buying,
                "browsing": browsing,
                "merger": merger,
                "union_ranker": union_ranker,
                "semantic_ranker": semantic,
                "guidance": guidance,
                "runtime_adaptation": adaptation,
                "extensions": extensions,
            }
        )
        candidate = AdaptiveHybridConfig.model_validate(value.model_dump())
        return AdaptiveArchitectureAudit.validate(candidate)


__all__ = [
    "AdaptiveArchitectureAudit",
    "AdaptiveHybridBinding",
    "AdaptiveHybridTrial",
    "generate_adaptive_trials",
]
