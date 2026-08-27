from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ghostlab.campaign.models import CandidateSpec
from ghostlab.competition.contract import AskAttribute
from ghostlab.research.technique_suite import (
    DenseBackend,
    Diversification,
    Normalizer,
    QueryExpansion,
    QueryVariant,
    QuestionVariant,
    Reranker,
    RetrievalRoute,
    RoutingVariant,
    StateVariant,
    UnifiedTechniqueConfig,
    load_suite_config,
)

BindingDisposition = Literal[
    "composable", "anchor_only", "research_only", "unavailable"
]
ASSET_FIELDS = frozenset(
    {
        "compiled_config_path",
        "learned_question_asset",
        "joint_policy_asset",
        "normalizer_asset",
        "dense_model_path",
        "learned_sparse_asset",
        "late_interaction_asset",
        "reranker_model_asset",
        "cross_encoder_model_path",
        "router_asset",
    }
)


class TechniquePatch(BaseModel):
    """Typed partial update for the integration suite configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: Literal["compiled", "experimental"] | None = None
    compiled_config_path: str | None = None
    state_variant: StateVariant | None = None
    query_variant: QueryVariant | None = None
    question_variant: QuestionVariant | None = None
    question_order: tuple[AskAttribute, ...] | None = None
    learned_question_asset: str | None = None
    eig_candidate_k: int | None = None
    question_value_margin: float | None = None
    joint_policy_asset: str | None = None
    normalizer: Normalizer | None = None
    normalizer_asset: str | None = None
    constraint_confidence: float | None = None
    retrieval_route: RetrievalRoute | None = None
    dense_backend: DenseBackend | None = None
    dense_model_path: str | None = None
    sparse_weight: float | None = None
    dense_weight: float | None = None
    learned_sparse_asset: str | None = None
    late_interaction_asset: str | None = None
    query_expansion: QueryExpansion | None = None
    structured_filter: bool | None = None
    profile_prior_weight: float | None = None
    quality_prior_weight: float | None = None
    reranker: Reranker | None = None
    reranker_model_asset: str | None = None
    cross_encoder_enabled: bool | None = None
    cross_encoder_model_path: str | None = None
    cross_encoder_weight: float | None = None
    cross_encoder_rerank_k: int | None = None
    diversification: Diversification | None = None
    routing_variant: RoutingVariant | None = None
    router_asset: str | None = None
    component_fallback: bool | None = None

    @field_validator(*ASSET_FIELDS)
    @classmethod
    def safe_asset_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("binding asset paths must stay inside the project")
        return value

    def updates(self) -> dict[str, object]:
        return self.model_dump(exclude_unset=True)


class TechniqueBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    technique_id: str
    disposition: BindingDisposition
    patch: TechniquePatch | None = None
    reason: str
    requires: tuple[str, ...] = ()

    @property
    def asset_paths(self) -> tuple[str, ...]:
        if self.patch is None:
            return ()
        updates = self.patch.updates()
        return tuple(
            str(updates[field])
            for field in sorted(ASSET_FIELDS & updates.keys())
            if updates[field] is not None
        )


class BindingConflictError(ValueError):
    pass


class TechniqueBindingRegistry:
    def __init__(self, bindings: Iterable[TechniqueBinding]) -> None:
        self.bindings: dict[str, TechniqueBinding] = {}
        for binding in bindings:
            if binding.technique_id in self.bindings:
                raise ValueError(f"duplicate technique binding: {binding.technique_id}")
            if (binding.disposition == "composable") != (binding.patch is not None):
                raise ValueError(
                    f"{binding.technique_id}: only composable bindings have patches"
                )
            self.bindings[binding.technique_id] = binding

    def materialize(
        self,
        baseline: UnifiedTechniqueConfig,
        candidate: CandidateSpec,
    ) -> UnifiedTechniqueConfig:
        value = baseline.model_dump(mode="json")
        owners: dict[str, tuple[str, object]] = {}
        selected = set(candidate.techniques)
        for technique_id in candidate.techniques:
            binding = self.bindings.get(technique_id)
            if binding is None:
                raise ValueError(f"unknown technique binding: {technique_id}")
            if binding.disposition != "composable":
                raise ValueError(
                    f"{technique_id} is {binding.disposition}: {binding.reason}"
                )
            missing = set(binding.requires) - selected
            if missing:
                raise ValueError(f"{technique_id} requires {sorted(missing)}")
            assert binding.patch is not None
            for field, update in binding.patch.updates().items():
                previous = owners.get(field)
                if previous is not None and previous[1] != update:
                    raise BindingConflictError(
                        f"conflicting patches for {field}: "
                        f"{previous[0]}={previous[1]!r}, {technique_id}={update!r}"
                    )
                owners[field] = (technique_id, update)
                value[field] = update
        parameters = dict(candidate.parameters)
        unknown = set(parameters) - set(UnifiedTechniqueConfig.model_fields)
        if unknown:
            raise ValueError(f"unknown candidate parameters: {sorted(unknown)}")
        value.update(parameters)
        value["experiment_id"] = candidate.candidate_id
        return UnifiedTechniqueConfig.model_validate(value)

    def materialize_from_suite(
        self, suite_path: str | Path, candidate: CandidateSpec
    ) -> UnifiedTechniqueConfig:
        return self.materialize(load_suite_config(suite_path), candidate)


def _composable(
    technique_id: str,
    patch: TechniquePatch,
    *,
    reason: str,
    requires: tuple[str, ...] = (),
) -> TechniqueBinding:
    return TechniqueBinding(
        technique_id=technique_id,
        disposition="composable",
        patch=patch,
        reason=reason,
        requires=requires,
    )


def _classified(
    technique_id: str, disposition: BindingDisposition, reason: str
) -> TechniqueBinding:
    return TechniqueBinding(
        technique_id=technique_id,
        disposition=disposition,
        reason=reason,
    )


def default_binding_registry() -> TechniqueBindingRegistry:
    """Return exhaustive bindings for the current Wave 1/2 catalogs.

    An entry is composable only when ``UnifiedTechniqueConfig`` and the current
    local assets can construct it. Historical anchors and research/search
    procedures remain visible but cannot accidentally become runtime candidates.
    """

    learned_question = (
        "artifacts/experiments/learned_question_linear_v1/"
        "linear_action_value_model.json"
    )
    metadata_model = "artifacts/models/gbdt_reranker_v2_round56.json"
    bindings = [
        _composable("state.current", TechniquePatch(state_variant="current", query_variant=None), reason="current-turn state"),
        _composable("state.raw_history", TechniquePatch(state_variant="raw_history"), reason="raw history state"),
        _composable("state.multi", TechniquePatch(state_variant="multi"), reason="multi-value conversation state"),
        _composable("state.compressed", TechniquePatch(state_variant="compressed"), reason="compressed state"),
        _composable("query.structured", TechniquePatch(query_variant="structured_active"), reason="structured active query"),
        _composable("question.fixed", TechniquePatch(question_variant="fixed", question_order=(), learned_question_asset=None, joint_policy_asset=None), reason="fixed question control"),
        _composable("question.adaptive_heuristic", TechniquePatch(question_variant="adaptive", question_order=(), learned_question_asset=None, joint_policy_asset=None), reason="observable heuristic policy"),
        _composable("question.learned_linear", TechniquePatch(question_variant="learned", question_order=(), learned_question_asset=learned_question, joint_policy_asset=None), reason="local compiled linear question asset"),
        _composable("retrieval.sparse", TechniquePatch(retrieval_route="keyword", dense_backend="off", dense_model_path=None, learned_sparse_asset=None, late_interaction_asset=None), reason="core sparse retrieval"),
        _composable("ranking.fixed_lexical", TechniquePatch(reranker="linear", reranker_model_asset=None), reason="fixed lexical reranker"),
        _composable("ranking.metadata_gbdt", TechniquePatch(reranker="metadata_gbdt", reranker_model_asset=metadata_model), reason="local metadata GBDT asset"),
        _composable("filter.structured", TechniquePatch(structured_filter=True), reason="coverage-aware structured filter"),
        _composable("prior.profile", TechniquePatch(profile_prior_weight=0.1), reason="profile prior with tunable starting weight"),
        _composable("prior.quality", TechniquePatch(quality_prior_weight=0.2), reason="catalog quality prior"),
        _composable("state.catalog_normalizer.v1", TechniquePatch(normalizer="catalog_v1", normalizer_asset="artifacts/assets/catalog_ontology_v1.json"), reason="local catalog ontology asset"),
        _classified("state.attribute_ontology.v1", "anchor_only", "asset-producing dependency of catalog normalization"),
        _composable("state.confidence_gated_constraints.v1", TechniquePatch(constraint_confidence=0.9), reason="normalization confidence gate", requires=("state.catalog_normalizer.v1",)),
        _composable("question.candidate_eig.v1", TechniquePatch(question_variant="candidate_eig", question_order=(), learned_question_asset=None, joint_policy_asset=None), reason="candidate-statistics EIG policy"),
        _classified("question.reward_voi.v1", "unavailable", "UnifiedTechniqueConfig has no fold-fitted reward-VOI calibration asset binding"),
        _composable("termination.reward_aware.v1", TechniquePatch(question_value_margin=0.02), reason="explicit question/stop margin", requires=("question.candidate_eig.v1",)),
        _composable("policy.joint_observable.v1", TechniquePatch(question_variant="joint_observable", question_order=(), learned_question_asset=None, joint_policy_asset="configs/assets/joint_policy_control_v1.json"), reason="bounded local joint decision-list asset"),
        _classified("routing.joint_route.v1", "anchor_only", "routing behavior is inseparable from the selected joint policy asset"),
        _classified("research.counterfactual_expert.v2", "research_only", "offline label generator, not a runtime switch"),
        _classified("policy.distilled_expert.v1", "unavailable", "no fold-fitted distilled runtime asset is present"),
        _classified("search.expert_iteration.v1", "research_only", "offline dataset aggregation procedure"),
        _composable("ranking.reward_lambdamart.v1", TechniquePatch(reranker="reward_lambdamart", reranker_model_asset="artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json"), reason="local reward-aligned ranker asset"),
        _composable("ranking.turn_aware_lambdamart.v1", TechniquePatch(reranker="turn_aware_lambdamart", reranker_model_asset="artifacts/models/w2_ranking_v1/turn_aware_lambdamart_v1.json"), reason="local turn-aware ranker asset"),
        _composable("ranking.fold_ensemble.v1", TechniquePatch(reranker="rank_ensemble", reranker_model_asset="artifacts/models/w2_ranking_v1/fold_ensemble.json"), reason="local fold-ensemble asset"),
        _composable("fusion.rank_stack.v1", TechniquePatch(reranker="rank_ensemble", reranker_model_asset="artifacts/models/w2_ranking_v1/rank_stack.json"), reason="local rank-stack asset"),
        _composable("query.catalog_prf.v1", TechniquePatch(query_expansion="prf"), reason="core catalog PRF"),
        _classified("query.expansion_guard.v1", "anchor_only", "guard is intrinsic to catalog PRF and has no independent runtime toggle"),
        _composable("ranking.facet_diversity.v1", TechniquePatch(diversification="facet_mmr"), reason="core facet MMR"),
        _classified("ranking.mmr_early.v1", "anchor_only", "early-turn gate is intrinsic to facet MMR in the current config"),
    ]

    unavailable = {
        "retrieval.minilm": "local MiniLM model asset is absent",
        "retrieval.e5": "local E5 model asset is absent",
        "fusion.rrf": "dense model asset required by this route is absent",
        "fusion.weighted": "dense model asset required by this route is absent",
        "fusion.sparse_first_union": "dense model asset required by this route is absent",
        "ranking.cross_encoder": "local cross-encoder model asset is absent",
        "retrieval.splade_rescue.v1": "learned-sparse model/index asset is unavailable",
        "fusion.sparse_semantic_union.v1": "learned-sparse dependency is unavailable",
        "retrieval.late_interaction_rescue.v1": "late-interaction feasibility asset is unavailable",
        "retrieval.colbert_rescue.v1": "ColBERT feasibility gate did not produce an asset",
        "retrieval.bge_m3_rescue.v1": "BGE-M3 feasibility gate did not produce an asset",
        "fusion.late_interaction_union.v1": "late-interaction dependency is unavailable",
        "query.query2doc_local.v1": "optional local generation model was not admitted",
        "routing.calibrated_observable.v1": "no fold-fitted router asset is present",
        "guard.component_fallback.v1": "requires an unavailable calibrated router asset",
    }
    bindings.extend(
        _classified(technique_id, "unavailable", reason)
        for technique_id, reason in unavailable.items()
    )
    anchor_only = {
        "ranking.pairwise_linear": "historical anchor is not represented by the unified reranker enum",
        "ranking.constraint_gbdt": "selected compiled suite anchor; not an additive experimental patch",
        "ranking.deep_dense_gbdt": "historical standalone challenger without a unified binding",
        "ranking.neural_gbdt": "historical standalone challenger without a unified binding",
        "guard.override_fallback": "part of the compiled guarded champion anchor",
        "routing.decision_list": "supporting policy mechanism, selected through a joint policy asset",
        "routing.observable_stump": "historical route-policy anchor without a runtime asset binding",
        "routing.route_table": "historical route-policy anchor without a runtime asset binding",
    }
    bindings.extend(
        _classified(technique_id, "anchor_only", reason)
        for technique_id, reason in anchor_only.items()
    )
    research_only = {
        "research.counterfactual",
        "research.replay",
        "research.leakage_firewall",
        "search.random_grid_beam",
        "search.multifidelity_racing",
        "search.evidence_allocator",
        "search.family_ucb",
        "search.typed_patches",
        "search.crossover",
        "search.hyperband.v1",
        "search.bohb.v1",
        "evidence.decision_store",
        "evaluation.grouped_splits",
        "evaluation.paired_statistics",
    }
    bindings.extend(
        _classified(
            technique_id,
            "research_only",
            "campaign/evaluation procedure, not a runtime configuration patch",
        )
        for technique_id in sorted(research_only)
    )
    return TechniqueBindingRegistry(bindings)
