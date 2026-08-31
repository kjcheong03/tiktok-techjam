from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_hybrid import (
    AdaptiveArchitectureAudit,
    AdaptiveHybridBinding,
    AdaptiveHybridTrial,
)
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig

AdaptiveRole = Literal[
    "compulsory",
    "promotable",
    "control_only",
    "research_only",
    "unavailable",
]
AdaptiveControlClass = Literal["direct", "absorbed_dependency", "replacement_only"]
AdaptiveStage = Literal[
    "state",
    "routing",
    "retrieval",
    "merge",
    "union_ranking",
    "semantic_ranking",
    "guidance",
    "adaptation",
    "orchestration",
    "offline",
]


class AdaptiveTechniquePatch(BaseModel):
    """Typed nested updates applied before ordinary parameter materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    updates: tuple[tuple[str, object], ...] = ()


class AdaptiveTechniqueBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    technique_id: str
    role: AdaptiveRole
    stage: AdaptiveStage
    reason: str
    patch: AdaptiveTechniquePatch | None = None
    fit_required: bool = False
    override_priority: int = 0
    control_class: AdaptiveControlClass = "direct"
    absorbed_by: str | None = None

    @property
    def selectable(self) -> bool:
        return self.role == "promotable"

    @property
    def absorbed(self) -> bool:
        return self.control_class == "absorbed_dependency"

    @model_validator(mode="after")
    def control_detail_is_consistent(self) -> AdaptiveTechniqueBinding:
        if self.control_class != "direct" and self.role != "control_only":
            raise ValueError(
                "only control-only techniques have control classifications"
            )
        if (self.control_class == "absorbed_dependency") != (
            self.absorbed_by is not None
        ):
            raise ValueError("absorbed dependencies must name exactly one parent")
        return self


@dataclass(frozen=True)
class AdaptiveTechniqueInventory:
    total: int
    compulsory: tuple[str, ...]
    promotable: tuple[str, ...]
    control_only: tuple[str, ...]
    research_only: tuple[str, ...]
    unavailable: tuple[str, ...]


_COMPULSORY: dict[str, tuple[AdaptiveStage, str]] = {
    "state.baseline_v2": (
        "state",
        "State V2 accumulation, correction and override implementation",
    ),
    "query.coverage_adaptive_v2": (
        "state",
        "State V2 query projection used by every required retrieval track",
    ),
    "retrieval.sparse": (
        "retrieval",
        "required Buying precision generator and bounded Browsing support",
    ),
    "retrieval.e5": (
        "retrieval",
        "required diverse dense Browsing generator and bounded Buying support",
    ),
    "filter.structured": (
        "union_ranking",
        "required route-independent hard-constraint authority",
    ),
    "ranking.cross_encoder": (
        "semantic_ranking",
        "required deterministic fallback behind the bounded local LLM slot",
    ),
    "question.candidate_eig.v1": (
        "guidance",
        "required highest-value unresolved-attribute guidance implementation",
    ),
    "termination.reward_aware.v1": (
        "guidance",
        "required positive-value question stopping rule",
    ),
    "prior.profile": (
        "adaptation",
        "required conflict-safe supplied-profile influence",
    ),
    "recommendation.correction_scoped_history": (
        "orchestration",
        "required intent-epoch recommendation history",
    ),
    "guard.override_fallback": (
        "orchestration",
        "required complete-precision fallback on adaptive component failure",
    ),
    "routing.dual_track_observable.v1": (
        "routing",
        "required observable Buying/Browsing route decision",
    ),
    "retrieval.category_independent.v1": (
        "retrieval",
        "required independent category candidate route",
    ),
    "fusion.multi_route_union.v1": (
        "merge",
        "required keyword/category/vector union with provenance",
    ),
    "ranking.source_aware_union.v1": (
        "union_ranking",
        "required source-aware union ranking feature contract",
    ),
    "ranking.local_llm_semantic.v1": (
        "semantic_ranking",
        "required bounded local-LLM semantic ranking slot",
    ),
    "guidance.pre_dense_overload.v1": (
        "guidance",
        "required pre-dense overload cutoff and question action",
    ),
    "adaptation.profile_update.v1": (
        "adaptation",
        "required conflict-safe profile update with confidence and provenance",
    ),
    "orchestration.atomic_commit.v1": (
        "orchestration",
        "required validated selected-action-only atomic commit",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_for_family(family: str) -> AdaptiveStage:
    return {
        "state": "state",
        "routing": "routing",
        "retrieval": "retrieval",
        "fusion": "merge",
        "ranking": "union_ranking",
        "filter": "union_ranking",
        "question": "guidance",
        "termination": "guidance",
        "prior": "adaptation",
        "recommendation": "orchestration",
        "policy": "orchestration",
        "guard": "orchestration",
    }.get(family, "offline")  # type: ignore[return-value]


def _patch(*updates: tuple[str, object]) -> AdaptiveTechniquePatch:
    return AdaptiveTechniquePatch(updates=updates)


def _promotable_patches(project_root: Path) -> dict[str, AdaptiveTechniquePatch]:
    def auxiliary_learned(
        technique_id: str, backend: Literal["gbdt", "rank_ensemble"], relative: str
    ) -> AdaptiveTechniquePatch:
        path = project_root / relative
        if not path.is_file():
            return _patch()
        return _patch(
            ("union_ranker.auxiliary_technique_id", technique_id),
            ("union_ranker.auxiliary_backend", backend),
            ("union_ranker.auxiliary_model_path", relative),
            ("union_ranker.auxiliary_model_sha256", _sha256(path)),
            ("union_ranker.auxiliary_weight", 0.1),
            ("union_ranker.auxiliary_rerank_k", 50),
        )

    ontology = project_root / "artifacts/assets/catalog_ontology_v1.json"
    ontology_patch = (
        _patch(
            ("state.catalog_normalizer_enabled", True),
            (
                "state.catalog_ontology_path",
                "artifacts/assets/catalog_ontology_v1.json",
            ),
            ("state.catalog_ontology_sha256", _sha256(ontology)),
            ("state.constraint_normalization_confidence", 0.9),
        )
        if ontology.is_file()
        else _patch()
    )

    return {
        # The control already uses weighted fusion, so this optional seed must
        # change the evidence allocation as well as name the strategy. HPO can
        # then tune the route shares from this deliberately less extreme seed.
        "fusion.weighted": _patch(
            ("merger.strategy", "weighted"),
            ("merger.buying_keyword_weight", 0.8),
            ("merger.buying_category_weight", 0.1),
            ("merger.buying_vector_weight", 0.1),
            ("merger.browsing_keyword_weight", 0.15),
            ("merger.browsing_category_weight", 0.15),
            ("merger.browsing_vector_weight", 0.7),
        ),
        "fusion.rrf": _patch(("merger.strategy", "rrf")),
        "fusion.sparse_first_union": _patch(("merger.strategy", "sparse_first_union")),
        "prior.quality": _patch(("extensions.quality_prior_weight", 0.2)),
        "query.catalog_prf.v1": _patch(("extensions.query_prf_enabled", True)),
        "state.catalog_normalizer.v1": ontology_patch,
        "retrieval.minilm": _patch(
            ("extensions.minilm_dense_view_enabled", True),
            (
                "extensions.minilm_dense_model_path",
                "artifacts/cache/models/all-MiniLM-L6-v2",
            ),
            ("extensions.minilm_dense_retrieval_k", 80),
            ("extensions.minilm_dense_weight", 0.15),
        ),
        "ranking.facet_diversity.v1": _patch(
            ("extensions.facet_diversity_enabled", True)
        ),
        # Planning switch only. The candidate technique ID dispatches a fresh
        # fold-local fit during evaluation; no historical residual asset or new
        # AdaptiveHybridConfig field is materialized here.
        "ranking.top10_residual_reranker.v2": _patch(),
        "retrieval.dense_view_balanced.v1": _patch(
            ("browsing.selection", "view_balanced")
        ),
        "retrieval.dense_embedding_mmr.v1": _patch(
            ("browsing.selection", "embedding_mmr")
        ),
        "prior.profile_query_view.v1": _patch(
            ("browsing.profile_query_view_enabled", True)
        ),
        "prior.profile_question_suppression.v1": _patch(
            ("runtime_adaptation.profile_question_suppression_enabled", True)
        ),
        "prior.profile_union_feature.v1": _patch(
            ("runtime_adaptation.union_profile_feature_enabled", True)
        ),
        "ranking.fixed_lexical": _patch(
            ("union_ranker.auxiliary_technique_id", "ranking.fixed_lexical"),
            ("union_ranker.auxiliary_backend", "fixed_lexical"),
            ("union_ranker.auxiliary_weight", 0.1),
            ("union_ranker.auxiliary_rerank_k", 50),
        ),
        "ranking.metadata_gbdt": auxiliary_learned(
            "ranking.metadata_gbdt",
            "gbdt",
            "artifacts/models/gbdt_reranker_v2_round56.json",
        ),
        "ranking.reward_lambdamart.v1": auxiliary_learned(
            "ranking.reward_lambdamart.v1",
            "gbdt",
            "artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json",
        ),
        "ranking.turn_aware_lambdamart.v1": auxiliary_learned(
            "ranking.turn_aware_lambdamart.v1",
            "gbdt",
            "artifacts/models/w2_ranking_v1/turn_aware_lambdamart_v1.json",
        ),
        "ranking.fold_ensemble.v1": auxiliary_learned(
            "ranking.fold_ensemble.v1",
            "rank_ensemble",
            "artifacts/models/w2_ranking_v1/fold_ensemble.json",
        ),
        "fusion.rank_stack.v1": auxiliary_learned(
            "fusion.rank_stack.v1",
            "rank_ensemble",
            "artifacts/models/w2_ranking_v1/rank_stack.json",
        ),
    }


_ABSORBED_CONTROLS: dict[str, tuple[str, str]] = {
    "query.expansion_guard.v1": (
        "query.catalog_prf.v1",
        "the promotable catalog-PRF hook already applies its bounded overload and drift guard",
    ),
    "ranking.mmr_early.v1": (
        "ranking.facet_diversity.v1",
        "the promotable facet-diversity hook owns bounded early-turn MMR activation",
    ),
    "state.attribute_ontology.v1": (
        "state.catalog_normalizer.v1",
        "the pinned ontology is the intrinsic asset dependency of catalog normalization",
    ),
    "state.confidence_gated_constraints.v1": (
        "state.catalog_normalizer.v1",
        "the promotable normalizer applies the confidence threshold before changing State V2 constraints",
    ),
}

_REPLACEMENT_ONLY_CONTROLS: dict[str, str] = {
    "policy.joint_observable.v1": (
        "joint action selection would replace the fixed router/guidance coordination contract"
    ),
    "query.structured": (
        "it replaces the compulsory coverage-adaptive State V2 query projection"
    ),
    "question.adaptive_heuristic": (
        "it replaces the compulsory candidate-EIG guidance implementation"
    ),
    "question.fixed": (
        "it replaces the compulsory candidate-EIG guidance implementation"
    ),
    "question.learned_linear": (
        "it replaces candidate-EIG and also requires a new fold-fitted policy asset"
    ),
    "question.other_always": (
        "it replaces candidate-EIG with an unconditional fixed question policy"
    ),
    "ranking.constraint_gbdt": (
        "it is a historical whole-ranker anchor that replaces source-aware union ranking"
    ),
    "ranking.deep_dense_gbdt": (
        "it is a historical whole-ranker anchor that replaces source-aware union ranking"
    ),
    "ranking.neural_gbdt": (
        "it is a historical whole-ranker anchor that replaces source-aware union ranking"
    ),
    "ranking.pairwise_linear": (
        "it is a historical whole-ranker anchor that replaces source-aware union ranking"
    ),
    "routing.decision_list": (
        "it replaces the compulsory observable Buying/Browsing dual-track router"
    ),
    "routing.joint_route.v1": (
        "joint routing replaces the compulsory observable dual-track router"
    ),
    "routing.observable_stump": (
        "it is a historical routing anchor that replaces the dual-track router"
    ),
    "routing.route_table": (
        "it is a historical routing anchor that replaces the dual-track router"
    ),
    "state.compressed": "it replaces compulsory State V2 memory semantics",
    "state.current": "it replaces compulsory State V2 memory semantics",
    "state.multi": "it replaces compulsory State V2 memory semantics",
    "state.raw_history": "it replaces compulsory State V2 memory semantics",
}


def _adaptive_catalog_view(catalog: TechniqueCatalog) -> TechniqueCatalog:
    """Rebind legacy replacement metadata for fixed-architecture additive hooks.

    The source catalog records MiniLM and E5 as alternative dense backends. In the
    adaptive runtime E5 remains compulsory and MiniLM contributes only an auxiliary
    bounded view, so their historical exclusive-group relationship no longer applies.
    The source content hash is retained for evidence lineage.
    """

    techniques = dict(catalog.techniques)
    minilm = techniques.get("retrieval.minilm")
    if minilm is not None:
        techniques["retrieval.minilm"] = minilm.model_copy(
            update={
                "config_binding": "extensions.minilm_dense_view_enabled=true",
                "exclusive_group": None,
            }
        )
    return TechniqueCatalog(catalog.schema_version, techniques, catalog.content_hash)


class AdaptiveTechniqueRegistry:
    """Exhaustive bridge from the historical catalog to fixed 1A-3B slots.

    The registry intentionally keeps non-promotable controls visible. A technique
    disappears from neither the catalog nor the evidence ledger merely because it
    cannot legally alter the submission architecture.
    """

    def __init__(
        self,
        catalog: TechniqueCatalog,
        bindings: Iterable[AdaptiveTechniqueBinding],
    ) -> None:
        self.catalog = _adaptive_catalog_view(catalog)
        self.bindings: dict[str, AdaptiveTechniqueBinding] = {}
        for binding in bindings:
            if binding.technique_id in self.bindings:
                raise ValueError(
                    f"duplicate adaptive technique binding: {binding.technique_id}"
                )
            if (binding.role == "promotable") != (binding.patch is not None):
                raise ValueError(
                    f"{binding.technique_id}: only promotable bindings have patches"
                )
            self.bindings[binding.technique_id] = binding
        missing = set(catalog.techniques) - set(self.bindings)
        extra = set(self.bindings) - set(catalog.techniques)
        if missing or extra:
            raise ValueError(
                "adaptive registry must be exhaustive; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

    @classmethod
    def from_catalog(
        cls, catalog: TechniqueCatalog, *, project_root: str | Path
    ) -> AdaptiveTechniqueRegistry:
        root = Path(project_root).resolve()
        legacy = default_binding_registry().bindings
        promotable = _promotable_patches(root)
        bindings: list[AdaptiveTechniqueBinding] = []
        for technique_id, technique in sorted(catalog.techniques.items()):
            stage = _stage_for_family(technique.family)
            control_class: AdaptiveControlClass = "direct"
            absorbed_by: str | None = None
            if technique_id in _COMPULSORY:
                stage, reason = _COMPULSORY[technique_id]
                role: AdaptiveRole = "compulsory"
                patch = None
            elif technique_id in promotable and (
                promotable[technique_id].updates
                or technique_id == "ranking.top10_residual_reranker.v2"
            ):
                role = "promotable"
                patch = promotable[technique_id]
                reason = (
                    "fresh fold-fitted additive dispatched by technique ID"
                    if technique_id == "ranking.top10_residual_reranker.v2"
                    else "architecture-safe implementation or additive hook"
                )
            else:
                patch = None
                old = legacy.get(technique_id)
                if not technique.executable or (
                    old is not None and old.disposition == "unavailable"
                ):
                    role = "unavailable"
                    reason = (
                        old.reason if old is not None else "catalog asset unavailable"
                    )
                elif technique.execution_mode == "research_only" or (
                    old is not None and old.disposition == "research_only"
                ):
                    role = "research_only"
                    reason = "offline search/evaluation procedure, not a runtime switch"
                else:
                    role = "control_only"
                    absorbed = _ABSORBED_CONTROLS.get(technique_id)
                    replacement = _REPLACEMENT_ONLY_CONTROLS.get(technique_id)
                    if absorbed is not None:
                        control_class = "absorbed_dependency"
                        absorbed_by, detail = absorbed
                        reason = f"absorbed by {absorbed_by}: {detail}"
                    elif replacement is not None:
                        control_class = "replacement_only"
                        reason = f"replacement-only control: {replacement}"
                    else:
                        reason = (
                            "retained for ablation or adaptation research; its current "
                            "binding cannot be promoted without replacing/bypassing a "
                            "required 1A-3B capability"
                        )
            bindings.append(
                AdaptiveTechniqueBinding(
                    technique_id=technique_id,
                    role=role,
                    stage=stage,
                    reason=reason,
                    patch=patch,
                    fit_required=technique.fit_required,
                    control_class=control_class,
                    absorbed_by=absorbed_by,
                    override_priority=(
                        20
                        if technique_id == "fusion.rank_stack.v1"
                        else 10
                        if technique_id == "ranking.fold_ensemble.v1"
                        else 0
                    ),
                )
            )
        return cls(catalog, bindings)

    def inventory(self) -> AdaptiveTechniqueInventory:
        grouped: dict[AdaptiveRole, list[str]] = {
            "compulsory": [],
            "promotable": [],
            "control_only": [],
            "research_only": [],
            "unavailable": [],
        }
        for technique_id, binding in sorted(self.bindings.items()):
            grouped[binding.role].append(technique_id)
        return AdaptiveTechniqueInventory(
            total=len(self.bindings),
            compulsory=tuple(grouped["compulsory"]),
            promotable=tuple(grouped["promotable"]),
            control_only=tuple(grouped["control_only"]),
            research_only=tuple(grouped["research_only"]),
            unavailable=tuple(grouped["unavailable"]),
        )

    def materialize(
        self,
        baseline: AdaptiveHybridConfig,
        candidate: CandidateSpec,
        *,
        policy_id: str | None = None,
    ) -> AdaptiveHybridConfig:
        value = baseline.model_dump(mode="python")
        proposals: dict[str, list[tuple[AdaptiveTechniqueBinding, object]]] = {}
        for technique_id in candidate.techniques:
            binding = self.bindings.get(technique_id)
            if binding is None:
                raise ValueError(f"unknown adaptive technique: {technique_id}")
            if binding.role == "compulsory":
                continue
            if not binding.selectable or binding.patch is None:
                raise ValueError(f"{technique_id} is {binding.role}: {binding.reason}")
            for path, update in binding.patch.updates:
                proposals.setdefault(path, []).append((binding, update))
        for path, updates in proposals.items():
            distinct = {repr(update) for _, update in updates}
            if len(distinct) != 1:
                highest = max(binding.override_priority for binding, _ in updates)
                winners = [
                    update
                    for binding, update in updates
                    if binding.override_priority == highest
                ]
                if highest > 0 and len(winners) == 1:
                    self._set_path(value, path, winners[0])
                    continue
                rendered = ", ".join(
                    f"{binding.technique_id}={update!r}" for binding, update in updates
                )
                raise ValueError(f"conflicting adaptive patches for {path}: {rendered}")
            self._set_path(value, path, updates[0][1])
        patched = AdaptiveHybridConfig.model_validate(value)
        trial = AdaptiveHybridTrial.from_config(patched)
        parameter_updates = dict(candidate.parameters)
        unknown = set(parameter_updates) - set(AdaptiveHybridTrial.model_fields)
        if unknown:
            raise ValueError(f"unknown adaptive parameters: {sorted(unknown)}")
        trial = trial.model_copy(update=parameter_updates)
        result = AdaptiveHybridBinding.materialize(
            patched,
            trial,
            policy_id=policy_id or candidate.candidate_id,
        )
        return AdaptiveArchitectureAudit.validate(result)

    @staticmethod
    def _set_path(target: dict[str, object], path: str, update: object) -> None:
        parts = path.split(".")
        node: dict[str, object] = target
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                raise TypeError(f"adaptive patch path is not an object: {path}")
            node = child
        node[parts[-1]] = update

    def validate_candidate(self, candidate: CandidateSpec) -> None:
        selected = set(candidate.techniques)
        for technique_id in selected:
            binding = self.bindings.get(technique_id)
            if binding is None:
                raise ValueError(f"unknown adaptive technique: {technique_id}")
            if binding.role not in {"compulsory", "promotable"}:
                raise ValueError(
                    f"{technique_id} is not submission-promotable ({binding.role})"
                )
        additions = selected - set(self.inventory().compulsory)
        union_implementations = selected & {
            "ranking.fixed_lexical",
            "ranking.metadata_gbdt",
            "ranking.reward_lambdamart.v1",
            "ranking.turn_aware_lambdamart.v1",
            "ranking.fold_ensemble.v1",
        }
        if "fusion.rank_stack.v1" in selected:
            illegal = union_implementations - {"ranking.fold_ensemble.v1"}
            if illegal:
                raise ValueError(
                    "rank stack cannot be combined with another union-ranker "
                    f"implementation: {sorted(illegal)}"
                )
        elif len(union_implementations) > 1:
            raise ValueError(
                "union-ranker implementations are mutually exclusive: "
                f"{sorted(union_implementations)}"
            )
        catalog_result = self._catalog_compatibility(tuple(selected))
        if catalog_result:
            raise ValueError("; ".join(catalog_result))
        if candidate.complexity != len(additions):
            raise ValueError("candidate complexity must count optional additions")

    def _catalog_compatibility(self, techniques: tuple[str, ...]) -> tuple[str, ...]:
        from ghostlab.campaign.compatibility import validate_techniques

        return validate_techniques(self.catalog, techniques).reasons

    def parameter_names_for(self, technique_ids: tuple[str, ...]) -> frozenset[str]:
        selected = set(technique_ids)
        names = set(AdaptiveHybridTrial.model_fields) - {
            "quality_prior_weight",
            "quality_rerank_k",
            "facet_relevance_weight",
            "facet_rerank_k",
            "facet_output_k",
            "facet_max_turn",
            "facet_max_constraints",
            "merger_rrf_constant",
            "query_prf_feedback_k",
            "query_prf_minimum_support",
            "query_prf_max_terms",
            "query_prf_max_added_ratio",
            "dense_mmr_relevance_weight",
            "union_auxiliary_weight",
        }
        if "prior.quality" in selected:
            names.update(("quality_prior_weight", "quality_rerank_k"))
        if "ranking.facet_diversity.v1" in selected:
            names.update(
                (
                    "facet_relevance_weight",
                    "facet_rerank_k",
                    "facet_output_k",
                    "facet_max_turn",
                    "facet_max_constraints",
                )
            )
        if "fusion.rrf" in selected:
            names.add("merger_rrf_constant")
        if "query.catalog_prf.v1" in selected:
            names.update(
                (
                    "query_prf_feedback_k",
                    "query_prf_minimum_support",
                    "query_prf_max_terms",
                    "query_prf_max_added_ratio",
                )
            )
        if "retrieval.dense_embedding_mmr.v1" in selected:
            names.add("dense_mmr_relevance_weight")
        if selected & {
            "ranking.fixed_lexical",
            "ranking.metadata_gbdt",
            "ranking.reward_lambdamart.v1",
            "ranking.turn_aware_lambdamart.v1",
            "ranking.fold_ensemble.v1",
            "fusion.rank_stack.v1",
        }:
            names.add("union_auxiliary_weight")
        return frozenset(names)


__all__ = [
    "AdaptiveControlClass",
    "AdaptiveRole",
    "AdaptiveStage",
    "AdaptiveTechniqueBinding",
    "AdaptiveTechniqueInventory",
    "AdaptiveTechniquePatch",
    "AdaptiveTechniqueRegistry",
]
