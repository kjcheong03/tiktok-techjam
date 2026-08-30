from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

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

    @property
    def selectable(self) -> bool:
        return self.role == "promotable"


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
    def learned(
        backend: Literal["gbdt", "rank_ensemble"], relative: str
    ) -> AdaptiveTechniquePatch:
        path = project_root / relative
        if not path.is_file():
            return _patch()
        return _patch(
            ("union_ranker.backend", backend),
            ("union_ranker.model_path", relative),
            ("union_ranker.model_sha256", _sha256(path)),
        )

    return {
        "fusion.weighted": _patch(("merger.strategy", "weighted")),
        "fusion.rrf": _patch(("merger.strategy", "rrf")),
        "fusion.sparse_first_union": _patch(("merger.strategy", "sparse_first_union")),
        "prior.quality": _patch(("extensions.quality_prior_weight", 0.2)),
        "query.catalog_prf.v1": _patch(("extensions.query_prf_enabled", True)),
        "ranking.facet_diversity.v1": _patch(
            ("extensions.facet_diversity_enabled", True)
        ),
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
            ("union_ranker.backend", "deterministic"),
            ("union_ranker.model_path", None),
            ("union_ranker.model_sha256", None),
        ),
        "ranking.metadata_gbdt": learned(
            "gbdt", "artifacts/models/gbdt_reranker_v2_round56.json"
        ),
        "ranking.reward_lambdamart.v1": learned(
            "gbdt",
            "artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json",
        ),
        "ranking.turn_aware_lambdamart.v1": learned(
            "gbdt",
            "artifacts/models/w2_ranking_v1/turn_aware_lambdamart_v1.json",
        ),
        "ranking.fold_ensemble.v1": learned(
            "rank_ensemble",
            "artifacts/models/w2_ranking_v1/fold_ensemble.json",
        ),
        "fusion.rank_stack.v1": learned(
            "rank_ensemble", "artifacts/models/w2_ranking_v1/rank_stack.json"
        ),
    }


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
        self.catalog = catalog
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
            if technique_id in _COMPULSORY:
                stage, reason = _COMPULSORY[technique_id]
                role: AdaptiveRole = "compulsory"
                patch = None
            elif technique_id in promotable and promotable[technique_id].updates:
                role = "promotable"
                patch = promotable[technique_id]
                reason = "architecture-safe implementation or additive hook"
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
        return frozenset(names)


__all__ = [
    "AdaptiveRole",
    "AdaptiveStage",
    "AdaptiveTechniqueBinding",
    "AdaptiveTechniqueInventory",
    "AdaptiveTechniquePatch",
    "AdaptiveTechniqueRegistry",
]
