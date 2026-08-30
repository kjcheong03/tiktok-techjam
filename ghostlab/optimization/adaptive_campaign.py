from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ghostlab.campaign.interaction_search import (
    CandidateEvidence,
    InteractionSearchPlan,
    SearchLimits,
    plan_higher_order_round,
    plan_standalones_and_pairs,
)
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_hybrid import AdaptiveHybridTrial
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.optimization.bohb import Observation
from ghostlab.optimization.conditional import (
    ConditionalParameter,
    ConditionalSearchSpace,
    TuningContext,
    suggest_for_combination,
)
from ghostlab.optimization.racing import Decision, racing_decide
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig

Fidelity = Literal["f0", "f1", "f2"]


@dataclass(frozen=True)
class AdaptiveEvaluation:
    candidate_id: str
    fidelity: Fidelity
    score: float
    session_rewards: tuple[float, ...]
    behavior_novelty: float = 0.0
    latency_p95_ms: float = 0.0
    fit_verified: bool = False
    gate_metrics: tuple[tuple[str, float], ...] = ()
    constraint_violations: int = 0
    hit_rate_at_10: float = 0.0
    mrr: float = 0.0
    mttc: float = 0.0
    lineage_cluster_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_rewards:
            raise ValueError("adaptive evaluations require session-level rewards")
        if not 0.0 <= self.behavior_novelty <= 1.0:
            raise ValueError("behavior novelty must be in [0, 1]")
        if self.lineage_cluster_ids and len(self.lineage_cluster_ids) != len(
            self.session_rewards
        ):
            raise ValueError("lineage clusters must align with session rewards")


@dataclass(frozen=True)
class AdaptiveRaceRecord:
    candidate: CandidateSpec
    evaluation: AdaptiveEvaluation
    paired_deltas: tuple[float, ...]
    decision: Decision
    fit_required: tuple[str, ...]
    gate_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptiveCampaignResult:
    incumbent: CandidateSpec
    selected: CandidateSpec
    promoted: bool
    stages: dict[Fidelity, tuple[AdaptiveRaceRecord, ...]]
    architecture_rejections: tuple[tuple[str, str], ...]
    search_rounds: tuple[InteractionSearchPlan, ...]


Evaluator = Callable[
    [AdaptiveHybridConfig, CandidateSpec, Fidelity], AdaptiveEvaluation
]


def _candidate_hash(
    baseline_id: str,
    techniques: tuple[str, ...],
    parameters: tuple[tuple[str, str | int | float | bool], ...] = (),
) -> str:
    encoded = repr((baseline_id, sorted(techniques), sorted(parameters))).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _adaptive_search_space() -> ConditionalSearchSpace:
    always: tuple[ConditionalParameter, ...] = (
        ConditionalParameter(
            name="buying_min_specific_constraints", kind="int", low=1, high=4
        ),
        ConditionalParameter(
            name="router_abstain_confidence", kind="float", low=0.5, high=0.9
        ),
        ConditionalParameter(
            name="router_specificity_threshold", kind="float", low=0.0, high=4.0
        ),
        ConditionalParameter(
            name="buying_retrieval_k",
            kind="categorical",
            choices=(50, 100, 200, 300, 500),
        ),
        ConditionalParameter(
            name="dense_retrieval_per_view",
            kind="categorical",
            choices=(100, 200, 400, 600, 800),
        ),
        ConditionalParameter(
            name="dense_output_k", kind="categorical", choices=(50, 100, 200, 300, 400)
        ),
        ConditionalParameter(
            name="overload_dense_retrieval_per_view",
            kind="categorical",
            choices=(20, 40, 80, 120, 200),
        ),
        ConditionalParameter(
            name="overload_dense_output_k",
            kind="categorical",
            choices=(20, 40, 80, 120, 200),
        ),
        ConditionalParameter(
            name="category_k", kind="categorical", choices=(20, 60, 120, 200, 300)
        ),
        ConditionalParameter(
            name="merged_k", kind="categorical", choices=(100, 200, 320, 450, 600)
        ),
        ConditionalParameter(
            name="buying_vector_support_k", kind="int", low=5, high=100
        ),
        ConditionalParameter(
            name="browsing_keyword_support_k", kind="int", low=5, high=100
        ),
        ConditionalParameter(
            name="buying_keyword_share", kind="float", low=0.5, high=0.95
        ),
        ConditionalParameter(
            name="browsing_vector_share", kind="float", low=0.5, high=0.95
        ),
        ConditionalParameter(
            name="union_rerank_k", kind="categorical", choices=(50, 100, 200, 320, 500)
        ),
        ConditionalParameter(
            name="buying_residual_weight", kind="float", low=0.0, high=0.49
        ),
        ConditionalParameter(name="semantic_weight", kind="float", low=0.05, high=0.75),
        ConditionalParameter(
            name="semantic_rerank_k", kind="categorical", choices=(5, 10, 20, 30, 50)
        ),
        ConditionalParameter(
            name="semantic_fallback_weight", kind="float", low=0.05, high=0.75
        ),
        ConditionalParameter(
            name="overload_min_candidates", kind="int", low=50, high=400
        ),
        ConditionalParameter(
            name="preview_min_candidates", kind="int", low=10, high=200
        ),
        ConditionalParameter(
            name="question_value_margin", kind="float", low=0.0, high=0.25
        ),
        ConditionalParameter(name="broad_discovery_turns", kind="int", low=0, high=4),
        ConditionalParameter(
            name="profile_weight", kind="float", low=0.01, high=0.25, scale="log"
        ),
        ConditionalParameter(
            name="profile_max_explicit_constraints", kind="int", low=0, high=4
        ),
    )
    optional: tuple[ConditionalParameter, ...] = (
        ConditionalParameter(
            name="dense_mmr_relevance_weight",
            kind="float",
            low=0.5,
            high=1.0,
            requires_all=("retrieval.dense_embedding_mmr.v1",),
        ),
        ConditionalParameter(
            name="merger_rrf_constant",
            kind="int",
            low=1,
            high=200,
            scale="log",
            requires_all=("fusion.rrf",),
        ),
        ConditionalParameter(
            name="quality_prior_weight",
            kind="float",
            low=0.01,
            high=1.0,
            scale="log",
            requires_all=("prior.quality",),
        ),
        ConditionalParameter(
            name="quality_rerank_k",
            kind="categorical",
            choices=(10, 20, 50, 100, 200),
            requires_all=("prior.quality",),
        ),
        ConditionalParameter(
            name="query_prf_feedback_k",
            kind="int",
            low=2,
            high=50,
            requires_all=("query.catalog_prf.v1",),
        ),
        ConditionalParameter(
            name="query_prf_minimum_support",
            kind="float",
            low=0.05,
            high=1.0,
            requires_all=("query.catalog_prf.v1",),
        ),
        ConditionalParameter(
            name="query_prf_max_terms",
            kind="int",
            low=0,
            high=20,
            requires_all=("query.catalog_prf.v1",),
        ),
        ConditionalParameter(
            name="query_prf_max_added_ratio",
            kind="float",
            low=0.0,
            high=1.0,
            requires_all=("query.catalog_prf.v1",),
        ),
        ConditionalParameter(
            name="facet_relevance_weight",
            kind="float",
            low=0.25,
            high=1.0,
            requires_all=("ranking.facet_diversity.v1",),
        ),
        ConditionalParameter(
            name="facet_rerank_k",
            kind="categorical",
            choices=(10, 20, 30, 50, 100),
            requires_all=("ranking.facet_diversity.v1",),
        ),
        ConditionalParameter(
            name="facet_output_k",
            kind="categorical",
            choices=(5, 10, 20, 30),
            requires_all=("ranking.facet_diversity.v1",),
        ),
        ConditionalParameter(
            name="facet_max_turn",
            kind="int",
            low=1,
            high=9,
            requires_all=("ranking.facet_diversity.v1",),
        ),
        ConditionalParameter(
            name="facet_max_constraints",
            kind="int",
            low=0,
            high=10,
            requires_all=("ranking.facet_diversity.v1",),
        ),
    )
    return ConditionalSearchSpace(schema_version=2, parameters=(*always, *optional))


class AdaptiveGhostLabEngine:
    """Champion/challenger racing constrained by the immutable 1A-3B contract."""

    def __init__(
        self,
        *,
        baseline: AdaptiveHybridConfig,
        registry: AdaptiveTechniqueRegistry,
        candidate_limit: int = 500,
        beam_width: int = 24,
        exploration_fraction: float = 0.2,
        max_extra_techniques: int | None = None,
        seed: int = 20260826,
    ) -> None:
        inventory = registry.inventory()
        maximum = len(inventory.promotable)
        if max_extra_techniques is not None and max_extra_techniques <= 0:
            raise ValueError("max_extra_techniques must be positive or None")
        self.baseline = baseline
        self.registry = registry
        self.candidate_limit = candidate_limit
        self.seed = seed
        self.search_space = _adaptive_search_space()
        self.limits = SearchLimits(
            max_order=min(maximum, max_extra_techniques or maximum) or 1,
            max_candidates=candidate_limit,
            beam_width=beam_width,
            exploration_fraction=exploration_fraction,
            seed=seed,
        )
        mandatory = inventory.compulsory
        self.incumbent = CandidateSpec(
            candidate_id=f"control-{baseline.policy_id}",
            baseline_id=baseline.policy_id,
            techniques=mandatory,
            complexity=0,
            generation="control",
        )

    def initial_plan(self) -> InteractionSearchPlan:
        inventory = self.registry.inventory()
        plan = plan_standalones_and_pairs(
            self.registry.catalog,
            baseline_id=self.baseline.policy_id,
            baseline_techniques=inventory.compulsory,
            technique_ids=inventory.promotable,
            limits=self.limits,
        )
        accepted: list[CandidateSpec] = []
        skipped = list(plan.skipped)
        for candidate in plan.candidates:
            try:
                self.registry.validate_candidate(candidate)
                self.registry.materialize(self.baseline, candidate)
            except Exception as error:  # noqa: BLE001 - preflight evidence boundary
                from ghostlab.campaign.planner import SkippedCandidate

                skipped.append(
                    SkippedCandidate(
                        candidate.baseline_id,
                        tuple(
                            item
                            for item in candidate.techniques
                            if item not in inventory.compulsory
                        ),
                        (f"adaptive_preflight:{type(error).__name__}:{error}",),
                    )
                )
            else:
                accepted.append(candidate)
        return InteractionSearchPlan(
            candidates=tuple(accepted),
            skipped=tuple(skipped),
            cap_exhausted=plan.cap_exhausted,
        )

    def materialize(self, candidate: CandidateSpec) -> AdaptiveHybridConfig:
        self.registry.validate_candidate(candidate)
        return self.registry.materialize(self.baseline, candidate)

    def hpo_candidate(
        self,
        candidate: CandidateSpec,
        parameters: tuple[tuple[str, str | int | float | bool], ...],
        *,
        ordinal: int,
    ) -> CandidateSpec:
        allowed = self.registry.parameter_names_for(candidate.techniques)
        unknown = {name for name, _ in parameters} - allowed
        if unknown:
            raise ValueError(
                f"parameters are inactive for this technique stack: {sorted(unknown)}"
            )
        result = candidate.model_copy(
            update={
                "candidate_id": (
                    f"{candidate.candidate_id}-hpo-{ordinal:03d}-"
                    f"{_candidate_hash(candidate.baseline_id, candidate.techniques, parameters)}"
                ),
                "parameters": tuple(sorted(parameters)),
            }
        )
        self.materialize(result)
        return result

    def suggest_hpo_candidates(
        self,
        candidate: CandidateSpec,
        *,
        count: int,
        observations: tuple[Observation, ...] = (),
    ) -> tuple[CandidateSpec, ...]:
        """Generate architecture-safe local BOHB trials for one structure."""

        if count < 0:
            raise ValueError("HPO trial count cannot be negative")
        if count == 0:
            return ()
        config = self.materialize(candidate)
        center = tuple(
            sorted(AdaptiveHybridTrial.from_config(config).model_dump().items())
        )
        results: list[CandidateSpec] = []
        seen = {candidate.canonical_hash()}
        attempts = 0
        while len(results) < count and attempts < count * 8:
            attempts += 1
            suggestion = suggest_for_combination(
                self.search_space,
                candidate.techniques,
                observations,
                context=TuningContext(outer_fold=0, inner_fold=0),
                seed=self.seed + attempts,
                center=center,
                max_changes=3,
                trust_region=0.2,
                block_index=attempts - 1,
            )
            parameters = dict(candidate.parameters)
            parameters.update(suggestion)
            try:
                trial = self.hpo_candidate(
                    candidate,
                    tuple(sorted(parameters.items())),
                    ordinal=len(results) + 1,
                )
            except ValueError:
                continue
            key = trial.canonical_hash()
            if key not in seen:
                seen.add(key)
                results.append(trial)
        return tuple(results)

    def higher_order_plan(
        self,
        evaluated: tuple[CandidateSpec, ...],
        records: dict[str, AdaptiveRaceRecord],
        *,
        consumed_wall_seconds: float,
        estimated_candidate_seconds: float,
    ) -> InteractionSearchPlan:
        control = records.get(self.incumbent.candidate_id)
        evidence: dict[str, CandidateEvidence] = {}
        for candidate in evaluated:
            record = records.get(candidate.candidate_id)
            if record is None:
                continue
            deltas = record.paired_deltas
            mean = statistics.fmean(deltas)
            spread = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            radius = 1.96 * spread / max(1.0, len(deltas) ** 0.5)
            evidence[candidate.candidate_id] = CandidateEvidence(
                candidate_id=candidate.candidate_id,
                mean_delta=mean,
                confidence_lower=mean - radius,
                confidence_upper=mean + radius,
                repeated_evaluations=1,
                invalid_reason=("matched control missing" if control is None else None),
            )
        plan = plan_higher_order_round(
            self.registry.catalog,
            evaluated_candidates=evaluated,
            evidence=evidence,
            technique_ids=self.registry.inventory().promotable,
            baseline_techniques=self.registry.inventory().compulsory,
            limits=self.limits,
            consumed_wall_seconds=consumed_wall_seconds,
            estimated_candidate_seconds=estimated_candidate_seconds,
        )
        accepted: list[CandidateSpec] = []
        skipped = list(plan.skipped)
        for candidate in plan.candidates:
            try:
                self.materialize(candidate)
            except Exception as error:  # noqa: BLE001 - preflight evidence boundary
                from ghostlab.campaign.planner import SkippedCandidate

                skipped.append(
                    SkippedCandidate(
                        candidate.baseline_id,
                        tuple(
                            item
                            for item in candidate.techniques
                            if item not in self.registry.inventory().compulsory
                        ),
                        (f"adaptive_preflight:{type(error).__name__}:{error}",),
                    )
                )
            else:
                accepted.append(candidate)
        return InteractionSearchPlan(
            candidates=tuple(accepted),
            skipped=tuple(skipped),
            reserve_candidate_ids=plan.reserve_candidate_ids,
            exploration_candidate_ids=plan.exploration_candidate_ids,
            permanently_pruned=plan.permanently_pruned,
            pruning_audit=plan.pruning_audit,
            resurrected=plan.resurrected,
            cap_exhausted=plan.cap_exhausted,
            wall_exhausted=plan.wall_exhausted,
        )

    def run(
        self,
        evaluator: Evaluator,
        *,
        f1_candidates: int = 24,
        f2_candidates: int = 6,
        higher_order_rounds: int = 8,
        hpo_trials_per_structure: int = 0,
    ) -> AdaptiveCampaignResult:
        if f1_candidates <= 0 or f2_candidates <= 0:
            raise ValueError("fidelity candidate limits must be positive")
        plan = self.initial_plan()
        architecture_rejections = tuple(
            (
                "+".join(item.roots) or "control",
                "; ".join(item.reasons),
            )
            for item in plan.skipped
        )
        f0 = self._evaluate_stage(plan.candidates, "f0", evaluator)
        search_rounds: list[InteractionSearchPlan] = [plan]
        all_candidates = list(plan.candidates)
        f0_by_id = {item.candidate.candidate_id: item for item in f0}
        for _ in range(higher_order_rounds):
            expansion = self.higher_order_plan(
                tuple(all_candidates),
                f0_by_id,
                consumed_wall_seconds=0.0,
                estimated_candidate_seconds=1.0,
            )
            search_rounds.append(expansion)
            if not expansion.candidates:
                break
            new_records = self._evaluate_stage(
                expansion.candidates, "f0", evaluator, control=f0[0]
            )
            f0 = (*f0, *new_records)
            all_candidates.extend(expansion.candidates)
            f0_by_id.update({item.candidate.candidate_id: item for item in new_records})
        f1_roots = self._survivors(f0, f1_candidates)
        hpo: list[CandidateSpec] = []
        if hpo_trials_per_structure < 0:
            raise ValueError("HPO trial count cannot be negative")
        f0_scores = {item.candidate.candidate_id: item.evaluation.score for item in f0}
        for root in f1_roots:
            if root.generation == "control":
                continue
            observations = (Observation(root.parameters, f0_scores[root.candidate_id]),)
            hpo.extend(
                self.suggest_hpo_candidates(
                    root,
                    count=hpo_trials_per_structure,
                    observations=observations,
                )
            )
        f1 = self._evaluate_stage((*f1_roots, *hpo), "f1", evaluator)
        f2_roots = self._survivors(f1, f2_candidates)
        f2 = self._evaluate_stage(f2_roots, "f2", evaluator)
        eligible = [
            item
            for item in f2
            if item.candidate.generation != "control"
            and item.decision == "PROMOTE"
            and (not item.fit_required or item.evaluation.fit_verified)
        ]
        winner = min(
            eligible,
            key=lambda item: (
                -item.evaluation.score,
                item.candidate.complexity,
                item.candidate.candidate_id,
            ),
            default=None,
        )
        selected = winner.candidate if winner is not None else self.incumbent
        return AdaptiveCampaignResult(
            incumbent=self.incumbent,
            selected=selected,
            promoted=winner is not None,
            stages={"f0": tuple(f0), "f1": tuple(f1), "f2": tuple(f2)},
            architecture_rejections=architecture_rejections,
            search_rounds=tuple(search_rounds),
        )

    def _evaluate_stage(
        self,
        candidates: tuple[CandidateSpec, ...],
        fidelity: Fidelity,
        evaluator: Evaluator,
        *,
        control: AdaptiveRaceRecord | None = None,
    ) -> tuple[AdaptiveRaceRecord, ...]:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda item: (item.generation != "control", item.candidate_id),
            )
        )
        if not ordered:
            raise ValueError("adaptive race stage cannot be empty")
        evaluations: list[tuple[CandidateSpec, AdaptiveEvaluation]] = []
        for candidate in ordered:
            config = self.materialize(candidate)
            evaluation = evaluator(config, candidate, fidelity)
            if evaluation.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a mismatched candidate ID")
            if evaluation.fidelity != fidelity:
                raise ValueError("evaluator returned a mismatched fidelity")
            evaluations.append((candidate, evaluation))
        matched_control = control
        if matched_control is None:
            control_pair = next(
                (pair for pair in evaluations if pair[0].generation == "control"),
                None,
            )
            if control_pair is None:
                baseline_evaluation = evaluator(
                    self.materialize(self.incumbent), self.incumbent, fidelity
                )
                control_pair = (self.incumbent, baseline_evaluation)
            matched_control = AdaptiveRaceRecord(
                candidate=control_pair[0],
                evaluation=control_pair[1],
                paired_deltas=tuple(0.0 for _ in control_pair[1].session_rewards),
                decision="HOLD_MORE_DATA",
                fit_required=(),
                gate_failures=(),
            )
        control_rewards = matched_control.evaluation.session_rewards
        records: list[AdaptiveRaceRecord] = []
        for candidate, evaluation in evaluations:
            if len(evaluation.session_rewards) != len(control_rewards):
                raise ValueError("candidate/control session rewards are not paired")
            deltas = tuple(
                candidate_reward - control_reward
                for candidate_reward, control_reward in zip(
                    evaluation.session_rewards, control_rewards, strict=True
                )
            )
            decision: Decision = (
                "HOLD_MORE_DATA"
                if candidate.generation == "control"
                else racing_decide(
                    list(deltas),
                    fidelity=fidelity,
                    behavior_novelty=evaluation.behavior_novelty,
                    seed=self.seed,
                    cluster_ids=(
                        evaluation.lineage_cluster_ids
                        if evaluation.lineage_cluster_ids
                        else None
                    ),
                )
            )
            gate_failures: list[str] = []
            if evaluation.constraint_violations:
                gate_failures.append(
                    f"constraint_violations:{evaluation.constraint_violations}"
                )
            control_metrics = dict(matched_control.evaluation.gate_metrics)
            for name, value in evaluation.gate_metrics:
                if name in control_metrics and value < control_metrics[name] - 0.02:
                    gate_failures.append(
                        f"non_regression:{name}:{value:.6f}<"
                        f"{control_metrics[name]:.6f}-0.02"
                    )
            if candidate.generation != "control" and gate_failures:
                decision = "REJECT"
            fit_required = tuple(
                technique_id
                for technique_id in candidate.techniques
                if self.registry.bindings[technique_id].fit_required
            )
            records.append(
                AdaptiveRaceRecord(
                    candidate,
                    evaluation,
                    deltas,
                    decision,
                    fit_required,
                    tuple(gate_failures),
                )
            )
        return tuple(records)

    def _survivors(
        self, records: tuple[AdaptiveRaceRecord, ...], limit: int
    ) -> tuple[CandidateSpec, ...]:
        control = next(
            (
                item.candidate
                for item in records
                if item.candidate.generation == "control"
            ),
            self.incumbent,
        )
        eligible = [
            item
            for item in records
            if item.candidate.generation != "control" and item.decision != "REJECT"
        ]
        eligible.sort(
            key=lambda item: (
                -item.evaluation.score,
                item.candidate.complexity,
                item.candidate.candidate_id,
            )
        )
        return (control, *(item.candidate for item in eligible[: max(0, limit - 1)]))


__all__ = [
    "AdaptiveCampaignResult",
    "AdaptiveEvaluation",
    "AdaptiveGhostLabEngine",
    "AdaptiveRaceRecord",
]
