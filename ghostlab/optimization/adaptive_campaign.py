from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace
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
SEMANTIC_WEIGHT_GRID = (0.05, 0.10, 0.15, 0.20)
SEMANTIC_F0_DEPTH = 10
SEMANTIC_F1_DEPTH = 20


@dataclass(frozen=True)
class AdaptiveEvaluation:
    candidate_id: str
    fidelity: Fidelity
    score: float
    session_rewards: tuple[float, ...]
    behavior_novelty: float = 0.0
    latency_p95_ms: float = 0.0
    semantic_latency_p95_ms: float = 0.0
    semantic_activations: int = 0
    semantic_rescue_opportunities: int = 0
    semantic_rescues: int = 0
    semantic_regressions: int = 0
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
    selected_semantic_weight: float
    selected_semantic_depth: int
    semantic_weight_survivors: tuple[float, ...]


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
            name="router_history_specificity_weight",
            kind="float",
            low=0.0,
            high=1.0,
        ),
        ConditionalParameter(
            name="router_current_attribute_weight",
            kind="float",
            low=0.0,
            high=1.0,
        ),
        ConditionalParameter(
            name="router_query_length_weight", kind="float", low=0.0, high=0.5
        ),
        ConditionalParameter(
            name="router_category_only_browsing_weight",
            kind="float",
            low=0.0,
            high=4.0,
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
        ConditionalParameter(
            name="semantic_weight", kind="categorical", choices=SEMANTIC_WEIGHT_GRID
        ),
        ConditionalParameter(
            name="semantic_rerank_k",
            kind="categorical",
            choices=(SEMANTIC_F0_DEPTH, SEMANTIC_F1_DEPTH),
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
            name="union_auxiliary_weight",
            kind="float",
            low=0.02,
            high=0.25,
            requires_any=(
                "ranking.fixed_lexical",
                "ranking.metadata_gbdt",
                "ranking.reward_lambdamart.v1",
                "ranking.turn_aware_lambdamart.v1",
                "ranking.fold_ensemble.v1",
                "fusion.rank_stack.v1",
            ),
        ),
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
        warm_start: CandidateSpec | None = None,
        candidate_limit: int = 500,
        beam_width: int = 24,
        exploration_fraction: float = 0.2,
        max_extra_techniques: int | None = None,
        seed: int = 20260826,
    ) -> None:
        semantic = baseline.semantic_ranker
        if (
            semantic.model_id != "smollm2-1.7b-instruct"
            or semantic.activation_policy != "browsing_only"
            or semantic.weight != SEMANTIC_WEIGHT_GRID[0]
            or semantic.rerank_k != SEMANTIC_F0_DEPTH
        ):
            raise ValueError(
                "GhostLab requires the frozen SmolLM2 Browsing-only control at "
                "weight 0.05 and depth 10"
            )
        inventory = registry.inventory()
        maximum = len(inventory.promotable)
        if max_extra_techniques is not None and max_extra_techniques <= 0:
            raise ValueError("max_extra_techniques must be positive or None")
        self.baseline = baseline
        self.registry = registry
        self.warm_start = warm_start
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
        if self.warm_start is not None:
            if self.warm_start.baseline_id != baseline.policy_id:
                raise ValueError("warm-start baseline ID does not match the control")
            if not self.warm_start.candidate_id.startswith("warm-start-"):
                raise ValueError(
                    "warm-start candidate ID must use the warm-start prefix"
                )
            self.registry.validate_candidate(self.warm_start)
            materialized = self.registry.materialize(self.baseline, self.warm_start)
            if materialized.architecture != self.baseline.architecture:
                raise ValueError("warm start changed the fixed architecture")

    def _execution_identity(self, candidate: CandidateSpec) -> str:
        """Hash runtime behavior plus fit-dispatch IDs, excluding provenance."""

        config = self.materialize(candidate).model_dump(mode="json")
        config.pop("policy_id", None)
        fit_required = tuple(
            sorted(
                technique_id
                for technique_id in candidate.techniques
                if self.registry.bindings[technique_id].fit_required
            )
        )
        encoded = json.dumps(
            {"config": config, "fit_required": fit_required},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _warm_start_drop_one_ablations(self) -> tuple[CandidateSpec, ...]:
        if self.warm_start is None:
            return ()
        compulsory = set(self.registry.inventory().compulsory)
        additions = tuple(sorted(set(self.warm_start.techniques) - compulsory))
        results: list[CandidateSpec] = []
        for removed in additions:
            techniques = tuple(
                item for item in self.warm_start.techniques if item != removed
            )
            active_parameters = self.registry.parameter_names_for(techniques)
            parameters = tuple(
                item
                for item in self.warm_start.parameters
                if item[0] in active_parameters
            )
            candidate = CandidateSpec(
                candidate_id=f"{self.warm_start.candidate_id}-without-{removed}",
                baseline_id=self.warm_start.baseline_id,
                techniques=techniques,
                parameters=parameters,
                complexity=len(additions) - 1,
                generation="ablation",
            )
            self.registry.validate_candidate(candidate)
            self.registry.materialize(self.baseline, candidate)
            results.append(candidate)
        return tuple(results)

    def plan_coverage(self, candidates: tuple[CandidateSpec, ...]) -> dict[str, object]:
        """Return explicit C/add-one/warm-ablation coverage evidence."""

        inventory = self.registry.inventory()
        compulsory = set(inventory.compulsory)
        controls = tuple(item for item in candidates if item.generation == "control")
        clean_control = (
            len(controls) == 1
            and set(controls[0].techniques) == compulsory
            and not controls[0].parameters
        )

        def dependency_closure(technique_id: str) -> set[str]:
            selected = set(compulsory) | {technique_id}
            pending = [technique_id]
            while pending:
                technique = self.registry.catalog.techniques.get(pending.pop())
                if technique is None:
                    continue
                for required in technique.requires:
                    if required not in selected:
                        selected.add(required)
                        pending.append(required)
            return selected

        add_one = {
            technique_id
            for technique_id in inventory.promotable
            if any(
                item.generation in {"single", "ablation"}
                and set(item.techniques) == dependency_closure(technique_id)
                for item in candidates
            )
        }
        missing_add_one = tuple(sorted(set(inventory.promotable) - add_one))
        selected_control_only = tuple(
            sorted(
                {
                    technique_id
                    for item in candidates
                    for technique_id in item.techniques
                    if technique_id in set(inventory.control_only)
                }
            )
        )
        absorbed_controls = {
            technique_id: binding.absorbed_by
            for technique_id, binding in self.registry.bindings.items()
            if binding.role == "control_only"
            and getattr(binding, "absorbed", False)
            and binding.absorbed_by is not None
        }
        missing_absorbed_parents = tuple(
            sorted(
                technique_id
                for technique_id, parent in absorbed_controls.items()
                if parent not in add_one
            )
        )
        warm_additions: tuple[str, ...] = ()
        expected_drop_one_ids: tuple[str, ...] = ()
        missing_drop_one_ids: tuple[str, ...] = ()
        if self.warm_start is not None:
            warm_additions = tuple(sorted(set(self.warm_start.techniques) - compulsory))
            expected_drop_one_ids = tuple(
                f"{self.warm_start.candidate_id}-without-{item}"
                for item in warm_additions
            )
            planned_ids = {item.candidate_id for item in candidates}
            missing_drop_one_ids = tuple(
                item for item in expected_drop_one_ids if item not in planned_ids
            )
        return {
            "clean_control_verified": clean_control,
            "control_candidate_ids": tuple(item.candidate_id for item in controls),
            "promotable_safe_optionals": inventory.promotable,
            "add_one_covered": tuple(sorted(add_one)),
            "missing_add_one": missing_add_one,
            "add_one_coverage_complete": not missing_add_one,
            "control_only_explicitly_excluded": inventory.control_only,
            "control_only_selected": selected_control_only,
            "absorbed_control_dependencies": absorbed_controls,
            "missing_absorbed_control_parents": missing_absorbed_parents,
            "absorbed_control_coverage_complete": not missing_absorbed_parents,
            "warm_start_additions": warm_additions,
            "expected_warm_drop_one_candidate_ids": expected_drop_one_ids,
            "missing_warm_drop_one_candidate_ids": missing_drop_one_ids,
            "warm_drop_one_coverage_complete": not missing_drop_one_ids,
        }

    def initial_plan(self) -> InteractionSearchPlan:
        inventory = self.registry.inventory()
        # Reserve one beam for evidence-guided higher-order expansion. Without
        # this split, standalone/pair enumeration consumes the complete focused
        # budget and the advertised warm-start combination round can never run.
        initial_limits = replace(
            self.limits,
            max_candidates=max(1, self.limits.max_candidates - self.limits.beam_width),
        )
        plan = plan_standalones_and_pairs(
            self.registry.catalog,
            baseline_id=self.baseline.policy_id,
            baseline_techniques=inventory.compulsory,
            technique_ids=inventory.promotable,
            limits=initial_limits,
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
        semantic_calibration = tuple(
            self.semantic_candidate(weight=weight, depth=SEMANTIC_F0_DEPTH, suffix="f0")
            for weight in SEMANTIC_WEIGHT_GRID[1:]
        )
        candidates = [*accepted]
        if self.warm_start is not None:
            provenance_candidates = (
                self.warm_start,
                *self._warm_start_drop_one_ablations(),
            )
            provenance_identities = {
                self._execution_identity(item) for item in provenance_candidates
            }
            controls = [item for item in candidates if item.generation == "control"]
            ordinary = [
                item
                for item in candidates
                if item.generation != "control"
                and self._execution_identity(item) not in provenance_identities
            ]
            # A warm start is useful only if it is evaluated before the broad
            # add-one/pair field. Keep C first, then D0 and its drop-one
            # ablations, while removing execution-equivalent ordinary entries.
            candidates = [*controls, *provenance_candidates, *ordinary]
        candidates.extend(semantic_calibration)
        result = InteractionSearchPlan(
            candidates=tuple(candidates),
            skipped=tuple(skipped),
            cap_exhausted=plan.cap_exhausted,
        )
        coverage = self.plan_coverage(result.candidates)
        if not coverage["clean_control_verified"]:
            raise RuntimeError("adaptive plan contaminated the fixed C control")
        if not coverage["add_one_coverage_complete"]:
            raise RuntimeError(
                "adaptive plan omitted promotable add-one candidates: "
                f"{coverage['missing_add_one']}"
            )
        if coverage["control_only_selected"]:
            raise RuntimeError(
                "adaptive plan selected control-only techniques: "
                f"{coverage['control_only_selected']}"
            )
        if not coverage["absorbed_control_coverage_complete"]:
            raise RuntimeError(
                "adaptive plan omitted add-one parents for absorbed control-only "
                f"dependencies: {coverage['missing_absorbed_control_parents']}"
            )
        if not coverage["warm_drop_one_coverage_complete"]:
            raise RuntimeError(
                "adaptive plan omitted warm-seed drop-one ablations: "
                f"{coverage['missing_warm_drop_one_candidate_ids']}"
            )
        return result

    def materialize(self, candidate: CandidateSpec) -> AdaptiveHybridConfig:
        self.registry.validate_candidate(candidate)
        return self.registry.materialize(self.baseline, candidate)

    @staticmethod
    def _semantic_parameters(candidate: CandidateSpec) -> tuple[float, int]:
        parameters = dict(candidate.parameters)
        return (
            float(parameters.get("semantic_weight", SEMANTIC_WEIGHT_GRID[0])),
            int(parameters.get("semantic_rerank_k", SEMANTIC_F0_DEPTH)),
        )

    @staticmethod
    def _is_semantic_calibration(candidate: CandidateSpec) -> bool:
        return candidate.candidate_id.startswith("semantic-calibration-")

    def semantic_candidate(
        self,
        *,
        weight: float,
        depth: int,
        suffix: str,
    ) -> CandidateSpec:
        if weight not in SEMANTIC_WEIGHT_GRID:
            raise ValueError(f"semantic weight is outside the fixed grid: {weight}")
        if depth not in {SEMANTIC_F0_DEPTH, SEMANTIC_F1_DEPTH}:
            raise ValueError(f"semantic depth is outside the staged grid: {depth}")
        rendered_weight = f"{weight:.2f}".replace(".", "p")
        candidate = CandidateSpec(
            candidate_id=(f"semantic-calibration-w{rendered_weight}-d{depth}-{suffix}"),
            baseline_id=self.baseline.policy_id,
            techniques=self.incumbent.techniques,
            parameters=(
                ("semantic_rerank_k", depth),
                ("semantic_weight", weight),
            ),
            complexity=0,
            generation="ablation",
        )
        self.materialize(candidate)
        return candidate

    def _with_semantic_policy(
        self,
        candidate: CandidateSpec,
        *,
        weight: float,
        depth: int,
        suffix: str,
    ) -> CandidateSpec:
        if candidate.generation == "control":
            return candidate
        parameters = {
            name: value
            for name, value in candidate.parameters
            if name not in {"semantic_weight", "semantic_rerank_k"}
        }
        parameters.update({"semantic_weight": weight, "semantic_rerank_k": depth})
        rendered_weight = f"{weight:.2f}".replace(".", "p")
        result = candidate.model_copy(
            update={
                "candidate_id": (
                    f"{candidate.candidate_id}-sem-w{rendered_weight}-d{depth}-{suffix}"
                ),
                "parameters": tuple(sorted(parameters.items())),
            }
        )
        self.materialize(result)
        return result

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
            suggestion = dict(
                suggest_for_combination(
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
            )
            # Semantic weight/depth are tuned by the explicit F0 -> F1 lane below.
            # Ordinary HPO remains free to tune every other active parameter but
            # cannot bypass the staged grid or re-open arbitrary semantic depths.
            suggestion.pop("semantic_weight", None)
            suggestion.pop("semantic_rerank_k", None)
            suggestion.pop("semantic_fallback_weight", None)
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
        all_candidates = [
            candidate
            for candidate in plan.candidates
            if not self._is_semantic_calibration(candidate)
        ]
        f0_by_id = {
            item.candidate.candidate_id: item
            for item in f0
            if not self._is_semantic_calibration(item.candidate)
        }
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

        control_f0 = next(item for item in f0 if item.candidate.generation == "control")
        semantic_f0 = tuple(
            item for item in f0 if self._is_semantic_calibration(item.candidate)
        )
        semantic_survivors = [SEMANTIC_WEIGHT_GRID[0]]
        semantic_survivors.extend(
            self._semantic_parameters(item.candidate)[0]
            for item in semantic_f0
            if item.decision != "REJECT"
        )
        semantic_survivors = sorted(set(semantic_survivors))
        semantic_records = (control_f0, *semantic_f0)
        eligible_semantic_records = [
            item
            for item in semantic_records
            if item.candidate.generation == "control" or item.decision != "REJECT"
        ]
        selected_semantic_weight = self._semantic_parameters(
            min(
                eligible_semantic_records,
                key=lambda item: (
                    -item.evaluation.score,
                    -item.evaluation.hit_rate_at_10,
                    -item.evaluation.mrr,
                    item.evaluation.latency_p95_ms,
                    self._semantic_parameters(item.candidate)[0],
                ),
            ).candidate
        )[0]

        structural_f0 = tuple(
            item for item in f0 if not self._is_semantic_calibration(item.candidate)
        )
        raw_f1_roots = self._survivors(
            structural_f0,
            max(2, f1_candidates - 2),
        )
        f1_roots = tuple(
            root
            if root.generation == "control"
            else self._with_semantic_policy(
                root,
                weight=selected_semantic_weight,
                depth=SEMANTIC_F0_DEPTH,
                suffix="f1",
            )
            for root in raw_f1_roots
        )
        semantic_depth_10 = (
            None
            if selected_semantic_weight == SEMANTIC_WEIGHT_GRID[0]
            else self.semantic_candidate(
                weight=selected_semantic_weight,
                depth=SEMANTIC_F0_DEPTH,
                suffix="f1",
            )
        )
        semantic_depth_20 = self.semantic_candidate(
            weight=selected_semantic_weight,
            depth=SEMANTIC_F1_DEPTH,
            suffix="f1",
        )
        hpo: list[CandidateSpec] = []
        if hpo_trials_per_structure < 0:
            raise ValueError("HPO trial count cannot be negative")
        f0_scores = {
            item.candidate.candidate_id: item.evaluation.score for item in structural_f0
        }
        for root in f1_roots:
            if root.generation == "control":
                continue
            original_id = root.candidate_id.split("-sem-w", maxsplit=1)[0]
            observations = (
                Observation(
                    root.parameters,
                    f0_scores.get(original_id, control_f0.evaluation.score),
                ),
            )
            hpo.extend(
                self.suggest_hpo_candidates(
                    root,
                    count=hpo_trials_per_structure,
                    observations=observations,
                )
            )
        f1_candidates_to_run = [*f1_roots]
        if semantic_depth_10 is not None:
            f1_candidates_to_run.append(semantic_depth_10)
        f1_candidates_to_run.extend((semantic_depth_20, *hpo))
        f1 = self._evaluate_stage(tuple(f1_candidates_to_run), "f1", evaluator)

        depth_10_record = (
            next(
                item
                for item in f1
                if semantic_depth_10 is not None
                and item.candidate.candidate_id == semantic_depth_10.candidate_id
            )
            if semantic_depth_10 is not None
            else next(item for item in f1 if item.candidate.generation == "control")
        )
        depth_20_record = next(
            item
            for item in f1
            if item.candidate.candidate_id == semantic_depth_20.candidate_id
        )
        depth_20_has_value = (
            depth_20_record.evaluation.hit_rate_at_10
            > depth_10_record.evaluation.hit_rate_at_10 + 1e-12
            or depth_20_record.evaluation.mrr > depth_10_record.evaluation.mrr + 1e-6
            or depth_20_record.evaluation.semantic_rescues
            > depth_10_record.evaluation.semantic_rescues
        )
        selected_semantic_depth = (
            SEMANTIC_F1_DEPTH
            if depth_20_record.decision != "REJECT" and depth_20_has_value
            else SEMANTIC_F0_DEPTH
        )

        structural_f1 = tuple(
            item for item in f1 if not self._is_semantic_calibration(item.candidate)
        )
        raw_f2_roots = self._survivors(structural_f1, f2_candidates)
        f2_candidates_to_run: list[CandidateSpec] = [self.incumbent]
        if (
            selected_semantic_weight != SEMANTIC_WEIGHT_GRID[0]
            or selected_semantic_depth != SEMANTIC_F0_DEPTH
        ):
            f2_candidates_to_run.append(
                self.semantic_candidate(
                    weight=selected_semantic_weight,
                    depth=selected_semantic_depth,
                    suffix="f2",
                )
            )
        for root in raw_f2_roots:
            if root.generation == "control":
                continue
            candidate = self._with_semantic_policy(
                root,
                weight=selected_semantic_weight,
                depth=selected_semantic_depth,
                suffix="f2",
            )
            if candidate.canonical_hash() not in {
                item.canonical_hash() for item in f2_candidates_to_run
            }:
                f2_candidates_to_run.append(candidate)
            if len(f2_candidates_to_run) >= f2_candidates:
                break
        f2 = self._evaluate_stage(tuple(f2_candidates_to_run), "f2", evaluator)
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
            selected_semantic_weight=selected_semantic_weight,
            selected_semantic_depth=selected_semantic_depth,
            semantic_weight_survivors=tuple(semantic_survivors),
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
                key=lambda item: (
                    item.generation != "control",
                    not item.candidate_id.startswith("warm-start-"),
                    item.candidate_id,
                ),
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
            control_evaluation = matched_control.evaluation
            if (
                candidate.generation != "control"
                and evaluation.hit_rate_at_10 + 1e-12
                < control_evaluation.hit_rate_at_10
            ):
                gate_failures.append(
                    "hit_at_10_regression:"
                    f"{evaluation.hit_rate_at_10:.6f}<"
                    f"{control_evaluation.hit_rate_at_10:.6f}"
                )
            if (
                candidate.generation != "control"
                and evaluation.mrr + 1e-12 < control_evaluation.mrr
            ):
                gate_failures.append(
                    f"mrr_regression:{evaluation.mrr:.6f}<{control_evaluation.mrr:.6f}"
                )
            control_metrics = dict(matched_control.evaluation.gate_metrics)
            for name, value in evaluation.gate_metrics:
                if name in control_metrics and value < control_metrics[name] - 0.02:
                    gate_failures.append(
                        f"non_regression:{name}:{value:.6f}<"
                        f"{control_metrics[name]:.6f}-0.02"
                    )
            quality_or_rescue_gain = (
                evaluation.hit_rate_at_10 > control_evaluation.hit_rate_at_10 + 1e-12
                or evaluation.mrr > control_evaluation.mrr + 1e-6
                or evaluation.semantic_rescues > control_evaluation.semantic_rescues
            )
            if (
                candidate.generation != "control"
                and control_evaluation.latency_p95_ms > 0
            ):
                latency_ratio = (
                    evaluation.latency_p95_ms / control_evaluation.latency_p95_ms
                )
                latency_limit = 2.5 if quality_or_rescue_gain else 1.5
                if latency_ratio > latency_limit:
                    gate_failures.append(
                        f"latency_ratio:{latency_ratio:.3f}>{latency_limit:.3f}"
                    )
            if (
                candidate.generation != "control"
                and control_evaluation.semantic_latency_p95_ms > 0
                and evaluation.semantic_activations > 0
            ):
                semantic_latency_ratio = (
                    evaluation.semantic_latency_p95_ms
                    / control_evaluation.semantic_latency_p95_ms
                )
                semantic_latency_limit = 2.25 if quality_or_rescue_gain else 1.5
                if semantic_latency_ratio > semantic_latency_limit:
                    gate_failures.append(
                        "semantic_latency_ratio:"
                        f"{semantic_latency_ratio:.3f}>{semantic_latency_limit:.3f}"
                    )
            _, semantic_depth = self._semantic_parameters(candidate)
            if (
                candidate.generation != "control"
                and semantic_depth == SEMANTIC_F1_DEPTH
                and not quality_or_rescue_gain
            ):
                gate_failures.append("depth20_without_quality_or_rescue_gain")
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
