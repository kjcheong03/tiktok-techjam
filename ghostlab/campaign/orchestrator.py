from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ghostlab.campaign.analyze import CandidateEvaluation, paired_analysis
from ghostlab.campaign.bindings import TechniqueBindingRegistry
from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.controller import CampaignStage, promote_stage
from ghostlab.campaign.interaction_search import (
    CandidateEvidence,
    SearchLimits,
    plan_higher_order_round,
    plan_standalones_and_pairs,
)
from ghostlab.campaign.jobs import build_jobs
from ghostlab.campaign.models import (
    CampaignJob,
    CampaignManifest,
    CandidateSpec,
    ChampionComparison,
    Fidelity,
    JobOutcome,
    MetricSnapshot,
)
from ghostlab.campaign.planner import backward_ablations
from ghostlab.campaign.runner import CampaignCheckpoint, run_jobs
from ghostlab.optimization.bohb import Observation
from ghostlab.optimization.conditional import (
    ConditionalSearchSpace,
    TuningContext,
    suggest_for_combination,
)
from ghostlab.research.technique_suite import (
    UnifiedTechniqueConfig,
    load_suite_config,
)

EvaluatorFactory = Callable[
    [tuple[CandidateSpec, ...]], Callable[[CampaignJob], JobOutcome]
]
CHAMPION_PRESET = "configs/suites/champion_guarded.json"


@dataclass(frozen=True)
class FrozenInputs:
    catalog_path: Path
    dataset_path: Path
    adaptive_split_path: Path
    nested_split_path: Path


@dataclass(frozen=True)
class CampaignOptions:
    f1_candidates: int = 24
    f2_candidates: int = 6
    hpo_trials_per_structure: int = 8
    higher_order_rounds: int = 2
    maximum_scenario_regression: float = 0.02
    bootstrap_resamples: int = 1000

    def __post_init__(self) -> None:
        if min(self.f1_candidates, self.f2_candidates) <= 0:
            raise ValueError("promotion limits must be positive")
        if self.hpo_trials_per_structure < 0 or self.higher_order_rounds < 0:
            raise ValueError("search limits cannot be negative")
        if self.maximum_scenario_regression < 0.0:
            raise ValueError("scenario regression limit cannot be negative")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap resamples must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_ids_hash(sample_ids: set[str]) -> str:
    encoded = json.dumps(
        sorted(sample_ids), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _robust_proposal_key(
    item: tuple[CandidateSpec, CandidateEvaluation, dict[str, object]],
) -> tuple[float, float, int, float, str]:
    candidate, evaluation, analysis = item
    interval = analysis.get("confidence_interval", ())
    lower = (
        float(interval[0])
        if isinstance(interval, (list, tuple)) and interval
        else float("-inf")
    )
    raw_scenarios = analysis.get("scenario_deltas", {})
    scenario_deltas = (
        [float(value) for value in raw_scenarios.values()]
        if isinstance(raw_scenarios, dict)
        else []
    )
    worst_scenario = min(scenario_deltas, default=float("-inf"))
    raw_wins = analysis.get("wins", 0)
    raw_losses = analysis.get("losses", 0)
    wins = int(raw_wins) if isinstance(raw_wins, (int, float)) else 0
    losses = int(raw_losses) if isinstance(raw_losses, (int, float)) else 0
    return (
        -lower,
        -worst_scenario,
        -(wins - losses),
        -evaluation.score,
        candidate.candidate_id,
    )


def _efficient_proposal_key(
    item: tuple[CandidateSpec, CandidateEvaluation, dict[str, object]],
) -> tuple[float, float, int, float, str]:
    candidate, evaluation, _ = item
    return (
        evaluation.latency_p95_ms,
        evaluation.memory_mb,
        candidate.complexity,
        -evaluation.score,
        candidate.candidate_id,
    )


def _record_candidate_id(record: dict[str, object]) -> str | None:
    candidate = record.get("candidate")
    if not isinstance(candidate, dict):
        return None
    candidate_id = candidate.get("candidate_id")
    return candidate_id if isinstance(candidate_id, str) else None


def verify_frozen_inputs(
    manifest: CampaignManifest,
    catalog: TechniqueCatalog,
    inputs: FrozenInputs,
) -> dict[str, str]:
    """Verify every development input covered by the frozen manifest."""

    expected = {
        "catalog": manifest.catalog_hash,
        "dataset": manifest.dataset_hash,
        "adaptive_split": manifest.adaptive_split_hash,
        "nested_split": manifest.nested_split_hash,
    }
    actual = {
        "catalog": catalog.content_hash,
        "dataset": sha256_file(inputs.dataset_path),
        "adaptive_split": sha256_file(inputs.adaptive_split_path),
        "nested_split": sha256_file(inputs.nested_split_path),
    }
    mismatches = {
        name: f"expected {expected[name]}, got {value}"
        for name, value in actual.items()
        if value != expected[name]
    }
    if mismatches:
        raise ValueError(f"frozen campaign input mismatch: {mismatches}")
    adaptive = json.loads(inputs.adaptive_split_path.read_text(encoding="utf-8"))
    if adaptive.get("dataset_sha256") != manifest.dataset_hash:
        raise ValueError("adaptive split belongs to another dataset")
    nested = json.loads(inputs.nested_split_path.read_text(encoding="utf-8"))
    adaptive_ids = tuple(str(value) for value in adaptive["sample_ids"])
    nested_ids = tuple(str(value) for value in nested["adaptive_sample_ids"])
    if set(adaptive_ids) != set(nested_ids) or len(nested_ids) != len(set(nested_ids)):
        raise ValueError("nested split does not match the frozen adaptive split")
    raw_folds = nested.get("outer_folds")
    if not isinstance(raw_folds, list):
        raise TypeError("nested outer_folds must be a list")
    folds = tuple(tuple(str(value) for value in fold) for fold in raw_folds)
    flattened = tuple(sample_id for fold in folds for sample_id in fold)
    if set(flattened) != set(adaptive_ids) or len(flattened) != len(set(flattened)):
        raise ValueError("nested outer folds must partition the adaptive split")
    manifest.validate_fold_partition(len(folds))
    return actual


class AutonomousCampaign:
    """Bounded F0/F1/F2 proposal campaign; it never promotes production state."""

    def __init__(
        self,
        *,
        manifest: CampaignManifest,
        catalog: TechniqueCatalog,
        registry: TechniqueBindingRegistry,
        evaluator_factory: EvaluatorFactory,
        checkpoint_path: Path,
        evidence_path: Path,
        outer_folds: tuple[tuple[str, ...], ...],
        project_root: Path = Path("."),
        search_space_path: Path = Path("configs/search/wave2_weight_space_v1.json"),
        verified_input_hashes: dict[str, str] | None = None,
        fit_capable_techniques: frozenset[str] = frozenset(),
        options: CampaignOptions | None = None,
    ) -> None:
        self.manifest = manifest
        self.catalog = catalog
        self.registry = registry
        self.evaluator_factory = evaluator_factory
        self.checkpoint_path = checkpoint_path
        self.evidence_path = evidence_path
        manifest.validate_fold_partition(len(outer_folds))
        self.outer_folds = outer_folds
        self.options = options or CampaignOptions()
        self.project_root = project_root
        self.verified_input_hashes = dict(verified_input_hashes or {})
        self.fit_capable_techniques = fit_capable_techniques
        resolved_space = (
            search_space_path
            if search_space_path.is_absolute()
            else project_root / search_space_path
        )
        if manifest.search_space_hash is not None:
            actual_search_hash = sha256_file(resolved_space)
            if actual_search_hash != manifest.search_space_hash:
                raise ValueError("frozen conditional search-space hash mismatch")
        self.search_space = ConditionalSearchSpace.model_validate_json(
            resolved_space.read_text(encoding="utf-8")
        )
        self.manifest_hash = manifest.canonical_hash()
        self._baselines = {
            preset: load_suite_config(project_root / preset)
            for preset in manifest.baseline_presets
        }
        self._started = 0.0

    def _additions(self, candidate: CandidateSpec) -> tuple[str, ...]:
        inherited = set(self.manifest.techniques_for_preset(candidate.baseline_id))
        return tuple(item for item in candidate.techniques if item not in inherited)

    def _fit_receipts_complete(
        self,
        candidate: CandidateSpec,
        job_ids: tuple[str, ...],
        outcomes: dict[str, JobOutcome],
    ) -> bool:
        required = {
            item
            for item in self._additions(candidate)
            if self.catalog.techniques[item].fit_required
        }
        if not required:
            return True
        for job_id in job_ids:
            outcome = outcomes.get(job_id)
            if outcome is None:
                return False
            received = {item.technique_id for item in outcome.fit_receipts}
            if not required <= received:
                return False
        return True

    @staticmethod
    def _weighted_metric(rows: list[JobOutcome], field: str) -> float | None:
        values = [getattr(row, field) for row in rows]
        if any(value is None for value in values):
            return None
        total = sum(len(row.session_rewards) for row in rows)
        if total == 0:
            return None
        return (
            sum(
                float(value) * len(row.session_rewards)
                for row, value in zip(rows, values, strict=True)
            )
            / total
        )

    def materialize(self, candidate: CandidateSpec) -> UnifiedTechniqueConfig:
        baseline = self._baselines[candidate.baseline_id]
        inherited = set(self.manifest.techniques_for_preset(candidate.baseline_id))
        additions = tuple(
            item for item in candidate.techniques if item not in inherited
        )
        patch_candidate = candidate.model_copy(update={"techniques": additions})
        return self.registry.materialize(baseline, patch_candidate)

    def _runnable(
        self, candidates: tuple[CandidateSpec, ...]
    ) -> tuple[tuple[CandidateSpec, ...], list[dict[str, object]]]:
        accepted: list[CandidateSpec] = []
        rejected: list[dict[str, object]] = []
        for candidate in candidates:
            try:
                self.materialize(candidate)
            except Exception as error:  # noqa: BLE001 - preflight evidence boundary
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "baseline_id": candidate.baseline_id,
                        "classification": "materialization_blocked",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                accepted.append(candidate)
        return tuple(accepted), rejected

    def _initial_candidates(
        self,
    ) -> tuple[tuple[CandidateSpec, ...], list[dict[str, object]]]:
        composable = tuple(
            technique_id
            for technique_id in self.manifest.technique_ids
            if technique_id in self.registry.bindings
            and self.registry.bindings[technique_id].disposition == "composable"
            and technique_id in self.catalog.techniques
            and self.catalog.techniques[technique_id].executable
            and self.catalog.techniques[technique_id].execution_mode == "runtime"
        )
        excluded: list[dict[str, object]] = [
            {
                "technique_id": technique_id,
                "classification": (
                    self.registry.bindings[technique_id].disposition
                    if technique_id in self.registry.bindings
                    else "unavailable"
                ),
                "reason": (
                    self.registry.bindings[technique_id].reason
                    if technique_id in self.registry.bindings
                    else "missing binding"
                ),
            }
            for technique_id in self.manifest.technique_ids
            if technique_id not in composable
        ]
        discovery_presets = tuple(
            preset
            for preset in self.manifest.baseline_presets
            if self.manifest.search_mode_for_preset(preset) == "composable"
        )
        control_count = len(self.manifest.baseline_presets) - len(discovery_presets)
        discovery_budget = self.manifest.candidate_limit - control_count
        if discovery_budget < len(discovery_presets):
            raise ValueError("candidate limit cannot fit every declared anchor control")
        per_anchor = max(1, discovery_budget // len(discovery_presets))
        planned: list[CandidateSpec] = []
        for preset in self.manifest.baseline_presets:
            baseline = self.manifest.techniques_for_preset(preset)
            if self.manifest.search_mode_for_preset(preset) == "control_only":
                planned.append(
                    CandidateSpec(
                        candidate_id=f"control-{hashlib.sha256(preset.encode()).hexdigest()[:12]}",
                        baseline_id=preset,
                        techniques=baseline,
                        complexity=0,
                        generation="control",
                    )
                )
                continue
            search = plan_standalones_and_pairs(
                self.catalog,
                baseline_id=preset,
                baseline_techniques=baseline,
                technique_ids=composable,
                limits=SearchLimits(
                    max_order=self.manifest.max_order,
                    max_candidates=per_anchor,
                    max_wall_seconds=self.manifest.max_wall_seconds,
                    exploration_fraction=self.manifest.exploration_fraction,
                    seed=self.manifest.seeds[0],
                ),
            )
            planned.extend(search.candidates)
        runnable, rejected = self._runnable(
            tuple(planned[: self.manifest.candidate_limit])
        )
        return runnable, [*excluded, *rejected]

    def _stage(
        self, fidelity: Fidelity, candidates: tuple[CandidateSpec, ...]
    ) -> CampaignStage:
        seeds = (self.manifest.seeds[0],) if fidelity == "f2" else self.manifest.seeds
        return self._stage_with_seeds(fidelity, candidates, seeds)

    def _stage_with_seeds(
        self,
        fidelity: Fidelity,
        candidates: tuple[CandidateSpec, ...],
        seeds: tuple[int, ...],
    ) -> CampaignStage:
        outer_fold_count = len(self.outer_folds)
        jobs = build_jobs(
            self.catalog,
            candidates,
            fidelity=fidelity,
            outer_fold_count=outer_fold_count,
            seeds=seeds,
        )
        if fidelity == "f2":
            jobs = tuple(
                job
                for job in jobs
                if job.outer_fold in self.manifest.confirmation_outer_folds
            )
        return CampaignStage(
            fidelity=fidelity,
            candidates=candidates,
            jobs=jobs,
        )

    def _run_stage(self, stage: CampaignStage) -> CampaignCheckpoint:
        return run_jobs(
            stage.jobs,
            manifest_hash=self.manifest_hash,
            resources=self.manifest.resources,
            checkpoint_path=self.checkpoint_path,
            evaluator=self.evaluator_factory(stage.candidates),
            progress_path=self.evidence_path.parent / "live_status.json",
        )

    @staticmethod
    def _evaluations(
        stage: CampaignStage, outcomes: dict[str, JobOutcome]
    ) -> dict[str, CandidateEvaluation]:
        by_hash = {item.canonical_hash(): item for item in stage.candidates}
        grouped: dict[str, list[JobOutcome]] = {}
        for job in stage.jobs:
            outcome = outcomes.get(job.job_id)
            if outcome is not None and outcome.state == "complete":
                grouped.setdefault(job.candidate_hash, []).append(outcome)
        result: dict[str, CandidateEvaluation] = {}
        for candidate_hash, rows in grouped.items():
            candidate = by_hash[candidate_hash]
            scenario_names = {name for row in rows for name in row.scenario_scores}
            session_rewards = tuple(
                value for row in rows for value in row.session_rewards
            )
            result[candidate.candidate_id] = CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                complexity=candidate.complexity,
                score=statistics.fmean(session_rewards),
                session_rewards=session_rewards,
                scenario_scores={
                    name: statistics.fmean(
                        row.scenario_scores[name]
                        for row in rows
                        if name in row.scenario_scores
                    )
                    for name in sorted(scenario_names)
                },
                latency_p95_ms=max(row.latency_p95_ms for row in rows),
                memory_mb=max(row.memory_mb for row in rows),
                hit_rate_at_10=AutonomousCampaign._weighted_metric(
                    rows, "hit_rate_at_10"
                ),
                mrr=AutonomousCampaign._weighted_metric(rows, "mrr"),
                mttc=AutonomousCampaign._weighted_metric(rows, "mttc"),
            )
        return result

    def _champion_comparison(
        self,
        candidate: CandidateEvaluation,
        champion: CandidateEvaluation,
        *,
        fit_receipts_verified: bool,
    ) -> ChampionComparison:
        analysis = paired_analysis(
            candidate,
            champion,
            resamples=self.options.bootstrap_resamples,
            seed=self.manifest.seeds[0],
        )
        scenario_safe = all(
            delta >= -self.options.maximum_scenario_regression
            for delta in analysis.scenario_deltas.values()
        )
        statistically_supported = (
            analysis.confidence_interval[0] > 0.0
            and analysis.randomization_pvalue <= 0.05
        )
        beats = analysis.mean_delta > 0.0

        def delta(left: float | None, right: float | None) -> float | None:
            return None if left is None or right is None else left - right

        return ChampionComparison(
            champion_candidate_id=champion.candidate_id,
            champion_baseline_id=CHAMPION_PRESET,
            candidate_metrics=MetricSnapshot(
                technical_score=candidate.score,
                hit_rate_at_10=candidate.hit_rate_at_10,
                mrr=candidate.mrr,
                mttc=candidate.mttc,
            ),
            champion_metrics=MetricSnapshot(
                technical_score=champion.score,
                hit_rate_at_10=champion.hit_rate_at_10,
                mrr=champion.mrr,
                mttc=champion.mttc,
            ),
            technical_score_delta=candidate.score - champion.score,
            hit_rate_at_10_delta=delta(
                candidate.hit_rate_at_10, champion.hit_rate_at_10
            ),
            mrr_delta=delta(candidate.mrr, champion.mrr),
            mttc_delta=delta(candidate.mttc, champion.mttc),
            paired_mean_delta=analysis.mean_delta,
            confidence_interval=analysis.confidence_interval,
            randomization_pvalue=analysis.randomization_pvalue,
            paired_session_count=len(candidate.session_rewards),
            wins=analysis.wins,
            ties=analysis.ties,
            losses=analysis.losses,
            scenario_deltas=analysis.scenario_deltas,
            beats_champion_point_estimate=beats,
            statistically_supported=statistically_supported,
            maximum_scenario_regression=self.options.maximum_scenario_regression,
            no_material_scenario_regression=scenario_safe,
            fit_receipts_verified=fit_receipts_verified,
            promotion_recommended=(
                beats
                and statistically_supported
                and scenario_safe
                and fit_receipts_verified
            ),
        )

    def _evidence(
        self, stage: CampaignStage, outcomes: dict[str, JobOutcome]
    ) -> tuple[dict[str, CandidateEvidence], dict[str, object]]:
        evaluations = self._evaluations(stage, outcomes)
        controls = {
            item.baseline_id: evaluations[item.candidate_id]
            for item in stage.candidates
            if item.generation == "control" and item.candidate_id in evaluations
        }
        evidence: dict[str, CandidateEvidence] = {}
        analyses: dict[str, object] = {}
        completed_counts = {
            item.canonical_hash(): sum(
                outcomes.get(job.job_id) is not None
                and outcomes[job.job_id].state == "complete"
                for job in stage.jobs
                if job.candidate_hash == item.canonical_hash()
            )
            for item in stage.candidates
        }
        for candidate in stage.candidates:
            current = evaluations.get(candidate.candidate_id)
            control = controls.get(candidate.baseline_id)
            if current is None or control is None:
                continue
            if candidate.generation == "control":
                evidence[candidate.candidate_id] = CandidateEvidence(
                    candidate.candidate_id,
                    0.0,
                    0.0,
                    0.0,
                    repeated_evaluations=max(
                        1, completed_counts[candidate.canonical_hash()]
                    ),
                )
                continue
            analysis = paired_analysis(
                current,
                control,
                resamples=self.options.bootstrap_resamples,
                seed=self.manifest.seeds[0],
            )
            evidence[candidate.candidate_id] = CandidateEvidence(
                candidate.candidate_id,
                analysis.mean_delta,
                analysis.confidence_interval[0],
                analysis.confidence_interval[1],
                repeated_evaluations=max(
                    1, completed_counts[candidate.canonical_hash()]
                ),
            )
            analyses[candidate.candidate_id] = analysis.__dict__
        return evidence, analyses

    def _expand_interactions(
        self,
        stage: CampaignStage,
        checkpoint: CampaignCheckpoint,
    ) -> tuple[CampaignStage, CampaignCheckpoint, list[dict[str, object]]]:
        diagnostics: list[dict[str, object]] = []
        current = stage
        for round_index in range(self.options.higher_order_rounds):
            additions: list[CandidateSpec] = []
            evidence, _ = self._evidence(current, checkpoint.outcomes)
            elapsed = time.perf_counter() - self._started
            completed = [
                row.elapsed_seconds
                for row in checkpoint.outcomes.values()
                if row.state == "complete" and row.elapsed_seconds > 0.0
            ]
            estimate = statistics.median(completed) if completed else 1.0
            for preset in self.manifest.baseline_presets:
                if self.manifest.search_mode_for_preset(preset) == "control_only":
                    continue
                anchor = tuple(
                    item for item in current.candidates if item.baseline_id == preset
                )
                remaining_cap = max(
                    1, self.manifest.candidate_limit - len(current.candidates)
                )
                plan = plan_higher_order_round(
                    self.catalog,
                    evaluated_candidates=anchor,
                    evidence={
                        item.candidate_id: evidence[item.candidate_id]
                        for item in anchor
                        if item.candidate_id in evidence
                    },
                    technique_ids=self.manifest.technique_ids,
                    baseline_techniques=self.manifest.techniques_for_preset(preset),
                    limits=SearchLimits(
                        max_order=self.manifest.max_order,
                        max_candidates=len(anchor) + remaining_cap,
                        max_wall_seconds=self.manifest.max_wall_seconds,
                        exploration_fraction=self.manifest.exploration_fraction,
                        seed=self.manifest.seeds[0] + round_index,
                    ),
                    consumed_wall_seconds=elapsed,
                    estimated_candidate_seconds=estimate,
                )
                additions.extend(plan.candidates)
                ablation_parents = sorted(
                    (
                        item
                        for item in anchor
                        if item.complexity >= 3 and item.candidate_id in evidence
                    ),
                    key=lambda item: (
                        -evidence[item.candidate_id].mean_delta,
                        item.candidate_id,
                    ),
                )[:3]
                ablations = tuple(
                    child
                    for parent in ablation_parents
                    for child in backward_ablations(parent)
                )
                additions.extend(ablations)
                diagnostics.append(
                    {
                        "round": round_index,
                        "baseline_id": preset,
                        "planned": len(plan.candidates),
                        "backward_ablations": len(ablations),
                        "reserve_candidate_ids": plan.reserve_candidate_ids,
                        "permanently_pruned": [
                            item.__dict__ for item in plan.permanently_pruned
                        ],
                        "cap_exhausted": plan.cap_exhausted,
                        "wall_exhausted": plan.wall_exhausted,
                    }
                )
            known = {item.canonical_hash() for item in current.candidates}
            unique_additions = {
                item.canonical_hash(): item
                for item in additions
                if item.canonical_hash() not in known
            }
            runnable_additions, rejected = self._runnable(
                tuple(unique_additions.values())
            )
            diagnostics.extend(rejected)
            available = self.manifest.candidate_limit - len(current.candidates)
            additions = list(runnable_additions[: max(0, available)])
            if not additions:
                break
            added_stage = self._stage("f0", tuple(additions))
            checkpoint = self._run_stage(added_stage)
            current = CampaignStage(
                fidelity="f0",
                candidates=(*current.candidates, *additions),
                jobs=(*current.jobs, *added_stage.jobs),
            )
        return current, checkpoint, diagnostics

    def _promote(
        self,
        previous: CampaignStage,
        outcomes: dict[str, JobOutcome],
        *,
        fidelity: Fidelity,
        limit: int,
    ) -> CampaignStage:
        selected: list[CandidateSpec] = []
        control_only = tuple(
            preset
            for preset in self.manifest.baseline_presets
            if self.manifest.search_mode_for_preset(preset) == "control_only"
        )
        composable = tuple(
            preset
            for preset in self.manifest.baseline_presets
            if preset not in control_only
        )
        if limit < len(self.manifest.baseline_presets):
            raise ValueError("promotion limit must provide one slot per anchor")
        composable_budget = limit - len(control_only)
        base_limit, extra_slots = (
            divmod(composable_budget, len(composable)) if composable else (0, 0)
        )
        per_preset_limit = {
            **{preset: 1 for preset in control_only},
            **{
                preset: base_limit + (index < extra_slots)
                for index, preset in enumerate(composable)
            },
        }
        for offset, preset in enumerate(self.manifest.baseline_presets):
            candidates = tuple(
                item for item in previous.candidates if item.baseline_id == preset
            )
            if fidelity == "f2":
                candidates = tuple(
                    item
                    for item in candidates
                    if item.generation == "control"
                    or all(
                        self.catalog.techniques[technique_id].selection_safe
                        and (
                            not self.catalog.techniques[technique_id].fit_required
                            or technique_id in self.fit_capable_techniques
                        )
                        for technique_id in self._additions(item)
                    )
                )
            jobs = tuple(
                job
                for job in previous.jobs
                if job.candidate_hash in {item.canonical_hash() for item in candidates}
            )
            anchor_stage = CampaignStage(previous.fidelity, candidates, jobs)
            promoted = promote_stage(
                self.catalog,
                anchor_stage,
                outcomes,
                next_fidelity=fidelity,
                candidate_limit=min(per_preset_limit[preset], len(candidates)),
                exploration_fraction=self.manifest.exploration_fraction,
                seed=self.manifest.seeds[0] + offset,
                outer_fold_count=len(self.outer_folds),
                seeds=(self.manifest.seeds[0],)
                if fidelity == "f2"
                else self.manifest.seeds,
            )
            selected.extend(promoted.candidates)
        unique = {item.canonical_hash(): item for item in selected}
        return self._stage(
            fidelity, tuple(sorted(unique.values(), key=lambda item: item.candidate_id))
        )

    def _run_hpo(
        self, stage: CampaignStage, checkpoint: CampaignCheckpoint
    ) -> tuple[CampaignStage, CampaignCheckpoint, list[dict[str, object]]]:
        """Adapt BOHB from prior batches and prune each batch by seed budget."""

        diagnostics: list[dict[str, object]] = []
        root_evaluations = self._evaluations(stage, checkpoint.outcomes)
        family_roots: dict[tuple[str, tuple[str, ...]], CandidateSpec] = {}
        observations: dict[tuple[str, tuple[str, ...]], list[Observation]] = {}
        for candidate in stage.candidates:
            if candidate.generation == "control":
                continue
            evaluation = root_evaluations.get(candidate.candidate_id)
            if evaluation is None:
                continue
            key = (candidate.baseline_id, candidate.techniques)
            current = family_roots.get(key)
            if (
                current is None
                or evaluation.score > root_evaluations[current.candidate_id].score
            ):
                family_roots[key] = candidate
            observations.setdefault(key, []).append(
                Observation(candidate.parameters, evaluation.score)
            )

        total_trials = self.options.hpo_trials_per_structure
        if not family_roots or total_trials == 0:
            return stage, checkpoint, diagnostics
        batch_sizes = tuple(
            size
            for size in (
                math.ceil(total_trials / 2),
                total_trials // 2,
            )
            if size > 0
        )
        retained: list[CandidateSpec] = []
        accumulated_jobs: list[CampaignJob] = []
        known = {item.canonical_hash() for item in stage.candidates}

        for batch_index, batch_size in enumerate(batch_sizes):
            proposed: list[CandidateSpec] = []
            for key, root in family_roots.items():
                root_seed = int(root.canonical_hash()[:8], 16)
                for trial in range(batch_size):
                    parameters = suggest_for_combination(
                        self.search_space,
                        root.techniques,
                        tuple(observations[key]),
                        context=TuningContext(outer_fold=0, inner_fold=0),
                        seed=(
                            self.manifest.seeds[0]
                            + root_seed
                            + batch_index * 10_000
                            + trial
                        ),
                        exploration_fraction=self.manifest.exploration_fraction,
                    )
                    if not parameters:
                        continue
                    digest = hashlib.sha256(
                        repr((root.canonical_hash(), parameters)).encode()
                    ).hexdigest()[:10]
                    proposed.append(
                        root.model_copy(
                            update={
                                "candidate_id": f"{root.candidate_id}-hpo-{digest}",
                                "parameters": parameters,
                                "generation": "beam",
                            }
                        )
                    )
            unique = {
                item.canonical_hash(): item
                for item in proposed
                if item.canonical_hash() not in known
            }
            runnable, rejected = self._runnable(tuple(unique.values()))
            diagnostics.extend(rejected)
            if not runnable:
                continue

            survivors = runnable
            batch_jobs: list[CampaignJob] = []
            elimination_rungs = min(2, max(0, len(self.manifest.seeds) - 1))
            for rung, seed in enumerate(self.manifest.seeds):
                added = self._stage_with_seeds("f1", survivors, (seed,))
                checkpoint = self._run_stage(added)
                batch_jobs.extend(added.jobs)
                accumulated_jobs.extend(added.jobs)
                before = len(survivors)
                if rung < elimination_rungs:
                    survivor_hashes = {item.canonical_hash() for item in survivors}
                    rung_stage = CampaignStage(
                        "f1",
                        survivors,
                        tuple(
                            job
                            for job in batch_jobs
                            if job.candidate_hash in survivor_hashes
                        ),
                    )
                    evaluations = self._evaluations(rung_stage, checkpoint.outcomes)
                    grouped: dict[tuple[str, tuple[str, ...]], list[CandidateSpec]] = {}
                    for candidate in survivors:
                        grouped.setdefault(
                            (candidate.baseline_id, candidate.techniques), []
                        ).append(candidate)
                    selected: list[CandidateSpec] = []
                    for family in grouped.values():
                        ranked = sorted(
                            family,
                            key=lambda item: (
                                -evaluations[item.candidate_id].score,
                                item.candidate_id,
                            ),
                        )
                        selected.extend(ranked[: max(1, math.ceil(len(ranked) / 2))])
                    survivors = tuple(selected)
                diagnostics.append(
                    {
                        "adaptive_batch": batch_index,
                        "prior_observations": sum(
                            len(items) for items in observations.values()
                        ),
                        "rung": rung,
                        "seed_budget": rung + 1,
                        "evaluated": before,
                        "survivors": len(survivors),
                        "pruned": before - len(survivors),
                    }
                )

            survivor_hashes = {item.canonical_hash() for item in survivors}
            survivor_stage = CampaignStage(
                "f1",
                survivors,
                tuple(
                    job for job in batch_jobs if job.candidate_hash in survivor_hashes
                ),
            )
            survivor_evaluations = self._evaluations(
                survivor_stage, checkpoint.outcomes
            )
            for candidate in survivors:
                key = (candidate.baseline_id, candidate.techniques)
                observations[key].append(
                    Observation(
                        candidate.parameters,
                        survivor_evaluations[candidate.candidate_id].score,
                    )
                )
                known.add(candidate.canonical_hash())
            retained.extend(survivors)

        final_candidates = (*stage.candidates, *retained)
        final_hashes = {item.canonical_hash() for item in final_candidates}
        final_jobs = (
            *stage.jobs,
            *(job for job in accumulated_jobs if job.candidate_hash in final_hashes),
        )
        return (
            CampaignStage("f1", final_candidates, final_jobs),
            checkpoint,
            diagnostics,
        )

    def _safety(
        self, stage: CampaignStage, checkpoint: CampaignCheckpoint
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        evaluations = self._evaluations(stage, checkpoint.outcomes)
        _, analyses = self._evidence(stage, checkpoint.outcomes)
        champion_controls = tuple(
            item
            for item in stage.candidates
            if item.generation == "control" and item.baseline_id == CHAMPION_PRESET
        )
        if len(champion_controls) > 1:
            raise ValueError("campaign contains more than one frozen champion control")
        champion_evaluation = (
            evaluations.get(champion_controls[0].candidate_id)
            if champion_controls
            else None
        )
        records: list[dict[str, object]] = []
        eligible: list[
            tuple[CandidateSpec, CandidateEvaluation, dict[str, object]]
        ] = []
        for candidate in stage.candidates:
            candidate_hash = candidate.canonical_hash()
            confirmation_job_ids = tuple(
                job.job_id for job in stage.jobs if job.candidate_hash == candidate_hash
            )
            jobs_complete = bool(confirmation_job_ids) and all(
                job_id in checkpoint.outcomes
                and checkpoint.outcomes[job_id].state == "complete"
                for job_id in confirmation_job_ids
            )
            evaluation = evaluations.get(candidate.candidate_id)
            fit_receipts_verified = self._fit_receipts_complete(
                candidate, confirmation_job_ids, checkpoint.outcomes
            )
            champion_comparison = (
                self._champion_comparison(
                    evaluation,
                    champion_evaluation,
                    fit_receipts_verified=fit_receipts_verified,
                )
                if evaluation is not None
                and champion_evaluation is not None
                and candidate.baseline_id != CHAMPION_PRESET
                else None
            )
            if evaluation is None or not jobs_complete:
                classification = "evaluation_incomplete"
                reason = "one or more required jobs did not complete"
                analysis = None
            elif candidate.generation == "control":
                classification = "anchor_control"
                reason = "matched comparison anchor"
                analysis = None
            elif any(
                not self.catalog.techniques[item].selection_safe
                or (
                    self.catalog.techniques[item].fit_required
                    and item not in self.fit_capable_techniques
                )
                for item in self._additions(candidate)
            ):
                classification = "research_only_not_selection_safe"
                reason = "catalog selection_safe/fit_required gate failed"
                analysis = analyses.get(candidate.candidate_id)
            elif not fit_receipts_verified:
                classification = "fit_evidence_incomplete"
                reason = "fold-fitted candidate lacks complete fit receipts"
                analysis = analyses.get(candidate.candidate_id)
            else:
                analysis = analyses.get(candidate.candidate_id)
                scenario_deltas = (
                    analysis["scenario_deltas"] if isinstance(analysis, dict) else {}
                )
                if any(
                    float(delta) < -self.options.maximum_scenario_regression
                    for delta in scenario_deltas.values()
                ):
                    classification = "scenario_regression"
                    reason = "matched scenario regression exceeded the declared gate"
                elif (
                    not isinstance(analysis, dict)
                    or float(analysis["mean_delta"]) <= 0.0
                ):
                    classification = "no_confirmed_improvement"
                    reason = (
                        "independent development confirmation did not beat the "
                        "matched control"
                    )
                else:
                    classification = "proposal_eligible"
                    reason = (
                        "passed independent development confirmation and the "
                        "declared scenario gate; production promotion is still manual"
                    )
                    eligible.append((candidate, evaluation, analysis))
            records.append(
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "candidate_hash": candidate_hash,
                    "confirmation_job_ids": confirmation_job_ids,
                    "evaluation": evaluation.__dict__
                    if evaluation is not None
                    else None,
                    "analysis": analysis,
                    "champion_comparison": (
                        champion_comparison.model_dump(mode="json")
                        if champion_comparison is not None
                        else None
                    ),
                    "classification": classification,
                    "reason": reason,
                }
            )
        ranked = sorted(
            eligible,
            key=lambda item: (
                -item[1].score,
                item[0].complexity,
                item[0].candidate_id,
            ),
        )
        if ranked:
            proposal_baseline = ranked[0][0].baseline_id
            ranked = [
                item for item in ranked if item[0].baseline_id == proposal_baseline
            ]
        distinct: list[
            tuple[CandidateSpec, CandidateEvaluation, dict[str, object]]
        ] = []
        seen_behaviors: set[tuple[float, ...]] = set()
        for item in ranked:
            behavior = tuple(round(value, 12) for value in item[1].session_rewards)
            if behavior in seen_behaviors:
                continue
            seen_behaviors.add(behavior)
            distinct.append(item)
        if len(distinct) < 3:
            return records, []
        score_leader = distinct[0]
        remaining = distinct[1:]
        robust_leader = min(remaining, key=_robust_proposal_key)
        remaining = [item for item in remaining if item is not robust_leader]
        efficient_pool = [
            item for item in remaining if item[1].score >= score_leader[1].score - 0.02
        ] or remaining
        efficient_alternative = min(efficient_pool, key=_efficient_proposal_key)
        selected = (
            ("score_leader", score_leader),
            ("robust_leader", robust_leader),
            ("efficient_alternative", efficient_alternative),
        )
        proposals = []
        for role, (candidate, evaluation, analysis) in selected:
            record = next(
                item
                for item in records
                if _record_candidate_id(item) == candidate.candidate_id
            )
            proposals.append(
                {
                    "proposal_role": role,
                    "candidate_id": candidate.candidate_id,
                    "baseline_id": candidate.baseline_id,
                    "score": evaluation.score,
                    "mean_delta": analysis["mean_delta"],
                    "champion_comparison": record["champion_comparison"],
                    "classification": "package_eligible_proposal_only",
                    "candidate_hash": candidate.canonical_hash(),
                    "confirmation_job_ids": tuple(
                        job.job_id
                        for job in stage.jobs
                        if job.candidate_hash == candidate.canonical_hash()
                    ),
                }
            )
        return records, proposals

    def _leaderboard(
        self, stage: CampaignStage, outcomes: dict[str, JobOutcome]
    ) -> list[dict[str, object]]:
        evaluations = self._evaluations(stage, outcomes)
        return [
            {
                "rank": rank,
                "candidate": candidate.model_dump(mode="json"),
                "score": evaluation.score,
                "latency_p95_ms": evaluation.latency_p95_ms,
                "memory_mb": evaluation.memory_mb,
            }
            for rank, (candidate, evaluation) in enumerate(
                sorted(
                    (
                        (candidate, evaluations[candidate.candidate_id])
                        for candidate in stage.candidates
                        if candidate.candidate_id in evaluations
                    ),
                    key=lambda item: (
                        -item[1].score,
                        item[0].complexity,
                        item[0].candidate_id,
                    ),
                ),
                start=1,
            )
        ]

    def _write_evidence(self, payload: dict[str, object]) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.evidence_path.with_suffix(self.evidence_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.evidence_path)

    def _split_evidence(self) -> dict[str, object]:
        search_ids = {
            sample_id
            for fold in self.manifest.search_outer_folds
            for sample_id in self.outer_folds[fold]
        }
        confirmation_ids = {
            sample_id
            for fold in self.manifest.confirmation_outer_folds
            for sample_id in self.outer_folds[fold]
        }
        return {
            "search_outer_folds": self.manifest.search_outer_folds,
            "confirmation_outer_folds": self.manifest.confirmation_outer_folds,
            "search_sample_count": len(search_ids),
            "confirmation_sample_count": len(confirmation_ids),
            "search_sample_ids_sha256": sample_ids_hash(search_ids),
            "confirmation_sample_ids_sha256": sample_ids_hash(confirmation_ids),
            "overlap_count": len(search_ids & confirmation_ids),
            "f2_seeds": (self.manifest.seeds[0],),
        }

    def run(self) -> dict[str, object]:
        self._started = time.perf_counter()
        initial, exclusions = self._initial_candidates()
        f0 = self._stage("f0", initial)
        checkpoint = self._run_stage(f0)
        f0, checkpoint, interaction_diagnostics = self._expand_interactions(
            f0, checkpoint
        )
        f1_seed = self._promote(
            f0, checkpoint.outcomes, fidelity="f1", limit=self.options.f1_candidates
        )
        checkpoint = self._run_stage(f1_seed)
        f1, checkpoint, hpo_diagnostics = self._run_hpo(f1_seed, checkpoint)
        f2 = self._promote(
            f1, checkpoint.outcomes, fidelity="f2", limit=self.options.f2_candidates
        )
        checkpoint = self._run_stage(f2)
        safety, confirmed_top3 = self._safety(f2, checkpoint)
        confirmed_ids = tuple(item["candidate_id"] for item in confirmed_top3)
        payload: dict[str, object] = {
            "schema_version": 1,
            "campaign_id": self.manifest.campaign_id,
            "manifest_hash": self.manifest_hash,
            "parent_commit": self.manifest.parent_commit,
            "verified_input_hashes": self.verified_input_hashes,
            "protected_holdout_access": self.manifest.protected_holdout_access,
            "highest_fidelity": "f2",
            "selection_evidence_class": "prospective_disjoint_confirmation",
            "confirmation_status": "independent_development_confirmation",
            "split_evidence": self._split_evidence(),
            "independent_confirmation": {
                "status": "confirmed",
                "method": "prospective_disjoint_development_confirmation",
                "manifest_hash": self.manifest_hash,
                "candidate_ids": confirmed_ids,
            },
            "promotion_effect": "proposal_only_no_runtime_or_champion_mutation",
            "stage_counts": {
                "f0": len(f0.candidates),
                "f1": len(f1.candidates),
                "f2": len(f2.candidates),
            },
            "excluded": exclusions,
            "interaction_search": interaction_diagnostics,
            "conditional_hpo": {
                "classification": "adaptive_conditional_bohb_f1_racing",
                "space": self.manifest.search_space_path
                or "configs/search/wave2_weight_space_v1.json",
                "diagnostics": hpo_diagnostics,
            },
            "leaderboards": {
                "f0": self._leaderboard(f0, checkpoint.outcomes),
                "f1": self._leaderboard(f1, checkpoint.outcomes),
                "f2": self._leaderboard(f2, checkpoint.outcomes),
            },
            "safety": safety,
            "proposal": confirmed_top3[0] if confirmed_top3 else None,
            "confirmed_top3": confirmed_top3,
            "checkpoint": str(self.checkpoint_path),
            "elapsed_seconds": round(time.perf_counter() - self._started, 6),
        }
        self._write_evidence(payload)
        return payload
