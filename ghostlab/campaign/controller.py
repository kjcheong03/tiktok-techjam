from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.jobs import build_jobs
from ghostlab.campaign.models import (
    CampaignJob,
    CampaignManifest,
    CandidateSpec,
    Fidelity,
    JobOutcome,
)
from ghostlab.campaign.planner import CandidatePlan, plan_candidates


@dataclass(frozen=True)
class CampaignStage:
    fidelity: Fidelity
    candidates: tuple[CandidateSpec, ...]
    jobs: tuple[CampaignJob, ...]


def initial_stage(
    catalog: TechniqueCatalog, manifest: CampaignManifest
) -> tuple[CandidatePlan, CampaignStage]:
    anchor_count = len(manifest.baseline_presets)
    per_anchor_limit = max(1, manifest.candidate_limit // anchor_count)
    plans = tuple(
        plan_candidates(
            catalog,
            baseline_id=preset,
            baseline_techniques=manifest.techniques_for_preset(preset),
            technique_ids=(
                manifest.techniques_for_preset(preset)
                if manifest.search_mode_for_preset(preset) == "control_only"
                else manifest.technique_ids
            ),
            max_order=manifest.max_order,
            candidate_limit=per_anchor_limit,
        )
        for preset in manifest.baseline_presets
    )
    plan = CandidatePlan(
        candidates=tuple(
            candidate for anchor_plan in plans for candidate in anchor_plan.candidates
        )[: manifest.candidate_limit],
        skipped=tuple(item for anchor_plan in plans for item in anchor_plan.skipped),
    )
    stage = CampaignStage(
        fidelity="f0",
        candidates=plan.candidates,
        jobs=build_jobs(
            catalog,
            plan.candidates,
            fidelity="f0",
            outer_fold_count=5,
            seeds=manifest.seeds,
        ),
    )
    return plan, stage


def promote_stage(
    catalog: TechniqueCatalog,
    previous: CampaignStage,
    outcomes: dict[str, JobOutcome],
    *,
    next_fidelity: Fidelity,
    candidate_limit: int,
    exploration_fraction: float,
    seed: int,
    outer_fold_count: int,
    seeds: tuple[int, ...],
) -> CampaignStage:
    """Promote by mean score while reserving a deterministic pruning-audit sample."""

    if candidate_limit <= 0:
        raise ValueError("promotion candidate limit must be positive")
    candidate_by_hash = {item.canonical_hash(): item for item in previous.candidates}
    grouped: dict[str, list[float]] = {}
    for job in previous.jobs:
        outcome = outcomes.get(job.job_id)
        if outcome is None or outcome.state != "complete" or outcome.score is None:
            continue
        grouped.setdefault(job.candidate_hash, []).append(outcome.score)
    ranked = sorted(
        (
            (candidate_by_hash[key], statistics.fmean(values))
            for key, values in grouped.items()
            if key in candidate_by_hash
        ),
        key=lambda item: (-item[1], item[0].complexity, item[0].candidate_id),
    )
    if not ranked:
        raise ValueError("no completed candidate outcomes are available for promotion")
    limit = min(candidate_limit, len(ranked))
    audit_count = min(limit - 1, round(limit * exploration_fraction))
    exploit_count = limit - audit_count
    selected = [item[0] for item in ranked[:exploit_count]]
    remainder = [item[0] for item in ranked[exploit_count:]]
    rng = random.Random(seed)
    rng.shuffle(remainder)
    selected.extend(remainder[:audit_count])
    ordered = tuple(sorted(selected, key=lambda item: item.canonical_hash()))
    return CampaignStage(
        fidelity=next_fidelity,
        candidates=ordered,
        jobs=build_jobs(
            catalog,
            ordered,
            fidelity=next_fidelity,
            outer_fold_count=outer_fold_count,
            seeds=seeds,
        ),
    )
