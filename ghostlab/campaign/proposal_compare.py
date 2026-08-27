from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from ghostlab.campaign.analyze import CandidateEvaluation, paired_analysis
from ghostlab.research.replay import session_reward


def compare_proposal_reports(
    baseline_path: str | Path,
    candidate_paths: Mapping[str, str | Path],
) -> dict[str, object]:
    if not candidate_paths or len(candidate_paths) > 3:
        raise ValueError("comparison requires one to three candidate reports")
    baseline = _load_report(baseline_path)
    baseline_evaluation = _evaluation("baseline", baseline)
    baseline_ids = tuple(str(item["sample_id"]) for item in baseline["sessions"])
    results: dict[str, object] = {}
    for role, path in sorted(candidate_paths.items()):
        candidate = _load_report(path)
        candidate_ids = tuple(str(item["sample_id"]) for item in candidate["sessions"])
        if candidate_ids != baseline_ids:
            raise ValueError(f"candidate report is not paired in baseline order: {role}")
        evaluation = _evaluation(role, candidate)
        analysis = paired_analysis(evaluation, baseline_evaluation)
        results[role] = {
            "score": evaluation.score,
            "mean_paired_delta": analysis.mean_delta,
            "paired_confidence_interval": list(analysis.confidence_interval),
            "randomization_pvalue": analysis.randomization_pvalue,
            "wins": analysis.wins,
            "ties": analysis.ties,
            "losses": analysis.losses,
            "scenario_deltas": analysis.scenario_deltas,
        }
    return {
        "schema_version": 1,
        "baseline_score": baseline_evaluation.score,
        "sample_count": len(baseline_ids),
        "candidates": results,
        "decision_boundary": "comparison only; no automatic promotion or F3 access",
    }


def _load_report(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sessions"), list):
        raise TypeError(f"invalid unified preset report: {path}")
    if not value["sessions"]:
        raise ValueError(f"unified preset report is empty: {path}")
    return value


def _evaluation(identifier: str, report: dict) -> CandidateEvaluation:
    rewards = tuple(session_reward(item) for item in report["sessions"])
    grouped: dict[str, list[float]] = defaultdict(list)
    for session, reward in zip(report["sessions"], rewards, strict=True):
        grouped[str(session["scenario_type"])].append(reward)
    return CandidateEvaluation(
        candidate_id=identifier,
        complexity=0,
        score=statistics.fmean(rewards),
        session_rewards=rewards,
        scenario_scores={
            name: statistics.fmean(values) for name, values in sorted(grouped.items())
        },
    )
