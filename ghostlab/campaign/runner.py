from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ghostlab.campaign.models import CampaignJob, CampaignResources, JobOutcome
from ghostlab.campaign.scheduler import schedule_waves

Evaluator = Callable[[CampaignJob], JobOutcome]


class CampaignCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    manifest_hash: str
    outcomes: dict[str, JobOutcome] = Field(default_factory=dict)


def _save_checkpoint(path: Path, checkpoint: CampaignCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_progress(
    path: Path,
    *,
    jobs: tuple[CampaignJob, ...],
    outcomes: dict[str, JobOutcome],
    stage: str | None = None,
    control_path: Path | None = None,
) -> None:
    relevant = [outcomes[job.job_id] for job in jobs if job.job_id in outcomes]
    complete = [item for item in relevant if item.state == "complete"]
    best = max(
        (item for item in complete if item.score is not None),
        key=lambda item: item.score if item.score is not None else float("-inf"),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "stage": stage,
        "total_jobs": len(jobs),
        "recorded": len(relevant),
        "complete": len(complete),
        "failed": sum(item.state == "failed" for item in relevant),
        "highest_individual_job": None
        if best is None
        else {
            "job_id": best.job_id,
            "score": best.score,
            "scenario_scores": best.scenario_scores,
        },
    }
    if control_path is not None:
        from ghostlab.campaign.control import load_campaign_control

        payload["control"] = load_campaign_control(control_path).model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_checkpoint(path: Path, manifest_hash: str) -> CampaignCheckpoint:
    if not path.exists():
        return CampaignCheckpoint(manifest_hash=manifest_hash)
    checkpoint = CampaignCheckpoint.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if checkpoint.manifest_hash != manifest_hash:
        raise ValueError("checkpoint belongs to a different campaign manifest")
    return checkpoint


def run_jobs(
    jobs: tuple[CampaignJob, ...],
    *,
    manifest_hash: str,
    resources: CampaignResources,
    checkpoint_path: Path,
    evaluator: Evaluator,
    progress_path: Path | None = None,
    stage: str | None = None,
    control_path: Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> CampaignCheckpoint:
    """Run deterministic resource batches and atomically checkpoint every outcome."""

    checkpoint = load_checkpoint(checkpoint_path, manifest_hash)
    outcomes = dict(checkpoint.outcomes)
    pending = tuple(job for job in jobs if job.job_id not in outcomes)
    if progress_path is not None:
        _save_progress(
            progress_path,
            jobs=jobs,
            outcomes=outcomes,
            stage=stage,
            control_path=control_path,
        )
    for wave in schedule_waves(pending, resources):
        if stop_requested is not None and stop_requested():
            break

        def evaluate_one(job: CampaignJob) -> JobOutcome:
            try:
                result = evaluator(job)
                if result.job_id != job.job_id:
                    raise ValueError("evaluator returned an outcome for another job")
                return result
            except Exception as error:  # noqa: BLE001 - worker boundary records failure
                return JobOutcome(
                    job_id=job.job_id,
                    state="failed",
                    error=f"{type(error).__name__}: {error}",
                )

        with ThreadPoolExecutor(max_workers=len(wave.jobs)) as executor:
            futures = {
                job.job_id: executor.submit(evaluate_one, job) for job in wave.jobs
            }
            wave_outcomes = {
                job_id: future.result() for job_id, future in futures.items()
            }
        for job in wave.jobs:
            outcome = wave_outcomes[job.job_id]
            outcomes[job.job_id] = outcome
            checkpoint = CampaignCheckpoint(
                manifest_hash=manifest_hash, outcomes=outcomes
            )
            _save_checkpoint(checkpoint_path, checkpoint)
            if progress_path is not None:
                _save_progress(
                    progress_path,
                    jobs=jobs,
                    outcomes=outcomes,
                    stage=stage,
                    control_path=control_path,
                )
        if stop_requested is not None and stop_requested():
            break
    return checkpoint
