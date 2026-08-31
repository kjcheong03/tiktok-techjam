from __future__ import annotations

from dataclasses import dataclass

from ghostlab.campaign.models import CampaignJob, CampaignResources, ResourceRequest


@dataclass(frozen=True)
class ScheduledWave:
    jobs: tuple[CampaignJob, ...]


def _fits(
    used: ResourceRequest,
    job: ResourceRequest,
    limits: CampaignResources,
    heavy_count: int,
) -> bool:
    return (
        used.cpu + job.cpu <= limits.cpu_jobs
        and used.gpu + job.gpu <= limits.gpu_jobs
        and used.memory_gb + job.memory_gb <= limits.memory_gb
        and heavy_count + int(job.heavy_model) <= limits.heavy_model_jobs
    )


def schedule_waves(
    jobs: tuple[CampaignJob, ...], limits: CampaignResources
) -> tuple[ScheduledWave, ...]:
    """Produce deterministic resource-safe batches without starting processes."""

    remaining = sorted(jobs, key=lambda item: item.job_id)
    waves: list[ScheduledWave] = []
    while remaining:
        selected: list[CampaignJob] = []
        deferred: list[CampaignJob] = []
        used = ResourceRequest(cpu=0, gpu=0, memory_gb=0.0)
        heavy_count = 0
        for job in remaining:
            request = job.resources
            if _fits(used, request, limits, heavy_count):
                selected.append(job)
                used = ResourceRequest(
                    cpu=used.cpu + request.cpu,
                    gpu=used.gpu + request.gpu,
                    memory_gb=used.memory_gb + request.memory_gb,
                )
                heavy_count += int(request.heavy_model)
            else:
                deferred.append(job)
        if not selected:
            job = remaining[0]
            raise ValueError(f"job exceeds campaign resource limits: {job.job_id}")
        waves.append(ScheduledWave(tuple(selected)))
        remaining = deferred
    return tuple(waves)
