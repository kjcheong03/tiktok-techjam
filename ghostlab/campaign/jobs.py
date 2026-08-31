from __future__ import annotations

import hashlib

from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.models import (
    CampaignJob,
    CandidateSpec,
    Fidelity,
    ResourceRequest,
)


def candidate_resources(
    catalog: TechniqueCatalog, candidate: CandidateSpec
) -> ResourceRequest:
    requests = [catalog.techniques[item].resources for item in candidate.techniques]
    if not requests:
        return ResourceRequest()
    return ResourceRequest(
        cpu=max(item.cpu for item in requests),
        gpu=max(item.gpu for item in requests),
        memory_gb=sum(item.memory_gb for item in requests),
        heavy_model=any(item.heavy_model for item in requests),
    )


def build_jobs(
    catalog: TechniqueCatalog,
    candidates: tuple[CandidateSpec, ...],
    *,
    fidelity: Fidelity,
    outer_fold_count: int,
    seeds: tuple[int, ...],
) -> tuple[CampaignJob, ...]:
    if outer_fold_count <= 0 or not seeds:
        raise ValueError("job construction requires folds and seeds")
    folds: tuple[int | None, ...] = (
        tuple(range(outer_fold_count)) if fidelity == "f2" else (None,)
    )
    jobs: list[CampaignJob] = []
    for candidate in sorted(candidates, key=lambda item: item.canonical_hash()):
        candidate_hash = candidate.canonical_hash()
        resources = candidate_resources(catalog, candidate)
        for seed in seeds:
            for fold in folds:
                encoded = f"{candidate_hash}\0{fidelity}\0{fold}\0{seed}".encode()
                jobs.append(
                    CampaignJob(
                        job_id=f"job-{hashlib.sha256(encoded).hexdigest()[:16]}",
                        candidate_hash=candidate_hash,
                        fidelity=fidelity,
                        outer_fold=fold,
                        seed=seed,
                        resources=resources,
                    )
                )
    return tuple(jobs)
