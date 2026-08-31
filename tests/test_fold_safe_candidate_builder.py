from __future__ import annotations

from pathlib import Path

from ghostlab.campaign.models import CampaignJob, CandidateSpec
from ghostlab.research.technique_suite import UnifiedTechniqueConfig
from ghostlab.retrieval.residual import TECHNIQUE_ID
from ghostlab.training.campaign import FoldSafeCandidateBuilder


def _candidate() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="residual",
        baseline_id="configs/suites/unfitted_keyword_search.json",
        techniques=("retrieval.sparse", TECHNIQUE_ID),
        complexity=1,
        generation="single",
    )


def _builder(tmp_path: Path) -> FoldSafeCandidateBuilder:
    return FoldSafeCandidateBuilder(
        materialize=lambda _: UnifiedTechniqueConfig(residual_reranker_enabled=True),
        dataset_path=tmp_path / "development.jsonl",
        catalog_path=tmp_path / "catalog.jsonl",
        outer_folds=(("a", "b"), ("c",), ("d",), ("e",), ("f",)),
        search_outer_folds=(0, 2, 3),
        confirmation_outer_folds=(1, 4),
        campaign_id="fit-test",
        artifact_root=tmp_path / "fits",
    )


def test_search_fit_excludes_entire_validation_fold(tmp_path: Path) -> None:
    candidate = _candidate()
    request = _builder(tmp_path)._fit_request(
        candidate,
        CampaignJob(
            job_id="job-search",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f1",
            seed=7,
        ),
        0,
    )

    assert request.train_sample_ids == ("d", "e")
    assert request.validation_sample_ids == ("a", "b")
    assert not set(request.train_sample_ids) & set(request.validation_sample_ids)


def test_confirmation_fit_uses_only_frozen_search_partition(tmp_path: Path) -> None:
    candidate = _candidate()
    request = _builder(tmp_path)._fit_request(
        candidate,
        CampaignJob(
            job_id="job-confirm",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f2",
            outer_fold=4,
            seed=11,
        ),
        4,
    )

    assert request.train_sample_ids == ("a", "b", "d", "e")
    assert request.validation_sample_ids == ("f",)
    assert "c" not in request.train_sample_ids


def test_fit_asset_identity_changes_with_candidate_parameters(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    candidate = _candidate()
    tuned = candidate.model_copy(
        update={"parameters": (("residual_regularization", 0.5),)}
    )
    job = CampaignJob(
        job_id="job-search",
        candidate_hash=candidate.canonical_hash(),
        fidelity="f0",
        seed=7,
    )

    assert builder._asset_path(candidate, job, 0) != builder._asset_path(
        tuned,
        job.model_copy(update={"candidate_hash": tuned.canonical_hash()}),
        0,
    )
