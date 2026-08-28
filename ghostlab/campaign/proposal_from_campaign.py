from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ghostlab.campaign.analyze import CandidateEvaluation, PairedAnalysis
from ghostlab.campaign.bindings import ASSET_FIELDS, default_binding_registry
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.freeze import resolve_repository_path, sha256_file
from ghostlab.campaign.models import CampaignManifest, CandidateSpec, ChampionComparison
from ghostlab.campaign.proposal_materializer import (
    MaterializedProposalBundle,
    materialize_top_three,
)
from ghostlab.campaign.runner import CampaignCheckpoint
from ghostlab.campaign.top_three import CandidatePackage, select_top_three
from ghostlab.research.technique_suite import load_suite_config
from ghostlab.retrieval.residual import TECHNIQUE_ID as RESIDUAL_TECHNIQUE_ID

CONFIRMATION_STATUS = "independent_development_confirmation"
EVIDENCE_CLASS = "prospective_disjoint_confirmation"
CONFIRMATION_METHOD = "prospective_disjoint_development_confirmation"
ELIGIBLE_CLASSIFICATION = "proposal_eligible"
SUMMARY_CLASSIFICATION = "package_eligible_proposal_only"
_ALLOWED_EXTRAS = frozenset({"core", "gbdt", "dense", "neural", "all"})
CHAMPION_PRESET = "configs/suites/champion_guarded.json"


class SplitEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_outer_folds: tuple[int, ...]
    confirmation_outer_folds: tuple[int, ...]
    search_sample_count: int = Field(gt=0)
    confirmation_sample_count: int = Field(gt=0)
    search_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    overlap_count: int = Field(ge=0)
    f2_seeds: tuple[int, ...] = Field(min_length=1)


class IndependentConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    method: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: tuple[str, ...] = Field(min_length=3)


class ConfirmedProposalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    baseline_id: str
    score: float
    mean_delta: float
    classification: str
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_job_ids: tuple[str, ...] = Field(min_length=1)
    champion_comparison: ChampionComparison | None = None


def _json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _ids_hash(values: set[str]) -> str:
    encoded = json.dumps(
        sorted(values), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _validate_split_evidence(
    manifest: CampaignManifest,
    split: SplitEvidence,
    nested_path: Path,
) -> None:
    if sha256_file(nested_path) != manifest.nested_split_hash:
        raise ValueError("nested split hash does not match the frozen manifest")
    nested = _json_object(nested_path, "nested split")
    raw_folds = nested.get("outer_folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise TypeError("nested split outer_folds must be a non-empty list")
    search_folds = split.search_outer_folds
    confirmation_folds = split.confirmation_outer_folds
    _unique(search_folds, "search fold indices")
    _unique(confirmation_folds, "confirmation fold indices")
    all_indices = set(range(len(raw_folds)))
    if set(search_folds) & set(confirmation_folds):
        raise ValueError("search and confirmation folds overlap")
    if set(search_folds) | set(confirmation_folds) != all_indices:
        raise ValueError("search and confirmation folds must partition all outer folds")
    if split.overlap_count != 0:
        raise ValueError("confirmation evidence reports sample overlap")

    def sample_ids(indices: tuple[int, ...]) -> set[str]:
        result: set[str] = set()
        for index in indices:
            if index < 0 or index >= len(raw_folds):
                raise ValueError(f"outer fold index is out of range: {index}")
            fold = raw_folds[index]
            if not isinstance(fold, list) or not all(
                isinstance(item, str) for item in fold
            ):
                raise TypeError(f"nested outer fold {index} must contain string IDs")
            if len(fold) != len(set(fold)):
                raise ValueError(f"nested outer fold {index} contains duplicate IDs")
            result.update(fold)
        return result

    search_ids = sample_ids(search_folds)
    confirmation_ids = sample_ids(confirmation_folds)
    if search_ids & confirmation_ids:
        raise ValueError("recomputed search and confirmation sample IDs overlap")
    if len(search_ids) != split.search_sample_count:
        raise ValueError("search sample count does not match the nested split")
    if len(confirmation_ids) != split.confirmation_sample_count:
        raise ValueError("confirmation sample count does not match the nested split")
    if _ids_hash(search_ids) != split.search_sample_ids_sha256:
        raise ValueError("search sample hash does not match the nested split")
    if _ids_hash(confirmation_ids) != split.confirmation_sample_ids_sha256:
        raise ValueError("confirmation sample hash does not match the nested split")
    if len(split.f2_seeds) != 1 or split.f2_seeds[0] not in manifest.seeds:
        raise ValueError("confirmation must use exactly one frozen campaign seed")


def _job_id(candidate: CandidateSpec, fold: int, seed: int) -> str:
    encoded = f"{candidate.canonical_hash()}\0f2\0{fold}\0{seed}".encode()
    return f"job-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _evaluation(candidate: CandidateSpec, value: object) -> CandidateEvaluation:
    if not isinstance(value, dict):
        raise TypeError(f"missing evaluation for {candidate.candidate_id}")
    result = CandidateEvaluation(
        candidate_id=str(value.get("candidate_id", "")),
        complexity=int(value.get("complexity", -1)),
        score=float(value["score"]),
        session_rewards=tuple(float(item) for item in value["session_rewards"]),
        scenario_scores={
            str(name): float(score)
            for name, score in dict(value["scenario_scores"]).items()
        },
        latency_p95_ms=float(value.get("latency_p95_ms", 0.0)),
        memory_mb=float(value.get("memory_mb", 0.0)),
    )
    if result.candidate_id != candidate.candidate_id:
        raise ValueError("evaluation candidate ID does not match its candidate")
    if result.complexity != candidate.complexity:
        raise ValueError("evaluation complexity does not match its candidate")
    return result


def _analysis(candidate: CandidateSpec, value: object) -> PairedAnalysis:
    if not isinstance(value, dict):
        raise TypeError(f"missing paired analysis for {candidate.candidate_id}")
    result = PairedAnalysis(
        candidate_id=str(value.get("candidate_id", "")),
        baseline_id=str(value.get("baseline_id", "")),
        mean_delta=float(value["mean_delta"]),
        confidence_interval=tuple(float(item) for item in value["confidence_interval"]),  # type: ignore[arg-type]
        randomization_pvalue=float(value["randomization_pvalue"]),
        wins=int(value["wins"]),
        ties=int(value["ties"]),
        losses=int(value["losses"]),
        scenario_deltas={
            str(name): float(delta)
            for name, delta in dict(value["scenario_deltas"]).items()
        },
    )
    if result.candidate_id != candidate.candidate_id:
        raise ValueError("paired analysis candidate ID does not match its candidate")
    if len(result.confidence_interval) != 2:
        raise ValueError("paired confidence interval must contain two values")
    return result


def _materialize_config(
    manifest: CampaignManifest,
    candidate: CandidateSpec,
    project_root: Path,
):
    registry = default_binding_registry()
    baseline = load_suite_config(project_root / candidate.baseline_id)
    inherited = set(manifest.techniques_for_preset(candidate.baseline_id))
    additions = tuple(item for item in candidate.techniques if item not in inherited)
    patch_candidate = candidate.model_copy(update={"techniques": additions})
    return registry.materialize(baseline, patch_candidate)


def _candidate_assets(config: object) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(getattr(config, field))
                for field in ASSET_FIELDS
                if getattr(config, field, None) is not None
            }
        )
    )


def materialize_confirmed_campaign_top_three(
    *,
    project_root: str | Path,
    manifest_path: str,
    catalog_path: str,
    evidence_path: str,
    checkpoint_path: str,
    adaptive_split_path: str,
    nested_split_path: str,
    baseline_id: str,
    output_dir: str,
) -> MaterializedProposalBundle:
    """Materialize three development-confirmed proposals; never promote or access F3."""

    root = Path(project_root).resolve()
    manifest_file = resolve_repository_path(root, manifest_path, label="manifest")
    catalog_file = resolve_repository_path(root, catalog_path, label="catalog")
    evidence_file = resolve_repository_path(root, evidence_path, label="evidence")
    checkpoint_file = resolve_repository_path(root, checkpoint_path, label="checkpoint")
    adaptive_file = resolve_repository_path(
        root, adaptive_split_path, label="adaptive split"
    )
    nested_file = resolve_repository_path(root, nested_split_path, label="nested split")
    baseline_file = resolve_repository_path(root, baseline_id, label="baseline preset")

    manifest = CampaignManifest.model_validate_json(
        manifest_file.read_text(encoding="utf-8")
    )
    if baseline_id not in manifest.baseline_presets:
        raise ValueError("baseline preset is not declared by the frozen manifest")
    if sha256_file(adaptive_file) != manifest.adaptive_split_hash:
        raise ValueError("adaptive split hash does not match the frozen manifest")
    catalog = load_catalog(catalog_file)
    if catalog.content_hash != manifest.catalog_hash:
        raise ValueError("catalog hash does not match the frozen manifest")
    checkpoint = CampaignCheckpoint.model_validate_json(
        checkpoint_file.read_text(encoding="utf-8")
    )
    manifest_hash = manifest.canonical_hash()
    champion_comparison_required = CHAMPION_PRESET in manifest.baseline_presets
    if checkpoint.manifest_hash != manifest_hash:
        raise ValueError("checkpoint belongs to another frozen manifest")
    evidence = _json_object(evidence_file, "campaign evidence")
    if evidence.get("manifest_hash") != manifest_hash:
        raise ValueError("evidence belongs to another frozen manifest")
    if evidence.get("campaign_id") != manifest.campaign_id:
        raise ValueError("evidence campaign ID does not match the manifest")
    if evidence.get("parent_commit") != manifest.parent_commit:
        raise ValueError("evidence parent commit does not match the manifest")
    if evidence.get("protected_holdout_access") != "forbidden":
        raise ValueError("campaign evidence does not forbid protected holdout access")
    if evidence.get("highest_fidelity") != "f2":
        raise ValueError("campaign evidence is not complete through F2")
    if evidence.get("confirmation_status") != CONFIRMATION_STATUS:
        raise ValueError("independent development confirmation is absent")
    if evidence.get("selection_evidence_class") != EVIDENCE_CLASS:
        raise ValueError("campaign evidence is not prospective disjoint confirmation")

    split = SplitEvidence.model_validate(evidence.get("split_evidence"))
    _validate_split_evidence(manifest, split, nested_file)
    nested_payload = _json_object(nested_file, "nested split")
    raw_outer_folds = nested_payload["outer_folds"]
    assert isinstance(raw_outer_folds, list)
    outer_folds = tuple(
        tuple(str(sample_id) for sample_id in fold) for fold in raw_outer_folds
    )
    search_fit_ids = {
        sample_id
        for fold in split.search_outer_folds
        for sample_id in outer_folds[fold]
    }
    confirmation = IndependentConfirmation.model_validate(
        evidence.get("independent_confirmation")
    )
    if confirmation.status != "confirmed" or confirmation.method != CONFIRMATION_METHOD:
        raise ValueError("independent confirmation declaration is not accepted")
    if confirmation.manifest_hash != manifest_hash:
        raise ValueError("independent confirmation manifest hash mismatch")
    _unique(confirmation.candidate_ids, "confirmed candidate IDs")

    raw_summaries = evidence.get("confirmed_top3")
    if not isinstance(raw_summaries, list) or len(raw_summaries) < 3:
        raise ValueError("at least three confirmed proposal summaries are required")
    summaries = tuple(
        ConfirmedProposalSummary.model_validate(item) for item in raw_summaries
    )
    summary_by_id = {item.candidate_id: item for item in summaries}
    if len(summary_by_id) != len(summaries):
        raise ValueError("confirmed proposal summaries contain duplicate IDs")
    confirmed_ids = set(confirmation.candidate_ids)
    if set(summary_by_id) != confirmed_ids:
        raise ValueError("confirmation and proposal candidate IDs do not match")
    if any(item.classification != SUMMARY_CLASSIFICATION for item in summaries):
        raise ValueError("confirmed proposal summary is not package eligible")

    raw_safety = evidence.get("safety")
    if not isinstance(raw_safety, list):
        raise TypeError("campaign safety evidence must be a list")
    safety_by_id: dict[str, dict[str, object]] = {}
    for raw in raw_safety:
        if not isinstance(raw, dict) or not isinstance(raw.get("candidate"), dict):
            raise TypeError("invalid campaign safety record")
        candidate_id = str(raw["candidate"].get("candidate_id", ""))
        if candidate_id in confirmed_ids:
            if candidate_id in safety_by_id:
                raise ValueError("duplicate confirmed safety record")
            safety_by_id[candidate_id] = raw
    if set(safety_by_id) != confirmed_ids:
        raise ValueError("confirmed candidates are missing safety records")

    evaluations: list[CandidateEvaluation] = []
    analyses: dict[str, PairedAnalysis] = {}
    packages: dict[str, CandidatePackage] = {}
    analysis_baseline: str | None = None
    evidence_ref = evidence_file.relative_to(root).as_posix()
    checkpoint_ref = checkpoint_file.relative_to(root).as_posix()
    seed = split.f2_seeds[0]
    for candidate_id in sorted(confirmed_ids):
        record = safety_by_id[candidate_id]
        if record.get("classification") != ELIGIBLE_CLASSIFICATION:
            raise ValueError(
                f"candidate is not independently proposal eligible: {candidate_id}"
            )
        candidate = CandidateSpec.model_validate(record["candidate"])
        if (
            candidate.candidate_id != candidate_id
            or candidate.baseline_id != baseline_id
        ):
            raise ValueError("confirmed candidate baseline/ID mismatch")
        if candidate.generation == "control":
            raise ValueError("a matched control cannot be a proposal candidate")
        candidate_hash = candidate.canonical_hash()
        if record.get("candidate_hash") != candidate_hash:
            raise ValueError("confirmed safety candidate hash mismatch")
        if any(item not in catalog.techniques for item in candidate.techniques):
            raise ValueError("confirmed candidate contains an unknown technique")
        inherited = set(manifest.techniques_for_preset(candidate.baseline_id))
        additions = tuple(
            item for item in candidate.techniques if item not in inherited
        )
        if any(
            not catalog.techniques[item].selection_safe
            or (catalog.techniques[item].fit_required and item != RESIDUAL_TECHNIQUE_ID)
            for item in additions
        ):
            raise ValueError(
                "confirmed candidate contains an unsafe or unsupported fitted addition"
            )
        expected_job_ids = tuple(
            _job_id(candidate, fold, seed)
            for fold in sorted(split.confirmation_outer_folds)
        )
        raw_job_ids = record.get("confirmation_job_ids")
        if not isinstance(raw_job_ids, list) or tuple(raw_job_ids) != expected_job_ids:
            raise ValueError("confirmed safety job IDs do not match frozen folds/seed")
        residual_receipts = []
        for fold, job_id in zip(
            sorted(split.confirmation_outer_folds), expected_job_ids, strict=True
        ):
            outcome = checkpoint.outcomes.get(job_id)
            if outcome is None or outcome.state != "complete" or outcome.score is None:
                raise ValueError(
                    f"confirmed candidate lacks a complete confirmation checkpoint job: {candidate_id}"
                )
            if RESIDUAL_TECHNIQUE_ID in additions:
                matches = tuple(
                    item
                    for item in outcome.fit_receipts
                    if item.technique_id == RESIDUAL_TECHNIQUE_ID
                )
                if len(matches) != 1:
                    raise ValueError(
                        "residual proposal requires one fit receipt per confirmation job"
                    )
                receipt = matches[0]
                asset = resolve_repository_path(
                    root, receipt.asset_path, label="residual fitted asset"
                )
                if sha256_file(asset) != receipt.asset_sha256:
                    raise ValueError(
                        "residual fitted asset hash does not match receipt"
                    )
                if receipt.seed != seed:
                    raise ValueError("residual fit receipt uses another seed")
                if receipt.outer_fold != fold:
                    raise ValueError("residual fit receipt uses another outer fold")
                if receipt.train_sample_ids_sha256 != _ids_hash(search_fit_ids):
                    raise ValueError("residual fit did not use the frozen search IDs")
                if receipt.validation_sample_ids_sha256 != _ids_hash(
                    set(outer_folds[fold])
                ):
                    raise ValueError(
                        "residual fit receipt validation IDs do not match its fold"
                    )
                residual_receipts.append(receipt)
        evaluation = _evaluation(candidate, record.get("evaluation"))
        paired = _analysis(candidate, record.get("analysis"))
        if analysis_baseline is None:
            analysis_baseline = paired.baseline_id
        elif paired.baseline_id != analysis_baseline:
            raise ValueError("confirmed candidates do not share a matched control")
        summary = summary_by_id[candidate_id]
        detailed_champion = (
            ChampionComparison.model_validate(record["champion_comparison"])
            if record.get("champion_comparison") is not None
            else None
        )
        if champion_comparison_required and detailed_champion is None:
            raise ValueError(
                "confirmed proposal is missing its same-fold champion comparison"
            )
        if (
            summary.baseline_id != baseline_id
            or summary.candidate_hash != candidate_hash
            or summary.confirmation_job_ids != expected_job_ids
            or abs(summary.score - evaluation.score) > 1e-12
            or abs(summary.mean_delta - paired.mean_delta) > 1e-12
            or summary.champion_comparison != detailed_champion
        ):
            raise ValueError(
                "confirmed proposal summary does not match detailed evidence"
            )
        config = _materialize_config(manifest, candidate, root)
        if residual_receipts:
            # Both confirmation jobs use the same frozen search partition and
            # hyperparameters. Keep one independently confirmed fold-fitted asset
            # as the reproducible proposal runtime asset.
            config = config.model_copy(
                update={"residual_model_asset": residual_receipts[0].asset_path}
            )
        assets = _candidate_assets(config)
        extras = tuple(
            sorted(
                {
                    catalog.techniques[item].execution_class
                    for item in candidate.techniques
                }
                or {"core"}
            )
        )
        if not set(extras) <= _ALLOWED_EXTRAS:
            raise ValueError(f"candidate has unsupported dependency extras: {extras}")
        evaluations.append(evaluation)
        analyses[candidate_id] = paired
        packages[candidate_id] = CandidatePackage(
            candidate_id=candidate_id,
            config=config,
            dependency_extras=extras,
            assets=assets,
            evidence_refs=(evidence_ref, checkpoint_ref),
            confirmed=True,
            safe=True,
            notes=(
                "development-confirmed on prospectively disjoint public folds; not final generalization proof",
            ),
            enabled_techniques=candidate.techniques,
            technique_sources=tuple(
                (
                    item,
                    catalog.techniques[item].source or "not_declared",
                    default_binding_registry().bindings[item].reason,
                )
                for item in candidate.techniques
            ),
            tuned_parameters=candidate.parameters,
            champion_comparison=detailed_champion,
        )

    assert analysis_baseline is not None
    selection = select_top_three(
        tuple(evaluations),
        analyses,
        packages,
        baseline_id=analysis_baseline,
        project_root=root,
        minimum_mean_delta=1e-12,
    )
    return materialize_top_three(
        selection,
        project_root=root,
        output_dir=root / output_dir,
        baseline_config_path=baseline_file.relative_to(root).as_posix(),
        split_path=adaptive_file.relative_to(root).as_posix(),
        rollback_commit=manifest.parent_commit,
    )
