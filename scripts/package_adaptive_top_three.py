from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config

ROOT = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_top_three(
    campaign_report_path: Path,
    base_config_path: Path,
    technique_catalog_path: Path,
    output_dir: Path,
    report_path: Path,
    lineage_manifest_path: Path | None = None,
) -> dict[str, Any]:
    campaign = json.loads(campaign_report_path.read_text(encoding="utf-8"))
    if campaign.get("mode") != "race":
        raise ValueError("Top-three packaging requires a completed race report")
    records = campaign.get("records", {}).get("f2", [])
    if not isinstance(records, list) or not records:
        raise ValueError("campaign report contains no F2 evaluations")

    baseline = load_adaptive_hybrid_config(base_config_path)
    registry = AdaptiveTechniqueRegistry.from_catalog(
        load_catalog(technique_catalog_path), project_root=ROOT
    )
    challengers = [
        item
        for item in records
        if item.get("candidate", {}).get("generation") != "control"
    ]
    challengers.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["candidate"].get("complexity", 0)),
            str(item["candidate"]["candidate_id"]),
        )
    )
    selected = challengers[:3]
    if not selected:
        raise ValueError("campaign F2 stage contains no challenger")

    output_dir.mkdir(parents=True, exist_ok=True)
    finalists: list[dict[str, Any]] = []
    for rank, record in enumerate(selected, start=1):
        candidate = CandidateSpec.model_validate(record["candidate"])
        config = registry.materialize(baseline, candidate)
        config_path = output_dir / f"rank_{rank}_{candidate.candidate_id}.json"
        config_path.write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        relative_config = config_path.relative_to(ROOT).as_posix()
        candidate_report = (
            f"artifacts/reports/finalists/{candidate.candidate_id}_evaluation.json"
        )
        validation_report = (
            f"artifacts/reports/finalists/{candidate.candidate_id}_validation.json"
        )
        eligible = (
            record.get("decision") == "PROMOTE"
            and not record.get("gate_failures")
            and int(record.get("constraint_violations", 0)) == 0
            and (
                not bool(record.get("fit_required")) or bool(record.get("fit_verified"))
            )
        )
        digest = _file_sha256(config_path)
        evaluation_command = (
            "PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid.py "
            f"--config {relative_config} "
            "--dataset data/public_set.jsonl "
            "--dataset data/synthetic_1000_public_like.jsonl "
            "--dataset data/independent_template_1000.jsonl "
            "--lineage-manifest "
            "data/splits/adaptive_hybrid_lineage_75_25_v1.json "
            "--partition development "
            f"--output {candidate_report}"
        )
        validation_command = (
            "PYTHONPATH=. .venv/bin/python scripts/validate_adaptive_hybrid.py "
            f"--config {relative_config} --adaptive-report {candidate_report} "
            "--training-report artifacts/reports/"
            "adaptive_hybrid_training_2200_structural_v2.json "
            f"--output {validation_report}"
        )
        activation_command = (
            "PYTHONPATH=. .venv/bin/python scripts/activate_adaptive_candidate.py "
            f"--preset {relative_config} --expected-sha256 {digest} "
            f"--top3-report {report_path.relative_to(ROOT).as_posix()} "
            "--holdout-report artifacts/reports/adaptive_final_holdout.json"
        )
        finalists.append(
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "config_path": relative_config,
                "config_sha256": digest,
                "techniques": list(candidate.techniques),
                "parameters": dict(candidate.parameters),
                "metrics": {
                    key: record.get(key)
                    for key in (
                        "score",
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "mean_paired_delta",
                        "latency_p95_ms",
                    )
                },
                "campaign_decision": record.get("decision"),
                "promotion_eligible": eligible,
                "gate_failures": record.get("gate_failures", []),
                "constraint_violations": record.get("constraint_violations", 0),
                "commands": {
                    "evaluate": evaluation_command,
                    "validate": validation_command,
                    "activate_after_validation": activation_command,
                    "verify_active": (
                        "PYTHONPATH=. .venv/bin/python "
                        "scripts/verify_active_candidate.py"
                    ),
                    "test": "PYTHONPATH=. .venv/bin/pytest -q",
                },
            }
        )

    recommended = next((item for item in finalists if item["promotion_eligible"]), None)
    manifest_hash = (
        _file_sha256(lineage_manifest_path)
        if lineage_manifest_path is not None and lineage_manifest_path.is_file()
        else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "architecture": baseline.architecture,
        "campaign_report": campaign_report_path.relative_to(ROOT).as_posix(),
        "requested_challenger_count": 3,
        "packaged_challenger_count": len(finalists),
        "recommended_candidate_id": (
            recommended["candidate_id"] if recommended is not None else None
        ),
        "recommendation": (
            "freeze_one_for_single_holdout_evaluation"
            if recommended is not None
            else "retain_current_champion"
        ),
        "automatic_activation": False,
        "selection_evidence": {
            "datasets": campaign.get("dataset_sources", []),
            "sample_count": campaign.get("sample_count"),
            "selection_data_held_out": False,
            "untouched_holdout_exists": True,
            "nested_oof_is_permanent_holdout": False,
            "f3_accessed": False,
            "note": (
                "Only the 1,650-session development partition may participate in "
                "fitting and campaign selection. The 550-session holdout remains "
                "untouched until exactly one challenger is frozen."
            ),
        },
        "frozen_proposal": (
            {
                "candidate_id": recommended["candidate_id"],
                "config_path": recommended["config_path"],
                "config_sha256": recommended["config_sha256"],
                "lineage_manifest_sha256": manifest_hash,
                "holdout_accessed": False,
            }
            if recommended is not None
            else None
        ),
        "promotion_process": [
            "Choose and freeze exactly one eligible finalist using development evidence only.",
            (
                "Evaluate frozen A/B references plus only the frozen finalist and "
                "frozen control once on the 550-session holdout; promotion remains "
                "C versus D only."
            ),
            "Require the predeclared holdout gates and artifact hashes to pass.",
            "Run activation with that holdout report; activation is never automatic.",
            "Run verify_active and the full test-suite command.",
        ],
        "finalists": finalists,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize and document the top three adaptive challengers"
    )
    parser.add_argument(
        "--campaign-report",
        default="artifacts/reports/adaptive_hybrid_campaign_1650.json",
    )
    parser.add_argument(
        "--base-config",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument(
        "--technique-catalog", default="configs/techniques/catalog_v2.json"
    )
    parser.add_argument(
        "--output-dir", default="configs/finalists/adaptive_hybrid_1650"
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    args = parser.parse_args()
    report = package_top_three(
        ROOT / args.campaign_report,
        ROOT / args.base_config,
        ROOT / args.technique_catalog,
        ROOT / args.output_dir,
        ROOT / args.output,
        ROOT / args.lineage_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
