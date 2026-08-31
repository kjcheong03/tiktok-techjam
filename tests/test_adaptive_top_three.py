from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from scripts.package_adaptive_top_three import package_top_three

ROOT = Path(__file__).resolve().parents[1]


def test_packages_three_ranked_challengers_without_activation() -> None:
    registry = AdaptiveTechniqueRegistry.from_catalog(
        load_catalog(ROOT / "configs/techniques/catalog_v2.json"), project_root=ROOT
    )
    compulsory = registry.inventory().compulsory
    records = []
    for ordinal, score in enumerate((0.83, 0.82, 0.81, 0.80), start=1):
        candidate = CandidateSpec(
            candidate_id=f"challenger-test-{ordinal}",
            baseline_id="adaptive_hybrid_1a_3b_v1",
            techniques=compulsory,
            parameters=(("router_abstain_confidence", 0.5 + ordinal * 0.01),),
            complexity=0,
            generation="single",
        )
        records.append(
            {
                "candidate": candidate.model_dump(mode="json"),
                "score": score,
                "hit_rate_at_10": 0.9,
                "mrr": 0.7,
                "mttc": 2.0,
                "decision": "PROMOTE",
                "fit_required": [],
                "fit_verified": True,
                "gate_failures": [],
                "mean_paired_delta": score - 0.79,
                "latency_p95_ms": 12.0,
                "constraint_violations": 0,
            }
        )
    campaign = {
        "schema_version": 2,
        "mode": "race",
        "dataset_sources": [{"path": "data/public_set.jsonl", "count": 200}],
        "sample_count": 2200,
        "records": {"f2": records},
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
        temp = Path(temporary)
        campaign_path = temp / "campaign.json"
        report_path = temp / "top3.json"
        output_dir = temp / "configs"
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        report = package_top_three(
            campaign_path,
            ROOT / "configs/adaptive_hybrid_1a_3b_v1.json",
            ROOT / "configs/techniques/catalog_v2.json",
            output_dir,
            report_path,
            ROOT / "data/splits/adaptive_hybrid_lineage_75_25_v1.json",
        )

        assert report["packaged_challenger_count"] == 3
        assert report["recommended_candidate_id"] == "challenger-test-1"
        assert report["automatic_activation"] is False
        assert report["selection_evidence"]["selection_data_held_out"] is False
        assert report["selection_evidence"][
            "one_time_final_selection_set_exists"
        ] is True
        assert [item["candidate_id"] for item in report["frozen_proposals"]] == [
            "challenger-test-1",
            "challenger-test-2",
            "challenger-test-3",
        ]
        assert all(
            item["final_selection_accessed"] is False
            for item in report["frozen_proposals"]
        )
        frozen = report["frozen_dependencies"]
        control = ROOT / frozen["control_config_path"]
        assert frozen["control_config_file_sha256"] == hashlib.sha256(
            control.read_bytes()
        ).hexdigest()
        assert frozen["control_config_canonical_sha256"]
        assert frozen["reference_a_implementation_sha256"]
        assert "reference_b_config_sha256" not in frozen
        assert frozen["gates_sha256"]
        assert report["selection_rule"]["no_post_selection_tuning"] is True
        assert len(report["selection_rule"]["tie_break_order"]) == 7
        validation = report["finalists"][0]["commands"]["validate"]
        assert "adaptive_hybrid_training_1650_final_v1.json" in validation
        assert "training_2200" not in validation
        assert all(item["promotion_eligible"] for item in report["finalists"])
        assert all(
            (ROOT / item["config_path"]).is_file() for item in report["finalists"]
        )
        assert all(
            "activate_adaptive_candidate.py"
            in item["commands"]["activate_after_validation"]
            for item in report["finalists"]
        )
        assert all(
            "--holdout-report" in item["commands"]["activate_after_validation"]
            for item in report["finalists"]
        )


def test_packages_one_positive_gate_clean_hold_candidate() -> None:
    registry = AdaptiveTechniqueRegistry.from_catalog(
        load_catalog(ROOT / "configs/techniques/catalog_v2.json"), project_root=ROOT
    )
    candidate = CandidateSpec(
        candidate_id="challenger-single-positive",
        baseline_id="adaptive_hybrid_1a_3b_v1",
        techniques=registry.inventory().compulsory,
        parameters=(("router_abstain_confidence", 0.61),),
        complexity=0,
        generation="single",
    )
    campaign = {
        "schema_version": 2,
        "mode": "race",
        "dataset_sources": [{"path": "data/public_set.jsonl", "count": 200}],
        "sample_count": 1650,
        "records": {
            "f2": [
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "score": 0.857,
                    "hit_rate_at_10": 0.96,
                    "mrr": 0.70,
                    "mttc": 2.75,
                    "decision": "HOLD_MORE_DATA",
                    "fit_required": [],
                    "fit_verified": True,
                    "gate_failures": [],
                    "mean_paired_delta": 0.0035,
                    "latency_p95_ms": 1300.0,
                    "constraint_violations": 0,
                }
            ]
        },
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
        temp = Path(temporary)
        campaign_path = temp / "campaign.json"
        report_path = temp / "finalists.json"
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        report = package_top_three(
            campaign_path,
            ROOT / "configs/adaptive_hybrid_1a_3b_v1.json",
            ROOT / "configs/techniques/catalog_v2.json",
            temp / "configs",
            report_path,
            ROOT / "data/splits/adaptive_hybrid_lineage_75_25_v1.json",
        )

        assert report["minimum_challenger_count"] == 1
        assert report["maximum_challenger_count"] == 3
        assert report["packaged_challenger_count"] == 1
        assert report["recommended_candidate_id"] == candidate.candidate_id
        assert report["selection_rule"]["eligible_systems"] == [
            "C_fixed_adaptive_architecture",
            "D1",
        ]
        assert report["finalists"][0]["campaign_decision"] == "HOLD_MORE_DATA"
        assert report["finalists"][0]["promotion_eligible"] is True


def test_rejects_hold_candidate_without_positive_development_delta() -> None:
    registry = AdaptiveTechniqueRegistry.from_catalog(
        load_catalog(ROOT / "configs/techniques/catalog_v2.json"), project_root=ROOT
    )
    candidate = CandidateSpec(
        candidate_id="challenger-single-flat",
        baseline_id="adaptive_hybrid_1a_3b_v1",
        techniques=registry.inventory().compulsory,
        complexity=0,
        generation="single",
    )
    campaign = {
        "mode": "race",
        "records": {
            "f2": [
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "score": 0.8,
                    "decision": "HOLD_MORE_DATA",
                    "fit_required": [],
                    "fit_verified": True,
                    "gate_failures": [],
                    "mean_paired_delta": 0.0,
                    "constraint_violations": 0,
                }
            ]
        },
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
        temp = Path(temporary)
        campaign_path = temp / "campaign.json"
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        with pytest.raises(ValueError, match="at least 1"):
            package_top_three(
                campaign_path,
                ROOT / "configs/adaptive_hybrid_1a_3b_v1.json",
                ROOT / "configs/techniques/catalog_v2.json",
                temp / "configs",
                temp / "finalists.json",
            )


def test_packages_residual_finalist_with_verified_checkpoint_asset() -> None:
    registry = AdaptiveTechniqueRegistry.from_catalog(
        load_catalog(ROOT / "configs/techniques/catalog_v2.json"), project_root=ROOT
    )
    candidate = CandidateSpec(
        candidate_id="challenger-residual",
        baseline_id="adaptive_hybrid_1a_3b_v1",
        techniques=tuple(
            sorted(
                {
                    *registry.inventory().compulsory,
                    "ranking.top10_residual_reranker.v2",
                }
            )
        ),
        complexity=1,
        generation="single",
    )
    campaign = {
        "mode": "race",
        "records": {
            "f2": [
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "score": 0.85,
                    "decision": "PROMOTE",
                    "fit_required": ["ranking.top10_residual_reranker.v2"],
                    "fit_verified": True,
                    "gate_failures": [],
                    "mean_paired_delta": 0.02,
                    "constraint_violations": 0,
                }
            ]
        },
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
        temp = Path(temporary)
        model = temp / "residual.joblib"
        receipt = temp / "residual.fit_receipt.json"
        model.write_bytes(b"verified residual model")
        receipt.write_text('{"verified": true}\n', encoding="utf-8")
        relative_model = model.relative_to(ROOT).as_posix()
        relative_receipt = receipt.relative_to(ROOT).as_posix()
        checkpoint = {
            "evaluations": {
                f"f2:{candidate.candidate_id}": {
                    "residual_fit_receipts": [
                        {
                            "asset_path": relative_model,
                            "asset_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                            "receipt_path": relative_receipt,
                            "receipt_sha256": hashlib.sha256(
                                receipt.read_bytes()
                            ).hexdigest(),
                        }
                    ]
                }
            }
        }
        campaign_path = temp / "campaign.json"
        checkpoint_path = temp / "checkpoint.json"
        report_path = temp / "finalists.json"
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

        report = package_top_three(
            campaign_path,
            ROOT / "configs/adaptive_hybrid_1a_3b_v1.json",
            ROOT / "configs/techniques/catalog_v2.json",
            temp / "configs",
            report_path,
            campaign_checkpoint_path=checkpoint_path,
        )

        finalist = report["finalists"][0]
        assert finalist["runtime_assets"]["asset_path"] == relative_model
        packaged = json.loads((ROOT / finalist["config_path"]).read_text())
        assert packaged["extensions"]["top10_residual_enabled"] is True
        assert packaged["extensions"]["top10_residual_model_path"] == relative_model
