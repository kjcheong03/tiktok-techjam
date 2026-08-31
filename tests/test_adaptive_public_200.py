from __future__ import annotations

import pytest

from scripts.evaluate_ac_finalist_public_200 import build_public_comparison


def _report(score: float) -> dict:
    return {
        "hit_rate_at_10": score,
        "mrr": score,
        "mttc": 2.0,
        "efficiency": 0.9,
        "recommended_technical_score": score,
        "evaluation_contract": {"harness_id": "shared-v1"},
        "sessions": [
            {"sample_id": f"public_{index:04d}"} for index in range(200)
        ],
    }


def test_public_comparison_requires_same_ground_and_reports_deltas() -> None:
    report = build_public_comparison(
        reference_a=_report(0.2),
        control=_report(0.8),
        finalist=_report(0.85),
        dataset_path="data/public_set.jsonl",
        dataset_sha256="dataset-hash",
        catalog_path="data/catalog.jsonl",
        catalog_sha256="catalog-hash",
        report_paths={"A": "a.json", "C": "c.json", "D": "d.json"},
        report_hashes={"A": "a-hash", "C": "c-hash", "D": "d-hash"},
        control_config_path="control.json",
        control_config_file_sha256="control-file-hash",
        control_config_canonical_sha256="control-canonical-hash",
        finalist_config_path="finalist.json",
        finalist_config_file_sha256="finalist-file-hash",
        finalist_config_canonical_sha256="finalist-canonical-hash",
        local_assets={"e5_product_embeddings": {"sha256": "embedding-hash"}},
    )

    assert report["sample_count"] == 200
    assert report["independent_holdout"] is False
    assert report["inputs"]["local_assets"]["e5_product_embeddings"]["sha256"] == (
        "embedding-hash"
    )
    assert [item["system_id"] for item in report["systems"]] == [
        "A_official_stateless_bm25",
        "C_fixed_adaptive_architecture",
        "GhostLab_Champion",
    ]
    assert report["deltas"]["Champion_minus_C"]["technical_score"] == pytest.approx(
        0.05
    )


def test_public_comparison_rejects_different_session_order() -> None:
    finalist = _report(0.85)
    finalist["sessions"] = list(reversed(finalist["sessions"]))
    try:
        build_public_comparison(
            reference_a=_report(0.2),
            control=_report(0.8),
            finalist=finalist,
            dataset_path="data/public_set.jsonl",
            dataset_sha256="dataset-hash",
            catalog_path="data/catalog.jsonl",
            catalog_sha256="catalog-hash",
            report_paths={"A": "a.json", "C": "c.json", "D": "d.json"},
            report_hashes={"A": "a-hash", "C": "c-hash", "D": "d-hash"},
            control_config_path="control.json",
            control_config_file_sha256="control-file-hash",
            control_config_canonical_sha256="control-canonical-hash",
            finalist_config_path="finalist.json",
            finalist_config_file_sha256="finalist-file-hash",
            finalist_config_canonical_sha256="finalist-canonical-hash",
        )
    except ValueError as error:
        assert "different ordered session IDs" in str(error)
    else:
        raise AssertionError("comparison accepted mismatched session order")
