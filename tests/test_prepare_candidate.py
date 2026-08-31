from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_candidate


def _comparison() -> dict[str, object]:
    return {
        "champion_candidate_id": "champion",
        "champion_baseline_id": "configs/suites/champion_guarded.json",
        "candidate_metrics": {"technical_score": 0.8},
        "champion_metrics": {"technical_score": 0.7},
        "technical_score_delta": 0.1,
        "paired_mean_delta": 0.1,
        "confidence_interval": [0.05, 0.15],
        "randomization_pvalue": 0.01,
        "paired_session_count": 5,
        "wins": 4,
        "ties": 0,
        "losses": 1,
        "beats_champion_point_estimate": True,
        "statistically_supported": True,
        "no_material_scenario_regression": True,
        "fit_receipts_verified": True,
        "promotion_recommended": True,
        "automatic_promotion": False,
    }


def _write_bundle(root: Path, comparison: dict[str, object] | None) -> Path:
    source = root / "artifacts/proposals/campaign/score_leader.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"experiment_id":"candidate"}\n', encoding="utf-8")
    relative = source.relative_to(root).as_posix()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "champion_comparison_required": True,
        "candidates": [
            {
                "preset": {"path": relative, "sha256": digest},
                "champion_comparison": comparison,
            }
        ],
    }
    (source.parent / "proposal_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return source


def test_preparation_repeats_only_hash_bound_valid_champion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_bundle(tmp_path, _comparison())
    monkeypatch.setattr(prepare_candidate, "PROJECT_ROOT", tmp_path)

    comparison = prepare_candidate._proposal_champion_comparison(source)

    assert comparison is not None
    assert comparison["promotion_recommended"] is True
    assert comparison["automatic_promotion"] is False
    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preset hash"):
        prepare_candidate._proposal_champion_comparison(source)


def test_preparation_rejects_missing_required_champion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_bundle(tmp_path, None)
    monkeypatch.setattr(prepare_candidate, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="required champion comparison"):
        prepare_candidate._proposal_champion_comparison(source)
