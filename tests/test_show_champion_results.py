from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.show_champion_results import build_summary


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    preset = tmp_path / "configs" / "finalists" / "rank_1_candidate.json"
    preset.parent.mkdir(parents=True)
    preset.write_text('{"policy_id": "candidate"}\n', encoding="utf-8")
    relative_preset = str(preset.relative_to(tmp_path))
    active = {
        "preset_path": relative_preset,
        "preset_sha256": hashlib.sha256(preset.read_bytes()).hexdigest(),
        "activation_decision": "MANUAL_ADJUDICATED_PROMOTION",
    }

    def system(system_id: str, score: float, config_path: str | None = None) -> dict:
        payload = {
            "system_id": system_id,
            "metrics": {
                "hit_rate_at_10": score,
                "mrr": score - 0.1,
                "mttc": 2.5,
                "technical_score": score - 0.05,
            },
        }
        if config_path is not None:
            payload["config_path"] = config_path
        return payload

    report = {
        "evaluation_scope": "one_time_final_selection_set",
        "sample_count": 550,
        "systems": [
            system("A_official_stateless_bm25", 0.2),
            system("C_fixed_adaptive_architecture", 0.9),
            system("GhostLab_Challenger", 0.95, relative_preset),
        ],
    }
    return report, active


def test_build_summary_prints_acd_and_verified_active_champion(tmp_path: Path) -> None:
    report, active = _fixture(tmp_path)

    summary = build_summary(report, active, project_root=tmp_path)

    assert "550 ONE-TIME FINAL-SELECTION SESSIONS" in summary
    assert "A — Organizer BM25" in summary
    assert "C — Fixed Adaptive Architecture" in summary
    assert "D — GhostLab Champion" in summary
    assert "Active champion: D — GhostLab Champion" in summary
    assert active["preset_sha256"] in summary
    assert "MANUAL_ADJUDICATED_PROMOTION" not in summary


def test_build_summary_rejects_champion_hash_mismatch(tmp_path: Path) -> None:
    report, active = _fixture(tmp_path)
    active["preset_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_summary(report, active, project_root=tmp_path)


def test_build_summary_rejects_report_champion_mismatch(tmp_path: Path) -> None:
    report, active = _fixture(tmp_path)
    report["systems"][-1]["config_path"] = "configs/finalists/other.json"

    with pytest.raises(ValueError, match="does not match D"):
        build_summary(report, active, project_root=tmp_path)
