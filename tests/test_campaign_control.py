from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ghostlab.campaign.control import (
    CampaignControl,
    load_campaign_control,
    request_skip_hpo,
    save_campaign_control,
)
from scripts import control_autonomous_campaign


def test_campaign_control_is_atomic_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    assert load_campaign_control(path) == CampaignControl()

    requested = request_skip_hpo(path)

    assert requested.skip_hpo is True
    assert requested.requested_at is not None
    assert load_campaign_control(path) == requested
    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text())["skip_hpo"] is True


def test_campaign_control_can_be_preseeded_for_resume(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    save_campaign_control(path, CampaignControl(skip_hpo=True))
    assert load_campaign_control(path).skip_hpo is True


def test_control_cli_reports_f2_request_as_no_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign = tmp_path / "artifacts/campaigns/example"
    campaign.mkdir(parents=True)
    (campaign / "live_status.json").write_text(
        json.dumps({"stage": "f2", "complete": 1, "total_jobs": 2})
    )
    monkeypatch.setattr(control_autonomous_campaign, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control_autonomous_campaign",
            "--campaign-id",
            "example",
            "--skip-hpo",
        ],
    )

    control_autonomous_campaign.main()

    output = json.loads(capsys.readouterr().out)
    assert output["effect"] == "no_op_hpo_already_passed"
    assert output["control"]["skip_hpo"] is True
