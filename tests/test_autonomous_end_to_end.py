from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from ghostlab.campaign.admission import build_admission_report
from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.selected import resolve_active_preset
from scripts import run_autonomous_end_to_end

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/campaigns/autonomous_state_v2_v1.template.json"
TARGETED_TEMPLATE = (
    ROOT / "configs/campaigns/adaptive_autonomous_augment_v2.template.json"
)


def test_augment_mode_uses_fresh_v2_campaign_after_residual_adapter_fix() -> None:
    assert run_autonomous_end_to_end.MODE_TEMPLATES["augment"].endswith(
        "adaptive_autonomous_augment_v2.template.json"
    )


def test_complete_template_accounts_for_every_technique_and_minimum_trial() -> None:
    report = build_admission_report(
        project_root=ROOT,
        template_path=TEMPLATE,
        catalog=load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        registry=default_binding_registry(),
    )
    assert report.complete_catalog_accounting
    assert report.admitted_without_trial == ()
    assert report.planned_structure_count <= 600
    assert report.materializable_structure_count > 300
    assert report.blocked_structure_count > 0


def test_targeted_template_may_explicitly_omit_unrelated_runnable_techniques() -> None:
    strict = build_admission_report(
        project_root=ROOT,
        template_path=TARGETED_TEMPLATE,
        catalog=load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        registry=default_binding_registry(),
    )
    targeted = build_admission_report(
        project_root=ROOT,
        template_path=TARGETED_TEMPLATE,
        catalog=load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        registry=default_binding_registry(),
        require_all_runnable=False,
    )
    assert not strict.campaign_ready
    assert targeted.campaign_ready
    assert targeted.admitted_without_trial == ()
    assert targeted.blocked_count == 0


def test_dense_and_fusion_bindings_compose_with_deterministic_precedence() -> None:
    baseline = load_suite_config(ROOT / "configs/suites/unfitted_keyword_search.json")
    candidate = CandidateSpec(
        candidate_id="dense-fusion",
        baseline_id="pure",
        techniques=("retrieval.e5", "fusion.weighted"),
        complexity=2,
        generation="pair",
    )
    config = default_binding_registry().materialize(baseline, candidate)
    assert config.retrieval_route == "weighted"
    assert config.dense_backend == "e5_small_v2"
    assert config.dense_model_path == "artifacts/cache/models/e5-small-v2"
    assert config.sparse_weight + config.dense_weight == pytest.approx(1.0)


def test_active_pointer_is_hash_bound_and_project_relative(tmp_path: Path) -> None:
    preset = ROOT / "configs/suites/champion_guarded.json"
    digest = hashlib.sha256(preset.read_bytes()).hexdigest()
    pointer = tmp_path / "active.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preset_path": "configs/suites/champion_guarded.json",
                "preset_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    assert resolve_active_preset(pointer) == preset.resolve()
    pointer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preset_path": "configs/suites/champion_guarded.json",
                "preset_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash"):
        resolve_active_preset(pointer)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("confirmed", [True, False])
def test_campaign_runs_before_proposal_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    confirmed: bool,
) -> None:
    template = "configs/campaigns/adaptive.json"
    campaign_id = "adaptive-test"
    _write_json(tmp_path / template, {"campaign_id": campaign_id})
    calls: list[str] = []

    def fake_run(*arguments: str) -> None:
        module = arguments[1]
        calls.append(module)
        if module == "scripts.run_autonomous_campaign":
            evidence = tmp_path / "artifacts/campaigns" / campaign_id / "evidence.json"
            candidates = (
                [
                    {
                        "candidate_id": f"candidate-{index}",
                        "baseline_id": "configs/suites/baseline.json",
                        "score": 0.9 - index * 0.01,
                        "mean_delta": 0.1 - index * 0.01,
                        "champion_comparison": {
                            "technical_score_delta": 0.05 - index * 0.01,
                            "promotion_recommended": True,
                        },
                    }
                    for index in range(3)
                ]
                if confirmed
                else []
            )
            _write_json(evidence, {"confirmed_top3": candidates})
        if module == "scripts.materialize_campaign_top_three":
            proposal_manifest = (
                tmp_path
                / "artifacts/proposals"
                / campaign_id
                / "proposal_manifest.json"
            )
            _write_json(
                proposal_manifest,
                {
                    "candidates": [
                        {
                            "candidate_id": "candidate-0",
                            "role": "score_leader",
                            "preset": {"path": "candidate.json"},
                            "enabled_techniques": ["retrieval.sparse"],
                            "tuned_parameters": {},
                            "prepare_command": (
                                "uv run python -m scripts.prepare_candidate "
                                "--preset candidate.json"
                            ),
                        }
                    ]
                },
            )

    monkeypatch.setattr(run_autonomous_end_to_end, "ROOT", tmp_path)
    monkeypatch.setattr(run_autonomous_end_to_end, "MODE_TEMPLATES", {"full": template})
    monkeypatch.setattr(run_autonomous_end_to_end, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_autonomous_end_to_end"])

    run_autonomous_end_to_end.main()

    assert calls[:4] == [
        "scripts.preflight_autonomous",
        "scripts.freeze_wave2_campaign",
        "scripts.plan_wave2_campaign",
        "scripts.run_autonomous_campaign",
    ]
    output = json.loads(capsys.readouterr().out)
    if confirmed:
        assert calls[-1] == "scripts.materialize_campaign_top_three"
        assert output["prepare_commands"] == [
            "uv run python -m scripts.prepare_candidate --preset candidate.json"
        ]
        assert output["candidate_summaries"] == [
            {
                "candidate_id": "candidate-0",
                "role": "score_leader",
                "score": 0.9,
                "mean_delta_vs_matched_state_v2": 0.1,
                "technical_score_delta_vs_champion": 0.05,
                "promotion_recommended": True,
                "enabled_techniques": ["retrieval.sparse"],
                "tuned_parameters": {},
                "prepare_command": (
                    "uv run python -m scripts.prepare_candidate "
                    "--preset candidate.json"
                ),
            }
        ]
    else:
        assert "scripts.materialize_campaign_top_three" not in calls
        assert output["proposal_count"] == 0
        assert output["retain_current_champion"] is True
