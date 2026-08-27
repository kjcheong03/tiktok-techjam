from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ghostlab.campaign.admission import build_admission_report
from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.selected import resolve_active_preset

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/campaigns/autonomous_state_v2_v1.template.json"


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
