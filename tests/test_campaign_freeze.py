from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ghostlab.campaign.freeze import (
    build_campaign_manifest,
    freeze_campaign,
    require_clean_worktree,
    sha256_file,
    validate_frozen_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path("configs/campaigns/wave2_smoke_v1.template.json")
FIXTURE_PATHS = (
    TEMPLATE_PATH,
    Path("configs/techniques/catalog_v1.json"),
    Path("configs/techniques/catalog_v2.json"),
    Path("configs/splits/adaptive_v1.json"),
    Path("configs/splits/nested_v1.json"),
    Path("configs/suites/keyword_research.json"),
    Path("configs/suites/unfitted_keyword_search.json"),
    Path("data/public_set.jsonl"),
    Path("artifacts/models/gbdt_reranker_v2_round56.json"),
    Path("artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json"),
)


def copy_freeze_fixture(target: Path) -> dict[str, object]:
    for relative in FIXTURE_PATHS:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)
    return json.loads((target / TEMPLATE_PATH).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def initialize_repository(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Freeze Test",
            "-c",
            "user.email=freeze@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_build_manifest_hashes_actual_data_splits_catalog_and_presets(
    tmp_path: Path,
) -> None:
    template = copy_freeze_fixture(tmp_path)
    manifest = build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)
    assert manifest.dataset_hash == sha256_file(tmp_path / "data/public_set.jsonl")
    assert manifest.adaptive_split_hash == sha256_file(
        tmp_path / "configs/splits/adaptive_v1.json"
    )
    assert manifest.nested_split_hash == sha256_file(
        tmp_path / "configs/splits/nested_v1.json"
    )
    assert manifest.protected_holdout_access == "forbidden"
    assert manifest.technique_ids
    validate_frozen_manifest(tmp_path, manifest)


def test_frozen_manifest_hash_tampering_is_rejected(tmp_path: Path) -> None:
    template = copy_freeze_fixture(tmp_path)
    manifest = build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)
    tampered = manifest.model_copy(update={"dataset_hash": "0" * 64})
    with pytest.raises(ValueError, match="dataset_hash"):
        validate_frozen_manifest(tmp_path, tampered)


def test_freeze_pins_clean_head_and_refuses_dirty_or_untracked_inputs(
    tmp_path: Path,
) -> None:
    copy_freeze_fixture(tmp_path)
    head = initialize_repository(tmp_path)
    require_clean_worktree(tmp_path)
    output = tmp_path / "artifacts/campaigns/smoke/manifest.json"
    manifest = freeze_campaign(tmp_path, tmp_path / TEMPLATE_PATH, output)
    assert manifest.parent_commit == head
    assert output.is_file()
    with pytest.raises(RuntimeError, match="clean worktree"):
        require_clean_worktree(tmp_path)


def test_protected_baseline_preset_path_is_rejected(tmp_path: Path) -> None:
    template = copy_freeze_fixture(tmp_path)
    original = str(template["baseline_presets"][0])
    protected = "configs/protected/f3_preset.json"
    template["baseline_presets"] = [protected]
    template["baseline_techniques_by_preset"] = {
        protected: template["baseline_techniques_by_preset"][original]
    }
    template["baseline_search_modes"] = {
        protected: template["baseline_search_modes"][original]
    }
    with pytest.raises(ValueError, match="protected data"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


@pytest.mark.parametrize("marker", ["f3", "holdout", "protected", "sealed"])
def test_protected_dataset_markers_are_rejected_before_read(
    tmp_path: Path, marker: str
) -> None:
    template = copy_freeze_fixture(tmp_path)
    template["dataset_path"] = f"data/{marker}/public.jsonl"
    with pytest.raises(ValueError, match="protected data"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_missing_runtime_baseline_asset_is_rejected(tmp_path: Path) -> None:
    template = copy_freeze_fixture(tmp_path)
    preset_path = tmp_path / "configs/suites/unfitted_keyword_search.json"
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    preset["reranker"] = "metadata_gbdt"
    preset["reranker_model_asset"] = "artifacts/models/missing.json"
    write_json(preset_path, preset)
    with pytest.raises(FileNotFoundError, match="baseline preset asset"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


@pytest.mark.parametrize(
    ("technique_id", "message"),
    [
        ("missing.technique", "unknown catalog ID"),
        ("research.counterfactual_expert.v2", "not runtime code"),
    ],
)
def test_unknown_or_nonruntime_campaign_bindings_are_rejected(
    tmp_path: Path,
    technique_id: str,
    message: str,
) -> None:
    template = copy_freeze_fixture(tmp_path)
    template["technique_ids"] = [technique_id]
    with pytest.raises(ValueError, match=message):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_unknown_default_baseline_id_is_rejected_even_with_preset_mapping(
    tmp_path: Path,
) -> None:
    template = copy_freeze_fixture(tmp_path)
    template["baseline_techniques"] = ["missing.baseline"]
    with pytest.raises(ValueError, match="unknown catalog ID"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_split_declared_dataset_hash_must_match_actual_bytes(tmp_path: Path) -> None:
    template = copy_freeze_fixture(tmp_path)
    adaptive_path = tmp_path / "configs/splits/adaptive_v1.json"
    adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
    adaptive["dataset_sha256"] = hashlib.sha256(b"different").hexdigest()
    write_json(adaptive_path, adaptive)
    with pytest.raises(ValueError, match="adaptive split dataset hash"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_nested_folds_must_partition_the_adaptive_split(tmp_path: Path) -> None:
    template = copy_freeze_fixture(tmp_path)
    nested_path = tmp_path / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    nested["outer_folds"][0].pop()
    write_json(nested_path, nested)
    with pytest.raises(ValueError, match="partition the adaptive split"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_search_and_confirmation_fold_indices_are_disjoint_and_in_range(
    tmp_path: Path,
) -> None:
    template = copy_freeze_fixture(tmp_path)
    template["search_outer_folds"] = [0, 1, 5]
    template["confirmation_outer_folds"] = [2, 3, 4]
    with pytest.raises(ValueError, match="partition every nested outer fold"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)

    template["search_outer_folds"] = [0, 1, 2]
    template["confirmation_outer_folds"] = [2, 3, 4]
    with pytest.raises(ValueError, match="outer folds overlap"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_search_and_confirmation_fold_groups_must_both_be_nonempty(
    tmp_path: Path,
) -> None:
    template = copy_freeze_fixture(tmp_path)
    template["search_outer_folds"] = []
    with pytest.raises(ValueError):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_search_and_confirmation_sample_sets_cannot_overlap(tmp_path: Path) -> None:
    template = copy_freeze_fixture(tmp_path)
    nested_path = tmp_path / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    nested["outer_folds"][3].append(nested["outer_folds"][0][0])
    write_json(nested_path, nested)
    with pytest.raises(ValueError, match="non-overlapping"):
        build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)


def test_frozen_fold_roles_resolve_to_disjoint_nonempty_sample_sets(
    tmp_path: Path,
) -> None:
    template = copy_freeze_fixture(tmp_path)
    manifest = build_campaign_manifest(tmp_path, template, parent_commit="a" * 40)
    nested = json.loads(
        (tmp_path / "configs/splits/nested_v1.json").read_text(encoding="utf-8")
    )
    search = {
        sample_id
        for fold_index in manifest.search_outer_folds
        for sample_id in nested["outer_folds"][fold_index]
    }
    confirmation = {
        sample_id
        for fold_index in manifest.confirmation_outer_folds
        for sample_id in nested["outer_folds"][fold_index]
    }
    assert len(search) == 90
    assert len(confirmation) == 60
    assert search.isdisjoint(confirmation)
    assert len(search | confirmation) == 150
