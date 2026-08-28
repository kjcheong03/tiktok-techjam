from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePath

from ghostlab.campaign.bindings import (
    ASSET_FIELDS,
    TechniqueBindingRegistry,
    default_binding_registry,
)
from ghostlab.campaign.catalog import TechniqueCatalog, load_catalog
from ghostlab.campaign.compatibility import validate_techniques
from ghostlab.campaign.models import CampaignManifest
from ghostlab.research.technique_suite import load_suite_config

_FORBIDDEN_PATH_MARKERS = ("f3", "holdout", "protected", "sealed")
_DEFAULT_DATASET_PATH = "data/public_set.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repository_path(
    project_root: Path,
    value: str,
    *,
    label: str,
    must_exist: bool = True,
    allow_directory: bool = False,
) -> Path:
    """Resolve a non-protected path without permitting repository escape."""

    relative = PurePath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError(f"{label} must stay inside the project")
    lowered = "/".join(relative.parts).casefold()
    if any(marker in lowered for marker in _FORBIDDEN_PATH_MARKERS):
        raise ValueError(f"{label} cannot reference protected data: {value}")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the project") from error
    if must_exist:
        exists = resolved.exists() if allow_directory else resolved.is_file()
        if not exists:
            expected = "file or directory" if allow_directory else "file"
            raise FileNotFoundError(f"{label} {expected} does not exist: {value}")
    return resolved


def git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_worktree(project_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "freeze requires a clean worktree so HEAD contains every input"
        )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _string_path(
    payload: dict[str, object], key: str, default: str | None = None
) -> str:
    value = payload.pop(key, default)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string path")
    return value


def _string_ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{label} must be a list of string IDs")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate IDs")
    return result


def _validate_split_integrity(
    dataset_path: Path,
    adaptive_path: Path,
    nested_path: Path,
) -> tuple[str, str, str]:
    dataset_hash = sha256_file(dataset_path)
    adaptive = _load_json_object(adaptive_path, "adaptive split")
    nested = _load_json_object(nested_path, "nested split")
    for label, split in (("adaptive", adaptive), ("nested", nested)):
        if split.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"{label} split dataset hash does not match the dataset")

    adaptive_ids = _string_ids(adaptive.get("sample_ids"), "adaptive sample_ids")
    nested_ids = _string_ids(
        nested.get("adaptive_sample_ids"), "nested adaptive_sample_ids"
    )
    if set(adaptive_ids) != set(nested_ids):
        raise ValueError("nested split does not reference the exact adaptive split")

    rows: set[str] = set()
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise TypeError(f"dataset row {line_number} has no string sample_id")
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"duplicate dataset sample_id: {sample_id}")
            rows.add(sample_id)
    missing = set(adaptive_ids) - rows
    if missing:
        raise ValueError(f"adaptive split has unknown dataset IDs: {sorted(missing)}")

    outer_folds = nested.get("outer_folds")
    if not isinstance(outer_folds, list) or not outer_folds:
        raise TypeError("nested outer_folds must be a non-empty list")
    flattened: list[str] = []
    for index, fold in enumerate(outer_folds):
        flattened.extend(_string_ids(fold, f"nested outer_folds[{index}]"))
    if len(flattened) != len(set(flattened)) or set(flattened) != set(adaptive_ids):
        raise ValueError(
            "nested outer folds must be non-overlapping and partition the adaptive "
            "split exactly"
        )
    return dataset_hash, sha256_file(adaptive_path), sha256_file(nested_path)


def _validate_fold_roles(nested_path: Path, manifest: CampaignManifest) -> None:
    nested = _load_json_object(nested_path, "nested split")
    outer_folds = nested.get("outer_folds")
    if not isinstance(outer_folds, list) or not outer_folds:
        raise TypeError("nested outer_folds must be a non-empty list")
    manifest.validate_fold_partition(len(outer_folds))
    folds = tuple(
        _string_ids(fold, f"nested outer_folds[{index}]")
        for index, fold in enumerate(outer_folds)
    )
    search_samples = {
        sample_id
        for fold_index in manifest.search_outer_folds
        for sample_id in folds[fold_index]
    }
    confirmation_samples = {
        sample_id
        for fold_index in manifest.confirmation_outer_folds
        for sample_id in folds[fold_index]
    }
    if not search_samples or not confirmation_samples:
        raise ValueError("search and confirmation fold sample sets must be non-empty")
    overlap = search_samples & confirmation_samples
    if overlap:
        raise ValueError(
            f"search and confirmation fold sample sets overlap: {sorted(overlap)}"
        )


def _validate_asset_paths(
    project_root: Path,
    paths: tuple[str, ...],
    *,
    label: str,
) -> None:
    for path in paths:
        resolve_repository_path(project_root, path, label=label, allow_directory=True)


def _validate_baseline_presets(
    project_root: Path,
    manifest: CampaignManifest,
) -> None:
    if len(manifest.baseline_presets) != len(set(manifest.baseline_presets)):
        raise ValueError("baseline_presets contains duplicate paths")
    for relative in manifest.baseline_presets:
        path = resolve_repository_path(project_root, relative, label="baseline preset")
        preset = load_suite_config(path)
        assets = tuple(
            str(getattr(preset, field))
            for field in sorted(ASSET_FIELDS)
            if getattr(preset, field, None) is not None
        )
        _validate_asset_paths(
            project_root,
            assets,
            label=f"baseline preset asset ({relative})",
        )


def _dependency_closure(
    catalog: TechniqueCatalog,
    registry: TechniqueBindingRegistry,
    roots: tuple[str, ...],
) -> tuple[str, ...]:
    selected = set(roots)
    pending = list(roots)
    while pending:
        technique_id = pending.pop()
        technique = catalog.techniques.get(technique_id)
        if technique is None:
            raise ValueError(f"campaign references unknown catalog ID: {technique_id}")
        binding = registry.bindings.get(technique_id)
        if binding is None:
            raise ValueError(f"catalog ID has no declared binding: {technique_id}")
        for required in (*technique.requires, *binding.requires):
            if required not in selected:
                selected.add(required)
                pending.append(required)
    return tuple(sorted(selected))


def _validate_catalog_and_bindings(
    project_root: Path,
    catalog: TechniqueCatalog,
    manifest: CampaignManifest,
    registry: TechniqueBindingRegistry,
) -> None:
    candidate_ids = _dependency_closure(catalog, registry, manifest.technique_ids)
    for technique_id in candidate_ids:
        technique = catalog.techniques[technique_id]
        binding = registry.bindings[technique_id]
        if not technique.executable:
            raise ValueError(f"campaign technique is unavailable: {technique_id}")
        if technique.execution_mode != "runtime":
            raise ValueError(f"campaign technique is not runtime code: {technique_id}")
        if binding.disposition != "composable":
            raise ValueError(
                f"campaign technique has no composable binding: {technique_id} "
                f"({binding.disposition})"
            )
        _validate_asset_paths(
            project_root,
            binding.asset_paths,
            label=f"technique binding asset ({technique_id})",
        )

    default_baseline = _dependency_closure(
        catalog, registry, manifest.baseline_techniques
    )
    default_compatibility = validate_techniques(catalog, default_baseline)
    if not default_compatibility.valid:
        raise ValueError(
            "invalid default baseline techniques: "
            + "; ".join(default_compatibility.reasons)
        )
    for technique_id in default_baseline:
        technique = catalog.techniques[technique_id]
        binding = registry.bindings[technique_id]
        if not technique.executable:
            raise ValueError(f"baseline technique is unavailable: {technique_id}")
        if binding.disposition not in {"composable", "anchor_only"}:
            raise ValueError(
                "default baseline technique has no runtime binding: "
                f"{technique_id} ({binding.disposition})"
            )
        _validate_asset_paths(
            project_root,
            binding.asset_paths,
            label=f"baseline technique asset ({technique_id})",
        )

    for preset in manifest.baseline_presets:
        baseline = manifest.techniques_for_preset(preset)
        closure = _dependency_closure(catalog, registry, baseline)
        compatibility = validate_techniques(catalog, closure)
        if not compatibility.valid:
            raise ValueError(
                f"invalid baseline techniques for {preset}: "
                + "; ".join(compatibility.reasons)
            )
        mode = manifest.search_mode_for_preset(preset)
        for technique_id in closure:
            technique = catalog.techniques[technique_id]
            binding = registry.bindings[technique_id]
            if not technique.executable:
                raise ValueError(f"baseline technique is unavailable: {technique_id}")
            allowed = (
                binding.disposition == "composable"
                if mode == "composable"
                else binding.disposition in {"composable", "anchor_only"}
            )
            if not allowed:
                raise ValueError(
                    f"baseline technique binding is incompatible with {mode}: "
                    f"{technique_id} ({binding.disposition})"
                )
            _validate_asset_paths(
                project_root,
                binding.asset_paths,
                label=f"baseline technique asset ({technique_id})",
            )


def build_campaign_manifest(
    project_root: Path,
    template: Mapping[str, object],
    *,
    parent_commit: str,
    registry: TechniqueBindingRegistry | None = None,
) -> CampaignManifest:
    """Validate all frozen inputs and construct their immutable manifest."""

    payload = dict(template)
    catalog_path = resolve_repository_path(
        project_root,
        _string_path(payload, "catalog_path"),
        label="technique catalog",
    )
    dataset_path = resolve_repository_path(
        project_root,
        _string_path(payload, "dataset_path", _DEFAULT_DATASET_PATH),
        label="development dataset",
    )
    adaptive_path = resolve_repository_path(
        project_root,
        _string_path(payload, "adaptive_split_path"),
        label="adaptive split",
    )
    nested_path = resolve_repository_path(
        project_root,
        _string_path(payload, "nested_split_path"),
        label="nested split",
    )
    search_space_value = payload.pop("search_space_path", None)
    search_space_path = None
    search_space_hash = None
    if search_space_value is not None:
        if not isinstance(search_space_value, str) or not search_space_value:
            raise TypeError("search_space_path must be a non-empty string path")
        search_space_path = resolve_repository_path(
            project_root, search_space_value, label="conditional search space"
        )
        search_space_hash = sha256_file(search_space_path)
    dataset_hash, adaptive_hash, nested_hash = _validate_split_integrity(
        dataset_path, adaptive_path, nested_path
    )
    catalog = load_catalog(catalog_path)
    manifest = CampaignManifest.model_validate(
        {
            **payload,
            "parent_commit": parent_commit,
            "catalog_hash": catalog.content_hash,
            "dataset_hash": dataset_hash,
            "adaptive_split_hash": adaptive_hash,
            "nested_split_hash": nested_hash,
            "search_space_path": search_space_value,
            "search_space_hash": search_space_hash,
        }
    )
    _validate_fold_roles(nested_path, manifest)
    _validate_baseline_presets(project_root, manifest)
    _validate_catalog_and_bindings(
        project_root, catalog, manifest, registry or default_binding_registry()
    )
    return manifest


def validate_frozen_manifest(
    project_root: Path,
    manifest: CampaignManifest,
    *,
    catalog_path: str = "configs/techniques/catalog_v2.json",
    dataset_path: str = _DEFAULT_DATASET_PATH,
    adaptive_split_path: str = "configs/splits/adaptive_v1.json",
    nested_split_path: str = "configs/splits/nested_v1.json",
    registry: TechniqueBindingRegistry | None = None,
) -> None:
    """Revalidate a frozen manifest against the current committed inputs."""

    catalog_file = resolve_repository_path(
        project_root, catalog_path, label="technique catalog"
    )
    dataset_file = resolve_repository_path(
        project_root, dataset_path, label="development dataset"
    )
    adaptive_file = resolve_repository_path(
        project_root, adaptive_split_path, label="adaptive split"
    )
    nested_file = resolve_repository_path(
        project_root, nested_split_path, label="nested split"
    )
    dataset_hash, adaptive_hash, nested_hash = _validate_split_integrity(
        dataset_file, adaptive_file, nested_file
    )
    expected = {
        "catalog_hash": load_catalog(catalog_file).content_hash,
        "dataset_hash": dataset_hash,
        "adaptive_split_hash": adaptive_hash,
        "nested_split_hash": nested_hash,
    }
    actual = {
        "catalog_hash": manifest.catalog_hash,
        "dataset_hash": manifest.dataset_hash,
        "adaptive_split_hash": manifest.adaptive_split_hash,
        "nested_split_hash": manifest.nested_split_hash,
    }
    if manifest.search_space_path is not None:
        assert manifest.search_space_hash is not None
        search_space_file = resolve_repository_path(
            project_root,
            manifest.search_space_path,
            label="conditional search space",
        )
        expected["search_space_hash"] = sha256_file(search_space_file)
        actual["search_space_hash"] = manifest.search_space_hash
    mismatches = sorted(key for key in expected if expected[key] != actual[key])
    if mismatches:
        raise ValueError(f"frozen manifest hash mismatch: {', '.join(mismatches)}")
    _validate_fold_roles(nested_file, manifest)
    catalog = load_catalog(catalog_file)
    _validate_baseline_presets(project_root, manifest)
    _validate_catalog_and_bindings(
        project_root, catalog, manifest, registry or default_binding_registry()
    )


def freeze_campaign(
    project_root: Path,
    template_path: Path,
    output_path: Path,
) -> CampaignManifest:
    """Freeze a campaign only from a clean, committed repository state."""

    root = project_root.resolve()
    template = resolve_repository_path(
        root,
        str(template_path.resolve().relative_to(root)),
        label="campaign template",
    )
    output = resolve_repository_path(
        root,
        str(output_path.resolve().relative_to(root)),
        label="campaign output",
        must_exist=False,
    )
    require_clean_worktree(root)
    manifest = build_campaign_manifest(
        root,
        _load_json_object(template, "campaign template"),
        parent_commit=git_head(root),
    )
    serialized = (
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    if output.exists():
        if output.read_text(encoding="utf-8") == serialized:
            return manifest
        raise FileExistsError(
            "frozen campaign manifest already exists with other content"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return manifest
