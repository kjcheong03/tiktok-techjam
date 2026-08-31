from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any

from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.selected import ACTIVE_POINTER, PROJECT_ROOT, sha256_file

PUBLIC_REPORT = PROJECT_ROOT / "artifacts/reports/adaptive_public_200.json"
BENCHMARK_INDEX = (
    PROJECT_ROOT / "artifacts/reports/adaptive_ac_finalist_benchmark_index.json"
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required reproduction file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _project_path(relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("reproduction path must be a string")
    safe = PurePath(relative)
    if safe.is_absolute() or ".." in safe.parts or not safe.name:
        raise ValueError(f"reproduction path escapes the repository: {relative}")
    path = (PROJECT_ROOT / safe).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    return path


def _verify_file(relative: object, expected_sha256: object) -> Path:
    path = _project_path(relative)
    if not path.is_file():
        raise FileNotFoundError(f"required reproduction file is missing: {relative}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {relative}: {actual} != {expected_sha256}"
        )
    return path


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if ".cache" in item.parts:
            continue
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _metrics_by_id(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    systems = report.get("systems")
    if not isinstance(systems, list):
        raise TypeError("benchmark report has no systems list")
    result: dict[str, dict[str, float]] = {}
    for item in systems:
        if not isinstance(item, dict) or not isinstance(item.get("metrics"), dict):
            raise TypeError("benchmark system entry is malformed")
        result[str(item["system_id"])] = {
            str(key): float(value) for key, value in item["metrics"].items()
        }
    return result


def main() -> None:
    pointer = _load(ACTIVE_POINTER)
    if pointer.get("schema_version") != 1:
        raise ValueError("unsupported active-candidate pointer schema")
    preset_path = _verify_file(pointer.get("preset_path"), pointer.get("preset_sha256"))
    adjudication_path = _verify_file(
        pointer.get("adjudication_path"), pointer.get("adjudication_sha256")
    )

    config = AdaptiveArchitectureAudit.validate(
        load_adaptive_hybrid_config(preset_path)
    )
    adjudication = _load(adjudication_path)
    if adjudication.get("candidate_id") != config.policy_id:
        raise ValueError("active preset and adjudication candidate IDs differ")
    if adjudication.get("preset_file_sha256") != pointer.get("preset_sha256"):
        raise ValueError("active pointer and adjudication preset hashes differ")
    if adjudication.get("preset_canonical_sha256") != config.canonical_hash():
        raise ValueError("active preset canonical hash does not match adjudication")

    verified_paths: set[str] = {
        str(pointer["preset_path"]),
        str(pointer["adjudication_path"]),
    }
    evidence = adjudication.get("evidence")
    if not isinstance(evidence, dict):
        raise TypeError("champion adjudication has no evidence object")
    for path_key, hash_key in (
        ("final_holdout_report_path", "final_holdout_report_sha256"),
        ("holdout_access_receipt_path", "holdout_access_receipt_sha256"),
        ("finalist_report_path", "finalist_report_sha256"),
    ):
        _verify_file(evidence.get(path_key), evidence.get(hash_key))
        verified_paths.add(str(evidence[path_key]))

    runtime_assets = adjudication.get("runtime_assets")
    if not isinstance(runtime_assets, list) or not runtime_assets:
        raise ValueError("champion adjudication has no runtime assets")
    for asset in runtime_assets:
        if not isinstance(asset, dict):
            raise TypeError("champion runtime asset entry is malformed")
        _verify_file(asset.get("path"), asset.get("sha256"))
        verified_paths.add(str(asset["path"]))

    union = config.union_ranker
    if union.model_path is None or union.model_sha256 is None:
        raise ValueError("champion union ranker has no hash-bound model")
    _verify_file(union.model_path, union.model_sha256)
    verified_paths.add(union.model_path)
    union_receipt_relative = str(Path(union.model_path).with_suffix(".fit_receipt.json"))
    union_receipt = _load(_project_path(union_receipt_relative))
    if union_receipt.get("model_sha256") != union.model_sha256:
        raise ValueError("union GBDT fit receipt does not bind the active model")
    verified_paths.add(union_receipt_relative)

    extensions = config.extensions
    if extensions.top10_residual_enabled:
        _verify_file(
            extensions.top10_residual_model_path,
            extensions.top10_residual_model_sha256,
        )
        _verify_file(
            extensions.top10_residual_fit_receipt_path,
            extensions.top10_residual_fit_receipt_sha256,
        )

    semantic = config.semantic_ranker
    semantic_model_path = _project_path(semantic.model_path)
    if not semantic_model_path.is_dir():
        raise FileNotFoundError(
            f"required semantic model directory is missing: {semantic.model_path}"
        )
    if _sha256_directory(semantic_model_path) != semantic.model_sha256:
        raise ValueError("active semantic model directory hash does not match config")

    index = _load(BENCHMARK_INDEX)
    evaluations = index.get("evaluations")
    if not isinstance(evaluations, dict):
        raise TypeError("benchmark index has no evaluations object")
    for evaluation in evaluations.values():
        if not isinstance(evaluation, dict):
            raise TypeError("benchmark index evaluation is malformed")
        for path_key, hash_key in (
            ("report_path", "report_sha256"),
            ("receipt_path", "receipt_sha256"),
            ("adjudication_path", "adjudication_sha256"),
        ):
            if path_key in evaluation:
                _verify_file(evaluation[path_key], evaluation[hash_key])
                verified_paths.add(str(evaluation[path_key]))

    public_report = _load(PUBLIC_REPORT)
    if int(public_report.get("sample_count", 0)) != 200:
        raise ValueError("official public report does not contain 200 sessions")
    for system in public_report.get("systems", []):
        if not isinstance(system, dict):
            raise TypeError("official public system entry is malformed")
        if "report_path" in system:
            _verify_file(system["report_path"], system["report_sha256"])
            verified_paths.add(str(system["report_path"]))
        if "config_path" in system:
            _verify_file(system["config_path"], system["config_file_sha256"])
            verified_paths.add(str(system["config_path"]))

    holdout_report = _load(_project_path(evidence["final_holdout_report_path"]))
    if int(holdout_report.get("sample_count", 0)) != 550:
        raise ValueError("final-selection report does not contain 550 sessions")

    print(
        json.dumps(
            {
                "verified": True,
                "active_candidate": config.policy_id,
                "architecture": config.architecture,
                "tracked_files_verified": len(verified_paths),
                "official_public_200": _metrics_by_id(public_report),
                "final_selection_550": _metrics_by_id(holdout_report),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
