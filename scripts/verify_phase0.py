from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from baseline.official_reference import Agent as OfficialReferenceAgent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/integrity/official_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/phase0_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{name} must be a string-to-string mapping")
    return value


def verify_hashes(root: Path, expected: dict[str, str]) -> dict[str, object]:
    mismatches: list[dict[str, str | None]] = []
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        if actual_hash != expected_hash:
            mismatches.append(
                {"path": relative, "expected": expected_hash, "actual": actual_hash}
            )
    return {
        "checked": len(expected),
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def expected_metrics(manifest: dict[str, object]) -> dict[str, int | float]:
    value = manifest.get("expected_metrics")
    if not isinstance(value, dict):
        raise TypeError("expected_metrics must be an object")
    allowed = (int, float)
    if not all(
        isinstance(key, str) and isinstance(item, allowed)
        for key, item in value.items()
    ):
        raise ValueError("expected_metrics values must be numeric")
    return value


def reproduce_metrics(root: Path) -> dict[str, object]:
    catalog_path = root / "data/catalog.jsonl"
    samples = load_jsonl(root / "data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index(catalog_path)
    result = evaluate(
        OfficialReferenceAgent(catalog_path),
        samples,
        catalog_ids,
        categories,
        products,
    )
    return {key: value for key, value in result.items() if key != "sessions"}


def verify(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = load_object(manifest_path)
    protected = string_mapping(manifest.get("protected_files"), "protected_files")
    reference = manifest.get("starter_reference")
    if not isinstance(reference, dict):
        raise TypeError("starter_reference must be an object")
    source_path = reference.get("source_path")
    frozen_path = reference.get("frozen_path")
    expected_reference_hash = reference.get("sha256")
    if not isinstance(source_path, str):
        raise TypeError("starter_reference source_path must be a string")
    if not isinstance(frozen_path, str):
        raise TypeError("starter_reference frozen_path must be a string")
    if not isinstance(expected_reference_hash, str):
        raise TypeError("starter_reference sha256 must be a string")

    integrity = verify_hashes(root, protected)
    frozen_hash = sha256_file(root / frozen_path)
    source_hash = sha256_file(root / source_path)
    starter = {
        "frozen_path": frozen_path,
        "frozen_hash": frozen_hash,
        "frozen_matches": frozen_hash == expected_reference_hash,
        "current_source_path": source_path,
        "current_source_matches_reference": source_hash == expected_reference_hash,
    }

    actual_metrics = reproduce_metrics(root)
    expected = expected_metrics(manifest)
    metric_mismatches = {
        key: {"expected": value, "actual": actual_metrics.get(key)}
        for key, value in expected.items()
        if actual_metrics.get(key) != value
    }
    metrics = {
        "passed": not metric_mismatches,
        "expected": expected,
        "actual": {key: actual_metrics.get(key) for key in expected},
        "mismatches": metric_mismatches,
    }
    passed = bool(
        integrity["passed"] and starter["frozen_matches"] and metrics["passed"]
    )
    return {
        "phase": 0,
        "status": "PASS" if passed else "FAIL",
        "manifest": str(manifest_path.relative_to(root)),
        "protected_integrity": integrity,
        "starter_reference": starter,
        "baseline_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen TechJam Phase 0 inputs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = verify(PROJECT_ROOT, args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
