from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ghostlab.research.technique_suite import PROJECT_ROOT, load_suite_config

MANIFEST = PROJECT_ROOT / "configs/integrity/unified_consolidation_v1.json"
OUTPUT = PROJECT_ROOT / "artifacts/reports/unified_consolidation_audit_v1.json"
PATH_FIELDS = (
    "source_paths",
    "test_paths",
    "manifest_paths",
    "report_paths",
    "runtime_assets",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing: list[str] = []
    hashes: dict[str, str] = {}
    component_results: list[dict[str, object]] = []
    for component in manifest["components"]:
        paths = [
            str(relative)
            for field in PATH_FIELDS
            for relative in component.get(field, [])
        ]
        component_missing = [
            relative for relative in paths if not (PROJECT_ROOT / relative).is_file()
        ]
        missing.extend(component_missing)
        for relative in paths:
            path = PROJECT_ROOT / relative
            if path.is_file():
                hashes[relative] = sha256(path)
        component_results.append(
            {
                "id": component["id"],
                "origin": component["origin"],
                "required_path_count": len(paths),
                "missing": component_missing,
                "archived_raw_paths": component.get("archived_raw_paths", []),
                "retest_trigger": component["retest_trigger"],
            }
        )

    catalog = json.loads(
        (PROJECT_ROOT / "configs/techniques/catalog_v1.json").read_text(
            encoding="utf-8"
        )
    )
    catalog_missing = [
        item["source"]
        for item in catalog["techniques"]
        if not (PROJECT_ROOT / item["source"]).is_file()
    ]
    missing.extend(catalog_missing)
    suites = sorted((PROJECT_ROOT / "configs/suites").glob("*.json"))
    suite_ids = [load_suite_config(path).experiment_id for path in suites]
    report = {
        "schema_version": 1,
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
        "unified_base": manifest["unified_base"],
        "component_count": len(component_results),
        "catalog_technique_count": len(catalog["techniques"]),
        "suite_count": len(suites),
        "suite_ids": suite_ids,
        "components": component_results,
        "required_file_hashes": dict(sorted(hashes.items())),
        "missing": sorted(set(missing)),
        "archived_raw_files_are_runtime_dependencies": False,
        "passed": not missing,
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
