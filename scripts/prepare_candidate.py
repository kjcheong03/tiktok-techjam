from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ghostlab.campaign.models import ChampionComparison
from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.selected import PROJECT_ROOT, sha256_file


def _safe_source(value: Path) -> Path:
    source = value.resolve()
    source.relative_to(PROJECT_ROOT.resolve())
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _proposal_champion_comparison(source: Path) -> dict[str, object] | None:
    manifest_path = source.parent / "proposal_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = source.relative_to(PROJECT_ROOT).as_posix()
    for record in manifest.get("candidates", []):
        if not isinstance(record, dict):
            continue
        preset = record.get("preset")
        if isinstance(preset, dict) and preset.get("path") == relative:
            if preset.get("sha256") != sha256_file(source):
                raise ValueError("proposal preset hash does not match its manifest")
            comparison = record.get("champion_comparison")
            if comparison is None:
                if manifest.get("champion_comparison_required") is True:
                    raise ValueError(
                        "proposal is missing its required champion comparison"
                    )
                return None
            return ChampionComparison.model_validate(comparison).model_dump(mode="json")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and stage a proposal without activating it"
    )
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()

    source = _safe_source(args.preset)
    champion_comparison = _proposal_champion_comparison(source)
    config = load_suite_config(source)
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config.experiment_id).strip("-")
    if not slug:
        slug = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = PROJECT_ROOT / "configs/candidates" / f"{slug}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    report = PROJECT_ROOT / "artifacts/prepared" / f"{slug}.json"
    if not args.structural_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.run_unified_preset",
                "--config",
                target.relative_to(PROJECT_ROOT).as_posix(),
                "--split",
                "configs/splits/adaptive_v1.json",
                "--output",
                report.relative_to(PROJECT_ROOT).as_posix(),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    preset_hash = sha256_file(target)
    command = (
        "uv run python -m scripts.activate_candidate --preset "
        f"{target.relative_to(PROJECT_ROOT).as_posix()} "
        f"--expected-sha256 {preset_hash}"
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "activated": False,
                "candidate": config.experiment_id,
                "preset": target.relative_to(PROJECT_ROOT).as_posix(),
                "preset_sha256": preset_hash,
                "validation": (
                    "structural_only"
                    if args.structural_only
                    else report.relative_to(PROJECT_ROOT).as_posix()
                ),
                "next_activation_command": command,
                "champion_comparison": champion_comparison,
                "promotion_recommended": (
                    champion_comparison.get("promotion_recommended")
                    if champion_comparison is not None
                    else None
                ),
                "automatic_promotion": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
