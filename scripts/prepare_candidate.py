from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.selected import PROJECT_ROOT, sha256_file


def _safe_source(value: Path) -> Path:
    source = value.resolve()
    source.relative_to(PROJECT_ROOT.resolve())
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and stage a proposal without activating it"
    )
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()

    source = _safe_source(args.preset)
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
