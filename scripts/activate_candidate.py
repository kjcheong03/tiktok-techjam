from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePath

from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.selected import ACTIVE_POINTER, PROJECT_ROOT, sha256_file

CHAMPION = "configs/suites/champion_guarded.json"


def _resolve(value: str) -> Path:
    relative = PurePath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError("preset must stay inside the project")
    path = (PROJECT_ROOT / relative).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically activate a prepared suite or roll back to champion"
    )
    parser.add_argument("--preset")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        relative = CHAMPION
    elif args.preset and args.expected_sha256:
        relative = args.preset
    else:
        parser.error("provide --preset and --expected-sha256, or --rollback")
    preset = _resolve(relative)
    load_suite_config(preset)
    actual = sha256_file(preset)
    if not args.rollback and actual != args.expected_sha256:
        raise ValueError("prepared preset hash changed; refusing activation")
    payload = {
        "schema_version": 1,
        "preset_path": preset.relative_to(PROJECT_ROOT).as_posix(),
        "preset_sha256": actual,
    }
    ACTIVE_POINTER.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_POINTER.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, ACTIVE_POINTER)
    print(
        json.dumps(
            {
                "activated": payload,
                "verify_command": "uv run python -m scripts.verify_active_candidate",
                "rollback_command": "uv run python -m scripts.activate_candidate --rollback",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
