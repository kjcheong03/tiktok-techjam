from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePath

from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.selected import ACTIVE_POINTER, PROJECT_ROOT, sha256_file


def _resolve(value: str) -> Path:
    relative = PurePath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError("path must stay inside the project")
    path = (PROJECT_ROOT / relative).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually activate a tested, eligible adaptive finalist"
    )
    parser.add_argument("--preset", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--top3-report", required=True)
    parser.add_argument("--holdout-report", required=True)
    args = parser.parse_args()

    preset = _resolve(args.preset)
    top3 = json.loads(_resolve(args.top3_report).read_text(encoding="utf-8"))
    holdout = json.loads(_resolve(args.holdout_report).read_text(encoding="utf-8"))
    config = AdaptiveArchitectureAudit.validate(load_adaptive_hybrid_config(preset))
    actual = sha256_file(preset)
    if actual != args.expected_sha256:
        raise ValueError("finalist config hash changed; refusing activation")
    finalist = next(
        (
            item
            for item in top3.get("finalists", [])
            if item.get("config_path") == args.preset
            and item.get("config_sha256") == actual
        ),
        None,
    )
    if finalist is None or not finalist.get("promotion_eligible"):
        raise ValueError("preset is not an eligible finalist in the Top-3 report")
    frozen = top3.get("frozen_proposals")
    if not isinstance(frozen, list) or len(frozen) != 3:
        raise ValueError("Top-3 report did not freeze exactly three challengers")
    frozen_item = next(
        (
            item
            for item in frozen
            if item.get("candidate_id") == finalist.get("candidate_id")
            and item.get("config_path") == args.preset
            and item.get("config_sha256") == actual
        ),
        None,
    )
    if frozen_item is None:
        raise ValueError("preset is not one of the three frozen challengers")
    if holdout.get("decision") != "PROMOTE" or not holdout.get("all_gates_passed"):
        raise ValueError("one-time final selection did not authorize promotion")
    challenger = holdout.get("challenger")
    if not isinstance(challenger, dict) or challenger.get(
        "config_sha256"
    ) != config.canonical_hash():
        raise ValueError("holdout report belongs to a different finalist config")
    if holdout.get("selected_candidate_id") != finalist.get("candidate_id"):
        raise ValueError("preset is not the selected final-selection winner")
    if set(holdout.get("frozen_candidate_ids", [])) != {
        str(item["candidate_id"]) for item in frozen
    }:
        raise ValueError("final-selection report used a different frozen Top 3")
    if holdout.get("challenger_count") != 3 or holdout.get("control_count") != 1:
        raise ValueError(
            "final-selection report did not compare exactly three challengers and C"
        )
    frozen_inputs = holdout.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict) or frozen_inputs.get(
        "proposal_report_sha256"
    ) != sha256_file(_resolve(args.top3_report)):
        raise ValueError("final-selection report belongs to a different Top-3 package")

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
                "candidate_id": finalist["candidate_id"],
                "verify_command": (
                    "PYTHONPATH=. .venv/bin/python scripts/verify_active_candidate.py"
                ),
                "rollback_command": (
                    "PYTHONPATH=. .venv/bin/python -m "
                    "scripts.activate_candidate --rollback"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
