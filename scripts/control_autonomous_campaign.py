from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.campaign.control import load_campaign_control, request_skip_hpo

ROOT = Path(__file__).resolve().parents[1]


def _object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or safely control a resumable autonomous campaign"
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--skip-hpo", action="store_true")
    args = parser.parse_args()

    campaign_root = ROOT / "artifacts" / "campaigns" / args.campaign_id
    if not campaign_root.is_dir():
        raise FileNotFoundError(f"campaign does not exist: {campaign_root}")
    control_path = campaign_root / "control.json"
    live = _object(campaign_root / "live_status.json")
    before = load_campaign_control(control_path)
    control = request_skip_hpo(control_path) if args.skip_hpo else before
    stage = live.get("stage")
    if args.skip_hpo and stage == "f2":
        effect = "no_op_hpo_already_passed"
    elif args.skip_hpo and stage == "hpo":
        effect = "requested_finish_current_wave_then_continue_to_f2"
    elif args.skip_hpo:
        effect = "requested_skip_hpo_at_next_safe_boundary"
    else:
        effect = "status_only"
    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "control": control.model_dump(mode="json"),
                "effect": effect,
                "live_status": live or None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
