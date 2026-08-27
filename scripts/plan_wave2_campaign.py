from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CampaignManifest
from ghostlab.campaign.planner import plan_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a frozen Wave 2 campaign")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "configs/techniques/catalog_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = CampaignManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8")
    )
    catalog = load_catalog(args.catalog)
    if manifest.catalog_hash != catalog.content_hash:
        raise ValueError("campaign catalog hash does not match the current catalog")
    plan = plan_candidates(
        catalog,
        baseline_id=manifest.baseline_presets[0],
        baseline_techniques=manifest.baseline_techniques,
        technique_ids=manifest.technique_ids,
        max_order=manifest.max_order,
        candidate_limit=manifest.candidate_limit,
    )
    payload = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "manifest_hash": manifest.canonical_hash(),
        "candidates": [item.model_dump(mode="json") for item in plan.candidates],
        "skipped": [
            {"roots": item.roots, "reasons": item.reasons} for item in plan.skipped
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
