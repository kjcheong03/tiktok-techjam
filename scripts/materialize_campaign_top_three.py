from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.campaign.proposal_from_campaign import (
    materialize_confirmed_campaign_top_three,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize three independently development-confirmed campaign proposals; "
            "never promote or access F3"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--catalog", default="configs/techniques/catalog_v2.json"
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--adaptive-split", default="configs/splits/adaptive_v1.json"
    )
    parser.add_argument(
        "--nested-split", default="configs/splits/nested_v1.json"
    )
    parser.add_argument(
        "--baseline-id",
        required=True,
        help="Exact baseline preset path declared by the frozen manifest",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = materialize_confirmed_campaign_top_three(
        project_root=PROJECT_ROOT,
        manifest_path=args.manifest,
        catalog_path=args.catalog,
        evidence_path=args.evidence,
        checkpoint_path=args.checkpoint,
        adaptive_split_path=args.adaptive_split,
        nested_split_path=args.nested_split,
        baseline_id=args.baseline_id,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "status": "development_confirmed_proposals_materialized",
                "automatic_promotion": False,
                "f3_access": "forbidden",
                "output": str(bundle.output_dir.relative_to(PROJECT_ROOT)),
                "manifest": str(bundle.manifest_path.relative_to(PROJECT_ROOT)),
                "guide": str(bundle.guide_path.relative_to(PROJECT_ROOT)),
                "presets": [
                    str(path.relative_to(PROJECT_ROOT))
                    for path in bundle.preset_paths
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
