from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.campaign.proposal_compare import compare_proposal_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare up to three paired proposal reports without promotion"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Candidate role and report path; repeat at most three times",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidates: dict[str, Path] = {}
    for value in args.candidate:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path or role in candidates:
            parser.error("--candidate values must be unique ROLE=PATH pairs")
        candidates[role] = Path(raw_path)
    result = compare_proposal_reports(args.baseline, candidates)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
