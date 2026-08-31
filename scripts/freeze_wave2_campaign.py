from __future__ import annotations

import argparse
from pathlib import Path

from ghostlab.campaign.freeze import freeze_campaign

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a Wave 2 campaign template")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze_campaign(PROJECT_ROOT, args.template, args.output)


if __name__ == "__main__":
    main()
