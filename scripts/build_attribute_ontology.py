from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.state.catalog_ontology import build_catalog_ontology


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Wave 2 catalog ontology")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    args = parser.parse_args()
    ontology = build_catalog_ontology(
        args.catalog, minimum_frequency=args.minimum_frequency
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ontology.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
