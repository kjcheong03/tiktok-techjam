from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ghostlab.research.technique_suite import (
    PROJECT_ROOT,
    load_suite_config,
    valid_combinations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and materialize a finite unified technique search space"
    )
    parser.add_argument(
        "--space",
        type=Path,
        default=PROJECT_ROOT / "configs/search/unified_space_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    value = json.loads(args.space.read_text(encoding="utf-8"))
    base = load_suite_config(PROJECT_ROOT / value["base_config"])
    accepted, rejected = valid_combinations(base, value["dimensions"])
    accepted_total = len(accepted)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        accepted = accepted[: args.limit]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for config in accepted:
                handle.write(config.model_dump_json() + "\n")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for config in accepted:
            encoded = config.model_dump_json()
            digest = hashlib.sha256(encoded.encode()).hexdigest()[:12]
            (args.output_dir / f"combination_{digest}.json").write_text(
                json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    print(
        json.dumps(
            {
                "space": str(args.space),
                "accepted_total": accepted_total,
                "materialized": len(accepted),
                "rejected": len(rejected),
                "output": None if args.output is None else str(args.output),
                "output_dir": (
                    None if args.output_dir is None else str(args.output_dir)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
