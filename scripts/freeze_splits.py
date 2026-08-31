from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.evaluation.splits import freeze_split, load_rows, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_frozen(path: Path, payload: dict[str, object]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace frozen split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze prospective and nested splits")
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl"
    )
    parser.add_argument("--seed", default="ghostlab-public-v1")
    args = parser.parse_args()

    adaptive, nested, guarded = freeze_split(load_rows(args.dataset), args.seed)
    dataset_hash = sha256_file(args.dataset)
    for payload in (adaptive, nested, guarded):
        payload["dataset_sha256"] = dataset_hash
    write_frozen(PROJECT_ROOT / "configs/splits/adaptive_v1.json", adaptive)
    write_frozen(PROJECT_ROOT / "configs/splits/nested_v1.json", nested)
    write_frozen(PROJECT_ROOT / "artifacts/guarded/f3_v1.json", guarded)
    print(
        json.dumps(
            {
                "adaptive": len(adaptive["sample_ids"]),
                "holdout": len(guarded["sample_ids"]),
                "outer_fold_sizes": [len(fold) for fold in nested["outer_folds"]],
                "holdout_ids_sha256": adaptive["prospective_meta"][
                    "holdout_ids_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
