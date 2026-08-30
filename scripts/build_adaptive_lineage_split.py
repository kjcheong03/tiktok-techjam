from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import build_lineage_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the verified cross-source lineage-safe 75/25 split"
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output", default="data/splits/adaptive_hybrid_lineage_75_25_v1.json"
    )
    parser.add_argument(
        "--audit-output",
        default="artifacts/reports/adaptive_lineage_reconstruction_audit_v1.json",
    )
    args = parser.parse_args()
    corpus = load_adaptive_training_corpus(ROOT, DATASETS)
    manifest, audit = build_lineage_manifest(
        corpus, seed=args.seed, fold_count=args.folds
    )
    manifest_path = ROOT / args.output
    audit_path = ROOT / args.audit_output
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": args.output,
                "audit": args.audit_output,
                "audit_status": audit["status"],
                "development": len(manifest["partitions"]["development"]["sample_ids"]),
                "holdout": len(manifest["partitions"]["holdout"]["sample_ids"]),
                "outer_fold_counts": [
                    len(item["sample_ids"])
                    for item in manifest["development_outer_folds"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
