from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)
LINEAGE_MANIFEST = "data/splits/adaptive_hybrid_lineage_75_25_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_finalists(top_three_path: Path, output_path: Path) -> dict[str, Any]:
    """Re-evaluate packaged D finalists on the exact shared development ground."""

    top_three = _load(top_three_path)
    finalists = top_three.get("finalists")
    if not isinstance(finalists, list) or not finalists:
        raise ValueError("Top-3 report contains no packaged finalists")
    if len(finalists) > 3:
        raise ValueError("development comparison supports at most three finalists")

    evaluations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="adaptive-finalists-") as temporary:
        temporary_root = Path(temporary)
        for item in finalists:
            candidate_id = str(item["candidate_id"])
            config_path = ROOT / str(item["config_path"])
            if _sha256(config_path) != str(item["config_sha256"]):
                raise ValueError(f"finalist config hash changed: {candidate_id}")
            temporary_report = temporary_root / f"{candidate_id}.json"
            command = [
                sys.executable,
                "scripts/run_adaptive_hybrid.py",
                "--config",
                str(item["config_path"]),
            ]
            for dataset in DATASETS:
                command.extend(("--dataset", dataset))
            command.extend(
                (
                    "--lineage-manifest",
                    LINEAGE_MANIFEST,
                    "--partition",
                    "development",
                    "--output",
                    str(temporary_report),
                )
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
            )
            print(f"START matched development evaluation: {candidate_id}", flush=True)
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            report = _load(temporary_report)
            if report.get("evaluation_partition") != "development":
                raise ValueError(f"finalist used the wrong partition: {candidate_id}")
            if int(report.get("sample_count", 0)) != 1650:
                raise ValueError(
                    f"finalist did not evaluate 1650 sessions: {candidate_id}"
                )
            if not isinstance(report.get("evaluation_contract"), dict):
                raise TypeError(
                    f"finalist has no shared evaluator contract: {candidate_id}"
                )
            evaluations.append(
                {
                    "rank": int(item["rank"]),
                    "candidate_id": candidate_id,
                    "config_path": str(item["config_path"]),
                    "config_sha256": str(item["config_sha256"]),
                    "report": report,
                }
            )
            print(f"DONE matched development evaluation: {candidate_id}", flush=True)

    payload = {
        "schema_version": 1,
        "evaluation_partition": "development",
        "holdout_accessed": False,
        "sample_count": 1650,
        "top_three_report": top_three_path.relative_to(ROOT).as_posix(),
        "top_three_report_sha256": _sha256(top_three_path),
        "evaluation_count": len(evaluations),
        "evaluations": evaluations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate packaged GhostLab finalists on the same shared 1,650-session "
            "development ground used by A/B/C"
        )
    )
    parser.add_argument(
        "--top-three", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/adaptive_finalist_development_evaluations.json",
    )
    args = parser.parse_args()
    payload = evaluate_finalists(ROOT / args.top_three, ROOT / args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "evaluation_count": payload["evaluation_count"],
                "sample_count": payload["sample_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
