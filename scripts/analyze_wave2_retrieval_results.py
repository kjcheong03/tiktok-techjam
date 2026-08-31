from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.research.replay import paired_delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched Wave 2 replays")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = json.loads(args.control.read_text(encoding="utf-8"))
    analyses: dict[str, object] = {}
    for path in args.candidate:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        deltas = paired_delta(candidate["sessions"], control["sessions"])
        analyses[str(candidate["experiment_id"])] = {
            "technical_score": candidate["metrics"]["recommended_technical_score"],
            "technical_score_delta": round(
                candidate["metrics"]["recommended_technical_score"]
                - control["metrics"]["recommended_technical_score"],
                6,
            ),
            "mean_session_reward_delta": round(statistics.fmean(deltas), 6),
            "bootstrap_95_interval": [
                round(value, 6)
                for value in bootstrap_mean_interval(deltas, resamples=10_000)
            ],
            "paired_randomization_pvalue": round(
                paired_randomization_pvalue(deltas, resamples=10_000), 6
            ),
            "wins": sum(value > 0.0 for value in deltas),
            "ties": sum(value == 0.0 for value in deltas),
            "losses": sum(value < 0.0 for value in deltas),
        }
    report = {
        "schema_version": 1,
        "evaluation_label": "F1 reused-development paired diagnostic; not OOF champion evidence",
        "holdout_accessed": False,
        "control": control["experiment_id"],
        "control_technical_score": control["metrics"]["recommended_technical_score"],
        "candidates": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
