from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.policy.calibrated_router import (
    RouterTrainingState,
    fit_calibrated_router,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a route model from disjoint outer-training partitions"
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-route", default="keyword")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fit_rows = [row for row in rows if row.get("partition") == "fit"]
    calibration_rows = [row for row in rows if row.get("partition") == "calibration"]
    if not fit_rows or not calibration_rows:
        raise ValueError("labels require non-empty fit and calibration partitions")
    fit_ids = {str(row["sample_id"]) for row in fit_rows}
    calibration_ids = {str(row["sample_id"]) for row in calibration_rows}
    if fit_ids & calibration_ids:
        raise ValueError("router fit and calibration sessions must be disjoint")

    def state(row: dict) -> RouterTrainingState:
        return RouterTrainingState(
            str(row["sample_id"]),
            {str(key): float(value) for key, value in row["features"].items()},
            {str(key): float(value) for key, value in row["route_rewards"].items()},
        )

    feature_names = tuple(sorted(fit_rows[0]["features"]))
    routes = tuple(sorted(fit_rows[0]["route_rewards"]))
    model = fit_calibrated_router(
        [state(row) for row in fit_rows],
        [state(row) for row in calibration_rows],
        feature_names=feature_names,
        routes=routes,
        base_route=args.base_route,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
