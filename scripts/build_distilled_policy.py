from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.policy.distilled_expert import fit_distilled_policy
from ghostlab.research.counterfactual_expert import ExpertLabel


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a compact fold-local expert tree")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--actions",
        type=Path,
        required=True,
        help="JSON object containing actions, allowed_routes, and allowed_depths",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-depth", type=int, default=2)
    parser.add_argument("--minimum-leaf-sessions", type=int, default=10)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [ExpertLabel(**row) for row in rows]
    if not labels:
        raise ValueError("no expert labels supplied")
    feature_names = tuple(sorted(labels[0].features))
    action_order = tuple(sorted(labels[0].action_rewards))
    model = fit_distilled_policy(
        labels,
        feature_names=feature_names,
        action_order=action_order,
        maximum_depth=args.maximum_depth,
        minimum_leaf_sessions=args.minimum_leaf_sessions,
    )
    action_spec = json.loads(args.actions.read_text(encoding="utf-8"))
    required = {"actions", "allowed_routes", "allowed_depths"}
    if not isinstance(action_spec, dict) or not required <= action_spec.keys():
        raise ValueError("action spec is missing required runtime fields")
    payload = {
        "schema_version": 1,
        "model": model.to_payload(),
        "actions": action_spec["actions"],
        "allowed_routes": action_spec["allowed_routes"],
        "allowed_depths": action_spec["allowed_depths"],
        "confidence_threshold": action_spec.get("confidence_threshold", 0.55),
        "fallback_action_id": action_spec.get("fallback_action_id", "base"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
