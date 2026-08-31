from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.policy.joint_policy import JointObservablePolicy
from ghostlab.research.replay import evaluate_replay
from ghostlab.runtime.unified_experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
CHAMPION_SEQUENCE = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)


def _agent(question_variant: str) -> ExperimentalAgent:
    if question_variant == "joint_observable":
        return ExperimentalAgent(
            ROOT / "data/catalog.jsonl",
            state_variant="raw_history",
            negative_evidence=False,
            retrieval_route="keyword",
            sparse_weights=FIELD_WEIGHTS,
            quality_prior_weight=0.2,
            question_variant="joint_observable",
            joint_policy=JointObservablePolicy.from_path(
                ROOT / "configs/assets/joint_policy_control_v1.json"
            ),
        )
    return ExperimentalAgent(
        ROOT / "data/catalog.jsonl",
        state_variant="raw_history",
        negative_evidence=False,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        question_variant="sequence",
        question_order=CHAMPION_SEQUENCE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded joint-policy comparison")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/w2_joint_policy_v1.json")
    )
    args = parser.parse_args()
    split = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    allowed = {str(value) for value in split["adaptive_sample_ids"]}
    samples = [
        row
        for row in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(row["sample_id"]) in allowed
    ][: args.limit]
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    results = {}
    for name in ("sequence", "joint_observable"):
        result = evaluate_replay(_agent(name), samples, categories, products)
        results[name] = {
            key: result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
                "scenario_metrics",
            )
        }
    payload = {
        "schema_version": 1,
        "evaluation_label": "fixed all-adaptive-development mechanism comparison",
        "selection_evidence": False,
        "protected_f3_access": False,
        "session_count": len(samples),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
