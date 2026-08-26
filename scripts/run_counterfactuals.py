from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from baseline.state import ASK_ORDER
from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.research.counterfactual import CounterfactualEvaluator
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = (None, *ASK_ORDER, "other")


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    f0_ids = set(nested["outer_folds"][0])
    samples = [sample for sample in samples if sample["sample_id"] in f0_ids]
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    evaluator = CounterfactualEvaluator(
        lambda: ExperimentalAgent(
            ROOT / "data/catalog.jsonl",
            state_variant="multi",
            question_variant="fixed",
        ),
        categories,
        products,
        continuation_id="manual_strong_multi_fixed_v1",
    )
    outcomes = []
    action_rewards: dict[str, list[float]] = defaultdict(list)
    oracle_rewards = []
    for sample in samples:
        branches = evaluator.branches(sample, ACTIONS)
        outcomes.extend(branches)
        for branch in branches:
            action_rewards[str(branch.action)].append(branch.reward)
        oracle_rewards.append(evaluator.best(branches).reward)
    first = evaluator.evaluate_action(samples[0], ACTIONS[0])
    repeated = evaluator.evaluate_action(samples[0], ACTIONS[0])
    report = {
        "phase": 7,
        "split": "adaptive_v1_outer_fold_0_screen",
        "sample_count": len(samples),
        "actions": list(ACTIONS),
        "mean_reward_by_action": {
            action: round(statistics.fmean(rewards), 6)
            for action, rewards in sorted(action_rewards.items())
        },
        "mean_oracle_reward": round(statistics.fmean(oracle_rewards), 6),
        "deterministic_repeat": first == repeated,
        "cache_hits": evaluator.cache_hits,
        "outcomes": evaluator.serialize(outcomes),
    }
    output = ROOT / "artifacts/reports/phase7_counterfactuals.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "outcomes"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
