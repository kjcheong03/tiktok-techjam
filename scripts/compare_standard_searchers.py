from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from ghostlab.optimization.meta_search import CandidateEvidence, cost_aware_search
from ghostlab.research.replay import session_reward

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("random", "grid", "beam", "allocated")
SEEDS = (17, 29, 43)
F2_EQUIVALENT_BUDGETS = (10, 25, 50, 100)


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [
            record
            for line in handle
            if (record := json.loads(line)).get("status") == "completed"
        ]


def candidate_family(config: dict) -> str:
    reranker = str(config["reranker"])
    return (
        ":".join(
            str(config[key])
            for key in ("retrieval_route", "question_variant", "state_variant")
        )
        + f":{reranker}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested, equal-session-budget comparison of standard searchers"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts/campaigns/standard_adaptive_v1.jsonl",
    )
    parser.add_argument("--expected-count", type=int, default=500)
    args = parser.parse_args()

    records = load_records(args.checkpoint)
    if len(records) < args.expected_count:
        raise RuntimeError(
            f"campaign incomplete: found {len(records)}, expected {args.expected_count}"
        )
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    all_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    reward_maps = {
        str(record["candidate_hash"]): {
            str(session["sample_id"]): session_reward(session)
            for session in record["sessions"]
        }
        for record in records
    }

    runs = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer_ids = {str(value) for value in outer_values}
        training_ids = sorted(all_ids - outer_ids)
        for seed in SEEDS:
            f0_ids = list(training_ids)
            random.Random(seed + fold_index * 1009).shuffle(f0_ids)
            f0_ids = f0_ids[:12]
            evidence = []
            for record in records:
                candidate_id = str(record["candidate_hash"])
                rewards = reward_maps[candidate_id]
                evidence.append(
                    CandidateEvidence(
                        candidate_id=candidate_id,
                        family=candidate_family(record["config"]),
                        complexity=int(record["complexity"]),
                        f0_score=statistics.fmean(rewards[key] for key in f0_ids),
                        f2_score=statistics.fmean(rewards[key] for key in training_ids),
                    )
                )
            for equivalents in F2_EQUIVALENT_BUDGETS:
                budget = equivalents * len(training_ids)
                for strategy in STRATEGIES:
                    result = cost_aware_search(
                        evidence,
                        strategy=strategy,  # type: ignore[arg-type]
                        budget=budget,
                        f0_cost=len(f0_ids),
                        f2_cost=len(training_ids),
                        seed=seed,
                    )
                    outer_reward = statistics.fmean(
                        reward_maps[result.selected_id][key] for key in outer_ids
                    )
                    runs.append(
                        {
                            "outer_fold": fold_index,
                            "seed": seed,
                            "strategy": strategy,
                            "f2_equivalent_budget": equivalents,
                            "session_evaluation_budget": budget,
                            "session_evaluations_used": result.session_evaluations,
                            "screened": result.screened,
                            "promoted": result.promoted,
                            "selected_candidate": result.selected_id,
                            "training_reward": round(result.selected_score, 6),
                            "outer_reward": round(outer_reward, 6),
                            "best_observed_candidate": result.best_observed_id,
                            "best_observed_training_reward": round(
                                result.best_observed_score, 6
                            ),
                        }
                    )

    summary = {}
    for equivalents in F2_EQUIVALENT_BUDGETS:
        summary[str(equivalents)] = {}
        for strategy in STRATEGIES:
            subset = [
                run
                for run in runs
                if run["f2_equivalent_budget"] == equivalents
                and run["strategy"] == strategy
            ]
            winners = Counter(run["selected_candidate"] for run in subset)
            summary[str(equivalents)][strategy] = {
                "mean_outer_reward": round(
                    statistics.fmean(run["outer_reward"] for run in subset), 6
                ),
                "outer_reward_stddev": round(
                    statistics.pstdev(run["outer_reward"] for run in subset), 6
                ),
                "mean_training_reward": round(
                    statistics.fmean(run["training_reward"] for run in subset), 6
                ),
                "unique_selected_candidates": len(winners),
                "most_common_selection_count": winners.most_common(1)[0][1],
                "mean_session_evaluations_used": round(
                    statistics.fmean(run["session_evaluations_used"] for run in subset),
                    1,
                ),
            }

    report = {
        "phase": 8,
        "gate": "nested_equal_session_budget_meta_search",
        "candidate_count": len(records),
        "outer_folds": len(nested["outer_folds"]),
        "seeds": list(SEEDS),
        "f0_sessions": 12,
        "f2_training_sessions": len(all_ids) - len(nested["outer_folds"][0]),
        "f2_equivalent_budgets": list(F2_EQUIVALENT_BUDGETS),
        "holdout_accessed": False,
        "summary": summary,
        "runs": runs,
    }
    output = ROOT / "artifacts/reports/phase8_standard_searchers.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
