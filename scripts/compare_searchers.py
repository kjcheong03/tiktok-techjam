from __future__ import annotations

import json
import statistics
from pathlib import Path

from ghostlab.optimization.search import PolicyCandidate, search
from ghostlab.research.replay import session_reward

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("random", "grid", "beam", "allocated")
SEEDS = (17, 29, 43)


def main() -> None:
    ablations = json.loads(
        (ROOT / "artifacts/reports/phase4_5_ablations.json").read_text()
    )["variants"]
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    candidates = [
        PolicyCandidate(
            name=name,
            family=("other" if "other" in name else name.split("_", 1)[0]),
            techniques=tuple(name.split("_")),
            complexity=len(name.split("_")),
        )
        for name in ablations
    ]
    rewards = {
        name: {
            str(session["sample_id"]): session_reward(session)
            for session in result["sessions"]
        }
        for name, result in ablations.items()
    }
    runs = []
    outer_results: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}
    for fold_index, outer_ids in enumerate(nested["outer_folds"]):
        outer = set(outer_ids)
        training = set(nested["adaptive_sample_ids"]) - outer
        f0 = set(nested["outer_folds"][(fold_index + 1) % 5])

        def objective(
            candidate: PolicyCandidate,
            fidelity: str,
            seed: int,
            training_ids: set[str] = training,
            f0_ids: set[str] = f0,
        ) -> float:
            del seed
            ids = (
                sorted(training_ids & f0_ids)
                if fidelity == "f0"
                else sorted(training_ids)
            )
            return statistics.fmean(
                rewards[candidate.name][sample_id] for sample_id in ids
            )

        for strategy in STRATEGIES:
            seed_winners = []
            for seed in SEEDS:
                result = search(
                    candidates,
                    objective,
                    strategy=strategy,  # type: ignore[arg-type]
                    budget=12,
                    seed=seed,
                )
                seed_winners.append(result.winner)
                runs.append(
                    {
                        "outer_fold": fold_index,
                        "strategy": strategy,
                        "seed": seed,
                        "winner": result.winner,
                        "training_score": result.winner_score,
                        "evaluations": [
                            evaluation.__dict__ for evaluation in result.evaluations
                        ],
                    }
                )
            winner = min(
                set(seed_winners), key=lambda name: (-seed_winners.count(name), name)
            )
            outer_results[strategy].append(
                statistics.fmean(rewards[winner][sample_id] for sample_id in outer)
            )
    summary = {
        strategy: {
            "mean_outer_reward": round(statistics.fmean(values), 6),
            "outer_fold_rewards": [round(value, 6) for value in values],
        }
        for strategy, values in outer_results.items()
    }
    report = {
        "phase": 8,
        "split": "nested_v1",
        "outer_folds": 5,
        "seeds": list(SEEDS),
        "candidate_budget": 12,
        "summary": summary,
        "runs": runs,
    }
    output = ROOT / "artifacts/reports/phase8_searchers.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
