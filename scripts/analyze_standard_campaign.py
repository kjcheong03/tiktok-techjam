from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from ghostlab.optimization.evidence import family_ucb_allocation
from ghostlab.optimization.racing import racing_decide
from ghostlab.research.replay import session_reward

ROOT = Path(__file__).resolve().parents[1]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("status") == "completed":
                records.append(record)
    return records


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + end - 1) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    left_ranks, right_ranks = rank(left), rank(right)
    left_mean, right_mean = statistics.fmean(left_ranks), statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = (
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze F0/F1/F2 campaign evidence")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts/campaigns/standard_adaptive_v1.jsonl",
    )
    args = parser.parse_args()
    records = load_records(args.checkpoint)
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    f0_ids = set(nested["outer_folds"][0][:12])
    f1_ids = set(nested["outer_folds"][0]) | set(nested["outer_folds"][1])
    all_ids = set(nested["adaptive_sample_ids"])
    baseline = next(
        record
        for record in records
        if record["config"]["question_variant"] == "other_always"
        and record["config"]["state_variant"] == "multi"
        and record["config"]["retrieval_route"] == "keyword"
        and "search_seed" not in record["config"]
    )
    baseline_rewards = {
        str(item["sample_id"]): session_reward(item) for item in baseline["sessions"]
    }
    family_gains: dict[str, list[float]] = defaultdict(list)
    analyses = []
    f0_scores, f2_scores = [], []
    for record in records:
        rewards = {
            str(item["sample_id"]): session_reward(item) for item in record["sessions"]
        }
        f0_deltas = [rewards[key] - baseline_rewards[key] for key in sorted(f0_ids)]
        f1_deltas = [rewards[key] - baseline_rewards[key] for key in sorted(f1_ids)]
        f2_deltas = [rewards[key] - baseline_rewards[key] for key in sorted(all_ids)]
        novelty = float(record["config"]["retrieval_route"] != "keyword")
        f0_decision = racing_decide(f0_deltas, fidelity="f0", behavior_novelty=novelty)
        f1_decision = racing_decide(f1_deltas, fidelity="f1", behavior_novelty=novelty)
        f2_mean = statistics.fmean(f2_deltas)
        family = ":".join(
            str(record["config"][key])
            for key in ("retrieval_route", "question_variant", "state_variant")
        )
        family_gains[family].append(f2_mean)
        f0_scores.append(statistics.fmean(f0_deltas))
        f2_scores.append(f2_mean)
        analyses.append(
            {
                "candidate_hash": record["candidate_hash"],
                "family": family,
                "f0_mean_delta": round(f0_scores[-1], 6),
                "f0_decision": f0_decision,
                "f1_mean_delta": round(statistics.fmean(f1_deltas), 6),
                "f1_decision": f1_decision,
                "f2_mean_delta": round(f2_mean, 6),
            }
        )
    top_f2 = sorted(analyses, key=lambda item: -item["f2_mean_delta"])[:20]
    false_pruned = [
        item["candidate_hash"]
        for item in top_f2
        if item["f0_decision"] == "REJECT" or item["f1_decision"] == "REJECT"
    ]
    signatures = Counter(
        tuple(
            (item["hit"], item["first_hit_turn"], item["best_rank"])
            for item in record["sessions"]
        )
        for record in records
    )
    report = {
        "phase": 8,
        "gate": "multi_fidelity_evidence_analysis",
        "candidate_count": len(records),
        "f0_count": len(f0_ids),
        "f1_count": len(f1_ids),
        "f2_count": len(all_ids),
        "f0_f2_spearman": round(spearman(f0_scores, f2_scores), 6),
        "top20_false_prune_count": len(false_pruned),
        "top20_false_pruned_candidates": false_pruned,
        "decision_counts_f0": dict(
            sorted(Counter(item["f0_decision"] for item in analyses).items())
        ),
        "decision_counts_f1": dict(
            sorted(Counter(item["f1_decision"] for item in analyses).items())
        ),
        "unique_outcome_signatures": len(signatures),
        "duplicate_outcome_policies": sum(count - 1 for count in signatures.values()),
        "next_family_allocation": family_ucb_allocation(family_gains),
        "candidates": analyses,
    }
    output = ROOT / "artifacts/reports/phase8_multifidelity_analysis.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "candidates"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
