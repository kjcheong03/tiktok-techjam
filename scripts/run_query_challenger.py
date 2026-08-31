from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    load_jsonl,
    materialize_hidden_fields,
)
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.policy.models import RuntimeConfig
from ghostlab.research.firewall import runtime_profile
from ghostlab.research.replay import evaluate_replay, paired_delta, session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiments/challenger_query_v1.json"
VARIANTS = {
    "raw_history": 0,
    "structured_active": 1,
    "category_constraints": 1,
    "compressed_raw": 1,
    "raw_plus_active": 2,
    "negation_safe_hybrid": 2,
}


class TimedAgent:
    def __init__(self, agent: ExperimentalAgent) -> None:
        self.agent = agent
        self.turn_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.turn_ms.append((time.perf_counter() - started) * 1000)
        return response


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def retrieval_diagnostics(
    agent: ExperimentalAgent, samples: list[dict], products: dict[str, dict]
) -> dict:
    traces: dict[str, list[dict]] = defaultdict(list)
    for item in agent.retrieval_trace:
        traces[str(item["session_id"])].append(item)
    cutoffs = (10, 50, 100, 200)
    first_hits = {cutoff: 0 for cutoff in cutoffs}
    any_hits = {cutoff: 0 for cutoff in cutoffs}
    ranks: list[int] = []
    for sample in samples:
        session_id = f"replay_{sample['sample_id']}"
        target = str(sample["ground_truth"]["parent_asin"])
        _, behavior = materialize_hidden_fields(sample, products)
        eligible_turn = (
            int((behavior.get("override") or {}).get("turn", 1))
            if sample["scenario_type"] == "intent_override"
            else 1
        )
        eligible = [
            item for item in traces[session_id] if int(item["turn"]) >= eligible_turn
        ]
        if not eligible:
            continue
        first = list(eligible[0]["retrieved"])
        for cutoff in cutoffs:
            first_hits[cutoff] += target in first[:cutoff]
            any_hits[cutoff] += any(
                target in list(item["retrieved"])[:cutoff] for item in eligible
            )
        candidate_ranks = [
            list(item["retrieved"]).index(target) + 1
            for item in eligible
            if target in list(item["retrieved"])
        ]
        if candidate_ranks:
            ranks.append(min(candidate_ranks))
    count = len(samples)
    return {
        "first_eligible_turn_recall": {
            f"recall_at_{cutoff}": round(first_hits[cutoff] / count, 6)
            for cutoff in cutoffs
        },
        "any_eligible_turn_recall": {
            f"recall_at_{cutoff}": round(any_hits[cutoff] / count, 6)
            for cutoff in cutoffs
        },
        "mean_best_candidate_rank_when_retrieved": (
            round(statistics.fmean(ranks), 6) if ranks else None
        ),
    }


def behavior_signature(agent: ExperimentalAgent) -> str:
    behavior = [
        {
            "turn": item["turn"],
            "top_10": list(item["ranked"])[:10],
        }
        for item in agent.retrieval_trace
    ]
    encoded = json.dumps(behavior, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    ]
    catalog_path = ROOT / "data/catalog.jsonl"
    _, categories, products = catalog_index(catalog_path)
    config = RuntimeConfig.model_validate_json(
        (ROOT / "configs/compiled_policy.json").read_text(encoding="utf-8")
    ).techniques
    results: dict[str, dict] = {}
    started_all = time.perf_counter()
    for ordinal, (name, complexity) in enumerate(VARIANTS.items(), start=1):
        agent = ExperimentalAgent(
            catalog_path,
            state_variant="raw_history",
            question_variant="sequence",
            question_order=tuple(config.question_order),
            retrieval_route="keyword",
            negative_evidence=True,
            provenance=True,
            override_invalidation=True,
            sparse_weights=config.sparse_field_weights,
            quality_prior_weight=config.quality_prior_weight,
            query_variant=name,  # type: ignore[arg-type]
        )
        timed = TimedAgent(agent)
        started = time.perf_counter()
        evaluation = evaluate_replay(timed, samples, categories, products)
        elapsed = time.perf_counter() - started
        results[name] = {
            "complexity": complexity,
            "metrics": {
                key: evaluation[key]
                for key in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "recommended_technical_score",
                    "scenario_metrics",
                )
            },
            "retrieval": retrieval_diagnostics(agent, samples, products),
            "runtime": {
                "elapsed_seconds": round(elapsed, 6),
                "mean_turn_ms": round(statistics.fmean(timed.turn_ms), 6),
                "p95_turn_ms": round(percentile(timed.turn_ms, 0.95), 6),
                "max_turn_ms": round(max(timed.turn_ms), 6),
                "new_model_asset_bytes": 0,
            },
            "behavior_signature": behavior_signature(agent),
            "sessions": evaluation["sessions"],
        }
        print(
            f"{ordinal}/{len(VARIANTS)} {name} "
            f"{evaluation['recommended_technical_score']}",
            flush=True,
        )

    baseline_sessions = results["raw_history"]["sessions"]
    comparisons = {}
    for name, result in results.items():
        deltas = paired_delta(result["sessions"], baseline_sessions)
        interval = bootstrap_mean_interval(deltas, resamples=5000)
        comparisons[name] = {
            "mean_paired_delta": round(statistics.fmean(deltas), 6),
            "bootstrap_95": [round(interval[0], 6), round(interval[1], 6)],
            "randomization_pvalue": round(
                paired_randomization_pvalue(deltas, resamples=5000), 6
            ),
            "wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "losses": sum(value < -1e-12 for value in deltas),
        }

    reward_maps = {
        name: {
            str(session["sample_id"]): session_reward(session)
            for session in result["sessions"]
        }
        for name, result in results.items()
    }
    fold_selections = []
    stitched_rewards: list[float] = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        training_scores = {
            name: statistics.fmean(rewards[item] for item in training)
            for name, rewards in reward_maps.items()
        }
        best = max(training_scores.values())
        selected = min(
            (name for name, score in training_scores.items() if score >= best - 0.005),
            key=lambda name: (VARIANTS[name], -training_scores[name], name),
        )
        outer_rewards = [reward_maps[selected][item] for item in outer]
        stitched_rewards.extend(outer_rewards)
        fold_selections.append(
            {
                "outer_fold": fold_index,
                "selected": selected,
                "training_reward": round(training_scores[selected], 6),
                "outer_reward": round(statistics.fmean(outer_rewards), 6),
            }
        )

    signatures: dict[str, list[str]] = defaultdict(list)
    for name, result in results.items():
        signatures[result["behavior_signature"]].append(name)
    report = {
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "split": manifest["split"],
        "sample_count": len(samples),
        "holdout_accessed": False,
        "candidate_count": len(results),
        "evaluation_control": "fixed_field_bm25_plus_quality_without_label_fitted_reranker",
        "baseline": "raw_history",
        "nested_selected_reward": round(statistics.fmean(stitched_rewards), 6),
        "fold_selections": fold_selections,
        "behavioral_equivalence_groups": [
            names for names in signatures.values() if len(names) > 1
        ],
        "comparisons_to_raw_history": comparisons,
        "variants": results,
        "elapsed_seconds": round(time.perf_counter() - started_all, 6),
        "runtime_profile_field_check": all(
            "ground_truth" not in runtime_profile(sample) for sample in samples
        ),
    }
    output = ROOT / "artifacts/reports/challenger_query_v1.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nested_selected_reward": report["nested_selected_reward"],
                "fold_selections": fold_selections,
                "scores": {
                    name: result["metrics"]["recommended_technical_score"]
                    for name, result in results.items()
                },
                "recall_at_200": {
                    name: result["retrieval"]["any_eligible_turn_recall"][
                        "recall_at_200"
                    ]
                    for name, result in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
