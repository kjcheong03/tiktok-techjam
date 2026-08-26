from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import ReplayEnvironment, session_reward
from ghostlab.research.route_policy import RouteFeatures, fit_route_table
from ghostlab.runtime.experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("keyword", "dense", "rrf", "weighted")
FEATURE_SETS = (
    ("has_initial_constraint",),
    ("has_initial_constraint", "critical_rater"),
)


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = set(nested["adaptive_sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in adaptive_ids]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    existing = json.loads(
        (ROOT / "artifacts/reports/phase4_5_ablations.json").read_text()
    )["variants"]["multi_fixed"]
    route_results = {"keyword": existing}
    latency_seconds = {"keyword": None}
    for route in ROUTES[1:]:
        started = time.perf_counter()
        route_results[route] = evaluate(
            ExperimentalAgent(
                ROOT / "data/catalog.jsonl",
                state_variant="multi",
                question_variant="fixed",
                retrieval_route=route,
            ),
            samples,
            catalog_ids,
            categories,
            products,
        )
        latency_seconds[route] = round(time.perf_counter() - started, 6)
        print(route, route_results[route]["recommended_technical_score"], flush=True)

    rewards = {
        route: {
            str(session["sample_id"]): session_reward(session)
            for session in result["sessions"]
        }
        for route, result in route_results.items()
    }
    features = {}
    for sample in samples:
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        state = ConversationState(observation.session_id, sample["user_profile"])
        state.observe(observation.user_message, 1)
        features[str(sample["sample_id"])] = RouteFeatures(
            has_initial_constraint=len(state.active_values()) > 1,
            critical_rater=str(
                sample["user_profile"].get("rating_style", "")
            ).casefold()
            == "critical",
        )

    table_reports = {}
    for feature_names in FEATURE_SETS:
        predictions = []
        folds = []
        for outer_ids in nested["outer_folds"]:
            outer = set(outer_ids)
            table = fit_route_table(
                adaptive_ids - outer, rewards, features, ROUTES, feature_names
            )
            fold_rewards = []
            for sample_id in outer:
                route = table.predict(features[sample_id])
                reward = rewards[route][sample_id]
                predictions.append((sample_id, route, reward))
                fold_rewards.append(reward)
            folds.append(statistics.fmean(fold_rewards))
        key = "+".join(feature_names)
        table_reports[key] = {
            "mean_oof_reward": round(
                statistics.fmean(reward for _, _, reward in predictions), 6
            ),
            "fold_rewards": [round(value, 6) for value in folds],
            "route_counts": dict(
                sorted(Counter(route for _, route, _ in predictions).items())
            ),
        }

    oracle = statistics.fmean(
        max(rewards[route][sample_id] for route in ROUTES) for sample_id in adaptive_ids
    )
    route_summary = {
        route: {
            key: value
            for key, value in result.items()
            if key in {"hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"}
        }
        for route, result in route_results.items()
    }
    unique_rescues = {
        route: sum(
            rewards[route][sample_id] > rewards["keyword"][sample_id]
            for sample_id in adaptive_ids
        )
        for route in ROUTES[1:]
    }
    report = {
        "phase": 10,
        "split": "nested_v1_oof",
        "sample_count": len(samples),
        "route_summary": route_summary,
        "wall_seconds": latency_seconds,
        "route_oracle_mean_reward": round(oracle, 6),
        "unique_rescued_sessions_vs_keyword": unique_rescues,
        "route_tables": table_reports,
        "sessions": {
            route: result["sessions"] for route, result in route_results.items()
        },
    }
    output = ROOT / "artifacts/reports/phase10_route_policy.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "sessions"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
