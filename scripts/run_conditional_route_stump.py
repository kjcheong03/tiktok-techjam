from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from baseline.retrieval import DenseRetriever
from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.policy.signals import retrieval_signals
from ghostlab.research.replay import ReplayEnvironment, session_reward
from ghostlab.research.route_stump import fit_route_stump
from ghostlab.retrieval.sparse import OFFICIAL_WEIGHTS, SparseIndex, query_terms
from ghostlab.state.memory import ConversationState

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("keyword", "dense", "rrf", "weighted")


def main() -> None:
    report = json.loads(
        (ROOT / "artifacts/reports/phase10_route_policy.json").read_text()
    )
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    sparse = SparseIndex(ROOT / "data/catalog.jsonl")
    features = {}
    queries = {}
    sparse_rankings = {}
    for sample_id, sample in samples.items():
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        state = ConversationState(observation.session_id, sample["user_profile"])
        state.observe(observation.user_message, 1)
        query = state.build_query()
        ranking = sparse.search(query, 200, OFFICIAL_WEIGHTS)
        queries[sample_id] = query
        sparse_rankings[sample_id] = [item.parent_asin for item in ranking.items]
        signals = retrieval_signals(
            [item.raw_score for item in ranking.items if item.raw_score is not None]
        )
        profile = sample["user_profile"]
        features[sample_id] = {
            "active_slots": float(len(state.active_values())),
            "average_prior_rating": float(profile.get("average_prior_rating") or 0.0),
            "candidate_count": float(signals.candidate_count),
            "critical_rater": float(
                str(profile.get("rating_style", "")).casefold() == "critical"
            ),
            "normalized_entropy": float(signals.normalized_entropy or 0.0),
            "preference_tag_count": float(len(profile.get("preference_tags") or [])),
            "query_characters": float(len(query)),
            "query_terms": float(len(query_terms(query))),
            "top1_margin": float(signals.top1_margin or 0.0),
        }

    dense = DenseRetriever(ROOT / "data/catalog.jsonl")
    ordered_ids = sorted(queries)
    query_embeddings = dense.model.encode(
        [queries[sample_id] for sample_id in ordered_ids],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    for sample_id, query_embedding in zip(ordered_ids, query_embeddings, strict=True):
        scores = np.asarray(dense.embeddings @ query_embedding)
        count = min(200, len(scores))
        candidate_indices = np.argpartition(scores, -count)[-count:]
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        dense_ids = [dense.identifiers[int(index)] for index in ranked_indices]
        top_scores = scores[ranked_indices[:10]]
        overlap = retrieval_signals(
            [],
            sparse_ids=sparse_rankings[sample_id],
            dense_ids=dense_ids,
        ).sparse_dense_top10_overlap
        features[sample_id].update(
            {
                "dense_top1_margin": float(top_scores[0] - top_scores[1]),
                "dense_top1_score": float(top_scores[0]),
                "dense_top10_mean": float(np.mean(top_scores)),
                "sparse_dense_top10_overlap": float(overlap or 0.0),
            }
        )

    rewards = {
        route: {
            str(session["sample_id"]): session_reward(session)
            for session in report["sessions"][route]
        }
        for route in ROUTES
    }
    folds = []
    predictions = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        stump = fit_route_stump(training, rewards, features, ROUTES)
        fold_predictions = []
        for sample_id in sorted(outer):
            route = stump.predict(features[sample_id])
            reward = rewards[route][sample_id]
            fold_predictions.append((sample_id, route, reward))
            predictions.append((sample_id, route, reward))
        folds.append(
            {
                "outer_fold": fold_index,
                "stump": {
                    "default_route": stump.default_route,
                    "feature": stump.feature,
                    "threshold": stump.threshold,
                    "lower_route": stump.lower_route,
                    "upper_route": stump.upper_route,
                },
                "outer_reward": round(
                    statistics.fmean(item[2] for item in fold_predictions), 6
                ),
                "route_counts": dict(
                    sorted(Counter(item[1] for item in fold_predictions).items())
                ),
            }
        )

    keyword_reward = statistics.fmean(
        rewards["keyword"][sample_id] for sample_id in adaptive_ids
    )
    output_report = {
        "phase": 14,
        "gate": "observable_conditional_route_stump",
        "split": "nested_v1_oof",
        "sample_count": len(samples),
        "feature_names": sorted(next(iter(features.values()))),
        "holdout_accessed": False,
        "constant_keyword_reward": round(keyword_reward, 6),
        "conditional_oof_reward": round(
            statistics.fmean(item[2] for item in predictions), 6
        ),
        "conditional_delta": round(
            statistics.fmean(item[2] for item in predictions) - keyword_reward, 6
        ),
        "route_counts": dict(sorted(Counter(item[1] for item in predictions).items())),
        "folds": folds,
    }
    output = ROOT / "artifacts/reports/phase14_conditional_route.json"
    output.write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
