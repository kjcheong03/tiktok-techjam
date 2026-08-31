from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from baseline.retrieval import DenseRetriever
from baseline.state import SessionState
from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from ghostlab.retrieval.fusion import jaccard_at
from ghostlab.retrieval.sparse import FIELD_NAMES, OFFICIAL_WEIGHTS, SparseIndex

ROOT = Path(__file__).resolve().parents[1]
KS = (50, 100, 200)
WEIGHT_VARIANTS = {
    "official": OFFICIAL_WEIGHTS,
    "title_category_heavy": (8.0, 6.0, 2.0, 1.5, 0.5, 0.5),
    "balanced": (4.0, 4.0, 3.0, 3.0, 1.0, 2.0),
}


def recall_at(rankings: dict[str, list[str]], targets: dict[str, str], k: int) -> float:
    return sum(targets[key] in ranking[:k] for key, ranking in rankings.items()) / len(
        targets
    )


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    allowed = set(split["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in allowed]
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    queries: dict[str, str] = {}
    targets: dict[str, str] = {}
    scenarios: dict[str, str] = {}
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        message = initial_message(
            effective, coarse_category(categories[target]), disclosed
        )
        state = SessionState(str(sample["sample_id"]), sample["user_profile"])
        state.observe(message, 1)
        sample_id = str(sample["sample_id"])
        queries[sample_id] = state.build_query()
        targets[sample_id] = target
        scenarios[sample_id] = str(sample["scenario_type"])

    sparse = SparseIndex(ROOT / "data/catalog.jsonl")
    rankings: dict[str, dict[str, list[str]]] = {}
    latencies: dict[str, list[float]] = defaultdict(list)
    for name, weights in WEIGHT_VARIANTS.items():
        route: dict[str, list[str]] = {}
        for sample_id, query in queries.items():
            result = sparse.search(query, 200, weights)
            route[sample_id] = [item.parent_asin for item in result.items]
            latencies[name].append(result.elapsed_ms)
        rankings[name] = route

    for index, field in enumerate(FIELD_NAMES):
        weights = list(OFFICIAL_WEIGHTS)
        weights[index] = 0.0
        name = f"without_{field}"
        rankings[name] = {
            sample_id: [
                item.parent_asin
                for item in sparse.search(query, 200, tuple(weights)).items  # type: ignore[arg-type]
            ]
            for sample_id, query in queries.items()
        }

    dense = DenseRetriever(ROOT / "data/catalog.jsonl")
    query_ids = list(queries)
    encoded = dense.model.encode(
        [queries[key] for key in query_ids],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    score_matrix = encoded @ np.asarray(dense.embeddings).T
    dense_rankings: dict[str, list[str]] = {}
    for row_index, sample_id in enumerate(query_ids):
        scores = score_matrix[row_index]
        indices = np.argpartition(scores, -200)[-200:]
        ordered = indices[np.argsort(scores[indices])[::-1]]
        dense_rankings[sample_id] = [dense.identifiers[int(index)] for index in ordered]
    rankings["dense_minilm"] = dense_rankings

    route_metrics: dict[str, object] = {}
    for name, route in rankings.items():
        scenario_metrics = {}
        for scenario in sorted(set(scenarios.values())):
            ids = [key for key in query_ids if scenarios[key] == scenario]
            scenario_metrics[scenario] = {
                f"recall_at_{k}": round(
                    sum(targets[key] in route[key][:k] for key in ids) / len(ids), 6
                )
                for k in KS
            }
        route_metrics[name] = {
            **{f"recall_at_{k}": round(recall_at(route, targets, k), 6) for k in KS},
            "scenario_metrics": scenario_metrics,
            "warm_latency_ms_mean": round(
                statistics.fmean(latencies.get(name, [0.0])), 6
            ),
        }

    sparse_route = rankings["official"]
    dense_route = rankings["dense_minilm"]
    unique_sparse = sum(
        targets[key] in sparse_route[key][:200]
        and targets[key] not in dense_route[key][:200]
        for key in query_ids
    )
    unique_dense = sum(
        targets[key] in dense_route[key][:200]
        and targets[key] not in sparse_route[key][:200]
        for key in query_ids
    )
    union = sum(
        targets[key] in set(sparse_route[key][:200]) | set(dense_route[key][:200])
        for key in query_ids
    )
    overlaps = [
        jaccard_at(sparse_route[key], dense_route[key], 10) for key in query_ids
    ]
    report = {
        "phase": 3,
        "split": "adaptive_v1",
        "sample_count": len(samples),
        "routes": route_metrics,
        "route_complementarity": {
            "sparse_only_hits_at_200": unique_sparse,
            "dense_only_hits_at_200": unique_dense,
            "union_recall_at_200": round(union / len(query_ids), 6),
            "mean_top10_jaccard": round(
                statistics.fmean(value for value in overlaps if value is not None), 6
            ),
        },
    }
    output = ROOT / "artifacts/reports/phase3_retrieval.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
