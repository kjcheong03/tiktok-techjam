from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np

from evaluator.local_evaluator import catalog_index
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.retrieval.dense_diversity import (
    embedding_mmr_select,
    max_relevance_select,
    view_balanced_select,
)
from ghostlab.retrieval.dense_query_views import build_dense_query_views
from ghostlab.runtime.adaptive_components import DiverseDenseTrack
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.state.baseline_v2 import StateBaselineV2
from ghostlab.state.v2_view import V2SessionController

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare deterministic dense selectors on identical E5 pools."
    )
    parser.add_argument("--source", default="data/public_set.jsonl")
    parser.add_argument("--config", default="configs/adaptive_hybrid_1a_3b_2200_v1.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_dense_diversity_v2.json"
    )
    return parser


def _mean_similarity(ids: list[str], embeddings: dict[str, np.ndarray]) -> float:
    values = [embeddings[item] for item in ids if item in embeddings]
    if len(values) < 2:
        return 0.0
    matrix = np.stack(values)
    similarities = matrix @ matrix.T
    upper = similarities[np.triu_indices(len(matrix), k=1)]
    return float(np.mean(upper)) if len(upper) else 0.0


def main() -> None:
    args = _parser().parse_args()
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = ROOT / args.source
    samples = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    samples = [item for item in samples if item.get("scenario_type") == "browsing"]
    if args.max_samples:
        samples = samples[: args.max_samples]
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    leaves = {
        identifier: " / ".join(values[2:] or values[-1:]).casefold()
        for identifier, values in categories.items()
    }
    config = load_adaptive_hybrid_config(ROOT / args.config)
    track = DiverseDenseTrack(
        ROOT / "data/catalog.jsonl", config.browsing, project_root=ROOT
    )
    index = track.index
    embedding_map = {
        identifier: np.asarray(index.embeddings[row])
        for row, identifier in enumerate(index.identifiers)
    }
    rows: list[dict[str, object]] = []
    for sample in samples:
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        state = StateBaselineV2(observation.session_id, {})
        state.observe(observation.user_message, 1)
        query = state.build_coverage_adaptive_query() or observation.user_message
        context = V2SessionController(state).snapshot(
            query_text=query,
            turn=1,
            current_message=observation.user_message,
        )
        rankings: dict[str, list[str]] = {}
        relevance: dict[str, float] = {}
        candidates: list[str] = []
        for view in build_dense_query_views(context):
            result = index.search(view.query_text, config.browsing.retrieval_per_view)
            rankings[view.name] = [item.parent_asin for item in result.items]
            for item in result.items:
                candidates.append(item.parent_asin)
                relevance[item.parent_asin] = max(
                    relevance.get(item.parent_asin, 0.0),
                    float(item.normalized_score or 0.0),
                )
        selections = {
            "multiview_max_relevance": max_relevance_select(
                candidates, relevance, output_k=config.browsing.output_k
            ),
            "view_balanced": view_balanced_select(
                rankings, relevance, output_k=config.browsing.output_k
            ),
            "embedding_mmr": embedding_mmr_select(
                candidates,
                relevance,
                embedding_map,
                output_k=config.browsing.output_k,
                relevance_weight=config.browsing.mmr_relevance_weight,
            ),
        }
        target = str(sample["ground_truth"]["parent_asin"])
        category_heads = {
            name: [leaves.get(item, "unknown") for item in ids[:50]]
            for name, ids in selections.items()
        }
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "category_bucket": sample.get("category_bucket"),
                "target": target,
                "selectors": {
                    name: {
                        "ids": ids,
                        "target_rank": (ids.index(target) + 1 if target in ids else None),
                        "category_count_50": len({leaves.get(item, "") for item in ids[:50]}),
                        "maximum_category_share_50": max(
                            (
                                category_heads[name].count(item)
                                / max(1, len(category_heads[name]))
                                for item in set(category_heads[name])
                            ),
                            default=0.0,
                        ),
                        "mean_similarity_50": _mean_similarity(ids[:50], embedding_map),
                    }
                    for name, ids in selections.items()
                },
            }
        )

    summary: dict[str, object] = {}
    for selector in (
        "multiview_max_relevance",
        "view_balanced",
        "embedding_mmr",
    ):
        selector_rows = [row["selectors"][selector] for row in rows]  # type: ignore[index]
        summary[selector] = {
            f"recall_at_{depth}": sum(
                item["target_rank"] is not None and item["target_rank"] <= depth
                for item in selector_rows
            )
            / max(1, len(selector_rows))
            for depth in (50, 100, 200)
        }
        summary[selector].update(  # type: ignore[union-attr]
            {
                "mean_category_count_50": statistics.fmean(
                    float(item["category_count_50"]) for item in selector_rows
                )
                if selector_rows
                else 0.0,
                "mean_pairwise_similarity_50": statistics.fmean(
                    float(item["mean_similarity_50"]) for item in selector_rows
                )
                if selector_rows
                else 0.0,
            }
        )
    control = summary["multiview_max_relevance"]
    decisions = {}
    for selector in ("view_balanced", "embedding_mmr"):
        candidate = summary[selector]
        decisions[selector] = {
            "recall_200_preserved": candidate["recall_at_200"] >= control["recall_at_200"],  # type: ignore[index]
            "redundancy_reduced": candidate["mean_pairwise_similarity_50"] < control["mean_pairwise_similarity_50"],  # type: ignore[index]
            "category_coverage_improved": candidate["mean_category_count_50"] > control["mean_category_count_50"],  # type: ignore[index]
        }
    payload = {
        "schema_version": 2,
        "evaluation_scope": "development_diagnostic_not_independent_claim",
        "source": str(source.relative_to(ROOT)),
        "browsing_sessions": len(rows),
        "identical_retrieval_pools": True,
        "summary": summary,
        "decisions": decisions,
        "per_session": rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_session"}, indent=2))


if __name__ == "__main__":
    main()
