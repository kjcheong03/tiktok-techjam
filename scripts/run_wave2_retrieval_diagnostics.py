from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from ghostlab.retrieval.diversify import DiversificationContext, FacetMMRDiversifier
from ghostlab.retrieval.pseudo_relevance import CatalogPseudoRelevanceFeedback
from ghostlab.retrieval.sparse import SparseIndex

CHAMPION_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
KS = (10, 50, 100, 200)


def rank(ranking: list[str], target: str) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def recall(rankings: list[list[str]], targets: list[str], k: int) -> float:
    return sum(
        target in ranking[:k] for ranking, target in zip(rankings, targets, strict=True)
    ) / len(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wave 2 core retrieval gates")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    allowed = set(json.loads(args.split.read_text(encoding="utf-8"))["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in allowed]
    _, categories, products = catalog_index(args.catalog)
    sparse = SparseIndex(args.catalog)
    prf = CatalogPseudoRelevanceFeedback(args.catalog)
    mmr = FacetMMRDiversifier(args.catalog)
    targets: list[str] = []
    base_rankings: list[list[str]] = []
    prf_rankings: list[list[str]] = []
    mmr_rankings: list[list[str]] = []
    expansion_counts: list[int] = []
    base_coverage: list[int] = []
    mmr_coverage: list[int] = []
    latencies: list[float] = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        query = initial_message(
            effective, coarse_category(categories[target]), disclosed=set()
        )
        started = time.perf_counter()
        base = [
            item.parent_asin
            for item in sparse.search(query, 200, CHAMPION_WEIGHTS).items
        ]
        expansion = prf.expand(query, base)
        expanded = [
            item.parent_asin
            for item in sparse.search(
                expansion.expanded_query, 200, CHAMPION_WEIGHTS
            ).items
        ]
        diversified = mmr.rerank(
            base, DiversificationContext(turn=1, active_constraint_count=0)
        ).ranking
        latencies.append((time.perf_counter() - started) * 1000.0)
        targets.append(target)
        base_rankings.append(base)
        prf_rankings.append(expanded)
        mmr_rankings.append(diversified)
        expansion_counts.append(len(expansion.terms))
        base_coverage.append(mmr.facet_coverage(base))
        mmr_coverage.append(mmr.facet_coverage(diversified))

    report = {
        "schema_version": 1,
        "evaluation_label": "F0 first-turn mechanism gate; not OOF champion evidence",
        "split": args.split.stem,
        "sample_count": len(samples),
        "holdout_accessed": False,
        "base_recall": {
            f"at_{k}": round(recall(base_rankings, targets, k), 6) for k in KS
        },
        "prf": {
            "recall": {
                f"at_{k}": round(recall(prf_rankings, targets, k), 6) for k in KS
            },
            "unique_rescues_at_200": sum(
                target not in base[:200] and target in candidate[:200]
                for base, candidate, target in zip(
                    base_rankings, prf_rankings, targets, strict=True
                )
            ),
            "unique_losses_at_200": sum(
                target in base[:200] and target not in candidate[:200]
                for base, candidate, target in zip(
                    base_rankings, prf_rankings, targets, strict=True
                )
            ),
            "mean_expansion_terms": round(statistics.fmean(expansion_counts), 6),
        },
        "facet_mmr": {
            "recall": {
                f"at_{k}": round(recall(mmr_rankings, targets, k), 6) for k in KS
            },
            "mean_facet_coverage_at_10_base": round(statistics.fmean(base_coverage), 6),
            "mean_facet_coverage_at_10_candidate": round(
                statistics.fmean(mmr_coverage), 6
            ),
            "target_rank_improved": sum(
                (rank(candidate, target) or 10**9) < (rank(base, target) or 10**9)
                for base, candidate, target in zip(
                    base_rankings, mmr_rankings, targets, strict=True
                )
            ),
            "target_rank_regressed": sum(
                (rank(candidate, target) or 10**9) > (rank(base, target) or 10**9)
                for base, candidate, target in zip(
                    base_rankings, mmr_rankings, targets, strict=True
                )
            ),
        },
        "combined_warm_latency_ms": {
            "mean": round(statistics.fmean(latencies), 6),
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 6),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
