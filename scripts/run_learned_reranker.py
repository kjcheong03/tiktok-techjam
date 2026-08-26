from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
)
from ghostlab.research.replay import ReplayEnvironment, session_reward
from ghostlab.retrieval.learned import (
    CandidateFeatureStore,
    LearnedLinearReranker,
    LinearRerankerModel,
    PairwiseExample,
    fit_pairwise_linear,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
QUESTION_ORDER = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)


def collect_examples(
    samples: dict[str, dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    quality: CatalogQualityReranker,
    feature_store: CandidateFeatureStore,
) -> tuple[dict[str, list[PairwiseExample]], dict[str, int]]:
    by_sample: dict[str, list[PairwiseExample]] = defaultdict(list)
    queries = positives = 0
    for sample_id, sample in samples.items():
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        messages: list[str] = []
        target = str(sample["ground_truth"]["parent_asin"])
        while not environment.done:
            messages.append(observation.user_message)
            query = ". ".join(messages)
            ranking = [
                item.parent_asin
                for item in sparse.search(query, 200, FIELD_WEIGHTS).items
            ]
            ranking = quality.rerank(ranking, weight=0.2, rerank_k=50)
            head = ranking[:50]
            queries += 1
            if target in head:
                positives += 1
                count = len(head)
                target_rank = head.index(target) + 1
                target_base = (
                    1.0 if count == 1 else 1.0 - (target_rank - 1) / max(1, count - 1)
                )
                target_features = feature_store.features(query, target)
                negatives = list(dict.fromkeys([*head[:20], *head[20:50:5]]))
                for identifier in negatives:
                    if identifier == target:
                        continue
                    rank = head.index(identifier) + 1
                    negative_base = (
                        1.0 if count == 1 else 1.0 - (rank - 1) / max(1, count - 1)
                    )
                    negative_features = feature_store.features(query, identifier)
                    by_sample[sample_id].append(
                        PairwiseExample(
                            base_margin=target_base - negative_base,
                            feature_delta=tuple(
                                left - right
                                for left, right in zip(
                                    target_features,
                                    negative_features,
                                    strict=True,
                                )
                            ),
                        )
                    )
            turn = observation.turn
            question = QUESTION_ORDER[turn - 1] if turn <= len(QUESTION_ORDER) else None
            next_observation = environment.step(
                {
                    "message": "training trajectory",
                    "ask_attribute": question,
                    "recommendations": [],
                }
            )
            if next_observation is not None:
                observation = next_observation
    return dict(by_sample), {
        "trajectory_queries": queries,
        "queries_with_target_in_top50": positives,
        "samples_with_training_pairs": len(by_sample),
        "training_pairs": sum(len(values) for values in by_sample.values()),
    }


def agent(
    feature_store: CandidateFeatureStore, model: LinearRerankerModel
) -> ExperimentalAgent:
    return ExperimentalAgent(
        ROOT / "data/catalog.jsonl",
        state_variant="raw_history",
        question_variant="sequence",
        question_order=QUESTION_ORDER,
        negative_evidence=False,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        learned_reranker=LearnedLinearReranker(feature_store, model),
    )


def summarized_metrics(sessions: list[dict]) -> dict[str, object]:
    overall = metric_summary(sessions)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "recommended_technical_score": round(
            statistics.fmean(session_reward(session) for session in sessions), 6
        ),
        "scenario_metrics": {
            name: metric_summary(values) for name, values in sorted(grouped.items())
        },
    }


def main() -> None:
    started = time.perf_counter()
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    sparse = SparseIndex(ROOT / "data/catalog.jsonl")
    quality = CatalogQualityReranker(ROOT / "data/catalog.jsonl")
    feature_store = CandidateFeatureStore(ROOT / "data/catalog.jsonl")
    examples, collection = collect_examples(
        samples, categories, products, sparse, quality, feature_store
    )
    print(json.dumps(collection, sort_keys=True), flush=True)

    folds = []
    oof_sessions = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        training_examples = [
            example for sample_id in training for example in examples.get(sample_id, [])
        ]
        model = fit_pairwise_linear(training_examples)
        fold_samples = [samples[sample_id] for sample_id in sorted(outer)]
        result = evaluate(
            agent(feature_store, model),
            fold_samples,
            catalog_ids,
            categories,
            products,
        )
        oof_sessions.extend(result["sessions"])
        folds.append(
            {
                "outer_fold": fold_index,
                "training_samples": len(training),
                "training_pairs": len(training_examples),
                "model": {
                    "weights": list(model.weights),
                    "l2": model.l2,
                },
                "outer_metrics": {
                    key: result[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "recommended_technical_score",
                    )
                },
            }
        )
        print(
            f"fold {fold_index} {result['recommended_technical_score']}",
            flush=True,
        )

    all_examples = [
        example for sample_id in sorted(examples) for example in examples[sample_id]
    ]
    full_model = fit_pairwise_linear(all_examples)
    full_result = evaluate(
        agent(feature_store, full_model),
        [samples[sample_id] for sample_id in sorted(samples)],
        catalog_ids,
        categories,
        products,
    )
    baseline = json.loads(
        (ROOT / "artifacts/reports/phase18_quality_priors.json").read_text()
    )["variants"]["title2_category8_feature4_quality_0.2"]
    report = {
        "phase": 19,
        "gate": "nested_pairwise_linear_reranker",
        "split": "nested_v1_oof",
        "holdout_accessed": False,
        "collection": collection,
        "optimizer": {
            "l2": 0.1,
            "learning_rate": 0.5,
            "iterations": 500,
            "hpo_performed": False,
        },
        "baseline_metrics": baseline["metrics"],
        "oof_metrics": summarized_metrics(oof_sessions),
        "oof_sessions": oof_sessions,
        "folds": folds,
        "full_training_model": {
            "weights": list(full_model.weights),
            "l2": full_model.l2,
            "training_pairs": full_model.training_pairs,
        },
        "full_training_metrics": {
            key: full_result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
                "scenario_metrics",
            )
        },
        "full_training_sessions": full_result["sessions"],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    output = ROOT / "artifacts/reports/phase19_learned_reranker.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "oof_metrics": report["oof_metrics"],
                "full_training_metrics": report["full_training_metrics"],
                "full_training_model": report["full_training_model"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
