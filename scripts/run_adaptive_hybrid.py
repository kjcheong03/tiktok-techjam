from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    load_jsonl,
    metric_summary,
)
from ghostlab.research.replay import evaluate_shared
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the complete adaptive hybrid"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--lineage-manifest")
    parser.add_argument(
        "--partition", choices=("development", "holdout", "all"), default="all"
    )
    parser.add_argument("--config", default="configs/adaptive_hybrid_1a_3b_v1.json")
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_hybrid_1a_3b_v1.json"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="optional deterministic prefix for bounded smoke validation",
    )
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument("--semantic-rerank-k", type=int)
    parser.add_argument("--buying-keyword-weight", type=float)
    parser.add_argument("--profile-weight", type=float)
    parser.add_argument("--browsing-safe-weight", type=float)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    root = Path(__file__).resolve().parents[1]
    catalog = root / args.catalog
    config = load_adaptive_hybrid_config(root / args.config)
    if args.semantic_weight is not None or args.semantic_rerank_k is not None:
        semantic = config.semantic_ranker.model_copy(
            update={
                "weight": (
                    args.semantic_weight
                    if args.semantic_weight is not None
                    else config.semantic_ranker.weight
                ),
                "rerank_k": (
                    args.semantic_rerank_k
                    if args.semantic_rerank_k is not None
                    else config.semantic_ranker.rerank_k
                ),
            }
        )
        config = config.model_copy(update={"semantic_ranker": semantic})
        config = AdaptiveHybridConfig.model_validate(config.model_dump())
    if args.buying_keyword_weight is not None:
        support_weight = (1.0 - args.buying_keyword_weight) / 2.0
        merger = config.merger.model_copy(
            update={
                "buying_keyword_weight": args.buying_keyword_weight,
                "buying_category_weight": support_weight,
                "buying_vector_weight": support_weight,
            }
        )
        config = config.model_copy(update={"merger": merger})
    if args.profile_weight is not None:
        adaptation = config.runtime_adaptation.model_copy(
            update={"profile_weight": args.profile_weight}
        )
        config = config.model_copy(update={"runtime_adaptation": adaptation})
    if args.browsing_safe_weight is not None:
        browsing = config.browsing.model_copy(
            update={"safe_ranker_weight": args.browsing_safe_weight}
        )
        config = config.model_copy(update={"browsing": browsing})
    config = AdaptiveHybridConfig.model_validate(config.model_dump())
    agent = AdaptiveHybridAgent(catalog, config, project_root=root)
    dataset_paths = args.datasets or ["data/public_set.jsonl"]
    origins: dict[str, str]
    if args.lineage_manifest is not None:
        corpus = load_adaptive_training_corpus(root, dataset_paths)
        manifest = load_lineage_manifest(root / args.lineage_manifest, corpus)
        if args.partition != "all":
            corpus = subset_corpus(corpus, manifest, args.partition)
        samples = [corpus.samples[item] for item in sorted(corpus.samples)]
        origins = corpus.origins
    elif len(dataset_paths) > 1:
        corpus = load_adaptive_training_corpus(root, dataset_paths)
        samples = [corpus.samples[item] for item in sorted(corpus.samples)]
        origins = corpus.origins
    else:
        samples = load_jsonl(root / dataset_paths[0])
        origins = {str(item["sample_id"]): dataset_paths[0] for item in samples}
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    _, categories, products = catalog_index(catalog)
    result = evaluate_shared(
        agent,
        samples,
        categories,
        products,
        catalog_path=catalog,
    )
    session_order = list(dict.fromkeys(trace.session_id for trace in agent.traces))
    if len(session_order) != len(samples):
        raise RuntimeError("runtime trace/session alignment failed")
    survival_rows: list[dict[str, object]] = []
    for sample, session_id in zip(samples, session_order, strict=True):
        target = str(sample["ground_truth"]["parent_asin"])
        snapshots = [
            item for item in agent.candidate_snapshots if item.session_id == session_id
        ]
        traces = [item for item in agent.traces if item.session_id == session_id]
        survival_rows.append(
            {
                "sample_id": sample["sample_id"],
                "source": origins[str(sample["sample_id"])],
                "scenario_type": sample["scenario_type"],
                "pre_authority_seen": any(
                    target in item.pre_authority_candidates for item in snapshots
                ),
                "post_authority_seen": any(
                    target in item.candidates for item in snapshots
                ),
                "removed_as_confirmed_contradiction": any(
                    target in item.authority_removed_ids for item in snapshots
                ),
                "final_top10_seen": any(target in item.top_ids for item in traces),
            }
        )
    source_sessions: dict[str, list[dict[str, object]]] = {}
    route_sessions: dict[str, list[dict[str, object]]] = {}
    for row in result["sessions"]:
        source_sessions.setdefault(origins[str(row["sample_id"])], []).append(row)
    for row, session_id in zip(result["sessions"], session_order, strict=True):
        session_traces = [
            trace for trace in agent.traces if trace.session_id == session_id
        ]
        route = session_traces[0].route if session_traces else "unknown"
        route_sessions.setdefault(route, []).append(row)
    target_audit = {
        "sample_count": len(survival_rows),
        "pre_authority_recall": sum(
            bool(row["pre_authority_seen"]) for row in survival_rows
        )
        / max(1, len(survival_rows)),
        "post_authority_recall": sum(
            bool(row["post_authority_seen"]) for row in survival_rows
        )
        / max(1, len(survival_rows)),
        "confirmed_target_removal_count": sum(
            bool(row["removed_as_confirmed_contradiction"]) for row in survival_rows
        ),
        "final_top10_recall": sum(
            bool(row["final_top10_seen"]) for row in survival_rows
        )
        / max(1, len(survival_rows)),
        "rows": survival_rows,
    }
    result["evaluation_partition"] = args.partition
    result["dataset_sources"] = dataset_paths
    result["source_metrics"] = {
        source: metric_summary(rows) for source, rows in sorted(source_sessions.items())
    }
    result["route_metrics"] = {
        route: metric_summary(rows) for route, rows in sorted(route_sessions.items())
    }
    result["target_survival_audit"] = target_audit
    result["adaptive_runtime"] = {
        "config_sha256": agent.config_sha256,
        "trace_count": len(agent.traces),
        "route_counts": {
            route: sum(trace.route == route for trace in agent.traces)
            for route in ("buying", "browsing")
        },
        "overload_count": sum(trace.overloaded for trace in agent.traces),
        "fallback_count": sum(
            trace.fallback_reason is not None for trace in agent.traces
        ),
        "output_constraint_violation_count": sum(
            trace.output_constraint_violations for trace in agent.traces
        ),
        "overload_cutoff_trace_violations": sum(
            trace.overloaded
            and (
                not trace.safe_ranker_executed
                or trace.normal_union_executed
                or trace.semantic_executed
            )
            for trace in agent.traces
        ),
        "semantic_activation_count": sum(
            not trace.semantic_backend.startswith("skipped:")
            and trace.semantic_backend != "not_run"
            for trace in agent.traces
        ),
        "semantic_skip_count": sum(
            trace.semantic_backend.startswith("skipped:") for trace in agent.traces
        ),
        "semantic_change_count": sum(trace.semantic_changed for trace in agent.traces),
        "semantic_failure_counts": {
            reason: sum(
                trace.semantic_failure_reason == reason for trace in agent.traces
            )
            for reason in sorted(
                {
                    trace.semantic_failure_reason
                    for trace in agent.traces
                    if trace.semantic_failure_reason is not None
                }
            )
        },
        "profile_activation_count": sum(trace.profile_active for trace in agent.traces),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sessions"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
