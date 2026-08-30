from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the complete adaptive hybrid")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--config", default="configs/adaptive_hybrid_1a_3b_v1.json"
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_hybrid_1a_3b_v1.json"
    )
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument("--semantic-rerank-k", type=int)
    parser.add_argument("--buying-keyword-weight", type=float)
    parser.add_argument("--profile-weight", type=float)
    parser.add_argument("--browsing-safe-weight", type=float)
    args = parser.parse_args()
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
    samples = load_jsonl(root / args.dataset)
    identifiers, categories, products = catalog_index(catalog)
    result = evaluate(
        cast(Agent, agent), samples, identifiers, categories, products
    )
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
        "semantic_activation_count": sum(
            not trace.semantic_backend.startswith("skipped:")
            and trace.semantic_backend != "not_run"
            for trace in agent.traces
        ),
        "semantic_skip_count": sum(
            trace.semantic_backend.startswith("skipped:") for trace in agent.traces
        ),
        "semantic_change_count": sum(trace.semantic_changed for trace in agent.traces),
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
