from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baseline.agent import BaselineAgent
from baseline.retrieval import DenseRetriever, KeywordRetriever
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent as OfficialAgent


VARIANTS = (
    "official_keyword",
    "keyword_current_turn",
    "keyword_state",
    "dense_current_turn",
    "dense_state",
    "hybrid_current_turn",
    "hybrid_state",
)


def _table(results: dict, sample_count: int) -> str:
    lines = [
        f"# Baseline comparison - {sample_count} public sessions",
        "",
        "## Configuration",
        "",
        "- Keyword: organizer SQLite FTS5/BM25 retriever.",
        "- Dense: `sentence-transformers/all-MiniLM-L6-v2`, exact cosine search on CPU.",
        "- State: deterministic slot accumulation with provenance and intent-override invalidation.",
        "- Hybrid: top-200 keyword and dense rankings fused with RRF (`k=60`), without tuning.",
        "- All variants use the unchanged organizer evaluator with at most 10 turns and 10 recommendations.",
        "- Evaluation seconds are warm-cache diagnostics and are not a controlled speed benchmark.",
        "",
        "Reproduce with: `uv run python -m scripts.run_baselines`",
        "",
        "## Overall results",
        "",
        "| Variant | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore | Eval seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        result = results[name]
        lines.append(
            f"| {name} | {result['hit_rate_at_10']:.6f} | {result['mrr']:.6f} | "
            f"{result['mttc']:.6f} | {result['efficiency']:.6f} | "
            f"{result['recommended_technical_score']:.6f} | {result['evaluation_seconds']:.3f} |"
        )

    lines.extend(["", "## Scenario Technical Inputs", ""])
    for name in VARIANTS:
        lines.extend(
            [
                f"### {name}",
                "",
                "| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for scenario, metrics in results[name]["scenario_metrics"].items():
            lines.append(
                f"| {scenario} | {metrics['sample_count']} | {metrics['hit_rate_at_10']:.6f} | "
                f"{metrics['mrr']:.6f} | {metrics['mttc']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible baseline ablations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/baseline_results.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    keyword = KeywordRetriever(args.catalog)
    dense = DenseRetriever(args.catalog)
    agents = {
        "official_keyword": OfficialAgent(args.catalog),
        "keyword_current_turn": BaselineAgent(
            mode="keyword", stateful=False, keyword=keyword, dense=None
        ),
        "keyword_state": BaselineAgent(
            mode="keyword", stateful=True, keyword=keyword, dense=None
        ),
        "dense_current_turn": BaselineAgent(
            mode="dense", stateful=False, keyword=keyword, dense=dense
        ),
        "dense_state": BaselineAgent(
            mode="dense", stateful=True, keyword=keyword, dense=dense
        ),
        "hybrid_current_turn": BaselineAgent(
            mode="hybrid", stateful=False, keyword=keyword, dense=dense
        ),
        "hybrid_state": BaselineAgent(
            mode="hybrid", stateful=True, keyword=keyword, dense=dense
        ),
    }

    results: dict[str, dict] = {}
    for name in VARIANTS:
        started = time.perf_counter()
        result = evaluate(agents[name], samples, catalog_ids, categories, products)
        elapsed = time.perf_counter() - started
        results[name] = {**result, "evaluation_seconds": round(elapsed, 6)}
        print(
            f"{name}: score={result['recommended_technical_score']:.6f} "
            f"hit={result['hit_rate_at_10']:.6f} mrr={result['mrr']:.6f} "
            f"mttc={result['mttc']:.6f} seconds={elapsed:.3f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_table(results, len(samples)), encoding="utf-8")
    print(f"Wrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
