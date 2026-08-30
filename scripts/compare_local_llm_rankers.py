from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import cast

from evaluator.local_evaluator import catalog_index, evaluate
from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen2.5-0.5b-instruct": {
        "path": "artifacts/cache/models/qwen2.5-0.5b-instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
    "smollm2-360m-instruct": {
        "path": "artifacts/cache/models/smollm2-360m-instruct",
        "revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
    },
    "qwen3-0.6b": {
        "path": "artifacts/cache/models/qwen3-0.6b",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
}
DEFAULT_DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if ".cache" in item.parts:
            continue
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded same-pipeline local causal-LLM ranking comparison"
    )
    parser.add_argument(
        "--config", default="configs/adaptive_hybrid_1a_3b_1650_final_v1.json"
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--depth", action="append", type=int, dest="depths")
    parser.add_argument("--weight", action="append", type=float, dest="weights")
    parser.add_argument(
        "--output", default="artifacts/reports/local_llm_ranker_comparison_v1.json"
    )
    parser.add_argument(
        "--selected-config-output",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    depths = tuple(args.depths or (10, 20, 30))
    weights = tuple(args.weights or (0.20, 0.35, 0.50))
    if any(not 2 <= depth <= 50 for depth in depths):
        raise ValueError("semantic depths must be between 2 and 50")
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("semantic weights must be between zero and one")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    base = load_adaptive_hybrid_config(ROOT / args.config)
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    complete = load_adaptive_training_corpus(ROOT, datasets)
    manifest = load_lineage_manifest(ROOT / args.lineage_manifest, complete)
    development = subset_corpus(complete, manifest, "development")
    samples = [
        item
        for _, item in sorted(development.samples.items())
        if item.get("scenario_type") == "browsing"
    ][: args.max_samples]
    catalog_path = ROOT / "data/catalog.jsonl"
    identifiers, categories, products = catalog_index(catalog_path)
    results: list[dict[str, object]] = []
    unavailable: list[dict[str, str]] = []
    qwen_selected_depth: int | None = None
    qwen_selected_weight: float | None = None
    for model_id, definition in MODELS.items():
        model_path = ROOT / definition["path"]
        if not model_path.is_dir():
            unavailable.append({"model_id": model_id, "reason": "asset_missing"})
            continue
        model_hash = _tree_sha256(model_path)
        model_depths = (
            depths
            if model_id == "qwen2.5-0.5b-instruct"
            else ((qwen_selected_depth,) if qwen_selected_depth is not None else ())
        )
        model_weights = (
            weights
            if model_id == "qwen2.5-0.5b-instruct"
            else ((qwen_selected_weight,) if qwen_selected_weight is not None else ())
        )
        for depth in model_depths:
            for weight in model_weights:
                assert weight is not None
                semantic = base.semantic_ranker.model_copy(
                    update={
                        "backend": (
                            "qwen_causal_relevance"
                            if model_id == "qwen2.5-0.5b-instruct"
                            else "local_causal_relevance"
                        ),
                        "model_id": model_id,
                        "model_path": definition["path"],
                        "model_revision": definition["revision"],
                        "model_sha256": model_hash,
                        "rerank_k": depth,
                        "weight": weight,
                        "timeout_ms": 120000,
                    }
                )
                config = base.model_copy(update={"semantic_ranker": semantic})
                started = time.perf_counter()
                agent = AdaptiveHybridAgent(catalog_path, config, project_root=ROOT)
                result = evaluate(
                    cast(Agent, agent), samples, identifiers, categories, products
                )
                elapsed = time.perf_counter() - started
                semantic_traces = [
                    trace
                    for trace in agent.traces
                    if not trace.semantic_backend.startswith("skipped:")
                    and trace.semantic_backend != "not_run"
                ]
                semantic_latencies = [
                    trace.semantic_elapsed_ms for trace in semantic_traces
                ]
                primary = sum(
                    trace.semantic_backend == model_id for trace in semantic_traces
                )
                fallback = sum(
                    trace.semantic_backend == "fallback_minilm_cross_encoder"
                    for trace in semantic_traces
                )
                results.append(
                    {
                        "model_id": model_id,
                        "revision": definition["revision"],
                        "tree_sha256": model_hash,
                        "depth": depth,
                        "weight": weight,
                        "sessions": len(samples),
                        "hit_rate_at_10": result["hit_rate_at_10"],
                        "mrr": result["mrr"],
                        "recommended_technical_score": result[
                            "recommended_technical_score"
                        ],
                        "semantic_activations": len(semantic_traces),
                        "primary_activations": primary,
                        "fallback_activations": fallback,
                        "ordering_change_rate": (
                            sum(trace.semantic_changed for trace in semantic_traces)
                            / len(semantic_traces)
                            if semantic_traces
                            else 0.0
                        ),
                        "mean_semantic_latency_ms": (
                            sum(semantic_latencies) / len(semantic_latencies)
                            if semantic_latencies
                            else 0.0
                        ),
                        "p95_semantic_latency_ms": (
                            sorted(semantic_latencies)[
                                min(
                                    len(semantic_latencies) - 1,
                                    int(0.95 * len(semantic_latencies)),
                                )
                            ]
                            if semantic_latencies
                            else 0.0
                        ),
                        "elapsed_seconds": elapsed,
                        "mean_seconds_per_session": elapsed / max(1, len(samples)),
                        "primary_backend_valid": bool(semantic_traces)
                        and primary == len(semantic_traces),
                    }
                )
                del agent
                gc.collect()
        if model_id == "qwen2.5-0.5b-instruct":
            qwen_feasible = [
                item
                for item in results
                if item["model_id"] == model_id and item["primary_backend_valid"]
            ]
            if qwen_feasible:
                qwen_selected_depth = cast(
                    int,
                    min(
                        qwen_feasible,
                        key=lambda item: (
                            -cast(float, item["recommended_technical_score"]),
                            cast(float, item["mean_seconds_per_session"]),
                            cast(int, item["depth"]),
                        ),
                    )["depth"],
                )
                qwen_selected_weight = cast(
                    float,
                    min(
                        qwen_feasible,
                        key=lambda item: (
                            -cast(float, item["recommended_technical_score"]),
                            cast(float, item["mean_seconds_per_session"]),
                            cast(int, item["depth"]),
                            cast(float, item["weight"]),
                        ),
                    )["weight"],
                )
    feasible = [item for item in results if item["primary_backend_valid"]]
    selected = min(
        feasible,
        key=lambda item: (
            -cast(float, item["recommended_technical_score"]),
            cast(float, item["mean_seconds_per_session"]),
            cast(int, item["depth"]),
            cast(float, item["weight"]),
            str(item["model_id"]),
        ),
        default=None,
    )
    selected_config_path: Path | None = None
    selected_config_sha256: str | None = None
    if selected is not None:
        selected_id = cast(str, selected["model_id"])
        selected_definition = MODELS[selected_id]
        selected_semantic = base.semantic_ranker.model_copy(
            update={
                "backend": (
                    "qwen_causal_relevance"
                    if selected_id == "qwen2.5-0.5b-instruct"
                    else "local_causal_relevance"
                ),
                "model_id": selected_id,
                "model_path": selected_definition["path"],
                "model_revision": selected_definition["revision"],
                "model_sha256": cast(str, selected["tree_sha256"]),
                "rerank_k": cast(int, selected["depth"]),
                "weight": cast(float, selected["weight"]),
            }
        )
        selected_config = base.model_copy(
            update={
                "policy_id": f"{base.policy_id}_llm_selected",
                "semantic_ranker": selected_semantic,
            }
        )
        AdaptiveArchitectureAudit.validate(selected_config)
        selected_config_path = ROOT / args.selected_config_output
        selected_config_path.parent.mkdir(parents=True, exist_ok=True)
        selected_config_path.write_text(
            json.dumps(
                selected_config.model_dump(mode="json"), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        selected_config_sha256 = selected_config.canonical_hash()
    payload = {
        "schema_version": 1,
        "evaluation_scope": "bounded_development_comparison",
        "config": args.config,
        "datasets": list(datasets),
        "lineage_manifest": args.lineage_manifest,
        "lineage_manifest_sha256": manifest.manifest_sha256,
        "partition": "development",
        "browsing_samples": len(samples),
        "candidate_models_frozen_before_results": list(MODELS),
        "depths": list(depths),
        "weights": list(weights),
        "methodology": (
            "tune Qwen depth first, then compare genuine local-LLM families at "
            "the selected Qwen depth under the identical pipeline"
        ),
        "qwen_selected_depth": qwen_selected_depth,
        "qwen_selected_weight": qwen_selected_weight,
        "results": results,
        "unavailable": unavailable,
        "selected": selected,
        "selected_config": (
            str(selected_config_path.relative_to(ROOT))
            if selected_config_path is not None
            else None
        ),
        "selected_config_sha256": selected_config_sha256,
        "unrestricted_model_search_performed": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
