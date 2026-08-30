from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from evaluator.local_evaluator import catalog_index, evaluate, metric_summary
from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.training.adaptive_datasets import (
    AdaptiveTrainingCorpus,
    load_adaptive_training_corpus,
)
from ghostlab.training.adaptive_lineage import (
    AdaptiveLineageManifest,
    load_lineage_manifest,
    subset_corpus,
)
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]
PROMPT_CONTRACT = (
    "shopping-relevance-v1: decide whether the product matches the request and "
    "constraints; answer yes or no"
)
MODELS: dict[str, dict[str, str]] = {
    "qwen2.5-0.5b-instruct": {
        "path": "artifacts/cache/models/qwen2.5-0.5b-instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "backend": "qwen_causal_relevance",
    },
    "qwen3-0.6b": {
        "path": "artifacts/cache/models/qwen3-0.6b",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "backend": "local_causal_relevance",
    },
    "gemma-3-1b-it": {
        "path": "artifacts/cache/models/gemma-3-1b-it",
        "revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "backend": "local_causal_relevance",
    },
    "smollm2-1.7b-instruct": {
        "path": "artifacts/cache/models/smollm2-1.7b-instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
        "backend": "local_causal_relevance",
    },
}
MINILM_CONTROL = {
    "model_id": "minilm-cross-encoder-control",
    "path": "artifacts/cache/models/ms-marco-MiniLM-L6-v2",
    "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "backend": "minilm_cross_encoder_control",
}
DEFAULT_DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _int_value(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer-compatible value, got {type(value).__name__}")
    return int(value)


def _float_value(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def symmetric_trial_matrix(
    model_ids: Sequence[str], depths: Sequence[int], weights: Sequence[float]
) -> tuple[dict[str, object], ...]:
    """Return the identical Cartesian setting grid for every model family."""

    if not model_ids or not depths or not weights:
        raise ValueError("model, depth and weight grids must be non-empty")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model IDs must be unique")
    return tuple(
        {"model_id": model_id, "depth": int(depth), "weight": float(weight)}
        for model_id in model_ids
        for depth in depths
        for weight in weights
    )


def _development_fold_groups(
    corpus: AdaptiveTrainingCorpus,
    manifest: AdaptiveLineageManifest,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    groups = {
        str(item["group_id"]): tuple(str(value) for value in item["member_ids"])
        for item in manifest.payload["lineage_groups"]
    }
    folds: list[tuple[tuple[str, ...], ...]] = []
    for fold in manifest.payload["development_outer_folds"]:
        eligible: list[tuple[str, ...]] = []
        for group_id in sorted(str(value) for value in fold["group_ids"]):
            sample_ids = tuple(
                sample_id
                for sample_id in groups[group_id]
                if sample_id in corpus.samples
                and corpus.samples[sample_id].get("scenario_type") == "browsing"
            )
            if sample_ids:
                eligible.append(sample_ids)
        folds.append(tuple(eligible))
    if not folds or any(not fold for fold in folds):
        raise ValueError("every development outer fold must contain Browsing lineage")
    return tuple(folds)


def lineage_safe_sample_ids(
    corpus: AdaptiveTrainingCorpus,
    manifest: AdaptiveLineageManifest,
    max_samples: int,
) -> tuple[tuple[str, ...], ...]:
    """Select complete Browsing lineage groups, balanced across outer folds."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    fold_groups = _development_fold_groups(corpus, manifest)
    selected: list[list[str]] = [[] for _ in fold_groups]
    offsets = [0 for _ in fold_groups]
    total = 0
    progressed = True
    while progressed:
        progressed = False
        for fold_index, groups in enumerate(fold_groups):
            if offsets[fold_index] >= len(groups):
                continue
            group = groups[offsets[fold_index]]
            offsets[fold_index] += 1
            if total + len(group) > max_samples:
                continue
            selected[fold_index].extend(group)
            total += len(group)
            progressed = True
    if not total or any(not fold for fold in selected):
        raise ValueError(
            "max_samples is too small to include a complete lineage group per fold"
        )
    flattened = [item for fold in selected for item in fold]
    group_by_sample = manifest.group_by_sample
    owners: dict[str, int] = {}
    for fold_index, fold in enumerate(selected):
        for sample_id in fold:
            group_id = group_by_sample[sample_id]
            previous = owners.setdefault(group_id, fold_index)
            if previous != fold_index:
                raise ValueError("a selected lineage group crosses development folds")
    if len(flattened) != len(set(flattened)):
        raise ValueError("development sample selection contains duplicates")
    return tuple(tuple(fold) for fold in selected)


def _peak_memory_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def _candidate_pool_evidence(agent: AdaptiveHybridAgent) -> tuple[str, int]:
    rows = [
        {
            "session_id": item.session_id,
            "turn": item.turn,
            "candidates": list(item.candidates),
        }
        for item in sorted(
            agent.candidate_snapshots, key=lambda item: (item.session_id, item.turn)
        )
    ]
    return _canonical_sha256(rows), len(rows)


def _semantic_rescue_metrics(
    agent: AdaptiveHybridAgent, samples: Mapping[str, dict[str, Any]]
) -> dict[str, int]:
    traces = {(item.session_id, item.turn): item for item in agent.traces}
    rescued = 0
    demoted = 0
    evaluated = 0
    confirmed_target_removals = 0
    for snapshot in agent.candidate_snapshots:
        trace = traces.get((snapshot.session_id, snapshot.turn))
        sample = samples.get(snapshot.session_id)
        if trace is None or sample is None or not trace.semantic_executed:
            continue
        target = str(sample.get("ground_truth", {}).get("parent_asin", ""))
        if not target:
            continue
        evaluated += 1
        before = (
            snapshot.candidates.index(target) + 1
            if target in snapshot.candidates
            else None
        )
        after = trace.top_ids.index(target) + 1 if target in trace.top_ids else None
        rescued += bool(
            before is not None and before > 10 and after is not None and after <= 10
        )
        demoted += bool(
            before is not None and before <= 10 and (after is None or after > 10)
        )
        confirmed_target_removals += target in snapshot.authority_removed_ids
    return {
        "semantic_target_turns": evaluated,
        "target_rescued_into_top10": rescued,
        "target_demoted_from_top10": demoted,
        "confirmed_target_removal_count": confirmed_target_removals,
    }


def _worker_trial(spec: dict[str, Any]) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    base = load_adaptive_hybrid_config(ROOT / str(spec["config"]))
    datasets = tuple(str(value) for value in spec["datasets"])
    complete = load_adaptive_training_corpus(ROOT, datasets)
    manifest = load_lineage_manifest(ROOT / str(spec["lineage_manifest"]), complete)
    development = subset_corpus(complete, manifest, "development")
    fold_ids = tuple(
        tuple(str(value) for value in fold) for fold in spec["fold_sample_ids"]
    )
    ordered_ids = tuple(value for fold in fold_ids for value in fold)
    samples = [development.samples[sample_id] for sample_id in ordered_ids]
    definition = cast(dict[str, str], spec["definition"])
    model_path = ROOT / definition["path"]
    model_hash = _tree_sha256(model_path)
    control = definition["backend"] == "minilm_cross_encoder_control"
    semantic = base.semantic_ranker.model_copy(
        update={
            "backend": definition["backend"],
            "model_id": str(spec["model_id"]),
            "model_path": definition["path"],
            "model_revision": definition["revision"],
            "model_sha256": model_hash,
            "rerank_k": int(spec["depth"]),
            "weight": float(spec["weight"]),
            "fallback_weight": (
                float(spec["weight"])
                if control
                else base.semantic_ranker.fallback_weight
            ),
            "timeout_ms": int(spec["component_timeout_ms"]),
        }
    )
    config = base.model_copy(update={"semantic_ranker": semantic})
    catalog_path = ROOT / "data/catalog.jsonl"
    identifiers, categories, products = catalog_index(catalog_path)
    started = time.perf_counter()
    agent = AdaptiveHybridAgent(catalog_path, config, project_root=ROOT)
    result = evaluate(cast(Agent, agent), samples, identifiers, categories, products)
    elapsed = time.perf_counter() - started
    semantic_traces = [item for item in agent.traces if item.semantic_executed]
    latencies = sorted(item.semantic_elapsed_ms for item in semantic_traces)
    primary = sum(
        item.semantic_backend == str(spec["model_id"]) for item in semantic_traces
    )
    if control:
        primary = sum(
            item.semantic_backend == "minilm_cross_encoder_control"
            for item in semantic_traces
        )
    fallback = sum(
        item.semantic_backend == "fallback_minilm_cross_encoder"
        for item in semantic_traces
    )
    sessions_by_id = {str(item["sample_id"]): item for item in result["sessions"]}
    fold_metrics = [
        {
            "fold": fold_index,
            **metric_summary([sessions_by_id[sample_id] for sample_id in sample_ids]),
        }
        for fold_index, sample_ids in enumerate(fold_ids)
    ]
    candidate_pool_hash, candidate_pool_turns = _candidate_pool_evidence(agent)
    rescue = _semantic_rescue_metrics(agent, development.samples)
    diagnostics = agent.semantic.diagnostics()
    attempts = _int_value(diagnostics["primary_attempts"])
    successes = _int_value(diagnostics["primary_successes"])
    return {
        "model_id": spec["model_id"],
        "control": control,
        "revision": definition["revision"],
        "tree_sha256": model_hash,
        "depth": int(spec["depth"]),
        "weight": float(spec["weight"]),
        "sessions": len(samples),
        "ordered_session_ids_sha256": _canonical_sha256(ordered_ids),
        "fold_metrics": fold_metrics,
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "recommended_technical_score": result["recommended_technical_score"],
        "output_constraint_violations": sum(
            item.output_constraint_violations for item in agent.traces
        ),
        **rescue,
        "semantic_activations": len(semantic_traces),
        "primary_activations": primary,
        "fallback_activations": fallback,
        "fallback_rate": fallback / max(1, len(semantic_traces)),
        "ordering_change_rate": (
            sum(item.semantic_changed for item in semantic_traces)
            / max(1, len(semantic_traces))
        ),
        "failure_counts": diagnostics["failure_counts"],
        "mean_semantic_latency_ms": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        "p95_semantic_latency_ms": (
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies
            else 0.0
        ),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_session": elapsed / max(1, len(samples)),
        "peak_worker_memory_mb": _peak_memory_mb(),
        "candidate_pool_sha256": candidate_pool_hash,
        "candidate_pool_turns": candidate_pool_turns,
        "prompt_meaning_sha256": diagnostics["prompt_meaning_sha256"],
        "chat_template_used": diagnostics["chat_template_used"],
        "chat_template_sha256": diagnostics["chat_template_sha256"],
        "primary_backend_valid": (
            bool(semantic_traces)
            and primary == len(semantic_traces)
            and (control or (attempts == successes and fallback == 0))
        ),
    }


def _trial_key(item: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int_value(item["output_constraint_violations"]),
        _int_value(item["confirmed_target_removal_count"]),
        _int_value(item["target_demoted_from_top10"]),
        -_float_value(item["recommended_technical_score"]),
        -_float_value(item["mrr"]),
        -_float_value(item["hit_rate_at_10"]),
        _float_value(item["fallback_rate"]),
        _float_value(item["p95_semantic_latency_ms"]),
        _float_value(item["peak_worker_memory_mb"]),
        _int_value(item["depth"]),
        _float_value(item["weight"]),
        str(item["model_id"]),
    )


def select_model_winners(
    results: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    winners: list[dict[str, object]] = []
    for model_id in MODELS:
        eligible = [
            item
            for item in results
            if item["model_id"] == model_id
            and bool(item["primary_backend_valid"])
            and _int_value(item["output_constraint_violations"]) == 0
            and _int_value(item["confirmed_target_removal_count"]) == 0
        ]
        if eligible:
            winners.append(min(eligible, key=_trial_key))
    return tuple(winners)


def paired_trial_evidence(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Verify that successful trials used identical sessions and candidate pools."""

    complete = [item for item in results if item.get("status") == "complete"]
    pool_hashes = {str(item["candidate_pool_sha256"]) for item in complete}
    ordered_hashes = {str(item["ordered_session_ids_sha256"]) for item in complete}
    return {
        "complete_trial_count": len(complete),
        "paired_candidate_pools": bool(complete) and len(pool_hashes) == 1,
        "paired_ordered_sessions": bool(complete) and len(ordered_hashes) == 1,
        "candidate_pool_sha256": (
            next(iter(pool_hashes)) if len(pool_hashes) == 1 else None
        ),
        "ordered_session_ids_sha256": (
            next(iter(ordered_hashes)) if len(ordered_hashes) == 1 else None
        ),
    }


def _model_config(
    base_path: Path,
    result: Mapping[str, object],
    output: Path,
) -> tuple[str, str]:
    base = load_adaptive_hybrid_config(base_path)
    model_id = str(result["model_id"])
    definition = MODELS[model_id]
    semantic = base.semantic_ranker.model_copy(
        update={
            "backend": definition["backend"],
            "model_id": model_id,
            "model_path": definition["path"],
            "model_revision": definition["revision"],
            "model_sha256": str(result["tree_sha256"]),
            "rerank_k": _int_value(result["depth"]),
            "weight": _float_value(result["weight"]),
        }
    )
    config = base.model_copy(
        update={
            "policy_id": f"{base.policy_id}_{model_id.replace('.', '_')}",
            "semantic_ranker": semantic,
        }
    )
    AdaptiveArchitectureAudit.validate(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output.relative_to(ROOT).as_posix(), config.canonical_hash()


def _run_worker(
    spec: dict[str, object], *, temporary: Path, timeout_seconds: int
) -> dict[str, object]:
    token = _canonical_sha256(spec)[:16]
    spec_path = temporary / f"{token}.spec.json"
    output_path = temporary / f"{token}.result.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker-spec",
                str(spec_path),
                "--worker-output",
                str(output_path),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            **spec,
            "status": "failed",
            "failure_reason": "worker_timeout",
            "elapsed_seconds": time.perf_counter() - started,
        }
    if completed.returncode != 0 or not output_path.is_file():
        return {
            **spec,
            "status": "failed",
            "failure_reason": "worker_error",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
            "elapsed_seconds": time.perf_counter() - started,
        }
    return {"status": "complete", **json.loads(output_path.read_text())}


def _worker_main(spec_path: Path, output_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = _worker_trial(spec)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symmetric four-family local-LLM semantic-ranking comparison"
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
    parser.add_argument("--trial-timeout-seconds", type=int, default=1800)
    parser.add_argument("--component-timeout-ms", type=int, default=120000)
    parser.add_argument(
        "--output", default="artifacts/reports/local_llm_ranker_comparison_v2.json"
    )
    parser.add_argument(
        "--selected-config-output",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument(
        "--candidate-config-dir", default="configs/finalists/local_llm_development"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker-spec")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker_spec or args.worker_output:
        if not args.worker_spec or not args.worker_output:
            raise ValueError("worker-spec and worker-output must be supplied together")
        _worker_main(Path(args.worker_spec), Path(args.worker_output))
        return
    depths = tuple(dict.fromkeys(args.depths or (10, 20, 30)))
    weights = tuple(dict.fromkeys(args.weights or (0.20, 0.35, 0.50)))
    if any(not 2 <= depth <= 50 for depth in depths):
        raise ValueError("semantic depths must be between 2 and 50")
    if any(not 0.0 < weight <= 1.0 for weight in weights):
        raise ValueError("semantic weights must be in (0, 1]")
    if args.trial_timeout_seconds <= 0 or args.component_timeout_ms <= 0:
        raise ValueError("timeouts must be positive")
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    corpus = load_adaptive_training_corpus(ROOT, datasets)
    manifest = load_lineage_manifest(ROOT / args.lineage_manifest, corpus)
    development = subset_corpus(corpus, manifest, "development")
    fold_sample_ids = lineage_safe_sample_ids(development, manifest, args.max_samples)
    model_definitions = {**MODELS, MINILM_CONTROL["model_id"]: MINILM_CONTROL}
    matrix = symmetric_trial_matrix(tuple(model_definitions), depths, weights)
    common = {
        "config": args.config,
        "datasets": list(datasets),
        "lineage_manifest": args.lineage_manifest,
        "fold_sample_ids": [list(fold) for fold in fold_sample_ids],
        "component_timeout_ms": args.component_timeout_ms,
    }
    unavailable: list[dict[str, str]] = []
    runnable: list[dict[str, object]] = []
    for trial in matrix:
        model_id = str(trial["model_id"])
        definition = model_definitions[model_id]
        path = ROOT / definition["path"]
        if not path.is_dir():
            if not any(item["model_id"] == model_id for item in unavailable):
                unavailable.append({"model_id": model_id, "reason": "asset_missing"})
            continue
        runnable.append({**common, **trial, "definition": definition})
    if args.plan_only:
        print(
            json.dumps(
                {
                    "model_ids": list(model_definitions),
                    "depths": list(depths),
                    "weights": list(weights),
                    "trial_count": len(matrix),
                    "runnable_trial_count": len(runnable),
                    "fold_sample_counts": [len(fold) for fold in fold_sample_ids],
                    "unavailable": unavailable,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="local-llm-grid-") as temporary_value:
        temporary = Path(temporary_value)
        for index, spec in enumerate(runnable, start=1):
            print(
                f"START local-LLM trial {index}/{len(runnable)} "
                f"model={spec['model_id']} depth={spec['depth']} weight={spec['weight']}",
                flush=True,
            )
            result = _run_worker(
                spec, temporary=temporary, timeout_seconds=args.trial_timeout_seconds
            )
            results.append(result)
            print(
                f"DONE local-LLM trial {index}/{len(runnable)} "
                f"status={result['status']}",
                flush=True,
            )
    complete = [item for item in results if item.get("status") == "complete"]
    paired_evidence = paired_trial_evidence(results)
    paired_candidate_pools = bool(paired_evidence["paired_candidate_pools"])
    paired_ordered_sessions = bool(paired_evidence["paired_ordered_sessions"])
    winners = select_model_winners(complete) if paired_candidate_pools else ()
    ranked_winners = tuple(sorted(winners, key=_trial_key))
    config_dir = ROOT / args.candidate_config_dir
    candidate_configs: list[dict[str, object]] = []
    for rank, winner in enumerate(ranked_winners, start=1):
        path = config_dir / f"rank_{rank}_{winner['model_id']}.json"
        relative, canonical_hash = _model_config(ROOT / args.config, winner, path)
        candidate_configs.append(
            {
                "rank": rank,
                "model_id": winner["model_id"],
                "config_path": relative,
                "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "config_canonical_sha256": canonical_hash,
                "depth": winner["depth"],
                "weight": winner["weight"],
                "metrics": {
                    key: winner[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "recommended_technical_score",
                        "fallback_rate",
                        "p95_semantic_latency_ms",
                        "peak_worker_memory_mb",
                    )
                },
            }
        )
    selected = ranked_winners[0] if ranked_winners else None
    selected_config = None
    selected_config_sha256 = None
    if selected is not None:
        selected_path = ROOT / args.selected_config_output
        selected_config, selected_config_sha256 = _model_config(
            ROOT / args.config, selected, selected_path
        )
    payload = {
        "schema_version": 2,
        "evaluation_scope": "lineage_safe_development_component_selection",
        "config": args.config,
        "datasets": list(datasets),
        "lineage_manifest": args.lineage_manifest,
        "lineage_manifest_sha256": manifest.manifest_sha256,
        "partition": "development",
        "holdout_accessed": False,
        "model_ids": list(MODELS),
        "control_model_id": MINILM_CONTROL["model_id"],
        "depths": list(depths),
        "weights": list(weights),
        "symmetric_grid": True,
        "grid_trial_count": len(matrix),
        "executed_trial_count": len(results),
        "fold_sample_ids": [list(fold) for fold in fold_sample_ids],
        "fold_sample_counts": [len(fold) for fold in fold_sample_ids],
        "ordered_session_ids_sha256": _canonical_sha256(
            tuple(value for fold in fold_sample_ids for value in fold)
        ),
        "prompt_contract": PROMPT_CONTRACT,
        "prompt_contract_sha256": hashlib.sha256(PROMPT_CONTRACT.encode()).hexdigest(),
        "model_specific_chat_templates_allowed": True,
        "paired_candidate_pools": paired_candidate_pools,
        "paired_ordered_sessions": paired_ordered_sessions,
        "candidate_pool_sha256": paired_evidence["candidate_pool_sha256"],
        "results": results,
        "unavailable": unavailable,
        "per_model_winners": list(winners),
        "ranked_model_winners": list(ranked_winners),
        "development_candidate_configs": candidate_configs,
        "top_three_development_configs": candidate_configs[:3],
        "selected": selected,
        "selected_config": selected_config,
        "selected_config_sha256": selected_config_sha256,
        "selection_valid": bool(selected)
        and paired_candidate_pools
        and paired_ordered_sessions,
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
