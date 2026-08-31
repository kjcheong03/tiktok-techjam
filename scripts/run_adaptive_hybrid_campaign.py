from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from evaluator.local_evaluator import catalog_index, evaluate, metric_summary
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_campaign import (
    AdaptiveEvaluation,
    AdaptiveGhostLabEngine,
)
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.optimization.adaptive_warm_start import load_adaptive_warm_start
from ghostlab.retrieval.residual import TECHNIQUE_ID as RESIDUAL_TECHNIQUE_ID
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.training.adaptive_datasets import (
    load_adaptive_training_corpus,
    progressive_stratified_samples,
)
from ghostlab.training.adaptive_lineage import (
    AdaptiveLineageManifest,
    cluster_ids_for_samples,
    load_lineage_manifest,
    subset_corpus,
)
from ghostlab.training.adaptive_residual import (
    collect_adaptive_residual_turns,
    config_with_adaptive_residual_asset,
    fit_adaptive_residual_asset,
    lineage_outer_folds,
)
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]


def _float_value(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("campaign metric must be numeric")
    return float(value)


def _int_value(value: object) -> int:
    if not isinstance(value, (int, float)):
        raise TypeError("campaign count must be numeric")
    return int(value)


def _session_reward(session: dict[str, object]) -> float:
    hit = float(bool(session["hit"]))
    raw_reciprocal = session["reciprocal_rank"]
    if not isinstance(raw_reciprocal, (int, float)):
        raise TypeError("session reciprocal rank must be numeric")
    reciprocal = float(raw_reciprocal)
    first_hit = session.get("first_hit_turn")
    turn = float(first_hit) if isinstance(first_hit, int) else 11.0
    efficiency = max(0.0, min(1.0, (11.0 - turn) / 10.0))
    return 0.5 * hit + 0.3 * reciprocal + 0.2 * efficiency


def _candidate_payload(candidate: CandidateSpec) -> dict[str, object]:
    return candidate.model_dump(mode="json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_verified(config, candidate: CandidateSpec, registry) -> bool:  # type: ignore[no-untyped-def]
    required = [
        item for item in candidate.techniques if registry.bindings[item].fit_required
    ]
    if not required:
        return True
    model_value = config.union_ranker.model_path
    expected_hash = config.union_ranker.model_sha256
    if not model_value or not expected_hash:
        return False
    model_path = ROOT / model_value
    receipt_path = model_path.with_name(f"{model_path.stem}.fit_receipt.json")
    if not model_path.is_file() or not receipt_path.is_file():
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return (
        _sha256(model_path) == expected_hash
        and receipt.get("model_sha256") == expected_hash
        and bool(receipt.get("selected_by_oof"))
        and not bool(receipt.get("holdout_accessed"))
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return float(ordered[index])


def _semantic_evidence(
    agent: AdaptiveHybridAgent,
    selected_samples: list[dict[str, object]],
    *,
    rerank_k: int,
) -> dict[str, object]:
    """Measure semantic work and Top-10 movement strictly offline."""

    session_ids = list(agent.sessions)
    target_by_session = {
        session_id: str(sample["ground_truth"]["parent_asin"])  # type: ignore[index]
        for session_id, sample in zip(session_ids, selected_samples, strict=True)
    }
    opportunities = 0
    rescues = 0
    regressions = 0
    for snapshot in agent.candidate_snapshots:
        target = target_by_session.get(snapshot.session_id)
        if target is None or not snapshot.pre_semantic_candidates:
            continue
        before = snapshot.pre_semantic_candidates
        after = snapshot.post_semantic_candidates
        if target not in before[:10] and target in before[:rerank_k]:
            opportunities += 1
        if target not in before[:10] and target in after[:10]:
            rescues += 1
        if target in before[:10] and target not in after[:10]:
            regressions += 1
    semantic_traces = [trace for trace in agent.traces if trace.semantic_executed]
    final_route_by_session: dict[str, str] = {}
    for trace in agent.traces:
        final_route_by_session[trace.session_id] = trace.route
    return {
        "semantic_latency_p95_ms": _p95(
            [trace.semantic_elapsed_ms for trace in semantic_traces]
        ),
        "semantic_activations": len(semantic_traces),
        "semantic_rescue_opportunities": opportunities,
        "semantic_rescues": rescues,
        "semantic_regressions": regressions,
        "routes": tuple(
            final_route_by_session.get(session_id, "unknown")
            for session_id in session_ids
        ),
    }


def _technical_result(sessions: list[dict[str, object]]) -> dict[str, object]:
    metrics = metric_summary(cast(list[dict], sessions))
    efficiency = max(0.0, min(1.0, (11.0 - float(metrics["mttc"])) / 10.0))
    return {
        **metrics,
        "recommended_technical_score": (
            0.50 * float(metrics["hit_rate_at_10"])
            + 0.30 * float(metrics["mrr"])
            + 0.20 * efficiency
        ),
        "sessions": sessions,
    }


def _evaluate_fold_safe_residual(
    *,
    config,
    candidate: CandidateSpec,
    fidelity: str,
    selected_samples: list[dict[str, Any]],
    lineage_manifest: AdaptiveLineageManifest,
    catalog_path: Path,
    identifiers: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    seed: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, Any]],
    int,
    tuple[dict[str, object], ...],
]:
    """Fresh-fit and evaluate D out of fold; C is only used as the collector."""

    turns, collection = collect_adaptive_residual_turns(
        config,
        selected_samples,
        catalog_path=catalog_path,
        project_root=ROOT,
    )
    selected_by_id = {str(item["sample_id"]): item for item in selected_samples}
    selected_ids = set(selected_by_id)
    folds = lineage_outer_folds(lineage_manifest, selected_ids)
    if len(folds) < 2:
        raise ValueError("residual OOF evaluation requires at least two lineage folds")
    token = hashlib.sha256(
        f"{candidate.candidate_id}\0{fidelity}\0{seed}".encode()
    ).hexdigest()[:16]
    sessions: list[dict[str, object]] = []
    evaluated_samples: list[dict[str, Any]] = []
    agents: list[AdaptiveHybridAgent] = []
    receipts: list[dict[str, object]] = []
    for fold_index, validation_ids in enumerate(folds):
        training_ids = selected_ids - validation_ids
        artifact = fit_adaptive_residual_asset(
            turns,
            config=config,
            training_ids=training_ids,
            validation_ids=validation_ids,
            group_by_sample=lineage_manifest.group_by_sample,
            outer_fold=fold_index,
            seed=seed,
            project_root=ROOT,
            output_prefix=(
                f"artifacts/models/adaptive_top10_residual/{token}/{fidelity}"
            ),
        )
        fold_config = config_with_adaptive_residual_asset(config, artifact)
        fold_samples = [
            item
            for item in selected_samples
            if str(item["sample_id"]) in validation_ids
        ]
        agent = AdaptiveHybridAgent(catalog_path, fold_config, project_root=ROOT)
        fold_result = evaluate(
            cast(Agent, agent),
            fold_samples,
            identifiers,
            categories,
            products,
        )
        agents.append(agent)
        sessions.extend(cast(list[dict[str, object]], fold_result["sessions"]))
        evaluated_samples.extend(fold_samples)
        receipts.append(
            {
                "outer_fold": artifact.outer_fold,
                "asset_path": artifact.asset_path,
                "asset_sha256": artifact.asset_sha256,
                "receipt_path": artifact.receipt_path,
                "receipt_sha256": artifact.receipt_sha256,
                "training_sample_ids_sha256": (artifact.training_sample_ids_sha256),
                "validation_sample_ids_sha256": (artifact.validation_sample_ids_sha256),
            }
        )
    semantic_latencies = [
        trace.semantic_elapsed_ms
        for agent in agents
        for trace in agent.traces
        if trace.semantic_executed
    ]
    semantic_activations = 0
    opportunities = rescues = regressions = 0
    routes: list[str] = []
    for agent, fold_samples in zip(
        agents,
        (
            [
                item
                for item in selected_samples
                if str(item["sample_id"]) in validation_ids
            ]
            for validation_ids in folds
        ),
        strict=True,
    ):
        evidence = _semantic_evidence(
            agent, cast(list[dict[str, object]], fold_samples), rerank_k=10
        )
        semantic_activations += _int_value(evidence["semantic_activations"])
        opportunities += _int_value(evidence["semantic_rescue_opportunities"])
        rescues += _int_value(evidence["semantic_rescues"])
        regressions += _int_value(evidence["semantic_regressions"])
        routes.extend(cast(tuple[str, ...], evidence["routes"]))
    semantic = {
        "semantic_latency_p95_ms": _p95(semantic_latencies),
        "semantic_activations": semantic_activations,
        "semantic_rescue_opportunities": opportunities,
        "semantic_rescues": rescues,
        "semantic_regressions": regressions,
        "routes": tuple(routes),
        "residual_collection": collection,
    }
    violations = sum(
        trace.output_constraint_violations for agent in agents for trace in agent.traces
    )
    return (
        _technical_result(sessions),
        semantic,
        evaluated_samples,
        violations,
        tuple(receipts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run architecture-safe GhostLab champion/challenger racing inside "
            "the fixed adaptive 1A-3B workflow"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument(
        "--technique-catalog", default="configs/techniques/catalog_v2.json"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help=(
            "repeat for each evaluation JSONL; defaults to the complete "
            "200+1000+1000 corpus, then selects development by lineage"
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-extra-techniques", type=int)
    parser.add_argument("--higher-order-rounds", type=int, default=8)
    parser.add_argument("--f1-candidates", type=int, default=24)
    parser.add_argument("--f2-candidates", type=int, default=6)
    parser.add_argument("--hpo-trials-per-structure", type=int, default=2)
    parser.add_argument(
        "--warm-start",
        help=(
            "optional architecture-safe translated warm-start specification; "
            "the historical runtime is never executed directly"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print the complete race plan without evaluating/training",
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_hybrid_campaign.json"
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/campaigns/adaptive_hybrid/checkpoint.json",
        help="resumable per-evaluation checkpoint",
    )
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    baseline = load_adaptive_hybrid_config(ROOT / args.config)
    catalog = load_catalog(ROOT / args.technique_catalog)
    registry = AdaptiveTechniqueRegistry.from_catalog(catalog, project_root=ROOT)
    warm_start_spec = None
    warm_start_candidate = None
    if args.warm_start:
        warm_start_spec, warm_start_candidate = load_adaptive_warm_start(
            args.warm_start,
            project_root=ROOT,
            baseline=baseline,
            registry=registry,
        )
    engine = AdaptiveGhostLabEngine(
        baseline=baseline,
        registry=registry,
        warm_start=warm_start_candidate,
        candidate_limit=args.candidate_limit,
        beam_width=args.beam_width,
        max_extra_techniques=args.max_extra_techniques,
        seed=args.seed,
    )
    inventory = registry.inventory()
    initial_plan = engine.initial_plan()
    planning_coverage = engine.plan_coverage(initial_plan.candidates)
    if args.plan_only:
        report: dict[str, Any] = {
            "schema_version": 2,
            "mode": "plan_only",
            "architecture": baseline.architecture,
            "workflow_static": True,
            "catalog_total": inventory.total,
            "inventory": {
                "compulsory": inventory.compulsory,
                "promotable": inventory.promotable,
                "control_only": inventory.control_only,
                "research_only": inventory.research_only,
                "unavailable": inventory.unavailable,
            },
            "initial_candidate_count": len(initial_plan.candidates),
            "initial_candidates": [
                _candidate_payload(item) for item in initial_plan.candidates
            ],
            "planning_coverage": planning_coverage,
            "skipped": [
                {"roots": item.roots, "reasons": item.reasons}
                for item in initial_plan.skipped
            ],
            "maximum_extra_techniques": engine.limits.max_order,
            "fixed_six_capability_limit": False,
            "hpo_trials_per_surviving_structure": args.hpo_trials_per_structure,
            "semantic_tuning": {
                "model_fixed": "smollm2-1.7b-instruct",
                "activation_policy_fixed": "browsing_only",
                "f0_weight_grid": [0.05, 0.10, 0.15, 0.20],
                "f0_depth": 10,
                "f1_survivor_depth": 20,
                "depth_30_or_50_tested": False,
            },
            "warm_start": (
                None
                if warm_start_spec is None
                else {
                    "specification": args.warm_start,
                    "warm_start_id": warm_start_spec.warm_start_id,
                    "source_candidate_id": warm_start_spec.source_candidate_id,
                    "candidate": _candidate_payload(
                        cast(CandidateSpec, warm_start_candidate)
                    ),
                    "historical_runtime_executed": False,
                }
            ),
        }
    else:
        dataset_paths = tuple(
            args.datasets
            or (
                "data/public_set.jsonl",
                "data/synthetic_1000_public_like.jsonl",
                "data/independent_template_1000.jsonl",
            )
        )
        complete_corpus = load_adaptive_training_corpus(ROOT, dataset_paths)
        lineage_manifest_path = ROOT / args.lineage_manifest
        lineage_manifest = load_lineage_manifest(lineage_manifest_path, complete_corpus)
        corpus = subset_corpus(complete_corpus, lineage_manifest, "development")
        samples = progressive_stratified_samples(corpus, seed=args.seed)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        catalog_path = ROOT / "data/catalog.jsonl"
        identifiers, categories, products = catalog_index(catalog_path)
        evaluation_ordinal = 0
        checkpoint_path = ROOT / args.checkpoint
        checkpoint_signature = {
            "evaluation_schema": 4,
            "config_sha256": baseline.canonical_hash(),
            "datasets": list(dataset_paths),
            "lineage_manifest_sha256": _sha256(lineage_manifest_path),
            "partition": "development",
            "sample_count": len(samples),
            "seed": args.seed,
        }
        checkpoint: dict[str, Any] = {
            "schema_version": 1,
            "signature": checkpoint_signature,
            "evaluations": {},
        }
        if checkpoint_path.is_file():
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if loaded.get("signature") != checkpoint_signature:
                archive = checkpoint_path.with_name(
                    f"{checkpoint_path.stem}.incompatible-{int(time.time())}.json"
                )
                checkpoint_path.replace(archive)
                print(
                    json.dumps(
                        {
                            "event": "incompatible_checkpoint_archived",
                            "archive": str(archive.relative_to(ROOT)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                checkpoint = loaded

        def save_checkpoint() -> None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp.json")
            temporary.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, checkpoint_path)

        def evaluator(
            config,
            candidate: CandidateSpec,
            fidelity: str,  # type: ignore[no-untyped-def]
        ) -> AdaptiveEvaluation:
            nonlocal evaluation_ordinal
            evaluation_ordinal += 1
            checkpoint_key = f"{fidelity}:{candidate.candidate_id}"
            cached = checkpoint["evaluations"].get(checkpoint_key)
            if cached is not None and cached.get("candidate") != _candidate_payload(
                candidate
            ):
                cached = None
            if cached is not None:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_resumed",
                            "ordinal": evaluation_ordinal,
                            "candidate_id": candidate.candidate_id,
                            "fidelity": fidelity,
                            "score": cached["score"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return AdaptiveEvaluation(
                    candidate_id=candidate.candidate_id,
                    fidelity=cast(Any, fidelity),
                    score=float(cached["score"]),
                    session_rewards=tuple(
                        float(item) for item in cached["session_rewards"]
                    ),
                    behavior_novelty=float(cached.get("behavior_novelty", 0.0)),
                    latency_p95_ms=float(cached.get("latency_p95_ms", 0.0)),
                    semantic_latency_p95_ms=float(
                        cached.get("semantic_latency_p95_ms", 0.0)
                    ),
                    semantic_activations=int(cached.get("semantic_activations", 0)),
                    semantic_rescue_opportunities=int(
                        cached.get("semantic_rescue_opportunities", 0)
                    ),
                    semantic_rescues=int(cached.get("semantic_rescues", 0)),
                    semantic_regressions=int(cached.get("semantic_regressions", 0)),
                    fit_verified=bool(cached.get("fit_verified", False)),
                    gate_metrics=tuple(
                        (str(name), float(value))
                        for name, value in cached.get("gate_metrics", [])
                    ),
                    constraint_violations=int(cached.get("constraint_violations", 0)),
                    hit_rate_at_10=float(cached.get("hit_rate_at_10", 0.0)),
                    mrr=float(cached.get("mrr", 0.0)),
                    mttc=float(cached.get("mttc", 0.0)),
                    lineage_cluster_ids=tuple(
                        str(item) for item in cached.get("lineage_cluster_ids", [])
                    ),
                )
            fractions = {"f0": 0.2, "f1": 0.5, "f2": 1.0}
            count = max(1, round(len(samples) * fractions[fidelity]))
            print(
                json.dumps(
                    {
                        "event": "evaluation_started",
                        "ordinal": evaluation_ordinal,
                        "candidate_id": candidate.candidate_id,
                        "fidelity": fidelity,
                        "sample_count": count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            started = time.perf_counter()
            selected_samples = cast(list[dict[str, Any]], samples[:count])
            residual_fit_receipts: tuple[dict[str, object], ...] = ()
            if RESIDUAL_TECHNIQUE_ID in candidate.techniques:
                (
                    result,
                    semantic_evidence,
                    selected_samples,
                    constraint_violations,
                    residual_fit_receipts,
                ) = _evaluate_fold_safe_residual(
                    config=config,
                    candidate=candidate,
                    fidelity=fidelity,
                    selected_samples=selected_samples,
                    lineage_manifest=lineage_manifest,
                    catalog_path=catalog_path,
                    identifiers=identifiers,
                    categories=categories,
                    products=products,
                    seed=args.seed,
                )
                fit_verified = bool(residual_fit_receipts)
            else:
                agent = AdaptiveHybridAgent(catalog_path, config, project_root=ROOT)
                result = evaluate(
                    cast(Agent, agent),
                    selected_samples,
                    identifiers,
                    categories,
                    products,
                )
                semantic_evidence = _semantic_evidence(
                    agent,
                    cast(list[dict[str, object]], selected_samples),
                    rerank_k=config.semantic_ranker.rerank_k,
                )
                constraint_violations = sum(
                    trace.output_constraint_violations for trace in agent.traces
                )
                fit_verified = _fit_verified(config, candidate, registry)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            sessions = cast(list[dict[str, object]], result["sessions"])
            semantic_routes = cast(tuple[str, ...], semantic_evidence["routes"])
            grouped_rewards: dict[str, list[float]] = defaultdict(list)
            for sample, session, route in zip(
                selected_samples,
                sessions,
                semantic_routes,
                strict=True,
            ):
                reward = _session_reward(session)
                sample_id = str(sample["sample_id"])
                grouped_rewards[f"scenario:{sample['scenario_type']}"].append(reward)
                grouped_rewards[f"source:{corpus.origins[sample_id]}"].append(reward)
                grouped_rewards[f"route:{route}"].append(reward)
            gate_metrics = tuple(
                (name, sum(values) / len(values))
                for name, values in sorted(grouped_rewards.items())
                if values
            )
            evaluation = AdaptiveEvaluation(
                candidate_id=candidate.candidate_id,
                fidelity=cast(Any, fidelity),
                score=_float_value(result["recommended_technical_score"]),
                session_rewards=tuple(_session_reward(item) for item in sessions),
                behavior_novelty=0.0,
                latency_p95_ms=elapsed_ms / count,
                semantic_latency_p95_ms=_float_value(
                    semantic_evidence["semantic_latency_p95_ms"]
                ),
                semantic_activations=_int_value(
                    semantic_evidence["semantic_activations"]
                ),
                semantic_rescue_opportunities=_int_value(
                    semantic_evidence["semantic_rescue_opportunities"]
                ),
                semantic_rescues=_int_value(semantic_evidence["semantic_rescues"]),
                semantic_regressions=_int_value(
                    semantic_evidence["semantic_regressions"]
                ),
                fit_verified=fit_verified,
                gate_metrics=gate_metrics,
                constraint_violations=constraint_violations,
                hit_rate_at_10=_float_value(result["hit_rate_at_10"]),
                mrr=_float_value(result["mrr"]),
                mttc=_float_value(result["mttc"]),
                lineage_cluster_ids=cluster_ids_for_samples(
                    lineage_manifest,
                    [str(item["sample_id"]) for item in selected_samples],
                ),
            )
            checkpoint["evaluations"][checkpoint_key] = {
                "candidate": _candidate_payload(candidate),
                "score": evaluation.score,
                "session_rewards": list(evaluation.session_rewards),
                "behavior_novelty": evaluation.behavior_novelty,
                "latency_p95_ms": evaluation.latency_p95_ms,
                "semantic_latency_p95_ms": evaluation.semantic_latency_p95_ms,
                "semantic_activations": evaluation.semantic_activations,
                "semantic_rescue_opportunities": (
                    evaluation.semantic_rescue_opportunities
                ),
                "semantic_rescues": evaluation.semantic_rescues,
                "semantic_regressions": evaluation.semantic_regressions,
                "fit_verified": evaluation.fit_verified,
                "gate_metrics": list(evaluation.gate_metrics),
                "constraint_violations": evaluation.constraint_violations,
                "hit_rate_at_10": evaluation.hit_rate_at_10,
                "mrr": evaluation.mrr,
                "mttc": evaluation.mttc,
                "lineage_cluster_ids": list(evaluation.lineage_cluster_ids),
                "sample_count": count,
                "residual_fit_receipts": list(residual_fit_receipts),
                "completed_at_unix": time.time(),
            }
            save_checkpoint()
            print(
                json.dumps(
                    {
                        "event": "evaluation_finished",
                        "ordinal": evaluation_ordinal,
                        "candidate_id": candidate.candidate_id,
                        "fidelity": fidelity,
                        "sample_count": count,
                        "score": evaluation.score,
                        "elapsed_seconds": elapsed_ms / 1000.0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return evaluation

        campaign_started = time.perf_counter()
        result = engine.run(
            evaluator,
            f1_candidates=args.f1_candidates,
            f2_candidates=args.f2_candidates,
            higher_order_rounds=args.higher_order_rounds,
            hpo_trials_per_structure=args.hpo_trials_per_structure,
        )
        report = {
            "schema_version": 2,
            "mode": "race",
            "architecture": baseline.architecture,
            "workflow_static": True,
            "catalog_total": inventory.total,
            "compulsory_count": len(inventory.compulsory),
            "promotable_count": len(inventory.promotable),
            "dataset_sources": [source.__dict__ for source in corpus.sources],
            "sample_count": len(samples),
            "partition": "development",
            "lineage_manifest": args.lineage_manifest,
            "lineage_manifest_sha256": _sha256(lineage_manifest_path),
            "lineage_cluster_count": len(
                {
                    lineage_manifest.group_by_sample[str(item["sample_id"])]
                    for item in samples
                }
            ),
            "fidelity_sample_counts": {
                fidelity: max(1, round(len(samples) * fraction))
                for fidelity, fraction in {"f0": 0.2, "f1": 0.5, "f2": 1.0}.items()
            },
            "incumbent": _candidate_payload(result.incumbent),
            "selected": _candidate_payload(result.selected),
            "promoted": result.promoted,
            "semantic_tuning": {
                "model_fixed": "smollm2-1.7b-instruct",
                "activation_policy_fixed": "browsing_only",
                "f0_weight_grid": [0.05, 0.10, 0.15, 0.20],
                "f0_depth": 10,
                "f0_surviving_weights": list(result.semantic_weight_survivors),
                "selected_weight": result.selected_semantic_weight,
                "f1_survivor_depth": 20,
                "selected_depth": result.selected_semantic_depth,
                "depth_30_or_50_tested": False,
                "model_family_search_reopened": False,
            },
            "warm_start": (
                None
                if warm_start_spec is None
                else {
                    "specification": args.warm_start,
                    "warm_start_id": warm_start_spec.warm_start_id,
                    "source_candidate_id": warm_start_spec.source_candidate_id,
                    "candidate": _candidate_payload(
                        cast(CandidateSpec, warm_start_candidate)
                    ),
                    "inherited_mechanisms": list(warm_start_spec.inherited_mechanisms),
                    "excluded_source_techniques": dict(
                        warm_start_spec.excluded_source_techniques
                    ),
                    "historical_runtime_executed": False,
                }
            ),
            "stage_counts": {
                fidelity: len(records) for fidelity, records in result.stages.items()
            },
            "records": {
                fidelity: [
                    {
                        "candidate": _candidate_payload(item.candidate),
                        "score": item.evaluation.score,
                        "decision": item.decision,
                        "fit_required": item.fit_required,
                        "fit_verified": item.evaluation.fit_verified,
                        "gate_failures": item.gate_failures,
                        "mean_paired_delta": sum(item.paired_deltas)
                        / len(item.paired_deltas),
                        "latency_p95_ms": item.evaluation.latency_p95_ms,
                        "semantic_latency_p95_ms": (
                            item.evaluation.semantic_latency_p95_ms
                        ),
                        "semantic_activations": (item.evaluation.semantic_activations),
                        "semantic_rescue_opportunities": (
                            item.evaluation.semantic_rescue_opportunities
                        ),
                        "semantic_rescues": item.evaluation.semantic_rescues,
                        "semantic_regressions": (item.evaluation.semantic_regressions),
                        "hit_rate_at_10": item.evaluation.hit_rate_at_10,
                        "mrr": item.evaluation.mrr,
                        "mttc": item.evaluation.mttc,
                        "gate_metrics": list(item.evaluation.gate_metrics),
                        "constraint_violations": (
                            item.evaluation.constraint_violations
                        ),
                    }
                    for item in records
                ]
                for fidelity, records in result.stages.items()
            },
            "architecture_rejections": result.architecture_rejections,
            "search_round_count": len(result.search_rounds),
            "hpo_trials_per_surviving_structure": args.hpo_trials_per_structure,
            "fixed_six_capability_limit": False,
            "elapsed_seconds": time.perf_counter() - campaign_started,
        }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
