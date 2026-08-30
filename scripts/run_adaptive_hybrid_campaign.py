from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from evaluator.local_evaluator import catalog_index, evaluate
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_campaign import (
    AdaptiveEvaluation,
    AdaptiveGhostLabEngine,
)
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.training.adaptive_datasets import (
    load_adaptive_training_corpus,
    progressive_stratified_samples,
)
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run architecture-safe GhostLab champion/challenger racing inside "
            "the fixed adaptive 1A-3B workflow"
        )
    )
    parser.add_argument("--config", default="configs/adaptive_hybrid_1a_3b_v1.json")
    parser.add_argument(
        "--technique-catalog", default="configs/techniques/catalog_v2.json"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help=(
            "repeat for each evaluation JSONL; defaults to data/public_set.jsonl"
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-extra-techniques", type=int)
    parser.add_argument("--higher-order-rounds", type=int, default=8)
    parser.add_argument("--f1-candidates", type=int, default=24)
    parser.add_argument("--f2-candidates", type=int, default=6)
    parser.add_argument("--hpo-trials-per-structure", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print the complete race plan without evaluating/training",
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_hybrid_campaign.json"
    )
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    baseline = load_adaptive_hybrid_config(ROOT / args.config)
    catalog = load_catalog(ROOT / args.technique_catalog)
    registry = AdaptiveTechniqueRegistry.from_catalog(catalog, project_root=ROOT)
    engine = AdaptiveGhostLabEngine(
        baseline=baseline,
        registry=registry,
        candidate_limit=args.candidate_limit,
        beam_width=args.beam_width,
        max_extra_techniques=args.max_extra_techniques,
        seed=args.seed,
    )
    inventory = registry.inventory()
    initial_plan = engine.initial_plan()
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
            "skipped": [
                {"roots": item.roots, "reasons": item.reasons}
                for item in initial_plan.skipped
            ],
            "maximum_extra_techniques": engine.limits.max_order,
            "fixed_six_capability_limit": False,
            "hpo_trials_per_surviving_structure": args.hpo_trials_per_structure,
        }
    else:
        dataset_paths = tuple(args.datasets or ("data/public_set.jsonl",))
        corpus = load_adaptive_training_corpus(ROOT, dataset_paths)
        samples = progressive_stratified_samples(corpus, seed=args.seed)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        catalog_path = ROOT / "data/catalog.jsonl"
        identifiers, categories, products = catalog_index(catalog_path)
        evaluation_ordinal = 0

        def evaluator(
            config,
            candidate: CandidateSpec,
            fidelity: str,  # type: ignore[no-untyped-def]
        ) -> AdaptiveEvaluation:
            nonlocal evaluation_ordinal
            evaluation_ordinal += 1
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
            agent = AdaptiveHybridAgent(catalog_path, config, project_root=ROOT)
            result = evaluate(
                cast(Agent, agent),
                samples[:count],
                identifiers,
                categories,
                products,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            sessions = result["sessions"]
            evaluation = AdaptiveEvaluation(
                candidate_id=candidate.candidate_id,
                fidelity=cast(Any, fidelity),
                score=float(result["recommended_technical_score"]),
                session_rewards=tuple(_session_reward(item) for item in sessions),
                behavior_novelty=0.0,
                latency_p95_ms=elapsed_ms / count,
                fit_verified=False,
            )
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
            "fidelity_sample_counts": {
                fidelity: max(1, round(len(samples) * fraction))
                for fidelity, fraction in {"f0": 0.2, "f1": 0.5, "f2": 1.0}.items()
            },
            "incumbent": _candidate_payload(result.incumbent),
            "selected": _candidate_payload(result.selected),
            "promoted": result.promoted,
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
                        "mean_paired_delta": sum(item.paired_deltas)
                        / len(item.paired_deltas),
                        "latency_p95_ms": item.evaluation.latency_p95_ms,
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
