from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl, metric_summary
from ghostlab.research.replay import evaluate_replay
from ghostlab.research.technique_suite import (
    PROJECT_ROOT,
    UnifiedTechniqueConfig,
    build_suite_agent,
    load_suite_config,
)

CORE_FACTORS = ("N", "Q", "J", "X", "D")
RANKING_FACTORS = ("R", "E")


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _factor_sets(*, ranking: bool) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for normalizer, expansion, diversify in product((False, True), repeat=3):
        for question in (None, "Q", "J"):
            for ranker in (None, "R", "E") if ranking else (None,):
                factors = tuple(
                    factor
                    for factor, active in (
                        ("N", normalizer),
                        (question or "", question is not None),
                        (ranker or "", ranker is not None),
                        ("X", expansion),
                        ("D", diversify),
                    )
                    if active
                )
                result.append(factors)
    return sorted(set(result), key=lambda item: (len(item), item))


def _config(
    base: UnifiedTechniqueConfig,
    factors: tuple[str, ...],
    parameters: dict[str, object] | None = None,
) -> UnifiedTechniqueConfig:
    value = base.model_dump(mode="json")
    value["experiment_id"] = "w2-" + ("control" if not factors else "-".join(factors))
    if "N" in factors:
        value.update(
            normalizer="catalog_v1",
            normalizer_asset="artifacts/assets/catalog_ontology_v1.json",
        )
    if "Q" in factors:
        value.update(question_variant="candidate_eig", question_order=[])
    if "J" in factors:
        value.update(
            question_variant="joint_observable",
            question_order=[],
            joint_policy_asset="configs/assets/joint_policy_control_v1.json",
        )
    if "R" in factors:
        value.update(
            reranker="reward_lambdamart",
            reranker_model_asset=(
                "artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json"
            ),
        )
    if "E" in factors:
        value.update(
            reranker="rank_ensemble",
            reranker_model_asset=("artifacts/models/w2_ranking_v1/fold_ensemble.json"),
        )
    if "X" in factors:
        value["query_expansion"] = "prf"
    if "D" in factors:
        value["diversification"] = "facet_mmr"
    value.update(parameters or {})
    return UnifiedTechniqueConfig.model_validate(value)


def _score_sessions(sessions: list[dict]) -> dict[str, object]:
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical, 6),
        "scenario_metrics": {
            name: metric_summary(rows) for name, rows in sorted(grouped.items())
        },
    }


class ResultCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "complete":
                    self.records[str(record["key"])] = record

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if record.get("status") == "complete":
            self.records[str(record["key"])] = record


def _evaluate(
    config: UnifiedTechniqueConfig,
    sample_ids: tuple[str, ...],
    samples_by_id: dict[str, dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    cache: ResultCache,
    *,
    label: str,
) -> dict:
    key = _hash(
        {
            "config": config.model_dump(mode="json"),
            "sample_ids": sample_ids,
            "label": label,
        }
    )
    if key in cache.records:
        return cache.records[key]
    started = time.perf_counter()
    try:
        result = evaluate_replay(
            build_suite_agent(config, PROJECT_ROOT / "data/catalog.jsonl"),
            [samples_by_id[sample_id] for sample_id in sample_ids],
            categories,
            products,
        )
        record = {
            "status": "complete",
            "key": key,
            "label": label,
            "experiment_id": config.experiment_id,
            "config": config.model_dump(mode="json"),
            "sample_ids": sample_ids,
            "metrics": {
                name: result[name]
                for name in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "recommended_technical_score",
                    "scenario_metrics",
                )
            },
            "sessions": result["sessions"],
            "wall_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as error:  # noqa: BLE001 - campaign boundary records failure
        record = {
            "status": "failed",
            "key": key,
            "label": label,
            "experiment_id": config.experiment_id,
            "config": config.model_dump(mode="json"),
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": round(time.perf_counter() - started, 6),
        }
    cache.append(record)
    print(
        f"{label} {config.experiment_id} "
        f"{record.get('metrics', {}).get('recommended_technical_score', 'FAILED')} "
        f"{record['wall_seconds']}s",
        flush=True,
    )
    return record


def _inner_ids(
    training_ids: set[str], samples_by_id: dict[str, dict], fold: int
) -> tuple[str, ...]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in training_ids:
        grouped[str(samples_by_id[sample_id]["scenario_type"])].append(sample_id)
    selected: list[str] = []
    for scenario, identifiers in sorted(grouped.items()):
        ordered = sorted(
            identifiers,
            key=lambda item: hashlib.sha256(
                f"w2-inner-{fold}-{scenario}-{item}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[: max(1, round(len(ordered) * 0.3))])
    return tuple(sorted(selected))


def _parameter_trials(factors: tuple[str, ...]) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = [{}]
    if "N" in factors:
        trials.extend({"constraint_confidence": value} for value in (0.78, 0.9, 0.98))
    if "Q" in factors:
        trials.extend({"eig_candidate_k": value} for value in (50, 100, 200))
        trials.extend({"question_value_margin": value} for value in (0.0, 0.01, 0.02))
    if "X" in factors:
        trials.extend({"expansion_max_terms": value} for value in (2, 4, 6))
        trials.extend({"expansion_min_support": value} for value in (0.3, 0.4, 0.5))
    if "D" in factors:
        trials.extend({"diversification_weight": value} for value in (0.7, 0.85, 0.95))
    unique = {_hash(item): item for item in trials}
    return [unique[key] for key in sorted(unique)]


def _rank(
    records: list[tuple[tuple[str, ...], dict[str, object], dict]],
) -> list[tuple[tuple[str, ...], dict[str, object], dict]]:
    complete = [item for item in records if item[2]["status"] == "complete"]
    return sorted(
        complete,
        key=lambda item: (
            -float(item[2]["metrics"]["recommended_technical_score"]),
            len(item[0]),
            _hash(item[1]),
            item[0],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded Wave 2 interaction campaign"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/campaign/w2_combinations.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/w2_combination_campaign.json"),
    )
    parser.add_argument("--top-structures", type=int, default=4)
    args = parser.parse_args()
    if args.top_structures <= 0:
        raise ValueError("top-structures must be positive")

    base = load_suite_config(PROJECT_ROOT / "configs/suites/keyword_research.json")
    samples = load_jsonl(PROJECT_ROOT / "data/public_set.jsonl")
    nested = json.loads((PROJECT_ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples_by_id = {
        str(sample["sample_id"]): sample
        for sample in samples
        if str(sample["sample_id"]) in adaptive_ids
    }
    _, categories, products = catalog_index(PROJECT_ROOT / "data/catalog.jsonl")
    cache = ResultCache(PROJECT_ROOT / args.checkpoint)
    started = time.perf_counter()

    screen_ids = tuple(sorted(adaptive_ids))
    screen: list[tuple[tuple[str, ...], dict[str, object], dict]] = []
    for factors in _factor_sets(ranking=True):
        record = _evaluate(
            _config(base, factors),
            screen_ids,
            samples_by_id,
            categories,
            products,
            cache,
            label="all-dev-screen",
        )
        screen.append((factors, {}, record))

    outer_records: list[dict] = []
    core = _factor_sets(ranking=False)
    for fold, raw_outer in enumerate(nested["outer_folds"]):
        outer_ids = tuple(sorted(str(value) for value in raw_outer))
        inner_ids = _inner_ids(adaptive_ids - set(outer_ids), samples_by_id, fold)
        structural: list[tuple[tuple[str, ...], dict[str, object], dict]] = []
        for factors in core:
            record = _evaluate(
                _config(base, factors),
                inner_ids,
                samples_by_id,
                categories,
                products,
                cache,
                label=f"fold-{fold}-inner-structure",
            )
            structural.append((factors, {}, record))
        finalists = _rank(structural)[: args.top_structures]
        tuned: list[tuple[tuple[str, ...], dict[str, object], dict]] = list(finalists)
        for factors, _, _ in finalists:
            for parameters in _parameter_trials(factors):
                record = _evaluate(
                    _config(base, factors, parameters),
                    inner_ids,
                    samples_by_id,
                    categories,
                    products,
                    cache,
                    label=f"fold-{fold}-inner-hpo",
                )
                tuned.append((factors, parameters, record))
        winner_factors, winner_parameters, inner_winner = _rank(tuned)[0]
        outer = _evaluate(
            _config(base, winner_factors, winner_parameters),
            outer_ids,
            samples_by_id,
            categories,
            products,
            cache,
            label=f"fold-{fold}-outer-confirmation",
        )
        outer_records.append(
            {
                "fold": fold,
                "factors": winner_factors,
                "parameters": winner_parameters,
                "inner_score": inner_winner["metrics"]["recommended_technical_score"],
                "outer": outer,
            }
        )

    oof_sessions = [
        session for item in outer_records for session in item["outer"]["sessions"]
    ]
    ranked_screen = _rank(screen)
    report = {
        "schema_version": 1,
        "campaign": "wave2_default_interactions_and_nested_core",
        "protected_f3_access": False,
        "screening_warning": (
            "All-dev ranking screens use deployment assets and are exploratory only; "
            "they are not promotion evidence."
        ),
        "core_oof_label": (
            "Five-fold outer confirmation with fold-local inner structure and "
            "conditional parameter selection; core factors use no target-fitted assets."
        ),
        "screened_candidates": len(screen),
        "screen_top_10": [
            {
                "factors": factors,
                "score": record["metrics"]["recommended_technical_score"],
            }
            for factors, _, record in ranked_screen[:10]
        ],
        "outer_folds": [
            {
                "fold": item["fold"],
                "factors": item["factors"],
                "parameters": item["parameters"],
                "inner_score": item["inner_score"],
                "outer_metrics": item["outer"]["metrics"],
            }
            for item in outer_records
        ],
        "core_nested_oof_metrics": _score_sessions(oof_sessions),
        "factor_selection_counts": {
            factor: sum(factor in item["factors"] for item in outer_records)
            for factor in CORE_FACTORS
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(args.checkpoint),
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
