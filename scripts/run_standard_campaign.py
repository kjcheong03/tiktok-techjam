from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

from baseline.state import ASK_ORDER
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES = (*ASK_ORDER, "other")
STATE_VARIANTS = ("multi", "multi", "multi", "compressed", "raw_history", "single")


def canonical_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def complexity(config: dict) -> int:
    order = config.get("question_order") or []
    score = 1 + len(set(order))
    score += int(config["state_variant"] not in {"multi", "single"})
    score += int(config["retrieval_route"] != "keyword") * 3
    score += int(config["reranker"] != "none") * 2
    return score


def anchors() -> list[dict]:
    return [
        {
            "state_variant": "multi",
            "question_variant": "other_always",
            "question_order": None,
            "repeat_last_question": False,
            "negative_evidence": True,
            "override_invalidation": True,
            "retrieval_route": "keyword",
            "sparse_weight": 0.75,
            "dense_weight": 0.25,
            "reranker": "none",
        },
        {
            "state_variant": "multi",
            "question_variant": "sequence",
            "question_order": list(ASK_ORDER),
            "repeat_last_question": False,
            "negative_evidence": True,
            "override_invalidation": True,
            "retrieval_route": "keyword",
            "sparse_weight": 0.75,
            "dense_weight": 0.25,
            "reranker": "none",
        },
        {
            "state_variant": "raw_history",
            "question_variant": "other_always",
            "question_order": None,
            "repeat_last_question": False,
            "negative_evidence": True,
            "override_invalidation": True,
            "retrieval_route": "keyword",
            "sparse_weight": 0.75,
            "dense_weight": 0.25,
            "reranker": "none",
        },
    ]


def random_config(rng: random.Random) -> dict:
    roll = rng.random()
    if roll < 0.12:
        question_variant = "other_always"
        order = None
    elif roll < 0.22:
        question_variant = "missing_priority"
        order = None
    else:
        question_variant = "sequence"
        length = rng.randint(1, 10)
        weights = [1, 1, 1, 1, 1, 1, 1, 3]
        order = rng.choices(ATTRIBUTES, weights=weights, k=length)
    route_roll = rng.random()
    if route_roll < 0.84:
        route = "keyword"
    elif route_roll < 0.89:
        route = "dense"
    elif route_roll < 0.95:
        route = "rrf"
    else:
        route = "weighted"
    sparse_weight = rng.choice((0.75, 0.8, 0.85, 0.9, 0.95))
    return {
        "state_variant": rng.choice(STATE_VARIANTS),
        "question_variant": question_variant,
        "question_order": order,
        "repeat_last_question": bool(order) and rng.random() < 0.25,
        "negative_evidence": rng.random() < 0.85,
        "override_invalidation": rng.random() < 0.9,
        "retrieval_route": route,
        "sparse_weight": sparse_weight,
        "dense_weight": round(1.0 - sparse_weight, 2),
        "reranker": "linear" if route == "keyword" and rng.random() < 0.08 else "none",
    }


def generate_candidates(max_candidates: int, seeds: list[int]) -> list[dict]:
    candidates = anchors()
    seen = {canonical_hash(config) for config in candidates}
    seed_index = 0
    generators = [random.Random(seed) for seed in seeds]
    while len(candidates) < max_candidates:
        config = random_config(generators[seed_index % len(generators)])
        config["search_seed"] = seeds[seed_index % len(seeds)]
        seed_index += 1
        key = canonical_hash(config)
        if key not in seen:
            seen.add(key)
            candidates.append(config)
    return candidates


def load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "completed":
                completed[str(record["candidate_hash"])] = record
    return completed


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(records: list[dict], nested: dict, tie_band: float) -> dict:
    by_hash = {record["candidate_hash"]: record for record in records}
    ranked = sorted(
        records,
        key=lambda item: (
            -item["metrics"]["recommended_technical_score"],
            item["complexity"],
            item["candidate_hash"],
        ),
    )
    best_score = ranked[0]["metrics"]["recommended_technical_score"]
    selected = min(
        (
            record
            for record in ranked
            if record["metrics"]["recommended_technical_score"] >= best_score - tie_band
        ),
        key=lambda item: (
            item["complexity"],
            -item["metrics"]["recommended_technical_score"],
        ),
    )
    nested_folds = []
    winner_counts: dict[str, int] = {}
    all_ids = set(nested["adaptive_sample_ids"])
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = set(outer_values)
        training = all_ids - outer

        def mean_reward(record: dict, ids: set[str]) -> float:
            sessions = {str(item["sample_id"]): item for item in record["sessions"]}
            return statistics.fmean(
                session_reward(sessions[sample_id]) for sample_id in ids
            )

        training_scores = [
            (record, mean_reward(record, training)) for record in records
        ]
        fold_best = max(score for _, score in training_scores)
        winner = min(
            (
                (record, score)
                for record, score in training_scores
                if score >= fold_best - tie_band
            ),
            key=lambda item: (
                item[0]["complexity"],
                -item[1],
                item[0]["candidate_hash"],
            ),
        )[0]
        winner_counts[winner["candidate_hash"]] = (
            winner_counts.get(winner["candidate_hash"], 0) + 1
        )
        nested_folds.append(
            {
                "outer_fold": fold_index,
                "winner": winner["candidate_hash"],
                "outer_mean_reward": round(mean_reward(winner, outer), 6),
            }
        )
    checkpoints = [10, 25, 50, 100, 250, 500]
    convergence = []
    best = float("-inf")
    for index, record in enumerate(records, start=1):
        best = max(best, record["metrics"]["recommended_technical_score"])
        if index in checkpoints or index == len(records):
            convergence.append({"candidates": index, "best_score": round(best, 6)})
    return {
        "completed_candidates": len(records),
        "best_candidate": ranked[0]["candidate_hash"],
        "best_score": best_score,
        "tie_band_selected_candidate": selected["candidate_hash"],
        "tie_band_selected_score": selected["metrics"]["recommended_technical_score"],
        "tie_band_selected_complexity": selected["complexity"],
        "selected_config": selected["config"],
        "top_10": [
            {
                "candidate_hash": item["candidate_hash"],
                "score": item["metrics"]["recommended_technical_score"],
                "complexity": item["complexity"],
                "config": item["config"],
            }
            for item in ranked[:10]
        ],
        "nested_outer_folds": nested_folds,
        "nested_mean_outer_reward": round(
            statistics.fmean(item["outer_mean_reward"] for item in nested_folds), 6
        ),
        "nested_winner_counts": winner_counts,
        "convergence": convergence,
        "record_hashes": sorted(by_hash),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded adaptive-only policy campaign"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/search/standard.json"
    )
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()
    campaign = json.loads(args.config.read_text())
    max_candidates = args.max_candidates or int(campaign["max_candidates"])
    max_minutes = args.max_minutes or float(campaign["max_minutes"])
    seeds = [int(seed) for seed in campaign["seeds"]]
    checkpoint = ROOT / campaign["checkpoint"]
    completed = load_completed(checkpoint)
    candidates = generate_candidates(max_candidates, seeds)
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    adaptive_ids = set(split["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in adaptive_ids]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    started = time.monotonic()
    for ordinal, config in enumerate(candidates, start=1):
        candidate_hash = canonical_hash(config)
        if candidate_hash in completed:
            continue
        if time.monotonic() - started >= max_minutes * 60:
            break
        options = {key: value for key, value in config.items() if key != "search_seed"}
        if options["question_order"] is not None:
            options["question_order"] = tuple(options["question_order"])
        candidate_started = time.perf_counter()
        try:
            result = evaluate(
                ExperimentalAgent(ROOT / "data/catalog.jsonl", **options),  # type: ignore[arg-type]
                samples,
                catalog_ids,
                categories,
                products,
            )
            record = {
                "status": "completed",
                "ordinal": ordinal,
                "candidate_hash": candidate_hash,
                "config": config,
                "complexity": complexity(config),
                "metrics": {
                    key: result[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "recommended_technical_score",
                        "scenario_metrics",
                    )
                },
                "sessions": result["sessions"],
                "wall_seconds": round(time.perf_counter() - candidate_started, 6),
            }
        except Exception as error:  # noqa: BLE001 - campaign records failures and continues
            record = {
                "status": "failed",
                "ordinal": ordinal,
                "candidate_hash": candidate_hash,
                "config": config,
                "error_type": type(error).__name__,
                "wall_seconds": round(time.perf_counter() - candidate_started, 6),
            }
        append_record(checkpoint, record)
        if record["status"] == "completed":
            completed[candidate_hash] = record
        print(
            f"{len(completed)}/{max_candidates} {candidate_hash[:10]} "
            f"{record.get('metrics', {}).get('recommended_technical_score', 'FAILED')} "
            f"{record['wall_seconds']}s",
            flush=True,
        )
    records = list(completed.values())
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    summary = {
        "phase": 8,
        "campaign": campaign["name"],
        "status": "complete" if len(records) >= max_candidates else "budget_exhausted",
        "max_candidates": max_candidates,
        "max_minutes": max_minutes,
        "seeds": seeds,
        "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        **summarize(records, nested, float(campaign["tie_band"])),
    }
    output = ROOT / "artifacts/reports/phase8_standard_campaign.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "record_hashes"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
