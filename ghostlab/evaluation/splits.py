from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCENARIO_HOLDOUT_COUNTS = {
    "buying": 20,
    "browsing": 20,
    "intent_override": 8,
    "boundary": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def profile_fingerprint(profile: object) -> str:
    serialized = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if total < 0 or total > population:
        raise ValueError("allocation total must be within the population")
    if population == 0:
        return {}
    exact = {key: total * count / population for key, count in counts.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def freeze_split(rows: list[dict[str, object]], seed: str) -> tuple[dict, dict, dict]:
    sample_ids = [row.get("sample_id") for row in rows]
    if not all(isinstance(sample_id, str) for sample_id in sample_ids):
        raise TypeError("every sample_id must be a string")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")

    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        scenario = row.get("scenario_type")
        difficulty = row.get("difficulty_bucket")
        if not isinstance(scenario, str) or not isinstance(difficulty, str):
            raise TypeError("scenario_type and difficulty_bucket must be strings")
        by_scenario[scenario].append(row)

    holdout_ids: set[str] = set()
    for scenario, target_count in SCENARIO_HOLDOUT_COUNTS.items():
        scenario_rows = by_scenario.get(scenario, [])
        if len(scenario_rows) < target_count:
            raise ValueError(f"not enough {scenario} rows for the declared holdout")
        difficulty_counts = Counter(
            str(row["difficulty_bucket"]) for row in scenario_rows
        )
        allocation = largest_remainder(dict(difficulty_counts), target_count)
        by_difficulty: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in scenario_rows:
            by_difficulty[str(row["difficulty_bucket"])].append(row)
        for difficulty, count in allocation.items():
            ordered = sorted(
                by_difficulty[difficulty],
                key=lambda row: stable_hash(str(row["sample_id"]), f"{seed}:holdout"),
            )
            holdout_ids.update(str(row["sample_id"]) for row in ordered[:count])

    adaptive_rows = [row for row in rows if str(row["sample_id"]) not in holdout_ids]
    adaptive_ids = sorted(str(row["sample_id"]) for row in adaptive_rows)
    ordered_holdout_ids = sorted(holdout_ids)
    if len(adaptive_ids) != 150 or len(ordered_holdout_ids) != 50:
        raise ValueError("expected an exact 150/50 public split")

    folds: list[list[str]] = [[] for _ in range(5)]
    strata: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in adaptive_rows:
        strata[(str(row["scenario_type"]), str(row["difficulty_bucket"]))].append(row)
    for stratum, stratum_rows in sorted(strata.items()):
        ordered = sorted(
            stratum_rows,
            key=lambda row: stable_hash(
                str(row["sample_id"]), f"{seed}:nested:{stratum}"
            ),
        )
        for index, row in enumerate(ordered):
            folds[index % len(folds)].append(str(row["sample_id"]))

    fingerprint_groups: dict[str, list[str]] = defaultdict(list)
    for row in adaptive_rows:
        fingerprint = profile_fingerprint(row.get("user_profile"))
        fingerprint_groups[fingerprint].append(str(row["sample_id"]))

    dataset_counts = Counter(str(row["scenario_type"]) for row in rows)
    adaptive_counts = Counter(str(row["scenario_type"]) for row in adaptive_rows)
    holdout_counts = Counter(
        str(row["scenario_type"])
        for row in rows
        if str(row["sample_id"]) in holdout_ids
    )
    adaptive = {
        "schema_version": 1,
        "name": "adaptive_v1",
        "seed": seed,
        "sample_ids": adaptive_ids,
        "scenario_counts": dict(sorted(adaptive_counts.items())),
    }
    nested = {
        "schema_version": 1,
        "name": "nested_v1",
        "seed": seed,
        "adaptive_sample_ids": adaptive_ids,
        "outer_folds": [sorted(fold) for fold in folds],
        "profile_fingerprint_groups": {
            key: sorted(values) for key, values in sorted(fingerprint_groups.items())
        },
    }
    guarded = {
        "schema_version": 1,
        "name": "f3_v1",
        "seed": seed,
        "sample_ids": ordered_holdout_ids,
        "scenario_counts": dict(sorted(holdout_counts.items())),
    }
    meta = {
        "dataset_scenario_counts": dict(sorted(dataset_counts.items())),
        "adaptive_count": len(adaptive_ids),
        "holdout_count": len(ordered_holdout_ids),
        "holdout_ids_sha256": hashlib.sha256(
            "\n".join(ordered_holdout_ids).encode()
        ).hexdigest(),
    }
    adaptive["prospective_meta"] = meta
    return adaptive, nested, guarded


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("every JSONL row must be an object")
            rows.append(value)
    return rows
