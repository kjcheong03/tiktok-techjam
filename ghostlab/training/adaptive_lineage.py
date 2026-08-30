from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ghostlab.training.adaptive_datasets import AdaptiveTrainingCorpus

Partition = Literal["development", "holdout"]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest_strings(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _stable_key(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()


@dataclass(frozen=True)
class LineageGroup:
    group_id: str
    family: Literal["public_derived", "independent_template"]
    scenario_type: str
    difficulty_bucket: str
    member_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "family": self.family,
            "scenario_type": self.scenario_type,
            "difficulty_bucket": self.difficulty_bucket,
            "member_ids": list(self.member_ids),
        }


@dataclass(frozen=True)
class AdaptiveLineageManifest:
    path: Path
    payload: dict[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.payload).encode()).hexdigest()

    @property
    def development_ids(self) -> frozenset[str]:
        return frozenset(self.payload["partitions"]["development"]["sample_ids"])

    @property
    def holdout_ids(self) -> frozenset[str]:
        return frozenset(self.payload["partitions"]["holdout"]["sample_ids"])

    @property
    def group_by_sample(self) -> dict[str, str]:
        return {
            member: str(group["group_id"])
            for group in self.payload["lineage_groups"]
            for member in group["member_ids"]
        }

    @property
    def family_by_group(self) -> dict[str, str]:
        return {
            str(group["group_id"]): str(group["family"])
            for group in self.payload["lineage_groups"]
        }

    def ids(self, partition: Partition) -> frozenset[str]:
        return self.development_ids if partition == "development" else self.holdout_ids


def _source_rows(corpus: AdaptiveTrainingCorpus, token: str) -> list[dict[str, Any]]:
    matched = [
        sample
        for sample_id, sample in corpus.samples.items()
        if token in corpus.origins[sample_id]
    ]
    return matched


def reconstruct_lineage_groups(
    corpus: AdaptiveTrainingCorpus,
) -> tuple[tuple[LineageGroup, ...], dict[str, Any]]:
    public = _source_rows(corpus, "public_set.jsonl")
    synthetic = _source_rows(corpus, "synthetic_1000_public_like.jsonl")
    independent = _source_rows(corpus, "independent_template_1000.jsonl")
    if (len(public), len(synthetic), len(independent)) != (200, 1000, 1000):
        raise ValueError(
            "lineage reconstruction requires the exact 200+1000+1000 corpus"
        )

    mismatches: list[dict[str, object]] = []
    groups: list[LineageGroup] = []
    for index, public_row in enumerate(public):
        variants = synthetic[index * 5 : (index + 1) * 5]
        expected = (
            str(public_row["scenario_type"]),
            str(public_row["difficulty_bucket"]),
            _canonical(public_row["user_profile"]),
        )
        invalid = [
            str(row["sample_id"])
            for row in variants
            if (
                str(row["scenario_type"]),
                str(row["difficulty_bucket"]),
                _canonical(row["user_profile"]),
            )
            != expected
        ]
        if invalid:
            mismatches.append(
                {
                    "public_sample_id": public_row["sample_id"],
                    "variant_mismatches": invalid,
                }
            )
        groups.append(
            LineageGroup(
                group_id=f"public_derived:{public_row['sample_id']}",
                family="public_derived",
                scenario_type=expected[0],
                difficulty_bucket=expected[1],
                member_ids=(
                    str(public_row["sample_id"]),
                    *(str(row["sample_id"]) for row in variants),
                ),
            )
        )

    independent_families: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in independent:
        independent_families[
            (str(row["scenario_type"]), _canonical(row["user_profile"]))
        ].append(row)
    for ordinal, ((scenario, profile), rows) in enumerate(
        sorted(independent_families.items()), start=1
    ):
        if len(rows) != 5:
            mismatches.append(
                {
                    "independent_family": hashlib.sha256(profile.encode()).hexdigest()[
                        :12
                    ],
                    "expected_members": 5,
                    "actual_members": len(rows),
                }
            )
        group_hash = hashlib.sha256(f"{scenario}\0{profile}".encode()).hexdigest()[:16]
        groups.append(
            LineageGroup(
                group_id=f"independent_template:{ordinal:03d}:{group_hash}",
                family="independent_template",
                scenario_type=scenario,
                difficulty_bucket=str(rows[0]["difficulty_bucket"]),
                member_ids=tuple(str(row["sample_id"]) for row in rows),
            )
        )

    members = [member for group in groups for member in group.member_ids]
    duplicate_members = sorted(
        member for member, count in Counter(members).items() if count != 1
    )
    missing_members = sorted(corpus.sample_ids - set(members))
    unexpected_members = sorted(set(members) - corpus.sample_ids)
    verified = (
        len(groups) == 400
        and len([g for g in groups if g.family == "public_derived"]) == 200
        and len([g for g in groups if g.family == "independent_template"]) == 200
        and not mismatches
        and not duplicate_members
        and not missing_members
        and not unexpected_members
    )
    audit = {
        "schema_version": 1,
        "algorithm": "adaptive_lineage_reconstruction_v1",
        "status": "verified" if verified else "failed",
        "candidate_group_count": len(groups),
        "family_group_counts": dict(Counter(group.family for group in groups)),
        "member_count": len(members),
        "mismatches": mismatches,
        "duplicate_members": duplicate_members,
        "missing_members": missing_members,
        "unexpected_members": unexpected_members,
        "limitations": [
            "generator template_family metadata is absent",
            "public lineage is reconstructed from verified row ordering plus exact scenario/difficulty/profile equality",
            "independent lineage is reconstructed from exact scenario/profile families",
        ],
    }
    if not verified:
        raise ValueError(f"lineage reconstruction audit failed: {audit}")
    return tuple(groups), audit


def _allocate_holdout_groups(
    groups: list[LineageGroup], *, count: int, seed: int
) -> set[str]:
    buckets: dict[tuple[str, str], list[LineageGroup]] = defaultdict(list)
    for group in groups:
        buckets[(group.scenario_type, group.difficulty_bucket)].append(group)
    quotas: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    for bucket, items in buckets.items():
        raw = count * len(items) / len(groups)
        quotas[bucket] = math.floor(raw)
        fractions.append((raw - math.floor(raw), bucket))
    remaining = count - sum(quotas.values())
    for _, bucket in sorted(fractions, key=lambda item: (-item[0], item[1]))[
        :remaining
    ]:
        quotas[bucket] += 1
    selected: set[str] = set()
    for bucket, items in sorted(buckets.items()):
        ordered = sorted(
            items,
            key=lambda group: _stable_key(
                seed, "holdout:" + repr(bucket), group.group_id
            ),
        )
        selected.update(group.group_id for group in ordered[: quotas[bucket]])
    if len(selected) != count:
        raise AssertionError("holdout group allocation produced the wrong count")
    return selected


def _outer_group_folds(
    groups: list[LineageGroup], *, fold_count: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    buckets: dict[tuple[str, str, str], list[LineageGroup]] = defaultdict(list)
    for group in groups:
        buckets[(group.family, group.scenario_type, group.difficulty_bucket)].append(
            group
        )
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for bucket, items in sorted(buckets.items()):
        ordered = sorted(
            items,
            key=lambda group: _stable_key(
                seed, "outer:" + repr(bucket), group.group_id
            ),
        )
        offset = (
            int.from_bytes(_stable_key(seed, "offset", repr(bucket))[:2], "big")
            % fold_count
        )
        for index, group in enumerate(ordered):
            folds[(offset + index) % fold_count].append(group.group_id)
    if any(not fold for fold in folds):
        raise ValueError("group stratification produced an empty outer fold")
    return tuple(tuple(sorted(fold)) for fold in folds)


def build_lineage_manifest(
    corpus: AdaptiveTrainingCorpus, *, seed: int = 20260831, fold_count: int = 5
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups, audit = reconstruct_lineage_groups(corpus)
    public_groups = [group for group in groups if group.family == "public_derived"]
    independent_groups = [
        group for group in groups if group.family == "independent_template"
    ]
    holdout_groups = _allocate_holdout_groups(public_groups, count=50, seed=seed)
    holdout_groups.update(
        _allocate_holdout_groups(independent_groups, count=50, seed=seed + 1)
    )
    development_groups = [
        group for group in groups if group.group_id not in holdout_groups
    ]
    development_ids = sorted(
        member for group in development_groups for member in group.member_ids
    )
    holdout_ids = sorted(
        member
        for group in groups
        if group.group_id in holdout_groups
        for member in group.member_ids
    )
    if (len(development_ids), len(holdout_ids)) != (1650, 550):
        raise AssertionError("lineage split did not produce 1650/550 sessions")
    if set(development_ids) & set(holdout_ids):
        raise AssertionError("development and holdout samples overlap")
    outer_group_folds = _outer_group_folds(
        development_groups, fold_count=fold_count, seed=seed
    )
    by_id = {group.group_id: group for group in groups}
    outer_folds = [
        {
            "group_ids": list(fold),
            "sample_ids": sorted(
                member for group_id in fold for member in by_id[group_id].member_ids
            ),
        }
        for fold in outer_group_folds
    ]
    source_counts = {
        partition: dict(Counter(corpus.origins[sample_id] for sample_id in identifiers))
        for partition, identifiers in (
            ("development", development_ids),
            ("holdout", holdout_ids),
        )
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "adaptive_lineage_75_25_v1",
        "seed": seed,
        "fold_count": fold_count,
        "source_files": [source.__dict__ for source in corpus.sources],
        "audit": audit,
        "lineage_groups": [group.as_dict() for group in groups],
        "partitions": {
            "development": {
                "sample_ids": development_ids,
                "sample_ids_sha256": _digest_strings(development_ids),
                "group_ids": sorted(group.group_id for group in development_groups),
                "source_counts": source_counts["development"],
            },
            "holdout": {
                "sample_ids": holdout_ids,
                "sample_ids_sha256": _digest_strings(holdout_ids),
                "group_ids": sorted(holdout_groups),
                "source_counts": source_counts["holdout"],
            },
        },
        "development_outer_folds": outer_folds,
        "checks": {
            "group_count": len(groups),
            "development_sample_count": len(development_ids),
            "holdout_sample_count": len(holdout_ids),
            "partition_intersection_count": len(
                set(development_ids) & set(holdout_ids)
            ),
            "all_samples_covered": set(development_ids) | set(holdout_ids)
            == corpus.sample_ids,
            "lineage_groups_disjoint": True,
            "target_ids_unique_in_source_corpus": True,
        },
    }
    return payload, audit


def load_lineage_manifest(
    path: Path, corpus: AdaptiveTrainingCorpus
) -> AdaptiveLineageManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("algorithm") != "adaptive_lineage_75_25_v1"
    ):
        raise ValueError("unsupported adaptive lineage manifest")
    expected_sources = {source.path: source.sha256 for source in corpus.sources}
    actual_sources = {
        str(item["path"]): str(item["sha256"]) for item in payload["source_files"]
    }
    if actual_sources != expected_sources:
        raise ValueError("lineage manifest source hashes do not match the corpus")
    manifest = AdaptiveLineageManifest(path, payload)
    if manifest.development_ids & manifest.holdout_ids:
        raise ValueError("lineage manifest partitions overlap")
    if manifest.development_ids | manifest.holdout_ids != corpus.sample_ids:
        raise ValueError("lineage manifest does not cover the corpus exactly")
    groups = manifest.group_by_sample
    if set(groups) != corpus.sample_ids or len(set(groups.values())) != 400:
        raise ValueError("lineage manifest group membership is invalid")
    return manifest


def subset_corpus(
    corpus: AdaptiveTrainingCorpus,
    manifest: AdaptiveLineageManifest,
    partition: Partition,
) -> AdaptiveTrainingCorpus:
    selected = manifest.ids(partition)
    samples = {sample_id: corpus.samples[sample_id] for sample_id in selected}
    origins = {sample_id: corpus.origins[sample_id] for sample_id in selected}
    return AdaptiveTrainingCorpus(samples, origins, corpus.sources)


def manifest_outer_folds(
    manifest: AdaptiveLineageManifest,
) -> tuple[tuple[str, ...], ...]:
    folds = tuple(
        tuple(str(item) for item in fold["sample_ids"])
        for fold in manifest.payload["development_outer_folds"]
    )
    seen: set[str] = set()
    group_by_sample = manifest.group_by_sample
    group_owner: dict[str, int] = {}
    for index, fold in enumerate(folds):
        if seen & set(fold):
            raise ValueError("outer folds overlap")
        seen.update(fold)
        for sample_id in fold:
            group_id = group_by_sample[sample_id]
            owner = group_owner.setdefault(group_id, index)
            if owner != index:
                raise ValueError("a lineage group crosses outer folds")
    if seen != manifest.development_ids:
        raise ValueError("outer folds do not cover development exactly")
    return folds


def cluster_ids_for_samples(
    manifest: AdaptiveLineageManifest, sample_ids: list[str] | tuple[str, ...]
) -> tuple[str, ...]:
    mapping = manifest.group_by_sample
    return tuple(mapping[sample_id] for sample_id in sample_ids)


__all__ = [
    "AdaptiveLineageManifest",
    "LineageGroup",
    "build_lineage_manifest",
    "cluster_ids_for_samples",
    "load_lineage_manifest",
    "manifest_outer_folds",
    "reconstruct_lineage_groups",
    "subset_corpus",
]
