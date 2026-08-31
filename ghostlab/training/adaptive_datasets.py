from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptiveDatasetSource:
    path: str
    sha256: str
    sample_count: int


@dataclass(frozen=True)
class AdaptiveTrainingCorpus:
    samples: dict[str, dict[str, Any]]
    origins: dict[str, str]
    sources: tuple[AdaptiveDatasetSource, ...]

    @property
    def sample_ids(self) -> set[str]:
        return set(self.samples)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_adaptive_training_corpus(
    project_root: str | Path,
    dataset_paths: list[str | Path] | tuple[str | Path, ...],
) -> AdaptiveTrainingCorpus:
    """Load multiple grounded evaluator datasets without silent duplication."""

    root = Path(project_root).resolve()
    if not dataset_paths:
        raise ValueError("at least one adaptive training dataset is required")
    samples: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    sources: list[AdaptiveDatasetSource] = []
    target_owner: dict[str, str] = {}
    for raw_path in dataset_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing adaptive dataset: {path}")
        origin = _relative_path(path, root)
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"{origin}:{line_number} is not an object")
                sample_id = str(payload.get("sample_id") or "")
                scenario = str(payload.get("scenario_type") or "")
                ground_truth = payload.get("ground_truth")
                target = (
                    str(ground_truth.get("parent_asin") or "")
                    if isinstance(ground_truth, dict)
                    else ""
                )
                profile = payload.get("user_profile")
                if not sample_id or not scenario or not target:
                    raise ValueError(
                        f"{origin}:{line_number} is missing sample/scenario/target"
                    )
                if not isinstance(profile, dict):
                    raise TypeError(
                        f"{origin}:{line_number} has no object user_profile"
                    )
                if sample_id in samples:
                    raise ValueError(f"duplicate sample_id across datasets: {sample_id}")
                if target in target_owner:
                    raise ValueError(
                        "duplicate target across datasets would leak between folds: "
                        f"{target} ({target_owner[target]}, {sample_id})"
                    )
                samples[sample_id] = payload
                origins[sample_id] = origin
                target_owner[target] = sample_id
                count += 1
        if count == 0:
            raise ValueError(f"adaptive dataset is empty: {origin}")
        sources.append(
            AdaptiveDatasetSource(
                path=origin,
                sha256=_sha256_file(path),
                sample_count=count,
            )
        )
    return AdaptiveTrainingCorpus(samples, origins, tuple(sources))


def stratified_outer_folds(
    corpus: AdaptiveTrainingCorpus,
    *,
    fold_count: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Create deterministic source/scenario-stratified outer folds."""

    if fold_count < 3:
        raise ValueError("nested training requires at least three folds")
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample_id, sample in corpus.samples.items():
        buckets[(corpus.origins[sample_id], str(sample["scenario_type"]))].append(
            sample_id
        )
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for bucket, identifiers in sorted(buckets.items()):
        ordered = sorted(
            identifiers,
            key=lambda sample_id: hashlib.sha256(
                f"{seed}\0{bucket[0]}\0{bucket[1]}\0{sample_id}".encode()
            ).digest(),
        )
        for index, sample_id in enumerate(ordered):
            folds[index % fold_count].append(sample_id)
    if any(not fold for fold in folds):
        raise ValueError("stratification produced an empty outer fold")
    return tuple(tuple(sorted(fold)) for fold in folds)


def progressive_stratified_samples(
    corpus: AdaptiveTrainingCorpus,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Order samples so every fidelity prefix approximates every source/scenario."""

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample_id, sample in corpus.samples.items():
        buckets[(corpus.origins[sample_id], str(sample["scenario_type"]))].append(
            sample_id
        )
    positioned: list[tuple[float, str, str]] = []
    for bucket, identifiers in sorted(buckets.items()):
        ordered = sorted(
            identifiers,
            key=lambda sample_id: hashlib.sha256(
                f"{seed}\0progressive\0{bucket[0]}\0{bucket[1]}\0{sample_id}".encode()
            ).digest(),
        )
        size = len(ordered)
        bucket_name = "\0".join(bucket)
        positioned.extend(
            ((index + 0.5) / size, bucket_name, sample_id)
            for index, sample_id in enumerate(ordered)
        )
    positioned.sort()
    return [corpus.samples[sample_id] for _, _, sample_id in positioned]


def fold_manifest(
    corpus: AdaptiveTrainingCorpus,
    folds: tuple[tuple[str, ...], ...],
    *,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "seed": seed,
        "sample_count": len(corpus.samples),
        "dataset_sources": [source.__dict__ for source in corpus.sources],
        "outer_folds": [list(fold) for fold in folds],
    }


__all__ = [
    "AdaptiveDatasetSource",
    "AdaptiveTrainingCorpus",
    "fold_manifest",
    "load_adaptive_training_corpus",
    "progressive_stratified_samples",
    "stratified_outer_folds",
]
