from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

PASSAGE_SCHEMA_VERSION = "catalog_fields_v2"
PASSAGE_FIELDS = (
    ("title", "title", 32),
    ("category", "categories", 24),
    ("price", "price", 8),
    ("store", "store", 16),
    ("features", "features", 50),
    ("details", "details", 35),
    ("description", "description", 15),
)


class PairScorer(Protocol):
    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]: ...


def _flatten(value: object) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return "" if value is None else str(value)


def product_passage(product: dict) -> str:
    parts = []
    for label, field, word_limit in PASSAGE_FIELDS:
        value = " ".join(_flatten(product.get(field)).split())
        bounded = " ".join(value.split()[:word_limit])
        if bounded:
            parts.append(f"{label}: {bounded}")
    return " | ".join(parts)


def blend_ranking(
    ranking: list[str], scores: Sequence[float], *, weight: float
) -> list[str]:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("cross-encoder weight must be between zero and one")
    if len(ranking) != len(scores):
        raise ValueError("ranking and score counts differ")
    count = len(ranking)
    original = {identifier: rank for rank, identifier in enumerate(ranking)}
    combined = {}
    for rank, (identifier, score) in enumerate(zip(ranking, scores, strict=True)):
        base = 1.0 if count == 1 else 1.0 - rank / max(1, count - 1)
        combined[identifier] = (1.0 - weight) * base + weight * float(score)
    return sorted(
        ranking,
        key=lambda identifier: (
            -combined[identifier],
            original[identifier],
            identifier,
        ),
    )


class CrossEncoderReranker:
    """Lazy, bounded Top-K neural reranker with query-head score caching."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        model_name: str,
        revision: str,
        cache_folder: str | Path,
        batch_size: int = 32,
        max_length: int = 256,
        local_files_only: bool = False,
        scorer: PairScorer | None = None,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("invalid cross-encoder batch or sequence length")
        started = time.perf_counter()
        self.documents: dict[str, str] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                self.documents[str(product["parent_asin"])] = product_passage(product)
        if scorer is None:
            import torch
            from sentence_transformers import CrossEncoder

            scorer = CrossEncoder(
                model_name,
                revision=revision,
                cache_folder=str(cache_folder),
                local_files_only=local_files_only,
                max_length=max_length,
                activation_fn=torch.nn.Sigmoid(),
                device="cpu",
            )
        self.scorer = scorer
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.initialization_seconds = time.perf_counter() - started
        self.score_seconds = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.score_calls = 0
        self._cache: dict[tuple[str, str], float] = {}

    def scores(self, query: str, ranking: list[str]) -> tuple[float, ...]:
        missing = [
            identifier
            for identifier in ranking
            if (query, identifier) not in self._cache
        ]
        self.cache_hits += len(ranking) - len(missing)
        if missing:
            pairs = [
                (query, self.documents.get(identifier, "")) for identifier in missing
            ]
            started = time.perf_counter()
            values = self.scorer.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.score_seconds += time.perf_counter() - started
            scores = tuple(float(value) for value in np.asarray(values).reshape(-1))
            if len(scores) != len(missing):
                raise ValueError("cross-encoder returned an unexpected score count")
            self._cache.update(
                ((query, identifier), score)
                for identifier, score in zip(missing, scores, strict=True)
            )
            self.cache_misses += len(missing)
            self.score_calls += 1
        return tuple(self._cache[(query, identifier)] for identifier in ranking)

    def rerank(
        self,
        query: str,
        ranking: list[str],
        *,
        rerank_k: int,
        weight: float,
    ) -> list[str]:
        if rerank_k <= 0:
            raise ValueError("rerank_k must be positive")
        if weight == 0.0:
            return list(ranking)
        head = ranking[:rerank_k]
        ordered = blend_ranking(head, self.scores(query, head), weight=weight)
        return [*ordered, *ranking[rerank_k:]]
