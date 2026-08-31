from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.gbdt import METADATA_FEATURES, GBDTFeatureStore

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
PASSAGE_SCHEMA_VERSION = "catalog_fields_v2"
MAX_LENGTH = 256
BATCH_SIZE = 32
NEURAL_FEATURES = ("cross_encoder_score", "cross_encoder_score_missing")
NEURAL_METADATA_FEATURES = (*METADATA_FEATURES, *NEURAL_FEATURES)
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


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def score_cache_identity(catalog_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "catalog_sha256": catalog_sha256,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "passage_schema_version": PASSAGE_SCHEMA_VERSION,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "activation": "sigmoid",
        "device": "cpu",
    }


class PinnedCrossEncoderScorer:
    """Exact local-only zero-shot scorer frozen by the interaction manifest."""

    def __init__(
        self,
        catalog_path: str | Path,
        model_cache: str | Path,
        *,
        scorer: PairScorer | None = None,
    ) -> None:
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
                MODEL_NAME,
                revision=MODEL_REVISION,
                cache_folder=str(model_cache),
                local_files_only=True,
                max_length=MAX_LENGTH,
                activation_fn=torch.nn.Sigmoid(),
                device="cpu",
            )
        self.scorer = scorer
        self.initialization_seconds = time.perf_counter() - started
        self.score_seconds = 0.0
        self.score_calls = 0
        self.scored_pairs = 0

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        inputs = [
            (query, self.documents.get(identifier, "")) for query, identifier in pairs
        ]
        started = time.perf_counter()
        values = self.scorer.predict(
            inputs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.score_seconds += time.perf_counter() - started
        self.score_calls += 1
        scores = tuple(float(value) for value in np.asarray(values).reshape(-1))
        if len(scores) != len(pairs):
            raise ValueError("cross-encoder returned an unexpected score count")
        self.scored_pairs += len(scores)
        return scores


class NeuralScoreCache:
    """Content-identified query-candidate scores, independent of learned folds."""

    def __init__(self, path: str | Path, expected_identity: dict[str, object]) -> None:
        self.path = Path(path)
        self.scores: dict[tuple[str, str], float] = {}
        with self.path.open(encoding="utf-8") as handle:
            header = json.loads(next(handle))
            if (
                header.get("type") != "header"
                or header.get("identity") != expected_identity
            ):
                raise ValueError("neural score cache identity mismatch")
            self.header = header
            for line in handle:
                record = json.loads(line)
                key = (str(record["query_sha256"]), str(record["parent_asin"]))
                if key in self.scores:
                    raise ValueError("duplicate neural score cache key")
                score = float(record["score"])
                if not math.isfinite(score):
                    raise ValueError("non-finite neural score cache value")
                self.scores[key] = score
        if int(self.header["row_count"]) != len(self.scores):
            raise ValueError("neural score cache row count mismatch")

    def get(self, query: str, identifier: str) -> float | None:
        return self.scores.get((query_hash(query), identifier))


def write_score_cache(
    path: str | Path,
    identity: dict[str, object],
    pairs: Sequence[tuple[str, str]],
    scorer: PinnedCrossEncoderScorer,
    *,
    chunk_size: int = 2048,
) -> None:
    unique = sorted(
        {(query_hash(query), query, identifier) for query, identifier in pairs}
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "header", "identity": identity, "row_count": len(unique)},
                sort_keys=True,
            )
            + "\n"
        )
        for start in range(0, len(unique), chunk_size):
            chunk = unique[start : start + chunk_size]
            scores = scorer.score_pairs(
                [(query, identifier) for _, query, identifier in chunk]
            )
            for (hashed, _, identifier), score in zip(chunk, scores, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "query_sha256": hashed,
                            "parent_asin": identifier,
                            "score": score,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                f"cached neural scores {min(start + chunk_size, len(unique))}/{len(unique)}",
                flush=True,
            )
    temporary.replace(destination)


class NeuralGBDTFeatureStore:
    """Audited metadata features plus one pinned score and its missingness."""

    def __init__(
        self,
        base: GBDTFeatureStore,
        *,
        cache: NeuralScoreCache | None = None,
        live_scorer: PinnedCrossEncoderScorer | None = None,
    ) -> None:
        if cache is None and live_scorer is None:
            raise ValueError("a cached or live cross-encoder scorer is required")
        self.base = base
        self.cache = cache
        self.live_scorer = live_scorer
        self._live_scores: dict[tuple[str, str], float] = {}
        self.missing_count = 0

    def matrix(
        self,
        query: str,
        ranking: list[str] | tuple[str, ...],
        feature_names: tuple[str, ...],
    ) -> NDArray[np.float64]:
        if feature_names != NEURAL_METADATA_FEATURES:
            raise ValueError("neural rank uses only the predeclared feature ordering")
        base = self.base.matrix(query, ranking, METADATA_FEATURES)
        values: list[float | None] = []
        for identifier in ranking:
            value = self._live_scores.get((query, identifier))
            if value is None and self.cache is not None:
                value = self.cache.get(query, identifier)
            values.append(value)
        missing_indices = [index for index, value in enumerate(values) if value is None]
        if missing_indices and self.live_scorer is not None:
            pairs = [(query, ranking[index]) for index in missing_indices]
            observed = self.live_scorer.score_pairs(pairs)
            for index, score in zip(missing_indices, observed, strict=True):
                identifier = ranking[index]
                self._live_scores[(query, identifier)] = score
                values[index] = score
        missing = np.asarray([value is None for value in values], dtype=np.float64)
        self.missing_count += int(missing.sum())
        scores = np.asarray(
            [math.nan if value is None else float(value) for value in values],
            dtype=np.float64,
        )
        return np.column_stack((base, scores, missing))
