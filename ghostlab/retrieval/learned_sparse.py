from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class SparseEncoder(Protocol):
    """Model-independent contract used by the index builder and runtime."""

    def encode(self, texts: Sequence[str]) -> list[Mapping[int, float]]: ...


@dataclass(frozen=True)
class LearnedSparseAsset:
    schema_version: int
    technique_id: str
    availability: str
    model_name: str
    revision: str | None
    model_path: str | None
    index_path: str | None
    index_sha256: str | None
    catalog_sha256: str | None
    max_terms: int
    unavailable_reason: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> LearnedSparseAsset:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        asset = cls(**value)
        if asset.schema_version != 1:
            raise ValueError("unsupported learned-sparse asset schema")
        if asset.technique_id != "retrieval.splade_rescue.v1":
            raise ValueError("unexpected learned-sparse technique ID")
        if asset.max_terms <= 0:
            raise ValueError("max_terms must be positive")
        return asset

    def require_available(self, root: str | Path) -> tuple[Path, Path]:
        if self.availability != "available":
            reason = self.unavailable_reason or "asset has not passed feasibility gates"
            raise RuntimeError(f"learned-sparse technique is unavailable: {reason}")
        if not self.model_path or not self.index_path or not self.index_sha256:
            raise ValueError("available learned-sparse manifest is incomplete")
        project = Path(root).resolve()
        model = _confined(project, self.model_path)
        index = _confined(project, self.index_path)
        if not model.is_dir() or not index.is_file():
            raise FileNotFoundError("learned-sparse model or index is missing")
        if sha256_file(index) != self.index_sha256:
            raise ValueError("learned-sparse index checksum mismatch")
        return model, index


def _confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("asset path escapes the project root") from error
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SpladeEncoder:
    """Offline-only SPLADE encoder; optional dependencies load on construction."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_terms: int = 128,
        max_length: int = 256,
        batch_size: int = 8,
    ) -> None:
        if max_terms <= 0 or max_length <= 0 or batch_size <= 0:
            raise ValueError("SPLADE bounds must be positive")
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForMaskedLM,
            AutoTokenizer,
        )

        self.torch = __import__("torch")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model.eval()
        self.max_terms = max_terms
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> list[Mapping[int, float]]:
        outputs: list[Mapping[int, float]] = []
        torch = self.torch
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**tokens).logits
                weights = torch.log1p(torch.relu(logits))
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (weights * mask).amax(dim=1)
            count = min(self.max_terms, int(pooled.shape[1]))
            values, indices = torch.topk(pooled, k=count, dim=1)
            for row_values, row_indices in zip(values, indices, strict=True):
                vector = {
                    int(index): float(value)
                    for index, value in zip(
                        row_indices.tolist(), row_values.tolist(), strict=True
                    )
                    if value > 0.0
                }
                outputs.append(vector)
        return outputs


class InvertedSparseIndex:
    """Compact CSR asset with an in-memory posting view for exact sparse search."""

    def __init__(
        self,
        identifiers: Sequence[str],
        indptr: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        if len(indptr) != len(identifiers) + 1:
            raise ValueError("invalid sparse row offsets")
        if len(indices) != len(values) or int(indptr[-1]) != len(indices):
            raise ValueError("invalid sparse index arrays")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sparse index identifiers must be unique")
        self.identifiers = tuple(str(item) for item in identifiers)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)
        postings: dict[int, list[tuple[int, float]]] = {}
        for row in range(len(self.identifiers)):
            for position in range(int(self.indptr[row]), int(self.indptr[row + 1])):
                token = int(self.indices[position])
                postings.setdefault(token, []).append(
                    (row, float(self.values[position]))
                )
        self.postings: dict[int, tuple[np.ndarray, np.ndarray]] = {
            token: (
                np.asarray([item[0] for item in items], dtype=np.int32),
                np.asarray([item[1] for item in items], dtype=np.float32),
            )
            for token, items in postings.items()
        }

    @classmethod
    def from_vectors(
        cls,
        identifiers: Sequence[str],
        vectors: Sequence[Mapping[int, float]],
    ) -> InvertedSparseIndex:
        if len(identifiers) != len(vectors):
            raise ValueError("identifier/vector counts differ")
        indptr = [0]
        indices: list[int] = []
        values: list[float] = []
        for vector in vectors:
            for token, value in sorted(vector.items()):
                if token < 0 or not np.isfinite(value) or value <= 0.0:
                    continue
                indices.append(int(token))
                values.append(float(value))
            indptr.append(len(indices))
        return cls(
            identifiers,
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            identifiers=np.asarray(self.identifiers, dtype=np.str_),
            indptr=self.indptr,
            indices=self.indices,
            values=self.values,
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> InvertedSparseIndex:
        with np.load(Path(path), allow_pickle=False) as value:
            return cls(
                value["identifiers"].tolist(),
                value["indptr"],
                value["indices"],
                value["values"],
            )

    def search(self, query_vector: Mapping[int, float], limit: int) -> list[str]:
        if limit <= 0:
            raise ValueError("search limit must be positive")
        scores: np.ndarray = np.zeros(len(self.identifiers), dtype=np.float32)
        touched: list[np.ndarray] = []
        for token, query_weight in query_vector.items():
            posting = self.postings.get(int(token))
            if posting is None or not np.isfinite(query_weight) or query_weight <= 0.0:
                continue
            rows, weights = posting
            scores[rows] += weights * float(query_weight)
            touched.append(rows)
        if not touched:
            return []
        candidates = np.unique(np.concatenate(touched))
        count = min(limit, len(candidates))
        selected = candidates[np.argpartition(scores[candidates], -count)[-count:]]
        ordered = sorted(
            (int(row) for row in selected),
            key=lambda row: (-float(scores[row]), self.identifiers[row]),
        )
        return [self.identifiers[row] for row in ordered]


class LearnedSparseRetriever:
    def __init__(self, encoder: SparseEncoder, index: InvertedSparseIndex) -> None:
        self.encoder = encoder
        self.index = index

    def search(self, query: str, limit: int) -> list[str]:
        if not query.strip():
            return []
        return self.index.search(self.encoder.encode([query])[0], limit)
