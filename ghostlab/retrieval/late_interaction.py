from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TokenEncoder(Protocol):
    def encode_tokens(self, texts: Sequence[str]) -> list[np.ndarray]: ...


def maxsim_score(query: np.ndarray, document: np.ndarray) -> float:
    """ColBERT-style sum of per-query-token maximum similarities."""

    if query.ndim != 2 or document.ndim != 2:
        raise ValueError("token embeddings must be matrices")
    if query.shape[1] != document.shape[1]:
        raise ValueError("query and document dimensions differ")
    if not len(query) or not len(document):
        return 0.0
    return float(np.max(query @ document.T, axis=1).sum())


@dataclass(frozen=True)
class LateInteractionFeasibility:
    document_count: int
    average_document_tokens: int
    embedding_dimension: int
    dtype_bytes: int
    query_tokens: int

    @property
    def index_gib(self) -> float:
        total = (
            self.document_count
            * self.average_document_tokens
            * self.embedding_dimension
            * self.dtype_bytes
        )
        return total / (1024**3)

    @property
    def dot_products_per_query(self) -> int:
        return self.document_count * self.average_document_tokens * self.query_tokens

    def passes(self, *, max_index_gib: float, max_dot_products_per_query: int) -> bool:
        return (
            self.index_gib <= max_index_gib
            and self.dot_products_per_query <= max_dot_products_per_query
        )


class TransformerTokenEncoder:
    """Local-only token encoder for a bounded late-interaction feasibility spike."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_length: int = 128,
        batch_size: int = 8,
    ) -> None:
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("encoder bounds must be positive")
        from transformers import (  # type: ignore[import-not-found]
            AutoModel,
            AutoTokenizer,
        )

        self.torch = __import__("torch")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def encode_tokens(self, texts: Sequence[str]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
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
                hidden = self.model(**tokens).last_hidden_state
                hidden = torch.nn.functional.normalize(hidden, dim=-1)
            masks = tokens["attention_mask"].bool()
            for row, mask in zip(hidden, masks, strict=True):
                outputs.append(row[mask].cpu().numpy().astype(np.float32, copy=False))
        return outputs


class TokenEmbeddingStore:
    """Pickle-free ragged token matrix store for reference retrieval and parity."""

    def __init__(
        self,
        identifiers: Sequence[str],
        offsets: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        if len(offsets) != len(identifiers) + 1:
            raise ValueError("invalid token offsets")
        if embeddings.ndim != 2 or int(offsets[-1]) != len(embeddings):
            raise ValueError("invalid token embedding matrix")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("token store identifiers must be unique")
        self.identifiers = tuple(str(item) for item in identifiers)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.embeddings = np.asarray(embeddings)

    @classmethod
    def from_documents(
        cls, identifiers: Sequence[str], documents: Sequence[np.ndarray]
    ) -> TokenEmbeddingStore:
        if len(identifiers) != len(documents):
            raise ValueError("identifier/document counts differ")
        dimensions = {item.shape[1] for item in documents if item.ndim == 2}
        if len(dimensions) != 1 or any(item.ndim != 2 for item in documents):
            raise ValueError("document matrices must share one dimension")
        offsets = [0]
        rows = []
        for document in documents:
            rows.append(np.asarray(document, dtype=np.float32))
            offsets.append(offsets[-1] + len(document))
        dimension = dimensions.pop()
        embeddings = (
            np.concatenate(rows, axis=0)
            if rows
            else np.empty((0, dimension), dtype=np.float32)
        )
        return cls(identifiers, np.asarray(offsets, dtype=np.int64), embeddings)

    def document(self, row: int) -> np.ndarray:
        return self.embeddings[self.offsets[row] : self.offsets[row + 1]]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            identifiers=np.asarray(self.identifiers, dtype=np.str_),
            offsets=self.offsets,
            embeddings=self.embeddings.astype(np.float16),
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> TokenEmbeddingStore:
        with np.load(Path(path), allow_pickle=False) as value:
            return cls(
                value["identifiers"].tolist(),
                value["offsets"],
                value["embeddings"],
            )


class LateInteractionRetriever:
    """Exact bounded reference retrieval; production use requires a measured gate."""

    def __init__(
        self,
        encoder: TokenEncoder,
        store: TokenEmbeddingStore,
        *,
        maximum_documents: int = 50_000,
    ) -> None:
        if len(store.identifiers) > maximum_documents:
            raise ValueError("late-interaction store exceeds the configured bound")
        self.encoder = encoder
        self.store = store

    def search(self, query: str, limit: int) -> list[str]:
        if limit <= 0:
            raise ValueError("search limit must be positive")
        if not query.strip():
            return []
        query_tokens = self.encoder.encode_tokens([query])[0]
        scored = [
            (maxsim_score(query_tokens, self.store.document(row)), identifier)
            for row, identifier in enumerate(self.store.identifiers)
        ]
        return [
            identifier
            for _, identifier in sorted(scored, key=lambda item: (-item[0], item[1]))[
                :limit
            ]
        ]


class LateInteractionReranker:
    """Bounded reference MaxSim reranker over locally stored product token vectors."""

    def __init__(
        self,
        encoder: TokenEncoder,
        documents: Mapping[str, np.ndarray],
    ) -> None:
        self.encoder = encoder
        self.documents = documents

    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]:
        if rerank_k <= 0:
            raise ValueError("rerank_k must be positive")
        head = ranking[:rerank_k]
        if not head or not query.strip():
            return list(ranking)
        query_tokens = self.encoder.encode_tokens([query])[0]
        scored = []
        for rank, identifier in enumerate(head):
            tokens = self.documents.get(identifier)
            score = maxsim_score(query_tokens, tokens) if tokens is not None else -1e9
            scored.append((identifier, score, rank))
        ordered = [
            identifier
            for identifier, _, _ in sorted(scored, key=lambda item: (-item[1], item[2]))
        ]
        return [*ordered, *ranking[rerank_k:]]


def load_feasibility_manifest(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported late-interaction asset schema")
    if value.get("technique_id") not in {
        "retrieval.colbert_rescue.v1",
        "retrieval.bge_m3_rescue.v1",
    }:
        raise ValueError("unexpected late-interaction technique ID")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
