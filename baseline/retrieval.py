from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import numpy as np

from starter.agent import Agent as OfficialKeywordAgent


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ". ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return ". ".join(str(item) for item in value)
    return str(value)


def catalog_document(product: dict) -> str:
    fields = (
        ("Title", product.get("title")),
        ("Category", product.get("categories")),
        ("Features", product.get("features")),
        ("Details", product.get("details")),
        ("Description", product.get("description")),
        ("Brand", product.get("store")),
    )
    return " ".join(f"{label}: {_text(value)}" for label, value in fields if _text(value))


class KeywordRetriever:
    """Adapter around the organizer's exact SQLite FTS5 BM25 implementation."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.agent = OfficialKeywordAgent(catalog_path)
        self._cache: dict[tuple[str, int], list[str]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def search(self, session_id: str, query: str, turn: int, limit: int) -> list[str]:
        cache_key = (query, limit)
        if cache_key not in self._cache:
            response = self.agent.respond(session_id, query, turn, limit)
            self._cache[cache_key] = [
                str(item["parent_asin"]) for item in response["recommendations"]
            ]
        return self._cache[cache_key]


class DenseRetriever:
    """Exact in-process cosine retrieval over cached Sentence Transformer embeddings."""

    def __init__(
        self,
        catalog_path: str | Path,
        cache_dir: str | Path = "artifacts/cache",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.catalog_path = Path(catalog_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")
        self.identifiers, documents = self._load_catalog()
        self.embeddings = self._load_or_build_embeddings(documents)
        self._ranking_cache: dict[tuple[str, int], list[str]] = {}

    def _load_catalog(self) -> tuple[list[str], list[str]]:
        identifiers: list[str] = []
        documents: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                identifiers.append(str(product["parent_asin"]))
                documents.append(catalog_document(product))
        return identifiers, documents

    def _cache_stem(self) -> str:
        digest = hashlib.sha256()
        with self.catalog_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        model_key = self.model_name.replace("/", "_")
        return f"{model_key}-{digest.hexdigest()[:12]}"

    def _load_or_build_embeddings(self, documents: Sequence[str]) -> np.ndarray:
        import numpy as np

        stem = self._cache_stem()
        embedding_path = self.cache_dir / f"{stem}.npy"
        ids_path = self.cache_dir / f"{stem}.ids.json"
        if embedding_path.exists() and ids_path.exists():
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if cached_ids == self.identifiers:
                return np.load(embedding_path, mmap_mode="r")

        embeddings = self.model.encode(
            list(documents),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        np.save(embedding_path, embeddings)
        ids_path.write_text(json.dumps(self.identifiers), encoding="utf-8")
        return np.load(embedding_path, mmap_mode="r")

    def search(self, query: str, limit: int) -> list[str]:
        import numpy as np

        query = query.strip()
        if not query:
            return []
        cache_key = (query, limit)
        if cache_key in self._ranking_cache:
            return self._ranking_cache[cache_key]

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32, copy=False)
        scores = np.asarray(self.embeddings @ query_embedding)
        count = min(limit, len(scores))
        candidate_indices = np.argpartition(scores, -count)[-count:]
        ordered_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        result = [self.identifiers[int(index)] for index in ordered_indices]
        self._ranking_cache[cache_key] = result
        return result


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]],
    *,
    rank_constant: int = 60,
    limit: int = 10,
) -> list[str]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (rank_constant + rank)
            best_rank[identifier] = min(best_rank.get(identifier, rank), rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))
    return ordered[:limit]
