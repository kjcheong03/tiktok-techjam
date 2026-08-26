from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from baseline.retrieval import catalog_document
from ghostlab.policy.models import RankedCandidate, RankedCandidates

DOCUMENT_FORMAT_VERSION = "labeled_catalog_fields_v1"


@dataclass(frozen=True)
class DenseModelSpec:
    key: str
    model_name: str
    revision: str
    query_prefix: str = ""
    passage_prefix: str = ""
    embedding_dimension: int = 384

    def format_query(self, query: str) -> str:
        return f"{self.query_prefix}{query.strip()}"

    def format_document(self, product: dict) -> str:
        return f"{self.passage_prefix}{catalog_document(product)}"

    def canonical_hash(self) -> str:
        payload = {
            **asdict(self),
            "document_format_version": DOCUMENT_FORMAT_VERSION,
            "normalized": True,
            "dtype": "float32",
        }
        value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


MINILM_CONTROL = DenseModelSpec(
    key="minilm_control",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)
E5_SMALL_V2 = DenseModelSpec(
    key="e5_small_v2",
    model_name="intfloat/e5-small-v2",
    revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
    query_prefix="query: ",
    passage_prefix="passage: ",
)
MODEL_SPECS = {spec.key: spec for spec in (MINILM_CONTROL, E5_SMALL_V2)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_biased_overlap(
    left: Sequence[str], right: Sequence[str], *, limit: int, persistence: float = 0.9
) -> float:
    if limit <= 0 or not 0.0 < persistence < 1.0:
        raise ValueError("invalid rank-biased-overlap parameters")
    depth = min(limit, len(left), len(right))
    if depth == 0:
        return 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    weighted = 0.0
    agreement = 0.0
    for index in range(depth):
        left_seen.add(left[index])
        right_seen.add(right[index])
        agreement = len(left_seen & right_seen) / (index + 1)
        weighted += agreement * persistence**index
    return (1.0 - persistence) * weighted + agreement * persistence**depth


class DenseIndex:
    """Exact offline cosine index with a content-addressed embedding cache."""

    def __init__(
        self,
        catalog_path: str | Path,
        spec: DenseModelSpec,
        *,
        cache_dir: str | Path = "artifacts/cache/dense",
        model_path: str | Path | None = None,
        batch_size: int = 128,
        local_files_only: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        self.catalog_path = Path(catalog_path)
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        source = str(model_path) if model_path is not None else spec.model_name
        kwargs: dict[str, object] = {
            "device": "cpu",
            "local_files_only": local_files_only,
        }
        if model_path is None:
            kwargs["revision"] = spec.revision
        self.model = SentenceTransformer(source, **kwargs)
        self.model_load_seconds = time.perf_counter() - started
        dimension = self.model.get_embedding_dimension()
        if dimension != spec.embedding_dimension:
            raise ValueError(
                f"{spec.key} embedding dimension {dimension} != {spec.embedding_dimension}"
            )
        self.identifiers, documents = self._load_catalog()
        self.catalog_sha256 = sha256_file(self.catalog_path)
        self.embeddings, self.cache_metadata = self._load_or_build_embeddings(documents)

    def _load_catalog(self) -> tuple[list[str], list[str]]:
        identifiers: list[str] = []
        documents: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                identifiers.append(str(product["parent_asin"]))
                documents.append(self.spec.format_document(product))
        return identifiers, documents

    def _expected_cache_metadata(self) -> dict[str, object]:
        identifiers_hash = hashlib.sha256(
            "\n".join(self.identifiers).encode()
        ).hexdigest()
        return {
            "schema_version": 1,
            "model": asdict(self.spec),
            "model_spec_sha256": self.spec.canonical_hash(),
            "document_format_version": DOCUMENT_FORMAT_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "identifiers_sha256": identifiers_hash,
            "row_count": len(self.identifiers),
            "embedding_dimension": self.spec.embedding_dimension,
            "dtype": "float32",
            "normalized": True,
        }

    def _cache_paths(self) -> tuple[Path, Path]:
        stem = (
            f"{self.spec.key}-{self.catalog_sha256[:12]}-"
            f"{self.spec.canonical_hash()[:12]}"
        )
        return self.cache_dir / f"{stem}.npy", self.cache_dir / f"{stem}.json"

    def _load_or_build_embeddings(
        self, documents: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, object]]:
        embedding_path, metadata_path = self._cache_paths()
        expected = self._expected_cache_metadata()
        started = time.perf_counter()
        if embedding_path.exists() and metadata_path.exists():
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in expected.items()):
                embeddings = np.load(embedding_path, mmap_mode="r")
                if embeddings.shape == (
                    len(self.identifiers),
                    self.spec.embedding_dimension,
                ):
                    return embeddings, {
                        **expected,
                        "cache_hit": True,
                        "build_seconds": cached.get("build_seconds"),
                        "build_peak_process_memory_mb": cached.get(
                            "build_peak_process_memory_mb"
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                        "embedding_path": str(embedding_path),
                        "metadata_path": str(metadata_path),
                    }

        embeddings = self.model.encode(
            list(documents),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        temporary_embedding = embedding_path.with_suffix(".tmp.npy")
        temporary_metadata = metadata_path.with_suffix(".tmp.json")
        np.save(temporary_embedding, embeddings)
        stored = {
            **expected,
            "build_seconds": round(time.perf_counter() - started, 6),
            "build_peak_process_memory_mb": None,
        }
        temporary_metadata.write_text(
            json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_embedding, embedding_path)
        os.replace(temporary_metadata, metadata_path)
        return np.load(embedding_path, mmap_mode="r"), {
            **stored,
            "cache_hit": False,
            "elapsed_seconds": time.perf_counter() - started,
            "embedding_path": str(embedding_path),
            "metadata_path": str(metadata_path),
        }

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        count = min(limit, len(scores))
        if count <= 0:
            return np.asarray([], dtype=np.int64)
        candidates = np.argpartition(scores, -count)[-count:]
        return candidates[np.argsort(scores[candidates], kind="stable")[::-1]]

    def search(self, query: str, limit: int) -> RankedCandidates:
        started = time.perf_counter()
        if not query.strip():
            return RankedCandidates(
                items=(), route="dense", requested_k=limit, elapsed_ms=0.0
            )
        query_embedding = self.model.encode(
            [self.spec.format_query(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        scores = np.asarray(self.embeddings @ query_embedding)
        ordered = self._top_indices(scores, limit)
        items = tuple(
            RankedCandidate(
                parent_asin=self.identifiers[int(index)],
                route="dense",
                rank=rank,
                raw_score=float(scores[index]),
                normalized_score=max(0.0, min(1.0, float((scores[index] + 1.0) / 2.0))),
            )
            for rank, index in enumerate(ordered, start=1)
        )
        return RankedCandidates(
            items=items,
            route="dense",
            requested_k=limit,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def search_many(self, queries: Sequence[str], limit: int) -> list[list[str]]:
        formatted = [self.spec.format_query(query) for query in queries]
        query_embeddings = self.model.encode(
            formatted,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
        rankings: list[list[str]] = []
        for query_embedding in query_embeddings:
            scores = np.asarray(self.embeddings @ query_embedding)
            ordered = self._top_indices(scores, limit)
            rankings.append([self.identifiers[int(index)] for index in ordered])
        return rankings
