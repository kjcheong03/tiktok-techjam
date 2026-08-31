from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostlab.retrieval.learned_sparse import (
    InvertedSparseIndex,
    LearnedSparseAsset,
    LearnedSparseRetriever,
)
from ghostlab.retrieval.sparse_semantic_fusion import sparse_semantic_union_ids


class StaticEncoder:
    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [{1: 1.0, 3: 1.0} for _ in texts]


def test_sparse_index_round_trip_and_exact_ranking(tmp_path: Path) -> None:
    index = InvertedSparseIndex.from_vectors(
        ["a", "b", "c"],
        [{1: 2.0}, {1: 0.5, 3: 2.0}, {2: 9.0}],
    )
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = InvertedSparseIndex.load(path)
    retriever = LearnedSparseRetriever(StaticEncoder(), loaded)
    assert retriever.search("semantic query", 3) == ["b", "a"]


def test_sparse_semantic_union_is_unique_and_weighted() -> None:
    result = sparse_semantic_union_ids(
        ["a", "b", "c"], ["d", "b", "e"], sparse_weight=0.8, semantic_weight=0.2
    )
    assert result[0] == "b"
    assert len(result) == len(set(result))
    with pytest.raises(ValueError, match="sum to one"):
        sparse_semantic_union_ids(["a"], ["b"], sparse_weight=0.8, semantic_weight=0.3)


def test_unavailable_manifest_fails_explicitly_without_model_import() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = LearnedSparseAsset.load(root / "configs/assets/splade_rescue_v1.json")
    assert manifest.availability == "unavailable"
    with pytest.raises(RuntimeError, match="unavailable"):
        manifest.require_available(root)


def test_manifest_rejects_wrong_technique_id(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "technique_id": "wrong",
                "availability": "unavailable",
                "model_name": "none",
                "revision": None,
                "model_path": None,
                "index_path": None,
                "index_sha256": None,
                "catalog_sha256": None,
                "max_terms": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="technique ID"):
        LearnedSparseAsset.load(path)
