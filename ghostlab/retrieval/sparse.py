from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from ghostlab.policy.models import RankedCandidate, RankedCandidates

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
}
FIELD_NAMES = ("title", "categories", "features", "details", "store", "description")
OFFICIAL_WEIGHTS = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def query_terms(text: str, limit: int = 40) -> list[str]:
    return list(
        dict.fromkeys(
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in STOPWORDS
        )
    )[:limit]


class SparseIndex:
    """One reusable FTS index with query-time field-weight ablations."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.connection = sqlite3.connect(":memory:")
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def search(
        self,
        query: str,
        limit: int,
        weights: tuple[float, float, float, float, float, float],
    ) -> RankedCandidates:
        started = time.perf_counter()
        terms = query_terms(query)
        expression = " OR ".join(f'"{term}"' for term in terms)
        if not expression:
            rows: list[tuple[str, float]] = []
        else:
            placeholders = ", ".join("?" for _ in range(7))
            statement = (
                "SELECT parent_asin, -bm25(products, " + placeholders + ") AS score "
                "FROM products WHERE products MATCH ? ORDER BY score DESC, parent_asin ASC LIMIT ?"
            )
            rows = self.connection.execute(
                statement, (0.0, *weights, expression, limit)
            ).fetchall()
        count = len(rows)
        items = tuple(
            RankedCandidate(
                parent_asin=str(identifier),
                route="keyword",
                rank=rank,
                raw_score=float(score),
                normalized_score=1.0
                if count == 1
                else 1.0 - (rank - 1) / max(1, count - 1),
            )
            for rank, (identifier, score) in enumerate(rows, start=1)
        )
        return RankedCandidates(
            items=items,
            route="keyword",
            requested_k=limit,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
