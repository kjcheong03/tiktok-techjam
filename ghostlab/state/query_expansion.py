from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpansionTerm:
    value: str
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("expansion term cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("expansion confidence must be between zero and one")


@dataclass(frozen=True)
class QueryExpansion:
    original_query: str
    terms: tuple[ExpansionTerm, ...]
    reason: str

    @property
    def expanded_query(self) -> str:
        additions = " ".join(item.value for item in self.terms)
        return (
            self.original_query
            if not additions
            else f"{self.original_query} {additions}"
        )


class ExpansionGuard:
    """Keep PRF additions bounded and subordinate to explicit user evidence."""

    def __init__(self, *, max_terms: int = 4, max_added_ratio: float = 0.5) -> None:
        if max_terms < 0:
            raise ValueError("max_terms cannot be negative")
        if not 0.0 <= max_added_ratio <= 1.0:
            raise ValueError("max_added_ratio must be between zero and one")
        self.max_terms = max_terms
        self.max_added_ratio = max_added_ratio

    def apply(
        self, query: str, proposed: list[ExpansionTerm]
    ) -> tuple[ExpansionTerm, ...]:
        explicit_count = max(1, len(query.split()))
        ratio_limit = int(explicit_count * self.max_added_ratio)
        limit = min(self.max_terms, ratio_limit)
        if limit <= 0:
            return ()
        explicit = {token.casefold() for token in query.split()}
        accepted: list[ExpansionTerm] = []
        seen: set[str] = set()
        for item in sorted(
            proposed, key=lambda value: (-value.confidence, value.value)
        ):
            normalized = item.value.casefold()
            if normalized in explicit or normalized in seen:
                continue
            seen.add(normalized)
            accepted.append(item)
            if len(accepted) >= limit:
                break
        return tuple(accepted)
