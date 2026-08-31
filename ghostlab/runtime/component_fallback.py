from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FallbackDecision:
    ranking: tuple[str, ...]
    used_fallback: bool
    reason: str


@dataclass(frozen=True)
class ComponentFallback:
    minimum_results: int = 10
    minimum_constraint_coverage: float = 0.8

    def __post_init__(self) -> None:
        if self.minimum_results < 1:
            raise ValueError("fallback minimum results must be positive")
        if not 0.0 <= self.minimum_constraint_coverage <= 1.0:
            raise ValueError("fallback coverage threshold must be between zero and one")

    def choose(
        self,
        candidate: Sequence[str],
        base: Sequence[str],
        *,
        candidate_constraint_coverage: Mapping[str, float] | None = None,
        base_constraint_coverage: Mapping[str, float] | None = None,
    ) -> FallbackDecision:
        unique_candidate = tuple(dict.fromkeys(candidate))
        unique_base = tuple(dict.fromkeys(base))
        if len(unique_candidate) < self.minimum_results:
            return FallbackDecision(unique_base, True, "insufficient_results")
        if candidate_constraint_coverage and base_constraint_coverage:
            for attribute, base_value in base_constraint_coverage.items():
                candidate_value = candidate_constraint_coverage.get(attribute, 0.0)
                if candidate_value < self.minimum_constraint_coverage * base_value:
                    return FallbackDecision(
                        unique_base, True, f"constraint_coverage:{attribute}"
                    )
        return FallbackDecision(unique_candidate, False, "component_accepted")
