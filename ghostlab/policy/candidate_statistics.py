from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ghostlab.competition.contract import AskAttribute
from ghostlab.state.catalog_ontology import ATTRIBUTE_KEYS, normalize_text

FACET_ATTRIBUTES: tuple[AskAttribute, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
)
MISSING_VALUE = "__missing__"


@dataclass(frozen=True)
class FacetDistribution:
    attribute: AskAttribute
    counts: tuple[tuple[str, int], ...]
    candidate_count: int
    covered_count: int
    entropy: float
    normalized_entropy: float
    partition_gain: float
    expected_reduction: float
    no_preference_probability: float

    @property
    def coverage(self) -> float:
        return self.covered_count / self.candidate_count if self.candidate_count else 0.0


@dataclass(frozen=True)
class CandidateStatistics:
    candidate_count: int
    facets: Mapping[AskAttribute, FacetDistribution]


def _budget_bucket(value: object) -> str | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    price = float(value)
    bounds = (25, 50, 100, 200, 500)
    lower = 0
    for upper in bounds:
        if price < upper:
            return f"${lower}-${upper}"
        lower = upper
    return "$500+"


class CandidateFacetStore:
    """Small observable catalog-facet view used only when EIG is enabled."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.values: dict[str, dict[AskAttribute, tuple[str, ...]]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = str(row["parent_asin"])
                facets: dict[AskAttribute, set[str]] = {
                    attribute: set() for attribute in FACET_ATTRIBUTES
                }
                for category in row.get("categories") or ():
                    normalized = normalize_text(str(category))
                    if normalized:
                        facets["category"].add(normalized)
                brand = normalize_text(str(row.get("store") or ""))
                if brand:
                    facets["brand"].add(brand)
                budget = _budget_bucket(row.get("price"))
                if budget:
                    facets["budget"].add(budget)
                details = row.get("details") or {}
                if isinstance(details, dict):
                    for key, raw in details.items():
                        normalized_key = normalize_text(str(key))
                        attribute = next(
                            (
                                name
                                for name, needles in ATTRIBUTE_KEYS.items()
                                if any(needle in normalized_key for needle in needles)
                            ),
                            None,
                        )
                        if attribute not in facets:
                            continue
                        normalized = normalize_text(str(raw))
                        if normalized and len(normalized) <= 120:
                            facets[attribute].add(normalized)  # type: ignore[index]
                self.values[identifier] = {
                    attribute: tuple(sorted(values))
                    for attribute, values in facets.items()
                    if values
                }

    def summarize(
        self, ranking: Iterable[str], *, limit: int
    ) -> CandidateStatistics:
        if limit <= 0:
            raise ValueError("candidate statistics limit must be positive")
        identifiers = list(dict.fromkeys(ranking))[:limit]
        count = len(identifiers)
        facets: dict[AskAttribute, FacetDistribution] = {}
        for attribute in FACET_ATTRIBUTES:
            counts: dict[str, float] = defaultdict(float)
            covered = 0
            for identifier in identifiers:
                values = self.values.get(identifier, {}).get(attribute, ())
                if values:
                    covered += 1
                    weight = 1.0 / len(values)
                    for value in values:
                        counts[value] += weight
                else:
                    counts[MISSING_VALUE] += 1
            integer_counts = tuple(
                (value, max(1, round(value_count)))
                for value, value_count in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )
            )
            probabilities = [value / count for value in counts.values()] if count else []
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
            maximum_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
            partition_gain = 1.0 - sum(p * p for p in probabilities)
            facets[attribute] = FacetDistribution(
                attribute=attribute,
                counts=integer_counts,
                candidate_count=count,
                covered_count=covered,
                entropy=entropy,
                normalized_entropy=entropy / maximum_entropy if maximum_entropy else 0.0,
                partition_gain=partition_gain,
                expected_reduction=count * partition_gain,
                no_preference_probability=1.0 - (covered / count) if count else 1.0,
            )
        return CandidateStatistics(count, facets)
