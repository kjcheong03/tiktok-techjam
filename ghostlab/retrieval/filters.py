from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ghostlab.state.v2_view import ConstraintView, V2StateView

BUDGET_RE = re.compile(r"(?:\$|under\s+|<=?\s*)(\d+(?:\.\d+)?)", re.IGNORECASE)
ATTRIBUTE_KEYS = {
    "color": ("Color",),
    "material": ("Material", "Fabric Type"),
    "size": ("Size",),
    "style": ("Style",),
    "brand": ("Brand", "Brand Name"),
}
ConstraintStatus = Literal[
    "CONFIRMED_MATCH",
    "CONFIRMED_VIOLATION",
    "UNKNOWN_METADATA",
    "SOFT_PREFERENCE",
]


@dataclass(frozen=True)
class ProductAttributes:
    price: float | None
    values: dict[str, frozenset[str]]
    document_terms: frozenset[str]


@dataclass(frozen=True)
class ConstraintDecision:
    parent_asin: str
    attribute: str
    values: tuple[str, ...]
    status: ConstraintStatus
    provenance: str
    reason: str


@dataclass(frozen=True)
class ConstraintAuthorityResult:
    ranking: tuple[str, ...]
    decisions: tuple[ConstraintDecision, ...]
    confirmed_match_count: dict[str, int]
    unknown_count: dict[str, int]
    soft_preference_count: dict[str, int]
    violation_count: int

    def counts(self) -> dict[str, int]:
        return {
            "confirmed_matches": sum(self.confirmed_match_count.values()),
            "confirmed_violations": self.violation_count,
            "unknown_metadata": sum(self.unknown_count.values()),
            "soft_preferences": sum(self.soft_preference_count.values()),
        }


def _tokens(value: object) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", str(value).casefold()))


class CoverageAwareFilter:
    """Fail-open hard filtering: absent catalog metadata never means mismatch."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.products: dict[str, ProductAttributes] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                details = product.get("details") or {}
                price = product.get("price")
                numeric_price = (
                    float(price) if isinstance(price, (int, float)) else None
                )
                values = {
                    attribute: frozenset().union(
                        *(_tokens(details[key]) for key in keys if details.get(key))
                    )
                    for attribute, keys in ATTRIBUTE_KEYS.items()
                }
                document_terms = _tokens(
                    " ".join(
                        str(value or "")
                        for value in (
                            product.get("title"),
                            product.get("description"),
                            product.get("features"),
                            product.get("categories"),
                            details,
                        )
                    )
                )
                self.products[str(product["parent_asin"])] = ProductAttributes(
                    numeric_price, values, document_terms
                )

    @staticmethod
    def _budget_max(values: tuple[str, ...]) -> float | None:
        matches = [BUDGET_RE.search(value) for value in values]
        return next((float(match.group(1)) for match in matches if match), None)

    @staticmethod
    def _is_authoritative(constraint: ConstraintView) -> bool:
        return (
            constraint.polarity == "exclude"
            or constraint.attribute == "budget"
            or constraint.strength == "hard"
            or (
                constraint.provenance in {"explicit", "simulator_answer"}
                and constraint.attribute in ATTRIBUTE_KEYS
            )
        )

    def _assess_constraint(
        self, identifier: str, constraint: ConstraintView
    ) -> ConstraintDecision:
        product = self.products.get(identifier)
        values = tuple(constraint.values)
        def decision(status: ConstraintStatus, reason: str) -> ConstraintDecision:
            return ConstraintDecision(
                parent_asin=identifier,
                attribute=constraint.attribute,
                values=values,
                provenance=constraint.provenance,
                status=status,
                reason=reason,
            )
        if not self._is_authoritative(constraint):
            return decision("SOFT_PREFERENCE", "non_authoritative_preference")
        if product is None:
            return decision("UNKNOWN_METADATA", "product_missing")
        if constraint.attribute == "budget":
            maximum = self._budget_max(values)
            if maximum is None or product.price is None:
                return decision("UNKNOWN_METADATA", "budget_or_price_unknown")
            violates = (
                product.price > maximum
                if constraint.polarity == "include"
                else product.price <= maximum
            )
            return decision(
                "CONFIRMED_VIOLATION" if violates else "CONFIRMED_MATCH",
                "known_price_comparison",
            )
        wanted = frozenset().union(*(_tokens(value) for value in values))
        known = product.values.get(constraint.attribute, frozenset())
        if not known and constraint.attribute in {"feature", "use_case", "other"}:
            known = product.document_terms
        if not known or not wanted:
            return decision("UNKNOWN_METADATA", "attribute_unknown")
        overlap = bool(known & wanted)
        violates = overlap if constraint.polarity == "exclude" else not overlap
        return decision(
            "CONFIRMED_VIOLATION" if violates else "CONFIRMED_MATCH",
            "catalog_attribute_comparison",
        )

    def enforce(
        self, ranking: list[str], context: V2StateView
    ) -> ConstraintAuthorityResult:
        """Remove only confirmed violations and place known matches before unknowns."""

        decisions: list[ConstraintDecision] = []
        confirmed: list[str] = []
        unknown: list[str] = []
        soft: list[str] = []
        match_counts: dict[str, int] = {}
        unknown_counts: dict[str, int] = {}
        soft_counts: dict[str, int] = {}
        violations = 0
        for identifier in ranking:
            candidate = [
                self._assess_constraint(identifier, constraint)
                for constraint in context.active_constraints
                if constraint.attribute != "category"
            ]
            decisions.extend(candidate)
            if any(item.status == "CONFIRMED_VIOLATION" for item in candidate):
                violations += 1
                continue
            match_count = sum(item.status == "CONFIRMED_MATCH" for item in candidate)
            unknown_count = sum(item.status == "UNKNOWN_METADATA" for item in candidate)
            soft_count = sum(item.status == "SOFT_PREFERENCE" for item in candidate)
            match_counts[identifier] = match_count
            unknown_counts[identifier] = unknown_count
            soft_counts[identifier] = soft_count
            if unknown_count:
                unknown.append(identifier)
            elif soft_count and not match_count:
                soft.append(identifier)
            else:
                confirmed.append(identifier)
        return ConstraintAuthorityResult(
            ranking=(*confirmed, *soft, *unknown),
            decisions=tuple(decisions),
            confirmed_match_count=match_counts,
            unknown_count=unknown_counts,
            soft_preference_count=soft_counts,
            violation_count=violations,
        )

    def apply(
        self,
        ranking: list[str],
        positive_constraints: dict[str, list[str]],
        *,
        minimum_results: int = 10,
    ) -> list[str]:
        budget_values = positive_constraints.get("budget", [])
        budget_matches = [BUDGET_RE.search(value) for value in budget_values]
        budget_max = next(
            (float(match.group(1)) for match in budget_matches if match), None
        )
        desired = {
            attribute: frozenset().union(*(_tokens(value) for value in values))
            for attribute, values in positive_constraints.items()
            if attribute in ATTRIBUTE_KEYS and values
        }

        def compatible(identifier: str) -> bool:
            product = self.products.get(identifier)
            if product is None:
                return False
            if (
                budget_max is not None
                and product.price is not None
                and product.price > budget_max
            ):
                return False
            for attribute, wanted in desired.items():
                known = product.values.get(attribute, frozenset())
                if known and wanted and not known.intersection(wanted):
                    return False
            return True

        filtered = [identifier for identifier in ranking if compatible(identifier)]
        return filtered if len(filtered) >= minimum_results else ranking

    def apply_strict(
        self,
        ranking: list[str],
        positive_constraints: dict[str, list[str]],
        negative_constraints: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Never return known violations; rank missing metadata after matches."""
        negative_constraints = negative_constraints or {}
        budget_values = positive_constraints.get("budget", [])
        budget_matches = [BUDGET_RE.search(value) for value in budget_values]
        budget_max = next(
            (float(match.group(1)) for match in budget_matches if match), None
        )
        desired = {
            attribute: frozenset().union(*(_tokens(value) for value in values))
            for attribute, values in positive_constraints.items()
            if attribute in ATTRIBUTE_KEYS and values
        }
        excluded = {
            attribute: frozenset().union(*(_tokens(value) for value in values))
            for attribute, values in negative_constraints.items()
            if attribute in ATTRIBUTE_KEYS and values
        }

        confirmed: list[str] = []
        unknown: list[str] = []
        for identifier in ranking:
            product = self.products.get(identifier)
            if product is None:
                unknown.append(identifier)
                continue
            violation = False
            incomplete = False
            if budget_max is not None:
                if product.price is None:
                    incomplete = True
                elif product.price > budget_max:
                    violation = True
            for attribute, wanted in desired.items():
                known = product.values.get(attribute, frozenset())
                if not known:
                    incomplete = True
                elif wanted and not known.intersection(wanted):
                    violation = True
            for attribute, forbidden in excluded.items():
                known = product.values.get(attribute, frozenset())
                if not known:
                    incomplete = True
                elif forbidden and known.intersection(forbidden):
                    violation = True
            if violation:
                continue
            (unknown if incomplete else confirmed).append(identifier)
        return [*confirmed, *unknown]
