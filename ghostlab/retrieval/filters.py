from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

BUDGET_RE = re.compile(r"(?:\$|under\s+|<=?\s*)(\d+(?:\.\d+)?)", re.IGNORECASE)
ATTRIBUTE_KEYS = {
    "color": ("Color",),
    "material": ("Material", "Fabric Type"),
    "size": ("Size",),
    "style": ("Style",),
    "brand": ("Brand", "Brand Name"),
}


@dataclass(frozen=True)
class ProductAttributes:
    price: float | None
    values: dict[str, frozenset[str]]


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
                self.products[str(product["parent_asin"])] = ProductAttributes(
                    numeric_price, values
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
