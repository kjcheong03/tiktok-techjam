from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

ONTOLOGY_SCHEMA_VERSION = 1
ATTRIBUTE_KEYS: Mapping[str, tuple[str, ...]] = {
    "brand": ("brand", "manufacturer"),
    "color": ("color", "colour"),
    "material": ("material", "fabric"),
    "size": ("size", "dimensions"),
    "style": ("style", "pattern", "theme"),
}
ALIASES: Mapping[str, str] = {
    "grey": "gray",
    "navy blue": "navy",
    "extra small": "xs",
    "extra large": "xl",
    "stainless-steel": "stainless steel",
}
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w.$%+/-]+", re.UNICODE)


def normalize_text(value: str) -> str:
    """Return a stable lexical form without erasing units or numeric ranges."""

    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip(" .")
    return ALIASES.get(text, text)


@dataclass(frozen=True)
class OntologyEntry:
    attribute: str
    canonical: str
    aliases: tuple[str, ...]
    frequency: int
    category_support: tuple[str, ...] = ()


@dataclass(frozen=True)
class OntologyResolution:
    attribute: str
    canonical: str
    confidence: float
    source: str


@dataclass(frozen=True)
class CatalogOntology:
    catalog_sha256: str
    entries: tuple[OntologyEntry, ...]
    _index: dict[str, list[OntologyEntry]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        index: dict[str, list[OntologyEntry]] = defaultdict(list)
        for entry in self.entries:
            for alias in entry.aliases:
                index[normalize_text(alias)].append(entry)
        object.__setattr__(self, "_index", dict(index))

    def resolve(
        self,
        value: str,
        *,
        attribute_hint: str | None = None,
        category: str | None = None,
    ) -> OntologyResolution | None:
        normalized = normalize_text(value)
        if not normalized:
            return None
        matches = list(dict.fromkeys(self._index.get(normalized, ())))
        if attribute_hint is not None:
            hinted = [item for item in matches if item.attribute == attribute_hint]
            if hinted:
                matches = hinted
        if not matches:
            return None
        normalized_category = normalize_text(category or "")

        def score(entry: OntologyEntry) -> tuple[float, int, str, str]:
            category_match = bool(
                normalized_category
                and any(
                    normalized_category == normalize_text(item)
                    or normalized_category in normalize_text(item)
                    for item in entry.category_support
                )
            )
            confidence = 0.98 if len(matches) == 1 else 0.78
            if attribute_hint == entry.attribute:
                confidence += 0.01
            if category_match:
                confidence += 0.01
            return (-min(confidence, 0.99), -entry.frequency, entry.attribute, entry.canonical)

        selected = min(matches, key=score)
        confidence = -score(selected)[0]
        return OntologyResolution(
            attribute=selected.attribute,
            canonical=selected.canonical,
            confidence=confidence,
            source="catalog_exact",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": ONTOLOGY_SCHEMA_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @classmethod
    def from_path(cls, path: str | Path) -> CatalogOntology:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != ONTOLOGY_SCHEMA_VERSION:
            raise ValueError("unsupported catalog ontology schema")
        return cls(
            catalog_sha256=str(payload["catalog_sha256"]),
            entries=tuple(OntologyEntry(**item) for item in payload["entries"]),
        )


def _catalog_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_catalog_ontology(
    catalog_path: str | Path,
    *,
    minimum_frequency: int = 2,
    maximum_values_per_attribute: int = 20_000,
) -> CatalogOntology:
    """Build a deterministic, bounded vocabulary from catalog metadata."""

    if minimum_frequency < 1 or maximum_values_per_attribute < 1:
        raise ValueError("ontology bounds must be positive")
    path = Path(catalog_path)
    counts: dict[tuple[str, str], int] = Counter()
    categories: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    display: dict[tuple[str, str], str] = {}

    def add(attribute: str, raw: object, row_categories: Iterable[object]) -> None:
        if not isinstance(raw, str):
            return
        canonical = normalize_text(raw)
        if not canonical or len(canonical) > 120:
            return
        key = (attribute, canonical)
        counts[key] += 1
        display.setdefault(key, canonical)
        categories[key].update(
            normalize_text(str(item)) for item in row_categories if str(item).strip()
        )

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_categories = row.get("categories") or ()
            for value in row_categories:
                add("category", value, row_categories)
            add("brand", row.get("store"), row_categories)
            details = row.get("details") or {}
            if isinstance(details, dict):
                for key, value in details.items():
                    normalized_key = normalize_text(str(key))
                    attribute = next(
                        (
                            name
                            for name, needles in ATTRIBUTE_KEYS.items()
                            if any(needle in normalized_key for needle in needles)
                        ),
                        None,
                    )
                    if attribute is not None:
                        add(attribute, value, row_categories)

    entries: list[OntologyEntry] = []
    per_attribute: Counter[str] = Counter()
    for (attribute, canonical), frequency in sorted(
        counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
    ):
        if frequency < minimum_frequency:
            continue
        if per_attribute[attribute] >= maximum_values_per_attribute:
            continue
        per_attribute[attribute] += 1
        aliases = {canonical}
        aliases.update(alias for alias, target in ALIASES.items() if target == canonical)
        entries.append(
            OntologyEntry(
                attribute=attribute,
                canonical=display[(attribute, canonical)],
                aliases=tuple(sorted(aliases)),
                frequency=frequency,
                category_support=tuple(
                    key for key, _ in categories[(attribute, canonical)].most_common(8)
                ),
            )
        )
    return CatalogOntology(_catalog_hash(path), tuple(entries))
