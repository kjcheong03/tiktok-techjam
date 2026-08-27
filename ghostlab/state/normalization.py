from __future__ import annotations

from dataclasses import dataclass

from ghostlab.state.catalog_ontology import CatalogOntology, OntologyResolution


@dataclass(frozen=True)
class NormalizedConstraint:
    attribute: str
    canonical: str
    confidence: float
    source: str


@dataclass(frozen=True)
class CatalogStateNormalizer:
    ontology: CatalogOntology
    confidence_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("normalization confidence must be between zero and one")

    def normalize(
        self, attribute: str, value: str, category: str | None
    ) -> NormalizedConstraint | None:
        resolved: OntologyResolution | None = self.ontology.resolve(
            value, attribute_hint=attribute, category=category
        )
        if resolved is None or resolved.confidence < self.confidence_threshold:
            return None
        return NormalizedConstraint(
            resolved.attribute,
            resolved.canonical,
            resolved.confidence,
            resolved.source,
        )
