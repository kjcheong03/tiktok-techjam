from __future__ import annotations

from dataclasses import dataclass, field

from ghostlab.state.catalog_ontology import CatalogOntology, OntologyResolution
from ghostlab.state.memory import ConversationState, Polarity, Provenance


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


@dataclass
class NormalizedConversationState(ConversationState):
    """Opt-in state adapter that leaves the historical state implementation intact."""

    catalog_normalizer: CatalogStateNormalizer | None = None
    normalization_trace: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.catalog_normalizer is None:
            raise ValueError("normalized conversation state requires a normalizer")

    def _add(
        self,
        attribute: str,
        value: str,
        turn: int,
        source_text: str,
        provenance: Provenance,
        *,
        polarity: Polarity = "positive",
        replace: bool = True,
        replace_reason: str = "replacement",
    ) -> None:
        assert self.catalog_normalizer is not None
        resolution = self.catalog_normalizer.normalize(
            attribute, value, self.active_category
        )
        if resolution is None:
            super()._add(
                attribute,
                value,
                turn,
                source_text,
                provenance,
                polarity=polarity,
                replace=replace,
                replace_reason=replace_reason,
            )
            return
        super()._add(
            resolution.attribute,
            resolution.canonical,
            turn,
            source_text,
            provenance,
            polarity=polarity,
            replace=replace,
            replace_reason=replace_reason,
        )
        stored = self.values[-1]
        stored.value = value
        self.normalization_trace.append(
            {
                "turn": turn,
                "attribute": resolution.attribute,
                "raw": value,
                "canonical": resolution.canonical,
                "confidence": resolution.confidence,
                "source": resolution.source,
            }
        )
