from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from baseline.state import (
    ASK_ORDER,
    CATEGORY_RE,
    CONSTRAINT_RE,
    NO_PREFERENCE_RE,
    OVERRIDE_RE,
    _clean,
    classify_constraint,
)

Provenance = Literal["explicit", "simulator_answer", "profile_prior"]
Polarity = Literal["positive", "negative"]
EXCLUSIVE_ATTRIBUTES = {"category", "color", "size", "budget", "brand"}
CATEGORY_SCOPED_ATTRIBUTES = {
    "material",
    "color",
    "size",
    "style",
    "brand",
    "feature",
    "use_case",
}
NEGATION_RE = re.compile(r"\b(?:not|avoid|without|exclude)\s+([^.;,]+)", re.IGNORECASE)
ALT_CATEGORY_RE = re.compile(
    r"\b(?:need|want|looking for)\s+(?:a |an |some )?(.+?)\s+(?:instead|now)(?:[.,]|$)",
    re.IGNORECASE,
)
GLOBAL_RESET_RE = re.compile(
    r"\b(?:start over|ignore (?:everything|all (?:my )?previous requirements?))\b",
    re.IGNORECASE,
)
EARLIER_PREFERENCE_RESET_RE = re.compile(
    r"\bignore my earlier preference\b", re.IGNORECASE
)


@dataclass
class MemoryValue:
    attribute: str
    value: str
    normalized: str
    source_turn: int
    source_text: str
    provenance: Provenance
    polarity: Polarity = "positive"
    category_scope: str | None = None
    active: bool = True
    invalidated_reason: str | None = None


@dataclass
class ConversationState:
    session_id: str
    user_profile: dict
    multi_value: bool = True
    negative_evidence: bool = True
    provenance_enabled: bool = True
    override_invalidation: bool = True
    messages: list[str] = field(default_factory=list)
    values: list[MemoryValue] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None

    def observe(self, message: str, turn: int) -> None:
        self.messages.append(message)
        no_preference = NO_PREFERENCE_RE.search(message)
        if no_preference and self.negative_evidence:
            self.no_preference_attributes.add(no_preference.group(1).lower())

        category = CATEGORY_RE.search(message) or ALT_CATEGORY_RE.search(message)
        category_value = _clean(category.group(1)) if category else None
        current_category = self.active_category

        constraint_match = CONSTRAINT_RE.search(message)
        constraints: list[str] = []
        if constraint_match:
            constraints = [
                _clean(value) for value in constraint_match.group(1).split(";")
            ]
        elif category and "." in message and "still exploring" not in message.lower():
            remainder = _clean(message.split(".", 1)[1])
            if remainder and "key requirement" not in remainder.lower():
                constraints = [remainder]

        if self.override_invalidation and GLOBAL_RESET_RE.search(message):
            self._invalidate_all("global_override")
            current_category = None
        elif (
            self.override_invalidation
            and constraints
            and EARLIER_PREFERENCE_RESET_RE.search(message)
        ):
            self._invalidate_non_category("earlier_preference_override")

        if (
            category_value
            and category_value.casefold() != (current_category or "").casefold()
        ):
            if current_category and self.override_invalidation:
                self._invalidate_category_scope(current_category)
            self._add("category", category_value, turn, message, "explicit")

        negative_values = []
        if self.negative_evidence:
            negative_values = [_clean(value) for value in NEGATION_RE.findall(message)]
            for value in negative_values:
                if value:
                    attribute = classify_constraint(value)
                    self._invalidate_matching(attribute, value)
                    self._add(
                        attribute, value, turn, message, "explicit", polarity="negative"
                    )

        explicit_override = bool(OVERRIDE_RE.search(message))
        for value in constraints:
            if not value or any(
                value.casefold() in item.casefold() for item in negative_values
            ):
                continue
            is_answer = message.lower().startswith("for that")
            attribute = (
                self.last_asked_attribute
                if is_answer and self.last_asked_attribute
                else classify_constraint(value)
            )
            provenance: Provenance = "simulator_answer" if is_answer else "explicit"
            replace = (
                explicit_override
                or attribute in EXCLUSIVE_ATTRIBUTES
                or not self.multi_value
            )
            self._add(
                attribute,
                value,
                turn,
                message,
                provenance,
                replace=replace,
                replace_reason=(
                    "override_replacement" if explicit_override else "replacement"
                ),
            )

    @property
    def active_category(self) -> str | None:
        categories = [
            value.value
            for value in self.values
            if value.active
            and value.attribute == "category"
            and value.polarity == "positive"
        ]
        return categories[-1] if categories else None

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
        normalized = value.casefold()
        for item in self.values:
            if (
                item.active
                and item.attribute == attribute
                and item.normalized == normalized
                and item.polarity == polarity
            ):
                return
            if item.active and item.attribute == attribute and replace:
                item.active = False
                item.invalidated_reason = replace_reason
        scope = (
            self.active_category if attribute in CATEGORY_SCOPED_ATTRIBUTES else None
        )
        self.values.append(
            MemoryValue(
                attribute=attribute,
                value=value,
                normalized=normalized,
                source_turn=turn,
                source_text=source_text,
                provenance=provenance if self.provenance_enabled else "explicit",
                polarity=polarity,
                category_scope=scope,
            )
        )

    def _invalidate_matching(self, attribute: str, value: str) -> None:
        normalized = value.casefold()
        for item in self.values:
            if (
                item.active
                and item.attribute == attribute
                and item.normalized in normalized
            ):
                item.active = False
                item.invalidated_reason = "negative_evidence"

    def _invalidate_category_scope(self, category: str) -> None:
        for item in self.values:
            if item.active and (
                item.attribute == "category" or item.category_scope == category
            ):
                item.active = False
                item.invalidated_reason = "category_override"

    def _invalidate_all(self, reason: str) -> None:
        for item in self.values:
            if item.active:
                item.active = False
                item.invalidated_reason = reason

    def _invalidate_non_category(self, reason: str) -> None:
        for item in self.values:
            if item.active and item.attribute != "category":
                item.active = False
                item.invalidated_reason = reason

    def active_values(self, polarity: Polarity = "positive") -> list[MemoryValue]:
        return [
            item for item in self.values if item.active and item.polarity == polarity
        ]

    def build_query(self, compressed: bool = False) -> str:
        active = sorted(
            self.active_values(),
            key=lambda item: (
                item.attribute != "category",
                item.source_turn,
                item.normalized,
            ),
        )
        values = list(dict.fromkeys(item.value for item in active))
        if compressed:
            values = values[:8]
        return ". ".join(values) if values else self.messages[-1]

    def choose_question(
        self, order: tuple[str, ...] = ASK_ORDER, *, allow_other: bool = False
    ) -> str | None:
        known = {item.attribute for item in self.active_values()}
        unavailable = known | self.no_preference_attributes | set(self.asked_attributes)
        candidates = (*order, "other") if allow_other else order
        for attribute in candidates:
            if attribute not in unavailable:
                self.asked_attributes.append(attribute)
                self.last_asked_attribute = attribute
                return attribute
        self.last_asked_attribute = None
        return None

    def observable_features(self, turn: int) -> dict[str, object]:
        counts = {
            attribute: 0 for attribute in (*ASK_ORDER, "category", "brand", "other")
        }
        for item in self.active_values():
            counts[item.attribute] = counts.get(item.attribute, 0) + 1
        features: dict[str, object] = {
            "turn": turn,
            "message_count": len(self.messages),
            "active_slot_total": sum(counts.values()),
            "negative_count": len(self.active_values("negative")),
            "asked_count": len(self.asked_attributes),
        }
        features.update(
            {f"active_slot_counts.{key}": value for key, value in counts.items()}
        )
        return features
