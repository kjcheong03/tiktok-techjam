from __future__ import annotations

import re
from dataclasses import dataclass, field

ASK_ORDER = ("material", "color", "style", "use_case", "feature", "budget", "size")
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
COLORS = (
    "black",
    "white",
    "blue",
    "navy",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)

CATEGORY_RE = re.compile(r"i'm looking for (.+?)(?:\.|,\s*but)", re.IGNORECASE)
CONSTRAINT_RE = re.compile(
    r"(?:a key requirement is|what matters is|what i need is):\s*(.+?)(?:\.$|$)",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"i don't have (?:an additional |a )?preference for ([a-z_]+)",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(r"\b(?:actually|ignore|instead)\b", re.IGNORECASE)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .;,\t\n")


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(color in lowered for color in COLORS) or "color" in lowered:
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(
        word in lowered for word in ("department", "style", "fit", "sleeve", "neck")
    ):
        return "style"
    if any(
        word in lowered
        for word in ("hiking", "running", "gym", "winter", "outdoor", "work")
    ):
        return "use_case"
    return "feature"


@dataclass
class SlotValue:
    attribute: str
    value: str
    source_turn: int
    source_text: str
    provenance: str
    active: bool = True


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    messages: list[str] = field(default_factory=list)
    slots: list[SlotValue] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None

    def observe(self, message: str, turn: int) -> None:
        self.messages.append(message)

        no_preference = NO_PREFERENCE_RE.search(message)
        if no_preference:
            attribute = no_preference.group(1).lower()
            self.no_preference_attributes.add(attribute)

        category = CATEGORY_RE.search(message)
        if category:
            self._replace_slot(
                "category",
                _clean(category.group(1)),
                turn,
                message,
                provenance="explicit",
            )

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

        if OVERRIDE_RE.search(message) and "what i need is" in message.lower():
            for slot in self.slots:
                if slot.attribute != "category":
                    slot.active = False

        for value in constraints:
            if not value:
                continue
            is_answer = message.lower().startswith("for that")
            attribute = (
                self.last_asked_attribute
                if is_answer and self.last_asked_attribute
                else classify_constraint(value)
            )
            provenance = "simulator_answer" if is_answer else "explicit"
            self._replace_slot(attribute, value, turn, message, provenance)

    def _replace_slot(
        self,
        attribute: str,
        value: str,
        turn: int,
        source_text: str,
        provenance: str,
    ) -> None:
        normalized = value.casefold()
        for slot in self.slots:
            if slot.attribute == attribute and slot.active:
                if slot.value.casefold() == normalized:
                    return
                slot.active = False
        self.slots.append(
            SlotValue(
                attribute=attribute,
                value=value,
                source_turn=turn,
                source_text=source_text,
                provenance=provenance,
            )
        )

    def build_query(self) -> str:
        active = [slot for slot in self.slots if slot.active]
        active.sort(key=lambda slot: (slot.attribute != "category", slot.source_turn))
        values = list(dict.fromkeys(slot.value for slot in active))
        return ". ".join(values) if values else self.messages[-1]

    def choose_question(self) -> str | None:
        known = {slot.attribute for slot in self.slots if slot.active}
        unavailable = known | self.no_preference_attributes | set(self.asked_attributes)
        for attribute in ASK_ORDER:
            if attribute not in unavailable:
                self.asked_attributes.append(attribute)
                self.last_asked_attribute = attribute
                return attribute
        self.last_asked_attribute = None
        return None


def fixed_question_for_turn(turn: int) -> str | None:
    index = turn - 1
    return ASK_ORDER[index] if 0 <= index < len(ASK_ORDER) else None
