"""Small deterministic conversation parser for State Baseline V2."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constraints import StructuredConstraint, normalize_value
from .state import COLORS, MATERIALS


@dataclass(frozen=True)
class ParseResult:
    constraints: tuple[StructuredConstraint, ...] = ()
    no_preference_attributes: frozenset[str] = frozenset()
    correction_attributes: frozenset[str] = frozenset()


_ATTRIBUTE_ALIASES = {
    "category": "category", "material": "material", "color": "color", "colour": "color",
    "size": "size", "sizing": "size", "style": "style", "fit": "style", "brand": "brand",
    "budget": "budget", "price": "budget", "feature": "feature", "features": "feature",
    "use_case": "use_case", "use case": "use_case", "usage": "use_case", "other": "other",
}
_VALUE_TO_ATTRIBUTE = {
    **{value.casefold(): "material" for value in MATERIALS},
    **{value.casefold(): "color" for value in COLORS},
    **dict.fromkeys(("size", "sizing", "wide", "narrow"), "size"),
    **dict.fromkeys(("style", "fit", "sleeve", "neck"), "style"),
    **dict.fromkeys(("hiking", "running", "gym", "winter", "outdoor", "work"), "use_case"),
    **dict.fromkeys(("waterproof", "lightweight", "breathable", "durable"), "feature"),
}
_VALUE_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(sorted(map(re.escape, _VALUE_TO_ATTRIBUTE), key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    r"\b(?:i['’]?m|i\s+am|im|we['’]?re|we\s+are)?\s*(?:looking|searching|shopping)\s+for\s+([^.!?;]+)", re.I,
)
_NO_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+)?(?:do\s*n't|do\s+not|dont)\s+have\s+(?:an?\s+)?(?:additional\s+)?preference\s+for\s+([a-z_]+)", re.I,
)
_CORRECTION_RE = re.compile(r"\b(?:actually|instead|ignore|replace|change|switch)\b", re.I)
_NEGATION_RE = re.compile(r"(?:\b(?:not|avoid|exclude|excluding|without)\b|\banything\s+but\b|\b(?:do\s+not|don't|dont)\s+(?:want|like)\b)", re.I)
_HARD_RE = re.compile(r"\b(?:must|required|mandatory|essential|need|needs|have to|has to)\b|\bkey\s+requirement\b", re.I)
_SOFT_RE = re.compile(r"\b(?:prefer|preferred|would be nice|nice to have|would like)\b", re.I)
_ANSWER_RE = re.compile(r"^\s*for\s+(?:that|this)\b.*?what\s+matters\s+is\s*:\s*(.+?)\s*[.!?]?\s*$", re.I)
_MAX_RE = re.compile(r"\b(?:under|below|less\s+than|at\s+most|up\s+to|no\s+more\s+than|max(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_MIN_RE = re.compile(r"\b(?:over|above|more\s+than|at\s+least|no\s+less\s+than|min(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_EQUAL_RE = re.compile(
    r"\b(?:exactly\s+)?(?:budget|price|cost)\s*"
    r"(?:is|of|=|:|around|about)?\s*\$?\s*(\d+(?:\.\d+)?)"
    r"|\b(?:exactly\s+)?\$(\d+(?:\.\d+)?)",
    re.I,
)


def _attribute(value: str) -> str | None:
    return _ATTRIBUTE_ALIASES.get(normalize_value(value).replace("-", " "))


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .;,:!?\t\n")


def _vague(value: str) -> bool:
    value = normalize_value(value)
    return value in {"something", "anything", "whatever", "nothing", "another", "different", "something else", "anything else", "not sure"}


def _focus(message: str) -> tuple[bool, str]:
    correcting = bool(_CORRECTION_RE.search(message))
    if not correcting:
        return False, message
    match = re.search(r"\bwhat\s+i\s+need\s+is\s*:\s*(.+)", message, re.I | re.S)
    if match:
        return True, "what I need is: " + match.group(1)
    match = re.search(r"\b(?:replace|change|switch)\b.*?\b(?:with|to)\s*:?[ \t]*(.+)", message, re.I | re.S)
    if match:
        return True, match.group(1)
    marker = _CORRECTION_RE.search(message)
    return True, message[marker.end():] if marker else message


def _category(message: str) -> str | None:
    match = _CATEGORY_RE.search(message)
    if not match:
        return None
    value = _clean(re.split(r"\s*,?\s+(?:but|while|preferably|with|under|below|up\s+to)\b", match.group(1), maxsplit=1, flags=re.I)[0])
    value = re.sub(r"^(?:a|an|some|the)\s+", "", value, flags=re.I)
    value = re.sub(r"^pair\s+of\s+", "", value, flags=re.I)
    return None if not value or _vague(value) else value


def _no_preference(message: str) -> tuple[frozenset[str], tuple[tuple[int, int], ...]]:
    result: set[str] = set()
    spans: list[tuple[int, int]] = []
    for match in _NO_PREFERENCE_RE.finditer(message):
        attribute = _attribute(match.group(1))
        if attribute:
            result.add(attribute)
            spans.append(match.span())
    return frozenset(result), tuple(spans)


def _constraint(
    attribute: str,
    values: list[str],
    message: str,
    turn: int,
    **kwargs: str,
) -> StructuredConstraint:
    return StructuredConstraint(attribute, values, source_turn=turn, source_text=message, **kwargs)  # type: ignore[arg-type]


def parse(message: str, turn: int = 0, last_asked_attribute: str | None = None) -> ParseResult:
    """Parse one message without catalog access or probabilistic components."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not isinstance(turn, int) or turn < 0:
        raise ValueError("turn must be a non-negative integer")
    no_preference, no_preference_spans = _no_preference(message)
    correcting, text = _focus(message)
    answer = _ANSWER_RE.match(message)
    provenance = "simulator_answer" if answer else "explicit"
    constraints: list[StructuredConstraint] = []
    category = _category(message)
    if category:
        constraints.append(_constraint("category", [category], message, turn))

    ignored = no_preference_spans if text == message else ()
    occurrences = [
        match for match in _VALUE_RE.finditer(text)
        if not any(match.start() < end and match.end() > start for start, end in ignored)
    ]
    groups: dict[tuple[str, str, str, int], list[re.Match[str]]] = {}
    for occurrence in occurrences:
        value = normalize_value(occurrence.group(0))
        attribute = _VALUE_TO_ATTRIBUTE[value]
        start = max(text.rfind(".", 0, occurrence.start()), text.rfind("!", 0, occurrence.start()), text.rfind("?", 0, occurrence.start())) + 1
        context = text[start:occurrence.start()]
        contrast = list(re.finditer(r"(?<!anything )\b(?:but|however)\b", context, re.I))
        if contrast:
            context = context[contrast[-1].end():]
        polarity = "exclude" if _NEGATION_RE.search(context) else "include"
        hard, soft = _HARD_RE.search(context), _SOFT_RE.search(context)
        strength = "hard" if hard and (not soft or hard.start() >= soft.start()) else "soft" if soft else "unspecified"
        groups.setdefault((attribute, polarity, strength, len(re.findall(r"[.!?]", text[:occurrence.start()]))), []).append(occurrence)
    for (attribute, polarity, strength, _), items in groups.items():
        values = list(dict.fromkeys(normalize_value(item.group(0)) for item in items))
        between = text[items[0].start():items[-1].end()]
        relation = "all" if len(items) > 1 and re.search(r"\band\b", between, re.I) else "any"
        constraints.append(
            _constraint(
                attribute,
                values,
                message,
                turn,
                relation=relation,
                polarity=polarity,
                strength=strength,
                provenance=provenance,
            )
        )

    bounded_budget_spans: list[tuple[int, int]] = []
    for pattern, operator in ((_MAX_RE, "at_most"), (_MIN_RE, "at_least")):
        for match in pattern.finditer(text):
            bounded_budget_spans.append(match.span())
            constraints.append(
                _constraint(
                    "budget",
                    [match.group(1)],
                    message,
                    turn,
                    operator=operator,
                    strength="hard",
                    provenance=provenance,
                )
            )
    for match in _EQUAL_RE.finditer(text):
        if any(
            match.start() < end and match.end() > start
            for start, end in bounded_budget_spans
        ):
            continue
        value = match.group(1) or match.group(2)
        constraints.append(
            _constraint(
                "budget",
                [value],
                message,
                turn,
                operator="equals",
                provenance=provenance,
            )
        )

    asked = _attribute(last_asked_attribute) if last_asked_attribute else None
    if asked and answer and not no_preference:
        unknown: list[str] = []
        for part in re.split(
            r"\s*;\s*|\s*,\s*|\s+\b(?:and|or)\b\s+",
            answer.group(1),
            flags=re.I,
        ):
            part = _clean(part)
            if not part or _vague(part) or _VALUE_RE.search(part) or re.search(r"\$?\d", part):
                continue
            value = normalize_value(part)
            if value not in unknown:
                unknown.append(value)
        if unknown:
            relation = "all" if re.search(r"\band\b", answer.group(1), re.I) else "any"
            constraints.append(_constraint(asked, unknown, message, turn, relation=relation, provenance="simulator_answer"))

    # The public initial-message template can contain a descriptive feature
    # that has no fixed-vocabulary token. It is explicit evidence, not an
    # ambiguous free-form guess, so retain it as a feature constraint.
    if category and len(constraints) == 1 and "." in message:
        remainder = _clean(message.split(".", 1)[1])
        requirement = re.sub(
            r"^(?:a\s+key\s+requirement\s+is|what\s+matters\s+is)\s*:\s*",
            "",
            remainder,
            flags=re.I,
        )
        if requirement and "still exploring" not in requirement.casefold() and not _vague(requirement):
            strength = "hard" if _HARD_RE.search(remainder) else "soft" if _SOFT_RE.search(remainder) else "unspecified"
            constraints.append(
                _constraint("feature", [requirement], message, turn, strength=strength)
            )

    unique: list[StructuredConstraint] = []
    seen: set[tuple[object, ...]] = set()
    for item in constraints:
        if item.normalized_key() not in seen:
            seen.add(item.normalized_key())
            unique.append(item)
    correction_attributes = frozenset(item.attribute for item in unique) if correcting else frozenset()
    return ParseResult(tuple(unique), no_preference, correction_attributes)


__all__ = ["ParseResult", "parse"]
