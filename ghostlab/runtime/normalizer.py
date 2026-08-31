from __future__ import annotations

from collections.abc import Iterable

from ghostlab.competition.contract import TurnResponse


def normalize_identifiers(
    values: Iterable[object], catalog_ids: set[str], top_k: int
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in values:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        identifier = str(value).strip()
        if not identifier or identifier in seen or identifier not in catalog_ids:
            continue
        seen.add(identifier)
        result.append({"parent_asin": identifier})
        if len(result) == top_k:
            break
    return result


def normalize_response(payload: object, catalog_ids: set[str], top_k: int) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("response must be an object")
    candidate = {
        "message": payload.get("message", ""),
        "ask_attribute": payload.get("ask_attribute"),
        "recommendations": normalize_identifiers(
            payload.get("recommendations", []), catalog_ids, top_k
        ),
        "usage": payload.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
    }
    validated = TurnResponse.model_validate(candidate)
    recommendations = []
    for item in validated.recommendations:
        recommendation: dict[str, object] = {"parent_asin": item.parent_asin}
        if item.score is not None:
            recommendation["score"] = item.score
        recommendations.append(recommendation)
    return {
        "message": validated.message,
        "ask_attribute": validated.ask_attribute,
        "recommendations": recommendations,
        "usage": validated.usage.model_dump(),
    }
