from __future__ import annotations

import operator
from collections.abc import Mapping

from ghostlab.policy.models import ActionPatch, DecisionList, JointAction, Predicate

_COMPARATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def matches(predicate: Predicate, features: Mapping[str, object]) -> bool:
    missing = predicate.feature not in features or features[predicate.feature] is None
    if predicate.operator == "is_missing":
        return missing
    if predicate.operator == "is_not_missing":
        return not missing
    if missing:
        return False
    actual = features[predicate.feature]
    if predicate.operator == "contains":
        return isinstance(actual, (str, list, tuple, set)) and predicate.value in actual
    try:
        return bool(
            _COMPARATORS[predicate.operator](actual, predicate.value)  # type: ignore[arg-type]
        )
    except TypeError:
        return False


def overlay(default: JointAction, patch: ActionPatch) -> JointAction:
    values = default.model_dump()
    for key, value in patch.model_dump().items():
        if value != "__inherit__":
            values[key] = value
    return JointAction.model_validate(values)


def select_action(policy: DecisionList, features: Mapping[str, object]) -> JointAction:
    for rule in policy.rules:
        if all(matches(predicate, features) for predicate in rule.all_conditions):
            return overlay(policy.default_action, rule.action_patch)
    return policy.default_action
