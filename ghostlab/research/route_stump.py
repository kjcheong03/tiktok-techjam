from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class RouteStump:
    default_route: str
    feature: str | None = None
    threshold: float | None = None
    lower_route: str | None = None
    upper_route: str | None = None

    def predict(self, features: Mapping[str, float]) -> str:
        if self.feature is None or self.threshold is None:
            return self.default_route
        value = features.get(self.feature)
        if value is None:
            return self.default_route
        if value <= self.threshold:
            return self.lower_route or self.default_route
        return self.upper_route or self.default_route


def _best_route(
    sample_ids: list[str],
    rewards: Mapping[str, Mapping[str, float]],
    routes: tuple[str, ...],
) -> tuple[str, float]:
    scored = [
        (
            route,
            statistics.fmean(rewards[route][sample_id] for sample_id in sample_ids),
        )
        for route in routes
    ]
    return min(scored, key=lambda item: (-item[1], item[0]))


def fit_route_stump(
    training_ids: Iterable[str],
    rewards: Mapping[str, Mapping[str, float]],
    features: Mapping[str, Mapping[str, float]],
    routes: tuple[str, ...],
    *,
    minimum_leaf_sessions: int = 15,
    simplicity_tie_band: float = 0.01,
) -> RouteStump:
    ids = sorted(training_ids)
    if len(ids) < 2 * minimum_leaf_sessions:
        raise ValueError("training set is too small for the requested leaf size")
    default_route, constant_score = _best_route(ids, rewards, routes)
    candidates: list[tuple[float, str, float, str, str]] = []
    feature_names = sorted(
        set.intersection(*(set(features[sample_id]) for sample_id in ids))
    )
    for feature in feature_names:
        values = sorted({features[sample_id][feature] for sample_id in ids})
        thresholds = [
            (left + right) / 2.0 for left, right in pairwise(values) if left != right
        ]
        for threshold in thresholds:
            lower = [
                sample_id
                for sample_id in ids
                if features[sample_id][feature] <= threshold
            ]
            lower_set = set(lower)
            upper = [sample_id for sample_id in ids if sample_id not in lower_set]
            if len(lower) < minimum_leaf_sessions or len(upper) < minimum_leaf_sessions:
                continue
            lower_route, lower_score = _best_route(lower, rewards, routes)
            upper_route, upper_score = _best_route(upper, rewards, routes)
            score = (len(lower) * lower_score + len(upper) * upper_score) / len(ids)
            candidates.append((score, feature, threshold, lower_route, upper_route))

    if not candidates:
        return RouteStump(default_route)
    best = min(
        candidates,
        key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
    )
    if constant_score >= best[0] - simplicity_tie_band:
        return RouteStump(default_route)
    return RouteStump(
        default_route=default_route,
        feature=best[1],
        threshold=best[2],
        lower_route=best[3],
        upper_route=best[4],
    )
