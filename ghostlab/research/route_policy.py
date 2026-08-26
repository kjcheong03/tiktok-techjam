from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteFeatures:
    has_initial_constraint: bool
    critical_rater: bool

    def key(self, names: tuple[str, ...]) -> tuple[bool, ...]:
        return tuple(bool(getattr(self, name)) for name in names)


@dataclass(frozen=True)
class RouteTable:
    feature_names: tuple[str, ...]
    cells: dict[tuple[bool, ...], str]
    default_route: str

    def predict(self, features: RouteFeatures) -> str:
        return self.cells.get(features.key(self.feature_names), self.default_route)


def fit_route_table(
    training_ids: Iterable[str],
    rewards: dict[str, dict[str, float]],
    features: dict[str, RouteFeatures],
    routes: tuple[str, ...],
    feature_names: tuple[str, ...],
    minimum_cell_sessions: int = 5,
) -> RouteTable:
    cell_ids: dict[tuple[bool, ...], list[str]] = defaultdict(list)
    ids = list(training_ids)
    for sample_id in ids:
        cell_ids[features[sample_id].key(feature_names)].append(sample_id)
    default = min(
        routes,
        key=lambda route: (
            -statistics.fmean(rewards[route][sample_id] for sample_id in ids),
            route,
        ),
    )
    cells = {}
    for key, sample_ids in cell_ids.items():
        if len(sample_ids) < minimum_cell_sessions:
            continue
        cells[key] = min(
            routes,
            key=lambda route: (
                -statistics.fmean(
                    rewards[route][sample_id] for sample_id in sample_ids
                ),
                route,
            ),
        )
    return RouteTable(feature_names, cells, default)
