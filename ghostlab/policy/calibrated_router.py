from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ghostlab.research.firewall import reject_forbidden_names


@dataclass(frozen=True)
class RouterTrainingState:
    sample_id: str
    features: Mapping[str, float]
    route_rewards: Mapping[str, float]


@dataclass(frozen=True)
class RouteDecision:
    route: str
    predicted_advantage: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class CalibratedRouteModel:
    feature_names: tuple[str, ...]
    base_route: str
    route_weights: Mapping[str, tuple[float, ...]]
    advantage_threshold: float
    calibration_precision: float
    training_sessions: int
    calibration_sessions: int

    @property
    def possible_routes(self) -> frozenset[str]:
        return frozenset({self.base_route, *self.route_weights})

    def _values(self, features: Mapping[str, float]) -> dict[str, float]:
        unknown = set(features) ^ set(self.feature_names)
        if unknown:
            raise ValueError(f"router feature mismatch: {sorted(unknown)}")
        vector: NDArray[np.float64] = np.asarray(
            [1.0, *(features[name] for name in self.feature_names)], dtype=np.float64
        )
        return {
            route: float(vector @ np.asarray(weights, dtype=np.float64))
            for route, weights in self.route_weights.items()
        }

    def decide(self, features: Mapping[str, float]) -> RouteDecision:
        values = self._values(features)
        base_value = values[self.base_route]
        alternatives = [route for route in values if route != self.base_route]
        if not alternatives:
            return RouteDecision(self.base_route, 0.0, 1.0, "always_base")
        route = min(
            alternatives,
            key=lambda item: (-(values[item] - base_value), item),
        )
        advantage = values[route] - base_value
        confidence = 1.0 / (1.0 + math.exp(-8.0 * (advantage - self.advantage_threshold)))
        if advantage < self.advantage_threshold:
            return RouteDecision(
                self.base_route, advantage, 1.0 - confidence, "below_calibrated_margin"
            )
        return RouteDecision(route, advantage, confidence, "predicted_positive_benefit")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            **asdict(self),
        }

    @classmethod
    def from_path(cls, path: str | Path) -> CalibratedRouteModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("schema_version", None) != 1:
            raise ValueError("unsupported calibrated router schema")
        payload["feature_names"] = tuple(payload["feature_names"])
        payload["route_weights"] = {
            route: tuple(weights)
            for route, weights in payload["route_weights"].items()
        }
        return cls(**payload)


def _fit_values(
    states: Sequence[RouterTrainingState],
    feature_names: tuple[str, ...],
    routes: tuple[str, ...],
    l2: float,
) -> dict[str, tuple[float, ...]]:
    matrix: NDArray[np.float64] = np.asarray(
        [
            [1.0, *(state.features[name] for name in feature_names)]
            for state in states
        ],
        dtype=np.float64,
    )
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    result = {}
    for route in routes:
        target: NDArray[np.float64] = np.asarray(
            [state.route_rewards[route] for state in states], dtype=np.float64
        )
        weights = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
        result[route] = tuple(float(value) for value in weights)
    return result


def fit_calibrated_router(
    fit_states: Sequence[RouterTrainingState],
    calibration_states: Sequence[RouterTrainingState],
    *,
    feature_names: tuple[str, ...],
    routes: tuple[str, ...],
    base_route: str = "keyword",
    l2: float = 1.0,
    minimum_precision: float = 0.6,
    minimum_routed_sessions: int = 5,
) -> CalibratedRouteModel:
    """Fit values, then choose a conservative threshold on separate train data."""

    if (
        not fit_states
        or not calibration_states
        or base_route not in routes
        or l2 < 0.0
        or not 0.0 <= minimum_precision <= 1.0
    ):
        raise ValueError("invalid calibrated-router fit inputs")
    reject_forbidden_names(feature_names)
    for state in (*fit_states, *calibration_states):
        if any(route not in state.route_rewards for route in routes):
            raise ValueError("router state is missing a registered route reward")
    weights = _fit_values(fit_states, feature_names, routes, l2)
    provisional = CalibratedRouteModel(
        feature_names,
        base_route,
        weights,
        0.0,
        0.0,
        len({state.sample_id for state in fit_states}),
        len({state.sample_id for state in calibration_states}),
    )
    predicted: list[tuple[float, bool]] = []
    for state in calibration_states:
        values = provisional._values(state.features)
        alternative = min(
            (route for route in routes if route != base_route),
            key=lambda route: (-(values[route] - values[base_route]), route),
        )
        advantage = values[alternative] - values[base_route]
        actually_positive = (
            state.route_rewards[alternative] > state.route_rewards[base_route]
        )
        predicted.append((advantage, actually_positive))
    thresholds = sorted({value for value, _ in predicted}, reverse=True)
    selected_threshold = math.inf
    selected_precision = 1.0
    for threshold in thresholds:
        routed = [positive for value, positive in predicted if value >= threshold]
        if len(routed) < minimum_routed_sessions:
            continue
        precision = sum(routed) / len(routed)
        if precision >= minimum_precision:
            selected_threshold = threshold
            selected_precision = precision
    return CalibratedRouteModel(
        feature_names,
        base_route,
        weights,
        selected_threshold,
        selected_precision,
        len({state.sample_id for state in fit_states}),
        len({state.sample_id for state in calibration_states}),
    )
