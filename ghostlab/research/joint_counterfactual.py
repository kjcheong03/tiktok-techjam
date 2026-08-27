from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ghostlab.policy.joint_actions import JointTrainingState


@dataclass(frozen=True)
class JointActionTable:
    feature_names: tuple[str, ...]
    cells: Mapping[tuple[bool, ...], str]
    default_action_id: str
    minimum_cell_sessions: int

    def predict(self, features: Mapping[str, float]) -> str:
        key = tuple(features[name] >= 0.5 for name in self.feature_names)
        return self.cells.get(key, self.default_action_id)


def fit_joint_action_table(
    states: Iterable[JointTrainingState],
    *,
    feature_names: tuple[str, ...],
    action_ids: tuple[str, ...],
    minimum_cell_sessions: int = 10,
) -> JointActionTable:
    """Fit a bounded fold-local table from dense counterfactual rewards."""

    rows = list(states)
    if not rows or not action_ids or minimum_cell_sessions < 1:
        raise ValueError("joint fit requires states, actions, and a positive leaf size")
    global_values: dict[str, list[float]] = defaultdict(list)
    cells: dict[tuple[bool, ...], list[JointTrainingState]] = defaultdict(list)
    for state in rows:
        if set(state.features) & {"target", "reward", "scenario_type"}:
            raise ValueError("research-only fields crossed the joint runtime schema")
        key = tuple(state.features[name] >= 0.5 for name in feature_names)
        cells[key].append(state)
        for action_id in action_ids:
            if action_id in state.action_rewards:
                global_values[action_id].append(state.action_rewards[action_id])
    if any(not global_values[action_id] for action_id in action_ids):
        raise ValueError("every joint action requires training rewards")
    default = min(
        action_ids,
        key=lambda action: (-statistics.fmean(global_values[action]), action),
    )
    selected: dict[tuple[bool, ...], str] = {}
    for key, cell_rows in sorted(cells.items()):
        session_count = len({row.sample_id for row in cell_rows})
        if session_count < minimum_cell_sessions:
            continue
        eligible = [
            action
            for action in action_ids
            if all(action in row.action_rewards for row in cell_rows)
        ]
        if eligible:
            selected[key] = min(
                eligible,
                key=lambda action: (
                    -statistics.fmean(
                        row.action_rewards[action] for row in cell_rows
                    ),
                    action,
                ),
            )
    return JointActionTable(feature_names, selected, default, minimum_cell_sessions)
