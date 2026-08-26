from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ghostlab.research.counterfactual import Action, ActionOutcome


@dataclass(frozen=True)
class QuestionFeatures:
    has_initial_constraint: bool
    critical_rater: bool
    material_profile_tag: bool

    def key(self, names: tuple[str, ...]) -> tuple[bool, ...]:
        return tuple(bool(getattr(self, name)) for name in names)


@dataclass(frozen=True)
class QuestionTable:
    feature_names: tuple[str, ...]
    cells: dict[tuple[bool, ...], Action]
    default_action: Action

    def predict(self, features: QuestionFeatures) -> Action:
        return self.cells.get(features.key(self.feature_names), self.default_action)


def fit_question_table(
    outcomes: Iterable[ActionOutcome],
    features: dict[str, QuestionFeatures],
    feature_names: tuple[str, ...],
    actions: tuple[Action, ...],
    *,
    minimum_cell_sessions: int = 5,
) -> QuestionTable:
    rewards: dict[tuple[tuple[bool, ...], Action], list[float]] = defaultdict(list)
    global_rewards: dict[Action, list[float]] = defaultdict(list)
    cell_sessions: dict[tuple[bool, ...], set[str]] = defaultdict(set)
    for outcome in outcomes:
        if outcome.action not in actions:
            continue
        key = features[outcome.sample_id].key(feature_names)
        rewards[(key, outcome.action)].append(outcome.reward)
        global_rewards[outcome.action].append(outcome.reward)
        cell_sessions[key].add(outcome.sample_id)
    default = min(
        actions,
        key=lambda action: (-statistics.fmean(global_rewards[action]), str(action)),
    )
    cells: dict[tuple[bool, ...], Action] = {}
    for key, sample_ids in cell_sessions.items():
        if len(sample_ids) < minimum_cell_sessions:
            continue
        cells[key] = min(
            actions,
            key=lambda action: (-statistics.fmean(rewards[(key, action)]), str(action)),
        )
    return QuestionTable(feature_names, cells, default)
