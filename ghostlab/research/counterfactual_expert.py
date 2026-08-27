from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ghostlab.research.firewall import reject_forbidden_names


@dataclass(frozen=True)
class ExpertState:
    sample_id: str
    turn: int
    features: Mapping[str, float]
    action_rewards: Mapping[str, float]


@dataclass(frozen=True)
class ExpertLabel:
    sample_id: str
    turn: int
    features: Mapping[str, float]
    action_rewards: Mapping[str, float]
    best_action_id: str
    best_reward: float
    runner_up_reward: float

    @property
    def margin(self) -> float:
        return self.best_reward - self.runner_up_reward


@dataclass(frozen=True)
class CounterfactualExpert:
    action_order: tuple[str, ...]
    simplicity_penalty: Mapping[str, float]

    def label(self, state: ExpertState) -> ExpertLabel:
        reject_forbidden_names(state.features)
        available = [
            action for action in self.action_order if action in state.action_rewards
        ]
        if not available:
            raise ValueError("expert state has no registered action outcomes")
        ordered = sorted(
            available,
            key=lambda action: (
                -(
                    state.action_rewards[action]
                    - self.simplicity_penalty.get(action, 0.0)
                ),
                self.action_order.index(action),
            ),
        )
        best = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else best
        return ExpertLabel(
            state.sample_id,
            state.turn,
            dict(state.features),
            dict(state.action_rewards),
            best,
            state.action_rewards[best],
            state.action_rewards[runner_up],
        )

    def labels(self, states: Iterable[ExpertState]) -> list[ExpertLabel]:
        return [self.label(state) for state in states]


def aggregate_expert_iterations(
    rounds: Iterable[Iterable[ExpertLabel]], *, maximum_rounds: int = 3
) -> list[ExpertLabel]:
    """Bounded DAgger-style aggregation with deterministic state replacement."""

    if maximum_rounds < 1:
        raise ValueError("expert iteration count must be positive")
    merged: dict[tuple[str, int, tuple[tuple[str, float], ...]], ExpertLabel] = {}
    for round_index, labels in enumerate(rounds):
        if round_index >= maximum_rounds:
            break
        for label in labels:
            key = (
                label.sample_id,
                label.turn,
                tuple(sorted(label.features.items())),
            )
            merged[key] = label
    return [merged[key] for key in sorted(merged)]
