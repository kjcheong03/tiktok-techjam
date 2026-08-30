from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.adaptive_questions import AdaptiveQuestionPolicy, QuestionContext
from ghostlab.policy.candidate_statistics import CandidateStatistics
from ghostlab.state.memory import ConversationState
from ghostlab.state.v2_view import V2StateView


@dataclass(frozen=True)
class RewardVOICalibration:
    action_adjustments: Mapping[AskAttribute, float]
    training_sessions: int
    shrinkage: float


@dataclass(frozen=True)
class EIGQuestionDecision:
    ask_attribute: AskAttribute | None
    reason: str
    values: Mapping[AskAttribute | None, float]


@dataclass(frozen=True)
class CandidateEIGPolicy:
    question_value_margin: float = 0.0
    minimum_coverage: float = 0.2
    maximum_no_preference: float = 0.8
    max_question_turn: int = 6
    broad_discovery_turns: int = 2
    official_turn_cost: float = 0.02
    information_reward_scale: float = 0.2
    entropy_weight: float = 0.35
    partition_weight: float = 0.65
    calibration: RewardVOICalibration | None = None
    fallback: AdaptiveQuestionPolicy = field(default_factory=AdaptiveQuestionPolicy)

    def __post_init__(self) -> None:
        if self.question_value_margin < 0.0:
            raise ValueError("question value margin must be non-negative")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum coverage must be between zero and one")
        if not 0.0 <= self.maximum_no_preference <= 1.0:
            raise ValueError("maximum no-preference must be between zero and one")
        if self.official_turn_cost < 0.0 or self.information_reward_scale < 0.0:
            raise ValueError("reward proxy terms must be non-negative")

    def decide(
        self,
        state: ConversationState | V2StateView,
        statistics: CandidateStatistics,
        *,
        turn: int,
        message: str,
        unavailable_attributes: frozenset[str] = frozenset(),
    ) -> EIGQuestionDecision:
        if turn > self.max_question_turn:
            return EIGQuestionDecision(None, "question_budget_exhausted", {None: 0.0})
        if turn <= self.broad_discovery_turns:
            return EIGQuestionDecision(
                "other", "broad_discovery", {None: 0.0, "other": 1.0}
            )
        active = {item.attribute for item in state.active_values()}
        unavailable = (
            active
            | set(state.asked_attributes)
            | state.no_preference_attributes
            | set(unavailable_attributes)
        )
        legal = {
            attribute
            for attribute in statistics.facets
            if attribute not in unavailable
        }
        values: dict[AskAttribute | None, float] = {None: 0.0}
        for attribute, facet in statistics.facets.items():
            if attribute not in legal:
                continue
            if (
                facet.coverage < self.minimum_coverage
                or facet.no_preference_probability > self.maximum_no_preference
            ):
                continue
            information = (
                self.entropy_weight * facet.normalized_entropy
                + self.partition_weight * facet.partition_gain
            ) * facet.coverage
            value = (
                self.information_reward_scale * information
                - self.official_turn_cost
            )
            if self.calibration is not None:
                value += self.calibration.action_adjustments.get(attribute, 0.0)
            values[attribute] = value
        if len(values) == 1:
            fallback = self.fallback.decide(
                QuestionContext(
                    turn=turn,
                    message=message,
                    active_attributes=frozenset(
                        {
                            *(item.attribute for item in state.active_values()),
                            *unavailable_attributes,
                        }
                    ),
                    asked_attributes=frozenset(state.asked_attributes),
                    no_preference_attributes=frozenset(
                        state.no_preference_attributes
                    ),
                    last_asked_attribute=state.last_asked_attribute,
                )
            )
            return EIGQuestionDecision(
                fallback.ask_attribute, "sparse_statistics_fallback", values
            )
        selected = min(
            (action for action in values if action is not None),
            key=lambda action: (-values[action], str(action)),
        )
        if values[selected] < self.question_value_margin:
            return EIGQuestionDecision(None, "reward_aware_stop", values)
        return EIGQuestionDecision(selected, "candidate_information_gain", values)
