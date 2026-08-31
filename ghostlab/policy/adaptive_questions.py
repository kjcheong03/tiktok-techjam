from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.signals import RetrievalSignals

DEFAULT_PRIORITY: tuple[AskAttribute, ...] = (
    "budget",
    "color",
    "feature",
    "size",
    "material",
    "use_case",
    "style",
)
DecisionReason = Literal[
    "initial_discovery",
    "recover_unhelpful_specific",
    "missing_constraint",
    "discovery_refresh",
    "confident_stop",
    "question_budget_exhausted",
    "no_question_remaining",
]


@dataclass(frozen=True)
class QuestionContext:
    turn: int
    message: str
    active_attributes: frozenset[str]
    asked_attributes: frozenset[str]
    no_preference_attributes: frozenset[str]
    last_asked_attribute: str | None
    retrieval: RetrievalSignals | None = None


@dataclass(frozen=True)
class QuestionDecision:
    ask_attribute: AskAttribute | None
    reason: DecisionReason


@dataclass(frozen=True)
class AdaptiveQuestionPolicy:
    priority: tuple[AskAttribute, ...] = DEFAULT_PRIORITY
    initial_other_turns: int = 2
    other_refresh_interval: int = 3
    max_question_turn: int = 9
    confident_entropy: float = 0.25
    confident_active_attributes: int = 3

    def decide(self, context: QuestionContext) -> QuestionDecision:
        if context.turn > self.max_question_turn:
            return QuestionDecision(None, "question_budget_exhausted")
        if context.turn <= self.initial_other_turns:
            return QuestionDecision("other", "initial_discovery")

        previous_was_unhelpful = "don't have an additional preference" in (
            context.message.casefold()
        )
        if previous_was_unhelpful and context.last_asked_attribute not in {
            None,
            "other",
        }:
            return QuestionDecision("other", "recover_unhelpful_specific")

        signals = context.retrieval
        if (
            signals is not None
            and signals.normalized_entropy is not None
            and signals.normalized_entropy <= self.confident_entropy
            and len(context.active_attributes) >= self.confident_active_attributes
        ):
            return QuestionDecision(None, "confident_stop")

        offset = context.turn - self.initial_other_turns
        if (
            self.other_refresh_interval > 0
            and offset % self.other_refresh_interval == 0
        ):
            return QuestionDecision("other", "discovery_refresh")

        unavailable = (
            context.active_attributes
            | context.asked_attributes
            | context.no_preference_attributes
        )
        question = next(
            (attribute for attribute in self.priority if attribute not in unavailable),
            None,
        )
        if question is None:
            return QuestionDecision(None, "no_question_remaining")
        return QuestionDecision(question, "missing_constraint")
