from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

from ghostlab.competition.contract import AgentProtocol, AskAttribute
from ghostlab.research.firewall import runtime_profile
from ghostlab.research.replay import ReplayEnvironment, session_reward

Action = AskAttribute | None
AgentFactory = Callable[[], AgentProtocol]


@dataclass(frozen=True)
class ActionOutcome:
    sample_id: str
    action: Action
    reward: float
    hit: bool
    first_hit_turn: int | None
    best_rank: int | None


class CounterfactualEvaluator:
    """First-action evaluator with a frozen continuation policy."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        categories: dict[str, list[str]],
        products: dict[str, dict],
        continuation_id: str,
    ) -> None:
        self.agent_factory = agent_factory
        self.categories = categories
        self.products = products
        self.continuation_id = continuation_id
        self.agent = agent_factory()
        self.cache: dict[str, ActionOutcome] = {}
        self.cache_hits = 0

    def _key(self, sample: dict, action: Action) -> str:
        payload = {
            "sample_id": sample["sample_id"],
            "action": action,
            "continuation_id": self.continuation_id,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def evaluate_action(self, sample: dict, action: Action) -> ActionOutcome:
        key = self._key(sample, action)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        environment = ReplayEnvironment(sample, self.categories, self.products)
        action_key = "none" if action is None else action
        environment.session_id = f"{environment.session_id}_{action_key}"
        observation = environment.observe()
        agent = self.agent
        agent.reset(observation.session_id, runtime_profile(sample))
        response = agent.respond(
            observation.session_id,
            observation.user_message,
            observation.turn,
            observation.top_k,
        )
        response = {**response, "ask_attribute": action}
        override = getattr(agent, "override_last_question", None)
        if callable(override):
            override(observation.session_id, action)
        next_observation = environment.step(response)
        while next_observation is not None:
            response = agent.respond(
                next_observation.session_id,
                next_observation.user_message,
                next_observation.turn,
                next_observation.top_k,
            )
            next_observation = environment.step(response)
        session = environment.session_result()
        outcome = ActionOutcome(
            sample_id=str(sample["sample_id"]),
            action=action,
            reward=session_reward(session),
            hit=bool(session["hit"]),
            first_hit_turn=session["first_hit_turn"],
            best_rank=session["best_rank"],
        )
        self.cache[key] = outcome
        return outcome

    def branches(self, sample: dict, actions: Iterable[Action]) -> list[ActionOutcome]:
        return [self.evaluate_action(sample, action) for action in actions]

    @staticmethod
    def best(outcomes: Iterable[ActionOutcome]) -> ActionOutcome:
        return min(
            outcomes,
            key=lambda item: (-item.reward, "" if item.action is None else item.action),
        )

    @staticmethod
    def serialize(outcomes: Iterable[ActionOutcome]) -> list[dict]:
        return [asdict(outcome) for outcome in outcomes]
