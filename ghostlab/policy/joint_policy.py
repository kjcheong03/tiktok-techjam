from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ghostlab.policy.decision_list import select_action
from ghostlab.policy.joint_actions import legalize_joint_action
from ghostlab.policy.models import DecisionList, JointAction
from ghostlab.state.memory import ConversationState


@dataclass(frozen=True)
class JointPolicyDecision:
    action: JointAction
    reason: str


@dataclass(frozen=True)
class JointObservablePolicy:
    policy: DecisionList
    allowed_routes: frozenset[str] = frozenset({"keyword"})
    allowed_depths: frozenset[int] = frozenset({100, 200})

    def __post_init__(self) -> None:
        if not self.allowed_routes or not self.allowed_depths:
            raise ValueError("joint policy requires bounded routes and depths")
        possible = {
            self.policy.default_action.retrieval_route,
            *(
                rule.action_patch.retrieval_route
                for rule in self.policy.rules
                if rule.action_patch.retrieval_route != "__inherit__"
            ),
        }
        if not possible <= self.allowed_routes:
            raise ValueError("joint policy contains an unregistered retrieval route")

    @property
    def possible_routes(self) -> frozenset[str]:
        return frozenset(
            {
                self.policy.default_action.retrieval_route,
                *(
                    rule.action_patch.retrieval_route
                    for rule in self.policy.rules
                    if rule.action_patch.retrieval_route != "__inherit__"
                ),
            }
        )

    def decide(
        self, state: ConversationState, features: Mapping[str, object]
    ) -> JointPolicyDecision:
        selected = select_action(self.policy, features)
        legal = legalize_joint_action(
            selected,
            state,
            allowed_routes=self.allowed_routes,
            allowed_depths=self.allowed_depths,
            base_action=self.policy.default_action,
        )
        return JointPolicyDecision(legal, "observable_decision_list")

    @classmethod
    def from_path(cls, path: str | Path) -> JointObservablePolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported joint policy schema")
        return cls(
            DecisionList.model_validate(payload["policy"]),
            frozenset(str(value) for value in payload["allowed_routes"]),
            frozenset(int(value) for value in payload["allowed_depths"]),
        )
