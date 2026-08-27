from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from ghostlab.policy.joint_actions import legalize_joint_action
from ghostlab.policy.joint_policy import JointPolicyDecision
from ghostlab.policy.models import JointAction
from ghostlab.research.counterfactual_expert import ExpertLabel
from ghostlab.state.memory import ConversationState


@dataclass(frozen=True)
class DistilledNode:
    action_id: str
    confidence: float
    sample_count: int
    feature: str | None = None
    threshold: float | None = None
    lower: DistilledNode | None = None
    upper: DistilledNode | None = None

    def predict(self, features: Mapping[str, float]) -> tuple[str, float]:
        if self.feature is None or self.threshold is None:
            return self.action_id, self.confidence
        value = features.get(self.feature)
        if value is None:
            return self.action_id, self.confidence
        child = self.lower if value <= self.threshold else self.upper
        return (self.action_id, self.confidence) if child is None else child.predict(features)


@dataclass(frozen=True)
class DistilledPolicyModel:
    feature_names: tuple[str, ...]
    action_order: tuple[str, ...]
    root: DistilledNode
    maximum_depth: int
    minimum_leaf_sessions: int
    training_states: int

    def predict(self, features: Mapping[str, float]) -> tuple[str, float]:
        missing = set(self.feature_names) - features.keys()
        if missing:
            raise ValueError(f"missing distilled features: {sorted(missing)}")
        return self.root.predict(features)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_type": "distilled_counterfactual_tree_v1",
            "feature_names": list(self.feature_names),
            "action_order": list(self.action_order),
            "root": asdict(self.root),
            "maximum_depth": self.maximum_depth,
            "minimum_leaf_sessions": self.minimum_leaf_sessions,
            "training_states": self.training_states,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> DistilledPolicyModel:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported distilled policy schema")

        def node(value: Mapping[str, object]) -> DistilledNode:
            lower = value.get("lower")
            upper = value.get("upper")
            return DistilledNode(
                action_id=str(value["action_id"]),
                confidence=float(cast("float | int | str", value["confidence"])),
                sample_count=int(cast("float | int | str", value["sample_count"])),
                feature=None if value.get("feature") is None else str(value["feature"]),
                threshold=None
                if value.get("threshold") is None
                else float(cast("float | int | str", value["threshold"])),
                lower=node(lower) if isinstance(lower, dict) else None,
                upper=node(upper) if isinstance(upper, dict) else None,
            )

        root = payload.get("root")
        if not isinstance(root, dict):
            raise TypeError("distilled policy root must be an object")
        feature_names = payload.get("feature_names")
        action_order = payload.get("action_order")
        if not isinstance(feature_names, list) or not isinstance(action_order, list):
            raise TypeError("distilled feature and action names must be arrays")
        return cls(
            tuple(str(value) for value in feature_names),
            tuple(str(value) for value in action_order),
            node(root),
            int(cast("float | int | str", payload["maximum_depth"])),
            int(cast("float | int | str", payload["minimum_leaf_sessions"])),
            int(cast("float | int | str", payload["training_states"])),
        )


def _best_action(
    labels: Sequence[ExpertLabel], action_order: tuple[str, ...]
) -> tuple[str, float, float]:
    eligible = [
        action
        for action in action_order
        if all(action in label.action_rewards for label in labels)
    ]
    if not eligible:
        raise ValueError("distillation leaf has no common action")
    action = min(
        eligible,
        key=lambda item: (
            -statistics.fmean(label.action_rewards[item] for label in labels),
            action_order.index(item),
        ),
    )
    confidence = sum(label.best_action_id == action for label in labels) / len(labels)
    regret = statistics.fmean(
        label.best_reward - label.action_rewards[action] for label in labels
    )
    return action, confidence, regret


def fit_distilled_policy(
    labels: Sequence[ExpertLabel],
    *,
    feature_names: tuple[str, ...],
    action_order: tuple[str, ...],
    maximum_depth: int = 2,
    minimum_leaf_sessions: int = 10,
    minimum_regret_improvement: float = 0.002,
) -> DistilledPolicyModel:
    """Distil fold-local expert rewards into a small deterministic regret tree."""

    if not labels or maximum_depth < 0 or minimum_leaf_sessions < 1:
        raise ValueError("invalid distilled-policy fit bounds")

    def build(rows: Sequence[ExpertLabel], depth: int) -> DistilledNode:
        action, confidence, leaf_regret = _best_action(rows, action_order)
        leaf = DistilledNode(action, confidence, len(rows))
        if depth >= maximum_depth or len(rows) < 2 * minimum_leaf_sessions:
            return leaf
        candidates: list[
            tuple[float, str, float, Sequence[ExpertLabel], Sequence[ExpertLabel]]
        ] = []
        for feature in feature_names:
            values = sorted({label.features[feature] for label in rows})
            for left, right in pairwise(values):
                threshold = (left + right) / 2.0
                lower = [label for label in rows if label.features[feature] <= threshold]
                upper = [label for label in rows if label.features[feature] > threshold]
                if (
                    len({label.sample_id for label in lower}) < minimum_leaf_sessions
                    or len({label.sample_id for label in upper})
                    < minimum_leaf_sessions
                ):
                    continue
                _, _, lower_regret = _best_action(lower, action_order)
                _, _, upper_regret = _best_action(upper, action_order)
                regret = (
                    len(lower) * lower_regret + len(upper) * upper_regret
                ) / len(rows)
                candidates.append((regret, feature, threshold, lower, upper))
        if not candidates:
            return leaf
        best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        if leaf_regret - best[0] < minimum_regret_improvement:
            return leaf
        return DistilledNode(
            action,
            confidence,
            len(rows),
            best[1],
            best[2],
            build(best[3], depth + 1),
            build(best[4], depth + 1),
        )

    return DistilledPolicyModel(
        feature_names,
        action_order,
        build(labels, 0),
        maximum_depth,
        minimum_leaf_sessions,
        len(labels),
    )


@dataclass(frozen=True)
class DistilledExpertPolicy:
    model: DistilledPolicyModel
    actions: Mapping[str, JointAction]
    allowed_routes: frozenset[str]
    allowed_depths: frozenset[int]
    confidence_threshold: float = 0.55
    fallback_action_id: str = "base"

    @property
    def possible_routes(self) -> frozenset[str]:
        return frozenset(action.retrieval_route for action in self.actions.values())

    def decide(
        self, state: ConversationState, features: Mapping[str, object]
    ) -> JointPolicyDecision:
        numeric = {
            name: float(cast("float | int | str", features[name]))
            for name in self.model.feature_names
        }
        action_id, confidence = self.model.predict(numeric)
        if confidence < self.confidence_threshold:
            action_id = self.fallback_action_id
        if action_id not in self.actions or self.fallback_action_id not in self.actions:
            raise ValueError("distilled policy references an unknown action")
        base = self.actions[self.fallback_action_id]
        action = legalize_joint_action(
            self.actions[action_id],
            state,
            allowed_routes=self.allowed_routes,
            allowed_depths=self.allowed_depths,
            base_action=base,
        )
        return JointPolicyDecision(action, "distilled_counterfactual_expert")

    @classmethod
    def from_path(cls, path: str | Path) -> DistilledExpertPolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = DistilledPolicyModel.from_payload(payload["model"])
        return cls(
            model,
            {
                str(key): JointAction.model_validate(value)
                for key, value in payload["actions"].items()
            },
            frozenset(str(value) for value in payload["allowed_routes"]),
            frozenset(int(value) for value in payload["allowed_depths"]),
            float(payload.get("confidence_threshold", 0.55)),
            str(payload.get("fallback_action_id", "base")),
        )
