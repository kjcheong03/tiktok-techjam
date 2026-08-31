from __future__ import annotations

import copy
import statistics
from collections import Counter, defaultdict
from itertools import pairwise

from ghostlab.policy.learned_questions import (
    QuestionAction,
    QuestionTrainingState,
    legal_question_actions,
    observable_question_features,
)
from ghostlab.research.counterfactual import ActionOutcome
from ghostlab.research.replay import ReplayEnvironment, session_reward
from ghostlab.runtime.experimental_questions import ExperimentalAgent
from ghostlab.state.memory import ConversationState


def _force_question(
    agent: ExperimentalAgent,
    session_id: str,
    state: ConversationState,
    action: QuestionAction,
) -> None:
    forced = copy.deepcopy(state)
    if action is not None:
        forced.asked_attributes.append(action)
    forced.last_asked_attribute = action
    agent.sessions[session_id] = forced


def _branch_outcome(
    agent: ExperimentalAgent,
    environment: ReplayEnvironment,
    state_after_observation: ConversationState,
    response: dict,
    action: QuestionAction,
) -> ActionOutcome:
    branch = environment.clone()
    session_id = branch.session_id
    _force_question(agent, session_id, state_after_observation, action)
    payload = {**response, "ask_attribute": action}
    observation = branch.step(payload)
    while observation is not None:
        before = copy.deepcopy(agent.sessions[session_id])
        continuation = agent.respond(
            observation.session_id,
            observation.user_message,
            observation.turn,
            observation.top_k,
        )
        if action is None:
            after = agent.sessions[session_id]
            assert isinstance(after, ConversationState)
            stopped = copy.deepcopy(after)
            stopped.asked_attributes = list(before.asked_attributes)
            stopped.last_asked_attribute = None
            agent.sessions[session_id] = stopped
            continuation = {**continuation, "ask_attribute": None}
        observation = branch.step(continuation)
    session = branch.session_result()
    return ActionOutcome(
        sample_id=str(session["sample_id"]),
        action=action,
        reward=session_reward(session),
        hit=bool(session["hit"]),
        first_hit_turn=session["first_hit_turn"],
        best_rank=session["best_rank"],
    )


def collect_counterfactual_question_states(
    agent: ExperimentalAgent,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[list[QuestionTrainingState], list[dict[str, object]]]:
    """Collect dense legal-action labels along the champion trajectory.

    Each branch changes only the current question. Question branches resume the
    fixed champion at the next absolute turn; stop branches remain stopped. The
    feature state is captured before the current action and contains no target,
    scenario, future answer, or reward fields.
    """

    states: list[QuestionTrainingState] = []
    labels: list[dict[str, object]] = []
    for sample in samples:
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        agent.reset(observation.session_id, sample["user_profile"])
        while not environment.done:
            session_id = observation.session_id
            before = copy.deepcopy(agent.sessions[session_id])
            assert isinstance(before, ConversationState)
            response = agent.respond(
                session_id,
                observation.user_message,
                observation.turn,
                observation.top_k,
            )
            after = agent.sessions[session_id]
            assert isinstance(after, ConversationState)
            feature_state = copy.deepcopy(after)
            feature_state.asked_attributes = list(before.asked_attributes)
            feature_state.last_asked_attribute = before.last_asked_attribute
            query, retrieval_scores = agent.last_runtime_inputs[session_id]
            features = observable_question_features(
                feature_state,
                message=observation.user_message,
                query=query,
                turn=observation.turn,
                retrieval_scores=retrieval_scores,
            )
            legal_actions = legal_question_actions(feature_state)
            champion_action = response.get("ask_attribute")
            if champion_action is not None and not isinstance(champion_action, str):
                champion_action = None
            evaluated_actions = list(legal_actions)
            if champion_action not in evaluated_actions:
                evaluated_actions.append(champion_action)  # type: ignore[arg-type]
            outcomes: dict[QuestionAction, ActionOutcome] = {}
            for action in evaluated_actions:
                outcomes[action] = _branch_outcome(
                    agent,
                    environment,
                    feature_state,
                    response,
                    action,
                )
            states.append(
                QuestionTrainingState(
                    sample_id=str(sample["sample_id"]),
                    turn=observation.turn,
                    features=features,
                    action_rewards={
                        action: outcomes[action].reward for action in legal_actions
                    },
                )
            )
            labels.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "turn": observation.turn,
                    "feature_schema": "observable_question_features_v1",
                    "features": features,
                    "legal_actions": [
                        "stop" if action is None else action for action in legal_actions
                    ],
                    "champion_action": (
                        "stop" if champion_action is None else champion_action
                    ),
                    "outcomes": {
                        "stop" if action is None else action: {
                            "reward": outcome.reward,
                            "hit": outcome.hit,
                            "first_hit_turn": outcome.first_hit_turn,
                            "best_rank": outcome.best_rank,
                        }
                        for action, outcome in outcomes.items()
                    },
                }
            )
            agent.sessions[session_id] = copy.deepcopy(after)
            next_observation = environment.step(response)
            if next_observation is not None:
                observation = next_observation
    return states, labels


def first_action_diagnostics(
    states: list[QuestionTrainingState], labels: list[dict[str, object]]
) -> dict[str, object]:
    action_counts: Counter[str] = Counter()
    advantages: list[float] = []
    stop_advantages: list[float] = []
    oracle_rewards: list[float] = []
    trajectory_rewards: list[float] = []
    for state, label in zip(states, labels, strict=True):
        rewards = state.action_rewards
        best_action = min(
            rewards,
            key=lambda action: (-rewards[action], "stop" if action is None else action),
        )
        action_counts["stop" if best_action is None else best_action] += 1
        oracle_rewards.append(rewards[best_action])
        outcomes = label["outcomes"]
        assert isinstance(outcomes, dict)
        champion_key = str(label["champion_action"])
        champion_payload = outcomes[champion_key]
        assert isinstance(champion_payload, dict)
        champion_reward = float(champion_payload["reward"])
        trajectory_rewards.append(champion_reward)
        advantages.append(rewards[best_action] - champion_reward)
        if None in rewards:
            stop_advantages.append(rewards[None] - champion_reward)
    return {
        "state_count": len(states),
        "oracle_action_counts": dict(sorted(action_counts.items())),
        "mean_oracle_reward": round(statistics.fmean(oracle_rewards), 6),
        "mean_trajectory_action_reward": round(statistics.fmean(trajectory_rewards), 6),
        "mean_oracle_advantage": round(statistics.fmean(advantages), 6),
        "mean_stop_advantage": round(statistics.fmean(stop_advantages), 6),
    }


def behavior_diagnostics(traces: list[dict[str, object]]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    by_session: dict[str, list[str]] = defaultdict(list)
    stopped_sessions: set[str] = set()
    illegal = 0
    for trace in traces:
        action = trace["ask_attribute"]
        name = "stop" if action is None else str(action)
        counts[name] += 1
        session_id = str(trace["session_id"])
        by_session[session_id].append(name)
        legal = trace.get("legal_actions")
        if isinstance(legal, tuple) and action not in legal:
            illegal += 1
        if action is None:
            stopped_sessions.add(session_id)
    repeats = sum(
        left == right and left != "stop"
        for actions in by_session.values()
        for left, right in pairwise(actions)
    )
    transitions = sum(max(0, len(actions) - 1) for actions in by_session.values())
    return {
        "action_distribution": dict(sorted(counts.items())),
        "turn_count": sum(counts.values()),
        "session_count": len(by_session),
        "sessions_with_stop": len(stopped_sessions),
        "stop_rate_per_session": round(
            len(stopped_sessions) / max(1, len(by_session)), 6
        ),
        "repeat_rate": round(repeats / max(1, transitions), 6),
        "illegal_action_count": illegal,
    }
