from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable

from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.eig_questions import RewardVOICalibration
from ghostlab.research.counterfactual import ActionOutcome


def fit_reward_voi_calibration(
    outcomes: Iterable[ActionOutcome], *, shrinkage: float = 10.0
) -> RewardVOICalibration:
    """Fit fold-local action offsets relative to stop reward.

    The caller must supply outer-training outcomes only. The result contains no
    session identifiers or target-derived runtime fields.
    """

    if shrinkage < 0.0:
        raise ValueError("shrinkage must be non-negative")
    by_session: dict[str, dict[AskAttribute | None, float]] = defaultdict(dict)
    for outcome in outcomes:
        by_session[outcome.sample_id][outcome.action] = outcome.reward
    deltas: dict[AskAttribute, list[float]] = defaultdict(list)
    for rewards in by_session.values():
        if None not in rewards:
            continue
        for action, reward in rewards.items():
            if action is not None:
                deltas[action].append(reward - rewards[None])
    adjustments = {
        action: statistics.fmean(values) * len(values) / (len(values) + shrinkage)
        for action, values in sorted(deltas.items())
    }
    return RewardVOICalibration(adjustments, len(by_session), shrinkage)
