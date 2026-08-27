from __future__ import annotations

import math

MAX_TURNS = 10
TOP_K = 10


def efficiency_at_turn(turn: int) -> float:
    """Return the organizer efficiency term for a first hit at ``turn``."""
    if not 1 <= turn <= MAX_TURNS + 1:
        raise ValueError("turn must be between 1 and 11")
    return max(0.0, min(1.0, (11.0 - turn) / 10.0))


def terminal_session_reward(rank: int | None, turn: int) -> float:
    """Exact organizer per-session reward for a terminal result at one turn.

    A target outside Top-10 does not terminate the conversation, so its immediate
    terminal reward is zero. Future outcomes are deliberately not estimated here.
    """
    efficiency_at_turn(turn)  # validates even for a miss
    if rank is None or rank > TOP_K:
        return 0.0
    if rank <= 0:
        raise ValueError("rank must be positive")
    return 0.50 + 0.30 / rank + 0.20 * efficiency_at_turn(turn)


def swap_reward_delta(left_rank: int, right_rank: int, turn: int) -> float:
    """Absolute organizer-reward change when the target swaps two ranks."""
    if left_rank <= 0 or right_rank <= 0:
        raise ValueError("ranks must be positive")
    return abs(
        terminal_session_reward(left_rank, turn)
        - terminal_session_reward(right_rank, turn)
    )


def mean_terminal_reward(
    target_ranks: list[int] | tuple[int, ...],
    turns: list[int] | tuple[int, ...],
) -> float:
    if len(target_ranks) != len(turns) or not target_ranks:
        raise ValueError("target ranks and turns must be non-empty and aligned")
    return math.fsum(
        terminal_session_reward(rank, turn)
        for rank, turn in zip(target_ranks, turns, strict=True)
    ) / len(turns)
