from __future__ import annotations

from typing import Literal

from .retrieval import DenseRetriever, KeywordRetriever, reciprocal_rank_fusion
from .state import SessionState, fixed_question_for_turn

RetrievalMode = Literal["keyword", "dense", "hybrid"]


class BaselineAgent:
    """Small baseline agent with independently switchable retrieval and state."""

    def __init__(
        self,
        *,
        mode: RetrievalMode,
        stateful: bool,
        keyword: KeywordRetriever,
        dense: DenseRetriever | None,
        retrieval_k: int = 200,
        rrf_constant: int = 60,
    ) -> None:
        if mode in {"dense", "hybrid"} and dense is None:
            raise ValueError(f"{mode} mode requires a dense retriever")
        self.mode = mode
        self.stateful = stateful
        self.keyword = keyword
        self.dense = dense
        self.retrieval_k = retrieval_k
        self.rrf_constant = rrf_constant
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.keyword.reset(session_id, user_profile)
        self.sessions[session_id] = SessionState(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")

        state = self.sessions[session_id]
        if self.stateful:
            state.observe(user_message, turn)
            query = state.build_query()
            ask_attribute = state.choose_question()
        else:
            query = user_message
            ask_attribute = fixed_question_for_turn(turn)

        if self.mode == "keyword":
            ranked = self.keyword.search(session_id, query, turn, self.retrieval_k)
        elif self.mode == "dense":
            assert self.dense is not None
            ranked = self.dense.search(query, self.retrieval_k)
        else:
            assert self.dense is not None
            sparse = self.keyword.search(session_id, query, turn, self.retrieval_k)
            semantic = self.dense.search(query, self.retrieval_k)
            ranked = reciprocal_rank_fusion(
                [sparse, semantic],
                rank_constant=self.rrf_constant,
                limit=self.retrieval_k,
            )

        if ask_attribute is None:
            message = "Here are the closest matches based on what you have shared."
        else:
            label = ask_attribute.replace("_", " ")
            message = f"Do you have a preference for {label}?"
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": identifier} for identifier in ranked[:top_k]
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
