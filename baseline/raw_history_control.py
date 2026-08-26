from __future__ import annotations

from .question_policy import QuestionPolicy, fixed_other
from .retrieval import KeywordRetriever
from .state import fixed_question_for_turn


class RawHistoryNoManagedStateAgent:
    """Keyword control that keeps raw history without managed constraints."""

    def __init__(self, keyword: KeywordRetriever, question_policy: QuestionPolicy) -> None:
        self.keyword = keyword
        self.question_policy = question_policy
        self.messages: dict[str, list[str]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.keyword.reset(session_id, user_profile)
        self.messages[session_id] = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.messages:
            raise RuntimeError("reset must be called before respond")

        history = self.messages[session_id]
        history.append(user_message)
        query = ". ".join(history)
        ranked = self.keyword.search(session_id, query, turn, 200)

        if self.question_policy is fixed_other:
            ask_attribute = self.question_policy(None, turn)
        else:
            ask_attribute = fixed_question_for_turn(turn)
        if ask_attribute is None:
            message = "Here are the closest matches based on what you have shared."
        else:
            message = f"Do you have a preference for {ask_attribute.replace('_', ' ')}?"
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": identifier} for identifier in ranked[:top_k]
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


__all__ = ["RawHistoryNoManagedStateAgent"]
