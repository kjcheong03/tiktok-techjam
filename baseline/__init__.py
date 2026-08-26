"""Minimal retrieval and conversation-state baselines.

Imports stay lazy so state and policy helpers do not require optional retrieval
dependencies such as NumPy and sentence-transformers.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import BaselineAgent
    from .retrieval import DenseRetriever, KeywordRetriever

__all__ = ["BaselineAgent", "DenseRetriever", "KeywordRetriever", "reciprocal_rank_fusion"]


def __getattr__(name: str) -> Any:
    if name == "BaselineAgent":
        from .agent import BaselineAgent

        return BaselineAgent
    if name in {"DenseRetriever", "KeywordRetriever", "reciprocal_rank_fusion"}:
        from . import retrieval

        return getattr(retrieval, name)
    raise AttributeError(name)
