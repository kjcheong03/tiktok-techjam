from __future__ import annotations

from pathlib import Path
from typing import cast

from baseline.retrieval import KeywordRetriever
from baseline.state import fixed_question_for_turn
from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.models import TechniqueConfig
from ghostlab.retrieval.learned import (
    FEATURE_NAMES,
    CandidateFeatureStore,
    LearnedLinearReranker,
    LinearRerankerModel,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.memory import ConversationState


class CompiledKeywordAgent:
    """Minimal compiled runtime for validated keyword policies."""

    def __init__(
        self,
        catalog_path: str | Path,
        catalog_ids: set[str],
        techniques: TechniqueConfig,
    ) -> None:
        if techniques.retrieval_route != "keyword":
            raise ValueError("compiled keyword runtime received a non-keyword route")
        self.keyword = KeywordRetriever(catalog_path)
        self.sparse = (
            SparseIndex(catalog_path)
            if techniques.sparse_field_weights is not None
            else None
        )
        self.quality = (
            CatalogQualityReranker(catalog_path)
            if techniques.quality_prior_weight > 0.0
            else None
        )
        self.learned = None
        if techniques.reranker == "learned_linear":
            assert techniques.learned_weights is not None
            model = LinearRerankerModel(
                weights=techniques.learned_weights,
                l2=techniques.learned_l2,
                training_pairs=techniques.learned_training_pairs,
            )
            enabled_features = tuple(
                name
                for name, weight in zip(
                    FEATURE_NAMES, techniques.learned_weights, strict=True
                )
                if weight != 0.0
            )
            self.learned = LearnedLinearReranker(
                CandidateFeatureStore(
                    catalog_path,
                    enabled_features=enabled_features,
                    quality=self.quality.quality if self.quality is not None else None,
                ),
                model,
            )
        self.catalog_ids = catalog_ids
        self.techniques = techniques
        self.sessions: dict[str, ConversationState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.keyword.reset(session_id, user_profile)
        self.sessions[session_id] = ConversationState(
            session_id,
            user_profile,
            multi_value=True,
            negative_evidence=self.techniques.negative_evidence,
            provenance_enabled=self.techniques.provenance,
            override_invalidation=self.techniques.override_invalidation,
        )

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        state = self.sessions[session_id]
        state.observe(user_message, turn)
        query = (
            ". ".join(state.messages)
            if self.techniques.state_mode == "raw_history"
            else state.build_query(
                compressed=self.techniques.state_mode == "compressed"
            )
        )
        question_policy = self.techniques.question_policy
        question: AskAttribute | None
        if question_policy == "other_always":
            question = "other"
        elif question_policy == "fixed":
            question = cast(AskAttribute | None, fixed_question_for_turn(turn))
        elif question_policy == "none":
            question = None
        elif question_policy == "sequence":
            order = self.techniques.question_order
            question = order[turn - 1] if turn <= len(order) else None
        else:
            question = cast(AskAttribute | None, state.choose_question())
        if self.sparse is None or self.techniques.sparse_field_weights is None:
            ranked = self.keyword.search(
                session_id, query, turn, self.techniques.retrieval_k
            )
        else:
            ranked = [
                item.parent_asin
                for item in self.sparse.search(
                    query,
                    self.techniques.retrieval_k,
                    self.techniques.sparse_field_weights,
                ).items
            ]
        if self.quality is not None:
            ranked = self.quality.rerank(
                ranked,
                weight=self.techniques.quality_prior_weight,
                rerank_k=self.techniques.rerank_k,
            )
        if self.learned is not None:
            ranked = self.learned.rerank(
                query, ranked, rerank_k=self.techniques.rerank_k
            )
        message = (
            "Here are the closest matches based on what you have shared."
            if question is None
            else f"Do you have a preference for {question.replace('_', ' ')}?"
        )
        return normalize_response(
            {
                "message": message,
                "ask_attribute": question,
                "recommendations": ranked,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            },
            self.catalog_ids,
            top_k,
        )
