from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostlab.retrieval.category import CategoryCandidateIndex
from ghostlab.retrieval.multi_route import merge_candidate_routes
from ghostlab.runtime.adaptive_components import (
    BoundedLocalLLMSemanticRanker,
    ConflictSafeContextAdapter,
    DiverseDenseResult,
    DualTrackRouter,
    SemanticRankingResult,
)
from ghostlab.runtime.adaptive_config import (
    AdaptiveExtensionsConfig,
    AdaptiveHybridConfig,
    DiverseDenseTrackConfig,
    ProactiveGuidanceConfig,
    RuntimeAdaptationConfig,
    UnionRankerConfig,
)
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint
from ghostlab.state.v2_view import V2SessionController


def _catalog(path: Path, count: int = 24) -> list[str]:
    identifiers = [f"P{index:03d}" for index in range(count)]
    rows = []
    for index, identifier in enumerate(identifiers):
        category = "Running Shoes" if index < count // 2 else "Summer Dresses"
        color = "Black" if index % 2 == 0 else "Blue"
        rows.append(
            {
                "parent_asin": identifier,
                "title": f"{color} {category} {index}",
                "categories": ["Clothing Shoes & Jewelry", category],
                "features": ["comfortable", "breathable"],
                "details": {"Color": color, "Material": "Cotton"},
                "description": f"A {color.casefold()} item for everyday use",
                "store": "Test Store",
                "price": 20 + index,
                "average_rating": 4.0,
                "rating_number": 10 + index,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return identifiers


class DenseStub:
    def __init__(self, identifiers: list[str]) -> None:
        self.identifiers = identifiers
        self.calls = 0

    def search(self, view: object) -> DiverseDenseResult:
        del view
        self.calls += 1
        return DiverseDenseResult(
            identifiers=tuple(self.identifiers),
            relevance_scores={
                item: 1.0 - index / max(1, len(self.identifiers))
                for index, item in enumerate(self.identifiers)
            },
            query_views=("complete_request", "features_style"),
            elapsed_ms=1.0,
        )


class SemanticStub:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def rank(self, query: str, ranking: list[str]) -> SemanticRankingResult:
        del query
        self.calls += 1
        if self.fail:
            raise RuntimeError("semantic failure")
        head = list(reversed(ranking[:3]))
        result = [*head, *ranking[3:]]
        return SemanticRankingResult(tuple(result), result != ranking, 1.0, "stub_llm")


def _config(
    *,
    overload_min_candidates: int = 1000,
    extensions: AdaptiveExtensionsConfig | None = None,
) -> AdaptiveHybridConfig:
    return AdaptiveHybridConfig(
        browsing=DiverseDenseTrackConfig(
            safe_ranker_backend="deterministic",
            safe_ranker_model_path=None,
            safe_ranker_model_sha256=None,
        ),
        union_ranker=UnionRankerConfig(
            backend="deterministic", model_path=None, model_sha256=None
        ),
        guidance=ProactiveGuidanceConfig(
            overload_min_candidates=overload_min_candidates,
            overload_max_specific_constraints=0,
        ),
        runtime_adaptation=RuntimeAdaptationConfig(
            maximum_explicit_constraints_for_profile=2
        ),
        extensions=extensions or AdaptiveExtensionsConfig(),
    )


def _agent(
    tmp_path: Path,
    *,
    overload_min_candidates: int = 1000,
    semantic_fail: bool = False,
) -> tuple[AdaptiveHybridAgent, DenseStub, SemanticStub, list[str]]:
    catalog = tmp_path / "catalog.jsonl"
    identifiers = _catalog(catalog)
    dense = DenseStub(list(reversed(identifiers)))
    semantic = SemanticStub(fail=semantic_fail)
    agent = AdaptiveHybridAgent(
        catalog,
        _config(overload_min_candidates=overload_min_candidates),
        project_root=tmp_path,
        dense_track=dense,  # type: ignore[arg-type]
        semantic_ranker=semantic,  # type: ignore[arg-type]
    )
    return agent, dense, semantic, identifiers


def test_config_has_no_disabled_submission_slot() -> None:
    with pytest.raises(ValueError):
        AdaptiveHybridConfig.model_validate(
            {"submission_eligible": False, "architecture": "adaptive_hybrid_1a_3b_v1"}
        )
    with pytest.raises(ValueError):
        AdaptiveHybridConfig.model_validate({"browsing": {"component": "off"}})


def test_router_reaches_both_tracks_from_observable_state() -> None:
    router = DualTrackRouter(AdaptiveHybridConfig().router)
    state = StateBaselineV2("s", {})
    state.observe("I'm looking for shoes, but I'm still exploring.", 1)
    view = V2SessionController(state).snapshot(query_text="shoes", turn=1)
    assert router.decide(view, state.messages[-1]).route == "browsing"

    state.observe("A key requirement is: black.", 2)
    view = V2SessionController(state).snapshot(query_text="black shoes", turn=2)
    assert router.decide(view, state.messages[-1]).route == "buying"


def test_category_route_independently_adds_candidates(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _catalog(catalog)
    hits = CategoryCandidateIndex(catalog).search(
        "running shoes", limit=10, preferred_categories=("running shoes",)
    )
    pool = merge_candidate_routes(
        route="buying",
        keyword_ids=("P020",),
        category_hits=hits,
        vector_ids=(),
        limit=20,
        keyword_weight=0.75,
        category_weight=0.25,
        vector_weight=0.0,
    )
    assert any(item.sources == frozenset({"category"}) for item in pool.candidates)


def test_complete_runtime_executes_buying_and_browsing(tmp_path: Path) -> None:
    agent, dense, semantic, _ = _agent(tmp_path)
    agent.reset("buy", {"preference_tags": ["comfort"]})
    buying = agent.respond(
        "buy", "I'm looking for running shoes. A key requirement is: black.", 1, 10
    )
    assert len(buying["recommendations"]) == 10
    assert agent.traces[-1].route == "buying"
    assert agent.traces[-1].contribution_counts["keyword"] > 0
    assert dense.calls == 1
    assert agent.traces[-1].contribution_counts["category"] > 0
    assert agent.traces[-1].contribution_counts["vector"] > 0
    assert semantic.calls == 0
    assert agent.traces[-1].semantic_backend == "skipped:high_confidence_buying"

    agent.reset("browse", {"preference_tags": ["comfort"]})
    browsing = agent.respond(
        "browse", "I'm looking for running shoes, but I'm still exploring.", 1, 10
    )
    assert len(browsing["recommendations"]) == 10
    assert agent.traces[-1].route == "browsing"
    assert agent.traces[-1].contribution_counts["vector"] > 0
    assert agent.traces[-1].contribution_counts["keyword"] > 0
    assert agent.traces[-1].contribution_counts["category"] > 0
    assert dense.calls == 2
    assert semantic.calls == 1
    assert agent.traces[-1].semantic_backend == "stub_llm"
    assert {trace.semantic_activation_reason for trace in agent.traces} == {
        "high_confidence_buying",
        "browsing_semantic_retrieval",
    }


def test_promotable_optional_rankers_execute_at_the_declared_hook(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    identifiers = _catalog(catalog)
    dense = DenseStub(list(reversed(identifiers)))
    semantic = SemanticStub()
    config = _config(
        extensions=AdaptiveExtensionsConfig(
            quality_prior_weight=0.2,
            query_prf_enabled=True,
            facet_diversity_enabled=True,
        )
    )
    agent = AdaptiveHybridAgent(
        catalog,
        config,
        project_root=tmp_path,
        dense_track=dense,  # type: ignore[arg-type]
        semantic_ranker=semantic,  # type: ignore[arg-type]
    )
    agent.reset("optional", {})
    agent.respond(
        "optional",
        "I'm looking for running shoes, but I'm still exploring.",
        1,
        10,
    )
    reasons = agent.traces[-1].reason_codes
    assert "optional:prior.quality" in reasons
    assert any(item.startswith("optional:query.catalog_prf.v1:") for item in reasons)
    assert any(
        item.startswith("optional:ranking.facet_diversity.v1:") for item in reasons
    )
    assert reasons.index("rank:union_aware") < reasons.index("optional:prior.quality")
    assert reasons.index("optional:prior.quality") < reasons.index("semantic:stub_llm")


def test_overload_caps_retrieval_but_uses_llm_for_browsing_semantics(
    tmp_path: Path,
) -> None:
    agent, _, semantic, _ = _agent(tmp_path, overload_min_candidates=10)
    agent.reset("s", {})
    response = agent.respond(
        "s", "I'm looking for running shoes, but I'm still exploring.", 1, 10
    )
    trace = agent.traces[-1]
    assert trace.overloaded is True
    assert semantic.calls == 1
    assert len(response["recommendations"]) == 10
    assert response["ask_attribute"] is not None
    assert "overload:cutoff" in trace.reason_codes
    assert "rank:union_aware" in trace.reason_codes
    assert "semantic:stub_llm" in trace.reason_codes
    assert all(
        trace.contribution_counts[source] > 0
        for source in ("keyword", "category", "vector")
    )
    assert agent.candidate_snapshots[-1].overloaded is True


def test_semantic_failure_uses_complete_precision_fallback(tmp_path: Path) -> None:
    agent, _, _, _ = _agent(tmp_path, semantic_fail=True)
    agent.reset("s", {})
    response = agent.respond(
        "s", "I'm looking for running shoes, but I'm still exploring.", 1, 10
    )
    assert len(response["recommendations"]) == 10
    assert agent.traces[-1].fallback_reason == "adaptive:RuntimeError"
    assert "fallback:complete_precision" in agent.traces[-1].reason_codes


def test_overload_semantic_failure_uses_complete_precision_fallback(
    tmp_path: Path,
) -> None:
    agent, _, semantic, _ = _agent(
        tmp_path, overload_min_candidates=10, semantic_fail=True
    )
    agent.reset("s", {})
    response = agent.respond(
        "s", "I'm looking for running shoes, but I'm still exploring.", 1, 10
    )
    assert len(response["recommendations"]) == 10
    assert semantic.calls == 1
    assert agent.traces[-1].fallback_reason == "adaptive:RuntimeError"
    assert "fallback:complete_precision" in agent.traces[-1].reason_codes


@pytest.mark.parametrize(
    ("component", "expected_reason"),
    (("router", "router:RuntimeError"), ("guidance", "guidance:RuntimeError")),
)
def test_control_failure_uses_complete_precision_fallback(
    tmp_path: Path, component: str, expected_reason: str
) -> None:
    agent, _, _, _ = _agent(tmp_path)

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("control failure")

    if component == "router":
        agent.router.decide = fail  # type: ignore[method-assign,assignment]
    else:
        agent.guidance.decide = fail  # type: ignore[method-assign,assignment]
    agent.reset("s", {})
    response = agent.respond(
        "s", "I'm looking for running shoes. A key requirement is: black.", 1, 10
    )
    assert response["recommendations"]
    assert agent.traces[-1].fallback_reason == expected_reason
    assert "fallback:complete_precision" in agent.traces[-1].reason_codes


def test_only_normalized_action_is_atomically_committed(tmp_path: Path) -> None:
    agent, _, _, _ = _agent(tmp_path)
    agent.reset("s", {})
    response = agent.respond(
        "s", "I'm looking for running shoes. A key requirement is: black.", 1, 3
    )
    session = agent.sessions["s"]
    expected = tuple(item["parent_asin"] for item in response["recommendations"])
    assert session.action_history[-1].shown_products == expected
    assert session.controller._shown_ids == set(expected)
    assert session.action_history[-1].asked_attribute == response["ask_attribute"]


def test_profile_conflict_is_suppressed() -> None:
    adapter = ConflictSafeContextAdapter(RuntimeAdaptationConfig())
    state = StateBaselineV2("s", {"preference_tags": ["formal", "comfort"]})
    state.apply_constraints(
        [
            StructuredConstraint(
                attribute="style",
                values=["formal"],
                polarity="exclude",
                source_turn=1,
                source_text="not formal",
            )
        ]
    )
    context = adapter.distil(state)
    assert context.active is False
    assert context.reason == "explicit_conflict"


def test_committed_profile_overlay_is_consumed_on_next_turn(tmp_path: Path) -> None:
    agent, _, _, _ = _agent(tmp_path)
    agent.reset("s", {})
    agent.respond(
        "s", "I'm looking for shoes. A key requirement is: lightweight.", 1, 10
    )
    update = agent.sessions["s"].profile_update
    assert update is not None
    assert "lightweight" in update.values
    assert update.confidence == 1.0
    assert update.provenance == "explicit_session_evidence"
    assert agent.profile_update("s") == update
    agent.respond("s", "Please refine those options.", 2, 10)
    assert agent.traces[-1].profile_active is True


def test_history_filter_never_restores_already_shown_products(tmp_path: Path) -> None:
    agent, _, _, _ = _agent(tmp_path)
    agent.reset("s", {})
    first = agent.respond(
        "s", "I'm looking for running shoes. A key requirement is: black.", 1, 10
    )
    second = agent.respond("s", "Please refine those options.", 2, 10)
    first_ids = {item["parent_asin"] for item in first["recommendations"]}
    second_ids = {item["parent_asin"] for item in second["recommendations"]}
    assert first_ids.isdisjoint(second_ids)
    assert len(second["recommendations"]) <= 10


def test_live_adaptive_reducer_emits_explicit_profile_conflict(tmp_path: Path) -> None:
    agent, _, _, _ = _agent(tmp_path)
    agent.reset("s", {"preference_tags": ["formal", "comfort"]})
    agent.respond(
        "s",
        "I'm looking for dresses, but I'm still exploring and not formal.",
        1,
        10,
    )
    exclusions = agent.sessions["s"].state.constraint_values(polarity="exclude")
    assert exclusions == ["formal"]
    assert agent.traces[-1].profile_reason == "explicit_conflict"


class _PairRankerStub:
    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int, weight: float
    ) -> list[str]:
        del query, rerank_k, weight
        return [*reversed(ranking)]


class _CausalScorerStub:
    def scores(self, query: str, ranking: list[str]) -> tuple[float, ...]:
        del query
        return tuple(float(index) for index, _ in enumerate(ranking))


def test_bounded_semantic_stage_preserves_membership(tmp_path: Path) -> None:
    ranker = BoundedLocalLLMSemanticRanker(
        tmp_path / "missing.jsonl",
        AdaptiveHybridConfig().semantic_ranker,
        project_root=tmp_path,
        reranker=_PairRankerStub(),  # type: ignore[arg-type]
    )
    result = ranker.rank("query", ["a", "b", "c"])
    assert result.ranking == ("c", "b", "a")
    assert result.changed is True
    assert result.backend == "fallback_minilm_cross_encoder"


def test_actual_causal_llm_stage_is_primary(tmp_path: Path) -> None:
    ranker = BoundedLocalLLMSemanticRanker(
        tmp_path / "missing.jsonl",
        AdaptiveHybridConfig().semantic_ranker,
        project_root=tmp_path,
        llm_scorer=_CausalScorerStub(),
    )
    result = ranker.rank("query", ["a", "b", "c"])
    assert result.ranking == ("c", "b", "a")
    assert result.backend == "qwen_causal_relevance"
