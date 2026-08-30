from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ghostlab.policy.models import RankedCandidate, RankedCandidates
from ghostlab.retrieval.dense_diversity import (
    embedding_mmr_select,
    view_balanced_select,
)
from ghostlab.retrieval.filters import CoverageAwareFilter
from ghostlab.retrieval.gbdt import GBDTFeatureStore, LambdaMARTModel, RegressionTree
from ghostlab.retrieval.multi_route import merge_candidate_routes
from ghostlab.retrieval.union_features import SOURCE_AWARE_FEATURES, UnionFeatureStore
from ghostlab.runtime.adaptive_components import (
    DiverseDenseResult,
    DiverseDenseTrack,
    OverGeneralityGuidance,
    UnionAwareRanker,
)
from ghostlab.runtime.adaptive_config import (
    AdaptiveHybridConfig,
    DiverseDenseTrackConfig,
    ProactiveGuidanceConfig,
    UnionRankerConfig,
)
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint
from ghostlab.state.v2_view import AdaptiveTurnContext, V2SessionController
from ghostlab.training.adaptive_hybrid import sha256_file


def _catalog(path: Path) -> list[str]:
    rows = [
        {
            "parent_asin": "MATCH",
            "title": "Black waterproof trail shoes",
            "categories": ["Shoes"],
            "features": ["waterproof"],
            "details": {"Color": "Black", "Material": "Rubber"},
            "description": "Waterproof outdoor shoes",
            "store": "A",
            "price": 80,
            "average_rating": 4.8,
            "rating_number": 100,
        },
        {
            "parent_asin": "OVER",
            "title": "Black waterproof premium shoes",
            "categories": ["Shoes"],
            "features": ["waterproof"],
            "details": {"Color": "Black", "Material": "Rubber"},
            "description": "Waterproof outdoor shoes",
            "store": "B",
            "price": 180,
            "average_rating": 4.7,
            "rating_number": 80,
        },
        {
            "parent_asin": "UNKNOWN",
            "title": "Outdoor shoes",
            "categories": ["Shoes"],
            "features": [],
            "details": {},
            "description": "Outdoor footwear",
            "store": "C",
            "price": None,
            "average_rating": None,
            "rating_number": None,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return [row["parent_asin"] for row in rows]


def _context(
    constraints: tuple[StructuredConstraint, ...], *, message: str = "request"
) -> AdaptiveTurnContext:
    state = StateBaselineV2("s", {})
    state.apply_constraints(constraints)
    return V2SessionController(state).snapshot(
        query_text=message, turn=1, current_message=message
    )


def test_frozen_turn_context_contains_one_consistent_projection() -> None:
    state = StateBaselineV2("s", {"preference_tags": ["minimal"]})
    state.observe("black shoes", 1)
    context = V2SessionController(state).snapshot(
        query_text="black shoes",
        turn=1,
        current_message="black shoes",
        supplied_profile_terms=frozenset({"minimal"}),
    )
    assert isinstance(context, AdaptiveTurnContext)
    assert context.session_id == "s"
    assert context.current_message == "black shoes"
    assert context.supplied_profile_terms == frozenset({"minimal"})
    try:
        context.query_text = "mutated"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("turn context must be immutable")


def test_constraint_authority_is_route_independent_and_preserves_unknowns(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _catalog(catalog)
    context = _context(
        (
            StructuredConstraint(
                attribute="budget",
                values=["under $100"],
                strength="hard",
                source_turn=1,
            ),
        )
    )
    result = CoverageAwareFilter(catalog).enforce(["OVER", "UNKNOWN", "MATCH"], context)
    assert result.ranking == ("MATCH", "UNKNOWN")
    assert result.violation_count == 1
    assert any(item.status == "UNKNOWN_METADATA" for item in result.decisions)


def test_known_exclusion_cannot_reappear_after_ranking(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _catalog(catalog)
    context = _context(
        (
            StructuredConstraint(
                attribute="color",
                values=["black"],
                polarity="exclude",
                strength="hard",
                source_turn=1,
            ),
        )
    )
    result = CoverageAwareFilter(catalog).enforce(["MATCH", "OVER", "UNKNOWN"], context)
    assert result.ranking == ("UNKNOWN",)
    assert result.violation_count == 2


def test_open_world_feature_equivalence_and_absence_never_create_false_violation(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    rows = [
        {
            "parent_asin": "GORE",
            "title": "Trail shell",
            "categories": ["Jackets"],
            "features": ["GORE-TEX membrane"],
            "details": {},
            "description": "Outdoor shell",
            "price": 90,
        },
        {
            "parent_asin": "PLAIN",
            "title": "Outdoor shell",
            "categories": ["Jackets"],
            "features": [],
            "details": {},
            "description": "General outdoor layer",
            "price": 70,
        },
    ]
    catalog.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    context = _context(
        (
            StructuredConstraint(
                attribute="feature",
                values=["waterproof"],
                strength="hard",
                source_turn=1,
            ),
        )
    )
    result = CoverageAwareFilter(catalog).enforce(["GORE", "PLAIN"], context)
    assert result.ranking == ("GORE", "PLAIN")
    assert result.violation_count == 0
    statuses = {item.parent_asin: item.status for item in result.decisions}
    assert statuses == {"GORE": "CONFIRMED_MATCH", "PLAIN": "UNKNOWN_METADATA"}


class _BudgetDense:
    def __init__(self, identifiers: list[str]) -> None:
        self.identifiers = identifiers
        self.overload_flags: list[bool] = []

    def search(
        self, view: AdaptiveTurnContext, *, overloaded: bool = False
    ) -> DiverseDenseResult:
        del view
        self.overload_flags.append(overloaded)
        depth = 2 if overloaded else len(self.identifiers)
        selected = self.identifiers[:depth]
        return DiverseDenseResult(
            identifiers=tuple(selected),
            relevance_scores={item: 1.0 for item in selected},
            query_views=("complete_request",),
            elapsed_ms=0.1,
            requested_per_view=depth,
            output_k=depth,
            selection="multiview_max_relevance",
        )


class _IdentitySemantic:
    def rank(self, query: str, ranking: list[str]):
        from ghostlab.runtime.adaptive_components import SemanticRankingResult

        del query
        return SemanticRankingResult(tuple(ranking), False, 0.1, "test_llm")


def test_pre_dense_preview_selects_reduced_budget_and_reaches_pipeline(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    ids = _catalog(catalog)
    dense = _BudgetDense(ids)
    config = AdaptiveHybridConfig(
        browsing=DiverseDenseTrackConfig(
            retrieval_per_view=20,
            output_k=20,
            overload_retrieval_per_view=5,
            overload_output_k=5,
            safe_ranker_backend="deterministic",
            safe_ranker_model_path=None,
            safe_ranker_model_sha256=None,
        ),
        guidance=ProactiveGuidanceConfig(
            overload_min_candidates=2,
            preview_min_candidates=2,
            overload_max_specific_constraints=0,
        ),
        union_ranker=UnionRankerConfig(
            backend="deterministic", model_path=None, model_sha256=None
        ),
    )
    agent = AdaptiveHybridAgent(
        catalog,
        config,
        project_root=tmp_path,
        dense_track=dense,  # type: ignore[arg-type]
        semantic_ranker=_IdentitySemantic(),  # type: ignore[arg-type]
    )
    agent.reset("s", {})
    agent.respond("s", "I'm still exploring shoes", 1, 3)
    trace = agent.traces[-1]
    assert dense.overload_flags == [True]
    assert trace.preview_reason == "preview_overloaded"
    assert trace.dense_output_k == 2
    assert "rank:browsing_safe" in trace.reason_codes
    assert trace.safe_merge_executed
    assert trace.safe_ranker_executed
    assert not trace.normal_union_executed
    assert trace.semantic_decision_reached
    assert not trace.semantic_executed
    assert trace.semantic_backend == "skipped:overload_cutoff"


def test_dense_selectors_are_deterministic_and_mmr_reduces_duplicates() -> None:
    views = {"a": ["a1", "a2", "x"], "b": ["b1", "b2", "x"]}
    relevance = {"a1": 1.0, "a2": 0.9, "b1": 0.95, "b2": 0.85, "x": 0.8}
    assert view_balanced_select(views, relevance, output_k=4) == [
        "a1",
        "b1",
        "a2",
        "b2",
    ]
    embeddings = {
        "a1": np.asarray([1.0, 0.0]),
        "a2": np.asarray([0.999, 0.001]),
        "b1": np.asarray([0.0, 1.0]),
    }
    selected = embedding_mmr_select(
        ["a1", "a2", "b1"],
        relevance,
        embeddings,
        output_k=2,
        relevance_weight=0.6,
    )
    assert selected == ["a1", "b1"]


def test_source_aware_union_features_distinguish_route_membership_and_missingness(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _catalog(catalog)
    pool = merge_candidate_routes(
        route="browsing",
        keyword_ids=["MATCH"],
        category_hits=(),
        vector_ids=["OVER"],
        limit=3,
        keyword_weight=0.2,
        category_weight=0.1,
        vector_weight=0.7,
    )
    store = UnionFeatureStore(GBDTFeatureStore(catalog))
    matrix = store.matrix("shoes", pool, SOURCE_AWARE_FEATURES)
    names = {name: index for index, name in enumerate(SOURCE_AWARE_FEATURES)}
    rows = {item.parent_asin: index for index, item in enumerate(pool.candidates)}
    assert matrix.shape == (2, len(SOURCE_AWARE_FEATURES))
    assert matrix[rows["MATCH"], names["keyword_member"]] == 1.0
    assert matrix[rows["MATCH"], names["vector_missing"]] == 1.0
    assert matrix[rows["OVER"], names["vector_member"]] == 1.0
    assert matrix[rows["OVER"], names["keyword_missing"]] == 1.0
    assert np.isfinite(matrix[:, names["source_count"]]).all()
    assert np.array_equal(
        matrix, store.matrix("shoes", pool, SOURCE_AWARE_FEATURES), equal_nan=True
    )


def test_buying_residual_prevents_learned_model_from_overturning_sparse_head(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _catalog(catalog)
    model = LambdaMARTModel(
        candidate_id="adversarial_source_model",
        feature_names=("keyword_member",),
        trees=(
            RegressionTree(
                children_left=(1, 1, 2),
                children_right=(2, 1, 2),
                feature=(0, -2, -2),
                threshold=(0.5, -2.0, -2.0),
                missing_go_to_left=(True, True, True),
                value=(0.0, 10.0, 0.0),
            ),
        ),
        learning_rate=1.0,
        best_iteration=1,
        training_groups=1,
        training_rows=2,
        seed=1,
    )
    model_path = tmp_path / "model.json"
    model.save(model_path)
    pool = merge_candidate_routes(
        route="buying",
        keyword_ids=["MATCH"],
        category_hits=(),
        vector_ids=["OVER"],
        limit=2,
        keyword_weight=0.9,
        category_weight=0.05,
        vector_weight=0.05,
    )
    residual = UnionAwareRanker(
        catalog,
        UnionRankerConfig(
            backend="gbdt",
            model_path="model.json",
            model_sha256=sha256_file(model_path),
            buying_mode="sparse_dominant_residual",
            buying_residual_weight=0.25,
        ),
        project_root=tmp_path,
    )
    direct = UnionAwareRanker(
        catalog,
        UnionRankerConfig(
            backend="gbdt",
            model_path="model.json",
            model_sha256=sha256_file(model_path),
            buying_mode="direct",
        ),
        project_root=tmp_path,
    )
    assert (
        residual.rank("shoes", pool, positive_constraints={}, negative_constraints={})[
            0
        ]
        == "MATCH"
    )
    assert (
        direct.rank("shoes", pool, positive_constraints={}, negative_constraints={})[0]
        == "OVER"
    )


class _FakeDenseIndex:
    def __init__(self) -> None:
        self.identifiers = ["a", "b", "c"]
        self.embeddings = np.asarray(
            [[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float32
        )

    def search(self, query: str, limit: int) -> RankedCandidates:
        order = ["a", "b", "c"] if "request" in query else ["c", "b", "a"]
        values = order[:limit]
        return RankedCandidates(
            items=tuple(
                RankedCandidate(
                    parent_asin=item,
                    route="dense",
                    rank=index,
                    raw_score=1.0 - index * 0.1,
                    normalized_score=1.0 - index * 0.1,
                )
                for index, item in enumerate(values, start=1)
            ),
            route="dense",
            requested_k=limit,
            elapsed_ms=0.1,
        )


def test_dense_track_exposes_view_evidence_and_selected_strategy(
    tmp_path: Path,
) -> None:
    context = AdaptiveTurnContext(
        query_text="complete request",
        active_constraints=(),
        intent_epoch=0,
        shown_ids=frozenset(),
        asked_attributes=(),
        no_preference_attributes=frozenset(),
        turn=1,
        session_id="s",
        current_message="complete request",
    )
    track = DiverseDenseTrack(
        tmp_path / "missing.jsonl",
        DiverseDenseTrackConfig(
            retrieval_per_view=10,
            output_k=10,
            overload_retrieval_per_view=5,
            overload_output_k=5,
            selection="embedding_mmr",
        ),
        project_root=tmp_path,
        index=_FakeDenseIndex(),  # type: ignore[arg-type]
    )
    result = track.search(context, overloaded=True)
    assert result.selection == "embedding_mmr"
    assert result.requested_per_view == 5
    assert result.output_k == 5
    assert result.per_view_ranks["complete_request"]["a"] == 1


def test_profile_query_view_is_optional_and_observable(tmp_path: Path) -> None:
    context = AdaptiveTurnContext(
        query_text="complete request",
        active_constraints=(),
        intent_epoch=0,
        shown_ids=frozenset(),
        asked_attributes=(),
        no_preference_attributes=frozenset(),
        turn=1,
        session_id="s",
        current_message="complete request",
        supplied_profile_terms=frozenset({"minimal", "breathable"}),
    )
    track = DiverseDenseTrack(
        tmp_path / "missing.jsonl",
        DiverseDenseTrackConfig(
            retrieval_per_view=10,
            output_k=10,
            overload_retrieval_per_view=5,
            overload_output_k=5,
            profile_query_view_enabled=True,
        ),
        project_root=tmp_path,
        index=_FakeDenseIndex(),  # type: ignore[arg-type]
    )
    result = track.search(context)
    assert "profile_context" in result.query_views


def test_profile_known_attribute_is_removed_from_question_race(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    ids = _catalog(catalog)
    guidance = OverGeneralityGuidance(
        catalog,
        ProactiveGuidanceConfig(
            broad_discovery_turns=0,
            overload_min_candidates=100,
            preview_min_candidates=30,
        ),
    )
    context = AdaptiveTurnContext(
        query_text="shoes",
        active_constraints=(),
        intent_epoch=0,
        shown_ids=frozenset(),
        asked_attributes=(),
        no_preference_attributes=frozenset(),
        turn=3,
        session_id="s",
        current_message="show me more",
    )
    control = guidance.decide(
        context,
        ids,
        turn=3,
        message="show me more",
        overloaded=False,
    )
    assert control.ask_attribute is not None
    suppressed = guidance.decide(
        context,
        ids,
        turn=3,
        message="show me more",
        overloaded=False,
        profile_known_attributes=frozenset({control.ask_attribute}),
    )
    assert suppressed.ask_attribute != control.ask_attribute
