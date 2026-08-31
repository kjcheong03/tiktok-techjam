from __future__ import annotations

from ghostlab.retrieval.dense_query_views import build_dense_query_views
from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint
from ghostlab.state.v2_view import V2SessionController


def test_dense_views_use_active_positive_state_without_exclusions() -> None:
    state = StateBaselineV2("session", {})
    state.apply_constraints(
        [
            StructuredConstraint(attribute="category", values=["clothing"]),
            StructuredConstraint(attribute="use_case", values=["summer wedding"]),
            StructuredConstraint(attribute="feature", values=["breathable"]),
            StructuredConstraint(attribute="style", values=["smart casual"]),
            StructuredConstraint(
                attribute="style", values=["formal"], polarity="exclude"
            ),
        ]
    )
    view = V2SessionController(state).snapshot(
        query_text="comfortable summer wedding clothing, not formal", turn=2
    )

    queries = build_dense_query_views(view)

    assert [(item.name, item.query_text) for item in queries] == [
        ("complete_request", "comfortable summer wedding clothing, not formal"),
        ("use_case", "clothing. summer wedding"),
        ("features_style", "clothing. breathable. smart casual"),
    ]
    assert all(item.query_text != "formal" for item in queries[1:])
