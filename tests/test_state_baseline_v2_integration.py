from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ghostlab.research.technique_suite import build_suite_agent, load_suite_config
from ghostlab.runtime.unified_experimental import ExperimentalAgent
from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint

ROOT = Path(__file__).resolve().parents[1]


def evidence(
    attribute: str,
    values: list[str],
    turn: int,
    text: str,
    **kwargs: str,
) -> StructuredConstraint:
    return StructuredConstraint(
        attribute=attribute,  # type: ignore[arg-type]
        values=values,
        source_turn=turn,
        source_text=text,
        **kwargs,
    )


class SequenceKeywordRetriever:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def reset(self, session_id: str, user_profile: dict) -> None:
        del session_id, user_profile

    def search(self, session_id: str, query: str, turn: int, limit: int) -> list[str]:
        del session_id, query, turn
        return self.ranking[:limit]


def write_catalog(path: Path) -> None:
    rows = [
        {
            "parent_asin": identifier,
            "title": f"product {identifier}",
            "categories": ["products"],
        }
        for identifier in ("A", "B", "C", "D")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_structured_contract_and_memory_adapter_preserve_fields() -> None:
    constraint = StructuredConstraint(
        attribute="budget",
        values=[80],  # type: ignore[list-item]
        strength="hard",
        operator="at_most",
        source_turn=4,
        source_text="It must be under $80",
    )
    state = StateBaselineV2("session", {})
    state.apply_constraints([constraint])

    assert constraint.values == ["80"]
    assert constraint.strength == "hard"
    assert constraint.operator == "at_most"
    assert state.active_values()[0].value == "80"


def test_compatible_values_and_no_preference_remain_retrievable() -> None:
    state = StateBaselineV2("session", {})
    state.last_asked_attribute = "other"
    state.observe(
        "For that, what matters is: 96% Nylon, 4% Spandex; Pull-On closure.",
        2,
    )
    state.last_asked_attribute = "other"
    state.observe("I don't have an additional preference for other.", 3)

    assert state.constraint_values("other") == [
        "96% nylon, 4% spandex",
        "pull-on closure",
    ]
    assert state.build_state_query() == "96% nylon, 4% spandex. pull-on closure"


def test_targeted_and_ambiguous_corrections_are_auditable() -> None:
    state = StateBaselineV2("session", {})
    state.observe("I'm looking for shoes. black.", 1)
    state.observe("A key requirement is: leather; under $80.", 1)
    state.observe(
        "Actually, ignore my earlier preference. What I need is: navy.",
        2,
    )

    colors = [item for item in state.constraints if item.attribute == "color"]
    assert [item.values for item in colors] == [["black"], ["navy"]]
    assert [item.status for item in colors] == ["superseded", "active"]
    assert "leather" in state.build_state_query()
    assert "under $80" in state.build_state_query()
    assert state.intent_epoch == 1

    before = state.build_state_query()
    state.observe(
        "Actually, ignore my earlier preference. What I need is: something.",
        3,
    )
    assert state.build_state_query() == before
    assert state.intent_epoch == 1


def test_coverage_adaptive_query_matches_teammate_rule() -> None:
    state = StateBaselineV2("session", {})
    messages = [
        "I'm looking for shoes. black.",
        "Actually, ignore my earlier preference. What I need is: navy.",
    ]
    for turn, message in enumerate(messages, 1):
        state.observe(message, turn)

    assert state.build_coverage_adaptive_query() == ". ".join(messages)

    dense_state = StateBaselineV2("dense", {})
    dense_state.apply_constraints(
        [
            evidence("category", ["shoes"], 1, "shoes"),
            evidence("material", ["leather"], 1, "leather"),
            evidence("color", ["black"], 1, "black"),
            evidence("style", ["casual"], 1, "casual"),
        ]
    )
    dense_state.observe(
        "Actually, navy.",
        2,
        parsed_constraints=[evidence("color", ["navy"], 2, "Actually, navy.")],
    )
    assert dense_state.build_coverage_adaptive_query() == (
        "shoes. leather. casual. navy"
    )


def test_correction_scoped_history_filters_and_resets(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    agent = ExperimentalAgent(
        catalog,
        state_variant="baseline_v2",
        query_variant="coverage_adaptive_v2",
        question_variant="other_always",
        retrieval_route="keyword",
        recommendation_history="correction_scoped",
    )
    agent.keyword = SequenceKeywordRetriever(["A", "B", "C", "D"])  # type: ignore[assignment]
    agent.reset("session", {})

    first = agent.respond("session", "I'm looking for shoes.", 1, 2)
    second = agent.respond("session", "I'm still exploring.", 2, 2)
    corrected = agent.respond(
        "session",
        "Actually, ignore my earlier preference. What I need is: navy.",
        3,
        2,
    )

    assert [item["parent_asin"] for item in first["recommendations"]] == ["A", "B"]
    assert [item["parent_asin"] for item in second["recommendations"]] == ["C", "D"]
    assert [item["parent_asin"] for item in corrected["recommendations"]] == [
        "A",
        "B",
    ]


def test_exact_presets_build_native_unified_agents() -> None:
    for name in (
        "state_baseline_v2_fixed.json",
        "state_baseline_v2_other.json",
        "state_baseline_v2_raw_control.json",
    ):
        config = load_suite_config(ROOT / "configs/suites" / name)
        agent = build_suite_agent(config, ROOT / "data/catalog.jsonl")
        assert isinstance(agent, ExperimentalAgent)


def test_committed_integration_report_is_current_and_keeps_holdout_sealed() -> None:
    report = json.loads(
        (ROOT / "artifacts/reports/state_baseline_v2_integration.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["sample_count"] == 150
    assert report["protected_holdout_accessed"] is False
    assert all(item["exact"] for item in report["parity"].values())
    for relative, expected in report["integration_input_sha256"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected
