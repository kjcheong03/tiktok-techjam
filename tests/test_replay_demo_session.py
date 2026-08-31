from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghostlab.training.adaptive_datasets import AdaptiveTrainingCorpus
from scripts import replay_demo_session as demo


@dataclass
class _Constraint:
    attribute: str
    values: list[str]
    polarity: str = "include"
    strength: str = "hard"
    source_turn: int = 1


@dataclass
class _State:
    messages: list[str] = field(default_factory=list)
    intent_epoch: int = 0
    active_constraints: list[_Constraint] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)

    def build_coverage_adaptive_query(self) -> str:
        return ". ".join(self.messages)


@dataclass
class _Session:
    state: _State


class _FakeAgent:
    def __init__(self) -> None:
        self.sessions: dict[str, _Session] = {}
        self.traces: list[SimpleNamespace] = []
        self.candidate_snapshots: list[SimpleNamespace] = []
        self.calls: list[tuple[str, int, str]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        del user_profile
        self.sessions[session_id] = _Session(_State())

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        del top_k
        self.calls.append((session_id, turn, user_message))
        state = self.sessions[session_id].state
        state.messages.append(user_message)
        if turn == 1:
            state.asked_attributes.append("color")
            state.active_constraints.append(
                _Constraint("color", ["red"], source_turn=1)
            )
            response = {
                "message": "I can narrow these down by color.",
                "ask_attribute": "color",
                "recommendations": [{"parent_asin": "A"}],
            }
            trace = {
                "session_id": session_id,
                "turn": turn,
                "route": "buying",
                "route_confidence": 0.9,
                "route_reason": "specific_requirement",
                "overloaded": False,
                "preview_reason": "specific",
                "preview_candidate_count": 1,
                "preview_score_flatness": 0.1,
                "contribution_counts": {"keyword": 1, "category": 1, "vector": 0},
                "union_candidate_count": 1,
                "semantic_backend": "skipped:buying_route",
                "semantic_activation_reason": "buying_route",
                "semantic_changed": False,
                "semantic_failure_reason": None,
                "semantic_elapsed_ms": 0.0,
                "constraint_counts": {
                    "confirmed_matches": 1,
                    "confirmed_violations": 0,
                    "unknown_metadata": 0,
                    "soft_preferences": 0,
                },
                "output_constraint_violations": 0,
                "state_query": user_message,
                "intent_epoch": state.intent_epoch,
                "query_sha256": "query-1",
                "query_views": (),
                "dense_requested_per_view": 0,
                "dense_output_k": 0,
                "dense_selection": "not_used",
                "reason_codes": ("route:buying", "union:executed"),
                "safe_merge_executed": False,
                "safe_ranker_executed": False,
                "normal_union_executed": True,
                "semantic_decision_reached": True,
                "semantic_executed": False,
                "fallback_reason": None,
            }
            snapshot = SimpleNamespace(
                session_id=session_id,
                turn=turn,
                authority_removed_ids=("C",),
            )
        else:
            state.intent_epoch = 1
            response = {
                "message": "The red option is the strongest match.",
                "ask_attribute": None,
                "recommendations": [{"parent_asin": "B"}],
            }
            trace = {
                "session_id": session_id,
                "turn": turn,
                "route": "buying",
                "route_confidence": 0.95,
                "route_reason": "specific_requirement",
                "overloaded": False,
                "preview_reason": "specific",
                "preview_candidate_count": 1,
                "preview_score_flatness": 0.1,
                "contribution_counts": {"keyword": 1, "category": 0, "vector": 0},
                "union_candidate_count": 1,
                "semantic_backend": "skipped:buying_route",
                "semantic_activation_reason": "buying_route",
                "semantic_changed": False,
                "semantic_failure_reason": None,
                "semantic_elapsed_ms": 0.0,
                "constraint_counts": {},
                "output_constraint_violations": 0,
                "state_query": ". ".join(state.messages),
                "intent_epoch": state.intent_epoch,
                "query_sha256": "query-2",
                "query_views": (),
                "dense_requested_per_view": 0,
                "dense_output_k": 0,
                "dense_selection": "not_used",
                "reason_codes": ("route:buying", "union:executed"),
                "safe_merge_executed": False,
                "safe_ranker_executed": False,
                "normal_union_executed": True,
                "semantic_decision_reached": True,
                "semantic_executed": False,
                "fallback_reason": None,
            }
            snapshot = SimpleNamespace(
                session_id=session_id,
                turn=turn,
                authority_removed_ids=(),
            )
        self.traces.append(SimpleNamespace(**trace))
        self.candidate_snapshots.append(snapshot)
        return response


def _sample() -> dict:
    return {
        "sample_id": "public_demo",
        "scenario_type": "buying",
        "ground_truth": {"parent_asin": "B"},
        "user_profile": {"preference_tags": ["comfort"]},
        "intent_card": {
            "target_category": "Red Shirt",
            "hard_constraints": ["red"],
            "soft_preferences": ["comfortable"],
        },
        "behavior": {"scenario_type": "buying"},
    }


def _products() -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
    products = {
        "A": {
            "parent_asin": "A",
            "title": "Blue Shirt",
            "categories": ["Clothing", "Shirts"],
        },
        "B": {
            "parent_asin": "B",
            "title": "Red Shirt",
            "categories": ["Clothing", "Shirts"],
        },
        "C": {
            "parent_asin": "C",
            "title": "Green Shirt",
            "categories": ["Clothing", "Shirts"],
        },
    }
    categories = {key: value["categories"] for key, value in products.items()}
    return set(products), categories, products


def test_replay_enforces_development_only_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample()
    corpus = AdaptiveTrainingCorpus(
        samples={
            "public_demo": sample,
            "holdout_demo": {**sample, "sample_id": "holdout_demo"},
        },
        origins={"public_demo": "demo.jsonl", "holdout_demo": "demo.jsonl"},
        sources=(),
    )
    manifest = SimpleNamespace(
        development_ids=frozenset({"public_demo"}),
        holdout_ids=frozenset({"holdout_demo"}),
    )
    monkeypatch.setattr(demo, "load_adaptive_training_corpus", lambda *_args: corpus)
    monkeypatch.setattr(demo, "load_lineage_manifest", lambda *_args: manifest)
    selected, _, _ = demo._development_sample(
        Path("."), "public_demo", ("demo.jsonl",), "manifest.json"
    )
    assert selected["sample_id"] == "public_demo"
    with pytest.raises(ValueError, match="holdout"):
        demo._development_sample(
            Path("."), "holdout_demo", ("demo.jsonl",), "manifest.json"
        )
    with pytest.raises(ValueError, match="development"):
        demo._development_sample(
            Path("."), "missing_demo", ("demo.jsonl",), "manifest.json"
        )


def test_replay_keeps_evaluator_only_labels_separate_and_follows_turns(
    tmp_path: Path,
) -> None:
    catalog_ids, categories, products = _products()
    output: list[str] = []
    payload = demo.replay_one_sample(
        _sample(),
        agent=_FakeAgent(),
        categories=categories,
        products=products,
        catalog_ids=catalog_ids,
        output_dir=tmp_path,
        print_fn=output.append,
    )
    visible = json.dumps(payload["agent_visible"], sort_keys=True)
    assert "target_asin" not in visible
    assert "scenario_type" not in visible
    assert "next_reply" not in visible
    assert payload["evaluator_only"]["turns"][0]["hit"] is False
    assert payload["evaluator_only"]["turns"][0]["next_reply"]
    assert payload["evaluator_only"]["turns"][1]["hit"] is True
    assert payload["evaluator_only"]["turns"][1]["rank"] == 1
    assert payload["evaluator_only"]["session_result"]["first_hit_turn"] == 2
    assert [item["turn"] for item in payload["agent_visible"]["turns"]] == [1, 2]
    assert len(output) == 4


def test_intent_override_is_applied_between_turns_for_evaluator_only_hit(
    tmp_path: Path,
) -> None:
    catalog_ids, categories, products = _products()
    sample = _sample()
    sample["scenario_type"] = "intent_override"
    sample["behavior"] = {
        "scenario_type": "intent_override",
        "override": {
            "turn": 3,
            "old_value": "blue",
            "new_value": "red",
            "message": "Actually, I need red instead.",
        },
    }
    payload = demo.replay_one_sample(
        sample,
        agent=_FakeAgent(),
        categories=categories,
        products=products,
        catalog_ids=catalog_ids,
        output_dir=tmp_path,
        print_fn=lambda _text: None,
    )
    turns = payload["evaluator_only"]["turns"]
    assert [turn["override_active"] for turn in turns] == [False, False, True]
    assert [turn["hit"] for turn in turns] == [False, False, True]
    assert "Actually, I need red instead." in turns[1]["next_reply"]
    assert payload["evaluator_only"]["session_result"]["first_hit_turn"] == 3


def test_demo_artifacts_and_console_have_readable_consistent_stages(
    tmp_path: Path,
) -> None:
    catalog_ids, categories, products = _products()
    output: list[str] = []
    payload = demo.replay_one_sample(
        _sample(),
        agent=_FakeAgent(),
        categories=categories,
        products=products,
        catalog_ids=catalog_ids,
        output_dir=tmp_path,
        print_fn=output.append,
    )
    json_path = tmp_path / demo.JSON_FILENAME
    markdown_path = tmp_path / demo.MARKDOWN_FILENAME
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    markdown = markdown_path.read_text(encoding="utf-8")
    console = "".join(output)
    for stage in (
        "Evaluator message:",
        "Dynamic conversation state:",
        "Changes since previous turn:",
        "Route:",
        "Preview/overload:",
        "Bounded-vs-full path:",
        "Evidence contribution counts:",
        "Union:",
        "Semantic:",
        "Constraints:",
        "Question:",
        "Top 10:",
        "Evaluator-only:",
    ):
        assert stage in console
        assert stage.split(":", 1)[0] in markdown
    assert "Red Shirt" in console
    assert "`B`" in markdown


def test_missing_config_fails_instead_of_silently_using_baseline(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "adaptive_hybrid_1a_3b_v1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing adaptive config"):
        demo._config_path(
            tmp_path, "configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json"
        )
