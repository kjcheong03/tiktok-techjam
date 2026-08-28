from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostlab.campaign.evaluator import OfflineCampaignEvaluator
from ghostlab.campaign.models import (
    CampaignJob,
    CandidateSpec,
    FidelityBudget,
)
from ghostlab.retrieval.residual import ResidualAgentAdapter, ResidualDecision
from ghostlab.state.memory import ConversationState


class TargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        del session_id, user_profile

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        del session_id, user_message, turn, top_k
        return {
            "message": "matches",
            "ask_attribute": None,
            "recommendations": ["target"],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class ResidualParentAgent:
    def __init__(self) -> None:
        self.sessions: dict[str, ConversationState] = {}
        self.last_runtime_inputs: dict[str, tuple[str, list[float]]] = {}
        self.retrieval_trace: list[dict[str, object]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = ConversationState(session_id, user_profile)
        self.last_runtime_inputs[session_id] = ("shoe", [1.0, 0.5])

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        del session_id, user_message, turn, top_k
        return {
            "message": "matches",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": "other"},
                {"parent_asin": "target"},
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class TargetFirstResidual:
    def rerank(
        self, query: str, ranking: tuple[str, ...], **kwargs: object
    ) -> ResidualDecision:
        del query, kwargs
        assert ranking == ("other", "target")
        return ResidualDecision(("target", "other"), True, "activated", 0.1, 2)


def _candidate() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="control",
        baseline_id="baseline",
        techniques=("retrieval.sparse",),
        generation="control",
    )


def _write_inputs(root: Path) -> tuple[Path, Path]:
    catalog = root / "catalog.jsonl"
    dataset = root / "public_set.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "parent_asin": "target",
                "title": "shoe",
                "categories": ["shoe"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "parent_asin": "other",
                "title": "other shoe",
                "categories": ["shoe"],
            }
        )
        + "\n"
    )
    rows = [
        {
            "sample_id": f"sample-{index}",
            "scenario_type": scenario,
            "user_profile": {},
            "ground_truth": {"parent_asin": "target"},
            "intent_card": {
                "target_category": "shoe",
                "hard_constraints": ["blue"],
                "soft_preferences": [],
            },
            "behavior": {"scenario_type": scenario},
        }
        for index, scenario in enumerate(("buying", "browsing"))
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return dataset, catalog


def test_evaluator_runs_bounded_fidelities_and_outer_fold(tmp_path: Path) -> None:
    dataset, catalog = _write_inputs(tmp_path)
    candidate = _candidate()
    evaluator = OfflineCampaignEvaluator(
        candidates=(candidate,),
        builder=lambda _: TargetAgent(),
        dataset_path=dataset,
        catalog_path=catalog,
        adaptive_sample_ids=("sample-0", "sample-1"),
        outer_folds=(("sample-0",), ("sample-1",)),
        budgets=FidelityBudget(f0=1, f1=2, f2=2),
    )
    f0 = evaluator(
        CampaignJob(
            job_id="f0",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f0",
            seed=7,
        )
    )
    f2 = evaluator(
        CampaignJob(
            job_id="f2",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f2",
            outer_fold=1,
            seed=7,
        )
    )
    assert len(f0.session_rewards) == 1
    assert len(f2.session_rewards) == 1
    assert f0.score == f2.score == 1.0
    assert f0.hit_rate_at_10 == f0.mrr == 1.0
    assert f0.mttc == 1.0
    assert f0.latency_p95_ms >= 0.0


def test_protected_paths_are_rejected_before_read(tmp_path: Path) -> None:
    dataset, catalog = _write_inputs(tmp_path)
    protected = tmp_path / "f3_holdout.jsonl"
    dataset.replace(protected)
    with pytest.raises(ValueError, match="protected dataset path"):
        OfflineCampaignEvaluator(
            candidates=(_candidate(),),
            builder=lambda _: TargetAgent(),
            dataset_path=protected,
            catalog_path=catalog,
            adaptive_sample_ids=("sample-0", "sample-1"),
            outer_folds=(("sample-0",), ("sample-1",)),
            budgets=FidelityBudget(f0=1, f1=2, f2=2),
        )


def test_search_and_confirmation_folds_are_strictly_disjoint(tmp_path: Path) -> None:
    dataset, catalog = _write_inputs(tmp_path)
    candidate = _candidate()
    evaluator = OfflineCampaignEvaluator(
        candidates=(candidate,),
        builder=lambda _: TargetAgent(),
        dataset_path=dataset,
        catalog_path=catalog,
        adaptive_sample_ids=("sample-0", "sample-1"),
        outer_folds=(("sample-0",), ("sample-1",)),
        budgets=FidelityBudget(f0=1, f1=2, f2=2),
        search_outer_folds=(0,),
        confirmation_outer_folds=(1,),
    )
    f1 = evaluator(
        CampaignJob(
            job_id="f1-search-only",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f1",
            seed=7,
        )
    )
    assert len(f1.session_rewards) == 1
    with pytest.raises(ValueError, match="frozen confirmation fold"):
        evaluator(
            CampaignJob(
                job_id="invalid-confirmation",
                candidate_hash=candidate.canonical_hash(),
                fidelity="f2",
                outer_fold=0,
                seed=7,
            )
        )
    confirmed = evaluator(
        CampaignJob(
            job_id="valid-confirmation",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f2",
            outer_fold=1,
            seed=7,
        )
    )
    assert len(confirmed.session_rewards) == 1


def test_campaign_accepts_residual_adapter_reordered_response_objects(
    tmp_path: Path,
) -> None:
    dataset, catalog = _write_inputs(tmp_path)
    candidate = _candidate()
    evaluator = OfflineCampaignEvaluator(
        candidates=(candidate,),
        builder=lambda _: ResidualAgentAdapter(
            ResidualParentAgent(),
            TargetFirstResidual(),  # type: ignore[arg-type]
        ),
        dataset_path=dataset,
        catalog_path=catalog,
        adaptive_sample_ids=("sample-0", "sample-1"),
        outer_folds=(("sample-0",), ("sample-1",)),
        budgets=FidelityBudget(f0=1, f1=2, f2=2),
    )

    outcome = evaluator(
        CampaignJob(
            job_id="residual-f0",
            candidate_hash=candidate.canonical_hash(),
            fidelity="f0",
            seed=7,
        )
    )

    assert outcome.state == "complete"
    assert outcome.score == 1.0
    assert outcome.hit_rate_at_10 == outcome.mrr == 1.0
