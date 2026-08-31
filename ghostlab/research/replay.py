from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from ghostlab.competition.contract import AgentProtocol
from ghostlab.research.firewall import runtime_profile

SHARED_EVALUATION_HARNESS = "ghostlab.research.replay.evaluate_shared.v1"
DEFAULT_EVALUATION_SEED = 2026


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_loaded_backends(seed: int) -> None:
    """Reset RNGs without importing heavyweight optional model libraries."""

    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def shared_evaluation_contract(
    samples: list[dict], catalog_path: str | Path, *, seed: int
) -> dict[str, object]:
    """Return the auditable contract shared by every comparable system."""

    catalog = Path(catalog_path)
    official_evaluator = (
        Path(__file__).resolve().parents[2] / "evaluator" / "local_evaluator.py"
    )
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    contract = {
        "harness_id": SHARED_EVALUATION_HARNESS,
        "harness_sha256": _sha256_file(Path(__file__)),
        "published_evaluator_sha256": _sha256_file(official_evaluator),
        "ordered_session_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "catalog_sha256": _sha256_file(catalog),
        "sample_count": len(sample_ids),
        "evaluation_seed": seed,
        "max_turns": MAX_TURNS,
        "top_k": TOP_K,
        "profile_contract": "sanitized user_profile supplied to reset",
        "response_contract": (
            "exceptions and invalid responses become empty responses; catalog IDs "
            "are validated, deduplicated and capped at Top K"
        ),
        "timeout_policy": (
            "no evaluator wall-clock cutoff; component timeouts are frozen in each "
            "hash-bound system configuration and latency is reported separately"
        ),
    }
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return contract


@dataclass(frozen=True)
class RuntimeObservation:
    session_id: str
    user_message: str
    turn: int
    top_k: int = TOP_K


@dataclass(frozen=True)
class ReplaySnapshot:
    user_message: str
    turn: int
    disclosed: frozenset[str]
    boundary_used: bool
    override_applied: bool
    hit_turn: int | None
    best_rank: int | None
    done: bool


class ReplayEnvironment:
    """Exact research replay of the published evaluator transition semantics."""

    def __init__(
        self, sample: dict, categories: dict[str, list[str]], products: dict[str, dict]
    ) -> None:
        self.sample = copy.deepcopy(sample)
        self.catalog_ids = set(products)
        self.target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        self.effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        self.category = coarse_category(categories.get(self.target, []))
        self.session_id = f"replay_{sample['sample_id']}"
        self.reset()

    def reset(self) -> RuntimeObservation:
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = self.sample["scenario_type"] != "intent_override"
        self.user_message = initial_message(
            self.effective_sample, self.category, self.disclosed
        )
        self.turn = 1
        self.hit_turn: int | None = None
        self.best_rank: int | None = None
        self.done = False
        return self.observe()

    def observe(self) -> RuntimeObservation:
        return RuntimeObservation(self.session_id, self.user_message, self.turn)

    def snapshot(self) -> ReplaySnapshot:
        return ReplaySnapshot(
            user_message=self.user_message,
            turn=self.turn,
            disclosed=frozenset(self.disclosed),
            boundary_used=self.boundary_used,
            override_applied=self.override_applied,
            hit_turn=self.hit_turn,
            best_rank=self.best_rank,
            done=self.done,
        )

    def restore(self, snapshot: ReplaySnapshot) -> None:
        self.user_message = snapshot.user_message
        self.turn = snapshot.turn
        self.disclosed = set(snapshot.disclosed)
        self.boundary_used = snapshot.boundary_used
        self.override_applied = snapshot.override_applied
        self.hit_turn = snapshot.hit_turn
        self.best_rank = snapshot.best_rank
        self.done = snapshot.done

    def clone(self) -> ReplayEnvironment:
        return copy.deepcopy(self)

    def step(self, response: object) -> RuntimeObservation | None:
        if self.done:
            raise RuntimeError("cannot step a completed replay")
        payload = response if isinstance(response, dict) else {}
        ranked = normalize_recommendations(
            payload.get("recommendations"), self.catalog_ids
        )
        if self.override_applied and self.target in ranked:
            self.best_rank = ranked.index(self.target) + 1
            self.hit_turn = self.turn
            self.done = True
            return None
        if self.turn == MAX_TURNS:
            self.done = True
            return None
        override = self.effective_sample.get("behavior", {}).get("override") or {}
        if not self.override_applied and self.turn + 1 == int(override.get("turn", 3)):
            self.override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                self.disclosed.add(new_value)
            self.user_message = str(
                override.get(
                    "message", "Actually, please ignore my earlier preference."
                )
            )
        else:
            self.user_message, self.boundary_used = customer_reply(
                self.effective_sample,
                payload.get("ask_attribute"),
                self.disclosed,
                self.boundary_used,
            )
        self.turn += 1
        return self.observe()

    def session_result(self) -> dict:
        if not self.done:
            raise RuntimeError("replay is not complete")
        return {
            "sample_id": self.sample["sample_id"],
            "scenario_type": self.sample["scenario_type"],
            "hit": self.hit_turn is not None,
            "first_hit_turn": self.hit_turn,
            "best_rank": self.best_rank,
            "reciprocal_rank": 0.0 if self.best_rank is None else 1.0 / self.best_rank,
        }


def evaluate_replay(
    agent: AgentProtocol,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    sessions: list[dict] = []
    prompt_tokens = completion_tokens = 0
    for sample in samples:
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        agent.reset(observation.session_id, runtime_profile(sample))
        while not environment.done:
            try:
                response = agent.respond(
                    observation.session_id,
                    observation.user_message,
                    observation.turn,
                    observation.top_k,
                )
            except Exception:  # noqa: BLE001 - evaluator parity requires containment
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(
                response.get("message"), str
            ):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                if (
                    isinstance(usage.get("prompt_tokens"), int)
                    and usage["prompt_tokens"] >= 0
                ):
                    prompt_tokens += usage["prompt_tokens"]
                if (
                    isinstance(usage.get("completion_tokens"), int)
                    and usage["completion_tokens"] >= 0
                ):
                    completion_tokens += usage["completion_tokens"]
            next_observation = environment.step(response)
            if next_observation is not None:
                observation = next_observation
        sessions.append(environment.session_result())

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = (
        0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
        "sessions": sessions,
    }


def evaluate_shared(
    agent: AgentProtocol,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    catalog_path: str | Path,
    seed: int = DEFAULT_EVALUATION_SEED,
) -> dict:
    """Run one agent through the canonical, deterministic research harness."""

    seed_loaded_backends(seed)
    result = evaluate_replay(agent, samples, categories, products)
    result["evaluation_contract"] = shared_evaluation_contract(
        samples, catalog_path, seed=seed
    )
    return result


def session_reward(session: dict) -> float:
    hit = float(bool(session["hit"]))
    reciprocal_rank = float(session["reciprocal_rank"])
    turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
    efficiency = max(0.0, min(1.0, (11.0 - float(turn)) / 10.0))
    return 0.50 * hit + 0.30 * reciprocal_rank + 0.20 * efficiency


def paired_delta(candidate: list[dict], baseline: list[dict]) -> list[float]:
    base = {str(item["sample_id"]): session_reward(item) for item in baseline}
    return [session_reward(item) - base[str(item["sample_id"])] for item in candidate]
