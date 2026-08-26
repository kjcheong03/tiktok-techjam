"""Run the keyword-only state/policy evaluation matrix.

The runner deliberately imports the keyword retriever lazily and never creates
``DenseRetriever``.  This keeps the contribution-isolation harness usable in a
checkout that has the public evaluator and catalog, but does not have a model
cache (or even sentence-transformers installed).

Results are stored as::

    {
      "variants": {
        "v1_keyword_state": {
          "current_order": <evaluator result>,
          "fixed_other": <evaluator result>
        }
      },
      "paired_comparisons": { ... }
    }

Additional cumulative variants can be supplied by adding ``VariantSpec``
instances to the registry returned by ``default_variant_registry``.  Each
variant is built with the same keyword retriever and evaluated under every
registered policy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from baseline.question_policy import (
    QUESTION_POLICIES,
    QuestionPolicy,
    fixed_other,
)


AgentFactory: TypeAlias = Callable[[Path, Any, QuestionPolicy], Any]

DEFAULT_COMPARISON_EDGES = (
    ("v1_keyword_state", "v2_state_only"),
    ("v2_state_only", "v2_state_prioritized_raw_history"),
    ("raw_history_no_state", "v2_state_prioritized_raw_history"),
)


@dataclass(frozen=True)
class VariantSpec:
    """Factory metadata for one state contribution variant.

    ``factory`` receives the catalog path, the shared unchanged keyword
    retriever, and the selected question policy.  A later V2 implementation
    can register cumulative factories without changing the evaluation loop.
    """

    name: str
    factory: AgentFactory


def _policy_aware_v1_agent(
    catalog_path: Path,
    keyword: Any,
    policy: QuestionPolicy,
) -> Any:
    """Build the V1 keyword-state agent without constructing a dense retriever.

    The current V1 agent predates policy injection.  When the agent exposes a
    ``question_policy`` (or ``policy``) constructor argument we use it
    directly.  The small adapter fallback keeps this harness executable before
    that integration lands while preserving the fixed-``other`` state
    semantics.
    """

    # Keep these imports out of module import time: policy and paired-session
    # tests should not require NumPy, sentence-transformers, or the catalog.
    from baseline.agent import BaselineAgent

    common = {
        "mode": "keyword",
        "stateful": True,
        "keyword": keyword,
        "dense": None,
    }
    try:
        parameters = inspect.signature(BaselineAgent.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "question_policy" in parameters:
        return BaselineAgent(**common, question_policy=policy)
    if "policy" in parameters:
        return BaselineAgent(**common, policy=policy)

    # A class with **kwargs can still support the canonical name even though
    # it does not list it explicitly.
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return BaselineAgent(**common, question_policy=policy)

    return _V1PolicyAdapter(BaselineAgent(**common), policy)


class _V1PolicyAdapter:
    """Inject a policy into the pre-policy V1 agent at the runner boundary.

    ``BaselineAgent.respond`` already observes the user message and performs
    the unchanged keyword search.  Its only policy-coupled side effect is the
    mutation of ``asked_attributes`` and ``last_asked_attribute``.  For the
    fixed-``other`` probe we undo the internally selected fixed-order question,
    then record ``other`` as the attribute that was actually presented.
    """

    def __init__(self, agent: Any, policy: QuestionPolicy) -> None:
        self._agent = agent
        self._policy = policy

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> Any:
        state = self._session_state(session_id)
        snapshot = self._question_state_snapshot(state) if self._is_fixed_other() else None
        try:
            response = self._agent.respond(session_id, user_message, turn, top_k)
        finally:
            if snapshot is not None:
                self._restore_question_state(state, snapshot)

        if self._is_fixed_other() and isinstance(response, dict):
            response = dict(response)
            response["ask_attribute"] = fixed_other(None, turn)
            response["message"] = "Are there any other requirements that matter to you?"
            if state is not None and hasattr(state, "last_asked_attribute"):
                state.last_asked_attribute = "other"
        return response

    def _is_fixed_other(self) -> bool:
        return self._policy is fixed_other

    def _session_state(self, session_id: str) -> Any:
        sessions = getattr(self._agent, "sessions", None)
        if isinstance(sessions, Mapping):
            return sessions.get(session_id)
        return None

    @staticmethod
    def _question_state_snapshot(state: Any) -> tuple[list[str], Any] | None:
        if state is None:
            return None
        asked = getattr(state, "asked_attributes", None)
        if not isinstance(asked, list):
            return None
        return list(asked), getattr(state, "last_asked_attribute", None)

    @staticmethod
    def _restore_question_state(state: Any, snapshot: tuple[list[str], Any] | None) -> None:
        if state is None or snapshot is None:
            return
        asked, last_asked = snapshot
        current_asked = getattr(state, "asked_attributes", None)
        if isinstance(current_asked, list):
            current_asked[:] = asked
        if hasattr(state, "last_asked_attribute"):
            state.last_asked_attribute = last_asked


def default_variant_registry() -> dict[str, VariantSpec]:
    """Return cumulative state and query-representation variants."""

    v1 = VariantSpec("v1_keyword_state", _policy_aware_v1_agent)
    state_only = VariantSpec("v2_state_only", _state_only_v2_agent)
    raw_control = VariantSpec("raw_history_no_state", _raw_history_control_agent)
    state_prioritized_raw_history = VariantSpec(
        "v2_state_prioritized_raw_history",
        _state_prioritized_raw_history_v2_agent,
    )
    return {
        v1.name: v1,
        state_only.name: state_only,
        raw_control.name: raw_control,
        state_prioritized_raw_history.name: state_prioritized_raw_history,
    }


def _state_only_v2_agent(
    catalog_path: Path,
    keyword: Any,
    policy: QuestionPolicy,
) -> Any:
    """Build V2 transitions with the legacy interpreter and unchanged BM25."""

    from baseline.agent import BaselineAgent
    from baseline.state_v2 import StructuredSessionState

    return BaselineAgent(
        mode="keyword",
        stateful=True,
        keyword=keyword,
        dense=None,
        state_factory=StructuredSessionState,
        question_policy=policy,
    )


def _state_prioritized_raw_history_v2_agent(
    catalog_path: Path,
    keyword: Any,
    policy: QuestionPolicy,
) -> Any:
    """Build V2 transitions with state-prioritized raw history for BM25."""

    from baseline.agent import BaselineAgent
    from baseline.query_state import StatePrioritizedRawHistorySessionState

    return BaselineAgent(
        mode="keyword",
        stateful=True,
        keyword=keyword,
        dense=None,
        state_factory=StatePrioritizedRawHistorySessionState,
        question_policy=policy,
    )


def _raw_history_control_agent(
    catalog_path: Path,
    keyword: Any,
    policy: QuestionPolicy,
) -> Any:
    """Build the lossless-history control without managed constraints."""

    from baseline.raw_history_control import RawHistoryNoManagedStateAgent

    return RawHistoryNoManagedStateAgent(keyword, policy)


def register_variant(registry: dict[str, VariantSpec], spec: VariantSpec) -> None:
    """Register one cumulative variant, rejecting accidental name reuse."""

    if spec.name in registry:
        raise ValueError(f"variant {spec.name!r} is already registered")
    registry[spec.name] = spec


def _session_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = result.get("sessions", ())
    if isinstance(payload, Mapping):
        values: Iterable[Any] = payload.values()
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        values = payload
    else:
        values = ()

    records: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping) and "sample_id" in value:
            records.append(dict(value))
    return records


def _session_index(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in _session_records(result):
        sample_id = str(record["sample_id"])
        # Evaluator output is unique by construction.  Keeping the first value
        # makes malformed input deterministic and avoids silently changing a
        # previously paired session on duplicate rows.
        indexed.setdefault(sample_id, record)
    return indexed


def _hit(record: Mapping[str, Any]) -> bool:
    value = record.get("hit")
    if isinstance(value, bool):
        return value
    if value is not None:
        return bool(value)
    return _number(record.get("first_hit_turn")) is not None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first_hit_turn(record: Mapping[str, Any]) -> float | None:
    value = _number(record.get("first_hit_turn"))
    return value if value is not None and value >= 0 else None


def _target_rank(record: Mapping[str, Any]) -> float | None:
    # ``best_rank`` is the evaluator's contract.  ``target_rank`` is accepted
    # as a small convenience for hand-authored comparison fixtures.
    value = record.get("best_rank", record.get("target_rank"))
    value = _number(value)
    return value if value is not None and value > 0 else None


def compare_paired_sessions(
    before_result: Mapping[str, Any],
    after_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare evaluator session rows aligned by ``sample_id``.

    A rank or turn comparison only includes sessions where both results have
    that value.  Miss/hit conversions are counted independently, so a newly
    hit session is not also treated as a rank improvement.  Counts are paired
    with stable sample-id lists to make the report auditable.
    """

    before = _session_index(before_result)
    after = _session_index(after_result)
    paired_ids = [sample_id for sample_id in before if sample_id in after]

    miss_to_hit: list[str] = []
    hit_to_miss: list[str] = []
    earlier_hit_turn: list[str] = []
    later_hit_turn: list[str] = []
    better_target_rank: list[str] = []
    worse_target_rank: list[str] = []

    for sample_id in paired_ids:
        old = before[sample_id]
        new = after[sample_id]
        old_hit = _hit(old)
        new_hit = _hit(new)
        if not old_hit and new_hit:
            miss_to_hit.append(sample_id)
        elif old_hit and not new_hit:
            hit_to_miss.append(sample_id)

        old_turn = _first_hit_turn(old)
        new_turn = _first_hit_turn(new)
        if old_hit and new_hit and old_turn is not None and new_turn is not None:
            if new_turn < old_turn:
                earlier_hit_turn.append(sample_id)
            elif new_turn > old_turn:
                later_hit_turn.append(sample_id)

        old_rank = _target_rank(old)
        new_rank = _target_rank(new)
        if old_rank is not None and new_rank is not None:
            if new_rank < old_rank:
                better_target_rank.append(sample_id)
            elif new_rank > old_rank:
                worse_target_rank.append(sample_id)

    report: dict[str, Any] = {
        "paired_count": len(paired_ids),
        "unpaired_before_count": len(before) - len(paired_ids),
        "unpaired_after_count": len(after) - len(paired_ids),
        "miss_to_hit": len(miss_to_hit),
        "hit_to_miss": len(hit_to_miss),
        "earlier_hit_turn": len(earlier_hit_turn),
        "later_hit_turn": len(later_hit_turn),
        "better_target_rank": len(better_target_rank),
        "worse_target_rank": len(worse_target_rank),
    }
    for name, sample_ids in (
        ("miss_to_hit", miss_to_hit),
        ("hit_to_miss", hit_to_miss),
        ("earlier_hit_turn", earlier_hit_turn),
        ("later_hit_turn", later_hit_turn),
        ("better_target_rank", better_target_rank),
        ("worse_target_rank", worse_target_rank),
    ):
        report[f"{name}_sample_ids"] = sample_ids
    return report


def _paired_variant_results(
    variant_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparison_edges: Sequence[tuple[str, str]],
    policy_order: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    comparisons: dict[str, dict[str, dict[str, Any]]] = {
        policy_name: {} for policy_name in policy_order
    }
    for policy_name in policy_order:
        for before_name, after_name in comparison_edges:
            before = variant_results[before_name][policy_name]
            after = variant_results[after_name][policy_name]
            comparisons[policy_name][f"{before_name} -> {after_name}"] = compare_paired_sessions(
                before,
                after,
            )
    return comparisons


def run_state_baselines(
    *,
    catalog_path: str | Path = "data/catalog.jsonl",
    dataset_path: str | Path = "data/public_set.jsonl",
    variants: Mapping[str, VariantSpec] | None = None,
    policies: Mapping[str, QuestionPolicy] | None = None,
) -> dict[str, Any]:
    """Run all registered keyword-state variants under all fixed policies."""

    # These imports are intentionally local.  Importing this module should be
    # enough for standard-library policy/math tests in a dependency-light
    # checkout.
    from baseline.retrieval import KeywordRetriever
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

    catalog = Path(catalog_path)
    dataset = Path(dataset_path)
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    keyword = KeywordRetriever(catalog)

    variant_registry = (
        default_variant_registry() if variants is None else dict(variants)
    )
    policy_registry = dict(QUESTION_POLICIES) if policies is None else dict(policies)
    variant_order = list(variant_registry)
    policy_order = list(policy_registry)

    variant_results: dict[str, dict[str, dict[str, Any]]] = {}
    for variant_name in variant_order:
        spec = variant_registry[variant_name]
        if spec.name != variant_name:
            raise ValueError(
                f"variant registry key {variant_name!r} does not match spec name {spec.name!r}"
            )
        per_policy: dict[str, dict[str, Any]] = {}
        for policy_name in policy_order:
            policy = policy_registry[policy_name]
            agent = spec.factory(catalog, keyword, policy)
            started = time.perf_counter()
            evaluated = evaluate(agent, samples, catalog_ids, categories, products)
            elapsed = time.perf_counter() - started
            per_policy[policy_name] = {
                **evaluated,
                "evaluation_seconds": round(elapsed, 6),
            }
        variant_results[variant_name] = per_policy

    return {
        "status": "ok",
        "catalog_path": str(catalog),
        "dataset_path": str(dataset),
        "sample_count": len(samples),
        "variant_order": variant_order,
        "policy_order": policy_order,
        "variants": variant_results,
        "paired_comparisons": _paired_variant_results(
            variant_results,
            (
                DEFAULT_COMPARISON_EDGES
                if variants is None
                else tuple(zip(variant_order, variant_order[1:]))
            ),
            policy_order,
        ),
    }


def write_results(result: Mapping[str, Any], output_path: str | Path) -> Path:
    """Persist a runner report as stable, human-readable JSON."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def _console_report(result: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": result.get("status"),
        "sample_count": result.get("sample_count", 0),
        "variants": {},
    }
    variants = result.get("variants", {})
    if isinstance(variants, Mapping):
        for variant_name, policy_results in variants.items():
            if not isinstance(policy_results, Mapping):
                continue
            summary["variants"][variant_name] = {}
            for policy_name, evaluation in policy_results.items():
                if not isinstance(evaluation, Mapping):
                    continue
                summary["variants"][variant_name][policy_name] = {
                    key: evaluation.get(key)
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "efficiency",
                        "recommended_technical_score",
                        "evaluation_seconds",
                    )
                }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run keyword-only state baseline policy probes")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/state_baseline_results.json")
    args = parser.parse_args(argv)

    try:
        result = run_state_baselines(catalog_path=args.catalog, dataset_path=args.dataset)
    except FileNotFoundError as exc:
        blocker = {
            "status": "blocked",
            "blocker": str(exc),
            "catalog_path": str(args.catalog),
            "dataset_path": str(args.dataset),
        }
        write_results(blocker, args.output)
        print(json.dumps(blocker, indent=2), file=sys.stderr)
        return 2

    output = write_results(result, args.output)
    print(json.dumps(_console_report(result), indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
