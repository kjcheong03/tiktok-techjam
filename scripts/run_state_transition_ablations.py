"""Run isolated V2 state-transition ablations under fixed question probes.

The harness reuses the existing keyword retriever, evaluator, and paired-session
comparison functions.  Every variant keeps the retained coverage-adaptive query,
per-session recommendation filtering, and correction-scoped history reset; each
ablation changes exactly one state-transition behavior.

Run with::

    python -m scripts.run_state_transition_ablations \
      --output artifacts/state_transition_ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from baseline.constraints import ALLOWED_ATTRIBUTES, StructuredConstraint
from baseline.question_policy import QUESTION_POLICIES, QuestionPolicy
from baseline.query_state import (
    CoverageAdaptiveSessionState,
    _LOW_COVERAGE_MAX_ACTIVE_CONSTRAINTS,
)
from scripts.run_state_baselines import (
    VariantSpec,
    _console_report,
    compare_paired_sessions,
    run_state_baselines,
    write_results,
)


FULL_VARIANT_NAME = "full_retained"
ABLATION_VARIANT_NAMES = (
    "no_compatible_accumulation",
    "no_targeted_correction",
    "no_ambiguous_preservation",
    "no_no_preference_preservation",
)
TRANSITION_POLICY_NAMES = ("fixed_turn_order", "fixed_other")
TRANSITION_COMPARISON_EDGES = tuple(
    (ablation_name, FULL_VARIANT_NAME)
    for ablation_name in ABLATION_VARIANT_NAMES
)

StateFactory: TypeAlias = Callable[[str, dict], CoverageAdaptiveSessionState]


class _TransitionAblationMixin:
    """Shared state helpers for transition-only ablations."""

    def _concrete_correction_attributes(
        self,
        constraints: Iterable[StructuredConstraint],
        *,
        correction: bool,
    ) -> set[str]:
        if not correction:
            return set()
        return {
            constraint.attribute
            for constraint in constraints
            if constraint.attribute in ALLOWED_ATTRIBUTES
            and self._confident_correction_attribute(constraint)
        }

    def _supersede_non_category(self, *, keep_attributes: set[str] | None = None) -> None:
        keep = keep_attributes or set()
        for constraint in self.constraints:
            if constraint.active and constraint.attribute != "category" and constraint.attribute not in keep:
                constraint.supersede()


class NoCompatibleAccumulationState(
    _TransitionAblationMixin,
    CoverageAdaptiveSessionState,
):
    """Reproduce V1 loss of compatible same-message values."""

    def observe(
        self,
        message: str,
        turn: int,
        parsed_constraints: Iterable[StructuredConstraint] | None = None,
        *,
        no_preference_attributes: Iterable[str] | None = None,
    ) -> None:
        super().observe(
            message,
            turn,
            parsed_constraints,
            no_preference_attributes=no_preference_attributes,
        )
        current_message = [
            constraint
            for constraint in self.constraints
            if constraint.active
            and constraint.source_turn == turn
            and constraint.source_text == message
        ]
        attributes = {constraint.attribute for constraint in current_message}
        for attribute in attributes:
            same_attribute = [
                constraint
                for constraint in current_message
                if constraint.attribute == attribute
            ]
            for constraint in same_attribute[:-1]:
                constraint.supersede()


class NoTargetedCorrectionState(
    _TransitionAblationMixin,
    CoverageAdaptiveSessionState,
):
    """Blanket-supersede non-category state for concrete corrections."""

    def apply_constraints(
        self,
        constraints: Iterable[StructuredConstraint],
        *,
        source_text: str | None = None,
        correction: bool = False,
        supersede_attributes: Iterable[str] | None = None,
    ) -> list[StructuredConstraint]:
        incoming = list(constraints)
        concrete_attributes = self._concrete_correction_attributes(
            incoming,
            correction=correction,
        )
        if concrete_attributes:
            self._supersede_non_category(keep_attributes=concrete_attributes)
        return super().apply_constraints(
            incoming,
            source_text=source_text,
            correction=correction,
            supersede_attributes=supersede_attributes,
        )


class NoAmbiguousPreservationState(
    _TransitionAblationMixin,
    CoverageAdaptiveSessionState,
):
    """Blanket-supersede non-category state for vague corrections."""

    def apply_constraints(
        self,
        constraints: Iterable[StructuredConstraint],
        *,
        source_text: str | None = None,
        correction: bool = False,
        supersede_attributes: Iterable[str] | None = None,
    ) -> list[StructuredConstraint]:
        incoming = list(constraints)
        concrete_attributes = self._concrete_correction_attributes(
            incoming,
            correction=correction,
        )
        if correction and incoming and not concrete_attributes:
            self._supersede_non_category()
        return super().apply_constraints(
            incoming,
            source_text=source_text,
            correction=correction,
            supersede_attributes=supersede_attributes,
        )


class NoNoPreferencePreservationState(CoverageAdaptiveSessionState):
    """Hide active no-preference constraints from the positive query."""

    def _build_query_without_no_preference(self) -> str:
        ordered = sorted(
            enumerate(self.constraints),
            key=lambda item: (
                item[1].attribute != "category",
                item[1].source_turn,
                item[0],
            ),
        )
        values: list[str] = []
        for _, constraint in ordered:
            if not constraint.active:
                continue
            if constraint.polarity != "include":
                continue
            if constraint.attribute in self.no_preference_attributes:
                continue
            for value in constraint.values:
                if value not in values:
                    values.append(value)
        return ". ".join(values)

    def build_query(self) -> str:
        state_query = self._build_query_without_no_preference()
        raw_history = ". ".join(self.messages)
        has_superseded = any(not constraint.active for constraint in self.constraints)
        if (
            has_superseded
            and len(self.active_constraints) <= _LOW_COVERAGE_MAX_ACTIVE_CONSTRAINTS
        ):
            return raw_history
        return state_query or raw_history


def _agent_factory(state_factory: StateFactory) -> Callable[[Path, Any, QuestionPolicy], Any]:
    def factory(catalog_path: Path, keyword: Any, policy: QuestionPolicy) -> Any:
        del catalog_path
        from baseline.agent import BaselineAgent

        return BaselineAgent(
            mode="keyword",
            stateful=True,
            keyword=keyword,
            dense=None,
            state_factory=state_factory,
            question_policy=policy,
            filter_seen_recommendations=True,
        )

    return factory


def default_variant_registry() -> dict[str, VariantSpec]:
    """Return the full retained state and each one-transition ablation."""

    state_factories: dict[str, StateFactory] = {
        FULL_VARIANT_NAME: CoverageAdaptiveSessionState,
        "no_compatible_accumulation": NoCompatibleAccumulationState,
        "no_targeted_correction": NoTargetedCorrectionState,
        "no_ambiguous_preservation": NoAmbiguousPreservationState,
        "no_no_preference_preservation": NoNoPreferencePreservationState,
    }
    return {
        name: VariantSpec(name, _agent_factory(state_factory))
        for name, state_factory in state_factories.items()
    }


def transition_policies() -> dict[str, QuestionPolicy]:
    """Return only the fixed-order and fixed-``other`` comparison policies."""

    return {
        name: QUESTION_POLICIES[name]
        for name in TRANSITION_POLICY_NAMES
    }


def paired_transition_comparisons(
    variant_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    policy_order: Sequence[str] = TRANSITION_POLICY_NAMES,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare each ablation to full state with the ablation on the left."""

    comparisons: dict[str, dict[str, dict[str, Any]]] = {
        policy_name: {} for policy_name in policy_order
    }
    for policy_name in policy_order:
        for ablation_name, full_name in TRANSITION_COMPARISON_EDGES:
            comparisons[policy_name][f"{ablation_name} -> {full_name}"] = compare_paired_sessions(
                variant_results[ablation_name][policy_name],
                variant_results[full_name][policy_name],
            )
    return comparisons


def run_state_transition_ablations(
    *,
    catalog_path: str | Path = "data/catalog.jsonl",
    dataset_path: str | Path = "data/public_set.jsonl",
) -> dict[str, Any]:
    """Evaluate full state and transition ablations on the unchanged evaluator."""

    result = run_state_baselines(
        catalog_path=catalog_path,
        dataset_path=dataset_path,
        variants=default_variant_registry(),
        policies=transition_policies(),
    )
    result = dict(result)
    result["full_variant"] = FULL_VARIANT_NAME
    result["ablation_order"] = list(ABLATION_VARIANT_NAMES)
    result["comparison_edges"] = [list(edge) for edge in TRANSITION_COMPARISON_EDGES]
    result["paired_comparisons"] = paired_transition_comparisons(
        result["variants"],
        result["policy_order"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run state-transition ablations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--output",
        default="artifacts/state_transition_ablation_results.json",
    )
    args = parser.parse_args(argv)

    try:
        result = run_state_transition_ablations(
            catalog_path=args.catalog,
            dataset_path=args.dataset,
        )
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


__all__ = [
    "ABLATION_VARIANT_NAMES",
    "FULL_VARIANT_NAME",
    "NoAmbiguousPreservationState",
    "NoCompatibleAccumulationState",
    "NoNoPreferencePreservationState",
    "NoTargetedCorrectionState",
    "TRANSITION_COMPARISON_EDGES",
    "TRANSITION_POLICY_NAMES",
    "default_variant_registry",
    "paired_transition_comparisons",
    "run_state_transition_ablations",
    "transition_policies",
]


if __name__ == "__main__":
    raise SystemExit(main())
