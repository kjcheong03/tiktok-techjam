from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.policy.signals import RetrievalSignals
from ghostlab.runtime.adaptive_components import (
    RouteDecision,
    SemanticActivationDecision,
    SemanticActivationPolicy,
)
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.state.v2_view import V2StateView
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "never",
    "always",
    "browsing_all",
    "browsing_refined",
    "browsing_ambiguous",
    "buying_all",
    "buying_semantic_constraints",
    "semantic_constraints",
    "selective",
)


class StudyActivationPolicy:
    """Research-only gates; only `selective` is submission-eligible."""

    def __init__(
        self,
        mode: str,
        selected: SemanticActivationPolicy,
        *,
        maximum_margin: float = 0.02,
        minimum_entropy: float = 0.85,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown semantic activation mode: {mode}")
        self.mode = mode
        self.selected = selected
        self.maximum_margin = maximum_margin
        self.minimum_entropy = minimum_entropy

    def decide(
        self,
        route: RouteDecision,
        view: V2StateView,
        *,
        overloaded: bool,
        signals: RetrievalSignals | None = None,
    ) -> SemanticActivationDecision:
        if self.mode == "selective":
            return self.selected.decide(
                route,
                view,
                overloaded=overloaded,
                signals=signals,
            )
        if self.mode == "never":
            return SemanticActivationDecision(False, "research_never")
        if self.mode == "always":
            return SemanticActivationDecision(True, "research_always")
        if self.mode == "browsing_all":
            return SemanticActivationDecision(
                route.route == "browsing",
                "research_browsing" if route.route == "browsing" else "research_buying",
            )
        if self.mode == "browsing_ambiguous":
            active = (
                route.route == "browsing"
                and not overloaded
                and signals is not None
                and signals.top1_margin is not None
                and signals.normalized_entropy is not None
                and signals.top1_margin <= self.maximum_margin
                and signals.normalized_entropy >= self.minimum_entropy
            )
            return SemanticActivationDecision(
                active,
                "research_ambiguous_browsing"
                if active
                else "research_confident_or_unavailable_browsing",
            )
        if self.mode == "buying_all":
            active = route.route == "buying" and not overloaded
            return SemanticActivationDecision(
                active,
                "research_bounded_buying" if active else "research_not_buying",
            )
        if self.mode == "buying_semantic_constraints":
            semantic_attributes = {"occasion", "use_case", "style", "feature", "other"}
            active = (
                route.route == "buying"
                and not overloaded
                and any(
                    constraint.polarity == "include"
                    and constraint.attribute in semantic_attributes
                    for constraint in view.active_constraints
                )
            )
            return SemanticActivationDecision(
                active,
                "research_semantic_buying"
                if active
                else "research_exact_or_not_buying",
            )
        if self.mode == "semantic_constraints":
            if overloaded:
                return SemanticActivationDecision(False, "research_overload")
            semantic_attributes = {"occasion", "use_case", "style", "feature", "other"}
            active = route.route == "browsing" or any(
                constraint.polarity == "include"
                and constraint.attribute in semantic_attributes
                for constraint in view.active_constraints
            )
            return SemanticActivationDecision(
                active,
                "research_semantic_constraint"
                if active
                else "research_exact_buying",
            )
        active = route.route == "browsing" and not overloaded
        return SemanticActivationDecision(
            active,
            "research_refined_browsing"
            if active
            else "research_not_refined_browsing",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare research-only local-LLM activation policies"
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-margin", type=float, default=0.02)
    parser.add_argument("--minimum-entropy", type=float, default=0.85)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = load_adaptive_hybrid_config(
        ROOT / "configs/adaptive_hybrid_1a_3b_v1.json"
    )
    catalog_path = ROOT / "data/catalog.jsonl"
    samples = load_jsonl(ROOT / args.dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    identifiers, categories, products = catalog_index(catalog_path)
    agent = AdaptiveHybridAgent(catalog_path, config, project_root=ROOT)
    agent.semantic_activation = StudyActivationPolicy(  # type: ignore[assignment]
        args.mode,
        agent.semantic_activation,
        maximum_margin=args.maximum_margin,
        minimum_entropy=args.minimum_entropy,
    )
    result = evaluate(
        cast(Agent, agent), samples, identifiers, categories, products
    )

    counts: Counter[str] = Counter()
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    session_ids = tuple(agent.sessions)
    for session_id, session_result in zip(
        session_ids, result["sessions"], strict=True
    ):
        scenario = str(session_result["scenario_type"])
        for trace in (item for item in agent.traces if item.session_id == session_id):
            status = (
                "skipped"
                if trace.semantic_backend.startswith("skipped:")
                else "activated"
            )
            counts[status] += 1
            scenario_counts[scenario][status] += 1

    report = {
        key: value for key, value in result.items() if key != "sessions"
    }
    report["semantic_activation_study"] = {
        "mode": args.mode,
        "submission_eligible": args.mode == "selective",
        "architecture_capability_present": True,
        "activation_counts": dict(counts),
        "scenario_activation_counts": {
            scenario: dict(values)
            for scenario, values in sorted(scenario_counts.items())
        },
        "config_sha256": config.canonical_hash(),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
