from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from baseline.state import ASK_ORDER
from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.research.counterfactual import Action, CounterfactualEvaluator
from ghostlab.research.firewall import session_set_hash
from ghostlab.research.question_policy import (
    QuestionFeatures,
    fit_question_table,
)
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.runtime.experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState

ROOT = Path(__file__).resolve().parents[1]
ALL_ACTIONS: tuple[Action, ...] = (None, *ASK_ORDER, "other")
NO_OTHER_ACTIONS: tuple[Action, ...] = (None, *ASK_ORDER)
FEATURE_SETS = (
    ("has_initial_constraint",),
    ("has_initial_constraint", "critical_rater"),
    ("has_initial_constraint", "critical_rater", "material_profile_tag"),
)


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = set(nested["adaptive_sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in adaptive_ids]
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    evaluator = CounterfactualEvaluator(
        lambda: ExperimentalAgent(
            ROOT / "data/catalog.jsonl",
            state_variant="multi",
            question_variant="fixed",
        ),
        categories,
        products,
        continuation_id="manual_strong_multi_fixed_v1",
    )
    outcomes = []
    features: dict[str, QuestionFeatures] = {}
    for sample in samples:
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        state = ConversationState(observation.session_id, sample["user_profile"])
        state.observe(observation.user_message, 1)
        sample_id = str(sample["sample_id"])
        profile = sample["user_profile"]
        tags = {str(value).casefold() for value in profile.get("preference_tags", [])}
        features[sample_id] = QuestionFeatures(
            has_initial_constraint=len(state.active_values()) > 1,
            critical_rater=str(profile.get("rating_style", "")).casefold()
            == "critical",
            material_profile_tag="material" in tags,
        )
        outcomes.extend(evaluator.branches(sample, ALL_ACTIONS))

    by_sample_action = {(item.sample_id, item.action): item.reward for item in outcomes}
    feature_reports = {}
    selected_tables = {}
    for action_set_name, actions in (
        ("with_other", ALL_ACTIONS),
        ("no_other", NO_OTHER_ACTIONS),
    ):
        for feature_names in FEATURE_SETS:
            fold_rewards = []
            predictions: list[tuple[str, Action, float]] = []
            fold_tables = []
            for fold_index, outer_ids in enumerate(nested["outer_folds"]):
                outer = set(outer_ids)
                training_outcomes = [
                    item for item in outcomes if item.sample_id not in outer
                ]
                table = fit_question_table(
                    training_outcomes, features, feature_names, actions
                )
                fold_tables.append(
                    {
                        "outer_fold": fold_index,
                        "default_action": table.default_action,
                        "cells": {
                            str(key): value for key, value in table.cells.items()
                        },
                    }
                )
                rewards = []
                for sample_id in sorted(outer):
                    action = table.predict(features[sample_id])
                    reward = by_sample_action[(sample_id, action)]
                    rewards.append(reward)
                    predictions.append((sample_id, action, reward))
                fold_rewards.append(statistics.fmean(rewards))
            key = f"{action_set_name}:{'+'.join(feature_names)}"
            feature_reports[key] = {
                "mean_oof_reward": round(
                    statistics.fmean(reward for _, _, reward in predictions), 6
                ),
                "fold_rewards": [round(value, 6) for value in fold_rewards],
                "action_counts": dict(
                    sorted(Counter(str(action) for _, action, _ in predictions).items())
                ),
                "fold_tables": fold_tables,
            }
        eligible = {
            key: value
            for key, value in feature_reports.items()
            if key.startswith(f"{action_set_name}:")
        }
        best_score = max(value["mean_oof_reward"] for value in eligible.values())
        chosen_key = next(
            key
            for key in (
                f"{action_set_name}:{'+'.join(names)}" for names in FEATURE_SETS
            )
            if eligible[key]["mean_oof_reward"] >= best_score - 0.01
        )
        final_names = tuple(chosen_key.split(":", 1)[1].split("+"))
        final_table = fit_question_table(outcomes, features, final_names, actions)
        selected_tables[action_set_name] = {
            "feature_names": list(final_names),
            "default_action": final_table.default_action,
            "cells": {str(key): value for key, value in final_table.cells.items()},
            "oof_reward": eligible[chosen_key]["mean_oof_reward"],
            "selection_rule": "simplest within 0.01 of best OOF",
        }

    material_reward = statistics.fmean(
        by_sample_action[(sample_id, "material")] for sample_id in adaptive_ids
    )
    report = {
        "phase": 9,
        "split": "nested_v1_oof",
        "sample_count": len(samples),
        "training_session_hash": session_set_hash(adaptive_ids),
        "manual_first_action_material_reward": round(material_reward, 6),
        "feature_reports": feature_reports,
        "selected_tables": selected_tables,
        "counterfactual_cache_hits": evaluator.cache_hits,
    }
    output = ROOT / "artifacts/reports/phase9_question_policy.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manual_first_action_material_reward": report[
                    "manual_first_action_material_reward"
                ],
                "selected_tables": selected_tables,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
