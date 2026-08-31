from __future__ import annotations

import unittest

from ghostlab.research.counterfactual import ActionOutcome
from ghostlab.research.question_policy import QuestionFeatures, fit_question_table


class QuestionPolicyTest(unittest.TestCase):
    def test_table_uses_cell_action_and_global_fallback(self) -> None:
        features = {
            f"s{index}": QuestionFeatures(index < 5, False, False)
            for index in range(10)
        }
        outcomes = []
        for sample_id, feature in features.items():
            outcomes.extend(
                [
                    ActionOutcome(
                        sample_id,
                        "material",
                        1.0 if feature.has_initial_constraint else 0.0,
                        True,
                        1,
                        1,
                    ),
                    ActionOutcome(
                        sample_id,
                        "color",
                        0.0 if feature.has_initial_constraint else 1.0,
                        True,
                        1,
                        1,
                    ),
                ]
            )
        table = fit_question_table(
            outcomes, features, ("has_initial_constraint",), ("material", "color")
        )
        self.assertEqual(table.predict(features["s0"]), "material")
        self.assertEqual(table.predict(features["s9"]), "color")


if __name__ == "__main__":
    unittest.main()
