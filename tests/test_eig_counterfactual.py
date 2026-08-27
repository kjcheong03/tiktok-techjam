from __future__ import annotations

import unittest

from ghostlab.research.counterfactual import ActionOutcome
from ghostlab.research.eig_counterfactual import fit_reward_voi_calibration


class EIGCounterfactualTests(unittest.TestCase):
    def test_calibration_is_relative_to_stop_and_shrunk(self) -> None:
        outcomes = [
            ActionOutcome("a", None, 0.2, False, None, None),
            ActionOutcome("a", "color", 0.8, True, 2, 1),
            ActionOutcome("b", None, 0.4, False, None, None),
            ActionOutcome("b", "color", 0.6, True, 3, 2),
        ]
        calibration = fit_reward_voi_calibration(outcomes, shrinkage=2)
        self.assertEqual(calibration.training_sessions, 2)
        self.assertAlmostEqual(calibration.action_adjustments["color"], 0.2)


if __name__ == "__main__":
    unittest.main()
