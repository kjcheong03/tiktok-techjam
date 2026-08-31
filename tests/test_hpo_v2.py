from __future__ import annotations

import unittest

from ghostlab.optimization.bohb import Observation, Parameter, suggest
from ghostlab.optimization.hyperband import Trial, successive_halving


class HPOTests(unittest.TestCase):
    def test_successive_halving_promotes_best_trials_deterministically(self) -> None:
        trials = tuple(
            Trial(str(index), (("weight", index / 10),)) for index in range(9)
        )
        results = successive_halving(
            trials,
            lambda trial, resource: float(dict(trial.parameters)["weight"]),
            resources=(10, 30, 90),
            reduction_factor=3,
        )
        final = [item for item in results if item.resource == 90]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].trial.trial_id, "8")

    def test_bohb_sampler_is_bounded_and_reproducible(self) -> None:
        space = (
            Parameter("weight", "float", 0.0, 1.0),
            Parameter("depth", "int", 1, 8),
            Parameter("route", "categorical", choices=("a", "b")),
        )
        observations = (
            Observation((("depth", 4), ("route", "b"), ("weight", 0.8)), 0.9),
            Observation((("depth", 2), ("route", "a"), ("weight", 0.2)), 0.1),
        )
        left = suggest(space, observations, seed=11, exploration_fraction=0.0)
        right = suggest(space, observations, seed=11, exploration_fraction=0.0)
        self.assertEqual(left, right)
        values = dict(left)
        self.assertTrue(0.0 <= float(values["weight"]) <= 1.0)
        self.assertTrue(1 <= int(values["depth"]) <= 8)
        self.assertIn(values["route"], {"a", "b"})


if __name__ == "__main__":
    unittest.main()
