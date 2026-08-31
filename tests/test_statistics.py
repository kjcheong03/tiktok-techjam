from __future__ import annotations

import unittest

from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)


class StatisticalTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_ordered(self) -> None:
        first = bootstrap_mean_interval([0.1, 0.2, 0.3], resamples=1000)
        second = bootstrap_mean_interval([0.1, 0.2, 0.3], resamples=1000)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_large_consistent_delta_has_small_pvalue(self) -> None:
        self.assertLess(paired_randomization_pvalue([1.0] * 20, resamples=2000), 0.01)


if __name__ == "__main__":
    unittest.main()
