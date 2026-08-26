from __future__ import annotations

import unittest

from scripts.analyze_standard_campaign import spearman


class CampaignAnalysisTest(unittest.TestCase):
    def test_spearman_known_orders(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)


if __name__ == "__main__":
    unittest.main()
