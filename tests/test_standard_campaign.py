from __future__ import annotations

import unittest

from scripts.run_standard_campaign import canonical_hash, generate_candidates


class StandardCampaignTest(unittest.TestCase):
    def test_generation_is_bounded_unique_and_deterministic(self) -> None:
        first = generate_candidates(100, [17, 29, 43])
        second = generate_candidates(100, [17, 29, 43])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len({canonical_hash(item) for item in first}), 100)


if __name__ == "__main__":
    unittest.main()
