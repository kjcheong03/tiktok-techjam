from __future__ import annotations

import unittest

from ghostlab.evaluation.splits import freeze_split, largest_remainder


class SplitTest(unittest.TestCase):
    def test_largest_remainder_preserves_total(self) -> None:
        result = largest_remainder({"easy": 3, "hard": 2, "medium": 5}, 4)
        self.assertEqual(sum(result.values()), 4)
        self.assertLessEqual(result["hard"], 2)

    def test_public_shape_is_disjoint_and_deterministic(self) -> None:
        scenarios = {
            "buying": 80,
            "browsing": 80,
            "intent_override": 30,
            "boundary": 10,
        }
        rows: list[dict[str, object]] = []
        index = 0
        for scenario, count in scenarios.items():
            for offset in range(count):
                index += 1
                rows.append(
                    {
                        "sample_id": f"sample_{index:04d}",
                        "scenario_type": scenario,
                        "difficulty_bucket": ("easy", "medium", "hard")[offset % 3],
                        "user_profile": {"group": offset % 7},
                    }
                )
        adaptive, nested, guarded = freeze_split(rows, "seed")
        second = freeze_split(rows, "seed")
        self.assertEqual((adaptive, nested, guarded), second)
        self.assertEqual(len(adaptive["sample_ids"]), 150)
        self.assertEqual(len(guarded["sample_ids"]), 50)
        self.assertFalse(set(adaptive["sample_ids"]) & set(guarded["sample_ids"]))
        self.assertEqual(sum(map(len, nested["outer_folds"])), 150)


if __name__ == "__main__":
    unittest.main()
