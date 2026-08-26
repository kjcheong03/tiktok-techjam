from __future__ import annotations

import unittest

from ghostlab.optimization.meta_search import CandidateEvidence, cost_aware_search


def evidence(index: int, *, family: str = "a", score: float | None = None):
    value = score if score is not None else index / 100
    return CandidateEvidence(str(index), family, index % 3 + 1, value, value)


class CostAwareSearchTests(unittest.TestCase):
    def test_random_respects_full_evaluation_budget(self) -> None:
        result = cost_aware_search(
            [evidence(index) for index in range(20)],
            strategy="random",
            budget=500,
            f0_cost=10,
            f2_cost=100,
            seed=17,
        )
        self.assertEqual(result.promoted, 5)
        self.assertEqual(result.screened, 0)
        self.assertEqual(result.session_evaluations, 500)

    def test_beam_reuses_screening_cost_on_promotion(self) -> None:
        result = cost_aware_search(
            [evidence(index) for index in range(20)],
            strategy="beam",
            budget=500,
            f0_cost=10,
            f2_cost=100,
            seed=17,
        )
        self.assertEqual(result.screened, 20)
        self.assertEqual(result.promoted, 3)
        self.assertEqual(result.session_evaluations, 470)

    def test_allocated_search_samples_multiple_families(self) -> None:
        pool = [
            *(evidence(index, family="good", score=0.8) for index in range(10)),
            *(evidence(index + 10, family="bad", score=0.1) for index in range(10)),
        ]
        result = cost_aware_search(
            pool,
            strategy="allocated",
            budget=500,
            f0_cost=10,
            f2_cost=100,
            seed=29,
        )
        self.assertEqual(result.selected_score, 0.8)
        self.assertLessEqual(result.session_evaluations, 500)

    def test_tie_band_prefers_simpler_candidate(self) -> None:
        pool = [
            CandidateEvidence("complex", "a", 5, 0.81, 0.81),
            CandidateEvidence("simple", "a", 1, 0.805, 0.805),
        ]
        result = cost_aware_search(
            pool,
            strategy="grid",
            budget=200,
            f0_cost=10,
            f2_cost=100,
            seed=1,
            tie_band=0.01,
        )
        self.assertEqual(result.best_observed_id, "complex")
        self.assertEqual(result.selected_id, "simple")


if __name__ == "__main__":
    unittest.main()
