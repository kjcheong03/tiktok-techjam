from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.optimization.search import (
    PolicyCandidate,
    interaction_gain,
    save_checkpoint,
    search,
    should_retest,
)


class OptimizerTest(unittest.TestCase):
    def candidates(self) -> list[PolicyCandidate]:
        return [
            PolicyCandidate(name=f"c{index}", family=str(index % 2), complexity=index)
            for index in range(5)
        ]

    def test_grid_recovers_known_optimum_and_obeys_budget(self) -> None:
        result = search(
            self.candidates(),
            lambda candidate, fidelity, seed: int(candidate.name[1:]),
            strategy="grid",
            budget=5,
            seed=1,
        )
        self.assertEqual(result.winner, "c4")
        self.assertEqual(len(result.evaluations), 5)

    def test_seeded_random_is_deterministic(self) -> None:
        objective = lambda candidate, fidelity, seed: int(candidate.name[1:])
        first = search(
            self.candidates(), objective, strategy="random", budget=3, seed=7
        )
        second = search(
            self.candidates(), objective, strategy="random", budget=3, seed=7
        )
        self.assertEqual(first, second)

    def test_deduplicates_canonical_candidates(self) -> None:
        duplicate = PolicyCandidate(name="c0", family="0", complexity=0)
        result = search(
            [duplicate, duplicate],
            lambda candidate, fidelity, seed: 1.0,
            strategy="grid",
            budget=5,
            seed=1,
        )
        self.assertEqual(len(result.evaluations), 1)

    def test_standalone_loser_can_be_retested_for_synergy(self) -> None:
        interaction = interaction_gain(0.5, 0.48, 0.55, 0.62)
        self.assertTrue(should_retest(-0.02, interaction))

    def test_checkpoint_is_complete_json(self) -> None:
        result = search(
            self.candidates(),
            lambda candidate, fidelity, seed: 1.0,
            strategy="grid",
            budget=1,
            seed=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(path, result)
            self.assertEqual(json.loads(path.read_text())["winner"], result.winner)


if __name__ == "__main__":
    unittest.main()
