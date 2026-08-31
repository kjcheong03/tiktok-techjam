from __future__ import annotations

import unittest

from ghostlab.optimization.patches import (
    PolicyPatch,
    SearchPolicyConfig,
    TypedOperation,
    crossover,
    materialize_patch,
)
from ghostlab.optimization.racing import racing_decide


def patch(identifier: str, field: str, value: object) -> PolicyPatch:
    return PolicyPatch(
        patch_id=identifier,
        parent_policy_id="parent",
        mutation_family="test",
        hypothesis="test",
        operations=(TypedOperation(field=field, value=value),),  # type: ignore[arg-type]
        falsification_condition="no gain",
        risk="low",
    )


class PatchRacingTest(unittest.TestCase):
    def parent(self) -> SearchPolicyConfig:
        return SearchPolicyConfig(
            state_variant="multi",
            question_variant="other_always",
            retrieval_route="keyword",
        )

    def test_typed_patch_materializes_and_hashes(self) -> None:
        result = materialize_patch(
            self.parent(), patch("p", "state_variant", "compressed")
        )
        self.assertEqual(result.state_variant, "compressed")
        self.assertEqual(result.policy_hash(), result.policy_hash())

    def test_crossover_allows_independent_fields_only(self) -> None:
        result = crossover(
            self.parent(),
            patch("a", "state_variant", "compressed"),
            patch("b", "negative_evidence", False),
        )
        self.assertEqual(result.state_variant, "compressed")
        self.assertFalse(result.negative_evidence)
        with self.assertRaises(ValueError):
            crossover(
                self.parent(),
                patch("c", "state_variant", "single"),
                patch("d", "state_variant", "raw_history"),
            )

    def test_racing_preserves_novelty_and_rejects_catastrophe(self) -> None:
        self.assertEqual(racing_decide([-0.2] * 10, fidelity="f0"), "REJECT")
        self.assertEqual(
            racing_decide([-0.001] * 10, fidelity="f0", behavior_novelty=0.8),
            "NOVELTY_RESERVE",
        )
        self.assertEqual(racing_decide([0.1] * 50, fidelity="f1"), "PROMOTE")


if __name__ == "__main__":
    unittest.main()
