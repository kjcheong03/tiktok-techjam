from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class GBDTQuestionInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "configs/experiments/gbdt_question_interaction_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_freezes_one_outcome_blind_candidate(self) -> None:
        self.assertEqual(
            self.manifest["parent_commit"],
            "cbfd7d5dd595c5637608ba28f46f57777c7e153e",
        )
        self.assertFalse(self.manifest["holdout_accessed"])
        self.assertEqual(self.manifest["candidate_limit"], 1)
        self.assertFalse(self.manifest["candidate"]["hyperparameter_search"])
        self.assertFalse(self.manifest["candidate"]["old_policy_weights_reused"])
        self.assertEqual(self.manifest["candidate"]["l2"], 1.0)

    def test_promotion_gates_are_executable_and_frozen(self) -> None:
        gates = self.manifest["promotion_gates"]
        self.assertEqual(gates["minimum_score_delta"], 0.005)
        self.assertEqual(gates["minimum_nonnegative_outer_folds"], 4)
        self.assertEqual(gates["minimum_scenario_score_delta"], -0.005)
        self.assertEqual(gates["maximum_failure_count"], 0)
        self.assertTrue(gates["determinism_required"])
        self.assertTrue(gates["control_reproduction_required"])

    def test_outer_folds_are_disjoint_and_cover_adaptive_set(self) -> None:
        split = json.loads(
            (ROOT / "configs/splits/nested_v1.json").read_text(encoding="utf-8")
        )
        adaptive = {str(value) for value in split["adaptive_sample_ids"]}
        folds = [{str(value) for value in fold} for fold in split["outer_folds"]]
        self.assertEqual(len(folds), 5)
        self.assertEqual(set().union(*folds), adaptive)
        for index, fold in enumerate(folds):
            self.assertFalse(
                fold & set().union(*(folds[:index] + folds[index + 1 :])),
            )

    def test_private_holdout_is_not_present(self) -> None:
        forbidden = (
            ROOT / "data/private_set.jsonl",
            ROOT / "data/f3.jsonl",
            ROOT / "data/holdout.jsonl",
        )
        self.assertTrue(all(not path.exists() for path in forbidden))

    def test_completed_report_preserves_control_and_parks_candidate(self) -> None:
        report = json.loads(
            (ROOT / "artifacts/reports/gbdt_question_interaction_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["holdout_accessed"])
        self.assertTrue(report["matched_control"]["reproduced"])
        self.assertEqual(
            report["matched_control"]["metrics"]["recommended_technical_score"],
            0.861417,
        )
        self.assertEqual(
            report["candidate_oof"]["metrics"]["recommended_technical_score"],
            0.847744,
        )
        self.assertTrue(
            all(value < 0 for value in report["candidate_oof"]["fold_score_deltas"])
        )
        self.assertEqual(report["decision"]["status"], "PARKED_INTERACTION")
        self.assertFalse(report["decision"]["promotion_rule_passed"])

    def test_report_hashes_and_training_fold_firewall(self) -> None:
        report = json.loads(
            (ROOT / "artifacts/reports/gbdt_question_interaction_v1.json").read_text(
                encoding="utf-8"
            )
        )
        for relative, expected in report["code_sha256"].items():
            self.assertEqual(digest(ROOT / relative), expected)
        label_path = ROOT / report["counterfactual_evidence"]["label_path"]
        if not label_path.exists():
            self.skipTest("raw counterfactual table remains on its archived branch")
        self.assertEqual(
            digest(label_path), report["counterfactual_evidence"]["label_sha256"]
        )
        split = json.loads(
            (ROOT / "configs/splits/nested_v1.json").read_text(encoding="utf-8")
        )
        outer_folds = [{str(value) for value in fold} for fold in split["outer_folds"]]
        rows = [json.loads(line) for line in label_path.read_text().splitlines()]
        self.assertEqual(
            len(rows), report["counterfactual_evidence"]["training_only_label_rows"]
        )
        self.assertTrue(
            all(
                str(row["sample_id"]) not in outer_folds[int(row["outer_fold"])]
                and row["training_only"] is True
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
