from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ghostlab.optimization.evidence import (
    TechniqueDecisionRecord,
    TechniqueDecisionStore,
)
from scripts.validate_decision_ledger import LEDGER, ROOT, validate_records


class DecisionLedgerTest(unittest.TestCase):
    def test_repository_ledger_is_valid(self) -> None:
        records = TechniqueDecisionStore(LEDGER).read()
        self.assertGreaterEqual(len(records), 20)
        self.assertEqual(validate_records(records, ROOT), [])

    def test_store_rejects_duplicate_identifiers(self) -> None:
        record = TechniqueDecisionRecord(
            decision_id="D001",
            technique_id="example.v1",
            family="example",
            status="NOT_TESTED",
            hypothesis="Example hypothesis.",
            mechanism="Example mechanism.",
            diagnosis="Not run.",
            evidence_refs=("docs/champion_checkpoint.md",),
            decided_at="2026-08-26",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = TechniqueDecisionStore(Path(directory) / "ledger.jsonl")
            store.append(record)
            with self.assertRaisesRegex(ValueError, "duplicate decision_id"):
                store.append(record.model_copy(update={"technique_id": "other.v1"}))


if __name__ == "__main__":
    unittest.main()
