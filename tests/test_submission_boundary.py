from __future__ import annotations

import json
import unittest
from pathlib import Path

from ghostlab.research.firewall import FORBIDDEN_RUNTIME_NAMES


class SubmissionBoundaryTest(unittest.TestCase):
    def test_submission_modules_do_not_import_research_or_evaluator_code(self) -> None:
        for path in (
            Path("starter/agent.py"),
            Path("ghostlab/runtime/agent.py"),
            Path("ghostlab/runtime/compiled.py"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("ghostlab.research", source, path)
            self.assertNotIn("evaluator", source, path)

    def test_compiled_policy_contains_no_research_labels(self) -> None:
        config = json.loads(
            Path("configs/compiled_policy.json").read_text(encoding="utf-8")
        )
        names: set[str] = set()

        def visit(value: object) -> None:
            if isinstance(value, dict):
                names.update(str(key).casefold() for key in value)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(config)
        self.assertFalse(names & FORBIDDEN_RUNTIME_NAMES)


if __name__ == "__main__":
    unittest.main()
