from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ghostlab.runtime.trace import (
    JsonlTraceSink,
    RuntimeTrace,
    assert_trace_has_no_research_labels,
)


class RuntimeTraceTest(unittest.TestCase):
    def test_trace_hashes_query_and_round_trips(self) -> None:
        trace = RuntimeTrace.from_observable(
            session_id="s",
            turn=1,
            policy_id="p",
            ask_attribute="other",
            retrieval_route="keyword",
            query="private user text",
            top_ids=["a", "b"],
        )
        self.assertNotIn("private user text", trace.model_dump_json())
        assert_trace_has_no_research_labels(trace)
        with tempfile.TemporaryDirectory() as directory:
            sink = JsonlTraceSink(Path(directory) / "trace.jsonl")
            sink.append(trace)
            self.assertEqual(sink.read(), [trace])


if __name__ == "__main__":
    unittest.main()
