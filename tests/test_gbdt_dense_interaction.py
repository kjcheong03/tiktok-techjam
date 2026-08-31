from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.policy.models import RankedCandidate, RankedCandidates
from ghostlab.retrieval.gbdt_dense import DeepGBDTAgent
from scripts.run_gbdt_dense_interaction import StageRecord, select_queries


class _Sparse:
    def search(self, query: str, limit: int, weights: tuple[float, ...]):
        del query, weights
        return RankedCandidates(
            items=tuple(
                RankedCandidate(parent_asin=value, route="keyword", rank=index)
                for index, value in enumerate(("s1", "shared", "s2")[:limit], 1)
            ),
            route="keyword",
            requested_k=limit,
            elapsed_ms=0.0,
        )


class _Dense:
    def search(self, query: str, limit: int):
        del query
        return RankedCandidates(
            items=tuple(
                RankedCandidate(parent_asin=value, route="dense", rank=index)
                for index, value in enumerate(("shared", "d1")[:limit], 1)
            ),
            route="dense",
            requested_k=limit,
            elapsed_ms=0.0,
        )


class _PassThroughQuality:
    def rerank(self, ranking: list[str], *, weight: float, rerank_k: int):
        del weight
        assert rerank_k == len(ranking)
        return list(ranking)


class _CaptureReranker:
    def __init__(self) -> None:
        self.ranking: list[str] = []

    def rerank(self, query: str, ranking: list[str], *, rerank_k: int):
        del query
        assert rerank_k == len(ranking)
        self.ranking = list(ranking)
        return list(ranking)


class GBDTDenseInteractionTests(unittest.TestCase):
    def _catalog(self, directory: str) -> Path:
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(
                json.dumps({"parent_asin": value, "title": value}) + "\n"
                for value in ("s1", "shared", "s2", "d1")
            ),
            encoding="utf-8",
        )
        return path

    def test_sparse_first_union_is_deduplicated_before_deep_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reranker = _CaptureReranker()
            agent = DeepGBDTAgent(
                self._catalog(directory),
                sparse=_Sparse(),
                dense=_Dense(),
                quality=_PassThroughQuality(),
                reranker=reranker,
                field_weights=(1.0,) * 6,
                question_order=("other",),
                dense_query_variant="raw_history",
            )
            agent.reset("session", {})
            response = agent.respond("session", "find item", 1, 10)
        self.assertEqual(reranker.ranking, ["s1", "shared", "s2", "d1"])
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            reranker.ranking,
        )
        self.assertEqual(agent.failure_count, 0)

    def test_sparse_only_arm_never_requires_dense_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reranker = _CaptureReranker()
            agent = DeepGBDTAgent(
                self._catalog(directory),
                sparse=_Sparse(),
                dense=None,
                quality=_PassThroughQuality(),
                reranker=reranker,
                field_weights=(1.0,) * 6,
                question_order=("other",),
                dense_query_variant=None,
            )
            agent.reset("session", {})
            agent.respond("session", "find item", 1, 10)
        self.assertEqual(reranker.ranking, ["s1", "shared", "s2"])

    def test_query_choice_uses_only_outer_training_complement(self) -> None:
        records = [
            StageRecord(
                sample_id="s0",
                target="t0",
                scenario_type="buying",
                turn=1,
                raw_query="q0",
                queries={
                    "raw_plus_active": "a0",
                    "negation_safe_structured": "n0",
                },
                sparse=(),
            ),
            StageRecord(
                sample_id="s1",
                target="t1",
                scenario_type="buying",
                turn=1,
                raw_query="q1",
                queries={
                    "raw_plus_active": "a1",
                    "negation_safe_structured": "n1",
                },
                sparse=(),
            ),
        ]
        rankings = {
            "raw_plus_active": [["t0"], ["miss"]],
            "negation_safe_structured": [["miss"], ["t1"]],
        }
        choices = select_queries(
            records,
            rankings,
            {"s0", "s1"},
            [{"s0"}, {"s1"}],
        )
        self.assertEqual(choices[0]["selected_query"], "negation_safe_structured")
        self.assertEqual(choices[1]["selected_query"], "raw_plus_active")
        self.assertNotIn("s0", choices[0]["selection_population_ids"])
        self.assertNotIn("s1", choices[1]["selection_population_ids"])

    def test_manifest_freezes_matched_depth_attribution_arms(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "configs/experiments/gbdt_dense_interaction_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(manifest["holdout_accessed"])
        self.assertEqual(manifest["candidate_limit"], 4)
        self.assertEqual(
            set(manifest["arms"]),
            {
                "A_current_gbdt_top50",
                "B_sparse_deep_gbdt",
                "C_raw_e5_union_deep_gbdt",
                "D_nested_query_e5_union_deep_gbdt",
            },
        )
        self.assertTrue(manifest["manifest_created_before_evaluation"])

    def test_report_preserves_control_and_parks_failed_interaction(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "artifacts/reports/gbdt_dense_interaction_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["holdout_accessed"])
        self.assertTrue(report["control_reproduction"]["sessions_exact"])
        self.assertEqual(
            report["arms"]["A"]["metrics"]["recommended_technical_score"],
            0.861417,
        )
        self.assertEqual(report["decision"], "PARK_STRUCTURED_DENSE_QUERY_INTERACTION")
        self.assertFalse(report["gate_for_arm_D"]["passed"])
        self.assertEqual(report["performance"]["peak_process_memory_mb"], 5133.9375)
        self.assertIn(
            "contention-affected",
            report["performance"]["latency_measurement_context"],
        )


if __name__ == "__main__":
    unittest.main()
