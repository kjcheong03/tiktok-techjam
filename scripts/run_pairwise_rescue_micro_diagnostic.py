from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from evaluator.local_evaluator import catalog_index, evaluate
from ghostlab.retrieval.cross_encoder import CrossEncoderReranker, product_passage
from ghostlab.retrieval.filters import CoverageAwareFilter
from ghostlab.runtime.adaptive_components import SemanticRankingResult
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent
from ghostlab.state.v2_view import V2StateView
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from scripts.compare_local_llm_rankers import lineage_safe_sample_ids
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)
EXPECTED_ORDERED_SESSION_HASH = (
    "e017d1f82e97813721d8d5b9856d0a9a647099bc81df0f1b65e45a22a51622cb"
)
EXPECTED_BASELINE_POOL_HASH = (
    "c18cc7f6ce30b42a2f2fb745a2c43470d242c70246110bb86edd801f83e9bcd5"
)
EXPECTED_OPPORTUNITIES = {
    ("independent_public_like_0067", 2): ("B01M70CKOE", 11, 270),
    ("independent_public_like_0140", 2): ("B01MQ2BIIU", 11, 320),
}
MODEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "qwen2.5-0.5b-instruct": {
        "path": "artifacts/cache/models/qwen2.5-0.5b-instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
    "smollm2-1.7b-instruct": {
        "path": "artifacts/cache/models/smollm2-1.7b-instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
    },
}
MINILM = {
    "path": "artifacts/cache/models/ms-marco-MiniLM-L6-v2",
    # Match the downloaded asset manifest/receipt used by this isolated control.
    "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
}


def structured_shopper_context(query: str, view: V2StateView) -> str:
    """Render only observable State V2 evidence with explicit semantic roles."""

    positive = view.positive_constraints()
    negative = view.negative_constraints()

    def values(attributes: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                value
                for attribute in attributes
                for value in positive.get(attribute, ())
                if value
            )
        )

    categories = values(("category",))
    intended_use = values(("occasion", "use_case"))
    reserved = {"category", "occasion", "use_case"}
    preferences = tuple(
        (attribute, tuple(dict.fromkeys(value for value in items if value)))
        for attribute, items in sorted(positive.items())
        if attribute not in reserved and any(items)
    )
    exclusions = tuple(
        (attribute, tuple(dict.fromkeys(value for value in items if value)))
        for attribute, items in sorted(negative.items())
        if any(items)
    )

    lines = [f"Current request: {query}"]
    if categories:
        lines.append(f"Category: {', '.join(categories)}")
    if intended_use:
        lines.append(f"Intended use: {', '.join(intended_use)}")
    if preferences:
        lines.append(
            "Positive preferences: "
            + "; ".join(
                f"{attribute}={', '.join(items)}"
                for attribute, items in preferences
            )
        )
    if exclusions:
        lines.append(
            "Explicit exclusions: "
            + "; ".join(
                f"{attribute}={', '.join(items)}"
                for attribute, items in exclusions
            )
        )
    else:
        lines.append("Explicit exclusions: none")
    return "\n".join(lines)


def shopper_context(query: str, view: V2StateView, mode: str) -> str:
    if mode == "flattened":
        return query
    if mode == "structured":
        return structured_shopper_context(query, view)
    raise ValueError(f"unknown shopper-context mode: {mode}")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if ".cache" in item.parts:
            continue
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TurnPoint:
    sample_id: str
    runtime_session_id: str
    turn: int
    query: str
    route: str
    ranking: tuple[str, ...]
    target_id: str
    view: V2StateView

    @property
    def target_rank(self) -> int | None:
        return (
            self.ranking.index(self.target_id) + 1
            if self.target_id in self.ranking
            else None
        )


@dataclass(frozen=True)
class Comparison:
    left_id: str
    right_id: str
    winner_id: str | None
    confidence: float


@dataclass(frozen=True)
class PromotionResult:
    ranking: tuple[str, ...]
    promoted_id: str | None
    comparisons: tuple[Comparison, ...]
    elapsed_ms: float
    failure_reason: str | None = None
    constraint_violation_count: int = 0


class BatchPairwiseScorer:
    model_id: str

    def compare(
        self, query: str, ordered_pairs: Sequence[tuple[str, str]]
    ) -> tuple[Comparison, ...]:
        raise NotImplementedError


class MiniLMPairwiseScorer(BatchPairwiseScorer):
    model_id = "minilm-cross-encoder-control"

    def __init__(self, catalog_path: Path, *, root: Path = ROOT) -> None:
        model_path = root / MINILM["path"]
        if not model_path.is_dir():
            raise FileNotFoundError(f"missing MiniLM asset: {model_path}")
        self.reranker = CrossEncoderReranker(
            catalog_path,
            model_name=str(model_path),
            revision=MINILM["revision"],
            cache_folder=root / "artifacts/cache/cross_encoder",
            local_files_only=True,
        )

    def compare(
        self, query: str, ordered_pairs: Sequence[tuple[str, str]]
    ) -> tuple[Comparison, ...]:
        identifiers = list(
            dict.fromkeys(item for pair in ordered_pairs for item in pair)
        )
        scores = dict(
            zip(identifiers, self.reranker.scores(query, identifiers), strict=True)
        )
        rows = []
        for left, right in ordered_pairs:
            delta = scores[left] - scores[right]
            winner = (
                None
                if math.isclose(delta, 0.0, abs_tol=1e-7)
                else (left if delta > 0.0 else right)
            )
            rows.append(Comparison(left, right, winner, abs(delta)))
        return tuple(rows)


class DirectLogitPairwiseScorer(BatchPairwiseScorer):
    """Order-audited A/B/tie scorer using only next-token logits."""

    def __init__(
        self,
        model_id: str,
        documents: Mapping[str, str],
        *,
        root: Path = ROOT,
        batch_size: int = 12,
        max_length: int = 512,
    ) -> None:
        if model_id not in MODEL_DEFINITIONS:
            raise ValueError(f"unsupported micro-diagnostic model: {model_id}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        definition = MODEL_DEFINITIONS[model_id]
        model_path = root / definition["path"]
        if not model_path.is_dir():
            raise FileNotFoundError(f"missing local LLM asset: {model_path}")
        self.model_id = model_id
        self.documents = documents
        self.batch_size = batch_size
        self.max_length = max_length
        self.torch = torch
        use_mps = torch.backends.mps.is_available()
        self.device = torch.device("mps" if use_mps else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=definition["revision"],
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options: dict[str, object] = {
            "revision": definition["revision"],
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if use_mps:
            options["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **options).to(
            self.device
        )
        self.model.eval()
        self.label_tokens = tuple(
            self._single_token_label(value) for value in ("A", "B", "C")
        )
        if len(set(self.label_tokens)) != 3:
            raise ValueError("A/B/tie labels do not resolve to distinct tokens")

    def _single_token_label(self, label: str) -> int:
        for candidate in (f" {label}", label):
            encoded = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                return int(encoded[0])
        raise ValueError(f"pairwise label is not a single token: {label}")

    def _prompt(self, query: str, left: str, right: str) -> str:
        content = (
            "Judge which product better matches the shopping request and all its "
            "constraints. Answer A if Product A is better, B if Product B is better, "
            "or C only if they are tied. Answer with exactly one letter.\n"
            f"Request: {query}\n"
            f"Product A: {self.documents.get(left, '')}\n"
            f"Product B: {self.documents.get(right, '')}"
        )
        if getattr(self.tokenizer, "chat_template", None):
            options = (
                {"enable_thinking": False} if "qwen3" in self.model_id.lower() else {}
            )
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **options,
            )
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("chat template returned an invalid prompt")
            return rendered
        return f"{content}\nDecision:"

    def compare(
        self, query: str, ordered_pairs: Sequence[tuple[str, str]]
    ) -> tuple[Comparison, ...]:
        prompts = [self._prompt(query, left, right) for left, right in ordered_pairs]
        output: list[Comparison] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            batch = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                logits = self.model(**batch).logits
            positions = batch["attention_mask"].sum(dim=1) - 1
            rows = self.torch.arange(len(positions), device=self.device)
            selected = logits[rows, positions][:, list(self.label_tokens)].float()
            probabilities = self.torch.softmax(selected, dim=-1).cpu()
            for offset, values in enumerate(probabilities):
                left, right = ordered_pairs[start + offset]
                ordered = self.torch.argsort(values, descending=True)
                choice = int(ordered[0])
                confidence = float(values[ordered[0]] - values[ordered[1]])
                winner = left if choice == 0 else right if choice == 1 else None
                output.append(Comparison(left, right, winner, confidence))
        if len(output) != len(ordered_pairs):
            raise ValueError("pairwise scorer returned an incomplete batch")
        return tuple(output)


class PairwiseBoundaryPromoter:
    def __init__(
        self, scorer: BatchPairwiseScorer, authority: CoverageAwareFilter
    ) -> None:
        self.scorer = scorer
        self.authority = authority

    @staticmethod
    def ordered_pairs(ranking: Sequence[str]) -> tuple[tuple[str, str], ...]:
        if len(ranking) < 13:
            return ()
        incumbents = ranking[8:10]
        outsiders = ranking[10:13]
        return tuple(
            pair
            for outsider in outsiders
            for incumbent in incumbents
            for pair in ((outsider, incumbent), (incumbent, outsider))
        )

    def promote(
        self, query: str, ranking: Sequence[str], view: V2StateView
    ) -> PromotionResult:
        started = time.perf_counter()
        original = tuple(ranking)
        pairs = self.ordered_pairs(original)
        if not pairs:
            return PromotionResult(
                original, None, (), (time.perf_counter() - started) * 1000.0
            )
        try:
            comparisons = self.scorer.compare(query, pairs)
            if len(comparisons) != 12:
                raise ValueError(
                    "the boundary protocol requires 12 ordered comparisons"
                )
            winners = {
                (item.left_id, item.right_id): item.winner_id for item in comparisons
            }
            eligible = []
            for outsider in original[10:13]:
                if all(
                    winners[(outsider, incumbent)] == outsider
                    and winners[(incumbent, outsider)] == outsider
                    for incumbent in original[8:10]
                ):
                    eligible.append(outsider)
            if not eligible:
                return PromotionResult(
                    original,
                    None,
                    comparisons,
                    (time.perf_counter() - started) * 1000.0,
                )
            promoted = eligible[0]
            candidate_check = self.authority.enforce([promoted], view)
            if (
                candidate_check.violation_count
                or promoted not in candidate_check.ranking
            ):
                return PromotionResult(
                    original,
                    None,
                    comparisons,
                    (time.perf_counter() - started) * 1000.0,
                    constraint_violation_count=max(1, candidate_check.violation_count),
                )
            changed = list(original)
            changed.remove(promoted)
            changed.insert(9, promoted)  # protect ranks 1-8 and retain rank 9
            full_check = self.authority.enforce(changed, view)
            removed = len(changed) - len(full_check.ranking)
            if removed:
                return PromotionResult(
                    original,
                    None,
                    comparisons,
                    (time.perf_counter() - started) * 1000.0,
                    constraint_violation_count=removed,
                )
            return PromotionResult(
                tuple(changed),
                promoted,
                comparisons,
                (time.perf_counter() - started) * 1000.0,
            )
        except Exception as error:  # noqa: BLE001 - declared no-op fallback
            return PromotionResult(
                original,
                None,
                (),
                (time.perf_counter() - started) * 1000.0,
                failure_reason=f"{type(error).__name__}: {error}",
            )


class RecordingNoopAgent(AdaptiveHybridAgent):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.recorded: dict[
            tuple[str, int], tuple[str, str, tuple[str, ...], V2StateView]
        ] = {}

    def _semantic_rank(
        self,
        query: str,
        ranking: list[str],
        route: object,
        view: V2StateView,
        **kwargs: object,
    ) -> SemanticRankingResult:
        del kwargs
        session_id = view.session_id
        self.recorded[(session_id, view.turn)] = (
            query,
            str(cast(Any, route).route),
            tuple(ranking),
            view,
        )
        return SemanticRankingResult(
            tuple(ranking), False, 0.0, "micro_noop", "micro_noop"
        )


class PairwiseExperimentalAgent(AdaptiveHybridAgent):
    def __init__(
        self,
        *args: object,
        promoter: PairwiseBoundaryPromoter,
        context_mode: str = "flattened",
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.promoter = promoter
        self.context_mode = context_mode
        self.promotion_results: dict[tuple[str, int], PromotionResult] = {}

    def _semantic_rank(
        self,
        query: str,
        ranking: list[str],
        route: object,
        view: V2StateView,
        **kwargs: object,
    ) -> SemanticRankingResult:
        del route, kwargs
        context = shopper_context(query, view, self.context_mode)
        result = self.promoter.promote(context, ranking, view)
        self.promotion_results[(view.session_id, view.turn)] = result
        return SemanticRankingResult(
            result.ranking,
            result.ranking != tuple(ranking),
            result.elapsed_ms,
            f"micro_pairwise:{self.promoter.scorer.model_id}",
            "micro_pairwise",
            result.failure_reason,
        )


def candidate_pool_hash(agent: AdaptiveHybridAgent) -> tuple[str, int]:
    traces = {
        (item.session_id, item.turn): item
        for item in agent.traces
        if item.semantic_decision_reached and not item.overloaded
    }
    first_by_session = {}
    for item in agent.candidate_snapshots:
        if (item.session_id, item.turn) in traces:
            first_by_session.setdefault(item.session_id, item)
    rows = [
        {
            "ordinal": ordinal,
            "turn": item.turn,
            "candidates": list(item.pre_semantic_candidates or item.candidates),
        }
        for ordinal, item in enumerate(first_by_session.values())
    ]
    return canonical_sha256(rows), len(rows)


def build_turn_points(
    agent: RecordingNoopAgent,
    ordered_ids: Sequence[str],
    samples: Mapping[str, dict[str, Any]],
) -> tuple[TurnPoint, ...]:
    runtime_ids = tuple(agent.sessions)
    if len(runtime_ids) != len(ordered_ids):
        raise RuntimeError("runtime session mapping is incomplete")
    sample_by_runtime = dict(zip(runtime_ids, ordered_ids, strict=True))
    points = []
    for (runtime_id, turn), (query, route, ranking, view) in agent.recorded.items():
        sample_id = sample_by_runtime[runtime_id]
        target = str(samples[sample_id]["ground_truth"]["parent_asin"])
        points.append(
            TurnPoint(sample_id, runtime_id, turn, query, route, ranking, target, view)
        )
    return tuple(
        sorted(points, key=lambda item: (ordered_ids.index(item.sample_id), item.turn))
    )


def select_micro_cases(
    points: Sequence[TurnPoint],
) -> tuple[tuple[TurnPoint, ...], dict[str, object]]:
    by_coordinate = {(item.sample_id, item.turn): item for item in points}
    opportunities = []
    for coordinate, expected in EXPECTED_OPPORTUNITIES.items():
        if coordinate not in by_coordinate:
            raise RuntimeError(f"audited opportunity is absent: {coordinate}")
        point = by_coordinate[coordinate]
        target, rank, pool_size = expected
        actual = (point.target_id, point.target_rank, len(point.ranking))
        if actual != (target, rank, pool_size):
            raise RuntimeError(
                f"audited opportunity drift for {coordinate}: expected "
                f"{(target, rank, pool_size)}, got {actual}"
            )
        opportunities.append(point)

    used = {item.sample_id for item in opportunities}
    negatives: list[TurnPoint] = []
    for opportunity in opportunities:
        for _ in range(2):
            eligible = [
                item
                for item in points
                if matched_negative_eligible(item, opportunity, used)
            ]
            if not eligible:
                raise RuntimeError(
                    "two deterministic rank-1-8 matched negatives are required "
                    "for every opportunity"
                )
            matched = min(
                eligible,
                key=lambda item: (
                    abs(len(item.ranking) - len(opportunity.ranking)),
                    abs(cast(int, item.target_rank) - 8),
                    item.sample_id,
                ),
            )
            negatives.append(matched)
            used.add(matched.sample_id)
    cases = (*opportunities, *negatives)
    evidence = {
        "opportunity_count": len(opportunities),
        "matched_negative_count": len(negatives),
        "cases": [
            {
                "sample_id": item.sample_id,
                "turn": item.turn,
                "target_id": item.target_id,
                "target_rank": item.target_rank,
                "pool_size": len(item.ranking),
                "kind": "opportunity" if item in opportunities else "matched_negative",
            }
            for item in cases
        ],
        "selection_sha256": canonical_sha256(
            [(item.sample_id, item.turn) for item in cases]
        ),
    }
    return tuple(cases), evidence


def matched_negative_eligible(
    point: TurnPoint, opportunity: TurnPoint, used_sample_ids: set[str]
) -> bool:
    return (
        point.turn == opportunity.turn
        and point.route == opportunity.route
        and point.sample_id not in used_sample_ids
        and point.target_rank is not None
        and 1 <= point.target_rank <= 8
    )


def rank_metrics(
    cases: Sequence[TurnPoint], rankings: Sequence[Sequence[str]]
) -> dict[str, float]:
    ranks = [
        ranking.index(case.target_id) + 1 if case.target_id in ranking else None
        for case, ranking in zip(cases, rankings, strict=True)
    ]
    return {
        "hit_rate_at_10": sum(rank is not None and rank <= 10 for rank in ranks)
        / len(ranks),
        "mrr": statistics.fmean(
            0.0 if rank is None or rank > 10 else 1.0 / rank for rank in ranks
        ),
    }


def summarize_micro_arm(
    model_id: str,
    cases: Sequence[TurnPoint],
    results: Sequence[PromotionResult],
    *,
    initialization_ms: float,
) -> dict[str, object]:
    baseline = rank_metrics(cases, [item.ranking for item in cases])
    after = rank_metrics(cases, [item.ranking for item in results])
    opportunity_coordinates = set(EXPECTED_OPPORTUNITIES)
    rescued = 0
    false_promotions = 0
    harmful = 0
    for case, result in zip(cases, results, strict=True):
        before = case.target_rank
        after_rank = (
            result.ranking.index(case.target_id) + 1
            if case.target_id in result.ranking
            else None
        )
        is_opportunity = (case.sample_id, case.turn) in opportunity_coordinates
        rescued += bool(
            is_opportunity
            and before
            and before > 10
            and after_rank
            and after_rank <= 10
        )
        false_promotions += bool(not is_opportunity and result.promoted_id is not None)
        harmful += bool(
            before and before <= 10 and (after_rank is None or after_rank > 10)
        )
    failures = Counter(
        item.failure_reason for item in results if item.failure_reason is not None
    )
    latencies = sorted(item.elapsed_ms for item in results)
    return {
        "model_id": model_id,
        "cases": len(cases),
        "comparisons_expected": len(cases) * 12 if model_id != "no-op" else 0,
        "comparisons_completed": sum(len(item.comparisons) for item in results),
        "promotions": sum(item.promoted_id is not None for item in results),
        "actual_rescues": rescued,
        "false_promotions_on_matched_negatives": false_promotions,
        "harmful_target_demotions": harmful,
        "constraint_revalidation_violations": sum(
            item.constraint_violation_count for item in results
        ),
        "failure_counts": dict(sorted(failures.items())),
        "fallback_to_noop_count": sum(
            item.failure_reason is not None for item in results
        ),
        "parsing_failures": 0,
        "baseline_micro_metrics": baseline,
        "micro_metrics": after,
        "hit_rate_at_10_delta": after["hit_rate_at_10"] - baseline["hit_rate_at_10"],
        "mrr_delta": after["mrr"] - baseline["mrr"],
        "initialization_ms": initialization_ms,
        "mean_case_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p95_case_latency_ms": latencies[
            min(len(latencies) - 1, int(0.95 * len(latencies)))
        ]
        if latencies
        else 0.0,
        "case_results": [
            {
                "sample_id": case.sample_id,
                "turn": case.turn,
                "target_rank_before": case.target_rank,
                "target_rank_after": (
                    result.ranking.index(case.target_id) + 1
                    if case.target_id in result.ranking
                    else None
                ),
                "promoted_id": result.promoted_id,
                "comparison_count": len(result.comparisons),
                "failure_reason": result.failure_reason,
                "elapsed_ms": result.elapsed_ms,
            }
            for case, result in zip(cases, results, strict=True)
        ],
    }


def release_model(scorer: object) -> None:
    del scorer
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def load_documents(catalog_path: Path) -> dict[str, str]:
    documents = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            documents[str(product["parent_asin"])] = product_passage(product)
    return documents


def evaluate_micro_arm(
    model_id: str,
    cases: Sequence[TurnPoint],
    catalog_path: Path,
    authority: CoverageAwareFilter,
    documents: Mapping[str, str],
    *,
    context_mode: str = "flattened",
) -> tuple[dict[str, object], BatchPairwiseScorer | None]:
    if model_id == "no-op":
        results = [PromotionResult(item.ranking, None, (), 0.0) for item in cases]
        return summarize_micro_arm(
            model_id, cases, results, initialization_ms=0.0
        ), None
    started = time.perf_counter()
    scorer: BatchPairwiseScorer
    if model_id == MiniLMPairwiseScorer.model_id:
        scorer = MiniLMPairwiseScorer(catalog_path)
    else:
        scorer = DirectLogitPairwiseScorer(model_id, documents)
    initialization_ms = (time.perf_counter() - started) * 1000.0
    promoter = PairwiseBoundaryPromoter(scorer, authority)
    results = [
        promoter.promote(
            shopper_context(item.query, item.view, context_mode),
            item.ranking,
            item.view,
        )
        for item in cases
    ]
    summary = summarize_micro_arm(
        model_id, cases, results, initialization_ms=initialization_ms
    )
    summary["shopper_context_mode"] = context_mode
    return summary, scorer


def full_replay_summary(
    scorer: BatchPairwiseScorer,
    samples: Sequence[dict[str, Any]],
    ordered_ids: Sequence[str],
    base_config: Path,
    catalog_path: Path,
    baseline_result: Mapping[str, object],
    *,
    context_mode: str = "flattened",
) -> dict[str, object]:
    identifiers, categories, products = catalog_index(catalog_path)
    authority = CoverageAwareFilter(catalog_path)
    agent = PairwiseExperimentalAgent(
        catalog_path,
        load_adaptive_hybrid_config(base_config),
        project_root=ROOT,
        promoter=PairwiseBoundaryPromoter(scorer, authority),
        context_mode=context_mode,
    )
    started = time.perf_counter()
    result = evaluate(
        cast(Agent, agent), list(samples), identifiers, categories, products
    )
    elapsed = time.perf_counter() - started
    runtime_to_sample = dict(zip(tuple(agent.sessions), ordered_ids, strict=True))
    samples_by_id = {str(item["sample_id"]): item for item in samples}
    rescues = harmful = false_promotions = 0
    for snapshot in agent.candidate_snapshots:
        key = (snapshot.session_id, snapshot.turn)
        decision = agent.promotion_results.get(key)
        sample_id = runtime_to_sample.get(snapshot.session_id)
        if decision is None or sample_id is None or decision.promoted_id is None:
            continue
        target = str(samples_by_id[sample_id]["ground_truth"]["parent_asin"])
        before_ranking = snapshot.pre_semantic_candidates
        after_ranking = decision.ranking
        before = before_ranking.index(target) + 1 if target in before_ranking else None
        after = after_ranking.index(target) + 1 if target in after_ranking else None
        rescues += bool(before and before > 10 and after and after <= 10)
        harmful += bool(before and before <= 10 and (after is None or after > 10))
        false_promotions += decision.promoted_id != target
    decisions = tuple(agent.promotion_results.values())
    failures = Counter(item.failure_reason for item in decisions if item.failure_reason)
    return {
        "model_id": scorer.model_id,
        "sessions": len(samples),
        "ordered_session_ids_sha256": canonical_sha256(tuple(ordered_ids)),
        "baseline_hit_rate_at_10": baseline_result["hit_rate_at_10"],
        "baseline_mrr": baseline_result["mrr"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "hit_rate_at_10_delta": float(result["hit_rate_at_10"])
        - float(baseline_result["hit_rate_at_10"]),
        "mrr_delta": float(result["mrr"]) - float(baseline_result["mrr"]),
        "promotions": sum(item.promoted_id is not None for item in decisions),
        "actual_rescues": rescues,
        "false_promotions": false_promotions,
        "harmful_target_demotions": harmful,
        "constraint_revalidation_violations": sum(
            item.constraint_violation_count for item in decisions
        ),
        "output_constraint_violations": sum(
            item.output_constraint_violations for item in agent.traces
        ),
        "comparisons_completed": sum(len(item.comparisons) for item in decisions),
        "failure_counts": dict(sorted(failures.items())),
        "fallback_to_noop_count": sum(
            item.failure_reason is not None for item in decisions
        ),
        "mean_turn_latency_ms": statistics.fmean(item.elapsed_ms for item in decisions)
        if decisions
        else 0.0,
        "shopper_context_mode": context_mode,
        "elapsed_seconds": elapsed,
    }


def asset_preflight() -> dict[str, object]:
    definitions = {**MODEL_DEFINITIONS, MiniLMPairwiseScorer.model_id: MINILM}
    rows = []
    for model_id, definition in definitions.items():
        path = ROOT / definition["path"]
        rows.append(
            {
                "model_id": model_id,
                "path": definition["path"],
                "available": path.is_dir() and (path / "config.json").is_file(),
                "tree_sha256": tree_sha256(path) if path.is_dir() else None,
                "revision": definition["revision"],
            }
        )
    return {
        "all_available": all(bool(item["available"]) for item in rows),
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated rank-boundary pairwise rescue micro-diagnostic"
    )
    parser.add_argument(
        "--base-config", default="configs/adaptive_hybrid_1a_3b_1650_final_v1.json"
    )
    parser.add_argument(
        "--protected-config",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument(
        "--output", default="artifacts/reports/pairwise_rescue_micro_diagnostic.json"
    )
    parser.add_argument("--stage-one-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--shopper-context",
        choices=("flattened", "structured"),
        default="flattened",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    base_config = ROOT / args.base_config
    protected_config = ROOT / args.protected_config
    protected_hash_before = file_sha256(protected_config)
    preflight = asset_preflight()
    protocol = {
        "research_only": True,
        "holdout_accessed": False,
        "promotion_allowed": False,
        "protected_ranks": [1, 8],
        "incumbent_ranks": [9, 10],
        "outsider_ranks": [11, 13],
        "ordered_comparisons_per_turn": 12,
        "orders": ["A/B", "B/A"],
        "max_promotions_per_turn": 1,
        "grid_search": False,
        "shopper_context": args.shopper_context,
        "stage_gate": (
            "Run both LLMs on the six-case micro-set; stop if neither rescues. "
            "If at least one rescues, replay only the best rescuing LLM on all 60 "
            "identical development sessions."
        ),
    }
    if args.plan_only:
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "asset_preflight": preflight,
                    "protected_config_sha256": protected_hash_before,
                },
                indent=2,
            )
        )
        return
    if not preflight["all_available"]:
        raise RuntimeError("one or more pinned local model assets are unavailable")

    corpus = load_adaptive_training_corpus(ROOT, DATASETS)
    manifest = load_lineage_manifest(ROOT / args.lineage_manifest, corpus)
    development = subset_corpus(corpus, manifest, "development")
    folds = lineage_safe_sample_ids(development, manifest, 60)
    ordered_ids = tuple(item for fold in folds for item in fold)
    ordered_hash = canonical_sha256(ordered_ids)
    if ordered_hash != EXPECTED_ORDERED_SESSION_HASH:
        raise RuntimeError(f"development session drift: {ordered_hash}")
    samples = [development.samples[sample_id] for sample_id in ordered_ids]
    catalog_path = ROOT / "data/catalog.jsonl"
    identifiers, categories, products = catalog_index(catalog_path)
    baseline_agent = RecordingNoopAgent(
        catalog_path,
        load_adaptive_hybrid_config(base_config),
        project_root=ROOT,
    )
    baseline_result = evaluate(
        cast(Agent, baseline_agent), samples, identifiers, categories, products
    )
    pool_hash, pool_turns = candidate_pool_hash(baseline_agent)
    if pool_hash != EXPECTED_BASELINE_POOL_HASH:
        raise RuntimeError(f"baseline candidate-pool drift: {pool_hash}")
    points = build_turn_points(baseline_agent, ordered_ids, development.samples)
    cases, selection = select_micro_cases(points)
    documents = load_documents(catalog_path)
    authority = CoverageAwareFilter(catalog_path)

    arms = []
    for model_id in (
        "no-op",
        MiniLMPairwiseScorer.model_id,
        "qwen2.5-0.5b-instruct",
        "smollm2-1.7b-instruct",
    ):
        summary, scorer = evaluate_micro_arm(
            model_id,
            cases,
            catalog_path,
            authority,
            documents,
            context_mode=args.shopper_context,
        )
        arms.append(summary)
        if scorer is not None:
            release_model(scorer)
            scorer = None

    llm_rescuers = [
        item
        for item in arms
        if item["model_id"] in MODEL_DEFINITIONS and int(item["actual_rescues"]) > 0
    ]
    winner: str | None = None
    full_replay: dict[str, object] | None = None
    if llm_rescuers:
        selected = min(
            llm_rescuers,
            key=lambda item: (
                -int(item["actual_rescues"]),
                int(item["harmful_target_demotions"]),
                int(item["false_promotions_on_matched_negatives"]),
                float(item["p95_case_latency_ms"]),
                str(item["model_id"]),
            ),
        )
        winner = str(selected["model_id"])
        if not args.stage_one_only:
            winner_scorer = DirectLogitPairwiseScorer(winner, documents)
            full_replay = full_replay_summary(
                winner_scorer,
                samples,
                ordered_ids,
                base_config,
                catalog_path,
                baseline_result,
                context_mode=args.shopper_context,
            )
            release_model(winner_scorer)

    protected_hash_after = file_sha256(protected_config)
    if protected_hash_after != protected_hash_before:
        raise RuntimeError("protected selected configuration changed during diagnostic")
    report = {
        "protocol": protocol,
        "asset_preflight": preflight,
        "development_evidence": {
            "fold_sizes": [len(fold) for fold in folds],
            "ordered_session_ids_sha256": ordered_hash,
            "candidate_pool_sha256": pool_hash,
            "candidate_pool_turns": pool_turns,
            "baseline_hit_rate_at_10": baseline_result["hit_rate_at_10"],
            "baseline_mrr": baseline_result["mrr"],
        },
        "micro_set": selection,
        "stage_one_arms": arms,
        "stage_one_llm_winner": winner,
        "stage_two_status": (
            "not_requested"
            if args.stage_one_only
            else "complete"
            if full_replay
            else "stopped_no_llm_rescue"
        ),
        "stage_two_full_replay": full_replay,
        "protected_config": {
            "path": args.protected_config,
            "sha256_before": protected_hash_before,
            "sha256_after": protected_hash_after,
            "unchanged": protected_hash_before == protected_hash_after,
        },
        "promotion_performed": False,
        "holdout_accessed": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
