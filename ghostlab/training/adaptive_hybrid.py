from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ghostlab.research.firewall import runtime_profile
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.retrieval.filters import ConstraintAuthorityResult
from ghostlab.retrieval.gbdt import (
    GBDTFeatureStore,
    LambdaMARTModel,
    fit_lambdamart,
)
from ghostlab.retrieval.multi_route import MergedCandidatePool
from ghostlab.retrieval.union_features import UNION_FEATURES, UnionFeatureStore
from ghostlab.runtime.adaptive_components import SemanticRankingResult
from ghostlab.runtime.adaptive_config import (
    AdaptiveHybridConfig,
    DiverseDenseTrackConfig,
    UnionRankerConfig,
)
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent


@dataclass(frozen=True)
class AdaptiveRankingGroup:
    sample_id: str
    source: str
    scenario_type: str
    turn: int
    route: str
    overloaded: bool
    query: str
    candidates: tuple[str, ...]
    labels: tuple[int, ...]
    matrix: NDArray[np.float64]


class IdentitySemanticRanker:
    """Training collector stub; candidate construction precedes semantic ranking."""

    def rank(self, query: str, ranking: list[str]) -> SemanticRankingResult:
        del query
        return SemanticRankingResult(
            ranking=tuple(ranking),
            changed=False,
            elapsed_ms=0.0,
            backend="training_identity",
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_config(
    config: AdaptiveHybridConfig, *, disable_browsing_safe: bool = False
) -> AdaptiveHybridConfig:
    browsing = config.browsing
    if disable_browsing_safe:
        browsing = DiverseDenseTrackConfig(
            **{
                **config.browsing.model_dump(),
                "safe_ranker_backend": "deterministic",
                "safe_ranker_model_path": None,
                "safe_ranker_model_sha256": None,
            }
        )
    return config.model_copy(
        update={
            "browsing": browsing,
            "union_ranker": UnionRankerConfig(
                backend="deterministic", model_path=None, model_sha256=None
            ),
        }
    )


def collect_adaptive_ranking_groups(
    *,
    samples: Mapping[str, dict],
    origins: Mapping[str, str] | None = None,
    categories: dict[str, list[str]],
    products: dict[str, dict],
    catalog_path: str | Path,
    config: AdaptiveHybridConfig,
    project_root: str | Path,
    features: GBDTFeatureStore,
    disable_browsing_safe: bool = False,
) -> tuple[dict[str, list[AdaptiveRankingGroup]], dict[str, int]]:
    """Replay exact runtime pools without exposing labels to the runtime agent."""
    agent = AdaptiveHybridAgent(
        catalog_path,
        collection_config(config, disable_browsing_safe=disable_browsing_safe),
        project_root=project_root,
        semantic_ranker=IdentitySemanticRanker(),  # type: ignore[arg-type]
    )
    groups: dict[str, list[AdaptiveRankingGroup]] = defaultdict(list)
    union_features = UnionFeatureStore(features)
    turns = candidate_pools = ordinary_pools = target_pools = rows = 0
    buying_pools = browsing_pools = overload_turns = 0
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        agent.reset(observation.session_id, runtime_profile(sample))
        target = str(sample["ground_truth"]["parent_asin"])
        while not environment.done:
            turns += 1
            before = len(agent.candidate_snapshots)
            response = agent.respond(
                observation.session_id,
                observation.user_message,
                observation.turn,
                observation.top_k,
            )
            if agent.traces[-1].overloaded:
                overload_turns += 1
            if len(agent.candidate_snapshots) > before:
                snapshot = agent.candidate_snapshots[-1]
                candidate_pools += 1
                ordinary_pools += not snapshot.overloaded
                buying_pools += snapshot.route == "buying"
                browsing_pools += snapshot.route == "browsing"
                if target in snapshot.candidates:
                    target_pools += 1
                    labels = tuple(
                        int(identifier == target) for identifier in snapshot.candidates
                    )
                    groups[sample_id].append(
                        AdaptiveRankingGroup(
                            sample_id=sample_id,
                            source=(origins or {}).get(sample_id, "unspecified"),
                            scenario_type=str(sample.get("scenario_type") or "unknown"),
                            turn=snapshot.turn,
                            route=snapshot.route,
                            overloaded=snapshot.overloaded,
                            query=snapshot.query,
                            candidates=snapshot.candidates,
                            labels=labels,
                            matrix=union_features.matrix(
                                snapshot.query,
                                MergedCandidatePool(
                                    route=snapshot.route,  # type: ignore[arg-type]
                                    candidates=snapshot.evidence,
                                ),
                                UNION_FEATURES,
                                authority=ConstraintAuthorityResult(
                                    ranking=snapshot.candidates,
                                    decisions=(),
                                    confirmed_match_count=snapshot.confirmed_match_count,
                                    unknown_count=snapshot.unknown_constraint_count,
                                    soft_preference_count=snapshot.soft_preference_count,
                                    violation_count=0,
                                ),
                                profile_terms=snapshot.profile_terms,
                            ),
                        )
                    )
                    rows += len(snapshot.candidates)
            next_observation = environment.step(
                {
                    "message": response["message"],
                    "ask_attribute": response["ask_attribute"],
                    "recommendations": [],
                }
            )
            if next_observation is not None:
                observation = next_observation
        if len(groups) % 100 == 0 and sample_id in groups:
            print(
                "adaptive collection progress: "
                f"sessions_with_groups={len(groups)} "
                f"candidate_pools={candidate_pools} rows={rows}",
                flush=True,
            )
    return dict(groups), {
        "sessions": len(samples),
        "trajectory_turns": turns,
        "candidate_pools": candidate_pools,
        "ordinary_merged_pools": ordinary_pools,
        "overload_target_pools": sum(
            group.overloaded for items in groups.values() for group in items
        ),
        "buying_pools": buying_pools,
        "browsing_pools": browsing_pools,
        "overload_turns": overload_turns,
        "pools_with_target": target_pools,
        "sessions_with_groups": len(groups),
        "ranking_groups": sum(len(items) for items in groups.values()),
        "candidate_rows": rows,
    }


def ranking_dataset(
    groups: Mapping[str, Sequence[AdaptiveRankingGroup]],
    sample_ids: set[str],
    *,
    route: str | None = None,
    overloaded: bool | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[int]]:
    selected = [
        group
        for sample_id in sorted(sample_ids)
        for group in groups.get(sample_id, ())
        if (route is None or group.route == route)
        and (overloaded is None or group.overloaded is overloaded)
    ]
    if not selected:
        raise ValueError("adaptive ranking dataset cannot be empty")
    return (
        np.vstack([group.matrix for group in selected]),
        np.concatenate(
            [np.asarray(group.labels, dtype=np.int64) for group in selected]
        ),
        [len(group.labels) for group in selected],
    )


def evaluate_group_ordering(
    groups: Mapping[str, Sequence[AdaptiveRankingGroup]],
    sample_ids: set[str],
    model: LambdaMARTModel | None,
    *,
    route: str | None = None,
    overloaded: bool | None = None,
    source: str | None = None,
    scenario_type: str | None = None,
) -> dict[str, float | int]:
    reciprocal: list[float] = []
    hits = 0
    count = 0
    for sample_id in sorted(sample_ids):
        for group in groups.get(sample_id, ()):
            if route is not None and group.route != route:
                continue
            if overloaded is not None and group.overloaded is not overloaded:
                continue
            if source is not None and group.source != source:
                continue
            if scenario_type is not None and group.scenario_type != scenario_type:
                continue
            if model is None:
                order = np.arange(len(group.candidates))
            else:
                scores = model.predict(group.matrix)
                order = np.argsort(-scores, kind="stable")
            ranked_labels = np.asarray(group.labels, dtype=np.int64)[order]
            positive = np.flatnonzero(ranked_labels > 0)
            if len(positive) != 1:
                raise ValueError("each adaptive group must contain one target")
            rank = int(positive[0]) + 1
            hits += rank <= 10
            reciprocal.append(1.0 / rank)
            count += 1
    if count == 0:
        raise ValueError("adaptive ordering evaluation cannot be empty")
    return {
        "groups": count,
        "hit_rate_at_10": hits / count,
        "mrr": statistics.fmean(reciprocal),
    }


def train_adaptive_union_model(
    groups: Mapping[str, Sequence[AdaptiveRankingGroup]],
    training_ids: set[str],
    validation_ids: set[str],
    *,
    max_rounds: int,
    seed: int,
) -> LambdaMARTModel:
    return fit_lambdamart(
        *ranking_dataset(groups, training_ids, overloaded=False),
        candidate_id="adaptive_union_source_aware_v2",
        feature_names=UNION_FEATURES,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        max_rounds=max_rounds,
        early_stopping_rounds=12,
        validation=ranking_dataset(groups, validation_ids, overloaded=False),
        seed=seed,
    )


__all__ = [
    "AdaptiveRankingGroup",
    "IdentitySemanticRanker",
    "collect_adaptive_ranking_groups",
    "collection_config",
    "evaluate_group_ordering",
    "ranking_dataset",
    "sha256_file",
    "train_adaptive_union_model",
]
