from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from joblib import dump  # type: ignore[import-untyped]
from numpy.typing import NDArray

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from ghostlab.competition.contract import AgentProtocol
from ghostlab.research.firewall import runtime_profile
from ghostlab.research.technique_suite import (
    UnifiedTechniqueConfig,
    build_suite_agent,
)
from ghostlab.retrieval.constraint_gbdt import (
    CONSTRAINT_METADATA_FEATURES,
    ConstraintContext,
    ConstraintGBDTFeatureStore,
)
from ghostlab.retrieval.residual import (
    DERIVED_FEATURES,
    FEATURE_SETS,
    RESIDUAL_FEATURES,
    TECHNIQUE_ID,
    ResidualPolicy,
)
from ghostlab.runtime.unified_experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState
from ghostlab.training.protocol import FitReceipt, FitRequest


@dataclass(frozen=True)
class ResidualFitConfig:
    feature_set: str
    model_variant: str
    regularization: float
    policy: ResidualPolicy

    @classmethod
    def from_suite(cls, config: UnifiedTechniqueConfig) -> ResidualFitConfig:
        if not config.residual_reranker_enabled:
            raise ValueError("residual fit requires the technique switch")
        return cls(
            feature_set=config.residual_feature_set,
            model_variant=config.residual_model_variant,
            regularization=config.residual_regularization,
            policy=ResidualPolicy(
                rerank_depth=config.residual_rerank_depth,
                model_weight=config.residual_model_weight,
                minimum_expected_gain=config.residual_minimum_expected_gain,
                minimum_probability_margin=(config.residual_minimum_probability_margin),
                maximum_moved_ids=config.residual_maximum_moved_ids,
            ),
        )


@dataclass(frozen=True)
class _TrainingTurn:
    features: NDArray[np.float64]
    labels: NDArray[np.int64]


@dataclass(frozen=True)
class MeanProbabilityModel:
    """Serializable equal-weight ensemble with the sklearn probability API."""

    members: tuple[object, ...]

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        predictions = [
            np.asarray(member.predict_proba(features), dtype=np.float64)  # type: ignore[attr-defined]
            for member in self.members
        ]
        return np.mean(predictions, axis=0)


def _config_hash(config: UnifiedTechniqueConfig) -> str:
    value = config.model_dump_json(exclude={"residual_model_asset"})
    return hashlib.sha256(value.encode()).hexdigest()


def _trace_features(
    agent: ExperimentalAgent,
    feature_store: ConstraintGBDTFeatureStore,
    session_id: str,
    turn: int,
    ranking: tuple[str, ...],
) -> NDArray[np.float64]:
    state = agent.sessions[session_id]
    query, retrieval_scores = agent.last_runtime_inputs[session_id]
    context = ConstraintContext.from_runtime(
        cast(ConversationState, state), turn=turn, retrieval_scores=retrieval_scores
    )
    matrix = feature_store.contextual_matrix(
        query, ranking, context, CONSTRAINT_METADATA_FEATURES
    )
    reciprocal = 1.0 / np.arange(1, len(ranking) + 1, dtype=np.float64)
    derived = np.column_stack(
        (
            reciprocal,
            reciprocal - 1.0,
            np.zeros(len(ranking), dtype=np.float64),
        )
    )
    if derived.shape[1] != len(DERIVED_FEATURES):
        raise RuntimeError("residual derived-feature schema drifted")
    return np.hstack((matrix, derived))


def _collect_training_turns(
    parent_config: UnifiedTechniqueConfig,
    *,
    sample_ids: frozenset[str],
    dataset_path: Path,
    catalog_path: Path,
) -> list[_TrainingTurn]:
    built = build_suite_agent(parent_config, catalog_path)
    if not isinstance(built, ExperimentalAgent):
        raise TypeError("residual training requires an experimental parent")
    agent = built
    catalog_ids, categories, products = catalog_index(catalog_path)
    feature_store = ConstraintGBDTFeatureStore(catalog_path)
    samples = [
        cast(dict[str, object], sample)
        for sample in load_jsonl(dataset_path)
        if str(sample["sample_id"]) in sample_ids
    ]
    if {str(sample["sample_id"]) for sample in samples} != set(sample_ids):
        raise ValueError("residual training IDs do not match the development data")
    result: list[_TrainingTurn] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = f"residual-fit:{sample_id}"
        agent.reset(session_id, runtime_profile(cast(dict, sample)))
        target = str(cast(dict, sample["ground_truth"])["parent_asin"])
        intent, behavior = materialize_hidden_fields(cast(dict, sample), products)
        effective = {**sample, "intent_card": intent, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            cast(dict, effective),
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        for turn in range(1, 11):
            response = agent.respond(session_id, message, turn, 10)
            ranking = tuple(
                normalize_recommendations(response.get("recommendations"), catalog_ids)
            )
            features = _trace_features(agent, feature_store, session_id, turn, ranking)
            labels = np.asarray(
                [
                    int(override_applied and identifier == target)
                    for identifier in ranking
                ],
                dtype=np.int64,
            )
            result.append(_TrainingTurn(features, labels))
            if override_applied and target in ranking:
                break
            if turn == 10:
                break
            override = cast(dict, effective).get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
            else:
                message, boundary_used = customer_reply(
                    cast(dict, effective),
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
    return result


def _rank_aware_dataset(
    turns: list[_TrainingTurn], feature_set: str
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    try:
        selected_names = FEATURE_SETS[feature_set]
    except KeyError as error:
        raise ValueError(f"unknown residual feature set: {feature_set}") from error
    indices = [RESIDUAL_FEATURES.index(name) for name in selected_names]
    matrices: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    weights: list[NDArray[np.float64]] = []
    for turn in turns:
        matrices.append(turn.features[:, indices])
        labels.append(turn.labels)
        group_weights = np.full(len(turn.labels), 0.35, dtype=np.float64)
        positives = np.flatnonzero(turn.labels > 0)
        if len(positives) == 1:
            group_weights.fill(1.0)
            target_index = int(positives[0])
            rank = target_index + 1
            group_weights[target_index] = 1.0 + 4.0 * (1.0 - 1.0 / rank)
        weights.append(group_weights)
    return np.vstack(matrices), np.concatenate(labels), np.concatenate(weights)


def _fit_probability_model(
    variant: str,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    weights: NDArray[np.float64],
    *,
    regularization: float,
    seed: int,
) -> object:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,  # type: ignore[import-untyped]
    )
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    def logistic() -> object:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
        )
        model.fit(features, labels, logisticregression__sample_weight=weights)
        return model

    tree_variants = {
        "hist_gbdt_d2_lr005": (2, 0.05),
        "hist_gbdt_d3_lr005": (3, 0.05),
        "hist_gbdt_d3_lr01": (3, 0.1),
        "ensemble_logistic_gbdt_d2_lr005": (2, 0.05),
        "ensemble_logistic_gbdt_d3_lr005": (3, 0.05),
        "ensemble_logistic_gbdt_d3_lr01": (3, 0.1),
    }
    if variant == "regularized_logistic":
        return logistic()
    try:
        depth, learning_rate = tree_variants[variant]
    except KeyError as error:
        raise ValueError(f"unknown residual model variant: {variant}") from error
    tree = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=depth,
            max_iter=100,
            min_samples_leaf=20,
            l2_regularization=regularization,
            random_state=seed,
        ),
    )
    tree.fit(
        features,
        labels,
        histgradientboostingclassifier__sample_weight=weights,
    )
    if variant.startswith("ensemble_"):
        return MeanProbabilityModel((logistic(), tree))
    return tree


class ResidualFoldTrainer:
    """Fit one content-addressed residual asset behind an explicit fold firewall."""

    def __init__(self, dataset_path: Path, catalog_path: Path) -> None:
        self.dataset_path = dataset_path
        self.catalog_path = catalog_path

    def fit(
        self,
        request: FitRequest,
        output_path: Path,
        *,
        candidate_config: UnifiedTechniqueConfig,
    ) -> FitReceipt:
        if request.technique_id != TECHNIQUE_ID:
            raise ValueError("residual trainer received another technique")
        fit_config = ResidualFitConfig.from_suite(candidate_config)
        parent_config = candidate_config.model_copy(
            update={
                "residual_reranker_enabled": False,
                "residual_model_asset": None,
            }
        )
        turns = _collect_training_turns(
            parent_config,
            sample_ids=frozenset(request.train_sample_ids),
            dataset_path=self.dataset_path,
            catalog_path=self.catalog_path,
        )
        features, labels, weights = _rank_aware_dataset(turns, fit_config.feature_set)
        if len(np.unique(labels)) != 2:
            raise ValueError("residual fit requires positive and negative labels")

        model = _fit_probability_model(
            fit_config.model_variant,
            features,
            labels,
            weights,
            regularization=fit_config.regularization,
            seed=request.seed,
        )
        payload = {
            "schema_version": 1,
            "technique_id": TECHNIQUE_ID,
            "parent_config_sha256": _config_hash(parent_config),
            "feature_names": list(FEATURE_SETS[fit_config.feature_set]),
            "policy": asdict(fit_config.policy),
            "fit_request": request.model_dump(mode="json"),
            "model": model,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        dump(payload, temporary)
        temporary.replace(output_path)
        return FitReceipt.from_fit(request, output_path)


class FoldDispatchAgent:
    """Route each replay session to the model that excluded its sample fold."""

    def __init__(self, agents_by_sample: dict[str, AgentProtocol]) -> None:
        if not agents_by_sample:
            raise ValueError("fold dispatch requires at least one sample")
        self.agents_by_sample = dict(agents_by_sample)
        self.sessions: dict[str, AgentProtocol] = {}

    @staticmethod
    def _sample_id(session_id: str) -> str:
        prefix = "replay_"
        if not session_id.startswith(prefix):
            raise ValueError("fold dispatch received an unknown session ID")
        return session_id[len(prefix) :]

    def reset(self, session_id: str, user_profile: dict) -> None:
        sample_id = self._sample_id(session_id)
        try:
            agent = self.agents_by_sample[sample_id]
        except KeyError as error:
            raise ValueError("sample has no fold-fitted residual agent") from error
        self.sessions[session_id] = agent
        agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        try:
            agent = self.sessions[session_id]
        except KeyError as error:
            raise ValueError("fold dispatch session was not reset") from error
        return agent.respond(session_id, user_message, turn, top_k)
