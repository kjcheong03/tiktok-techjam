from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from ghostlab.policy.models import ModelAssetConfig, RuntimeConfig, TechniqueConfig
from ghostlab.policy.registry import Technique, TechniqueRegistry
from ghostlab.runtime.normalizer import normalize_identifiers, normalize_response


class RuntimeBoundaryTest(unittest.TestCase):
    def test_default_config_is_typed(self) -> None:
        path = Path("configs/techniques/baseline_v1.json")
        config = RuntimeConfig.model_validate_json(path.read_text(encoding="utf-8"))
        self.assertEqual(config.policy_id, "baseline_v1_keyword_state")
        self.assertEqual(config.techniques.retrieval_route, "keyword")

    def test_registry_is_lazy_and_validates_dependencies(self) -> None:
        built: list[str] = []
        registry = TechniqueRegistry()
        registry.register(Technique("keyword", lambda: built.append("keyword")))
        registry.register(
            Technique("dense", lambda: built.append("dense"), requires=("asset",))
        )
        registry.register(Technique("asset", lambda: built.append("asset")))
        registry.build({"keyword"})
        self.assertEqual(built, ["keyword"])
        with self.assertRaises(ValueError):
            registry.build({"dense"})

    def test_normalizer_removes_duplicates_and_unknowns(self) -> None:
        values = ["a", {"parent_asin": "a"}, "unknown", {"parent_asin": "b"}]
        self.assertEqual(
            normalize_identifiers(values, {"a", "b"}, 10),
            [{"parent_asin": "a"}, {"parent_asin": "b"}],
        )

    def test_malformed_contract_is_rejected(self) -> None:
        with self.assertRaises((TypeError, ValidationError)):
            normalize_response(
                {"message": 7, "ask_attribute": "secret", "recommendations": []},
                set(),
                10,
            )

    def test_dense_is_not_declared_in_default_runtime_imports(self) -> None:
        config = json.loads(Path("configs/techniques/baseline_v1.json").read_text())
        config["techniques"]["retrieval_route"] = "keyword"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(config, handle)
            handle.flush()
            parsed = RuntimeConfig.model_validate_json(Path(handle.name).read_text())
        self.assertEqual(parsed.techniques.retrieval_route, "keyword")

    def test_champion_weight_validation_rejects_invalid_configs(self) -> None:
        with self.assertRaises(ValidationError):
            TechniqueConfig(sparse_field_weights=(2, -1, 4, 2.5, 1.5, 1))
        with self.assertRaises(ValidationError):
            TechniqueConfig(reranker="learned_linear")
        with self.assertRaises(ValidationError):
            TechniqueConfig(question_policy="sequence")

    def test_guarded_assets_are_relative_typed_and_complete(self) -> None:
        with self.assertRaises(ValidationError):
            ModelAssetConfig(path="/tmp/model.json", sha256="0" * 64)
        with self.assertRaises(ValidationError):
            ModelAssetConfig(path="../model.json", sha256="0" * 64)
        with self.assertRaises(ValidationError):
            ModelAssetConfig(path="artifacts/model.json", sha256="invalid")
        with self.assertRaises(ValidationError):
            TechniqueConfig(
                state_mode="raw_history",
                question_policy="sequence",
                question_order=("other",),
                sparse_field_weights=(2, 8, 4, 2.5, 1.5, 1),
                reranker="guarded_constraint_gbdt",
            )


if __name__ == "__main__":
    unittest.main()
