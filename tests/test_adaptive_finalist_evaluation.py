from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from scripts import evaluate_adaptive_development_finalists as module

ROOT = Path(__file__).resolve().parents[1]


def test_finalists_use_full_shared_development_ground(monkeypatch) -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
        directory = Path(temporary)
        config = directory / "finalist.json"
        config.write_text("{}\n", encoding="utf-8")
        relative_config = config.relative_to(ROOT).as_posix()
        top_three = directory / "top3.json"
        top_three.write_text(
            json.dumps(
                {
                    "finalists": [
                        {
                            "rank": 1,
                            "candidate_id": "candidate-one",
                            "config_path": relative_config,
                            "config_sha256": hashlib.sha256(
                                config.read_bytes()
                            ).hexdigest(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        observed: list[str] = []

        def fake_run(command, **kwargs) -> None:
            del kwargs
            observed.extend(command)
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "evaluation_partition": "development",
                        "sample_count": 1650,
                        "evaluation_contract": {
                            "harness_id": "shared-v1",
                            "contract_sha256": "contract",
                        },
                        "sessions": [
                            {"sample_id": f"sample-{index:04d}"}
                            for index in range(1650)
                        ],
                    }
                ),
                encoding="utf-8",
            )

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        output = directory / "evaluations.json"
        report = module.evaluate_finalists(top_three, output)

        assert report["evaluation_count"] == 1
        assert report["sample_count"] == 1650
        assert observed.count("--dataset") == 3
        assert observed[observed.index("--partition") + 1] == "development"
        assert "--lineage-manifest" in observed
        assert output.is_file()
