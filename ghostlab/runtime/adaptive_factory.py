from __future__ import annotations

import json
from pathlib import Path

from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.runtime.adaptive_hybrid import AdaptiveHybridAgent

DEFAULT_ADAPTIVE_CONFIG = Path("configs/adaptive_hybrid_1a_3b_v1.json")


def load_adaptive_hybrid_config(path: str | Path) -> AdaptiveHybridConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AdaptiveHybridConfig.model_validate(payload)


def build_adaptive_hybrid_agent(
    catalog_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_ADAPTIVE_CONFIG,
    project_root: str | Path | None = None,
) -> AdaptiveHybridAgent:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    catalog = Path(catalog_path)
    if not catalog.is_absolute():
        catalog = root / catalog
    return AdaptiveHybridAgent(
        catalog,
        load_adaptive_hybrid_config(config_file),
        project_root=root,
    )


__all__ = [
    "DEFAULT_ADAPTIVE_CONFIG",
    "build_adaptive_hybrid_agent",
    "load_adaptive_hybrid_config",
]
