from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Factory = Callable[[], object]


@dataclass(frozen=True)
class Technique:
    name: str
    factory: Factory
    requires: tuple[str, ...] = ()
    incompatible_with: tuple[str, ...] = ()


class TechniqueRegistry:
    """Explicit lazy registry; factories run only when requested."""

    def __init__(self) -> None:
        self._techniques: dict[str, Technique] = {}

    def register(self, technique: Technique) -> None:
        if technique.name in self._techniques:
            raise ValueError(f"duplicate technique: {technique.name}")
        self._techniques[technique.name] = technique

    def build(self, names: set[str]) -> dict[str, object]:
        unknown = names - self._techniques.keys()
        if unknown:
            raise ValueError(f"unknown techniques: {sorted(unknown)}")
        for name in names:
            technique = self._techniques[name]
            missing = set(technique.requires) - names
            conflicts = set(technique.incompatible_with) & names
            if missing:
                raise ValueError(f"{name} requires {sorted(missing)}")
            if conflicts:
                raise ValueError(f"{name} conflicts with {sorted(conflicts)}")
        return {name: self._techniques[name].factory() for name in sorted(names)}
