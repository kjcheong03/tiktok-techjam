from __future__ import annotations

from dataclasses import dataclass, field

from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint


@dataclass(frozen=True)
class ConstraintView:
    """Immutable runtime-observable projection of one active V2 constraint."""

    attribute: str
    values: tuple[str, ...]
    relation: str
    polarity: str
    strength: str
    operator: str
    source_turn: int
    provenance: str

    @classmethod
    def from_constraint(cls, constraint: StructuredConstraint) -> ConstraintView:
        if not constraint.active:
            raise ValueError("a state view cannot expose an inactive constraint")
        return cls(
            attribute=constraint.attribute,
            values=tuple(constraint.values),
            relation=constraint.relation,
            polarity=constraint.polarity,
            strength=constraint.strength,
            operator=constraint.operator,
            source_turn=constraint.source_turn,
            provenance=constraint.provenance,
        )


@dataclass(frozen=True)
class V2StateView:
    """Small read-only boundary consumed by retrieval and ranking components."""

    query_text: str
    active_constraints: tuple[ConstraintView, ...]
    intent_epoch: int
    shown_ids: frozenset[str]
    asked_attributes: tuple[str, ...]
    no_preference_attributes: frozenset[str]
    turn: int

    def positive_constraints(self) -> dict[str, list[str]]:
        return self.constraints_by_polarity("include")

    def negative_constraints(self) -> dict[str, list[str]]:
        return self.constraints_by_polarity("exclude")

    def constraints_by_polarity(self, polarity: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for constraint in self.active_constraints:
            if constraint.polarity != polarity:
                continue
            result.setdefault(constraint.attribute, []).extend(constraint.values)
        return result


@dataclass
class V2SessionController:
    """Own the V2 state snapshot and correction-scoped recommendation history."""

    state: StateBaselineV2
    _shown_ids: set[str] = field(default_factory=set, repr=False)
    _history_epoch: int = 0

    def _sync_epoch(self) -> None:
        if self.state.intent_epoch != self._history_epoch:
            self._shown_ids.clear()
            self._history_epoch = self.state.intent_epoch

    def snapshot(self, *, query_text: str, turn: int) -> V2StateView:
        if turn < 0:
            raise ValueError("turn must be non-negative")
        self._sync_epoch()
        return V2StateView(
            query_text=query_text,
            active_constraints=tuple(
                ConstraintView.from_constraint(item)
                for item in self.state.active_constraints
            ),
            intent_epoch=self.state.intent_epoch,
            shown_ids=frozenset(self._shown_ids),
            asked_attributes=tuple(self.state.asked_attributes),
            no_preference_attributes=frozenset(self.state.no_preference_attributes),
            turn=turn,
        )

    def filter_ranking(self, ranking: list[str]) -> list[str]:
        self._sync_epoch()
        return [
            identifier for identifier in ranking if identifier not in self._shown_ids
        ]

    def record_shown(self, identifiers: list[str]) -> None:
        self._sync_epoch()
        self._shown_ids.update(identifiers)


__all__ = ["ConstraintView", "V2SessionController", "V2StateView"]
