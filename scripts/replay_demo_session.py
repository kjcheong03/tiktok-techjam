"""Replay one public-development session with an auditable stage-by-stage trace.

The replay intentionally keeps the evaluator's labels on a separate evidence
branch.  The adaptive agent only receives the same profile and user messages it
would receive from :class:`ghostlab.research.replay.ReplayEnvironment`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluator.local_evaluator import catalog_index, normalize_recommendations
from ghostlab.competition.contract import AgentProtocol
from ghostlab.research.firewall import runtime_profile
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.runtime.adaptive_factory import (
    build_adaptive_hybrid_agent,
    load_adaptive_hybrid_config,
)
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import (
    AdaptiveLineageManifest,
    load_lineage_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/adaptive_hybrid_1a_3b_v1.json"
DEFAULT_CATALOG = "data/catalog.jsonl"
DEFAULT_DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)
DEFAULT_LINEAGE_MANIFEST = "data/splits/adaptive_hybrid_lineage_75_25_v1.json"
DEFAULT_OUTPUT_DIR = "artifacts/demo_replay"
JSON_FILENAME = "demo_replay.json"
MARKDOWN_FILENAME = "demo_replay.md"
DEFAULT_CONSOLE_WIDTH = 112
MIN_CONSOLE_WIDTH = 72
MAX_CONSOLE_WIDTH = 120


def _resolve_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_strings(value: object) -> list[str]:
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _safe_constraint(value: object) -> dict[str, object]:
    """Project runtime state to fields that cannot carry evaluator labels."""

    if isinstance(value, Mapping):
        source = value
        get = source.get
    else:
        get = lambda name, default=None: getattr(value, name, default)
    values = get("values", ())
    if isinstance(values, str):
        values = [values]
    elif isinstance(values, (set, frozenset)):
        values = sorted(values)
    if not isinstance(values, (list, tuple, set, frozenset)):
        values = []
    return {
        "attribute": str(get("attribute", "")),
        "values": [str(item) for item in values],
        "polarity": str(get("polarity", "include")),
        "strength": str(get("strength", "unspecified")),
        "source_turn": _as_int(get("source_turn", 0)),
    }


def _session_state(agent: object, session_id: str) -> object | None:
    sessions = getattr(agent, "sessions", None)
    if isinstance(sessions, Mapping):
        session = sessions.get(session_id)
        if session is not None:
            return _field(session, "state", session)
    getter = getattr(agent, "_session", None)
    if callable(getter):
        try:
            session = getter(session_id)
        except (KeyError, TypeError, ValueError):
            return None
        return _field(session, "state", session)
    state = getattr(agent, "state", None)
    return state if state is not None else None


def _state_evidence(
    agent: object, session_id: str, current_message: str
) -> dict[str, object]:
    state = _session_state(agent, session_id)
    if state is None:
        return {
            "messages": [current_message],
            "query": current_message,
            "intent_epoch": None,
            "active_constraints": [],
            "asked_attributes": [],
            "no_preference_attributes": [],
        }
    raw_messages = _field(state, "messages", ())
    messages = _as_strings(raw_messages)
    if not messages:
        messages = [current_message]
    raw_constraints = _field(state, "active_constraints", ())
    try:
        constraints = [_safe_constraint(item) for item in raw_constraints]
    except TypeError:
        constraints = []
    query = current_message
    query_builder = _field(state, "build_coverage_adaptive_query")
    if callable(query_builder):
        try:
            query = str(query_builder() or current_message)
        except Exception:  # noqa: BLE001 - trace rendering must not alter replay
            query = current_message
    return {
        "messages": messages,
        "query": query,
        "intent_epoch": _field(state, "intent_epoch"),
        "active_constraints": constraints,
        "asked_attributes": _as_strings(_field(state, "asked_attributes", ())),
        "no_preference_attributes": _as_strings(
            _field(state, "no_preference_attributes", ())
        ),
    }


def _latest_for_turn(
    agent: object, attribute: str, session_id: str, turn: int
) -> object | None:
    values = getattr(agent, attribute, ())
    if not isinstance(values, (list, tuple)):
        return None
    fallback: object | None = None
    for value in reversed(values):
        value_session = _field(value, "session_id")
        value_turn = _field(value, "turn")
        if value_turn is not None and _as_int(value_turn, -1) != turn:
            continue
        if value_session == session_id:
            return value
        if value_session is None:
            fallback = value
    return fallback


def _trace_data(
    agent: object, session_id: str, turn: int, response: Mapping[str, object]
) -> dict[str, object]:
    trace = _latest_for_turn(agent, "traces", session_id, turn)
    snapshot = _latest_for_turn(agent, "candidate_snapshots", session_id, turn)
    reason_codes = _as_strings(_field(trace, "reason_codes", ()))
    overloaded = _as_bool(_field(trace, "overloaded"), False)
    safe_merge = _as_bool(_field(trace, "safe_merge_executed"), False)
    safe_ranker = _as_bool(_field(trace, "safe_ranker_executed"), False)
    normal_union = _as_bool(_field(trace, "normal_union_executed"), False)
    semantic_decision = _as_bool(_field(trace, "semantic_decision_reached"), False)
    semantic_executed = _as_bool(_field(trace, "semantic_executed"), False)
    fallback_reason = _field(trace, "fallback_reason")
    if overloaded:
        path_mode = "bounded_overload"
        path_label = "bounded path (overload cutoff)"
    elif normal_union:
        path_mode = "full_union"
        path_label = "full path (union + optional semantic stages)"
    else:
        path_mode = "precision_fallback"
        path_label = "bounded fallback (complete precision path)"
    if semantic_executed:
        semantic_status = (
            "changed" if _as_bool(_field(trace, "semantic_changed")) else "unchanged"
        )
    elif semantic_decision:
        semantic_status = "skipped"
    else:
        semantic_status = "not_run"
    union_reason = next(
        (item for item in reason_codes if item.startswith(("union:", "rank:union"))),
        "union:executed" if normal_union else "union:not_reported",
    )
    active_sources = {
        key: _as_int(
            (_field(trace, "contribution_counts", {}) or {}).get(key, 0)
            if isinstance(_field(trace, "contribution_counts", {}), Mapping)
            else 0
        )
        for key in ("keyword", "category", "vector")
    }
    constraint_counts_raw = _field(trace, "constraint_counts", {})
    constraint_counts = (
        {str(key): _as_int(value) for key, value in constraint_counts_raw.items()}
        if isinstance(constraint_counts_raw, Mapping)
        else {}
    )
    removed_ids = _as_strings(_field(snapshot, "authority_removed_ids", ()))
    if not removed_ids:
        removed_ids = _as_strings(
            _field(
                trace,
                "constraint_removed_ids",
                _field(trace, "authority_removed_ids", ()),
            )
        )
    ask_attribute = response.get("ask_attribute")
    if ask_attribute is not None:
        ask_attribute = str(ask_attribute)
    return {
        "route": {
            "name": str(_field(trace, "route", "unavailable")),
            "confidence": _as_float(_field(trace, "route_confidence")),
            "reason": str(_field(trace, "route_reason", "not_reported")),
        },
        "preview": {
            "candidate_count": _as_int(_field(trace, "preview_candidate_count")),
            "score_flatness": _as_float(_field(trace, "preview_score_flatness")),
            "overloaded": overloaded,
            "reason": str(_field(trace, "preview_reason", "not_reported")),
        },
        "path": {
            "mode": path_mode,
            "label": path_label,
            "safe_merge_executed": safe_merge,
            "safe_ranker_executed": safe_ranker,
            "normal_union_executed": normal_union,
            "semantic_decision_reached": semantic_decision,
            "semantic_executed": semantic_executed,
            "fallback_reason": (
                str(fallback_reason) if fallback_reason is not None else None
            ),
        },
        "contributions": active_sources,
        "union": {
            "candidate_count": _as_int(_field(trace, "union_candidate_count")),
            "executed": normal_union,
            "reason": union_reason,
        },
        "semantic": {
            "status": semantic_status,
            "backend": str(_field(trace, "semantic_backend", "not_reported")),
            "activation_reason": str(
                _field(trace, "semantic_activation_reason", "not_reported")
            ),
            "changed": _as_bool(_field(trace, "semantic_changed")),
            "failure_reason": _field(trace, "semantic_failure_reason"),
            "elapsed_ms": _as_float(_field(trace, "semantic_elapsed_ms"), 0.0),
        },
        "constraints": {
            "counts": constraint_counts,
            "removed_ids": removed_ids,
            "output_violations": _as_int(_field(trace, "output_constraint_violations")),
        },
        "runtime": {
            "state_query": str(
                _field(trace, "state_query", _field(trace, "query", "")) or ""
            ),
            "intent_epoch": _field(trace, "intent_epoch"),
            "query_sha256": _field(trace, "query_sha256"),
            "query_views": _as_strings(_field(trace, "query_views", ())),
            "dense_requested_per_view": _as_int(
                _field(trace, "dense_requested_per_view")
            ),
            "dense_output_k": _as_int(_field(trace, "dense_output_k")),
            "dense_selection": str(_field(trace, "dense_selection", "not_reported")),
            "reason_codes": reason_codes,
        },
        "question": {
            "attribute": ask_attribute,
            "message": str(response.get("message", "")),
        },
    }


def _top_products(
    response: Mapping[str, object], catalog_ids: set[str], products: Mapping[str, dict]
) -> list[dict[str, object]]:
    identifiers = normalize_recommendations(
        response.get("recommendations"), catalog_ids
    )
    top: list[dict[str, object]] = []
    for rank, identifier in enumerate(identifiers, 1):
        product = products.get(identifier, {})
        title = product.get("title") if isinstance(product, Mapping) else None
        top.append(
            {
                "rank": rank,
                "title": str(title or "(untitled)"),
                "asin": identifier,
            }
        )
    return top


def _valid_response(agent: AgentProtocol, observation: object) -> dict[str, object]:
    """Call the agent using the evaluator's exception/shape containment policy."""

    try:
        response = agent.respond(
            observation.session_id,
            observation.user_message,
            observation.turn,
            observation.top_k,
        )
    except Exception:  # noqa: BLE001 - exact evaluator parity
        return {"message": "", "ask_attribute": None, "recommendations": []}
    if not isinstance(response, dict) or not isinstance(response.get("message"), str):
        return {"message": "", "ask_attribute": None, "recommendations": []}
    return response


def _format_constraint_list(values: object) -> str:
    if not isinstance(values, list) or not values:
        return "none"
    rendered: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            rendered.append(
                f"{item.get('attribute', '')}="
                f"{', '.join(str(v) for v in item.get('values', []))}"
            )
        else:
            rendered.append(str(item))
    return "; ".join(rendered)


def _state_changes(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> list[str]:
    """Describe only user-visible turn-to-turn state and routing changes."""

    if previous is None:
        return ["conversation state initialized from the first customer message"]

    changes: list[str] = []
    previous_state = previous.get("state", {})
    current_state = current.get("state", {})
    if not isinstance(previous_state, Mapping) or not isinstance(
        current_state, Mapping
    ):
        return ["state snapshot updated"]

    previous_epoch = previous_state.get("intent_epoch")
    current_epoch = current_state.get("intent_epoch")
    if previous_epoch != current_epoch:
        changes.append(f"intent epoch {previous_epoch} -> {current_epoch}")

    def constraint_keys(state: Mapping[str, object]) -> set[str]:
        constraints = state.get("active_constraints", [])
        if not isinstance(constraints, list):
            return set()
        return {
            _format_constraint_list([item])
            for item in constraints
            if isinstance(item, Mapping)
        }

    previous_constraints = constraint_keys(previous_state)
    current_constraints = constraint_keys(current_state)
    added = sorted(current_constraints - previous_constraints)
    removed = sorted(previous_constraints - current_constraints)
    if added:
        changes.append("constraints added: " + "; ".join(added))
    if removed:
        changes.append("constraints removed: " + "; ".join(removed))

    previous_asked = set(_as_strings(previous_state.get("asked_attributes", ())))
    current_asked = set(_as_strings(current_state.get("asked_attributes", ())))
    newly_asked = sorted(current_asked - previous_asked)
    if newly_asked:
        changes.append("newly asked: " + ", ".join(newly_asked))

    previous_declined = set(
        _as_strings(previous_state.get("no_preference_attributes", ()))
    )
    current_declined = set(
        _as_strings(current_state.get("no_preference_attributes", ()))
    )
    newly_declined = sorted(current_declined - previous_declined)
    if newly_declined:
        changes.append("no preference recorded: " + ", ".join(newly_declined))

    previous_route = previous.get("route", {})
    current_route = current.get("route", {})
    if isinstance(previous_route, Mapping) and isinstance(current_route, Mapping):
        before = previous_route.get("name")
        after = current_route.get("name")
        if before != after:
            changes.append(f"route {before} -> {after}")

    previous_path = previous.get("path", {})
    current_path = current.get("path", {})
    if isinstance(previous_path, Mapping) and isinstance(current_path, Mapping):
        before = previous_path.get("mode")
        after = current_path.get("mode")
        if before != after:
            changes.append(f"retrieval path {before} -> {after}")

    if previous_state.get("query") != current_state.get("query"):
        changes.append("retrieval query rebuilt")
    return changes or ["no material state or routing change"]


def _console_width(width: int | None = None) -> int:
    """Choose a stable, readable width for plain-terminal output."""

    if width is None:
        width = shutil.get_terminal_size(fallback=(DEFAULT_CONSOLE_WIDTH, 24)).columns
    return max(
        MIN_CONSOLE_WIDTH, min(MAX_CONSOLE_WIDTH, _as_int(width, DEFAULT_CONSOLE_WIDTH))
    )


def _wrapped(value: object, width: int, indent: str = "  ") -> list[str]:
    """Render one value with continuation lines aligned under its value."""

    text = str(value if value is not None else "none").replace("\n", " ").strip()
    if not text:
        text = "none"
    available = max(1, width - len(indent))
    wrapped = textwrap.wrap(
        text,
        width=available,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return [f"{indent}{line}" for line in wrapped] or [f"{indent}none"]


def _stage(lines: list[str], title: str) -> None:
    """Append a named stage with visual separation from the previous stage."""

    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"{title}:")


def _append_value(
    lines: list[str], title: str, value: object, width: int, *, indent: str = "  "
) -> None:
    lines.extend(_wrapped(value, width, indent=f"{indent}{title}: "))


def _append_labelled(
    lines: list[str], title: str, value: object, width: int, *, indent: str = "  "
) -> None:
    lines.append(f"{indent}{title}:")
    lines.extend(_wrapped(value, width, indent=f"{indent}  "))


def _readable_route_reason(value: object) -> str:
    """Keep the observable evidence while making its fields easy to scan."""

    reason = str(value if value is not None else "not reported")
    if reason.startswith("observable_evidence:"):
        fields = reason.split(":")
        return "observable evidence · " + " · ".join(fields[1:])
    return reason


def _console_header(sample_id: object, width: int | None = None) -> str:
    terminal_width = _console_width(width)
    lines = [f"Adaptive demo replay: sample {sample_id}"]
    lines.extend(
        _wrapped(
            "(Evaluator-only fields are shown to the replay operator and are "
            "never passed to the agent.)",
            terminal_width,
            "",
        )
    )
    return "\n".join(lines) + "\n"


def _console_result(result: object, width: int | None = None) -> str:
    terminal_width = _console_width(width)
    lines = ["Evaluator result:"]
    lines.extend(_wrapped(result, terminal_width, "  "))
    return "\n".join(lines) + "\n"


def _console_text(
    payload: Mapping[str, object],
    *,
    verbose: bool = False,
    width: int | None = None,
    include_header: bool = True,
) -> str:
    """Format an operator-facing trace without changing persisted evidence.

    The payload remains the source of truth for JSON/Markdown artifacts.  This
    renderer intentionally omits raw runtime diagnostics unless ``verbose`` is
    requested so the default output can be read during a live replay.
    """

    terminal_width = _console_width(width)
    lines: list[str] = []
    if include_header:
        lines.extend(_console_header(payload["sample_id"], terminal_width).splitlines())
    agent_visible = payload["agent_visible"]
    evaluator_only = payload["evaluator_only"]
    agent_turns = (
        agent_visible.get("turns", []) if isinstance(agent_visible, Mapping) else []
    )
    evaluator_turns = (
        evaluator_only.get("turns", []) if isinstance(evaluator_only, Mapping) else []
    )
    for visible, hidden in zip(agent_turns, evaluator_turns, strict=True):
        if not isinstance(visible, Mapping) or not isinstance(hidden, Mapping):
            continue
        state = visible.get("state", {})
        route = visible.get("route", {})
        preview = visible.get("preview", {})
        path = visible.get("path", {})
        union = visible.get("union", {})
        semantic = visible.get("semantic", {})
        constraints = visible.get("constraints", {})
        response = visible.get("response", {})
        runtime = visible.get("runtime", {})
        if lines:
            lines.append("")
        lines.extend(
            [
                f"=== Turn {visible.get('turn')} ===",
                "-" * min(terminal_width, 72),
            ]
        )

        _stage(lines, "Evaluator message")
        lines.extend(
            _wrapped(visible.get("evaluator_message", ""), terminal_width, "  ")
        )

        _stage(lines, "Dynamic conversation state")
        _append_labelled(lines, "Query", state.get("query", ""), terminal_width)
        _append_value(lines, "Intent epoch", state.get("intent_epoch"), terminal_width)
        _append_labelled(
            lines,
            "Active constraints",
            _format_constraint_list(state.get("active_constraints")),
            terminal_width,
        )

        _stage(lines, "Changes since previous turn")
        changes = _as_strings(visible.get("changes", ()))
        for change in changes or ["no material state or routing change"]:
            lines.extend(_wrapped(f"- {change}", terminal_width, "  "))

        _stage(lines, "Route")
        _append_value(lines, "Name", route.get("name"), terminal_width)
        _append_value(lines, "Confidence", route.get("confidence"), terminal_width)
        _append_labelled(
            lines,
            "Reason",
            _readable_route_reason(route.get("reason")),
            terminal_width,
        )

        _stage(lines, "Preview/overload")
        _append_value(
            lines, "Candidates", preview.get("candidate_count"), terminal_width
        )
        _append_value(lines, "Overloaded", preview.get("overloaded"), terminal_width)
        _append_value(
            lines, "Score flatness", preview.get("score_flatness"), terminal_width
        )
        _append_labelled(lines, "Reason", preview.get("reason"), terminal_width)

        _stage(lines, "Bounded-vs-full path")
        _append_labelled(lines, "Mode", path.get("label"), terminal_width)

        _stage(lines, "Evidence contribution counts")
        contributions = visible.get("contributions", {})
        for source in ("keyword", "category", "vector"):
            value = (
                contributions.get(source, 0)
                if isinstance(contributions, Mapping)
                else 0
            )
            _append_value(lines, source.title(), value, terminal_width)

        _stage(lines, "Union")
        _append_value(
            lines,
            "Status",
            "executed" if union.get("executed") else "skipped",
            terminal_width,
        )
        _append_value(lines, "Candidates", union.get("candidate_count"), terminal_width)
        _append_labelled(lines, "Reason", union.get("reason"), terminal_width)

        _stage(lines, "Semantic")
        _append_value(lines, "Status", semantic.get("status"), terminal_width)
        _append_value(lines, "Backend", semantic.get("backend"), terminal_width)
        _append_labelled(
            lines,
            "Activation reason",
            semantic.get("activation_reason"),
            terminal_width,
        )
        if semantic.get("failure_reason") not in (None, ""):
            _append_labelled(
                lines, "Failure", semantic.get("failure_reason"), terminal_width
            )

        _stage(lines, "Constraints")
        _append_labelled(lines, "Counts", constraints.get("counts", {}), terminal_width)
        _append_labelled(
            lines, "Removed", constraints.get("removed_ids", []), terminal_width
        )
        _append_value(
            lines,
            "Output violations",
            constraints.get("output_violations"),
            terminal_width,
        )

        _stage(lines, "Question")
        _append_value(lines, "Attribute", response.get("question"), terminal_width)
        _append_labelled(lines, "Message", response.get("message", ""), terminal_width)

        _stage(lines, "Top 10")
        top = response.get("top_10", [])
        if isinstance(top, list) and top:
            for item in top:
                if isinstance(item, Mapping):
                    lines.extend(
                        _wrapped(
                            f"{item.get('rank')}. {item.get('title')} [{item.get('asin')}]",
                            terminal_width,
                            "  ",
                        )
                    )
        else:
            lines.append("  (none)")

        if verbose:
            _stage(lines, "Runtime trace")
            _append_labelled(
                lines, "state_query", runtime.get("state_query"), terminal_width
            )
            _append_value(
                lines, "intent_epoch", runtime.get("intent_epoch"), terminal_width
            )
            _append_value(
                lines, "query_sha256", runtime.get("query_sha256"), terminal_width
            )
            _append_labelled(
                lines, "query_views", runtime.get("query_views", []), terminal_width
            )
            _append_value(
                lines,
                "dense_requested_per_view",
                runtime.get("dense_requested_per_view"),
                terminal_width,
            )
            _append_value(
                lines, "dense_output_k", runtime.get("dense_output_k"), terminal_width
            )
            _append_value(
                lines, "dense_selection", runtime.get("dense_selection"), terminal_width
            )
            _append_labelled(
                lines, "reason_codes", runtime.get("reason_codes", []), terminal_width
            )

            _stage(lines, "Path diagnostics")
            for key in (
                "safe_merge_executed",
                "safe_ranker_executed",
                "normal_union_executed",
                "semantic_decision_reached",
                "semantic_executed",
                "fallback_reason",
            ):
                _append_value(lines, key, path.get(key), terminal_width)
            _append_value(
                lines, "semantic_changed", semantic.get("changed"), terminal_width
            )
            _append_value(
                lines, "semantic_elapsed_ms", semantic.get("elapsed_ms"), terminal_width
            )

        _stage(lines, "Evaluator-only")
        lines.append(f"  Status: {'HIT' if hidden.get('hit') else 'MISS'}")
        lines.append(
            f"  Rank: {hidden.get('rank') if hidden.get('rank') is not None else '—'}"
        )
        _append_labelled(lines, "Next reply", hidden.get("next_reply"), terminal_width)
    result = (
        evaluator_only.get("session_result")
        if isinstance(evaluator_only, Mapping)
        else None
    )
    if result is not None:
        _stage(lines, "Evaluator result")
        lines.extend(_wrapped(result, terminal_width, "  "))
    return "\n".join(lines) + "\n"


def _markdown_text(payload: Mapping[str, object]) -> str:
    lines = [
        "# Adaptive demo replay",
        "",
        f"- Sample ID: `{payload['sample_id']}`",
        f"- Config: `{payload.get('config', {}).get('used', '')}`",
        (
            "- The `Agent-visible trace` section contains only messages and "
            "runtime-observable decisions."
        ),
        (
            "- The `Evaluator-only trace` section contains target, scenario, "
            "hit/rank, and next-reply evidence."
        ),
        "",
        "## Agent-visible trace",
    ]
    agent_visible = payload["agent_visible"]
    evaluator_only = payload["evaluator_only"]
    agent_turns = (
        agent_visible.get("turns", []) if isinstance(agent_visible, Mapping) else []
    )
    evaluator_turns = (
        evaluator_only.get("turns", []) if isinstance(evaluator_only, Mapping) else []
    )
    for visible in agent_turns:
        if not isinstance(visible, Mapping):
            continue
        state = visible.get("state", {})
        route = visible.get("route", {})
        preview = visible.get("preview", {})
        path = visible.get("path", {})
        union = visible.get("union", {})
        semantic = visible.get("semantic", {})
        constraints = visible.get("constraints", {})
        response = visible.get("response", {})
        runtime = visible.get("runtime", {})
        lines.extend(
            [
                "",
                f"### Turn {visible.get('turn')}",
                f"**Evaluator message:** {visible.get('evaluator_message', '')}",
                "",
                (
                    "**Dynamic conversation state:** "
                    f"query=`{state.get('query', '')}`; "
                    f"intent epoch=`{state.get('intent_epoch')}`; "
                    "active constraints: "
                    f"{_format_constraint_list(state.get('active_constraints'))}"
                ),
                "",
                "**Changes since previous turn:** "
                + "; ".join(_as_strings(visible.get("changes", ()))),
                "",
                (
                    f"**Route:** `{route.get('name')}`; "
                    f"confidence=`{route.get('confidence')}`; "
                    f"reason=`{route.get('reason')}`"
                ),
                (
                    f"**Preview/overload:** "
                    f"candidates=`{preview.get('candidate_count')}`; "
                    f"overloaded=`{preview.get('overloaded')}`; "
                    f"flatness=`{preview.get('score_flatness')}`; "
                    f"reason=`{preview.get('reason')}`"
                ),
                f"**Bounded-vs-full path:** {path.get('label')}",
                (
                    f"**Evidence contribution counts:** "
                    f"keyword=`{visible.get('contributions', {}).get('keyword', 0)}`; "
                    f"category=`{visible.get('contributions', {}).get('category', 0)}`; "
                    f"vector=`{visible.get('contributions', {}).get('vector', 0)}`"
                ),
                (
                    f"**Union:** "
                    f"status=`{'executed' if union.get('executed') else 'skipped'}`; "
                    f"candidates=`{union.get('candidate_count')}`; "
                    f"reason=`{union.get('reason')}`"
                ),
                (
                    f"**Semantic:** status=`{semantic.get('status')}`; "
                    f"backend=`{semantic.get('backend')}`; "
                    f"reason=`{semantic.get('activation_reason')}`; "
                    f"failure=`{semantic.get('failure_reason')}`"
                ),
                (
                    f"**Constraints:** "
                    f"counts=`{constraints.get('counts', {})}`; "
                    f"removed=`{constraints.get('removed_ids', [])}`; "
                    f"violations=`{constraints.get('output_violations')}`"
                ),
                (
                    f"**Question:** attribute=`{response.get('question')}`; "
                    f"message={response.get('message', '')}"
                ),
                (
                    f"**Runtime trace:** state query=`{runtime.get('state_query')}`; "
                    f"intent epoch=`{runtime.get('intent_epoch')}`; "
                    f"query views=`{runtime.get('query_views', [])}`; "
                    f"reason codes=`{runtime.get('reason_codes', [])}`"
                ),
                "",
                "**Top 10:**",
                "",
                "| Rank | Catalog title | ASIN |",
                "| ---: | --- | --- |",
            ]
        )
        top = response.get("top_10", [])
        if isinstance(top, list) and top:
            lines.extend(
                f"| {item.get('rank')} | {item.get('title', '')} | "
                f"`{item.get('asin', '')}` |"
                for item in top
                if isinstance(item, Mapping)
            )
        else:
            lines.append("| — | (none) | — |")
    lines.extend(["", "## Evaluator-only trace", ""])
    lines.extend(
        [
            "| Turn | Scenario | Target ASIN | Hit | Rank | Next reply |",
            "| ---: | --- | --- | --- | ---: | --- |",
        ]
    )
    for hidden in evaluator_turns:
        if not isinstance(hidden, Mapping):
            continue
        next_reply = (
            str(hidden.get("next_reply") or "")
            .replace("|", "\\|")
            .replace("\n", "<br>")
        )
        lines.append(
            f"| {hidden.get('turn')} | `{hidden.get('scenario_type')}` | "
            f"`{hidden.get('target_asin')}` | "
            f"{'HIT' if hidden.get('hit') else 'MISS'} | "
            f"{hidden.get('rank') or '—'} | {next_reply or '—'} |"
        )
    result = (
        evaluator_only.get("session_result")
        if isinstance(evaluator_only, Mapping)
        else None
    )
    if result is not None:
        lines.extend(
            [
                "",
                "**Session result:**",
                "",
                f"```json\n{json.dumps(result, sort_keys=True)}\n```",
            ]
        )
    return "\n".join(lines) + "\n"


def write_demo_artifacts(
    payload: Mapping[str, object], output_dir: str | Path
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_FILENAME
    markdown_path = directory / MARKDOWN_FILENAME
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown_text(payload), encoding="utf-8")
    return json_path, markdown_path


def replay_one_sample(
    sample: Mapping[str, object],
    *,
    agent: AgentProtocol,
    categories: dict[str, list[str]],
    products: dict[str, dict],
    catalog_ids: set[str] | None = None,
    output_dir: str | Path | None = None,
    metadata: Mapping[str, object] | None = None,
    print_fn: Callable[[str], object] = print,
    verbose: bool = False,
) -> dict[str, object]:
    """Run exactly one sample through the canonical evaluator transition."""

    if catalog_ids is None:
        catalog_ids = set(products)
    environment = ReplayEnvironment(dict(sample), categories, products)
    observation = environment.observe()
    agent.reset(observation.session_id, runtime_profile(sample))
    visible_turns: list[dict[str, object]] = []
    evaluator_turns: list[dict[str, object]] = []
    print_fn(_console_header(sample["sample_id"]))
    while not environment.done:
        response = _valid_response(agent, observation)
        state = _state_evidence(agent, observation.session_id, observation.user_message)
        runtime = _trace_data(agent, observation.session_id, observation.turn, response)
        top = _top_products(response, catalog_ids, products)
        visible_turn: dict[str, object] = {
            "turn": observation.turn,
            "evaluator_message": observation.user_message,
            "state": state,
            "route": runtime["route"],
            "preview": runtime["preview"],
            "path": runtime["path"],
            "contributions": runtime["contributions"],
            "union": runtime["union"],
            "semantic": runtime["semantic"],
            "constraints": runtime["constraints"],
            "runtime": runtime["runtime"],
            "question": runtime["question"],
            "response": {
                "message": str(response.get("message", "")),
                "question": response.get("ask_attribute"),
                "top_10": top,
            },
        }
        visible_turn["changes"] = _state_changes(
            visible_turns[-1] if visible_turns else None,
            visible_turn,
        )
        visible_turns.append(visible_turn)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        override_active = environment.override_applied
        hit = override_active and environment.target in ranked
        rank = ranked.index(environment.target) + 1 if hit else None
        next_observation = environment.step(response)
        evaluator_turn: dict[str, object] = {
            "turn": observation.turn,
            "scenario_type": str(sample["scenario_type"]),
            "target_asin": environment.target,
            "override_active": override_active,
            "hit": hit,
            "rank": rank,
            "next_reply": (
                next_observation.user_message if next_observation is not None else None
            ),
            "done": environment.done,
        }
        evaluator_turns.append(evaluator_turn)
        turn_payload = {
            "sample_id": str(sample["sample_id"]),
            "agent_visible": {"turns": [visible_turn]},
            "evaluator_only": {"turns": [evaluator_turn]},
        }
        print_fn(
            _console_text(
                turn_payload,
                verbose=verbose,
                include_header=False,
            )
        )
        if next_observation is not None:
            observation = next_observation
    payload: dict[str, object] = {
        "schema_version": 1,
        "sample_id": str(sample["sample_id"]),
        **(dict(metadata) if metadata is not None else {}),
        "agent_visible": {"turns": visible_turns},
        "evaluator_only": {
            "turns": evaluator_turns,
            "session_result": environment.session_result(),
        },
    }
    if output_dir is not None:
        write_demo_artifacts(payload, output_dir)
    print_fn("\n" + _console_result(environment.session_result()))
    return payload


def _development_sample(
    root: Path,
    sample_id: str,
    dataset_paths: tuple[str, ...],
    manifest_path: str,
) -> tuple[dict[str, object], object, AdaptiveLineageManifest]:
    corpus = load_adaptive_training_corpus(root, dataset_paths)
    manifest = load_lineage_manifest(_resolve_path(root, manifest_path), corpus)
    requested = str(sample_id).strip()
    if not requested:
        raise ValueError("sample-id must be non-empty")
    if requested in manifest.holdout_ids:
        raise ValueError(f"sample-id {requested!r} belongs to the holdout partition")
    if requested not in manifest.development_ids:
        raise ValueError(f"sample-id {requested!r} is not a known development ID")
    try:
        sample = corpus.samples[requested]
    except KeyError as error:
        raise ValueError(
            f"sample-id {requested!r} is absent from the loaded corpus"
        ) from error
    return sample, corpus, manifest


def _config_path(root: Path, requested: str) -> Path:
    path = _resolve_path(root, requested)
    if path.is_file():
        return path
    raise FileNotFoundError(f"missing adaptive config: {path}")


def run_demo_session(
    *,
    sample_id: str,
    config: str = DEFAULT_CONFIG,
    catalog: str = DEFAULT_CATALOG,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    lineage_manifest: str = DEFAULT_LINEAGE_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    project_root: str | Path = ROOT,
    agent: AgentProtocol | None = None,
    print_fn: Callable[[str], object] = print,
    verbose: bool = False,
) -> dict[str, object]:
    """Load one development sample and produce the deterministic demo files."""

    root = Path(project_root).resolve()
    sample, corpus, manifest = _development_sample(
        root, sample_id, tuple(datasets), lineage_manifest
    )
    catalog_path = _resolve_path(root, catalog)
    catalog_ids, categories, products = catalog_index(catalog_path)
    if agent is None:
        config_path = _config_path(root, config)
        loaded_config = load_adaptive_hybrid_config(config_path)
        agent = build_adaptive_hybrid_agent(
            catalog_path,
            config_path=config_path,
            project_root=root,
        )
    else:
        config_path = _resolve_path(root, config)
        loaded_config = None
    agent_config = getattr(agent, "config", None)
    policy_id = (
        loaded_config.policy_id
        if loaded_config is not None
        else _field(agent_config, "policy_id")
    )
    canonical_hash = (
        loaded_config.canonical_hash() if loaded_config is not None else None
    )
    metadata = {
        "config": {
            "requested": str(_resolve_path(root, config)),
            "used": str(config_path),
            "policy_id": policy_id,
            "canonical_sha256": canonical_hash,
        },
        "catalog": {"path": str(catalog_path), "sha256": _sha256(catalog_path)},
        "dataset_sources": [source.__dict__ for source in corpus.sources],
        "lineage_manifest": {
            "path": str(manifest.path),
            "sha256": manifest.manifest_sha256,
            "partition": "development",
        },
    }
    return replay_one_sample(
        sample,
        agent=agent,
        categories=categories,
        products=products,
        catalog_ids=catalog_ids,
        output_dir=_resolve_path(root, output_dir),
        metadata=metadata,
        print_fn=print_fn,
        verbose=verbose,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay exactly one adaptive session from the development partition"
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument(
        "--dataset",
        "--datasets",
        action="append",
        dest="datasets",
        help="adaptive dataset path; repeat for the three project datasets",
    )
    parser.add_argument("--lineage-manifest", default=DEFAULT_LINEAGE_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include full runtime state and low-level diagnostics in console output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_demo_session(
            sample_id=args.sample_id,
            config=args.config,
            catalog=args.catalog,
            datasets=tuple(args.datasets or DEFAULT_DATASETS),
            lineage_manifest=args.lineage_manifest,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CATALOG",
    "DEFAULT_CONFIG",
    "DEFAULT_DATASETS",
    "DEFAULT_LINEAGE_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "JSON_FILENAME",
    "MARKDOWN_FILENAME",
    "build_parser",
    "main",
    "replay_one_sample",
    "run_demo_session",
    "write_demo_artifacts",
]
