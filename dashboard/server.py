from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = {
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
}
MODEL_LABELS = {
    "A": "Organizer BM25 Starter",
    "C": "Fixed Adaptive Architecture",
    "D": "GhostLab Champion",
}
ACTIVE_CANDIDATE = PROJECT_ROOT / "configs" / "active_candidate.json"
COMPARISON_REPORTS = (
    PROJECT_ROOT / "artifacts" / "reports" / "adaptive_final_holdout.json",
    PROJECT_ROOT / "artifacts" / "reports" / "adaptive_system_comparison_1650.json",
)


def _has_metrics(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    metrics = value.get("metrics")
    if isinstance(metrics, dict) and METRIC_KEYS.intersection(metrics):
        return True
    return bool(METRIC_KEYS.intersection(value))


def count_visualizable_runs(payload: object) -> int:
    """Count dashboard-compatible runs without depending on a single report schema."""
    if _has_metrics(payload):
        return 1
    if not isinstance(payload, dict):
        return 0
    systems = payload.get("systems")
    if isinstance(systems, list):
        count = sum(_has_metrics(system) for system in systems)
        if count:
            return count
    records = payload.get("records")
    if isinstance(records, list):
        count = sum(_has_metrics(record) for record in records)
        if count:
            return count
    return sum(_has_metrics(value) for value in payload.values())


def _model_descriptor(
    model_id: str,
    path: Path,
    *,
    run_key: str | None = None,
    system_id: str | None = None,
) -> dict[str, object]:
    relative = path.relative_to(PROJECT_ROOT)
    return {
        "model_id": model_id,
        "label": MODEL_LABELS[model_id],
        "path": relative.as_posix(),
        "url": "/" + quote(relative.as_posix()),
        "run_key": run_key,
        "system_id": system_id,
        "role": (
            "explanatory_baseline"
            if model_id == "A"
            else "ghostlab_control"
            if model_id == "C"
            else "ghostlab_champion"
        ),
        "champion_eligible": model_id in {"C", "D"},
        "default": True,
    }


def _select_comparison_systems(
    payload: object,
    *,
    active_preset_path: str | None = None,
    preferred_system_id: str | None = None,
) -> dict[str, dict[str, object]]:
    """Resolve the baseline, adaptive control, and active champion rows."""
    if not isinstance(payload, dict) or not isinstance(payload.get("systems"), list):
        return {}
    systems = [item for item in payload["systems"] if isinstance(item, dict)]
    selected: dict[str, dict[str, object]] = {}
    for model_id in ("A", "C"):
        match = next(
            (
                item
                for item in systems
                if str(item.get("system_id", "")).startswith(f"{model_id}_")
            ),
            None,
        )
        if match is not None:
            selected[model_id] = match

    selected_system_id = preferred_system_id or str(
        payload.get("selected_system_id") or ""
    )
    challenger_rows = [
        item
        for item in systems
        if str(item.get("system_id", "")).startswith("D")
        or "challenger" in str(item.get("role", "")).lower()
        or "champion" in str(item.get("role", "")).lower()
    ]
    selected_challenger = next(
        (
            item
            for item in challenger_rows
            if active_preset_path
            and str(item.get("config_path", "")) == active_preset_path
        ),
        None,
    )
    if selected_challenger is None:
        selected_challenger = next(
            (
                item
                for item in systems
                if selected_system_id
                and not selected_system_id.startswith(("A_", "B_", "C_"))
                and str(item.get("system_id", "")) == selected_system_id
            ),
            None,
        )
    if selected_challenger is None and challenger_rows:
        # D1 is the frozen top-ranked challenger when no final selection exists yet.
        selected_challenger = min(
            challenger_rows, key=lambda item: str(item.get("system_id", ""))
        )
    if selected_challenger is not None:
        selected["D"] = selected_challenger
    return selected


def _repository_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = (PROJECT_ROOT / value).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _active_champion_context() -> tuple[str | None, str | None, list[Path]]:
    """Read the active preset and its canonical comparison evidence."""
    try:
        pointer = json.loads(ACTIVE_CANDIDATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, []
    if not isinstance(pointer, dict):
        return None, None, []

    preset_path = pointer.get("preset_path")
    active_preset_path = preset_path if isinstance(preset_path, str) else None
    system_id = pointer.get("champion_system_id")
    preferred_system_id = system_id if isinstance(system_id, str) else None
    report_paths: list[Path] = []

    comparison_path = _repository_path(pointer.get("comparison_report_path"))
    if comparison_path is not None:
        report_paths.append(comparison_path)

    adjudication_path = _repository_path(pointer.get("adjudication_path"))
    if adjudication_path is not None:
        try:
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            adjudication = None
        if isinstance(adjudication, dict):
            evidence = adjudication.get("evidence")
            if isinstance(evidence, dict):
                evidence_path = _repository_path(
                    evidence.get("final_holdout_report_path")
                )
                if evidence_path is not None:
                    report_paths.append(evidence_path)

    return active_preset_path, preferred_system_id, list(dict.fromkeys(report_paths))


def discover_models() -> list[dict[str, object]]:
    """Return the three stable dashboard systems, never raw experiment reports."""
    baseline = PROJECT_ROOT / "artifacts" / "baseline_results.json"
    models = {
        "A": _model_descriptor("A", baseline, run_key="official_keyword"),
        "C": _model_descriptor(
            "C",
            PROJECT_ROOT
            / "artifacts"
            / "reports"
            / "adaptive_hybrid_1a_3b_final_v2.json",
        ),
        "D": _model_descriptor(
            "D",
            PROJECT_ROOT
            / "artifacts"
            / "reports"
            / "unified_champion_verification_v1.json",
        ),
    }

    active_preset_path, preferred_system_id, active_reports = (
        _active_champion_context()
    )
    comparison_candidates = list(dict.fromkeys([*active_reports, *COMPARISON_REPORTS]))
    comparison_path: Path | None = None
    selected_systems: dict[str, dict[str, object]] = {}
    for path in comparison_candidates:
        if not path.is_file():
            continue
        try:
            comparison = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected = _select_comparison_systems(
            comparison,
            active_preset_path=active_preset_path,
            preferred_system_id=preferred_system_id,
        )
        if set(selected) == {"A", "C", "D"}:
            comparison_path = path
            selected_systems = selected
            break

    if comparison_path is not None:
        for model_id, system in selected_systems.items():
            system_id = system.get("system_id")
            if isinstance(system_id, str) and system_id:
                models[model_id] = _model_descriptor(
                    model_id, comparison_path, system_id=system_id
                )

    return [models[model_id] for model_id in ("A", "C", "D")]


def discover_reports() -> list[dict[str, object]]:
    candidates = [PROJECT_ROOT / "artifacts" / "baseline_results.json"]
    candidates.extend(sorted((PROJECT_ROOT / "artifacts" / "reports").glob("*.json")))
    default_result = PROJECT_ROOT / "results.json"
    if default_result.is_file():
        candidates.append(default_result)

    reports: list[dict[str, object]] = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_count = count_visualizable_runs(payload)
        if not run_count:
            continue
        relative = path.relative_to(PROJECT_ROOT)
        stat = path.stat()
        systems = payload.get("systems") if isinstance(payload, dict) else None
        fair_comparison = (
            isinstance(systems, list)
            and isinstance(payload.get("comparison_semantics"), dict)
            and payload["comparison_semantics"].get("same_ground") is True
        )
        reports.append(
            {
                "label": path.stem.replace("_", " ").title(),
                "path": relative.as_posix(),
                "url": "/" + quote(relative.as_posix()),
                "run_count": run_count,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "kind": "fair_system_comparison" if fair_comparison else "generic",
                "partition": (
                    payload.get("evaluation_partition")
                    if isinstance(payload, dict)
                    else None
                ),
                "sample_count": (
                    payload.get("sample_count") if isinstance(payload, dict) else None
                ),
                "featured": False,
            }
        )
    priorities = (
        "artifacts/reports/adaptive_final_holdout.json",
        "artifacts/reports/adaptive_system_comparison_1650.json",
        "artifacts/reports/unified_champion_verification_v1.json",
        "artifacts/reports/adaptive_hybrid_qwen_selective_v3.json",
    )
    for preferred in priorities:
        match = next((item for item in reports if item["path"] == preferred), None)
        if match is not None:
            match["featured"] = True
            break
    return sorted(
        reports, key=lambda item: (not bool(item["featured"]), str(item["label"]))
    )


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "GhostLabDashboard/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def _send_json(self, payload: object, status: int = 200) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/api/health":
            self._send_json({"ok": True, "service": "ghostlab-results-dashboard"})
            return
        if request_path == "/api/reports":
            reports = discover_reports()
            self._send_json({"reports": reports, "count": len(reports)})
            return
        if request_path == "/api/models":
            models = discover_models()
            self._send_json({"models": models, "count": len(models)})
            return
        if request_path == "/":
            self.send_response(302)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return

        allowed_dashboard = request_path == "/dashboard" or request_path.startswith(
            "/dashboard/"
        )
        allowed_report = (
            request_path.startswith("/artifacts/") or request_path == "/results.json"
        ) and request_path.endswith(".json")
        if not (allowed_dashboard or allowed_report):
            self.send_error(
                404, "Only dashboard assets and result JSON files are served"
            )
            return
        super().do_GET()

    def log_message(self, message: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {message % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the GhostLab results dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"GhostLab dashboard: http://{args.host}:{args.port}/dashboard/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
