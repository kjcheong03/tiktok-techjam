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
