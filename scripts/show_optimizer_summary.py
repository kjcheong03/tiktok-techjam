from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def build_summary(campaign: dict[str, Any], finalists: dict[str, Any]) -> str:
    if campaign.get("mode") != "race":
        raise ValueError("expected a completed GhostLab race report")
    if finalists.get("campaign_report") != (
        "artifacts/reports/adaptive_hybrid_additive_warm_start_1650.json"
    ):
        raise ValueError("finalist report does not reference the completed campaign")

    stage_counts = campaign.get("stage_counts")
    fidelity_counts = campaign.get("fidelity_sample_counts")
    frozen = finalists.get("finalists")
    if not isinstance(stage_counts, dict) or not isinstance(fidelity_counts, dict):
        raise TypeError("campaign report is missing progressive-racing counts")
    if not isinstance(frozen, list) or not frozen or not isinstance(frozen[0], dict):
        raise TypeError("finalist report contains no frozen finalist")

    winner = frozen[0]
    metrics = winner.get("metrics")
    techniques = winner.get("techniques")
    if not isinstance(metrics, dict) or not isinstance(techniques, list):
        raise TypeError("frozen finalist is missing metrics or techniques")

    key_techniques = [
        label
        for technique_id, label in (
            ("fusion.rrf", "RRF evidence fusion"),
            ("prior.quality", "catalogue quality prior"),
            ("ranking.top10_residual_reranker.v2", "Top-10 residual reranker"),
            ("ranking.local_llm_semantic.v1", "bounded local-LLM semantic stage"),
        )
        if technique_id in techniques
    ]
    elapsed_seconds = float(campaign.get("elapsed_seconds", 0.0))
    elapsed_hours = elapsed_seconds / 3600.0
    sample_count = int(campaign.get("sample_count", 0))

    lines = [
        "GHOSTLAB OFFLINE OPTIMIZATION — COMPLETED CAMPAIGN",
        "",
        f"Development sessions: {sample_count:,} (final-selection set not accessed)",
        f"Search strategy: {str(campaign.get('search_mode', 'unknown')).replace('_', ' ')}",
        "",
        "Progressive racing:",
    ]
    for fidelity in ("f0", "f1", "f2"):
        lines.append(
            f"  {fidelity.upper()}: {int(stage_counts.get(fidelity, 0)):>2} candidates "
            f"× {int(fidelity_counts.get(fidelity, 0)):,} sessions"
        )
    lines.extend(
        [
            f"  Recorded campaign time: {elapsed_hours:.1f} hours",
            "",
            "Frozen development winner:",
            f"  Candidate: {winner['candidate_id']}",
            f"  Techniques: {', '.join(key_techniques)}",
            f"  Hit@10: {float(metrics['hit_rate_at_10']):.4f}",
            f"  MRR: {float(metrics['mrr']):.4f}",
            f"  MTTC: {float(metrics['mttc']):.4f}",
            f"  TechnicalScore: {float(metrics['score']):.4f}",
            f"  Constraint violations: {int(winner['constraint_violations'])}",
            f"  Promotion eligible: {'yes' if winner['promotion_eligible'] else 'no'}",
            f"  Frozen SHA-256: {winner['config_sha256']}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the completed GhostLab F0/F1/F2 campaign summary"
    )
    parser.add_argument(
        "--campaign",
        default="artifacts/reports/adaptive_hybrid_additive_warm_start_1650.json",
    )
    parser.add_argument(
        "--finalists", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    args = parser.parse_args()
    print(
        build_summary(
            _load_json(ROOT / args.campaign),
            _load_json(ROOT / args.finalists),
        )
    )


if __name__ == "__main__":
    main()
