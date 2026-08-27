from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESSENTIAL_MIRRORS = {
    "README.md": "docs/essentials/project_overview.md",
    "docs/competition_specification.md": (
        "docs/essentials/competition_specification.md"
    ),
    "docs/submission_rules.md": "docs/essentials/submission_rules.md",
    "docs/unified_technique_operations.md": (
        "docs/essentials/unified_technique_operations.md"
    ),
    "docs/autonomous_unified_system_reference.md": (
        "docs/essentials/autonomous_unified_system_reference.md"
    ),
    "docs/champion_checkpoint.md": "docs/essentials/champion_checkpoint.md",
    "docs/final_candidate_checkpoint.md": (
        "docs/essentials/final_candidate_checkpoint.md"
    ),
    "docs/technique_decision_ledger.md": (
        "docs/essentials/technique_decision_ledger.md"
    ),
    "docs/wave2_advanced_challenger_and_autonomy_plan.md": (
        "docs/essentials/wave2_advanced_challenger_and_autonomy_plan.md"
    ),
    "docs/wave2_policy_track_validation.md": (
        "docs/essentials/wave2_policy_track_validation.md"
    ),
    "docs/wave2_retrieval_track_report.md": (
        "docs/essentials/wave2_retrieval_track_report.md"
    ),
    "docs/wave2_ranking_report.md": "docs/essentials/wave2_ranking_report.md",
    "docs/state_baseline_v2_integration.md": (
        "docs/essentials/state_baseline_v2_integration.md"
    ),
}


def test_essential_documents_match_their_sources() -> None:
    for source, mirror in ESSENTIAL_MIRRORS.items():
        source_path = PROJECT_ROOT / source
        mirror_path = PROJECT_ROOT / mirror
        assert source_path.is_file(), source
        assert mirror_path.is_file(), mirror
        assert mirror_path.read_bytes() == source_path.read_bytes(), (
            f"{mirror} drifted from {source}; update the source first and then "
            "synchronize its essential copy"
        )
