from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESSENTIAL_MARKDOWN = {
    "README.md",
    "DATA_ATTRIBUTION.md",
    "data/README.md",
    "dashboard/README.md",
    "docs/architecture_overview.md",
    "docs/competition_specification.md",
    "docs/engine_guide.md",
    "docs/offline_training_optimization_pipeline.md",
    "docs/submission_rules.md",
}
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
IGNORED_MARKDOWN_ROOTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "artifacts/cache",
    "artifacts/proposals",
}


def _ignored(path: Path) -> bool:
    relative = str(path.relative_to(PROJECT_ROOT))
    return any(
        relative == root or relative.startswith(f"{root}/")
        for root in IGNORED_MARKDOWN_ROOTS
    )


def test_only_canonical_markdown_documents_remain() -> None:
    actual = {
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("*.md")
        if not _ignored(path)
    }
    assert actual == ESSENTIAL_MARKDOWN


def test_canonical_markdown_links_resolve() -> None:
    for relative in sorted(ESSENTIAL_MARKDOWN):
        document = PROJECT_ROOT / relative
        assert document.is_file() and document.stat().st_size > 0
        text = document.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", 1)[0].replace("%20", " ")
            if not path_text:
                continue
            resolved = (document.parent / path_text).resolve()
            assert resolved.exists(), f"{relative}: missing link target {target}"


def test_readme_links_every_supporting_document() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for relative in sorted(ESSENTIAL_MARKDOWN - {"README.md"}):
        assert f"]({relative})" in readme
