from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TypedDict

from ghostlab.campaign.top_three import ProposalCandidate, TopThreeSelection

ROLE_FILENAMES = {
    "score_leader": "score_leader.json",
    "robust_leader": "robust_leader.json",
    "efficient_alternative": "efficient_alternative.json",
}


class PathRecord(TypedDict):
    path: str
    sha256: str
    bytes: int
    files: int


class CandidateRecord(TypedDict):
    role: str
    candidate_id: str
    reason: str
    score: float
    mean_delta: float
    confidence_interval: list[float]
    randomization_pvalue: float
    wins_ties_losses: list[int]
    scenario_deltas: dict[str, float]
    complexity: int
    latency_p95_ms: float
    memory_mb: float
    dependency_extras: list[str]
    assets: list[PathRecord]
    evidence: list[PathRecord]
    preset: PathRecord
    notes: list[str]
    confirmed: bool
    safe: bool
    enabled_techniques: list[str]
    technique_sources: list[dict[str, str]]
    tuned_parameters: dict[str, str | int | float | bool]
    configuration: dict[str, object]
    prepare_command: str


@dataclass(frozen=True)
class MaterializedProposalBundle:
    output_dir: Path
    manifest_path: Path
    guide_path: Path
    preset_paths: tuple[Path, ...]


def materialize_top_three(
    selection: TopThreeSelection,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    baseline_config_path: str,
    split_path: str = "configs/splits/adaptive_v1.json",
    rollback_commit: str,
    maximum_asset_files: int = 10000,
    maximum_asset_bytes: int = 10 * 1024**3,
) -> MaterializedProposalBundle:
    """Write a proposal bundle only; never mutate suite defaults or promote code."""
    root = Path(project_root).resolve()
    target = Path(output_dir).resolve()
    _require_below_root(target, root)
    baseline = _resolve_existing(root, baseline_config_path)
    split = _resolve_existing(root, split_path)
    if not re.fullmatch(r"[0-9a-f]{7,40}", rollback_commit):
        raise ValueError("rollback_commit must be a Git object ID")
    if not 1 <= maximum_asset_files <= 100000:
        raise ValueError("maximum_asset_files must be between 1 and 100000")
    if not 1 <= maximum_asset_bytes <= 100 * 1024**3:
        raise ValueError("maximum_asset_bytes must be between 1 and 100 GiB")
    candidates = selection.candidates
    if (
        len(candidates) != 3
        or len({item.evaluation.candidate_id for item in candidates}) != 3
    ):
        raise ValueError("materialization requires three distinct candidates")
    target.mkdir(parents=True, exist_ok=True)
    reports_dir = target / "reports"
    reports_dir.mkdir(exist_ok=True)

    preset_paths: list[Path] = []
    records: list[CandidateRecord] = []
    for candidate in candidates:
        preset_path = target / ROLE_FILENAMES[candidate.role]
        payload = candidate.package.config.model_dump(mode="json")
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        _write_if_changed(preset_path, encoded)
        preset_paths.append(preset_path)
        records.append(
            _candidate_record(
                candidate,
                root=root,
                preset_path=preset_path,
                maximum_asset_files=maximum_asset_files,
                maximum_asset_bytes=maximum_asset_bytes,
            )
        )

    manifest = {
        "schema_version": 1,
        "kind": "human_review_top_three",
        "baseline_id": selection.baseline_id,
        "baseline_config": _path_record(
            baseline, root, maximum_asset_files, maximum_asset_bytes
        ),
        "adaptive_split": _path_record(
            split, root, maximum_asset_files, maximum_asset_bytes
        ),
        "rollback": {
            "commit": rollback_commit,
            "preset": baseline_config_path,
            "preset_sha256": _hash_path(
                baseline, maximum_asset_files, maximum_asset_bytes
            )[0],
        },
        "candidates": records,
        "excluded": [list(item) for item in selection.excluded],
        "automatic_promotion": False,
        "f3_access": "forbidden",
        "human_boundaries": ["gate_a", "one_shot_f3", "gate_b"],
    }
    manifest_path = target / "proposal_manifest.json"
    _write_if_changed(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    guide_path = target / "README.md"
    _write_if_changed(
        guide_path,
        _render_guide(
            selection,
            root=root,
            target=target,
            baseline_config_path=baseline_config_path,
            split_path=split_path,
            rollback_commit=rollback_commit,
            records=records,
            maximum_asset_files=maximum_asset_files,
            maximum_asset_bytes=maximum_asset_bytes,
        ),
    )
    return MaterializedProposalBundle(
        target, manifest_path, guide_path, tuple(preset_paths)
    )


def _candidate_record(
    candidate: ProposalCandidate,
    *,
    root: Path,
    preset_path: Path,
    maximum_asset_files: int,
    maximum_asset_bytes: int,
) -> CandidateRecord:
    package = candidate.package
    preset_relative = preset_path.relative_to(root).as_posix()
    return {
        "role": candidate.role,
        "candidate_id": candidate.evaluation.candidate_id,
        "reason": candidate.reason,
        "score": candidate.evaluation.score,
        "mean_delta": candidate.analysis.mean_delta,
        "confidence_interval": list(candidate.analysis.confidence_interval),
        "randomization_pvalue": candidate.analysis.randomization_pvalue,
        "wins_ties_losses": [
            candidate.analysis.wins,
            candidate.analysis.ties,
            candidate.analysis.losses,
        ],
        "scenario_deltas": dict(sorted(candidate.analysis.scenario_deltas.items())),
        "complexity": candidate.evaluation.complexity,
        "latency_p95_ms": candidate.evaluation.latency_p95_ms,
        "memory_mb": candidate.evaluation.memory_mb,
        "dependency_extras": list(package.dependency_extras),
        "assets": [
            _path_record(
                _resolve_existing(root, value),
                root,
                maximum_asset_files,
                maximum_asset_bytes,
            )
            for value in package.assets
        ],
        "evidence": [
            _path_record(
                _resolve_existing(root, value),
                root,
                maximum_asset_files,
                maximum_asset_bytes,
            )
            for value in package.evidence_refs
        ],
        "preset": _path_record(
            preset_path, root, maximum_asset_files, maximum_asset_bytes
        ),
        "notes": list(package.notes),
        "confirmed": package.confirmed,
        "safe": package.safe,
        "enabled_techniques": list(package.enabled_techniques),
        "technique_sources": [
            {"technique_id": item, "source": source, "description": description}
            for item, source, description in package.technique_sources
        ],
        "tuned_parameters": dict(package.tuned_parameters),
        "configuration": package.config.model_dump(mode="json"),
        "prepare_command": (
            "uv run python -m scripts.prepare_candidate --preset "
            + shlex.quote(preset_relative)
        ),
    }


def _render_guide(
    selection: TopThreeSelection,
    *,
    root: Path,
    target: Path,
    baseline_config_path: str,
    split_path: str,
    rollback_commit: str,
    records: list[CandidateRecord],
    maximum_asset_files: int,
    maximum_asset_bytes: int,
) -> str:
    target_relative = target.relative_to(root).as_posix()
    lines = [
        "# Top-three Wave 2 candidate proposal",
        "",
        "> Proposal only. Nothing in this folder promotes a candidate, changes the",
        "> champion preset, opens F3, merges a branch, commits, or pushes.",
        "",
        "## Shortlist",
        "",
        "| Role | Candidate | Score | Mean paired delta | 95% interval | p95 ms | MB | Complexity |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    by_role = {item.role: item for item in selection.candidates}
    for role in ("score_leader", "robust_leader", "efficient_alternative"):
        item = by_role[role]
        interval = item.analysis.confidence_interval
        lines.append(
            f"| `{role}` | `{item.evaluation.candidate_id}` | "
            f"{item.evaluation.score:.6f} | {item.analysis.mean_delta:+.6f} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{item.evaluation.latency_p95_ms:.3f} | "
            f"{item.evaluation.memory_mb:.3f} | {item.evaluation.complexity} |"
        )
    lines.extend(
        [
            "",
            "The three roles are deliberately distinct. Score chooses the highest",
            "validated endpoint, robustness emphasizes lower-bound and scenario",
            "behavior, and efficiency emphasizes measured runtime/resource cost.",
            "",
            "## Dependencies, assets, and evidence",
            "",
        ]
    )
    for record in records:
        lines.extend(
            [
                f"### `{record['role']}` — `{record['candidate_id']}`",
                "",
                "- Dependency extras: "
                + ", ".join(f"`{item}`" for item in record["dependency_extras"]),
                "- Enabled techniques: "
                + ", ".join(f"`{item}`" for item in record["enabled_techniques"]),
                "- Tuned parameters: `"
                + json.dumps(record["tuned_parameters"], sort_keys=True)
                + "`",
                f"- Preset: `{record['preset']['path']}` (`{record['preset']['sha256']}`)",
                f"- Prepare command: `{record['prepare_command']}`",
            ]
        )
        assets = record["assets"]
        evidence = record["evidence"]
        lines.append(
            "- Assets: "
            + (
                "; ".join(
                    f"`{item['path']}` (`{item['sha256']}`, {item['bytes']} bytes)"
                    for item in assets
                )
                if assets
                else "none"
            )
        )
        lines.append(
            "- Evidence: "
            + (
                "; ".join(f"`{item['path']}` (`{item['sha256']}`)" for item in evidence)
                if evidence
                else "none"
            )
        )
        lines.append("")

    baseline_report = f"{target_relative}/reports/baseline.json"
    lines.extend(
        [
            "## Exact local comparison commands",
            "",
            "Run from the repository root. These use only the adaptive development",
            "split and do not access F3.",
            "",
            "```bash",
            _run_command(
                config=baseline_config_path,
                split=split_path,
                output=baseline_report,
                extras=("core",),
            ),
        ]
    )
    for record in records:
        role = str(record["role"])
        lines.append(
            _run_command(
                config=str(record["preset"]["path"]),
                split=split_path,
                output=f"{target_relative}/reports/{role}.json",
                extras=tuple(str(item) for item in record["dependency_extras"]),
            )
        )
    candidate_arguments = " ".join(
        f"--candidate {record['role']}={target_relative}/reports/{record['role']}.json"
        for record in records
    )
    lines.extend(
        [
            "```",
            "",
            "Compare the paired session outputs without selecting or promoting:",
            "",
            "```bash",
            (
                "uv run python -m scripts.compare_proposal_reports "
                f"--baseline {baseline_report} {candidate_arguments} "
                f"--output {target_relative}/reports/comparison.json"
            ),
            "```",
            "",
            "## Human decision boundaries",
            "",
            "### Gate A — freeze one candidate or reject all",
            "",
            "A human reviewer verifies the manifest/preset hashes, OOF pairing,",
            "scenario safety, dependencies, offline assets, runtime, licensing, and",
            "compiled parity. Gate A may freeze exactly one candidate for guarded F3",
            "or reject every proposal. The software cannot approve this gate.",
            "Running a candidate's prepare command validates it and prints the exact",
            "hash-bound activation command. Activation then prints verification and",
            "rollback commands; it is never performed by the campaign itself.",
            "",
            "### One-shot F3 — outside this proposal bundle",
            "",
            "This bundle contains no F3 path or command. Only after Gate A records one",
            "frozen commit, preset hash, asset hashes, and analysis plan may the separate",
            "guarded process expose F3 exactly once. The result is recorded even if bad;",
            "there is no candidate substitution or post-F3 tuning.",
            "",
            "### Gate B — accept or reject the frozen result",
            "",
            "A human reviews the one-shot result, integrity log, packaging, runtime, and",
            "private-evaluation readiness. Gate B may accept the already frozen candidate",
            "or retain the current champion. It cannot tune weights or choose another",
            "candidate using F3.",
            "",
            "## Rollback",
            "",
            f"- Known-good commit: `{rollback_commit}`",
            f"- Known-good preset: `{baseline_config_path}`",
            "- Preset SHA-256: `"
            + _hash_path(
                root / baseline_config_path,
                maximum_asset_files,
                maximum_asset_bytes,
            )[0]
            + "`",
            "- Rollback means retaining/re-running that preset; this proposal never",
            "  rewrites it. Use ordinary reviewed Git branch operations if code recovery",
            "  is required—do not reset or delete worktrees from an automated campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_command(
    *, config: str, split: str, output: str, extras: tuple[str, ...]
) -> str:
    extra_flags = _extra_flags(extras)
    prefix = "uv run" + (f" {extra_flags}" if extra_flags else "")
    return (
        f"{prefix} python -m scripts.run_unified_preset "
        f"--config {shlex.quote(config)} --split {shlex.quote(split)} "
        f"--output {shlex.quote(output)}"
    )


def _extra_flags(extras: tuple[str, ...]) -> str:
    values = sorted(set(extras) - {"core"})
    if "all" in values:
        values = ["all"]
    return " ".join(f"--extra {shlex.quote(value)}" for value in values)


def _path_record(
    path: Path,
    root: Path,
    maximum_asset_files: int,
    maximum_asset_bytes: int,
) -> PathRecord:
    digest, size, files = _hash_path(path, maximum_asset_files, maximum_asset_bytes)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "bytes": size,
        "files": files,
    }


def _hash_path(
    path: Path, maximum_asset_files: int, maximum_asset_bytes: int
) -> tuple[str, int, int]:
    if path.is_symlink():
        raise ValueError(f"proposal paths cannot be symlinks: {path}")
    if path.is_file():
        digest, size = _hash_file(path, maximum_asset_bytes)
        return digest, size, 1
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if len(files) > maximum_asset_files:
        raise ValueError(f"asset tree exceeds file bound: {path}")
    tree_digest = hashlib.sha256()
    size = 0
    for item in files:
        if item.is_symlink():
            raise ValueError(f"proposal asset tree contains a symlink: {item}")
        remaining = maximum_asset_bytes - size
        item_digest, item_size = _hash_file(item, remaining)
        relative = item.relative_to(path).as_posix().encode()
        tree_digest.update(len(relative).to_bytes(8, "big"))
        tree_digest.update(relative)
        tree_digest.update(bytes.fromhex(item_digest))
        size += item_size
    return tree_digest.hexdigest(), size, len(files)


def _hash_file(path: Path, maximum_bytes: int) -> tuple[str, int]:
    if maximum_bytes <= 0 or path.stat().st_size > maximum_bytes:
        raise ValueError(f"asset tree exceeds byte bound: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            if size > maximum_bytes:
                raise ValueError(f"asset tree exceeds byte bound: {path}")
            digest.update(block)
    return digest.hexdigest(), size


def _resolve_existing(root: Path, relative: str) -> Path:
    path = PurePath(relative)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError("proposal paths must stay inside the project")
    if any(part.lower() in {"f3", "protected"} for part in path.parts):
        raise ValueError("proposal materialization cannot reference protected data")
    resolved = (root / path).resolve()
    _require_below_root(resolved, root)
    if not resolved.exists():
        raise FileNotFoundError(relative)
    return resolved


def _require_below_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("proposal output must stay inside the project") from error


def _write_if_changed(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") == value:
            return
        raise FileExistsError(
            f"immutable proposal artifact already exists with other content: {path}"
        )
    path.write_text(value, encoding="utf-8")
