from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGE_ORDER = (
    "split",
    "fit",
    "diversity",
    "llm",
    "evaluate",
    "baselines",
    "validate",
    "campaign",
    "package",
    "finalists",
    "compare",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def signature(self) -> str:
        signature_payload: dict[str, object] = {
            "command": self.command,
            "outputs": self.outputs,
        }
        if len(self.command) > 1:
            script = ROOT / self.command[1]
            if script.is_file() and script.suffix == ".py":
                signature_payload["script_sha256"] = hashlib.sha256(
                    script.read_bytes()
                ).hexdigest()
        payload = json.dumps(signature_payload, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the adaptive structural fit, bounded LLM selection, public "
            "evaluation, validation, GhostLab campaign and Top-3 packaging as one "
            "resumable pipeline"
        )
    )
    parser.add_argument("--from-stage", choices=STAGE_ORDER, default="split")
    parser.add_argument("--through-stage", choices=STAGE_ORDER, default="compare")
    parser.add_argument(
        "--force-stage",
        action="append",
        choices=STAGE_ORDER,
        default=[],
        help="rerun a completed stage; repeat for multiple stages",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="print commands and checkpoint state without executing",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/campaigns/adaptive_hybrid_pipeline/checkpoint.json",
    )
    parser.add_argument("--log-dir", default="artifacts/logs/adaptive_hybrid_pipeline")
    parser.add_argument("--campaign-candidate-limit", type=int, default=500)
    parser.add_argument("--campaign-beam-width", type=int, default=24)
    parser.add_argument("--campaign-higher-order-rounds", type=int, default=8)
    parser.add_argument("--campaign-f1-candidates", type=int, default=24)
    parser.add_argument("--campaign-f2-candidates", type=int, default=6)
    parser.add_argument("--campaign-hpo-trials", type=int, default=2)
    parser.add_argument(
        "--campaign-max-samples",
        type=int,
        help="optional bounded campaign prefix; omit for all 2,200 sessions",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if STAGE_ORDER.index(args.from_stage) > STAGE_ORDER.index(args.through_stage):
        raise ValueError("from-stage must not come after through-stage")
    positive = {
        "campaign-candidate-limit": args.campaign_candidate_limit,
        "campaign-beam-width": args.campaign_beam_width,
        "campaign-f1-candidates": args.campaign_f1_candidates,
        "campaign-f2-candidates": args.campaign_f2_candidates,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.campaign_higher_order_rounds < 0 or args.campaign_hpo_trials < 0:
        raise ValueError("campaign round and HPO counts cannot be negative")
    if args.campaign_max_samples is not None and args.campaign_max_samples <= 0:
        raise ValueError("campaign-max-samples must be positive")


def stage_specs(args: argparse.Namespace) -> tuple[StageSpec, ...]:
    python = sys.executable
    selected_config = "configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json"
    public_report = "artifacts/reports/adaptive_hybrid_development_1650.json"
    validation_report = (
        "artifacts/reports/adaptive_hybrid_structural_v2_validation.json"
    )
    campaign_report = "artifacts/reports/adaptive_hybrid_campaign_1650.json"
    campaign_checkpoint = "artifacts/campaigns/adaptive_hybrid_1650_v1/checkpoint.json"
    campaign_command = [
        python,
        "scripts/run_adaptive_hybrid_campaign.py",
        "--config",
        selected_config,
        "--dataset",
        "data/public_set.jsonl",
        "--dataset",
        "data/synthetic_1000_public_like.jsonl",
        "--dataset",
        "data/independent_template_1000.jsonl",
        "--candidate-limit",
        str(args.campaign_candidate_limit),
        "--beam-width",
        str(args.campaign_beam_width),
        "--higher-order-rounds",
        str(args.campaign_higher_order_rounds),
        "--f1-candidates",
        str(args.campaign_f1_candidates),
        "--f2-candidates",
        str(args.campaign_f2_candidates),
        "--hpo-trials-per-structure",
        str(args.campaign_hpo_trials),
        "--checkpoint",
        campaign_checkpoint,
        "--output",
        campaign_report,
    ]
    if args.campaign_max_samples is not None:
        campaign_command.extend(("--max-samples", str(args.campaign_max_samples)))
    return (
        StageSpec(
            "split",
            (python, "scripts/build_adaptive_lineage_split.py"),
            (
                "data/splits/adaptive_hybrid_lineage_75_25_v1.json",
                "artifacts/reports/adaptive_lineage_reconstruction_audit_v1.json",
            ),
        ),
        StageSpec(
            "fit",
            (python, "scripts/train_adaptive_hybrid.py"),
            (
                "configs/adaptive_hybrid_1a_3b_1650_final_v1.json",
                "artifacts/models/adaptive_union_gbdt_1650_final_v1.json",
                "artifacts/models/adaptive_union_gbdt_1650_final_v1.fit_receipt.json",
                "artifacts/reports/adaptive_hybrid_training_1650_final_v1.json",
            ),
        ),
        StageSpec(
            "diversity",
            (
                python,
                "scripts/validate_adaptive_diversity.py",
                "--config",
                "configs/adaptive_hybrid_1a_3b_1650_final_v1.json",
            ),
            ("artifacts/reports/adaptive_dense_diversity_v2.json",),
        ),
        StageSpec(
            "llm",
            (python, "scripts/compare_local_llm_rankers.py"),
            (
                "artifacts/reports/local_llm_ranker_comparison_v1.json",
                selected_config,
            ),
        ),
        StageSpec(
            "evaluate",
            (
                python,
                "scripts/run_adaptive_hybrid.py",
                "--config",
                selected_config,
                "--dataset",
                "data/public_set.jsonl",
                "--dataset",
                "data/synthetic_1000_public_like.jsonl",
                "--dataset",
                "data/independent_template_1000.jsonl",
                "--lineage-manifest",
                "data/splits/adaptive_hybrid_lineage_75_25_v1.json",
                "--partition",
                "development",
                "--output",
                public_report,
            ),
            (public_report,),
        ),
        StageSpec(
            "baselines",
            (python, "scripts/evaluate_adaptive_reference_baselines.py"),
            (
                "artifacts/reports/adaptive_baseline_a_development_1650.json",
                "artifacts/reports/adaptive_baseline_b_development_1650.json",
            ),
        ),
        StageSpec(
            "validate",
            (
                python,
                "scripts/validate_adaptive_hybrid.py",
                "--config",
                selected_config,
                "--adaptive-report",
                public_report,
                "--training-report",
                "artifacts/reports/adaptive_hybrid_training_1650_final_v1.json",
                "--output",
                validation_report,
            ),
            (validation_report,),
        ),
        StageSpec("campaign", tuple(campaign_command), (campaign_report,)),
        StageSpec(
            "package",
            (
                python,
                "scripts/package_adaptive_top_three.py",
                "--campaign-report",
                campaign_report,
                "--base-config",
                selected_config,
                "--output",
                "artifacts/reports/adaptive_hybrid_top3.json",
            ),
            ("artifacts/reports/adaptive_hybrid_top3.json",),
        ),
        StageSpec(
            "finalists",
            (
                python,
                "scripts/evaluate_adaptive_development_finalists.py",
            ),
            ("artifacts/reports/adaptive_finalist_development_evaluations.json",),
        ),
        StageSpec(
            "compare",
            (python, "scripts/build_adaptive_system_comparison.py"),
            (
                "artifacts/reports/adaptive_system_comparison_1650.json",
                "artifacts/reports/adaptive_system_comparison_1650.md",
            ),
        ),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "stages": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("stages"), dict
    ):
        raise ValueError("adaptive pipeline checkpoint has an invalid schema")
    return payload


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _outputs_exist(stage: StageSpec) -> bool:
    return all((ROOT / output).is_file() for output in stage.outputs)


def _completed(stage: StageSpec, checkpoint: dict[str, Any], forced: set[str]) -> bool:
    record = checkpoint["stages"].get(stage.name, {})
    return (
        stage.name not in forced
        and record.get("status") == "complete"
        and record.get("signature") == stage.signature
        and _outputs_exist(stage)
    )


def _run_stage(
    stage: StageSpec,
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}.log"
    checkpoint["stages"][stage.name] = {
        "status": "running",
        "signature": stage.signature,
        "command": list(stage.command),
        "started_at_unix": time.time(),
        "log": str(log_path.relative_to(ROOT)),
    }
    _write_checkpoint(checkpoint_path, checkpoint)
    print(f"START stage={stage.name} log={log_path.relative_to(ROOT)}", flush=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(ROOT)
    )
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            stage.command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            checkpoint["stages"][stage.name].update(
                {"status": "interrupted", "finished_at_unix": time.time()}
            )
            _write_checkpoint(checkpoint_path, checkpoint)
            raise
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        checkpoint["stages"][stage.name].update(
            {
                "status": "failed",
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
            }
        )
        _write_checkpoint(checkpoint_path, checkpoint)
        raise RuntimeError(f"stage {stage.name} failed with exit code {return_code}")
    missing = [output for output in stage.outputs if not (ROOT / output).is_file()]
    if missing:
        checkpoint["stages"][stage.name].update(
            {
                "status": "failed",
                "reason": "missing_outputs",
                "missing_outputs": missing,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
            }
        )
        _write_checkpoint(checkpoint_path, checkpoint)
        raise FileNotFoundError(f"stage {stage.name} did not create: {missing}")
    checkpoint["stages"][stage.name].update(
        {
            "status": "complete",
            "return_code": 0,
            "outputs": list(stage.outputs),
            "elapsed_seconds": elapsed,
            "finished_at_unix": time.time(),
        }
    )
    _write_checkpoint(checkpoint_path, checkpoint)
    print(f"DONE stage={stage.name} elapsed={elapsed:.1f}s", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    checkpoint_path = ROOT / args.checkpoint
    log_dir = ROOT / args.log_dir
    checkpoint = _load_checkpoint(checkpoint_path)
    forced = set(args.force_stage)
    first = STAGE_ORDER.index(args.from_stage)
    last = STAGE_ORDER.index(args.through_stage)
    selected = stage_specs(args)[first : last + 1]
    plan = [
        {
            "stage": stage.name,
            "status": (
                "skip_complete" if _completed(stage, checkpoint, forced) else "run"
            ),
            "command": list(stage.command),
            "outputs": list(stage.outputs),
            "log": str((log_dir / f"{stage.name}.log").relative_to(ROOT)),
        }
        for stage in selected
    ]
    if args.show_plan:
        print(json.dumps({"schema_version": 1, "stages": plan}, indent=2))
        return
    print(json.dumps({"event": "pipeline_plan", "stages": plan}), flush=True)
    for stage in selected:
        if _completed(stage, checkpoint, forced):
            print(f"SKIP stage={stage.name} reason=checkpoint_complete", flush=True)
            continue
        _run_stage(
            stage,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            log_dir=log_dir,
        )
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
