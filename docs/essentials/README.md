# Essential GhostLab documentation

This folder contains full copies of the documents needed to understand and execute
the project. They are not abbreviated navigation pages.

The established source paths remain in place because the official competition files
are hash-pinned and historical reports link to those locations. The test
`tests/test_essential_docs.py` requires every copy here to remain byte-for-byte equal
to its source, preventing silent documentation drift.

## Reading order

1. [Project overview](project_overview.md) — problem, data, starter, interface,
   metrics, repository map, and submission entry points.
2. [Competition specification](competition_specification.md) — authoritative task,
   session protocol, allowed attributes, metrics, model policy, and deliverables.
3. [Submission rules](submission_rules.md) — allowed and disallowed submission
   contents, output rules, reproducibility, and packaging.
4. [Unified technique operations](unified_technique_operations.md) — complete folder
   map, every technique and switch, implementation paths, dependencies, assets,
   presets, combinations, commands, evidence, and retest rules.
5. [First champion checkpoint](champion_checkpoint.md) — pairwise-linear champion,
   selected question sequence, implementation, score, validation, and reproduction.
6. [Final guarded candidate](final_candidate_checkpoint.md) — guarded-GBDT candidate,
   exact pipeline, `0.878963` OOF score, parity, runtime, recovery, and holdout rule.
7. [Technique decision ledger](technique_decision_ledger.md) — what improved or
   regressed, why techniques were promoted or parked, interactions, and retest triggers.
8. [Wave 2 plan](wave2_advanced_challenger_and_autonomy_plan.md) — research basis,
   technique catalog, compatibility, validation design, and autonomy design.
9. Read the actual Wave 2 results:
   [policy](wave2_policy_track_validation.md),
   [retrieval](wave2_retrieval_track_report.md), and
   [ranking](wave2_ranking_report.md).
10. [Autonomous system reference](autonomous_unified_system_reference.md) — the
    implemented pure-baseline search, F0/F1/F2 execution, pruning, conditional HPO,
    dense/neural asset preflight, overfitting controls, one-command checkpoint/resume,
    top-three proposals, hash-bound activation/rollback, and human gates. Section 16 is
    the current `autonomous_state_v2_v1` eight-step execution guide.
11. [State Baseline V2 integration](state_baseline_v2_integration.md) — exact teammate
    parity, native state/query/history switches, presets, autonomous linkage,
    combination result, and overfitting-safe validation.
12. [Adaptive autonomous optimizer](../adaptive_autonomous_optimizer.md) — dual-track
    search modes, 42 conditional runtime parameters, observable activation, true
    successive halving, fold-fitted residual ranking, fit receipts, overfitting
    controls, commands, and recovery.

## What to use for each task

| Task | Document |
|---|---|
| Understand `other`, `size`, `use_case`, and other agent fields | `competition_specification.md` |
| Find the first champion's exact question sequence | `champion_checkpoint.md` |
| Locate or configure any technique | `unified_technique_operations.md` |
| Understand why a technique helped or failed | `technique_decision_ledger.md` and its track report |
| Install dependencies and run presets | `unified_technique_operations.md` |
| Run the autonomous campaign | `autonomous_unified_system_reference.md` |
| Understand the score to beat | `final_candidate_checkpoint.md` |
| Use or retest the teammate State V2 baseline | `state_baseline_v2_integration.md` |
| Run or extend adaptive structure-plus-parameter optimization | `../adaptive_autonomous_optimizer.md` |
| Understand Wave 2 hypotheses versus actual evidence | Wave 2 plan followed by the three result reports |

## Essential non-Markdown companions

These remain at their canonical repository paths:

- `docs/agent_api_contract.json`
- `docs/evaluation_config.json`
- `configs/techniques/catalog_v1.json`
- `configs/techniques/catalog_v2.json`
- `ghostlab/campaign/bindings.py`
- `artifacts/evidence/technique_decisions.jsonl`

## Maintenance rule

Edit the established source document first, then synchronize its copy in this folder.
Run:

```bash
uv run pytest -q tests/test_essential_docs.py
```

Do not independently edit an essential copy, because the synchronization test will
correctly treat that as drift.
