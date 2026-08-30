# Wave 2 Autonomous Campaign Checkpoint

Status: implementation checkpoint on `exp/w2-autonomous-campaign`; not yet integrated
or promoted.

## Implemented

- Backward-compatible technique catalog v2 with explicit planned/available states,
  dependencies, conflicts, resources, mechanism tags, and retest triggers.
- Thirty versioned Wave 2 component IDs across the twelve planned families.
- Deterministic dependency closure, compatibility rejection, canonical candidate
  identities, singles/pairs/higher-order planning, and backward ablations.
- Resource-safe scheduling that separates heavy model jobs.
- Concurrent execution inside resource-safe waves, atomic checkpoints, and resume.
- Content-addressed cache entries with SHA-256 integrity checks.
- Paired bootstrap/randomization analysis, scenario gates, interaction calculation,
  and proposal-only selection.
- Deterministic successive halving and a bounded BOHB-style sampler.
- Combination-conditional weight spaces that can be used only through an explicit
  inner-validation context.
- Campaign freezing that refuses dirty worktrees, pins code/catalog/data/split
  hashes, and permanently declares protected holdout access forbidden.

The system does not generate code, install arbitrary dependencies, open F3, change a
champion preset, commit, push, or promote a candidate.

## Current availability behavior

The v2 catalog extends all 40 Wave 1 entries and declares 30 Wave 2 component IDs.
At this checkpoint only the new Hyperband and BOHB optimizer components are marked
available. All unfinished challenger components are `planned` and therefore produce
explicit `unavailable technique` planning records.

A dry run over all declared Wave 2 IDs currently yields:

| Item | Count |
|---|---:|
| Catalog entries | 70 |
| Wave 2 component IDs | 30 |
| Runnable control/optimizer candidates | 4 |
| Skipped unavailable combinations through order two | 462 |

These counts are expected to change as policy, ranking, and retrieval implementations
are integrated and their exact IDs become available.

## Validation

- New autonomy/HPO tests: 16 passed.
- Complete repository unit suite: 194 passed, 1 skipped after attaching the verified
  shared catalog and installing the declared `gbdt` extra.
- Ruff: all changed autonomy files pass.
- Mypy: all 15 changed autonomy/script source files pass.
- One pre-existing Wave 1 script remains outside repository-wide Ruff formatting;
  it was not modified by this track.

## Commands after a clean implementation commit

Freeze a manifest:

```bash
uv run --frozen python -m scripts.freeze_wave2_campaign \
  --template configs/campaigns/wave2_smoke_v1.template.json \
  --output artifacts/campaigns/wave2_smoke_v1/manifest.json
```

Plan legal and skipped candidates:

```bash
uv run --frozen python -m scripts.plan_wave2_campaign \
  --manifest artifacts/campaigns/wave2_smoke_v1/manifest.json \
  --output artifacts/campaigns/wave2_smoke_v1/plan.json
```

The freeze command deliberately fails before commit because `parent_commit` must
contain every source/config input. Execution adapters for newly available techniques
are added in the integration phase; the controller must never import code directly
from another worktree.

## Remaining before integration validation

1. Reconcile exact IDs and config bindings from the three implementation tracks.
2. Mark a component available only after its source, tests, factory, asset manifest,
   and smoke preset coexist in integration.
3. Add execution adapters for available fold-fit and runtime techniques.
4. Freeze the first clean smoke campaign and verify interrupt/resume with real
   technique jobs.
5. Run standalone, matched 2x2, higher-order, conditional-weight, and backward-
   ablation campaigns under the nested validation contract.
