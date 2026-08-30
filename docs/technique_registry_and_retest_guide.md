# GhostLab Technique Registry and Retest Guide

Date: 2026-08-26

## Purpose

This is the authoritative human-readable map of the techniques tried in the
first GhostLab implementation and the advanced challenger tournament. It exists
so a future change to retrieval, state, ranking, dialogue, data, or evaluation
can reopen a parked technique without recreating its code or forgetting why it
previously won or lost.

The machine-readable chronological record remains
`artifacts/evidence/technique_decisions.jsonl`. Detailed session outcomes and
fold metrics remain in `artifacts/reports`. This guide adds the practical mapping
from technique to switch, worktree, commit, report, result, and retest condition.

On `ghostlab/unified-techniques`, reusable source from the isolated challengers is
also consolidated in this worktree. Installation, asset preparation, unified
presets, and combination commands are authoritative in
`docs/unified_technique_operations.md`; the branch/worktree references below
remain the immutable provenance and raw-evidence locations.

## Which directory means what

| Directory | Branch | Purpose |
|---|---|---|
| `techjam/` | `ghostlab/implementation` | Immutable original champion/recovery point at `189f0c6`. |
| `techjam-integration/` | `ghostlab/integration` | Current advanced integration candidate and the directory to use for final validation. |
| `techjam-compile/` | `exp/compile-guarded-gbdt` | Isolated official-runtime compilation worktree; temporary until its validated commits enter integration. |
| Other `techjam-*` directories | `exp/*` | Preserved prototype implementations, dependencies, tests, and evidence. |

Do not implement new experiments in the immutable `techjam/` worktree. Start a
new branch/worktree from the desired control, or resume the preserved worktree.

## Score hierarchy and evidence labels

All comparable figures below use the same 150-session grouped development split.
`OOF` means each complete session was evaluated by a model fitted without that
session. An all-development refit is a packaging/parity measurement, not an
independent generalization estimate.

| Candidate | OOF score | Status |
|---|---:|---|
| Guarded constraint-aware GBDT | **0.878963** | Selected development candidate; independent audit and compiled parity passed, with borderline CI caveat. |
| Corrected unguarded constraint GBDT | 0.876283 | Parked: Hit@10 and intent-override gates failed. |
| Metadata GBDT | 0.861417 | Audited integration fallback. |
| Cross-encoder score + GBDT | 0.858094 | Parked: lower score and latency failure. |
| Learned adaptive questions + GBDT | 0.847744 | Parked: negative in all five folds. |
| Structured E5 union + deep GBDT | 0.829423 | Parked: lost to Top-50 GBDT in every fold. |
| Pairwise linear champion | 0.817649 | Original compiled champion/control. |
| Learned adaptive questions + linear ranker | 0.808951 | Parked standalone. |
| Fixed field BM25 + catalog quality | 0.800591 | Strong nonlearned ranking control. |
| Compact cross-encoder standalone/nested | 0.790915 | Parked standalone. |

The earlier dense/query score `0.835125` is not a validated promotion result. An
independent audit found global query selection and rerank-depth confounds. Matched
controls later showed that the structured query contributed only about `0.001062`
over raw E5 union, and the complete dense/deep GBDT tournament lost to Top-50 GBDT.

## First-version technique inventory

These techniques were developed before the isolated advanced challengers.

| Area | Technique | Switch or implementation | Best evidence and decision |
|---|---|---|---|
| State | Current turn only | `state_mode: off` or baseline stateless path | Weak control; retained as fallback. |
| State | Single-value state | `state_mode: single` | Implemented control; replaced by stronger history/state variants. |
| State | Structured multi-value memory | `state_mode: multi` | Early gain; retained as an interaction/state feature source. |
| State | Raw conversation history | `state_mode: raw_history` | Promoted; preserved discriminative lexical terms. |
| State | Compressed active state | `state_mode: compressed` | Implemented; lost to raw history in query testing. |
| State | Negative evidence | `negative_evidence` | Useful inside structured state; retained and tested with invalidation. |
| State | Provenance and override invalidation | `provenance`, `override_invalidation` | Retained; later hardened by the constraint challenger audit. |
| Dialogue | No questions | `question_policy: none` | Essential negative control; post-GBDT score `0.356928`. |
| Dialogue | Fixed organizer questions | `question_policy: fixed` | Implemented control. |
| Dialogue | Other-always | `question_policy: other_always` | Strong early policy; superseded by the fixed sequence. |
| Dialogue | Missing/feature/uncertainty heuristics | question-policy variants | Implemented and tested; did not beat the sequence. |
| Dialogue | Fixed selected sequence | `question_policy: sequence` | Promoted: `other, other, use_case, other, size, other, other, size`. |
| Retrieval | Organizer keyword baseline | baseline keyword path | Reproduced at technical score `0.106710`. |
| Retrieval | Field-aware SQLite FTS5 BM25 | `sparse_field_weights` | Promoted Top-200 generator; feature/category/title interactions helped. |
| Retrieval | Generic MiniLM dense | dense route/model switch | Weak standalone recall; preserved as a baseline. |
| Fusion | RRF and weighted sparse/dense fusion | `retrieval_route`, fusion weights | Parked until a dense head is independently competitive. |
| Routing | Observable stump/decision list | research routing paths | Collapsed to always sparse; dependency-gated. |
| Filtering | Coverage-aware structured filter | `enabled_filters` / research path | Parked because false exclusions outweighed gains. |
| Profile | Fixed profile prior | profile-prior research path | Increasing fixed weights degraded score; only reconsider as gated learned evidence. |
| Ranking | Fixed lexical reranker | experimental reranker switch | Parked standalone. |
| Ranking | Catalog quality prior | `quality_prior_weight` | Promoted at weight `0.2`. |
| Ranking | Pairwise learned linear reranker | `reranker: learned_linear` | Original champion, OOF `0.817649`. |
| Search | Grid/beam control | optimization search utilities | Retained as auditable bounded-search control. |
| Search | Adaptive evidence allocator | optimization search utilities | Lost the equal-budget comparison; preserved for changed spaces. |

The original champion is documented in `docs/champion_checkpoint.md`. Its compiled
configuration is `configs/compiled_policy.json` in the immutable champion
worktree.

## Advanced challenger registry

### Query construction

| Field | Record |
|---|---|
| Worktree / branch | `techjam-query/` / `exp/query-construction` |
| Final commit | `c2620aa` |
| Variants | Raw, structured, raw+active, compressed, negation-safe raw fallback, category constraints. |
| Result | Raw and raw+active tied at `0.800591`; all other standalone forms lost. |
| Decision | Preserve raw history; raw+active/negation-safe remained dependency reserves. |
| Retest when | The dense model, parser, active-state semantics, or candidate ranker materially changes. |

### Learned counterfactual questioning

| Field | Record |
|---|---|
| Worktree / branch | `techjam-question-policy/` / `exp/learned-question-policy` |
| Final commit | `cab92c0` |
| Mechanism | Fold-local linear action-value model over legal questions plus stop. |
| Standalone result | `0.808951` versus fixed all-development control `0.819719`. |
| GBDT interaction | `techjam-gbdt-question/`, commit `6711182`; `0.847744` versus `0.861417`, negative in 5/5 folds. |
| Tree diagnosis | Observable depth-3 diagnostic collapsed to `other`; estimated gain `0.000878`. |
| Decision | Stop current policy search. Questions are useful; the current observable features cannot learn the oracle exceptions. |
| Retest when | New runtime-observable uncertainty/action-value signals or a materially different objective/continuation becomes available. |

### Retrieval-specialized dense search and query interaction

| Field | Record |
|---|---|
| Worktree / branch | `techjam-dense/` / `exp/dense-retrieval` |
| Commits | Dense base `25d7fc3`; query interaction `d8cf054`. |
| Model | Offline pinned E5-small-v2 index; MiniLM retained as control. |
| Recall finding | BM25 all-stage Recall@200 `0.952593`; E5 `0.797037`. |
| Initial interaction | `0.835125`, later downgraded by independent audit for selection/depth confounds. |
| Matched GBDT tournament | `techjam-gbdt-dense/`, commit `b6500ff`; Top-50 GBDT `0.861417`, best deep/dense arm `0.829423`. |
| Decision | Park E5/deep union. Dense assets and index remain isolated and switchable in the prototype. |
| Retest when | A new dense model demonstrates stable unique recall beyond BM25 before end-to-end evaluation, or the candidate distribution changes. |

### Nonlinear GBDT/LambdaMART ranking

| Field | Record |
|---|---|
| Worktree / branch | `techjam-gbdt/` / `exp/gbdt-reranker` |
| Final audited commit | `cbfd7d5` |
| OOF result | `0.861417` versus linear `0.817649`; all five folds positive. |
| Deployable refit | Frozen median 56 rounds, score `0.857554`, model SHA recorded in the deployment audit. |
| Runtime | About 48 ms warm p95, 77,779-byte model, zero failures in the audited run. |
| Decision | Promoted integration fallback and control for later interactions. |

### Compact cross-encoder

| Field | Record |
|---|---|
| Worktree / branch | `techjam-cross-encoder/` / `exp/cross-encoder` |
| Final commit | `071eda9` |
| Standalone/nested result | `0.790915` versus linear OOF `0.817649`. |
| GBDT interaction | `techjam-neural-rank/`, commit `e211434`; `0.858094` versus GBDT `0.861417`. |
| Runtime | Interaction warm p95 `525.367 ms`, above the 500 ms gate. |
| Decision | Park zero-shot CE and do not fine-tune on 150 repeatedly reused sessions. |
| Retest when | Candidate head, hardware/runtime budget, model, or independent training data materially changes. |

### Runtime constraint features and observable override guard

| Field | Record |
|---|---|
| Worktree / branch | `techjam-gbdt-constraints/` / `exp/interaction-gbdt-constraints` |
| Final commit | `93bf07b` |
| Defective v1 | `0.884943`, superseded because stale override evidence remained active. |
| Corrected v2 | `0.876283`, parked because Hit@10 and intent-override gates failed. |
| Guarded v2 | `0.878963`, same Hit@10 as base, 4/5 folds positive, all scenario gates pass. |
| Guard behavior | Constraint-aware GBDT normally; matched metadata GBDT after an observable override invalidation. |
| Audit | Independent replay and metrics exact; no target/profile/scenario/future/F3 leakage. CI versus base narrowly crosses zero. |
| Decision | Selected development candidate after independent audit and exact compiled/starter parity; audited GBDT remains the fallback. |
| Retest when | Override examples or independent sessions increase, particularly for global/category/targeted corrections not represented in current OOF data. |

## Compatibility results already tested

| Combination | Outcome | Do not repeat unless |
|---|---|---|
| Structured query + E5 + linear ranker | Confounded exploratory gain; invalid promotion attribution. | Re-evaluated with nested selection and matched depth. |
| E5/deep candidates + GBDT | Lost to Top-50 GBDT in every fold. | Dense unique recall materially improves. |
| Cross-encoder score + GBDT | `-0.003322` and latency failure. | Model/hardware/candidate head changes. |
| Learned questions + GBDT | `-0.013672`, 0/5 positive folds. | New observable question-value signals appear. |
| Corrected constraint state + GBDT | Aggregate gain but unsafe override/Hit behavior. | Use the already-tested observable guard. |
| Constraint GBDT + override fallback | `+0.017547` versus GBDT; current integration challenger. | Reconfirm on untouched holdout/private evaluation. |

No unrestricted three- or four-way sweep was run. Once dense, CE, and learned
questioning failed their matched GBDT interactions, adding them to the constraint
winner would increase runtime and selection risk without a supported mechanism.

## How on/off behavior is preserved

There are two kinds of switches.

### Lightweight configuration switches

`ghostlab/policy/models.py` defines typed runtime switches such as:

- `retrieval_route`;
- `state_mode`;
- `question_policy` and `question_order`;
- `sparse_field_weights`;
- `negative_evidence`, `provenance`, and `override_invalidation`;
- `quality_prior_weight`;
- `reranker` and `rerank_k`;
- `enabled_filters`.

A disabled lightweight technique follows a deterministic path and should not load
its optional model or asset. Add a new versioned config rather than mutating a
historical evidence config.

### Heavy or incompatible switches

Dense E5, cross-encoder, learned-question research, deep-candidate GBDT, and other
The original prototypes remain in isolated branches/worktrees, while their reusable
modules, tests, manifests, and small evidence are consolidated on
`ghostlab/unified-techniques`. They are intentionally lazy and optional: disabling
a heavy component avoids its imports, model assets, initialization cost, and failure
modes. Fold-local research interactions without a promoted fitted model remain
dedicated scripts rather than pretending to be deployable runtime toggles.

To resume one, open its preserved worktree, branch from its final commit, declare a
new versioned manifest, and compare against the current integration control. Do not
rewrite its historical report or decision.

## Reopening and retesting a technique

1. Read this guide, `docs/technique_decision_ledger.md`, and the named raw report.
2. Confirm the stated dependency or data condition has materially changed.
3. Create a new branch/worktree from the control you intend to beat.
4. Commit a manifest before evaluating outcomes. Record the candidate limit,
   frozen folds, gates, runtime budgets, and holdout status.
5. Fit learned components inside complete-session training folds.
6. Include an exact matched control and backward ablation.
7. Report paired session evidence, five fold deltas, scenario deltas, Hit@10, MRR,
   MTTC, technical score, failures, latency, memory, and assets.
8. Add a new decision version. Never replace the old result.
9. Keep F3 sealed until one complete final candidate is frozen.

List preserved worktrees with:

```bash
git -C "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam" worktree list
```

Open a specific prototype directly, for example:

```bash
code "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam-dense"
```

## Current selection and recovery rule

Compiled integration validation is complete:

1. guarded constraint GBDT (`0.878963` OOF) is the selected development
   candidate;
2. its official compiled/starter replay is exact on 150 sessions and 349 turns,
   with zero mismatches and all-development score `0.886852`;
3. audited metadata GBDT (`0.861417` OOF) is the validated integration fallback;
4. original linear champion commit `189f0c6` is the immutable recovery point.

The compiled implementation entered integration in commit `4ae52c0`. Exact
parity and runtime evidence are in
`artifacts/reports/guarded_compiled_parity_v1.json` and
`artifacts/reports/guarded_compiled_runtime_v1.json`. The full selected-candidate
handoff is `docs/final_candidate_checkpoint.md`.

The protected F3 result, when eventually run exactly once for a frozen candidate,
must be recorded even if negative. No technique or parameter may change after F3
access.
