# Guarded constraint GBDT compilation report

Date: 2026-08-26

Original compilation branch: `exp/compile-guarded-gbdt`

Integrated branch: `ghostlab/integration`

Research parent: `93bf07b`

Protected holdout: sealed and not accessed

## Scope and status

This branch compiles the guarded constraint-aware GBDT challenger into the ordinary
`GhostLabRuntime` and untouched starter adapter. It is an isolated preparation for
the integration tournament, not a change to the champion/new-baseline worktree and
not an independent promotion decision.

The production path preserves the frozen research mechanism exactly:

1. corrected single-value `ConversationState` with raw message history;
2. fixed question order `other, other, use_case, other, size, other, other, size`,
   including consecutive-question deduplication in state bookkeeping;
3. field-weighted BM25 Top-200 with weights `2, 8, 4, 2.5, 1.5, 1`;
4. the `0.2` catalog-quality prior on the Top-50 head;
5. the content-addressed constraint GBDT normally; and
6. the content-addressed metadata GBDT from the first observable override
   invalidation onward.

The guard reasons remain the frozen set: global reset, explicit earlier-preference
reset, category override, and explicit override replacement. The 150-session OOF
evaluation exercised only `earlier_preference_override`. The other reasons have
targeted semantic test coverage but no direct OOF support; compilation deliberately
does not narrow or retune the guard after observing outcomes.

## Production hardening

- Both local JSON models are explicit typed config assets with pinned SHA-256
  digests. Absolute paths and project-root escapes are rejected.
- Model deserialization and hash verification are lazy, locked, offline, and cached.
- Missing, corrupt, or schema-incompatible assets raise inside the primary runtime
  and use the existing contract-safe keyword fallback.
- Sparse retrieval is protected for cross-thread use; each session has an isolated
  state lock.
- The compiled path contains no routing/question trace, counterfactual object,
  evaluator import, hidden label, scenario route, or external model call.

## Exact parity gate

The separate candidate config passed before the default compiled policy was changed.
After that gate, the isolated branch default was pointed to the same techniques and
the full gate was repeated through the official starter adapter.

| Check | Result |
|---|---:|
| Adaptive sessions | 150 |
| Compared turns | 349 |
| Research vs compiled response mismatches | 0 |
| Research vs starter response mismatches | 0 |
| Question/recommendation trace SHA-256 | `c5fbe098a1ac27c869c71a100d0f3eb95b42494f53e1230112fc3015ba283f63` |
| All-development replay score | 0.886852 |
| Hit@10 / MRR / MTTC | 0.973333 / 0.757508 / 2.353333 |

This score is an all-development refit replay and is not independent OOF evidence.
The selection evidence remains the guarded OOF score `0.878963` from the parent
challenger report.

## Runtime and packaging gate

| Measurement | Result | Budget |
|---|---:|---:|
| Cold initialization | 4.543854 s | <= 30 s |
| First response | 31.996292 ms | reported |
| Warm response p95 | 46.659083 ms | <= 500 ms |
| Peak process memory | 1177.141 MB | <= 4096 MB |
| Combined model assets | 0.148161 MB | <= 500 MB |
| Response failures | 0 | 0 |
| External calls | 0 | 0 |

The measured runtime was forced offline. Both configured model hashes matched their
files, lazy loading occurred on the first response, and no experiment trace was
present. Exact machine-readable evidence is in
`artifacts/reports/guarded_compiled_parity_v1.json` and
`artifacts/reports/guarded_compiled_runtime_v1.json`.

The table above is the isolated, contention-free rerun from
`ghostlab/integration` after the compiled commit was cherry-picked. The integration
worktree also passed 132 tests, repository Ruff checks, production-boundary mypy,
lockfile, organizer integrity, decision-ledger, compiled parity, checkpoint, and
offline fallback gates. See `docs/final_candidate_checkpoint.md`.
