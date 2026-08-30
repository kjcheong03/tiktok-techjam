# GhostLab Guarded GBDT Candidate Checkpoint

Date: 2026-08-26

Branch: `ghostlab/integration`

Policy: `ghostlab_guarded_constraint_gbdt_v1`

Runtime code commit: `4ae52c0`

Status: selected and production-validated development candidate; protected F3
holdout remains sealed

## Decision

The guarded constraint-aware GBDT is the selected development candidate. It has
the strongest valid OOF estimate, passed the frozen fold/scenario/Hit/runtime
gates, passed an independent leakage and replay audit, and is byte/config pinned in
the official runtime with exact starter-adapter parity.

This is not yet a prospective holdout confirmation. The 50-session F3 split has
not been accessed, and no private-organizer result exists. The paired OOF interval
versus the metadata GBDT narrowly crosses zero, so the uncertainty must remain
visible even though all predeclared promotion gates passed.

## Selected pipeline

1. Store corrected, provenance-aware conversation state while retaining raw user
   message history for lexical retrieval.
2. Ask the fixed sequence
   `other, other, use_case, other, size, other, other, size`.
3. Retrieve Top-200 with field-aware SQLite FTS5 BM25 using weights
   `2.0, 8.0, 4.0, 2.5, 1.5, 1.0`.
4. Apply the catalog-quality prior at weight `0.2` over the Top-50 head.
5. Normally rerank Top-50 with the corrected constraint-aware 56-round GBDT.
6. After an observable override invalidation, use the audited metadata 56-round
   GBDT for that and subsequent turns.
7. Normalize, deduplicate, catalog-validate, and truncate to the official Top-10.

The guard reads only runtime `ConversationState` invalidation reasons. It does not
read scenario, target, profile, future answer, evaluator, counterfactual, F3, or
private fields. The production runtime contains no experiment trace.

## Selection evidence

All figures are over the 150-session grouped adaptive development split.

| Candidate | Hit@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Original pairwise linear OOF | 0.933333 | 0.631720 | 2.926667 | 0.817649 |
| Audited metadata GBDT OOF | 0.973333 | 0.680278 | 2.466667 | 0.861417 |
| Guarded constraint GBDT OOF | **0.973333** | **0.737878** | **2.453333** | **0.878963** |
| Guarded compiled all-development replay | 0.973333 | 0.757508 | 2.353333 | 0.886852 |

The selected OOF delta versus metadata GBDT is `+0.017547`. Fold deltas are
`+0.020591, +0.023603, +0.014317, +0.029770, -0.001063`; four of five folds are
positive. Scenario deltas are boundary `+0.027812`, browsing `+0.002964`, buying
`+0.037194`, and intent override `0`.

Paired evidence versus metadata GBDT is 37 wins, 99 ties, and 14 losses, with 95%
bootstrap interval `[-0.000998, 0.035733]` and randomization p-value `0.058994`.
The interval/p-value are borderline and do not establish conventional significance.
They were not predeclared rejection gates; they remain a reason to require the
sealed F3/private evaluation before claiming generalization.

The guard routed 22 of 150 OOF sessions and 25 turns, all through observable
`earlier_preference_override`. Other frozen override reasons have semantic and
concurrency tests but no direct OOF examples.

## Exact compiled parity

The research agent, compiled runtime, and official `starter.Agent` produced exactly
the same response on 150 sessions and 349 turns:

- response mismatches: `0`;
- question/recommendation trace SHA-256:
  `c5fbe098a1ac27c869c71a100d0f3eb95b42494f53e1230112fc3015ba283f63`;
- policy file SHA-256:
  `97b10856fbeff77f3001ed9ce456a681447045c8c38395d95aefc9dd4d2237cf`;
- canonical policy SHA-256:
  `15a8b67ba680a403f888120540f8b790035d2d34510742c1c61cb627c88b3772`.

Model assets are content-addressed and verified before lazy loading:

| Asset | SHA-256 |
|---|---|
| Metadata GBDT | `10782d08ce20f8c9a60d3e2482ff577c887a35cc74e456c69c781409eb4df4d6` |
| Constraint GBDT | `2a3dc13284bb5ca53b9b795c9ec69ac921883be55efe6a239072302c4d4f6e6b` |

## Integration performance and packaging

Measured offline in `techjam-integration` after cherry-picking the compiled runtime:

| Check | Result | Budget |
|---|---:|---:|
| Cold initialization | 4.543854 s | 30 s |
| First response | 31.996292 ms | reported |
| Warm response p95 | 46.659083 ms | 500 ms |
| Peak process memory | 1177.141 MB | 4096 MB |
| Combined model assets | 0.148161 MB | 500 MB |
| Response failures | 0 | 0 |
| External calls | 0 | 0 |

Missing, corrupt, hash-mismatched, or schema-incompatible models use the existing
contract-safe keyword fallback. Model initialization is lazy and locked. Session
state and sparse retrieval are safe under interleaved/concurrent session tests.

## Final quality gates

- 132 unit/integration tests: pass.
- Ruff format and lint over the repository: pass.
- Mypy over the production boundary with project dependencies: pass.
- Frozen lockfile consistency: pass.
- Official 9-file organizer integrity: pass.
- Official weak-baseline reproduction: exact score `0.106710`.
- Research/compiled/starter exact parity: pass.
- Decision-ledger validation: 31 valid records.
- Git whitespace validation: pass.
- Offline runtime/network guard: pass.
- F3/holdout access: absent and false.

Reproduce the core gates from `techjam-integration`:

```bash
uv lock --check
uvx ruff format --check .
uvx ruff check .
uv run --frozen python -m unittest discover -s tests -p 'test_*.py'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --frozen python -m scripts.validate_guarded_compiled --require-default
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --frozen python -m scripts.measure_guarded_compiled
uv run --frozen python -m scripts.verify_phase0
uv run --frozen python -m scripts.validate_decision_ledger
uv run --frozen python -m scripts.validate_compiled
uv run --frozen python -m scripts.validate_champion_checkpoint
uv run --frozen --with mypy mypy --follow-imports=skip \
  starter/agent.py ghostlab/policy/models.py \
  ghostlab/runtime/agent.py ghostlab/runtime/guarded_gbdt.py
git diff --check
```

## Recovery and future retesting

- Selected candidate: `techjam-integration`, branch `ghostlab/integration`.
- Validated fallback: audited metadata GBDT evidence at original commit `cbfd7d5`;
  its integrated commits precede the guarded constraint chain.
- Immutable original recovery champion: `techjam`, commit `189f0c6`.
- All parked techniques and on/off/retest instructions:
  `docs/technique_registry_and_retest_guide.md`.

No GitHub push was performed. Push `ghostlab/integration` to the private `origin`
only after reviewing the final local commits.

## Holdout rule

Before any F3 access, freeze the exact integration commit, policy/config hashes,
model hashes, primary analysis, and access log. Run F3 exactly once, report the
result even if negative, and make no subsequent policy/feature/parameter changes.
