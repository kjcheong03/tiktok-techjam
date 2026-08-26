# GhostLab Champion Checkpoint

Date: 2026-08-26
Branch: `ghostlab/implementation`
Policy: `ghostlab_champion_linear_v1`
Status: production-ready development checkpoint; not the final holdout candidate

## Decision

The current simpler champion is complete and recoverable. It is suitable as the
control implementation for subsequent incompatible challengers. The protected
50-session F3 split remains sealed and this checkpoint must not be described as
prospectively confirmed.

## Compiled pipeline

1. Concatenate raw user-message history for lexical recall.
2. Retrieve 200 candidates with field-aware SQLite FTS5 BM25 using weights:
   title `2.0`, categories `8.0`, features `4.0`, details `2.5`, store `1.5`,
   description `1.0`.
3. Apply the Bayesian catalog-quality prior with weight `0.2` over the top 50.
4. Apply the frozen pairwise linear reranker over the top 50. Only two features
   are enabled: product-feature overlap (`0.634672535385014`) and catalog quality
   (`0.4938576967870529`). The fit used 32,746 training pairs and L2 `0.1`.
5. Ask the fixed development-selected sequence:
   `other, other, use_case, other, size, other, other, size`.
6. Normalize, deduplicate, catalog-validate, and truncate recommendations to the
   official Top-10 contract. Token usage and external calls are zero.

## Implementation and policy inventory

The repository contains more techniques than the compiled champion activates.
This inventory distinguishes deployable champion behavior, implemented research
switches, experiment/search infrastructure, and future work that is still absent.
The detailed decision history and evidence links are in
`docs/technique_decision_ledger.md` and
`artifacts/evidence/technique_decisions.jsonl`.

### Active in the compiled champion

| Area | Active implementation/policy |
|---|---|
| State | Raw normalized message history; session-isolated memory. |
| Query | Concatenated raw user history. |
| Retrieval | SQLite FTS5 field-aware BM25, Top-200, validated six-field weights. |
| Ranking | Catalog-quality prior at `0.2`, then frozen two-feature pairwise linear reranker over Top-50. |
| Dialogue | Fixed eight-action sequence with a bounded stop after the sequence. |
| Output | Normalization, deduplication, catalog validation, and official Top-10 truncation. |
| Runtime | Offline compiled policy; no dense model initialization, model download, token use, or external call. |

### Implemented and experimentally switchable, but inactive

| Area | Available techniques and policies |
|---|---|
| State | Current-turn only, raw history, single-value, multi-value, and compressed representations; negative evidence, provenance, and override invalidation switches. |
| Question policy | No question, organizer fixed, explicit sequence, missing-priority, feature-first, uncertainty-limited, `other`-always, and heuristic adaptive policies. |
| Retrieval | Organizer keyword baseline, generic MiniLM dense retrieval, RRF, weighted sparse/dense fusion, and field-aware sparse retrieval. |
| Filtering/ranking | Coverage-aware structured filter, fixed lexical reranker, profile prior, catalog-quality prior, and learned linear reranker. |
| Routing | Observable decision-list and shallow route-stump research paths, including always-route controls. |

These paths are retained because an inactive technique may be a useful comparator or
interaction dependency. Inactive does not mean recommended for submission.

### Implemented research and optimization infrastructure

- deterministic replay, counterfactual action evaluation, route analysis, and
  question-policy evaluation;
- grid, random, beam/racing, and evidence-allocation search utilities with bounded
  manifests and cacheable results;
- grouped frozen splits, paired statistics, integrity checks, holdout firewall,
  compiled parity, runtime measurements, and offline contract validation;
- typed technique configurations, lazy technique registry, experimental runtime
  switches, traces, evidence records, and the validated technique-decision ledger.

### Explicitly not implemented in this checkpoint

- learned counterfactual question policy;
- retrieval-specialized dense/vector index and sparse semantic expansion;
- cross-fitted GBDT/LambdaMART reranker;
- compact cross-encoder reranker;
- raw-plus-structured hybrid query-construction challenger.

These remain isolated worktree proposals. Their presence in the execution plan or
decision ledger must not be interpreted as implemented code or positive evidence.

## Validation results

All figures below use only the 150-session adaptive development split.

| Estimate | Hit Rate@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Fixed field BM25 + quality | 0.913333 | 0.630860 | 3.266667 | 0.800591 |
| Selected two-feature learner, five-fold OOF | 0.933333 | 0.631720 | 2.926667 | 0.817649 |
| Selected learner refit on all 150 | 0.933333 | 0.639508 | 2.940000 | 0.819719 |
| Compiled official adapter | 0.933333 | 0.639508 | 2.940000 | 0.819719 |

The refit-to-OOF difference is `+0.002070`. The refit score is a deployment-fit
measurement, not an independent generalization estimate. The five-fold OOF score
is the appropriate development estimate.

Compiled-versus-research parity passed with zero mismatches across all sessions,
questions, recommendations, turn outcomes, aggregate metrics, and scenario
metrics. The compiled policy JSON SHA-256 is
`57857af805256240e3f6d4e51ba164cf9cbe8575174840da981faa4811390b19`.

## Performance and packaging

Measured on the adaptive split in the current local environment:

| Check | Result | Budget |
|---|---:|---:|
| Cold initialization | 4.035942 s | 30 s |
| Warm turn p95 | 44.304333 ms | 500 ms |
| Maximum observed turn | 57.975250 ms | — |
| Peak process memory | 932.266 MB | 4096 MB |
| Runtime source and config | 0.039492 MB | — |
| Bundled model assets | 0 MB | 500 MB |
| External calls per turn | 0 | 0 |

Loading only the two nonzero learned features reduced peak memory from roughly
1.28 GB to 0.93 GB and cold start from 5.45 s to 4.04 s without changing behavior.
The catalog is a 57.74 MB organizer-provided runtime input.

## Quality gates

- Official organizer/evaluator integrity: pass; 9 protected hashes unchanged.
- Frozen official weak-baseline reproduction: pass; score `0.106710`.
- Compiled/research parity: pass.
- Official adapter offline smoke test with frozen dependencies: pass.
- Ruff format and lint across participant-owned code: pass.
- Mypy over the submission runtime boundary: pass, 9 files.
- Unit/integration suite: pass, 79 tests using the frozen lockfile.
- Lockfile consistency: pass with `uv lock --check`.
- Git whitespace validation: pass.
- User-specific paths/private-key signatures in runtime/config: none found.
- F3 access log: absent/empty; holdout not accessed.

The frozen organizer files and the byte-exact official reference are explicitly
excluded from mechanical formatting so integrity validation remains meaningful.

## Reproduction commands

```bash
uv lock --check
uvx ruff format --check .
uvx ruff check .
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python -m scripts.verify_phase0
uv run python -m scripts.validate_compiled
uv run python -m scripts.validate_champion_checkpoint
```

The offline contract smoke test uses `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; the champion does not initialize the optional dense
retriever or download a model.

## Holdout firewall

`configs/validation/primary_analysis.json` is intentionally not updated to this
checkpoint. It must be frozen only after the incompatible challenger tournament
selects the actual final candidate. The guarded promotion command compares the
predeclared candidate ID and compiled-policy hash before recording or reading F3,
so the current stale declaration causes a safe refusal rather than holdout access.

No code or policy changes may be selected using F3. When a final candidate is
eventually frozen, F3 is run exactly once and its result is reported even if it is
negative.

## Remaining work outside this checkpoint

This checkpoint deliberately excludes the planned GBDT, cross-encoder, stronger
dense retrieval, learned question policy, and query-construction challengers. They
should begin from this implementation in isolated worktrees, use the same adaptive
evaluation contract, and merge only after cross-fitted evidence. This checkpoint
remains the control if none generalizes better.
