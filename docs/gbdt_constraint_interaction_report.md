# Constraint-aware GBDT interaction report

Date: 2026-08-26

Audited parent: `cbfd7d5dd595c5637608ba28f46f57777c7e153e`

Pre-evaluation commit: `b112554`

Split: frozen `nested_v1`, 150 adaptive sessions

Protected holdout: sealed and not accessed

## Decision

Promote `metadata_depth3_plus_runtime_constraints` to the integration tournament,
without modifying the compiled champion/new baseline. It is the only candidate in
the immutable manifest and passed every predeclared score, stability, scenario,
failure, determinism, asset, and provisional runtime gate. Runtime was measured
while other CPU-heavy worktrees were active, so the latency values are explicitly
contention-affected and require one isolated confirmation before integration
runtime sign-off.

## Frozen comparison

Both sides use the same field-aware BM25 candidate generator, quality prior,
raw-history query, fixed question sequence, Top-50 head, grouped outer folds,
depth-three/seven-leaf tree capacity, seed, and audited per-fold round counts
`[100, 4, 56, 98, 34]`. The backward ablation removes all added state features and
reproduces the audited GBDT control exactly at `0.861417`.

The candidate adds only deterministic public-runtime evidence:

- active positive constraint coverage count and token ratio;
- explicit-negative contradiction count and ratio;
- separate explicit and simulator-answer coverage ratios;
- neutral no-preference count/ratio;
- invalidation count/presence, without scoring inactive values as preferences;
- current turn, sparse Top-1 margin, and normalized sparse-score entropy.

No profile, target, scenario label, undisclosed intent, future observation, or F3
field enters training or inference. No-preference is deliberately neutral rather
than a product contradiction. Inactive overridden values are context indicators
only and never contribute to candidate coverage.

## Grouped outer-OOF results

| Variant | Hit@10 | MRR | MTTC | Technical score | Delta |
|---|---:|---:|---:|---:|---:|
| Audited metadata GBDT control | 0.973333 | 0.680278 | 2.466667 | 0.861417 | — |
| + runtime constraint context | 0.980000 | 0.738254 | 2.326667 | 0.884943 | +0.023526 |

The paired 10,000-resample bootstrap interval is
`[+0.006970, +0.041826]`, the paired randomization p-value is `0.006599`,
and the session counts are 42 wins, 87 ties, and 21 losses.

Fold deltas were `+0.027511`, `+0.046329`, `+0.009662`, `+0.043563`, and
`-0.010804`. The candidate therefore passed the declared four-of-five gate exactly;
the fifth-fold loss is retained as a generalization warning rather than hidden by
the aggregate.

| Scenario | Paired mean reward delta |
|---|---:|
| Boundary | +0.032812 |
| Browsing | +0.011498 |
| Buying | +0.035645 |
| Intent override | +0.019902 |

All aggregate scenarios cleared the no-regression threshold. Boundary has only
eight adaptive sessions, so its positive result is not a high-confidence standalone
claim.

## Diagnosis

Across the five outer models, the dominant added split feature was active constraint
coverage ratio (374 splits). Invalidation count appeared 77 times, sparse Top-1
margin 33 times, explicit coverage ratio 22 times, sparse entropy 17 times, and turn
once. Negative contradiction, no-preference, simulator-only coverage, and the
override-presence flag were available but not selected.

This pattern supports the intended interaction: structured state helps the ranker
interpret which lexical evidence is currently active instead of replacing the raw
history that already works well. Split counts are diagnostic, not causal
attribution; the only causal ablation in this run removes the complete added feature
block. Any later adaptive question policy or retrieval-head change must repeat that
backward ablation because it changes the distribution of state and confidence
features.

The all-development 56-round refit scored `0.892908` on the same development set.
That value is deployment-fit evidence only and must not be presented as independent
generalization. Two independent models were structurally identical, the model asset
is `0.073895 MB`, all 344 instrumented responses succeeded, and external calls were
zero. The contention-affected measurement was 6.356569 seconds cold start,
99.319917 ms warm-turn p95, and 1289.047 MB peak process memory.

The complete immutable manifest, per-session OOF outcomes, folds, scenario results,
paired evidence, feature importance, model hashes, and gate results are in
`configs/experiments/gbdt_constraint_interaction_v1.json` and
`artifacts/reports/gbdt_constraint_interaction_v1.json`.
