# Constrained GBDT reranker report

Date: 2026-08-26  
Parent control: `189f0c6338e2d2ec1a795dce543e881ff2037f2a`  
Split: frozen `nested_v1`, 150 adaptive sessions  
Protected holdout: sealed and not accessed

## Decision

Promote `shallow_metadata_depth3` to the integration tournament as
`ranking.gbdt_v2`. This does not modify or replace the compiled linear champion in
this worktree. The candidate cleared the predeclared outer-OOF, paired-evidence,
stability, and runtime gates. Its deployable all-development refit is a parity and
packaging result, not independent generalization evidence.

## Method

The challenger replaces only the final Top-50 ordering stage. Candidate generation,
the quality prior, raw-history state, and fixed question sequence remain identical
to the champion. Training uses grouped pairwise LambdaRank gradients and shallow
gradient-boosted regression trees. All turns from a session remain on the same side
of every split. Each outer fold selects its stopping round on the next frozen fold
inside the outer-training side, then refits on the complete outer-training side.

Features are deterministic catalog/runtime values: original rank and percentile,
query length, six field overlaps, catalog quality, rating, log popularity, and
metadata completeness. Native `NaN` values and explicit indicators represent
missing metadata; no target, scenario label, simulator state, or research-only
field reaches inference.

## Outer-OOF results

| Variant | Hit@10 | MRR | MTTC | Technical score | Delta vs linear |
|---|---:|---:|---:|---:|---:|
| Fixed field + quality | 0.913333 | 0.630860 | 3.266667 | 0.800591 | -0.017058 |
| Two-feature linear champion | 0.933333 | 0.631720 | 2.926667 | 0.817649 | — |
| Rank-only depth 1 | 0.913333 | 0.630860 | 3.266667 | 0.800591 | -0.017058 |
| Shallow lexical depth 2 | 0.953333 | 0.626336 | 2.793333 | 0.828701 | +0.011052 |
| Shallow metadata depth 3 | 0.973333 | 0.680278 | 2.466667 | 0.861417 | +0.043768 |

For the selected candidate versus the linear champion, the mean paired session
reward delta is `+0.043767`; the 10,000-resample bootstrap 95% interval is
`[+0.016708, +0.073183]`, the paired randomization p-value is `0.002600`, and the
session counts are 59 wins, 62 ties, and 29 losses.

The selected fold scores are `0.854194`, `0.849892`, `0.864528`, `0.857644`, and
`0.882011` (population standard deviation `0.011249`, worst fold `0.849892`). Every
fold exceeded its corresponding linear control. Hit@10 did not regress in any
aggregate scenario; the small Boundary slice remained at `0.875` and improved MRR
from `0.592857` to `0.598958`.

## Interpretation and cost

Across the five outer models, the most frequent split features were log rating
count (503), reciprocal rank (300), category overlap (276), rank percentile (168),
original rank (158), metadata completeness (111), and catalog quality (100). This
is not an unstable single-feature proxy: retrieval order, lexical relevance, and
catalog-quality/metadata signals all contribute. Missing indicators were available
but rarely selected; native missing routing remained active.

The isolated deployable measurement passed every checkpoint budget: model asset
`0.005963 MB`, cold initialization `6.220752 s`, warm-turn p95 `60.124708 ms`, and
peak process memory `1099.312 MB`. External calls, tokens, and runtime failures were
zero. The all-development refit scored `0.828369`; its stopping round was selected
inside an all-development training/validation split and its behavior matched the
isolated runtime measurement exactly.

The full machine-readable manifest, per-session predictions, fold IDs, scenario
metrics, paired evidence, feature importance, model hashes, and measurements are in
`configs/experiments/gbdt_reranker_v1.json` and
`artifacts/reports/gbdt_reranker_v1.json`.
