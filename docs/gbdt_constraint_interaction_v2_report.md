# Corrected constraint-aware GBDT report

Date: 2026-08-26

Amendment: `gbdt_constraint_interaction_v1_amendment_1`

Pre-outcome commits: amendment `ed8ebc2`, implementation `94e91c8`

Protected holdout: sealed and not accessed

## Decision

Park the corrected unguarded interaction. It retains a meaningful scalar and MRR
gain over the audited metadata GBDT, but it fails the unchanged Hit and
intent-override scenario robustness gates. The defective v1 result remains preserved
and explicitly superseded; this v2 result is the only valid constraint-interaction
evidence.

## Correctness repairs

The candidate, features, Top-50 head, folds, tree capacity, seed, and round counts
did not change. Before measuring v2, the branch added:

- scoped observable override handling for global resets, explicit earlier-preference
  resets, targeted corrections, category changes, ambiguous language, negation, and
  repeated correction;
- exact training/runtime deduplication of repeated `other` questions;
- invocation-scoped `ContextVar` binding and thread-local sparse indexes so concurrent
  sessions cannot share state or SQLite connections;
- targeted semantic, feature-parity, interleaving/concurrency, Ruff, and mypy checks.

## Grouped outer-OOF evidence

| Variant | Hit@10 | MRR | MTTC | Technical score | Delta |
|---|---:|---:|---:|---:|---:|
| Matched metadata GBDT | 0.973333 | 0.680278 | 2.466667 | 0.861417 | — |
| Corrected constraint v2 | 0.966667 | 0.743164 | 2.500000 | 0.876283 | +0.014866 |

The paired 10,000-resample interval is `[-0.007212, +0.035689]`, the paired
randomization p-value is `0.182982`, and wins/ties/losses are 40/92/18. Fold deltas
were `+0.020361`, `+0.005538`, `+0.015151`, `+0.037528`, and `-0.003994`, satisfying
the four-of-five gate.

| Scenario | Paired mean reward delta |
|---|---:|
| Boundary | +0.027812 |
| Browsing | +0.002964 |
| Buying | +0.037194 |
| Intent override | -0.018279 |

The candidate failed two frozen gates: overall Hit@10 regressed by `-0.006666`
against a floor of `-0.005`, and intent-override reward regressed by `-0.018279`.
The confidence interval also crosses zero. These are selection failures even though
the average technical score and MRR improved.

Active constraint coverage ratio remained the dominant added feature (374 outer
model splits). Invalidation count fell from the defective v1 diagnostic count of 77
to 17, consistent with corrected semantics rather than stale-state exploitation.
Split counts are diagnostic and do not override the paired/scenario gates.

## Deployment and scope

The isolated 56-round all-development refit scored `0.893319`; this is deployment-fit
evidence, not independent validation. Cold start was 5.761333 seconds, warm-turn p95
94.514042 ms, peak process memory 1292.125 MB, model size 0.073985 MB, and all 344
instrumented responses succeeded without external calls.

The full v2 report is `artifacts/reports/gbdt_constraint_interaction_v2.json`. The
unchanged v1 artifacts remain available only for audit provenance.
