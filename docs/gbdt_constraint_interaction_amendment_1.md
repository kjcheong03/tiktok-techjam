# Constraint-aware GBDT correctness amendment

This amendment was committed before measuring corrected outcomes. The original
`0.884943` result is preserved but superseded as promotion evidence because its
active-constraint features could include stale preferences after an observable
intent override.

The correction-only retest keeps the candidate features, Top-50 head, five grouped
folds, fold rounds, tree capacity, seed, all-development rounds, and promotion gates
unchanged. It permits only three correctness repairs:

1. scoped observable override invalidation as frozen in the JSON amendment;
2. exact training/runtime deduplication parity for repeated questions;
3. invocation-scoped runtime context instead of a shared mutable binding.

Targeted typing fixes and tests are allowed only to prove those repairs. The old
report/model remain untouched, corrected artifacts use v2 paths, and neither the
champion nor the protected holdout is accessed.
