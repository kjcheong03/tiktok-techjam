# Membership-Preserving Residual Reranker Decision

## Decision

**PARK_RETESTABLE** — The technique is mechanically safe, but evidence did not pass every promotion gate.

This was evaluated independently and has not been registered in the autonomous
engine. The protected holdout was not accessed.

## Nested out-of-fold result

| Variant | Hit@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Guarded GBDT parent | 0.973333 | 0.737878 | 2.453333 | 0.878963 |
| Residual reranker | 0.973333 | 0.784915 | 2.453333 | 0.893075 |

Technical-score delta: `+0.014112`.

Paired evidence: `25` wins, `10` losses and `115` ties; bootstrap 95% interval
`[+0.001944, +0.026311]`; paired randomization `p=0.025297`.

## Safety result

- Exact Top-10 membership failures: `0`.
- Hit@10 absolute difference: `0.000000`.
- MTTC absolute difference: `0.000000`.
- Nonnegative outer folds: `3/5`.
- Worst scenario delta: `-0.003409`.
- Parent score reproduced: `True`.

## Interpretation

The search adaptively selected model family, observable feature set, regularization,
rerank depth, champion/model blend, expected-gain threshold, probability-margin
threshold, and movement limit inside each outer fold. Every reported candidate
session remained unseen by both its parent GBDT and residual learner.

The aggregate signal is promising, but the automatic selector was unstable: the
regularized logistic family produced the strongest held-out folds, while two
selected shallow-tree variants regressed. Preserve this implementation for a
future pre-registered conservative-selector and interaction experiment; do not
enable it in the champion or autonomous promotion pool from this result alone.
