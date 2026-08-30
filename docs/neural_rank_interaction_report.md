# Pinned cross-encoder score + GBDT interaction report

Date: 2026-08-26
Audited parent: `404948d0952171113cbba238c67cdfa5d8f5541a`
Predeclaration commit: `1c0072b`
Split: frozen `nested_v1`, 150 adaptive sessions
Protected holdout: unavailable and not accessed

## Decision

Park `ranking.gbdt_ce_interaction_v1`. The single predeclared interaction lost to
the audited shallow metadata GBDT, was not stable across outer folds, regressed
Hit@10, and missed the warm-turn p95 budget. No deployable neural tree artifact was
created, and the audited GBDT report/model and compiled champion were not modified.

## Frozen comparison

The control is the audited `shallow_metadata_depth3` Top-50 LambdaMART. The only
candidate used the identical metadata features, depth 3, seven-leaf limit, learning
rate, maximum rounds, inner stopping rule, grouped session folds, and fold-local
refit procedure, then appended exactly two columns: the sigmoid score from
`cross-encoder/ms-marco-MiniLM-L6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, and an explicit missing-score
indicator. Passage construction was frozen as `catalog_fields_v2`. There was no
model, feature, weight, Top-K, or tree-hyperparameter search.

Scores were generated locally for all 74,300 unique query-candidate pairs in the
fixed quality-reranked Top-50 heads of all 1,500 trajectories. The content-identified
cache was complete, contained no label-derived fitting, and produced no missing
scores during training or OOF evaluation. Every learned tree fit stayed inside an
outer fold: five inner stopping fits and five outer-training refits. There was no
all-development refit.

## Complete OOF evidence

| Variant | Hit@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Audited metadata GBDT | 0.973333 | 0.680278 | 2.466667 | 0.861417 |
| Metadata GBDT + pinned CE score | 0.966667 | 0.683870 | 2.520000 | 0.858094 |

The candidate's paired session-reward delta was `-0.003322`; the 10,000-resample
paired bootstrap 95% interval was `[-0.016561, +0.007822]`, and the paired
randomization p-value was `0.642536`. Session counts were 19 wins, 114 ties, and 17
losses.

Candidate fold scores were `0.858629`, `0.849892`, `0.863000`, `0.835345`, and
`0.883966`. Deltas versus the audited control were `+0.004435`, `0.000000`,
`-0.001528`, `-0.022299`, and `+0.001955`; only two folds improved. Boundary MRR
regressed by `0.041666`, Buying Hit@10 regressed by `0.016667`, and Intent Override
MRR regressed by `0.011363`. Browsing MRR improved by `0.013565`, but that slice gain
did not offset the complete-policy loss.

## Interaction, importance, and cost

Relative to the two-feature linear baseline, the audited GBDT gained `+0.043768`,
the previously parked standalone cross-encoder selection gained `-0.026734`, and
the combination gained `+0.040445`. The arithmetic interaction term was therefore
positive (`+0.023411`), but this mostly reflects recovering a standalone neural
loss: the combined policy still scored below GBDT alone.

The cross-encoder score was used in 127 splits across the five outer models, behind
log rating count (510), category overlap (251), reciprocal rank (249), and original
rank (174). Its missing indicator was never used because the score cache was
complete. This confirms that the learner consumed the semantic feature; it does
not establish a stable benefit.

The isolated uncached runtime measurement used an outer-fold model only as a cost
reference. Cold start was `7.593400 s`, warm-turn p95 was `525.367292 ms`, peak
memory was `2160.156 MB`, and total unique model assets were `87.708862 MB`.
Failures, external calls, and missing scores were zero. The 500 ms p95 gate failed.

The machine-readable report contains complete OOF sessions, folds, scenarios,
paired evidence, backward ablations, feature importance, cache/model hashes, and
runtime measurements at `artifacts/reports/neural_rank_interaction_v1.json`.
