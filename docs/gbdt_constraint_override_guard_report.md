# Observable override-guard report

Date: 2026-08-26

Pre-outcome commits: amendment `37f2ed9`, implementation `37b1228`

Protected holdout: sealed and not accessed

## Decision

Advance the guarded constraint interaction to the integration tournament without
changing the champion/new baseline. It passed every frozen gate, exactly reproduced
both controls, and repaired the corrected v2 override regression with a single
observable routing rule. This is not yet a final champion decision: its paired
confidence interval narrowly crosses zero.

## Mechanism and isolation

The guarded candidate uses the corrected constraint-v2 reranker normally. If current
conversation state contains an actual override invalidation reason—global reset,
explicit earlier-preference reset, category override, or explicit override slot
replacement—it uses the matched metadata GBDT from that turn onward. Ordinary slot
replacement and negation do not activate the guard.

Runtime never receives a scenario label. Offline attribution may group the observed
routes by published scenario only after evaluation. Within each fold, base and
constraint models were fitted once under the already frozen rounds, then the same
objects were used for unguarded, guarded, and repeated-determinism replay. No feature,
model, threshold, fitting rule, or hyperparameter was added.

## Grouped outer-OOF results

| Variant | Hit@10 | MRR | MTTC | Technical score | Delta vs base |
|---|---:|---:|---:|---:|---:|
| Matched metadata GBDT | 0.973333 | 0.680278 | 2.466667 | 0.861417 | — |
| Corrected unguarded v2 | 0.966667 | 0.743164 | 2.500000 | 0.876283 | +0.014866 |
| Observable override guard | 0.973333 | 0.737878 | 2.453333 | 0.878963 | +0.017547 |

Guarded fold deltas versus base were `+0.020591`, `+0.023603`, `+0.014317`,
`+0.029770`, and `-0.001063`. Incremental deltas versus unguarded v2 were
`+0.000230`, `+0.018065`, `-0.000834`, `-0.007758`, and `+0.002931`.

Against base, the paired 10,000-resample interval is
`[-0.000998, +0.035733]`, p=`0.058994`, with 37 wins, 99 ties, and 14 losses.
Against unguarded v2, the incremental mean is `+0.002681`, interval
`[-0.005752, +0.015148]`, p=`0.826917`, and 4/143/3 wins/ties/losses. The guard is
mechanistically correct and passes predeclared gates, but its incremental statistical
evidence alone is not strong.

| Scenario | Guarded delta vs base | Guarded delta vs unguarded |
|---|---:|---:|
| Boundary | +0.027812 | 0.000000 |
| Browsing | +0.002964 | 0.000000 |
| Buying | +0.037194 | 0.000000 |
| Intent override | 0.000000 | +0.018279 |

## Exact routing attribution

The guard routed 22 of 150 sessions and 25 of 364 turns. All 25 routed turns carried
`earlier_preference_override`; no ordinary replacement or negative evidence routed.
Offline attribution found all 22 routed sessions were intent-override sessions and
all 22 intent-override sessions routed. The remaining 339 turns used constraint-v2.
Repeated replay with the same fitted models produced identical sessions and route
counts.

## Runtime and handoff

The isolated all-development runtime used the already frozen 56-round base and
constraint models. It scored `0.886852` on development-fit replay, had 5.825521
seconds cold start, 93.755666 ms warm-turn p95, 1288.688 MB peak memory, 0.148161 MB
combined model assets, and zero failures across 349 responses. The development-fit
score is not independent validation.

Integration must preserve both component model hashes, the guard reason set, exact
routing trace tests, the matched-base ablation, and the paired uncertainty caveat.
The machine-readable evidence is
`artifacts/reports/gbdt_constraint_override_guard_v1.json`.
