# GBDT × learned-question interaction decision

## Outcome

`PARKED_INTERACTION`. The frozen linear action-value question policy does not
improve the audited shallow-metadata GBDT. The current GBDT with its fixed
question sequence remains unchanged.

| Variant | Hit@10 | MRR | MTTC | Technical score | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched fold-local GBDT + fixed sequence | 0.973333 | 0.680278 | 2.466667 | 0.861417 | — |
| Matched fold-local GBDT + learned questions | 0.960000 | 0.677370 | 2.773333 | 0.847744 | -0.013672 |
| Matched fold-local GBDT + no questions | 0.406667 | 0.263093 | 7.266667 | 0.356928 | -0.504489 |

The matched control reproduced the prior nested OOF result exactly for all four
metrics. This is an important validity check: the interaction comparison did not
silently change the GBDT, retrieval path, split, or evaluator.

## Fold-safe design

The manifest was committed as `64aaabc` before any outcome was run. For outer
fold `k`, the experiment:

1. rebuilt the exact `shallow_metadata_depth3` model using only the outer-training
   side and the previously frozen inner stopping rule;
2. replayed the fixed question trajectory only on those 120 training sessions;
3. evaluated every legal current question plus absorbing stop with that fold's
   GBDT as the frozen continuation;
4. fitted the unchanged per-action ridge model (`l2=1.0`) from those training-only
   labels; and
5. evaluated the resulting policy once on the 30 unseen sessions in fold `k`.

No weights from the earlier linear-ranker question run were reused, no global
parameter was selected, and the protected holdout was absent and unaccessed.

## Generalization evidence

Fold score deltas were:

- fold 0: -0.008065;
- fold 1: -0.016130;
- fold 2: -0.002917;
- fold 3: -0.002587; and
- fold 4: -0.039252.

The candidate therefore achieved 0/5 nonnegative folds, while promotion required
at least 4/5. Its paired mean session-reward delta was -0.013672 with a 95%
bootstrap interval of `[-0.034967, 0.002828]`, randomization `p=0.200380`, and
12 wins / 114 ties / 24 losses.

Scenario reward deltas were boundary -0.107188, browsing -0.001667, buying
-0.016722, and intent override -0.004091. Both the overall Hit@10 gate and the
minimum scenario delta gate failed.

## Behavior and diagnosis

The learned policy made 410 question decisions across 150 sessions. It asked
`other` 139 times and `feature` 84 times; it emitted stop 53 times across 23
sessions. Every action was legal, behavior was deterministic, runtime/research
sessions matched, and 2,938 instrumented responses produced zero failures.

The no-question ablation is decisive context: removing questions collapsed the
score to 0.356928. Questioning is essential for this simulator. The rejected
component is specifically the frozen linear counterfactual policy: it did not
generalize its action values well enough to replace the strong fixed sequence,
especially for boundary and buying sessions. This result does not establish that
all adaptive question policies are inferior.

## Gate record

Passed: exact control reproduction, legality, zero failures, determinism, runtime
parity, and nominal packaging budgets.

Failed: +0.005 technical-score gain, at least four nonnegative folds, no overall
Hit@10 regression, and no scenario delta below -0.005.

Latency was measured while other CPU-heavy challengers were running. The observed
4.734748-second cold start and 48.790875-ms warm-turn p95 are therefore diagnostic,
not isolated performance evidence. An isolated rerun was not justified because
the accuracy gates already failed decisively.

## Durable decision

Do not merge this interaction into the champion or audited GBDT candidate, and do
not retry the same linear family or reuse its fitted weights. A future question
challenger is justified only if it changes model capacity or learning objective
materially and receives a fresh grouped, outcome-blind predeclaration.

Evidence:

- `configs/experiments/gbdt_question_interaction_v1.json`
- `artifacts/reports/gbdt_question_interaction_v1.json`
- `artifacts/experiments/gbdt_question_interaction_v1/counterfactual_labels.jsonl`
- `artifacts/experiments/gbdt_question_interaction_v1/oof_sessions.jsonl`
