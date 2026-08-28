# State V2 Adaptive Residual V2 Decision

## Verdict

Promote `ranking.top10_residual_reranker.v2` into the autonomous engine as an
optional, fit-required experimental technique. Keep it disabled by default until
the engine has produced a fold-safe fitted asset for the chosen parent pipeline.

The primary comparison is the fair State Baseline V2 ablation: the same parent
trace is evaluated with residual ranking off and on. The secondary comparison is
a compatibility test against the stronger ranked State V2 parent; it does not
replace the primary inclusion decision.

The protected holdout was not accessed. Every residual model and policy was
selected only by grouped inner-fold predictions, then fitted without that outer
fold and evaluated once on it. Fit receipts prove disjoint training and validation
IDs. The parent configuration was identical in each off/on pair.

| Parent | Residual off | Residual on | Delta | Nonnegative folds | Decision |
|---|---:|---:|---:|---:|---|
| primary | 0.782154 | 0.876700 | +0.094546 | 5/5 | PROMOTE_EXPERIMENTAL |
| secondary | 0.885391 | 0.912467 | +0.027076 | 5/5 | PROMOTE_EXPERIMENTAL |

### primary

- MRR: `0.562291` → `0.877444`.
- Hit@10 difference: `0.000000`.
- MTTC difference: `0.000000`.
- Paired interval: `[+0.077139, +0.111746]`; `p=0.000100`.
- Scenario deltas: `{'boundary': 0.07901800000000003, 'browsing': 0.10478600000000005, 'buying': 0.10042400000000007, 'intent_override': 0.056234000000000006}`.
- Fold deltas: `[0.10291899999999998, 0.08730000000000004, 0.08015500000000009, 0.09221299999999999, 0.11056299999999997]`.

### secondary

- MRR: `0.721526` → `0.811778`.
- Hit@10 difference: `0.000000`.
- MTTC difference: `0.000000`.
- Paired interval: `[+0.014462, +0.039703]`; `p=0.000200`.
- Scenario deltas: `{'boundary': 0.03749999999999998, 'browsing': 0.03516699999999995, 'buying': 0.027105000000000046, 'intent_override': 0.0011359999999999149}`.
- Fold deltas: `[0.030645000000000033, 0.023789999999999978, 0.004166000000000003, 0.043246999999999924, 0.03429799999999994]`.


## What was adaptive

For every outer fold, the training side alone screened 24 rank-aware model
specifications, shortlisted six, evaluated six single models plus 15 two-model
ensembles, and searched 216 safe activation policies per candidate. This is 4,536
inner-fold configurations per outer fold. The selectable dimensions included
model family, feature subset, regularization/tree settings, ensemble membership,
rerank depth, blend weight, expected-gain threshold, probability-margin threshold,
and maximum moved IDs. An explicit adaptive-off candidate was available.

The selection utility rewarded mean score gain and a conservative fold lower
bound, while penalizing scenario regression and unnecessary activation. Different
outer folds selected different models and gates, so the result is not a single
manually chosen weight disguised as adaptation.

## Safety contract

- Output contains exactly the parent's normalized Top 10, reordered only.
- Hit@10 and MTTC must be exactly unchanged.
- Runtime features exclude target ID, target profile, scenario label, future
  answers, and evaluator outcomes.
- The technique fails closed to the parent order when its gates do not activate.
- Promotion requires positive paired evidence, at least four nonnegative outer
  folds, no material scenario regression, and zero membership failures.

## Interpretation and limitation

The result supports engine inclusion, not immediate replacement of the runtime
champion. It is evidence from the public adaptive set, not the sealed competition
holdout. A deployable asset still needs one final outcome-blind configuration
selection using cross-fitted development predictions, a fit on all allowed
development IDs, an immutable fit receipt, and exact runtime/off-state parity
tests. The engine must continue to treat this technique as fit-required and must
never reuse an evaluation-fold fit at deployment.
