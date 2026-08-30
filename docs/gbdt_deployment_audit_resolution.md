# GBDT deployment audit resolution

Date: 2026-08-26  
Amendment commit: `e001d872011e7104b7648d3c3db10d18dedc468a`  
Protected holdout: sealed and not accessed

## Resolution

The audit is resolved and the selected GBDT family remains ready for the integration
tournament. The nested OOF result is unchanged. Before the amended deployment-fit
run, `configs/experiments/gbdt_reranker_v1_amendment_1.json` froze the final model at
the median of the five already-selected outer stopping rounds:

```text
outer rounds: 100, 4, 56, 98, 34
sorted:       4, 34, 56, 98, 100
median:       56
```

No alternative round aggregation was compared against all-development outcomes.
The 56-round configuration was fitted twice on all 150 adaptive sessions, and the
two serialized assets were byte-identical.

## OOF evidence versus deployment fit

| Measurement | Role | Technical score | Hit@10 | MRR | MTTC |
|---|---|---:|---:|---:|---:|
| Nested selected family | Generalization evidence | 0.861417 | 0.973333 | 0.680278 | 2.466667 |
| Initial 4-round refit | Superseded deployment-fit check | 0.828369 | 0.940000 | 0.649675 | 2.826667 |
| Frozen 56-round refit | Current deployment-fit check | 0.857554 | 0.973333 | 0.661624 | 2.380000 |

The `0.861417` result stitches models whose stopping rounds were selected strictly
inside their outer-training sides. It remains the evidence for family promotion.
The `0.857554` result uses every adaptive session for training and therefore is not
independent evidence; it verifies that a reproducibly frozen deployable model
retains the expected behavior. The earlier 4-round asset is preserved only as audit
provenance and is superseded by `gbdt_reranker_v2_round56.json`.

## Executable gates

All gates passed:

- Fold: the candidate beat its matched linear control in all five outer folds;
  deltas were `+0.063591`, `+0.044535`, `+0.062560`, `+0.037890`, and `+0.008193`.
- Scenario: no Hit@10 regression; scenario technical-score deltas were Boundary
  `+0.006831`, Browsing `+0.041024`, Buying `+0.033901`, and Intent Override
  `+0.091591`.
- Determinism: both 56-round refits produced SHA-256
  `10782d08ce20f8c9a60d3e2482ff577c887a35cc74e456c69c781409eb4df4d6`
  and zero mismatched session outcomes.
- Parity: research deployment-fit and isolated-runtime metrics matched exactly.
- Packaging: 77,779-byte model, `4.727217 s` cold start, `47.720541 ms` warm-turn
  p95, and `1174.922 MB` peak memory, all within budget.
- Reliability: 353 actual responses were instrumented. Reset exceptions, response
  exceptions, invalid responses, external calls, and total failures were all zero.

The executable results, per-fold and per-scenario checks, full provenance, model
hash, response counters, and performance measurements are in
`artifacts/reports/gbdt_deployment_audit_v1.json`.
