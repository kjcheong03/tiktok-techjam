# Query Construction Challenger V1

Date: 2026-08-26
Branch: `exp/query-construction`
Parent champion: `189f0c6338e2d2ec1a795dce543e881ff2037f2a`
Split: 150-session `nested_v1` adaptive set
Protected holdout: not accessed

## Decision

Do not promote a query-construction change standalone. Keep
`raw_plus_active` as an interaction reserve for a retrieval-specialized dense model;
park the remaining tested variants until a named dependency changes. Raw history
remains the sparse-retrieval control.

The standalone evaluation deliberately used fixed field-aware BM25 plus the catalog
quality prior, without the all-development learned reranker. This prevents a new
query representation from being selected using a reranker fitted on the same outer
sessions. Any future query+learned-ranker combination must fit the ranker inside its
training folds.

## Results

| Variant | Technical score | Paired delta | 95% paired bootstrap | Decision |
|---|---:|---:|---:|---|
| `raw_history` | 0.800591 | 0.000000 | [0.000000, 0.000000] | Control |
| `raw_plus_active` | 0.800591 | 0.000000 | [0.000000, 0.000000] | Dense interaction reserve |
| `compressed_raw` | 0.786369 | -0.014222 | [-0.031156, 0.000000] | Park standalone |
| `negation_safe_hybrid` | 0.736361 | -0.064230 | [-0.102624, -0.029825] | Retest only after parser improvement |
| `structured_active` | 0.580262 | -0.220329 | [-0.276549, -0.163604] | Park standalone |
| `category_constraints` | 0.143593 | -0.656998 | [-0.716894, -0.593254] | Prune this version |

Every outer fold selected `raw_history`; the stitched selected reward was
`0.800591`. Raw history, raw-plus-active, compressed raw, and negation-safe hybrid
all reached any-eligible-turn Recall@200 `0.993333`. The loss therefore occurs
primarily in Top-10 ordering and conversational timing, not broad candidate recall.

## Diagnosis

- Raw product language is unusually valuable to the field-aware sparse index.
- Structured-only representations remove catalog-matching details and sharply
  reduce ranking quality even when Recall@200 remains fairly high.
- Adding active constraints to raw text did not change final session rewards. It is
  not worth extra sparse-runtime complexity by itself.
- Compression changed only five sessions and predominantly hurt them; the current
  short simulator conversations do not need a term-budget intervention.
- The tested negation-safe parser removed useful context along with stale intent and
  made intent-override performance worse, so it must not replace raw history.
- A separate structured natural-language channel may still help a semantic encoder,
  which is why `raw_plus_active` is retained for the query+dense dependency test.

Detailed metrics, sessions, scenario results, recall diagnostics, latency, paired
tests, and fold selections are stored in
`artifacts/reports/challenger_query_v1.json`.
