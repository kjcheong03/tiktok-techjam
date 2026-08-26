# GBDT + dense retrieval interaction

## Decision

Park the structured dense/query interaction and retain arm A, the audited Top-50
metadata GBDT, unchanged. The matched OOF control reproduced the prior report
exactly at `0.861417`. Every deeper arm scored lower than A, and the fold-nested
structured query added only `0.000755` over a matched raw-history E5 union.

No protected/F3 data was present or accessed. The experiment ran only on the 150
frozen `nested_v1` development sessions. All query and GBDT choices were made
inside each outer-training complement; unseen outer folds were used only once for
OOF measurement.

## Why four arms were required

The earlier dense-query result mixed semantic retrieval with a larger rerank
budget. This run froze the following matched attribution chain before evaluation:

| Arm | Candidate pool and rank depth | Technical score | Hit@10 | MRR | MTTC |
|---|---|---:|---:|---:|---:|
| A | raw BM25 200; quality + GBDT Top-50 | **0.861417** | **0.973333** | **0.680278** | **2.466667** |
| B | raw BM25 200; quality + GBDT Top-200 | 0.823782 | 0.926667 | 0.667717 | 2.993333 |
| C | raw BM25 200 + raw E5 200; quality + GBDT Top-400 | 0.828668 | 0.953333 | 0.606228 | 2.493333 |
| D | raw BM25 200 + nested-query E5 200; quality + GBDT Top-400 | 0.829423 | 0.946667 | 0.629632 | 2.640000 |

The causal increments are therefore:

- B − A, deep reranking alone: `-0.037635`, with only 1/5 non-negative folds.
- C − B, raw dense union at matched depth: `+0.004886`, with 3/5 non-negative
  folds and paired CI `[-0.023944, 0.035299]`.
- D − C, nested structured query at matched depth: `+0.000755`, with 3/5
  non-negative folds, paired CI `[-0.014611, 0.014416]`, and Hit@10 `-0.006666`.
- D − B, complete dense + structured channel beyond sparse deep: `+0.005641`,
  but only 3/5 non-negative folds and paired CI `[-0.020512, 0.033456]`.
- D − A, complete proposed interaction versus the current GBDT: `-0.031994`;
  all five fold deltas were negative.

This means the earlier apparent improvement was not evidence that deeper GBDT or
structured dense retrieval improves the current GBDT. The current Top-50 ranker is
both stronger and materially smaller.

## Nested query selection

The only query choices were the two predeclared interaction reserves:
`raw_plus_active` and `negation_safe_structured`. Selection maximized all-stage
sparse-or-dense Recall@200 on the outer-training complement, with
`raw_plus_active` as the fixed tie break. The resulting OOF choices were:

- folds 0 and 1: `raw_plus_active`;
- folds 2, 3, and 4: `negation_safe_structured`.

The globally selected query from the earlier `d8cf0542` run is treated as
exploratory, not validated, because it selected and evaluated on the same 150
sessions.

## Reliability and packaging

- Control sessions and score reproduced byte-for-value against
  `gbdt_reranker_v1.json`; stopping rounds were `[100, 4, 56, 98, 34]`.
- An independent D fold-0 refit selected the same 156 rounds and serialized
  byte-identically.
- Full deterministic replay and evaluator/replay adapter parity matched exactly.
- All four arms produced zero instrumented response failures.
- E5 ran offline from the pinned revision and an existing content-addressed index;
  no index build or external call occurred.
- Unique E5 model + index assets were `201.491924 MB`.
- Peak process memory was `5133.9375 MB`, correctly converted from Darwin's
  byte-valued `ru_maxrss`; this exceeds the frozen 4096 MB budget because all
  research training matrices coexist in the tournament process. It is a research
  harness peak, not a deployment-only measurement.
- Warm D latency p95 was `70.978727 ms`, but it was measured while other CPU-heavy
  prototypes were running. It is explicitly contention-affected. No isolated
  rerun is warranted because D already failed multiple score, fold, Hit, and
  scenario gates.

## Artifacts

- Manifest: `configs/experiments/gbdt_dense_interaction_v1.json`
- Complete report: `artifacts/reports/gbdt_dense_interaction_v1.json`
- Runtime arm: `ghostlab/retrieval/gbdt_dense.py`
- Reproduction runner: `scripts/run_gbdt_dense_interaction.py`
- Contract tests: `tests/test_gbdt_dense_interaction.py`

The experiment branch remains isolated. It does not modify the compiled champion,
the champion worktree, or any protected holdout material.
