# Wave 2 ranking track: implementation and OOF decision

Status: complete standalone ranking-track validation; all new techniques remain off
by default and the selected Wave 1 champion is unchanged.

## Scope and firewall

This track implements the Wave 2 competition-aligned ranking and rank-ensemble
families. It uses only the 150 IDs in `configs/splits/nested_v1.json`. The protected
50-session F3 split was not imported, enumerated, read, or scored. Targets, observed
turns, and terminal outcomes are fold-local training labels only; runtime inference
uses the existing observable metadata feature matrix.

The comparison is a matched ranking-subsystem experiment on the frozen Wave 1
raw-history/fixed-question trajectory. It is not a direct challenger to the complete
`0.878963` guarded policy until integrated behind that policy in the Wave 2
integration worktree.

## Implemented techniques and exact locations

| ID | Switch | Implementation | Current decision |
|---|---|---|---|
| `ranking.reward_lambdamart.v1` | `reranker=reward_lambdamart` | `ghostlab/retrieval/reward_lambdamart.py` | Available, parked after negative standalone OOF result. |
| `ranking.turn_aware_lambdamart.v1` | `reranker=turn_aware_lambdamart` | `ghostlab/retrieval/reward_lambdamart.py` | Available, parked after negative standalone OOF result. |
| `ranking.fold_ensemble.v1` | `reranker=rank_ensemble` | `ghostlab/retrieval/ensemble.py` | Available, interaction reserve; positive but uncertain OOF delta. |
| `fusion.rank_stack.v1` | `reranker=rank_ensemble`, `ensemble_mode=fold_local_stack` | `ghostlab/retrieval/ensemble.py` | Available, interaction reserve; positive but uncertain OOF delta. |
| `ranking.uniform_pairwise.v1` | `reranker=uniform_pairwise` | `ghostlab/retrieval/reward_lambdamart.py` | Available control, parked. |
| `ranking.pointwise_gbdt.v1` | `reranker=pointwise_gbdt` | `ghostlab/retrieval/reward_lambdamart.py` | Available control, parked. |

The authoritative technique bindings are in
`configs/techniques/w2_ranking_v1.json`. Experiment contracts are in
`configs/experiments/w2_reward_lambdamart_v1.json` and
`configs/experiments/w2_rank_ensemble_v1.json`. All switches default to off.

## Exact reward delta

`ghostlab/evaluation/reward_deltas.py` implements the organizer reward at a first
hit on rank `r` and turn `t`:

```text
0.50 + 0.30 / r + 0.20 * clip((11 - t) / 10, 0, 1),  r <= 10
0,                                                       r > 10
```

The Lambda weight for swapping the target between two predicted ranks is the
absolute difference between those two rewards. Consequently:

- swaps within Top-10 capture reciprocal-rank movement;
- swaps across rank 10 capture inclusion, reciprocal rank, and efficiency;
- early rank-10 crossings are worth more than late crossings;
- pairs entirely below rank 10 have zero immediate terminal value;
- no future outcome is guessed or exposed as a runtime feature.

The mean of this per-session reward is exactly the published technical score on
valid turns because all three organizer terms are linear over the session sample.

## Validation design

- Five grouped outer folds stitch exactly 150 OOF session results per candidate.
- Each outer fold uses another frozen fold for inner round selection.
- Models are refit on the complete outer-training side at the selected round count.
- The historical NDCG, uniform-pairwise, pointwise, reward, and turn-aware arms use
  identical metadata features, candidate depth 50, tree depth 3, seven leaves,
  learning rate `0.03`, maximum 160 rounds, and minimum leaf size 40.
- The exact historical NDCG control reproduces `0.861417`.
- Equal ensemble weights are fixed controls. Stack weights are selected only on
  each outer fold's inner-validation side using a deterministic `0.25` simplex
  grid. Outer labels are never visible to the weight selector.
- Every ensemble reranks the same candidate head. There is no candidate-depth or
  union advantage.

## Corrected OOF results

| Candidate | Technical score | Hit@10 | MRR | MTTC | Delta vs NDCG |
|---|---:|---:|---:|---:|---:|
| Equal standardized-score ensemble | **0.865533** | **0.980000** | 0.677553 | 2.386667 | +0.004116 |
| Fold-local rank stack | 0.864371 | **0.980000** | 0.670571 | **2.340000** | +0.002955 |
| Exact NDCG@10 control | 0.861417 | 0.973333 | **0.680278** | 2.466667 | — |
| Uniform pairwise | 0.858373 | 0.966667 | 0.680799 | 2.460000 | -0.003044 |
| Equal mean-rank ensemble | 0.856640 | 0.973333 | 0.661688 | 2.426667 | -0.004777 |
| Turn-aware reward LambdaMART | 0.852850 | 0.966667 | 0.666389 | 2.520000 | -0.008567 |
| Reward LambdaMART | 0.848198 | 0.973333 | 0.631328 | 2.393333 | -0.013218 |
| Pointwise GBDT | 0.843026 | 0.940000 | 0.700310 | 2.853333 | -0.018390 |

The leading ensemble's paired 95% bootstrap interval is
`[-0.008292, 0.018221]`; its 150-session gain is therefore not decisive. It adds
one session hit but loses two net rank-1 sessions (seven gained and nine lost).
It remains an interaction-reserve technique, not a new champion.

## Preliminary screen versus corrected evidence

An initial engineering screen accidentally used a 120-round/15-patience ceiling
for every arm. It produced NDCG `0.858099`, uniform pairwise `0.864255`, fold-local
stack `0.862763`, reward `0.835885`, and turn-aware reward `0.850536`. Those values
are retained here only as diagnostic history and must not be used for decisions.

The manifest was corrected to the historical 160-round/20-patience capacity and all
arms were rerun from scratch. Only the corrected table above is promotion evidence.
This correction reproduced the historical NDCG score exactly and changed the
leader, demonstrating why capacity matching mattered.

## Runtime and assets

The corrected run took `631.068` seconds. On 100 cached Top-50 query heads:

| Runtime path | Milliseconds/query |
|---|---:|
| NDCG single model | 1.140440 |
| Reward single model | 1.189903 |
| Equal fold/model ensemble | 2.058666 |
| Fold-local rank stack | 2.053555 |

The complete compact development-refit asset set is approximately 211 KB before
removing redundant sidecar fields. Individual tree assets preserve the existing
JSON inference format byte-for-byte; objective/technique metadata lives in the
Wave 2 report and experiment manifests. Ensemble manifests contain only safe
project-relative model paths, aggregation mode, and bounded weights.

## Decision and integration guidance

1. Do not replace the `0.861417` NDCG fallback or the complete `0.878963` champion
   from this standalone evidence.
2. Integrate equal standardized-score ensemble and fold-local stacking behind
   disabled switches and test them with the complete guarded champion.
3. Keep both reward objectives available for interactions with improved retrieval,
   candidate-EIG questioning, or a different candidate distribution. Their current
   negative standalone results are not proof that every future interaction is bad.
4. Retune ensemble weights inside inner folds for every materially different
   combination. Do not reuse weights optimized for this candidate distribution.
5. Preserve equal weights as the fixed control; a tuned stack must beat it at the
   same depth and runtime accounting.
6. F3 remains sealed until one complete cross-family candidate is frozen.

Machine-readable evidence is in `artifacts/reports/w2_ranking_v1.json`; compact
assets and SHA-256 metadata are in `artifacts/models/w2_ranking_v1/` and the report.
