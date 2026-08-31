# Adaptive Hybrid 1A-3B Structural Implementation Report

## Outcome

The fixed Track 4 architecture and its GhostLab integration are implemented and pass
implementation-level validation. The architecture cannot be removed or reordered by a
challenger; GhostLab can tune required implementations and race compatible additions.

The final 1,650-development structural fit has completed. The first bounded LLM family
run was correctly rejected because no selected turn reached semantic execution; the
activation and paired-pool defects found by that run are now corrected and the model
comparison must be rerun. The large GhostLab campaign remains deferred. The 550-session
final selection set remains unaccessed, the previous active candidate was not modified,
and F3 was not accessed.

## Problem-statement coverage

| Requirement | Implemented contract | Adaptive freedom |
|---|---|---|
| 1A Dual-track routing | Observable Buying/Browsing selector; hard constraints remain authoritative on both routes | Specificity features, thresholds, confidence and abstention |
| 1B Pipeline base | Buying is BM25-primary; Browsing is diverse-E5-primary; keyword, independent category and vector evidence enter a bounded in-memory union; literal local-LLM activation decision follows union ranking | Retrieval depths, support weights, union model, LLM model/depth/weight/activation |
| 2A Dynamic state | One immutable State V2 turn context carries accumulated facts, exclusions, provenance, corrections and intent epoch | Parser confidence and bounded override behavior |
| 2B Proactive guidance | Cheap pre-E5 preview can cap expansion; recommendations plus the highest-value legal unresolved question are returned | Cutoff thresholds, EIG margin, candidate depth and discovery horizon |
| 3A Runtime adaptation | Conflict-safe profile context plus a separate profile update containing attributes, values, confidence, provenance, epoch and conflicts | Profile query view, union feature and question suppression are gated/tunable |
| 3B Adaptive orchestration | Canonical stage order, failure fallback, reason codes, response validation and atomic commit are fixed | Budgets and optional compatible rankers inside fixed slots |

The adaptive question slot is compulsory. The current implementation uses candidate EIG;
GhostLab may tune or replace it only with another highest-value legal-question policy.
Ask-only deferral is not the default architecture.

## Resolved structural gaps

| Gap | Resolution | Validation evidence |
|---|---|---|
| Observable router was too narrow | Separated current-query evidence from discounted historical state; added category-only, attribute-coverage, bounded query-length, provenance, exclusion and correction evidence without labels or evaluator data | Counterfactual tests hold accumulated state constant while current query specificity changes Buying/Browsing; standard route and abstention tests remain active |
| Constraints could lose authority on Browsing | Added route-independent confirmed-match/confirmed-violation/unknown/soft decisions before and after ranking; complete literal tokens or approved semantic equivalents are required | Adversarial tests cover shared-token false positives, incomplete approved-equivalent matches, exclusions, budget violations and missing metadata |
| Diverse dense was only multi-view max relevance | Added max-relevance, view-balanced and embedding-MMR selectors with per-view evidence and pinned E5 product embeddings | Full 80-session public Browsing diagnostic reports recall, category coverage and pairwise similarity |
| Overload cutoff occurred after full dense work | Added cheap keyword/category preview, overload-specific reduced E5 budgets and a bounded safe merge/ranker branch that skips normal union, optional full rerankers and local-LLM execution | Behavior-level trace tests verify reduced depth, safe-branch execution and prohibited-stage non-execution |
| GBDT could not learn source importance | Added normalized BM25/category/dense ranks and scores, missingness, source membership/count, reciprocal rank, agreement, constraint and profile features | Matrix parity/missingness tests and real structural smoke fit pass |
| Buying precision could be overturned | Added sparse-dominant residual mode with bounded learned influence and route-independent authority | Adversarial learned-ranker test cannot overturn the protected sparse head or hard constraints |
| Turn context was only partly immutable | Router, retrieval, guidance, ranking, clarification and profile adaptation consume one frozen per-turn context | Frozen-dataclass and atomic commit tests pass |
| Profile adaptation was too weak/decorative | Added optional profile query view, union feature and question suppression plus explicit update attributes | Conflict, query-view, feature and question tests pass; explicit current intent remains dominant |
| LLM configuration lacked a controlled family study | Added pinned manifests for Qwen2.5, Qwen3, Gemma 3 1B IT and SmolLM2-1.7B, a generic local-causal backend, symmetric depth/weight grids, isolated workers and MiniLM control/fallback | Selection now requires verified assets and an exact 45-trial attempt ledger; failures remain recorded but ineligible, partial downloads are rejected, and Qwen3 uses explicit non-thinking chat rendering for immediate yes/no scoring; the full comparison remains deferred |
| Clarified exploration collapsed into Buying | Persisted explicit exploratory intent within its correction epoch; after an overload clarification, exploratory sessions remain Browsing while hard constraints stay route-independent | The same 60-session development replay now reaches 75 semantic calls across 58 sessions; all five paired backends execute without fallback |
| Paired-pool evidence included random evaluator session IDs and pre-union order | Hashes now exclude runtime UUIDs and use the first actual ordered pre-semantic pool per session | Two independent 60-session replays produced the identical pool hash and 58 paired semantic cases |
| Descriptive negation became a hard exclusion | Restricted exclusion extraction so phrases such as `not including buckle` and `not quite right yet` are not product prohibitions while genuine exclusions such as `not formal` remain authoritative | Confirmed target removals on the paired smoke dropped from 8 to 0; focused adversarial tests and the complete suite pass |
| GhostLab could reject required architecture | Added compulsory/adaptive classifications, preflight architecture audit, fit receipts, resumable checkpoints and source/scenario/constraint gates | Plan contains all 19 compulsory capabilities in every one of 28 initial candidates |
| Final comparison could drift after development | Finalist validation uses the shared 1,650-session evaluator; packaging freezes the available D1-D3, C, A, gates, tie-breaks and lineage hashes before final selection | Tests reject changed paths/hashes and mismatched evaluation contracts/session order before any access receipt is written |

## Post-fix local-LLM activation smoke

A bounded one-setting smoke used the same 60 lineage-safe development sessions,
pre-semantic pools, depth 10 and weight 0.35. It is protocol validation, not the full
45-trial model selection.

| Backend | Technical score | Semantic calls | Rescues/demotions | Fallback | p95 latency |
|---|---:|---:|---:|---:|---:|
| No-op semantic control | 0.880895 | 75 | 0 / 0 | 0 | n/a |
| Qwen2.5-0.5B | 0.862853 | 75 | 0 / 0 | 0 | 209.5 ms |
| Qwen3-0.6B | 0.858651 | 75 | 0 / 0 | 0 | 398.5 ms |
| Gemma 3 1B IT | 0.835934 | 75 | 0 / 0 | 0 | 808.1 ms |
| SmolLM2-1.7B | 0.870437 | 75 | 0 / 0 | 0 | 609.7 ms |
| MiniLM control | 0.882234 | 75 | 0 / 0 | 0 | 63.1 ms |

The earlier missing-output error is resolved: all trials completed, candidate/session
pairing passed, selection was valid and a smoke config was emitted. However, no genuine
LLM beat the no-op control at this one setting. The smoke-selected SmolLM2 config must
not be promoted; the full depth/weight grid must test whether a smaller semantic weight
or different depth creates a net benefit.

## Diverse-dense result

The diagnostic used all 80 official public Browsing sessions. It demonstrates measured
candidate behavior, not general cross-category relevance.

| Selector | Recall@50 | Recall@100 | Recall@200 | Mean categories@50 | Mean pair similarity@50 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Multi-view max relevance | 0.3000 | 0.4000 | 0.5875 | 8.675 | 0.8763 | Default |
| View balanced | 0.3000 | 0.4000 | 0.5875 | 8.675 | 0.8763 | No measurable change; retain as challenger |
| Embedding MMR | 0.2250 | 0.2875 | 0.4250 | 13.275 | 0.8558 | More diverse but fails recall gate; rejected as default |

Limited cross-category candidate reach had previously been demonstrated in three
handcrafted scenarios. Neither that check nor query-view existence proves general
cross-category quality. The new validator therefore reports actual coverage,
concentration and recall.

## Source-aware ranker smoke result

The 200-session structural smoke replay exercised the real State V2, router, retrieval,
constraint authority and union path. It produced 2,000 candidate pools and 503,301
candidate rows. Nested out-of-fold results were:

| Ordering | Hit@10 | MRR |
|---|---:|---:|
| Existing merged order | 0.841387 | 0.604244 |
| Source-aware union GBDT | 0.888005 | 0.728429 |
| Delta | +0.046617 | +0.124185 |

This is strong historical smoke evidence that the structural feature change is useful.
It is not the final 1,650-development selection result.

## End-to-end structural smoke

Twenty deterministic public-prefix sessions used the 200-session smoke-fit config:

| Metric | Result |
|---|---:|
| Hit@10 | 0.950000 |
| MRR | 0.461250 |
| MTTC | 2.950000 |
| TechnicalScore | 0.774375 |
| Turns traced | 58 |
| Buying/Browsing turns | 50 / 8 |
| Overload turns | 8 |
| Literal-LLM activations/skips | 8 / 50 |
| LLM ordering changes | 7 |
| Runtime fallbacks | 0 |

This bounded result proves the complete pipeline executes. It is not comparable to the
200-session champion score because the sample scope differs.

The dedicated validator passed every required implementation check: config/schema,
model hash and fit receipt, deterministic response and trace, response contract, both
routes, three-source evidence on Buying and overloaded Browsing, dense views, real
cutoff, literal LLM activation and change, conflict-safe profile behavior, profile
update, intent epoch/history scoping, offline mode, zero fallback, untouched activation,
and no F3 access.

## GhostLab integration status

The catalog has 88 records. The adaptive registry classifies 19 as compulsory, 17 as
promotable, 25 as control-only, 16 as research-only and 11 as unavailable. The current
plan has 28 exhaustive control/standalone/compatible-pair candidates under a limit of
30. There is no six-technique champion cap, and higher-order compatible expansion and
conditional HPO remain supported.

Plan construction succeeds, and bounded live smoke scoring successfully evaluated the
structural control and multiple challengers with checkpoint writes. The exhaustive race
was intentionally stopped because each candidate reloads local models and the user
deferred optimization. No partial campaign is presented as a completed result.

## Code-quality validation

- Full regression suite: 475 passed, 1 skipped.
- Repository-wide Ruff: clean.
- Focused mypy over the changed trainer, LLM comparison and validator: clean.
- `git diff --check`: clean.
- Active-candidate pointer: unchanged/absent.
- Protected F3: not accessed.

## Deliberately deferred work

1. Retain the completed 1,650-development structural fit and its lineage-preserving
   outer/inner-fold receipts. Historical 2,200-session artifacts remain ineligible
   because they consumed what is now the one-time final selection set.
2. Rerun the symmetric bounded local-LLM comparison after the activation and paired-pool
   corrections. The fitted candidate pipeline remains frozen: Qwen2.5, Qwen3,
   Gemma 3 1B IT and SmolLM2-1.7B all receive the same Top-10/20/30 depth and
   weight grid; MiniLM remains the non-LLM control/fallback.
3. Use the emitted selected LLM config for a full public run and validation.
4. Run the resumable F0/F1/F2 GhostLab campaign with grouped source/scenario gates.
5. Package every eligible development finalist up to three, freeze D1-D3 as available plus A/C,
   gates and tie-breaks, then run the 550-session final selection once.
6. Review the final champion manually; do not auto-write the active-candidate pointer.

## Reproduction sequence

```bash
# Inspect the complete dependency plan without executing it.
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py --show-plan

# Run or resume fit -> LLM selection -> evaluation -> validation -> campaign.
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py

# Optional: stop after final validation and defer the long GhostLab campaign.
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --through-stage validate
```

The constituent scripts remain directly runnable for isolated diagnosis, but normal
operation should use the checkpointed wrapper so dependency order cannot be skipped.

## Machine-readable evidence

- `artifacts/reports/adaptive_pre_structural_baseline_manifest.json`
- `artifacts/reports/adaptive_dense_diversity_v2.json`
- `artifacts/reports/adaptive_hybrid_structural_training_smoke.json`
- `artifacts/reports/adaptive_hybrid_structural_e2e_smoke.json`
- `artifacts/reports/adaptive_hybrid_structural_smoke_validation.json`
- `artifacts/reports/adaptive_hybrid_campaign_structural_smoke_plan.json`
