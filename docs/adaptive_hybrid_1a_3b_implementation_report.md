# Adaptive Hybrid 1A-3B Structural Implementation Report

## Outcome

The fixed Track 4 architecture and its GhostLab integration are implemented and pass
implementation-level validation. The architecture cannot be removed or reordered by a
challenger; GhostLab can tune required implementations and race compatible additions.

The final 1,650-development fit, bounded LLM family comparison, and large GhostLab
campaign are deliberately deferred. They are optimization/model-selection work, not
missing runtime implementation. The 550-session lineage-safe holdout remains untouched,
the previous active candidate was not modified, and F3 was not accessed.

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
| LLM configuration lacked a controlled family study | Added pinned SmolLM2 and Qwen3 assets, generic local-causal backend, Qwen depth study and same-pipeline comparison script | Qwen, SmolLM2 and Qwen3 all load offline and return finite relevance scores; full comparison is deferred |
| GhostLab could reject required architecture | Added compulsory/adaptive classifications, preflight architecture audit, fit receipts, resumable checkpoints and source/scenario/constraint gates | Plan contains all 19 compulsory capabilities in every one of 28 initial candidates |
| Final comparison could drift after development | Finalist validation uses the shared 1,650-session evaluator; packaging freezes D, control C, A/B definitions, gates and lineage hashes before holdout | Tests reject changed control paths/hashes and mismatched evaluation contracts/session order before any holdout receipt is written |

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

- Full regression suite: 457 passed, 1 skipped.
- Repository-wide Ruff: clean.
- Focused mypy over the changed trainer, LLM comparison and validator: clean.
- `git diff --check`: clean.
- Active-candidate pointer: unchanged/absent.
- Protected F3: not accessed.

## Deliberately deferred work

1. Complete the 1,650-development five-fold structural fit using lineage-preserving
   outer and inner folds. Historical 2,200-session artifacts are not eligible because
   they consumed what is now the untouched holdout. No incomplete model is bound or
   claimed.
2. Run the bounded local-LLM comparison after the fitted candidate pools are frozen:
   Qwen Top 10/20/30, then SmolLM2 and Qwen3 at the winning depth.
3. Use the emitted selected LLM config for a full public run and validation.
4. Run the resumable F0/F1/F2 GhostLab campaign with grouped source/scenario gates.
5. Package up to three development finalists, freeze exactly one D plus control C and
   every holdout dependency, then run the 550-session holdout once.
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
