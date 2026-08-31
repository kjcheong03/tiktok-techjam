# TechJam Track 4 — Adaptive Shopping Copilot

This branch implements a conversational product-search agent whose runtime follows
the six required Track 4 capabilities while GhostLab optimizes the implementation
inside that fixed workflow.

The governing rule is:

> TikTok fixes the required architectural capabilities; GhostLab optimizes how
> each capability works.

The workflow cannot be removed or reordered by a challenger. GhostLab may tune its
budgets and thresholds, replace one implementation with another implementation of the
same required capability, and add compatible optional techniques.

## Current branch and status

- Worktree: `techjam-adaptive-optimizer`
- Branch: `feat/adaptive-hybrid-1a-3b`
- Required architecture: implemented and operational
- Current active runtime: hash-bound GhostLab Champion (adaptive D + RRF)
- Adaptive candidate: manually adjudicated and explicitly activated
- One-time 550-session final selection: accessed once; organizer-private evaluation remains unseen
- Latest regression result: 525 passed, 1 skipped before final documentation synchronization

The legacy guarded champion remains a deployment fallback, not the matched control for
the new architecture. C (the fixed adaptive architecture) is the GhostLab promotion
control; D1-D3 denote the available one to three frozen finalists. A GhostLab result
never changes the active agent without an explicit activation decision.

## Challenge contract

The agent receives a supplied preference profile and one customer message at a time.
For each turn it may return up to ten catalog `parent_asin` values, ask one structured
clarification question, or do both. A session succeeds when the hidden target appears
in the scored Top 10 within ten turns.

The public development set contains 200 Buying, Browsing, Intent Override, and Boundary
sessions over a frozen 50,000-product Clothing, Shoes and Jewelry catalog. The organizer
holds the private final evaluation set.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality counts as a hit.

## Fixed 1A–3B architecture

```text
                  2B. Over-generality guidance
            controls retrieval cutoff and question action

                    3B. Adaptive orchestration
             controls route, ranker and question workflow
                                  |
                                  v
Customer message + supplied profile
                |
                v
        2A. State V2 dynamic state
   accumulate facts and handle overrides
                |
                v
      1A. Buying/Browsing selector
          |                    |
       Buying              Browsing
          |                    |
 Field BM25 precision    Diverse multi-view E5
      retrieval          semantic retrieval
          |                    |
          +---------+----------+
                    |
                    v
       1B. Merge keyword/category/vector candidates
                    |
                    v
           union-aware complete-pool ranking
                    |
                    v
      bounded local-LLM semantic activation decision
                    |
                    v
        conflict-safe profile-context influence
                    |
                    v
      validate Top 10 + recommendations/question
                    |
                    v
        atomically commit the selected action
                    |
                    v
       3A. Emit session/profile update for next turn
```

Category retrieval is an independent candidate source inside the 1B merge. It is not
a third top-level user-intent route.

### Required capabilities and adaptive freedom

| Section | Static requirement | GhostLab may optimize |
|---|---|---|
| 1A Routing | Both Buying and Browsing remain reachable using observable features | Specificity signals, confidence and abstention threshold |
| 1B Retrieval | Buying is BM25-primary; Browsing is diverse-E5-primary; keyword, category and vector evidence enter the bounded union | Retrieval depths, supporting budgets, fusion strategy and source weights |
| 1B Semantic ranking | A literal local-LLM capability and semantic activation decision remain after union ranking | Activation gate, prompt, depth, weight, timeout and fallback |
| 2A State | State V2 accumulates facts, exclusions, corrections, provenance and intent epochs | Parser confidence and bounded override scope |
| 2B Guidance | Over-generality triggers bounded retrieval and the highest-value unresolved legal question | Overload threshold, question-value margin and discovery horizon |
| 3A Runtime adaptation | Supplied/session profile context is conflict-safe; explicit current intent wins | Profile confidence, ambiguity gate and ranking influence |
| 3B Orchestration | Canonical order, fallback, reason codes, response validation and atomic commit remain fixed | Budgets and activation thresholds inside the fixed order |

`AdaptiveArchitectureAudit` validates this contract before a GhostLab trial can be
scored. A submission-eligible configuration has no `off` value for a required slot.

## Current selected implementations

- **State:** State V2 with typed constraints, exclusions, corrections, provenance,
  intent epochs, and intent-scoped shown-product history.
- **Router:** deterministic Buying/Browsing selection using only observable state.
- **Buying retrieval:** field-weighted FTS5/BM25 precision retrieval.
- **Browsing retrieval:** deterministic State V2 query views over pinned E5-small-v2,
  retrieving up to 400 per view and selecting a semantic Top 200.
- **Category retrieval:** independent catalog category candidates with provenance.
- **Merge:** keyword/category/vector weighted union; Buying weights `0.90/0.05/0.05`,
  Browsing weights `0.10/0.10/0.80`.
- **Union ranking:** hash-bound GBDT over the complete merged pool.
- **Semantic ranking:** pinned SmolLM2-1.7B-Instruct for Browsing, starting at depth
  `10` and weight `0.05`; deterministic skip for Buying and overload cutoff. The LLM
  may reorder only supplied catalog IDs.
- **Guidance:** recommend-and-ask under overload; ask-only deferral is not promoted.
- **Profile adaptation:** conflict-safe supplied/session profile influence with current
  weight `0.02`, confidence, provenance, intent epoch and conflicts.
- **Fallback:** complete precision path on component timeout, invalid score or failure.
- **Commit:** response normalization, deduplication, Top-10 cap and atomic selected-action
  history update.

MiniLM is retained only as a bounded invalid-score/model fallback. It is not described
as the required LLM.

## GhostLab integration

GhostLab remains the research and optimization engine, but it cannot delete required
architecture. The current technique catalog contains 88 records. In the adaptive
registry, 19 compulsory capabilities define the fixed workflow and 17 optional
techniques are eligible to race or combine. The remaining records are retained as
controls, research procedures, or explicitly unavailable historical bindings.

| Classification | Count | Meaning |
|---|---:|---|
| Compulsory | 19 | Always represented in the fixed workflow |
| Promotable | 17 | Runnable alternative/addition that may race and combine |
| Control-only | 25 | Diagnostic evidence; cannot replace a required capability |
| Research-only | 16 | Search, evaluation and evidence procedures |
| Unavailable | 11 | Preserved with a blocker and retest trigger |

The engine:

1. materializes the matched incumbent;
2. evaluates every valid standalone and compatible pair;
3. races candidates through F0/F1/F2 using paired session rewards;
4. conservatively prunes dominated structures while retaining exploration;
5. expands compatible survivors into higher-order combinations;
6. performs conditional local BOHB tuning around surviving structures; and
7. promotes only architecture-valid candidates with required fit evidence.

There is no six-technique champion limit. The current catalog produces 62 valid initial
control/single/pair candidates; higher-order search is globally capped at 500 candidates.
Fit-required historical rankers may be evaluated, but remain promotion-ineligible until
refitted against the new candidate pools with verified disjoint-fold receipts.

### Runnable and research technique inventory

The inventory below is exhaustive for techniques with an operational default binding.
Unavailable records remain in the machine-readable catalog with their blockers, but are
deliberately omitted here so this section cannot imply that they are runnable.

Anchor/control techniques: `guard.override_fallback`, `query.expansion_guard.v1`,
`ranking.constraint_gbdt`, `ranking.deep_dense_gbdt`, `ranking.mmr_early.v1`,
`ranking.neural_gbdt`, `ranking.pairwise_linear`, `routing.decision_list`,
`routing.joint_route.v1`, `routing.observable_stump`, `routing.route_table`,
`state.attribute_ontology.v1`.

Composable techniques: `filter.structured`, `fusion.rank_stack.v1`, `fusion.rrf`,
`fusion.sparse_first_union`, `fusion.weighted`, `policy.joint_observable.v1`,
`prior.profile`, `prior.quality`, `query.catalog_prf.v1`,
`query.coverage_adaptive_v2`, `query.structured`,
`question.adaptive_heuristic`, `question.candidate_eig.v1`, `question.fixed`,
`question.learned_linear`, `question.other_always`, `ranking.cross_encoder`,
`ranking.facet_diversity.v1`, `ranking.fixed_lexical`,
`ranking.fold_ensemble.v1`, `ranking.metadata_gbdt`,
`ranking.reward_lambdamart.v1`, `ranking.top10_residual_reranker.v2`,
`ranking.turn_aware_lambdamart.v1`,
`recommendation.correction_scoped_history`, `retrieval.e5`, `retrieval.minilm`,
`retrieval.sparse`, `state.baseline_v2`, `state.catalog_normalizer.v1`,
`state.compressed`, `state.confidence_gated_constraints.v1`, `state.current`,
`state.multi`, `state.raw_history`, `termination.reward_aware.v1`.

Research and optimizer techniques: `evaluation.grouped_splits`,
`evaluation.paired_statistics`, `evidence.decision_store`,
`research.counterfactual`, `research.counterfactual_expert.v2`,
`research.leakage_firewall`, `research.replay`, `search.bohb.v1`,
`search.crossover`, `search.evidence_allocator`, `search.expert_iteration.v1`,
`search.family_ucb`, `search.hyperband.v1`, `search.multifidelity_racing`,
`search.random_grid_beam`, `search.typed_patches`.

The older flat unified/autonomous campaign remains in the repository as a control and
historical research system. It does not define this branch's submission architecture.

## Repository setup

Supported Python versions are 3.10–3.13; 3.12 is recommended. macOS and Linux are
supported. Native Windows is not validated; use WSL2.

```bash
uv sync --all-extras --group dev
uv pip check
```

Download the released catalog and place it at `data/catalog.jsonl`, then verify its
published SHA-256 value. Fetch the pinned optional assets:

```bash
uv run python -m scripts.fetch_optional_assets e5
uv run python -m scripts.fetch_optional_assets minilm
uv run python -m scripts.fetch_optional_assets cross_encoder
uv run python -m scripts.fetch_optional_assets qwen_ranker
uv run python -m scripts.fetch_optional_assets smollm2_ranker
uv run python -m scripts.fetch_optional_assets qwen3_ranker
uv run python -m scripts.fetch_dense_index_asset
```

The last command downloads the versioned 50,000-product E5 and MiniLM embedding
indexes from a GitHub Release. It verifies the archive and every extracted file by
SHA-256, then validates the catalog hash, model revisions, row counts, dimensions and
dtypes before installation. To check an existing installation without downloading:

```bash
uv run python -m scripts.fetch_dense_index_asset --verify-only
```

For a private repository, the downloader uses `GH_TOKEN`, `GITHUB_TOKEN`, or the active
GitHub CLI login. No authentication is required after the repository is public.

Runtime loading is forced offline after assets are present. If the released dense-index
asset is unavailable, the content-addressed indexes can still be rebuilt locally from
the pinned models and matching catalog when the dense runtime is first initialized.

## Validation

Run the regression suite:

```bash
uv run ruff check .
uv run mypy ghostlab
uv run pytest -q
```

Run focused architecture and behavior checks:

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_adaptive_diversity.py
PYTHONPATH=. .venv/bin/python scripts/validate_adaptive_hybrid.py
```

Evaluate the adaptive runtime on the 200 public sessions:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid.py \
  --config configs/adaptive_hybrid_structural_smoke.json \
  --max-samples 20 \
  --output artifacts/reports/adaptive_hybrid_structural_e2e_smoke.json
```

### Replay the demonstrated session

Replay one development-partition session turn by turn and write the readable
console trace plus deterministic `demo_replay.json` and `demo_replay.md` files:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. .venv/bin/python scripts/replay_demo_session.py \
  --sample-id public_0001 \
  --config configs/adaptive_hybrid_1a_3b_v1.json \
  --catalog data/catalog.jsonl \
  --lineage-manifest data/splits/adaptive_hybrid_lineage_75_25_v1.json \
  --output-dir artifacts/demo_replay
```

The command is a demonstrated-session replay, not a reproduction of aggregate
metrics. It accepts repeatable `--dataset` arguments (defaulting to the three
project datasets), rejects holdout or unknown IDs, and fails clearly if the
requested configuration is unavailable. Use the baseline config while the final
fitted configuration is still being produced, then pass the frozen finalist path.

### Results dashboard

Visualize and compare existing or newly generated evaluation reports with the local
dependency-free dashboard:

```bash
uv run python dashboard/server.py
```

Open <http://127.0.0.1:8787/dashboard/>. The dashboard discovers compatible reports in
`artifacts/reports/` automatically and also accepts JSON files by import or drag-and-drop.
When the final fair-comparison report exists, the dashboard shows A, C and the available
one to three D finalists on one shared evaluation ground. A is the organizer reference,
C is the fixed adaptive control, and D1-D3 are frozen GhostLab challengers. A and C stay
pinned while the displayed D is selected from the dropdown; the final-selection result
identifies the selected D, or retains C. Only a D-versus-C decision can change the
champion.
See `dashboard/README.md` for supported report shapes and port configuration.

The protected F3/private data must never be used for routing, ranking, HPO, selection or
debugging.

## Development datasets

The complete 2,200-session corpus is partitioned by verified lineage group before any
final fitting or selection:

| Source | Samples | Buying | Browsing | Override | Boundary |
|---|---:|---:|---:|---:|---:|
| Official public development | 200 | 80 | 80 | 30 | 10 |
| Public-like synthetic | 1,000 | 400 | 400 | 150 | 50 |
| Independent-template synthetic | 1,000 | 400 | 400 | 150 | 50 |
| **Total** | **2,200** | **880** | **880** | **330** | **110** |

| Source | Development | One-time final selection |
|---|---:|---:|
| Official public development | 150 | 50 |
| Public-like synthetic | 750 | 250 |
| Independent-template synthetic | 750 | 250 |
| **Total** | **1,650** | **550** |

Each official row remains with its five public-like variants, and each independent
five-session family remains intact. The same lineage grouping is enforced in every
development outer/inner fold and in clustered racing statistics.

All 2,200 target products are distinct. Once the independent-template set is used for
training or model selection, it must no longer be claimed as independent validation.
All current examples are catalog-grounded clothing-domain sessions; this corpus does not
by itself establish generalization to unrelated catalog domains.

## One-command fit and optimization pipeline

The recommended entrypoint runs every dependent stage in the safe order:

```text
lineage reconstruction and 1,650-session development fit
  -> identical-pool dense diversity validation
  -> freeze the fixed SmolLM2 semantic control
  -> full public evaluation
  -> end-to-end validation
  -> resumable GhostLab F0/F1/F2 campaign
  -> freeze every development-eligible D configuration, capped at three
  -> matched full-development evaluation of the available D1-D3
  -> fair A/C plus selectable-finalist comparison
```

Inspect the exact commands and outputs without running anything:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py --show-plan
```

Run the entire pipeline on macOS while preventing sleep:

```bash
mkdir -p artifacts/logs

nohup caffeinate -dimsu env PYTHONPATH=. .venv/bin/python \
  scripts/run_adaptive_hybrid_pipeline.py \
  > artifacts/logs/adaptive_hybrid_pipeline.log 2>&1 &

echo $!
```

Monitor overall and stage-specific progress:

```bash
tail -f artifacts/logs/adaptive_hybrid_pipeline.log
tail -f artifacts/logs/adaptive_hybrid_pipeline/fit.log
```

The wrapper writes
`artifacts/campaigns/adaptive_hybrid_pipeline/checkpoint.json`. If interrupted, run the
same command again: completed stages with matching signatures and outputs are skipped,
while the GhostLab stage resumes its per-evaluation checkpoint. Use
`--through-stage validate` to finish model fitting and validation without starting the
long campaign. Use `--force-stage llm` (repeatable) only when intentionally rerunning a
completed stage.

This is one user-facing command, not one mixed statistical fit. Each stage remains
isolated so models are frozen before selection, failures are attributable, and runtime
labels cannot leak backward into training.

### Focused architecture-safe warm start

When the full combinatorial campaign is too slow, use the translated historical seed
and a focused successive-halving budget. The historical runtime is never executed:
compulsory 1A-3B stages remain fixed, while only compatible optional settings seed the
race. F0 evaluates the translated seed, all important standalone additions, selected
pairs and a global exploration reserve on 330 development sessions. Only gated
survivors reach F1 (825 sessions), one conditional HPO trial is allocated per surviving
structure, and F2 reserves enough capacity for three structural D finalists plus the
matched control/semantic calibration (all 1,650 development sessions). The 550-session
final-selection set remains untouched until finalists are
frozen.

Both focused warm-start and fresh exhaustive search use the same adaptive technique
registry. Each covers every currently promotable safe optional. In particular, the
historical lexical, metadata-GBDT, reward/turn-aware LambdaMART, fold ensemble and rank
stack options now run only as bounded secondary signals after the compulsory
source-aware union GBDT; none can replace it or change candidate membership. Warm start
reduces search breadth and evaluates the translated quality-plus-residual seed first;
it does not hide those adapted alternatives.

```bash
nohup caffeinate -dimsu env PYTHONPATH=. PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --from-stage campaign \
  --campaign-search-profile focused_warm_start \
  --campaign-warm-start \
    configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  > artifacts/logs/adaptive_hybrid_warm_start_pipeline.log 2>&1 &
```

The focused profile searches at most 36 structural candidates in total. It first covers
every safe add-one plus the warm seed and its ablations, then reserves up to an
eight-candidate beam for one evidence-guided higher-order expansion round. Six F1
survivors, five F2 evaluation slots and one HPO trial per surviving structure complete
the budget. These are evaluation budgets rather than a wall-clock kill switch. The
existing per-evaluation checkpoint is reused only when a candidate's full payload
matches, so completed compatible F0 work is retained safely.

The development output is `artifacts/reports/adaptive_hybrid_top3.json`. It contains up to
three F2-evaluated non-control challengers, their Hit@10/MRR/MTTC/technical score,
paired delta, latency, gates, materialized config hashes, and exact commands to evaluate,
validate, and manually activate each eligible finalist. The pipeline never activates a
champion automatically.

The final `compare` stage writes
`artifacts/reports/adaptive_system_comparison_1650.json` and the matching Markdown
table. It shows A (official stateless BM25), C (the fixed adaptive architecture), and
the available one to three D GhostLab finalists on the same 1,650 development sessions.
A is the organizer reference; only C and D participate in champion selection.

Before `compare`, the `finalists` stage re-evaluates each packaged D1-D3 configuration
on the exact same ordered 1,650-session development partition and shared evaluator
contract as A/C. GhostLab's fold/racing metrics remain recorded separately as
selection evidence; they are never substituted for the matched leaderboard metrics.

All compared systems use `ghostlab.research.replay.evaluate_shared` as the common research
harness. It freezes ordered session IDs, catalog, evaluator code, deterministic seed,
profile/response handling, Top-K and turn limits in a hash-bound evaluation contract.
The published `evaluator.local_evaluator.evaluate` remains unchanged as the reference,
with parity tests proving that the shared harness produces identical core outcomes.

Finalist numbers are development selection evidence. The one-time 550-session final
selection run evaluates frozen A, C, and every available D configuration (one to three)
on the same ordered sessions, catalog and evaluator contract. A is explanatory only.
Each D is independently compared against C, and the winner is chosen only by the frozen
tie-break order. The 550 is therefore a final selection set, not an unbiased holdout,
and its result must never be fed back into GhostLab for more tuning.

After reviewing the frozen finalist package, run the one-time comparison. Activation
accepts only the selected D from a matching report containing A, C, and every frozen
challenger:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_adaptive_holdout.py
```

Then use the frozen finalist's `activate_after_validation` command from the finalist
report. Roll back with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.activate_candidate --rollback
```

## Fit the new-architecture ranker on 1,650 development sessions

The trainer loads all three sources only to verify the immutable manifest, then replays
the actual State V2, router, retrieval and merge path for the 1,650 development
sessions. It trains:

1. the source-aware union GBDT for ordinary merged pools.

Overloaded turns deliberately use a deterministic bounded safe scorer and skip the
normal union GBDT and local LLM, so no redundant overload model is trained.

The target ID creates the offline label only after candidate generation. It is never
passed into runtime retrieval or ranker features. Only target-containing pools can form
supervised LambdaMART groups, so receipts report both the 1,650 replayed sessions and the
exact eligible ranking-session count.

The semantic ranker is an identity step during pre-semantic pool collection. E5, Qwen,
BM25, routing rules and profile logic are not fine-tuned by this command.

Preflight without fitting:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_adaptive_hybrid.py --plan-only
```

Start fitting on macOS:

```bash
mkdir -p artifacts/logs

nohup caffeinate -dimsu env PYTHONPATH=. .venv/bin/python \
  scripts/train_adaptive_hybrid.py \
  > artifacts/logs/adaptive_hybrid_training_1650.log 2>&1 &

echo $!
```

Monitor it:

```bash
tail -f artifacts/logs/adaptive_hybrid_training_1650.log
```

The first detailed line appears only after candidate-pool collection. A typical Apple
Silicon run is expected to take roughly 40–90 minutes and several GB of memory.

The repository contains implementation and smoke evidence, not a completed final
1,650-session fit. Do not treat the paths below as selected assets until the trainer
finishes and their receipts validate.

Training emits:

```text
configs/adaptive_hybrid_1a_3b_1650_final_v1.json
configs/splits/adaptive_1650_group_nested_v1.json
artifacts/models/adaptive_union_gbdt_1650_final_v1.json
artifacts/models/adaptive_union_gbdt_1650_final_v1.fit_receipt.json
artifacts/reports/adaptive_hybrid_training_1650_final_v1.json
```

The union model is bound into the output configuration only if it passes out-of-fold
Hit@10 non-regression, strict MRR improvement, and protected-slice gates. Rejected
evidence is retained in the training report.

The one-command pipeline no longer reopens model-family research. After the structural
fit succeeds it freezes the selected SmolLM2 control reproducibly:

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_adaptive_semantic_control.py
```

The command verifies the pinned SmolLM2 asset and emits C at Browsing-only depth `10`,
weight `0.05`, with MiniLM retained only as the failure fallback. The earlier four-model
comparison remains an evidence artifact, not a stage that is rerun during final training.

## Run the architecture-safe GhostLab campaign

Do not start this campaign until the 1,650-development fit finishes and the output config,
models and receipts are validated.

Architecture-only plan:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json \
  --plan-only \
  --output artifacts/reports/adaptive_hybrid_campaign_plan_1650.json
```

Full campaign:

```bash
mkdir -p artifacts/logs

nohup caffeinate -dimsu env PYTHONPATH=. .venv/bin/python \
  scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json \
  --dataset data/public_set.jsonl \
  --dataset data/synthetic_1000_public_like.jsonl \
  --dataset data/independent_template_1000.jsonl \
  --candidate-limit 500 \
  --beam-width 24 \
  --higher-order-rounds 8 \
  --f1-candidates 24 \
  --f2-candidates 6 \
  --hpo-trials-per-structure 2 \
  --output artifacts/reports/adaptive_hybrid_campaign_1650.json \
  > artifacts/logs/adaptive_hybrid_campaign_1650.log 2>&1 &

echo $!
```

Every evaluation start and finish is logged with fidelity, sample count, score and wall
time.

### Default fidelity allocation

The progressive ordering is deterministic and proportional within every dataset source
and scenario:

| Phase | Public | Public-like | Independent-template | Total per candidate |
|---|---:|---:|---:|---:|
| F0 | 30 | 150 | 150 | 330 |
| F1 | 75 | 375 | 375 | 825 |
| F2 | 150 | 750 | 750 | 1,650 |

Each fidelity prefix is deterministically balanced across source and scenario; exact
counts are written to the campaign plan and checkpoint.

F0 starts with 62 candidates and may expand up to the 500-candidate cap. F1 retains at
most 24 roots and adds two local HPO trials per non-control root, for at most 70 F1
evaluations. F2 evaluates at most six finalists.

This uses the entire development partition for GhostLab fitting and finalist selection
without spending F2 evaluation on every weak F0 candidate. The 550-session final
selection set is excluded from every development fidelity.

### Racing non-regression gates

The runner constructs balanced source/scenario prefixes and applies grouped promotion
gates in addition to the combined paired session reward:

- official-public performance must not materially regress;
- Buying, Browsing and Intent Override must not materially regress;
- Hit@10 and MRR may not regress against matched C;
- confirmed output-constraint violations must remain zero;
- latency must remain within the predefined bound unless paired quality or an actual
  Top-10 rescue justifies the extra cost;
- sparse F0 Boundary evidence must produce `HOLD_MORE_DATA`, not permanent rejection;
- one to three eligible challengers are frozen from F2 over all 1,650 development sessions;
- frozen A, C, and every available D finalist (one to three) may access the 550-session final
  selection set once; each D is gated against C before immutable tie-breaking.

The exhaustive default campaign is expected to take hours on one Mac. Retrieval/LLM
caching and lower early candidate caps are valid engineering optimizations only when they
preserve paired behavior and the fixed architecture.

The semantic lane is deliberately small: F0 compares weights `0.05`, `0.10`, `0.15`
and `0.20` at depth `10`; rejected weights are pruned. F1 evaluates depth `20` only for
the selected surviving weight. Depth `20` is retained only with a Hit@10/MRR improvement
or a measured Top-10 rescue that also passes latency and route/scenario gates. Depths
`30` and `50`, new model families, pairwise ranking and listwise ranking are outside the
final campaign.

## Profile awareness

The runtime is profile-aware in two ways:

1. `reset(...)` receives supplied profile tags and creates bounded ranking context.
2. each committed turn emits a caller-persistable `ProfileUpdate` containing values,
   confidence, provenance, intent epoch and conflicts.

Profile influence activates only when the request remains ambiguous and explicit intent
is insufficient. Explicit exclusions suppress conflicting profile terms, and current
session intent always overrides profile history. The source-aware GBDT schema includes a
bounded profile-match feature, but activation remains separately gated and tunable; the
conflict-safe runtime stage is authoritative.

## Selected public evidence

| Agent | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Guarded champion | 0.980000 | 0.774839 | 2.280000 | 0.896852 |
| State V2 precision control | 0.990000 | 0.746895 | 2.185000 | 0.895369 |
| Historical Adaptive Hybrid, Browsing-only Qwen | 0.985000 | 0.578286 | 2.045000 | 0.845086 |

The adaptive candidate improves Hit@10 and MTTC relative to the champion but loses MRR.
It is the architecture-complete optimization base, not the active winner.

The historical semantic activation study used the same 200 public sessions:

| Policy | Qwen activations / 406 turns | TechnicalScore | Decision |
|---|---:|---:|---|
| Never | 0 | 0.843037 | Ineligible control |
| Always | 406 | 0.830356 | Rejected |
| Broad semantic gate | 239 | 0.832516 | Rejected |
| Browsing after overload resolves | 0 | 0.843037 | Decorative; rejected |
| Browsing route only | 96 | 0.845086 | Selected inside adaptive runtime |

These are public development results, not guarantees of private-set performance.

## Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."},
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

Only the coordinator may atomically commit returned recommendations and the question
actually sent to the user.

## Important files

| Path | Purpose |
|---|---|
| `configs/adaptive_hybrid_1a_3b_v1.json` | Implemented architecture baseline |
| `configs/adaptive_hybrid_structural_smoke.json` | Validation-only 200-session fitted smoke config |
| `configs/adaptive_hybrid_1a_3b_1650_final_v1.json` | Development-only trainer output |
| `configs/techniques/catalog_v2.json` | Complete technique inventory |
| `ghostlab/runtime/adaptive_hybrid.py` | Fixed 1A–3B coordinator |
| `ghostlab/runtime/adaptive_config.py` | Typed required configuration contract |
| `ghostlab/runtime/adaptive_components.py` | Router, guidance, rankers and profile adapters |
| `ghostlab/optimization/adaptive_hybrid.py` | Architecture audit and safe binding |
| `ghostlab/optimization/adaptive_techniques.py` | Technique classification/binding |
| `ghostlab/optimization/adaptive_campaign.py` | F0/F1/F2 race engine |
| `ghostlab/training/adaptive_hybrid.py` | Runtime-pool collection and ranking data |
| `ghostlab/training/adaptive_datasets.py` | Multi-source loading and balanced folds |
| `ghostlab/training/adaptive_lineage.py` | Cross-source lineage reconstruction and group-safe partitions |
| `scripts/run_adaptive_hybrid_pipeline.py` | One-command checkpointed fit/selection/validation/campaign wrapper |
| `scripts/evaluate_adaptive_reference_baselines.py` | Development evaluation of A; optional research-only B audit |
| `scripts/build_adaptive_system_comparison.py` | Unified A/C and optional D1-D3 report |
| `scripts/train_adaptive_hybrid.py` | Development-only source-aware union ranker trainer |
| `scripts/run_adaptive_hybrid_campaign.py` | Architecture-safe GhostLab runner |
| `scripts/evaluate_adaptive_holdout.py` | Guarded one-time A/C plus one-to-three-D final selection |
| `docs/adaptive_hybrid_1a_3b_implementation_process.md` | Detailed process |
| `docs/adaptive_hybrid_1a_3b_implementation_report.md` | Evidence and decisions |

## Safety, reproducibility and activation

- Scenario labels, target IDs, evaluator outcomes, future answers and hidden rewards are
  forbidden runtime features.
- Target IDs are permitted only as offline labels after candidate generation.
- Local model revisions and asset hashes are pinned; offline loading is enforced.
- Fit-required candidates cannot be promoted without disjoint-fold receipts.
- Campaign results do not write `configs/active_candidate.json`.
- Activation remains a separate, explicit human-reviewed operation.
- Never commit API credentials, model caches, live logs or protected evaluator data.

## Further documentation

- [Adaptive implementation process](docs/adaptive_hybrid_1a_3b_implementation_process.md)
- [Adaptive implementation report](docs/adaptive_hybrid_1a_3b_implementation_report.md)
- [Competition specification](docs/competition_specification.md)
- [Agent API contract](docs/agent_api_contract.json)
- [Submission rules](docs/submission_rules.md)
- [Data attribution](DATA_ATTRIBUTION.md)

The catalog and development data are derived from Amazon Reviews 2023 by McAuley Lab,
UCSD. Review `DATA_ATTRIBUTION.md` before redistribution.
