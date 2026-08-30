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
- Default competition-facing runtime: frozen guarded champion
- Adaptive candidate: architecture-complete but not automatically activated
- Protected F3/private holdout: not accessed
- Latest regression result: 431 passed, 1 skipped; Ruff and focused mypy checks clean

The guarded champion remains the comparison control because the current adaptive
candidate trails it on public MRR. A GhostLab result never changes the active agent
automatically.

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
- **Semantic ranking:** pinned Qwen2.5-0.5B-Instruct for Browsing; deterministic skip for
  high-confidence Buying. Qwen may reorder only supplied catalog IDs.
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
uv run python -m scripts.fetch_optional_assets cross_encoder
uv run python -m scripts.fetch_optional_assets qwen_ranker
uv run python -m scripts.fetch_optional_assets smollm2_ranker
uv run python -m scripts.fetch_optional_assets qwen3_ranker
```

Runtime loading is forced offline after assets are present.

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

The protected F3/private data must never be used for routing, ranking, HPO, selection or
debugging.

## Development datasets

The 2,200-session fitting corpus contains:

| Source | Samples | Buying | Browsing | Override | Boundary |
|---|---:|---:|---:|---:|---:|
| Official public development | 200 | 80 | 80 | 30 | 10 |
| Public-like synthetic | 1,000 | 400 | 400 | 150 | 50 |
| Independent-template synthetic | 1,000 | 400 | 400 | 150 | 50 |
| **Total** | **2,200** | **880** | **880** | **330** | **110** |

All 2,200 target products are distinct. Once the independent-template set is used for
training or model selection, it must no longer be claimed as independent validation.
All current examples are catalog-grounded clothing-domain sessions; this corpus does not
by itself establish generalization to unrelated catalog domains.

## Fit the new-architecture rankers on 2,200 sessions

The trainer replays the actual State V2, router, retrieval and merge path for all 2,200
sessions. It then trains:

1. the union GBDT for ordinary merged pools; and
2. the Browsing-safe GBDT for overloaded diverse-dense pools.

The target ID creates the offline label only after candidate generation. It is never
passed into runtime retrieval or ranker features. Only target-containing pools can form
supervised LambdaMART groups, so receipts report both the 2,200 replayed sessions and the
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
  > artifacts/logs/adaptive_hybrid_training_2200.log 2>&1 &

echo $!
```

Monitor it:

```bash
tail -f artifacts/logs/adaptive_hybrid_training_2200.log
```

The first detailed line appears only after candidate-pool collection. A typical Apple
Silicon run is expected to take roughly 40–90 minutes and several GB of memory.

The repository currently contains the successful 200-session structural smoke fit, not
the completed 2,200-session structural fit. The full run is intentionally deferred; do
not treat the future `structural_v2` paths below as existing selected assets until the
trainer finishes and their receipts validate.

Training emits:

```text
configs/adaptive_hybrid_1a_3b_2200_structural_v2.json
configs/splits/adaptive_2200_nested_v1.json
artifacts/models/adaptive_union_gbdt_2200_structural_v2.json
artifacts/models/adaptive_union_gbdt_2200_structural_v2.fit_receipt.json
artifacts/models/adaptive_browsing_gbdt_2200_structural_v2.json
artifacts/models/adaptive_browsing_gbdt_2200_structural_v2.fit_receipt.json
artifacts/reports/adaptive_hybrid_training_2200_structural_v2.json
```

Each model is bound into the output configuration only if it passes out-of-fold Hit@10
non-regression and strict MRR improvement. Both models are still fitted and reported so
rejected evidence is retained.

After the structural fit succeeds, tune Qwen depth and compare the bounded model family:

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_local_llm_rankers.py
```

This evaluates Qwen at Top 10/20/30, then compares SmolLM2 and Qwen3 only at the winning
depth. It emits a report and a separate hash-bound `structural_v2_selected` configuration;
it does not overwrite the trained base config.

## Run the architecture-safe GhostLab campaign

Do not start this campaign until 2,200-sample fitting finishes and the output config,
models and receipts are validated.

Architecture-only plan:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_2200_structural_v2.json \
  --plan-only \
  --output artifacts/reports/adaptive_hybrid_campaign_plan_2200.json
```

Full campaign:

```bash
mkdir -p artifacts/logs

nohup caffeinate -dimsu env PYTHONPATH=. .venv/bin/python \
  scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_2200_structural_v2.json \
  --dataset data/public_set.jsonl \
  --dataset data/synthetic_1000_public_like.jsonl \
  --dataset data/independent_template_1000.jsonl \
  --candidate-limit 500 \
  --beam-width 24 \
  --higher-order-rounds 8 \
  --f1-candidates 24 \
  --f2-candidates 6 \
  --hpo-trials-per-structure 2 \
  --output artifacts/reports/adaptive_hybrid_campaign_2200.json \
  > artifacts/logs/adaptive_hybrid_campaign_2200.log 2>&1 &

echo $!
```

Every evaluation start and finish is logged with fidelity, sample count, score and wall
time.

### Default fidelity allocation

The progressive ordering is deterministic and proportional within every dataset source
and scenario:

| Phase | Public | Public-like | Independent-template | Total per candidate |
|---|---:|---:|---:|---:|
| F0 | 40 | 200 | 200 | 440 |
| F1 | 100 | 500 | 500 | 1,100 |
| F2 | 200 | 1,000 | 1,000 | 2,200 |

F0 contains 176 Buying, 176 Browsing, 66 Intent Override and 22 Boundary sessions. F1
contains 440 Buying, 440 Browsing, 165 Intent Override and 55 Boundary sessions.

F0 starts with 62 candidates and may expand up to the 500-candidate cap. F1 retains at
most 24 roots and adds two local HPO trials per non-control root, for at most 70 F1
evaluations. F2 evaluates at most six finalists.

This uses the entire corpus for final selection without spending full-data evaluation on
every weak F0 candidate. Increasing early percentages reduces variance but does not
remove synthetic-data bias.

### Racing non-regression gates

The runner constructs balanced source/scenario prefixes and applies grouped promotion
gates in addition to the combined paired session reward:

- official-public performance must not materially regress;
- Buying, Browsing and Intent Override must not materially regress;
- sparse F0 Boundary evidence must produce `HOLD_MORE_DATA`, not permanent rejection;
- final promotion must use F2 over all 2,200 sessions.

The exhaustive default campaign is expected to take hours on one Mac. Retrieval/Qwen
caching and lower early candidate caps are valid engineering optimizations only when they
preserve paired behavior and the fixed architecture.

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
| Adaptive Hybrid, Browsing-only Qwen | 0.985000 | 0.578286 | 2.045000 | 0.845086 |

The adaptive candidate improves Hit@10 and MTTC relative to the champion but loses MRR.
It is the architecture-complete optimization base, not the active winner.

The semantic activation study used the same 200 public sessions:

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
| `configs/adaptive_hybrid_1a_3b_2200_structural_v2.json` | Deferred trainer output (created only after a full fit) |
| `configs/techniques/catalog_v2.json` | Complete technique inventory |
| `ghostlab/runtime/adaptive_hybrid.py` | Fixed 1A–3B coordinator |
| `ghostlab/runtime/adaptive_config.py` | Typed required configuration contract |
| `ghostlab/runtime/adaptive_components.py` | Router, guidance, rankers and profile adapters |
| `ghostlab/optimization/adaptive_hybrid.py` | Architecture audit and safe binding |
| `ghostlab/optimization/adaptive_techniques.py` | Technique classification/binding |
| `ghostlab/optimization/adaptive_campaign.py` | F0/F1/F2 race engine |
| `ghostlab/training/adaptive_hybrid.py` | Runtime-pool collection and ranking data |
| `ghostlab/training/adaptive_datasets.py` | Multi-source loading and balanced folds |
| `scripts/train_adaptive_hybrid.py` | Unified 2,200-session ranker trainer |
| `scripts/run_adaptive_hybrid_campaign.py` | Architecture-safe GhostLab runner |
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
