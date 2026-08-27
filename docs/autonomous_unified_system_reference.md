# Autonomous unified experimentation system reference

Status: canonical reference for the consolidated `techjam-unified` system. Development
may occur in an integration worktree, but the paths and commands below are repository
relative and remain the same after consolidation.

For the technique-by-technique historical registry, prior scores, challenger lineage, and
retest triggers, also see `docs/unified_technique_operations.md`. This document is the
canonical source for the current autonomous campaign contract and executable commands.

This document describes implemented behavior, not intended future behavior. A command or
orchestrator entry point that does not exist in this worktree is labelled **PENDING**. The
system is proposal-only: it may search, evaluate, compare, and package candidates, but it
must not promote a champion, expose the protected F3 split, merge, commit, or push.

## 1. Operating contract

The safe lifecycle is:

1. Commit a clean, reviewable experiment base.
2. Freeze a versioned campaign manifest. The manifest records the Git commit, the selected
   catalog file hash, the dataset hash declared by the adaptive split, and hashes of the
   adaptive and nested split files.
3. Start composable discovery from the pure unified keyword anchor, while evaluating
   previous champions/challengers only as non-composable controls.
4. Evaluate only F0, F1, and F2 development fidelities. Resume from checkpoints after an
   interruption.
5. Compare candidates on paired sessions, scenario deltas, uncertainty, runtime, memory,
   complexity, and interaction evidence.
6. Materialize three distinct human-review proposals: score leader, robust leader, and
   efficient alternative.
7. Stop at Gate A. F3 is outside this system and has no command in this document.

Invariant boundaries enforced in code:

- `CampaignManifest.protected_holdout_access` is literally `"forbidden"`.
- `Fidelity` is restricted to `f0`, `f1`, and `f2`; there is no F3 campaign fidelity.
- `OfflineCampaignEvaluator` rejects paths containing `f3`, `holdout`, `protected`, or
  `sealed`.
- Proposal materialization rejects protected/F3 paths and records
  `automatic_promotion: false`.
- Runtime candidates may use observable features only. Targets, correct products, future
  user turns, session rewards, and turn outcomes are training/evaluation labels, never
  runtime inputs.
- A result on all adaptive development data is exploratory screening, not promotion
  evidence. Nested outer-fold or other predeclared out-of-fold evidence is required.

### 1.1 Absolute-start fairness contract

The default autonomous campaign does **not** begin from the previous winner. Its sole
composable discovery anchor is `configs/suites/unfitted_keyword_search.json`, deliberately
reduced to the smallest switchable conversational runtime:

| Capability | Pure-anchor value |
|---|---|
| Conversation state | current user turn only |
| Question policy | fixed starter questions |
| Retrieval | keyword/BM25 only |
| Dense or hybrid retrieval | off |
| Query expansion and structured filtering | off |
| Learned/fixed reranker | off |
| Profile and quality priors | `0.0` |
| Negative-evidence, provenance, override logic | off |
| Fitted model assets | none |

This is the **pure unified conversational baseline**, not a claim that it is byte-for-byte
the organizer's stateless reference agent in `baseline/official_reference.py`. The latter
remains the permanent external floor. Fixed questions are retained only so question/state
techniques can be compared inside the common conversational runtime.

The campaign declares only `state.current`, `question.fixed`, and `retrieval.sparse` as
baseline techniques for this anchor. Raw/multi/compressed history, state normalization,
query rewriting, learned/adaptive questions, priors, reranking, fusion, and diversification
must therefore be added and evaluated as explicit techniques. `champion_guarded.json`,
`keyword_research.json`, and `learned_questions.json` are `control_only`: they are reported
for context but cannot seed beams, donate fitted weights, or determine which combinations
are explored. Any campaign that changes the pure anchor or makes a historical system
`composable` is a different experiment and must use a new manifest/campaign ID.

## 2. What is implemented and what remains pending

| Capability | Status | Actual entry point |
|---|---|---|
| Immutable campaign freeze | Implemented | `scripts/freeze_wave2_campaign.py` |
| Multi-anchor finite planning | Implemented | `scripts/plan_wave2_campaign.py` |
| Typed candidate-to-runtime binding | Implemented | `ghostlab/campaign/bindings.py` |
| Compatibility/dependency validation | Implemented | `ghostlab/campaign/compatibility.py` |
| Resource-safe deterministic scheduling | Implemented library | `ghostlab/campaign/scheduler.py` |
| Atomic checkpoint/resume | Implemented library | `ghostlab/campaign/runner.py` |
| Development-only replay evaluator | Implemented library | `ghostlab/campaign/evaluator.py` |
| F0/F1/F2 promotion stages | Implemented library | `ghostlab/campaign/controller.py` |
| Bounded interaction-aware structure search | Implemented library | `ghostlab/campaign/interaction_search.py` |
| Hyperband/successive halving | Implemented library | `ghostlab/optimization/hyperband.py` |
| BOHB-style suggestions | Implemented library | `ghostlab/optimization/bohb.py` |
| Paired statistical analysis | Implemented library | `ghostlab/campaign/analyze.py` |
| Specialized nested combination campaign | Implemented CLI | `scripts/run_wave2_combination_campaign.py` |
| Bounded frozen-manifest execute/resume CLI | Implemented | `scripts/run_autonomous_campaign.py` and `ghostlab/campaign/orchestrator.py` |
| Deterministic top-three selection | Implemented library | `ghostlab/campaign/top_three.py` |
| Immutable proposal bundle | Implemented library | `ghostlab/campaign/proposal_materializer.py` |
| Confirmed top-three selection/materialization CLI | Implemented | `scripts/materialize_campaign_top_three.py` and `ghostlab/campaign/proposal_from_campaign.py` |
| Paired proposal report comparison | Implemented CLI | `scripts/compare_proposal_reports.py` |
| Automatic champion promotion | Deliberately absent | Human Gate A/F3/Gate B only |

`scripts/run_autonomous_campaign.py` is the generic manifest-driven runner. It currently
replans from the frozen manifest rather than consuming the JSON audit preview emitted by
`plan_wave2_campaign.py`. `scripts/run_wave2_combination_campaign.py` remains a separate,
concrete nested experiment over factors `N/Q/J/R/E/X/D`.

## 3. Repository map

### 3.1 Competition and runtime boundary

| Path | Responsibility |
|---|---|
| `ghostlab/competition/contract.py` | Agent request/response contract and ask/recommend actions |
| `ghostlab/runtime/agent.py` | Main runtime agent assembly |
| `ghostlab/runtime/compiled.py` | Compiled configuration runtime |
| `ghostlab/runtime/guarded_gbdt.py` | Guarded constraint-GBDT champion path |
| `ghostlab/runtime/unified_experimental.py` | Unified switchable experimental runtime |
| `ghostlab/runtime/component_fallback.py` | Component-scoped fallback implementation |
| `ghostlab/runtime/trace.py` | Runtime observability/trace support |
| `ghostlab/research/technique_suite.py` | Typed `UnifiedTechniqueConfig`, preset loading, suite factory, finite combination validation |
| `scripts/run_unified_preset.py` | Evaluate one materialized unified preset against an optional registered split |

### 3.2 Campaign architecture

| Path | Responsibility |
|---|---|
| `ghostlab/campaign/models.py` | Immutable schemas for techniques, candidates, manifests, jobs, resources, and outcomes |
| `ghostlab/campaign/freeze.py` | Repository-path safety, input/split/asset validation, clean-commit freeze, immutable write |
| `ghostlab/campaign/catalog.py` | Load and hash the v1/v2 technique catalogs; classify historical/research techniques |
| `ghostlab/campaign/bindings.py` | Exhaustive typed runtime binding registry and candidate materialization |
| `ghostlab/campaign/compatibility.py` | Requirements, conflicts, exclusive groups, availability, and asset checks |
| `ghostlab/campaign/planner.py` | Control plus compatible combinations up to a bounded order; dependency closure and ablations |
| `ghostlab/campaign/jobs.py` | Deterministic F0/F1/F2 job construction |
| `ghostlab/campaign/cache.py` | Campaign cache helpers |
| `ghostlab/campaign/scheduler.py` | Deterministic CPU/GPU/memory/heavy-model scheduling waves |
| `ghostlab/campaign/runner.py` | Threaded job execution and atomic manifest-bound checkpoints |
| `ghostlab/campaign/evaluator.py` | Development-only replay, stratified fidelity samples, paired session rewards, latency and memory |
| `ghostlab/campaign/controller.py` | Initial F0 stage and deterministic score/exploration promotion to later fidelities |
| `ghostlab/campaign/orchestrator.py` | Manifest-driven F0/F1/F2 search, bounded HPO, resume, safety records, and proposal-only evidence |
| `ghostlab/campaign/interaction_search.py` | Standalones, pairs, diverse beams, reserve/resurrection, strict pruning, audit samples |
| `ghostlab/campaign/analyze.py` | Paired bootstrap/randomization/scenario and interaction analysis |
| `ghostlab/campaign/proposal.py` | Proposal records used by campaign reporting |
| `ghostlab/campaign/top_three.py` | Safe confirmed top-three role selection |
| `ghostlab/campaign/proposal_from_campaign.py` | Fail-closed campaign/checkpoint/split verification and package reconstruction |
| `ghostlab/campaign/proposal_materializer.py` | Immutable presets, hashes, dependencies, evidence, Gate text, rollback metadata |
| `ghostlab/campaign/proposal_compare.py` | Paired comparison of baseline and up to three proposal reports |
| `scripts/run_autonomous_campaign.py` | CLI for the bounded, resumable manifest-driven campaign |
| `scripts/materialize_campaign_top_three.py` | CLI for immutable development-confirmed score/robust/efficient proposals |

### 3.3 Search and optimization

| Path | Responsibility |
|---|---|
| `ghostlab/optimization/search.py` | Original random/grid/beam utilities and interaction gain |
| `ghostlab/optimization/racing.py` | Multi-fidelity racing utilities |
| `ghostlab/optimization/meta_search.py` | Evidence allocation/meta-search utilities |
| `ghostlab/optimization/evidence.py` | Family evidence and decision ledger support |
| `ghostlab/optimization/patches.py` | Typed patches and crossover helpers |
| `ghostlab/optimization/conditional.py` | Conditional parameter-space logic |
| `ghostlab/optimization/hyperband.py` | Deterministic successive halving over a frozen trial population |
| `ghostlab/optimization/bohb.py` | Seeded random exploration and elite-local BOHB-style suggestions |
| `configs/search/unified_space_v1.json` | Original finite unified switch space |
| `configs/search/wave2_weight_space_v1.json` | Wave 2 declared weight/parameter space |

The autonomous runner uses `ConditionalSearchSpace` and `suggest_for_combination()` for
bounded F1 racing. Each round rebuilds the observation history for a structure and feeds
it to the BOHB-style sampler, mixing seeded exploration with elite-local suggestions.
This optimization stays inside the prospectively frozen search folds. `successive_halving()`
is available as a library but is not separately wired into this runner.

### 3.4 State, policy, retrieval, and ranking implementations

| Area | Paths |
|---|---|
| Conversation memory | `ghostlab/state/memory.py` |
| Structured state/query | `ghostlab/state/query.py` |
| Catalog ontology and normalization | `ghostlab/state/catalog_ontology.py`, `ghostlab/state/normalization.py` |
| Query expansion | `ghostlab/state/query_expansion.py`, `ghostlab/retrieval/pseudo_relevance.py` |
| Candidate statistics/EIG | `ghostlab/policy/candidate_statistics.py`, `ghostlab/policy/eig_questions.py` |
| Adaptive/learned questions | `ghostlab/policy/adaptive_questions.py`, `ghostlab/policy/learned_questions.py` |
| Joint actions/policy | `ghostlab/policy/joint_actions.py`, `ghostlab/policy/joint_policy.py` |
| Distilled/calibrated policies | `ghostlab/policy/distilled_expert.py`, `ghostlab/policy/calibrated_router.py` |
| Sparse/dense/fusion | `ghostlab/retrieval/sparse.py`, `dense.py`, `fusion.py`, `sparse_semantic_fusion.py` |
| Learned sparse/late interaction | `ghostlab/retrieval/learned_sparse.py`, `late_interaction.py` |
| Filtering/priors | `ghostlab/retrieval/filters.py`, `profile.py`, `quality.py` |
| Fixed/learned/GBDT rankers | `ghostlab/retrieval/rerank.py`, `learned.py`, `gbdt.py`, `constraint_gbdt.py`, `gbdt_dense.py` |
| Neural/cross encoder | `ghostlab/retrieval/neural_rank.py`, `cross_encoder.py` |
| Reward rankers/ensembles | `ghostlab/retrieval/reward_lambdamart.py`, `ensemble.py` |
| Diversification | `ghostlab/retrieval/diversify.py` |
| Offline counterfactual/replay | `ghostlab/research/counterfactual.py`, `eig_counterfactual.py`, `joint_counterfactual.py`, `counterfactual_expert.py`, `replay.py` |
| Leakage firewall | `ghostlab/research/firewall.py` |

### 3.5 Configuration, evidence, and tests

| Path | Meaning |
|---|---|
| `configs/techniques/catalog_v1.json` | Original technique registry |
| `configs/techniques/catalog_v2.json` | Wave 2 extension; loaded together with v1 |
| `configs/campaigns/wave2_smoke_v1.template.json` | One-anchor smoke campaign template |
| `configs/campaigns/autonomous_full_v1.template.json` | Multi-anchor bounded campaign template |
| `configs/splits/adaptive_v1.json` | Adaptive development IDs and dataset hash |
| `configs/splits/nested_v1.json` | Five outer development folds |
| `configs/validation/primary_analysis.json` | Predeclared analysis settings |
| `configs/suites/*.json` | Runnable unified presets/anchors |
| `configs/experiments/*.json` | Historical and Wave 2 experiment manifests |
| `configs/integrity/unified_consolidation_v1.json` | Consolidation provenance and required paths |
| `artifacts/evidence/technique_decisions.jsonl` | Historical decision ledger |
| `artifacts/evidence/wave2_policy_decisions.jsonl` | Wave 2 policy decisions |
| `artifacts/reports/w2_*.json` | Wave 2 replay, mechanism, ranking, and decision evidence |
| `tests/test_campaign*.py` | Campaign schema, planning, execution, and integration tests |
| `tests/test_campaign_freeze.py` | Frozen input, path, asset, clean-worktree, and immutability tests |
| `tests/test_campaign_orchestrator.py` | Manifest verification, resume, safety, and proposal-only orchestration tests |
| `tests/test_campaign_bindings.py` | Binding completeness/conflict/materialization tests |
| `tests/test_campaign_evaluator.py` | Development-only evaluator tests |
| `tests/test_campaign_interaction_search.py` | Search/pruning/resurrection tests |
| `tests/test_top_three_proposals.py` | Proposal selection, hashing, safety, and immutability tests |
| `tests/test_campaign_proposal_cli.py` | Confirmation/split/checkpoint binding and fail-closed CLI tests |
| `tests/test_wave2_unified_factory.py` | Wave 2 unified factory tests |
| `docs/unified_technique_operations.md` | Full first-version/challenger technique registry, lineage, scores, evidence, and retest triggers |
| `docs/autonomous_unified_system_reference.md` | Current autonomous architecture, fairness contract, commands, search, validation, and human gates |

## 4. Technique registry and classification

The catalog is the record of discoverable techniques. The binding registry is the
runtime truth. A catalog item is not necessarily composable merely because source code
exists.

Classifications:

- **Composable**: can be materialized as a typed `UnifiedTechniqueConfig` patch now.
- **Anchor-only**: runnable only as a complete preset/compiled anchor, or inseparable
  from another switch. Do not add it as a candidate patch.
- **Research-only**: offline label generation, search, or evaluation procedure; never a
  runtime feature.
- **Unavailable**: source or a placeholder manifest exists, but a required local asset or
  typed binding is absent.

Binding capability and selection eligibility are separate axes. A typed composable patch
may still be barred from prospective candidates by catalog flags. In particular,
`question.learned_linear` and `ranking.metadata_gbdt` retain bindings so historical
presets remain runnable, while `fit_required=true` and `selection_safe=false` force those
presets to `control_only` until fold-safe refitting exists.

### 4.1 Original runtime techniques

| Technique ID | Class | Source / reason |
|---|---|---|
| `state.current` | Composable | `ghostlab/runtime/experimental.py` |
| `state.raw_history` | Composable | `ghostlab/state/memory.py` |
| `state.multi` | Composable | `ghostlab/state/memory.py` |
| `state.compressed` | Composable | `ghostlab/state/memory.py` |
| `query.structured` | Composable | `ghostlab/state/query.py` |
| `question.fixed` | Composable | fixed control |
| `question.adaptive_heuristic` | Composable | `ghostlab/policy/adaptive_questions.py` |
| `question.learned_linear` | Historical control-only anchor | typed binding exists, but catalog is `fit_required=true`, `selection_safe=false`; prior fitted asset cannot enter prospective selection |
| `retrieval.sparse` | Composable | `ghostlab/retrieval/sparse.py` |
| `retrieval.minilm` | Unavailable | local MiniLM asset absent |
| `retrieval.e5` | Unavailable | local E5 asset absent |
| `fusion.rrf` | Unavailable | dense asset required by this route is absent |
| `fusion.weighted` | Unavailable | dense asset required by this route is absent |
| `fusion.sparse_first_union` | Unavailable | dense asset required by this route is absent |
| `ranking.fixed_lexical` | Composable | `ghostlab/retrieval/rerank.py` |
| `ranking.pairwise_linear` | Anchor-only | historical ranker not represented by the unified reranker enum |
| `ranking.metadata_gbdt` | Historical control-only anchor | typed binding/asset exists, but catalog is `fit_required=true`, `selection_safe=false`; prior fit predates the 90/60 boundary |
| `ranking.constraint_gbdt` | Anchor-only | compiled suite anchor, not an additive patch |
| `ranking.deep_dense_gbdt` | Anchor-only | historical standalone challenger |
| `ranking.cross_encoder` | Unavailable | local cross-encoder asset absent |
| `ranking.neural_gbdt` | Anchor-only | historical standalone challenger |
| `filter.structured` | Composable | `ghostlab/retrieval/filters.py` |
| `prior.profile` | Composable | typed initial weight; tune only inside inner folds |
| `prior.quality` | Composable | typed initial weight |
| `guard.override_fallback` | Anchor-only | part of compiled guarded champion |
| `routing.decision_list` | Anchor-only | supporting mechanism selected through joint-policy asset |
| `routing.observable_stump` | Anchor-only | historical route-policy anchor |
| `routing.route_table` | Anchor-only | historical route-policy anchor |

### 4.2 Original research, search, and evaluation procedures

All items in this table are **Research-only**:

| Technique ID | Source |
|---|---|
| `research.counterfactual` | `ghostlab/research/counterfactual.py` |
| `research.replay` | `ghostlab/research/replay.py` |
| `research.leakage_firewall` | `ghostlab/research/firewall.py` |
| `search.random_grid_beam` | `ghostlab/optimization/search.py` |
| `search.multifidelity_racing` | `ghostlab/optimization/racing.py` |
| `search.evidence_allocator` | `ghostlab/optimization/meta_search.py` |
| `search.family_ucb` | `ghostlab/optimization/evidence.py` |
| `search.typed_patches` | `ghostlab/optimization/patches.py` |
| `search.crossover` | `ghostlab/optimization/patches.py` |
| `evidence.decision_store` | `ghostlab/optimization/evidence.py` |
| `evaluation.grouped_splits` | `ghostlab/evaluation/splits.py` |
| `evaluation.paired_statistics` | `ghostlab/evaluation/statistics.py` |

### 4.3 Wave 2 policy/state techniques

| Technique ID | Class | Source / requirement |
|---|---|---|
| `state.catalog_normalizer.v1` | Composable | `ghostlab/state/normalization.py`; ontology asset required |
| `state.attribute_ontology.v1` | Anchor-only | asset-producing dependency of the normalizer |
| `state.confidence_gated_constraints.v1` | Composable | requires `state.catalog_normalizer.v1` |
| `question.candidate_eig.v1` | Composable; interaction reserve | `ghostlab/policy/eig_questions.py` |
| `question.reward_voi.v1` | Unavailable/planned | no fold-fitted reward-VOI asset binding |
| `termination.reward_aware.v1` | Composable | requires candidate EIG |
| `policy.joint_observable.v1` | Composable | uses `configs/assets/joint_policy_control_v1.json` |
| `routing.joint_route.v1` | Anchor-only | inseparable from selected joint-policy asset |
| `research.counterfactual_expert.v2` | Research-only | offline label generator |
| `policy.distilled_expert.v1` | Unavailable/planned | no fold-fitted distilled runtime asset |
| `search.expert_iteration.v1` | Research-only | offline dataset aggregation |
| `routing.calibrated_observable.v1` | Unavailable/planned | no fold-fitted router asset |
| `guard.component_fallback.v1` | Unavailable | requires unavailable calibrated router asset |

### 4.4 Wave 2 ranking techniques

| Technique ID | Class | Asset / source |
|---|---|---|
| `ranking.reward_lambdamart.v1` | Composable; interaction reserve | `artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json` |
| `ranking.turn_aware_lambdamart.v1` | Composable; parked | `artifacts/models/w2_ranking_v1/turn_aware_lambdamart_v1.json` |
| `ranking.fold_ensemble.v1` | Composable; interaction reserve | `artifacts/models/w2_ranking_v1/fold_ensemble.json` |
| `fusion.rank_stack.v1` | Composable; interaction reserve | `artifacts/models/w2_ranking_v1/rank_stack.json` |

Supporting fixed controls remain in `artifacts/models/w2_ranking_v1/`:
`ndcg_at_10_control.json`, `pointwise_control.json`, and
`uniform_pairwise_control.json`. Ranking training, exact organizer reward deltas, and
fold-local auditing live in `ghostlab/retrieval/reward_lambdamart.py`,
`ghostlab/retrieval/ensemble.py`, `ghostlab/evaluation/reward_deltas.py`, and
`ghostlab/evaluation/gbdt_audit.py`.

### 4.5 Wave 2 retrieval, expansion, and diversity techniques

| Technique ID | Class | Source / reason |
|---|---|---|
| `retrieval.splade_rescue.v1` | Unavailable/planned | learned-sparse model/index asset absent |
| `fusion.sparse_semantic_union.v1` | Unavailable/planned | learned-sparse dependency absent |
| `retrieval.late_interaction_rescue.v1` | Unavailable/planned | feasibility asset absent |
| `retrieval.colbert_rescue.v1` | Unavailable/planned | ColBERT gate produced no admitted asset |
| `retrieval.bge_m3_rescue.v1` | Unavailable/planned | BGE-M3 gate produced no admitted asset |
| `fusion.late_interaction_union.v1` | Unavailable/planned | late-interaction dependency absent |
| `query.catalog_prf.v1` | Composable; parked | catalog pseudo-relevance feedback |
| `query.query2doc_local.v1` | Unavailable/planned | optional local generation model not admitted |
| `query.expansion_guard.v1` | Anchor-only | intrinsic to PRF, no independent toggle |
| `ranking.facet_diversity.v1` | Composable; interaction reserve | facet MMR |
| `ranking.mmr_early.v1` | Anchor-only | early-turn gate intrinsic to current facet MMR |

### 4.6 Wave 2 optimization techniques

| Technique ID | Class | Source |
|---|---|---|
| `search.hyperband.v1` | Research-only | `ghostlab/optimization/hyperband.py` |
| `search.bohb.v1` | Research-only | `ghostlab/optimization/bohb.py` |

These procedures optimize experiments; they are not on/off runtime features.

## 5. Installation, extras, data, and assets

Run from the repository root.

```bash
uv sync
```

Core requires Python `>=3.10,<3.14`, NumPy, and Pydantic. Optional extras are declared in
`pyproject.toml`:

| Extra | Install | Needed for |
|---|---|---|
| Core | `uv sync` | sparse/state/policy/fixed ranking/campaign logic |
| GBDT | `uv sync --extra gbdt` | scikit-learn GBDT/LambdaMART-related training and audits |
| Dense | `uv sync --extra dense` | sentence-transformer dense retrieval |
| Neural | `uv sync --extra neural` | cross-encoder/neural/late-interaction work; includes PyTorch |
| Everything | `uv sync --extra all` | all declared optional dependencies |
| Development tools | `uv sync --group dev` | pytest, Ruff, mypy |

The data paths expected by current runners are:

- `data/public_set.jsonl`
- `data/catalog.jsonl`
- `configs/splits/adaptive_v1.json`
- `configs/splits/nested_v1.json`

`data/public_set.jsonl` is checked in. `data/catalog.jsonl` is release data and may be
locally supplied/ignored, depending on the repository checkout; verify it before running.
Do not commit downloaded model caches or generated vector/index payloads merely because a
runner needs them. Keep lightweight manifests and approved compact runtime assets in the
repository; use the project asset scripts for reproducibility:

```bash
uv run python -m scripts.fetch_optional_assets e5 --verify-only
uv run python -m scripts.fetch_optional_assets minilm --verify-only
uv run python -m scripts.fetch_optional_assets cross_encoder --verify-only
uv run python -m scripts.fetch_dense_assets --verify-only
uv run python -m scripts.build_attribute_ontology \
  --catalog data/catalog.jsonl \
  --output artifacts/assets/catalog_ontology_v1.json
```

The learned-sparse and late-interaction builders require an already available model and a
manifest template; they do not download or bless a model:

```bash
uv run python -m scripts.build_learned_sparse_index \
  --catalog data/catalog.jsonl \
  --model /path/to/local/model \
  --manifest-template configs/assets/splade_rescue_v1.json \
  --index /path/to/output/index \
  --output-manifest /path/to/output/manifest.json

uv run python -m scripts.build_late_interaction_index \
  --catalog data/catalog.jsonl \
  --model /path/to/local/model \
  --manifest-template configs/assets/late_interaction_rescue_v1.json \
  --index /path/to/output/index \
  --output-manifest /path/to/output/manifest.json
```

Those example output/model paths are intentionally placeholders. A technique remains
unavailable until the binding registry is updated to a real local, reviewed asset and the
catalog/compatibility tests pass.

## 6. Exact workflows

### 6.1 Run one preset

Use this for controls, one technique, or a manually materialized combination:

```bash
uv run python -m scripts.run_unified_preset \
  --config configs/suites/unfitted_keyword_search.json \
  --split configs/splits/adaptive_v1.json \
  --output artifacts/reports/local_pure_keyword_baseline.json
```

Omitting `--split` evaluates all rows loaded by the runner and is not confirmatory evidence.
Use `configs/suites/champion_guarded.json` only when deliberately reproducing that
historical control; it is not the autonomous discovery starting point.

### 6.2 Materialize the original finite unified switch space

```bash
uv run python -m scripts.plan_unified_combinations \
  --space configs/search/unified_space_v1.json \
  --output artifacts/campaign/unified_combinations.jsonl \
  --output-dir artifacts/campaign/unified_presets
```

Use `--limit N` for a bounded smoke materialization. This command validates and writes
presets; it does not evaluate them.

### 6.3 Freeze a campaign

Freeze requires a clean worktree because the recorded `HEAD` must contain every input.

```bash
uv run python -m scripts.freeze_wave2_campaign \
  --template configs/campaigns/autonomous_full_v1.template.json \
  --output artifacts/campaigns/autonomous_full_v1/manifest.json
```

The freeze command hashes the selected catalog file and the actual development dataset,
checks that both split files declare that same dataset hash, verifies that outer folds
partition the adaptive IDs, and hashes the adaptive and nested split files. It also checks
repository-safe preset/asset paths and campaign binding compatibility. Never hand-edit a
frozen manifest; create a new versioned template and freeze a new campaign ID.

Current limitation: `catalog_v2.json` extends `catalog_v1.json`, but the current catalog
`content_hash` covers the selected v2 file bytes only, not a transitive hash of its parent.
Therefore any v1 catalog edit must also produce a reviewed, versioned v2/catalog change
before freezing. A future implementation should hash the resolved catalog closure.

### 6.4 Plan a frozen campaign

```bash
uv run python -m scripts.plan_wave2_campaign \
  --manifest artifacts/campaigns/autonomous_full_v1/manifest.json \
  --catalog configs/techniques/catalog_v2.json \
  --output artifacts/campaigns/autonomous_full_v1/plan.json
```

The planner validates the catalog hash, produces per-anchor controls and combinations,
applies dependency closure, rejects incompatible/non-runtime additions, bounds order and
candidate count, and emits F0 jobs plus explicit skip reasons.

### 6.5 Execute and resume the generic frozen-manifest campaign

```bash
uv run python -m scripts.run_autonomous_campaign \
  --manifest artifacts/campaigns/autonomous_full_v1/manifest.json \
  --technique-catalog configs/techniques/catalog_v2.json \
  --dataset data/public_set.jsonl \
  --product-catalog data/catalog.jsonl \
  --adaptive-split configs/splits/adaptive_v1.json \
  --nested-split configs/splits/nested_v1.json \
  --checkpoint artifacts/campaigns/autonomous_full_v1/checkpoint.json \
  --evidence artifacts/reports/autonomous_full_v1.json \
  --f1-candidates 24 \
  --f2-candidates 6 \
  --hpo-trials-per-structure 8 \
  --higher-order-rounds 2 \
  --bootstrap-resamples 1000
```

Run the identical command to resume. `verify_frozen_inputs()` recomputes the catalog,
dataset, and split hashes before evaluation. `load_checkpoint()` rejects another manifest,
and `run_jobs()` skips already recorded job IDs. Evidence is atomically replaced after a
completed orchestration pass; checkpoint outcomes are atomically updated after each job.

The runner directly replans from the frozen manifest; `plan.json` is an audit preview, not
an execution input. Its current procedure is:

1. load the pure keyword/fixed-question suite as the only composable anchor and load prior
   winner/challenger suites as controls only;
2. prospectively reserve the manifest's `search_outer_folds` for F0/F1 selection and its
   disjoint `confirmation_outer_folds` for F2 only;
3. enumerate controls, every compatible standalone, and bounded pairs per composable
   anchor;
4. run F0 and evidence-guided higher-order rounds on search-fold sessions;
5. promote a bounded score-leading/exploration set to F1;
6. run observation-informed bounded conditional HPO/racing on F1 search sessions;
7. remove candidates containing `selection_safe=false` or `fit_required=true` techniques,
   then promote the remaining bounded set to F2;
8. evaluate the frozen finalists with one frozen seed only on the disjoint F2 confirmation
   folds, then emit the full F0/F1/F2 leaderboards, safety records, and up to three
   package-eligible development-confirmed summaries.

This path reports `prospective_disjoint_confirmation`, not nested OOF. It prevents direct
selection/confirmation overlap by freezing disjoint public-development folds (currently
search `[0,2,3]`, confirmation `[1,4]`) and records their sample counts and canonical
ID hashes. This is independent **development** confirmation; it is not final
generalization proof and does not replace the human-gated one-shot F3 boundary.

### 6.6 Run/resume the specialized Wave 2 combination campaign

```bash
uv run python -m scripts.run_wave2_combination_campaign \
  --checkpoint artifacts/campaign/w2_combinations.jsonl \
  --output artifacts/reports/w2_combination_campaign.json \
  --top-structures 4
```

Run the identical command to resume. Completed records are keyed by the config, sample
IDs, and label and are reused from the append-only JSONL checkpoint. This campaign:

- screens `N/Q/J/R/E/X/D` combinations on development data for diagnosis only;
- performs five outer folds for the core `N/Q/J/X/D` factors;
- selects structures and tunes declared parameters using fold-local inner IDs;
- evaluates each chosen fold configuration once on its outer fold;
- labels all-dev ranking as exploratory and reports nested OOF metrics separately.

### 6.7 Generate the three development-confirmed proposals

```bash
uv run python -m scripts.materialize_campaign_top_three \
  --manifest artifacts/campaigns/autonomous_full_v1/manifest.json \
  --catalog configs/techniques/catalog_v2.json \
  --evidence artifacts/reports/autonomous_full_v1.json \
  --checkpoint artifacts/campaigns/autonomous_full_v1/checkpoint.json \
  --adaptive-split configs/splits/adaptive_v1.json \
  --nested-split configs/splits/nested_v1.json \
  --baseline-id configs/suites/unfitted_keyword_search.json \
  --output artifacts/proposals/autonomous_full_v1
```

The CLI fails closed unless all of the following match:

- manifest, catalog, adaptive split, nested split, evidence, checkpoint, campaign ID,
  parent commit, and manifest hashes;
- `confirmation_status=independent_development_confirmation` and
  `selection_evidence_class=prospective_disjoint_confirmation`;
- the frozen search/confirmation fold partition, exact canonical sample-ID hashes/counts,
  zero overlap, and exactly one frozen F2 seed;
- at least three identical candidate IDs across the independent-confirmation declaration,
  `confirmed_top3`, and `proposal_eligible` safety records;
- candidate hashes and the exact completed checkpoint job IDs for every confirmation fold;
- one common matched control and one common declared baseline preset;
- selection-safe, non-fit-required techniques, typed materialization, existing assets,
  paired/scenario/resource gates, and strictly positive paired mean delta.

It reconstructs configs from the frozen baseline plus typed bindings, derives dependency
extras/assets, references and hashes the campaign evidence/checkpoint, then writes three
distinct role presets: score leader, robust leader, and efficient alternative. The output
is immutable and proposal-only. If the campaign's confirmed candidates span baselines,
run/confirm a baseline-specific shortlist; the CLI will not compare unmatched anchors.

The campaign leaderboard and the three proposals are deliberately different artifacts:

- the checkpoint retains job outcomes across screened candidates and fidelities;
- the autonomous evidence report retains exclusions, interaction diagnostics, all stage
  leaderboards, all F2 safety records, and up to three score-ranked development-confirmed
  summaries;
- `TopThreeSelection` transforms those confirmed summaries into three distinct
  score/robust/efficient human-review roles under stricter package gates;
- losing, parked, reserve, anchor-only, research-only, and unavailable techniques remain
  documented in the catalogs/binding registry for future retesting; shortlist exclusion
  does not delete their code or switch identity.

Techniques that require fold-local fitting may enter the current runner's F0/F1 mechanism
screens, but are removed before F2. A `selection_safe=false` technique is rejected by the
current freeze path and may be studied only through an explicitly controlled research
runner, never as a selectable campaign candidate. Neither class can become a confirmed
top-three package until the technique is selection-safe and any fitting procedure is
implemented inside each outer-training fold and validated without protected data.

### 6.8 Compare a materialized proposal bundle

The proposal bundle README supplies its exact paths. The existing comparison CLI is:

```bash
uv run python -m scripts.compare_proposal_reports \
  --baseline artifacts/proposals/<bundle>/reports/baseline.json \
  --candidate score_leader=artifacts/proposals/<bundle>/reports/score_leader.json \
  --candidate robust_leader=artifacts/proposals/<bundle>/reports/robust_leader.json \
  --candidate efficient_alternative=artifacts/proposals/<bundle>/reports/efficient_alternative.json \
  --output artifacts/proposals/<bundle>/reports/comparison.json
```

`<bundle>` is a placeholder for an actually materialized proposal directory. Do not run
this command until the bundle exists.

### 6.9 Restart from pure baseline or retest after a new baseline

For fair from-scratch discovery, always use the full template unchanged: the pure keyword
suite stays the sole composable anchor and every admitted technique is reconsidered from
that root. A stronger new baseline is a separate sensitivity/retest campaign; it must not
silently replace the from-scratch result. There is no dedicated `retest-new-baseline`
command. The supported sensitivity workflow is:

1. Add a new immutable suite JSON under `configs/suites/`.
2. Copy `configs/campaigns/autonomous_full_v1.template.json` to a new versioned template.
3. Give it a new `campaign_id`, add the suite path to `baseline_presets`, declare its
   `baseline_techniques_by_preset`, and set its `baseline_search_modes` entry to
   `composable`. Preserve the pure anchor and old controls; do not replace their evidence.
4. Keep unavailable, anchor-only, and research-only items visible in catalogs but out of
   the runtime `technique_ids` list. Re-admit them only after assets/bindings change.
5. Commit the reviewed template/config so freeze has a clean `HEAD`.
6. Freeze and plan with the actual commands in sections 6.3 and 6.4, substituting the new
   template/campaign directory.
7. Run the generic command in section 6.5 with new versioned checkpoint/evidence paths.
8. Materialize proposals only if the prospective disjoint confirmation produces at least
   three safe candidates for the same matched baseline.

This preserves old negative/interaction evidence while allowing every still-compatible
switch to be reconsidered against the new baseline.

### 6.10 Verified autonomous smoke behavior

The final clean-commit prospective-disjoint smoke exercised freeze, planning, the real
runner, checkpoint resume, and strict proposal materialization from the pure anchor:

| Check | Observed result |
|---|---:|
| Frozen parent commit | `6c3b5e8ce6b8a12d793747a6c9fbd39ed97fe1f7` |
| Pure control techniques | `state.current`, `question.fixed`, `retrieval.sparse` |
| Initial wall time | approximately `109.52` seconds |
| Identical-command resume | approximately `1.72` seconds |
| Stage counts | F0 `7`, F1 `13`, F2 `5` |
| Search/confirmation partition | `90` / `60` sessions; overlap `0` |
| F2 seeds | exactly one: `20260826` |
| HPO | two observation-informed rounds, four evaluated proposals per round |
| F2 matched control score | `0.097222` |
| Highest diagnostic F2 score/delta | `0.450861` / `+0.353639` |
| `confirmed_top3` | three behaviorally distinct positive candidates |
| Proposal boundary | score/robust/efficient presets materialized; `automatic_promotion=false`, `f3_access=forbidden` |

The smoke template admits only candidate EIG, reward LambdaMART, and facet diversity; it is
not the 600-candidate full campaign. These are mechanism/resume diagnostics, not a new
champion leaderboard or final promotion evidence. The very large delta reflects the
intentionally weak absolute-start control and must not be interpreted as an expected
competition gain. On another run, fewer than three behaviorally distinct, strictly
improving candidates remains a valid result; the proposal CLI must fail rather than
package controls, ties, or duplicate behaviors.

## 7. Search, pruning, combinations, and HPO rules

### 7.1 Candidate enumeration

`plan_candidates()` always includes a baseline control. It enumerates combinations up to
`max_order`, closes dependencies, validates compatibility, deduplicates by canonical
behavior hash, and stops at `candidate_limit`. Orders one, two, and three are labelled
single, pair, and triple; larger orders are labelled beam candidates.

The current `autonomous_full_v1` template bounds search at order 6 and 600 candidates.
This is a ceiling, not proof that every possible six-way combination was evaluated.

### 7.2 Interaction-aware exploration

`interaction_search.py` is designed not to discard a mildly weak standalone too early:

- enumerate all compatible standalones before pairs;
- group beams by technique-family signatures to preserve diversity;
- reserve candidates with missing, uncertain, mildly negative, or unrepeated evidence;
- keep seeded exploration outside the score-leading beam;
- calculate pair interaction as `both - first - second + control` on aligned sessions;
- support backward ablations and deterministic resurrection/audit samples;
- retain a pruning-audit sample so pruning mistakes can be measured.

An invalid configuration or exact behavioral duplicate may be permanently pruned
immediately. Performance domination requires repeated evidence, a mean below the declared
mild-loss floor, and an upper confidence bound below the matched control. A single bad
fold is not enough.

### 7.3 Multi-fidelity and resource pruning

F0 and F1 use deterministic scenario-stratified samples drawn only from the frozen search
folds at the manifest budgets. F2 uses only the frozen, disjoint confirmation folds and
one predeclared seed. `controller.promote_stage()` ranks completed mean scores but reserves
a seeded exploration fraction for pruning audit. Scheduler waves respect CPU, GPU, memory,
and heavy-model limits; jobs that cannot fit are rejected instead of silently
oversubscribing.

This is prospective disjoint development confirmation, not nested OOF. The specialized
combination runner is the current narrower path that performs fold-local inner
structure/parameter selection and outer-fold estimation.

### 7.4 Parameter/weight optimization

Weights are parameters, not techniques. For the current prospective-disjoint procedure:

- compare every tuned technique against a fixed, depth/capacity-matched control;
- tune only on frozen search-fold sessions;
- freeze the candidate, weights, confirmation folds, and one F2 seed before confirmation;
- never optimize on confirmation-fold outcomes or F3;
- constrain values to declared spaces such as `configs/search/wave2_weight_space_v1.json`;
- record seeds, trial IDs, resources, assets, and failed trials;
- treat the independent confirmation result as development evidence, not final proof.

`successive_halving()` is deterministic for a frozen trial population and increasing
resource levels. `bohb.suggest()` alternates seeded random exploration with elite-local
sampling. The generic runner updates each structure's observation history during bounded
F1 racing, while all of those observations remain confined to search folds. The current
declared `question_value_margin` domain is `[0.0, 0.05]`.

## 8. Statistics and overfitting controls

Primary candidate comparison is paired at the session level. `paired_analysis()` reports:

- mean paired reward delta;
- deterministic bootstrap interval (default 5,000 resamples);
- paired randomization p-value (default 5,000 resamples);
- wins/ties/losses;
- scenario-level score deltas.

The technical score is the organizer-aligned combination used by the evaluator:
`0.50 * HitRate@10 + 0.30 * MRR + 0.20 * efficiency`, where
`efficiency = clip((11 - MTTC) / 10, 0, 1)`.

Minimum promotion evidence should include:

- paired independent-confirmation session rewards against the exact matched anchor;
- no material scenario regression beyond the predeclared threshold;
- bootstrap interval and randomization result interpreted as uncertainty, not a magic
  pass/fail oracle;
- consistency across outer folds/seeds, not only the best fold;
- runtime p95, memory, failure/fallback counts, and complexity;
- mechanism diagnostics and backward ablations for important combinations;
- asset/config/commit hashes and deterministic reproduction commands.

Anti-overfit boundaries:

1. Feature engineering may inspect only information observable at that turn.
2. Labels and counterfactual outcomes are generated only inside outer-training data.
3. Current `fit_required=true` techniques are excluded before F2. To admit one, its model,
   calibration, or stacker must be fitted without confirmation-fold labels and frozen
   before confirmation; stricter nested fitting remains preferable when feasible.
4. Frozen search folds select structures, parameters, weights, and stopping rules.
5. Frozen disjoint confirmation folds estimate the already selected development procedure.
6. The adaptive 150 is development data and may be used through registered folds; it is
   not a forever-fresh holdout after repeated decisions.
7. The protected 50/F3 is unavailable to search, diagnostics, top-three selection, and
   proposal packaging.
8. Multiple anchors and pruning audits reduce dependence on one baseline or one lucky
   path, but do not turn repeated development tuning into independent evidence.

## 9. Multi-anchor behavior

`CampaignManifest.baseline_presets` may contain multiple suite paths. Each preset may have
its own technique baseline through `baseline_techniques_by_preset` and its own search mode:

- `composable`: compare the compatible runtime technique pool against this anchor.
- `control_only`: emit only the anchor/control; do not add technique patches.

`initial_stage()` divides the global candidate limit across anchors, builds a separate
plan for each, concatenates them deterministically, and applies the global cap. Candidate
analysis must always use the control with the same `baseline_id`; cross-anchor raw score
differences are descriptive, not paired causal evidence.

The current full template declares:

- `configs/suites/champion_guarded.json` as `control_only`;
- `configs/suites/keyword_research.json` as `control_only` because its metadata GBDT and
  tuned priors predate the prospective 90/60 boundary;
- `configs/suites/learned_questions.json` as `control_only` because its learned question
  and metadata-ranker assets predate that boundary;
- `configs/suites/unfitted_keyword_search.json` as the only default `composable` search
  anchor.

This preserves historical systems as descriptive controls without allowing their fitted
assets to leak into candidate selection. New combinations are searched from the unfitted
keyword anchor.

The pure anchor itself is explicit and reviewable in two places:

- runtime fields: `configs/suites/unfitted_keyword_search.json`;
- declared baseline technique identity and search mode:
  `configs/campaigns/autonomous_full_v1.template.json`.

Before freezing, verify that the first file still has current-turn state, fixed questions,
keyword retrieval, no dense backend/reranker, zero profile/quality priors, and all
negative/provenance/override switches off; verify that the second still marks only this
suite `composable`. These are fairness invariants, not tuning choices.

## 10. Human Gate A, one-shot F3, and Gate B

### Gate A — freeze one candidate or reject all

A human verifies OOF pairing, scenario safety, uncertainty, runtime/memory, failure modes,
dependencies, licensing, asset and preset hashes, compiled parity, rollback metadata, and
the predeclared one-shot analysis. Gate A may freeze exactly one candidate or retain the
champion. No code in this system can approve Gate A.

### F3 — external guarded process

There is intentionally no F3 command or path here. After Gate A, a separate guarded human
process may evaluate the one frozen commit/preset/assets exactly once. The outcome must be
recorded even if it is poor. No candidate substitution, threshold adjustment, or weight
tuning is allowed after viewing it.

`scripts/promote_holdout.py` is a legacy guarded script and is **not** the autonomous
campaign entry point. Do not invoke it from this workflow.

### Gate B — accept or retain

A human reviews the one-shot result, integrity log, packaging, runtime, and private-test
readiness. Gate B may accept the already frozen candidate or retain the known-good
champion. It may not select a different candidate using F3 information.

## 11. Rollback and recovery

Before a campaign, record:

- clean base commit;
- champion preset path and SHA-256;
- catalog and split hashes;
- compact asset hashes;
- campaign manifest hash;
- checkpoint and report destinations.

Rollback means retaining or re-running the known-good preset/commit. Proposal
materialization records this information and never rewrites the champion. Use reviewed,
ordinary Git operations to recover code; never let an experiment runner reset branches,
delete worktrees, merge, commit, or push.

For interrupted execution:

- specialized campaign: rerun the identical command with the same checkpoint path;
- library runner: `load_checkpoint()` validates `manifest_hash` and `run_jobs()` skips
  recorded job IDs;
- corrupted or mismatched checkpoint: keep it for diagnosis, choose a new output path,
  and restart from the frozen manifest; do not edit outcome records into a pass.

## 12. Consolidation into the unified repository

The integration worktree is the proving ground. Consolidation into `techjam-unified` is a
separate, reviewed change and is not performed by this campaign.

For every technique retained for future retesting, copy/merge all of the following:

1. implementation source and focused tests;
2. catalog entry with stable ID, availability, requirements, conflicts, mode, assets,
   evidence references, and retest triggers;
3. exhaustive binding entry, even when its disposition is anchor-only, research-only, or
   unavailable;
4. default-off typed config fields and factory/runtime wiring for composable techniques;
5. compact runtime assets and manifests, never unnecessary caches/raw labels;
6. OOF mechanism/decision reports and decision-ledger entry;
7. install extras and asset build/fetch instructions;
8. updated integrity manifest and this system reference.

After a reviewed consolidation, run from that target worktree:

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run python -m scripts.audit_unified_consolidation
```

`scripts.audit_unified_consolidation.py` currently has no CLI flags; it reads
`configs/integrity/unified_consolidation_v1.json` and writes
`artifacts/reports/unified_consolidation_audit_v1.json`. Update the integrity manifest in
the same reviewed change when adding Wave 2 components. A passing old manifest proves only
that the paths it already lists exist; it does not prove new files were consolidated.

## 13. Troubleshooting

### Freeze says the worktree is dirty

Expected behavior. Review `git status`, commit the intended inputs locally, and retry.
Do not bypass the check; otherwise `parent_commit` would not contain the frozen inputs.

### Catalog hash mismatch during planning

The catalog changed after the manifest was frozen or the wrong catalog path was supplied.
Use the catalog that matches the manifest, or create a new versioned template and freeze a
new manifest. Never edit the hash.

### Unknown/unavailable/non-runtime technique

Check `configs/techniques/catalog_v2.json` and `default_binding_registry()` in
`ghostlab/campaign/bindings.py`. Source code alone is insufficient. The technique needs an
executable catalog status, runtime execution mode, typed patch, and all required assets.

### Binding conflict

Two selected techniques patch the same config field differently. Treat the combination as
incompatible or design one explicit combined technique; do not rely on patch ordering.

### Missing model or index

Install the correct optional extra, verify/build the declared asset, and preserve the
asset manifest/hash. Do not silently fall back while recording the intended technique as
active. Component fallback must be observable.

### Job exceeds resources

Reduce the candidate resource request or raise the frozen campaign limit and refreeze.
The scheduler correctly refuses a job that cannot fit in any wave.

### Resume starts a new run or rejects a checkpoint

Confirm the exact checkpoint path and manifest hash. Specialized JSONL cache keys also
include config, sample IDs, and label; changing them creates different work rather than a
valid resume.

### Top-three selection has fewer than three candidates

This is a valid outcome. Do not weaken `confirmed`, `safe`, paired, scenario, resource, or
asset gates merely to fill three roles. Gather more legitimate F2 evidence or present no
bundle.

### A high all-development score disagrees with nested OOF

Prefer the nested OOF estimate. The all-development result was used in selection or
diagnosis and is optimistically biased. Record both, clearly labelled.

### A weak standalone could help in a combination

Keep it in reserve when evidence is uncertain, mildly negative, or unrepeated; inspect
pair interaction and pruning-audit results. Permanently prune only invalid, duplicate, or
repeatedly dominated behavior under the declared rule.

### `uv run python -m scripts.audit_unified_consolidation --help` prints an audit

The audit script does not use `argparse`; invoking it runs the audit. Use the exact command
without `--help` when an audit is intended.

## 14. Current evidence and asset index

Wave 2 compact tracked runtime assets currently present:

- `configs/assets/joint_policy_control_v1.json`
- `artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json`
- `artifacts/models/w2_ranking_v1/turn_aware_lambdamart_v1.json`
- `artifacts/models/w2_ranking_v1/fold_ensemble.json`
- `artifacts/models/w2_ranking_v1/rank_stack.json`
- `artifacts/models/w2_ranking_v1/ndcg_at_10_control.json`
- `artifacts/models/w2_ranking_v1/pointwise_control.json`
- `artifacts/models/w2_ranking_v1/uniform_pairwise_control.json`

`configs/assets/catalog_ontology_v1.json` is the tracked ontology build manifest.
`artifacts/assets/catalog_ontology_v1.json` is generated locally by the Section 5 command,
is intentionally ignored, and is required only when the catalog-normalizer technique is
admitted. The minimal smoke template does not depend on this generated cache, so it works
from a fresh clone; the full template must fail closed until required generated assets
exist.

Wave 2 evidence currently present:

- `artifacts/reports/w2_candidate_eig_f0_20.json`
- `artifacts/reports/w2_candidate_eig_f1_150.json`
- `artifacts/reports/w2_control_replay_v1.json`
- `artifacts/reports/w2_facet_mmr_replay_v1.json`
- `artifacts/reports/w2_prf_replay_v1.json`
- `artifacts/reports/w2_prf_facet_mmr_replay_v1.json`
- `artifacts/reports/w2_ranking_v1.json`
- `artifacts/reports/w2_retrieval_decision_v1.json`
- `artifacts/reports/w2_retrieval_mechanism_v1.json`
- `artifacts/reports/w2_integration_control_screen.json`
- `artifacts/evidence/wave2_policy_decisions.jsonl`

Evidence presence does not automatically imply selection. Read each report's validation
label and the catalog availability/disposition before using a technique.

## 15. Autonomy scope and remaining completion criteria

`scripts/run_autonomous_campaign.py` now provides bounded, resumable, proposal-only
development autonomy. It verifies a frozen manifest, searches multiple anchors only on
frozen search folds, confirms safe non-fit-required finalists on disjoint development
folds, checkpoints, and writes full leaderboards/evidence without expanding authority.
`scripts/materialize_campaign_top_three.py` then verifies the evidence/checkpoint/fold
boundary and creates immutable score/robust/efficient proposal presets.

The implemented end-to-end development procedure:

1. load a frozen manifest and matching catalog;
2. plan multi-anchor candidates and record skips;
3. materialize candidates through the exhaustive binding registry;
4. execute/resume F0/F1/F2 within frozen candidate, fidelity, CPU/GPU, memory, and
   heavy-model bounds; use the declared wall budget to stop higher-order expansion;
5. perform search-fold structure/HPO selection and diverse pruning audits;
6. aggregate paired disjoint-confirmation statistics and mechanism/interaction diagnostics;
7. produce deterministic, bounded score/robust/efficient proposals with hashes;
8. stop for human Gate A and provide no F3, merge, commit, push, or promotion action.

The remaining improvements are strict nested fold-local training for fit-required
techniques, optional integration of successive-halving resource allocation, and a hard
whole-campaign wall stop rather than the current bounded job counts plus higher-order wall
estimate. Even with those,
the output remains a development-confirmed proposal—not the final/generalization-proof
champion—until human Gate A, the external one-shot F3 process, and Gate B are complete.
