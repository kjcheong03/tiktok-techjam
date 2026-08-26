# Unified Technique Catalog and Operations Guide

Date: 2026-08-26
Branch: `ghostlab/unified-techniques`
Worktree: `techjam-unified/`
Base candidate: `ghostlab/integration@ec4906a`

## 1. Purpose and safety boundary

This worktree is the single research and collaboration surface for every reusable
GhostLab technique implemented during the first champion and advanced challenger
program. It contains technique code, switches, presets, dependency declarations,
asset manifests, tests, historical evidence, and commands for legal combination
testing.

It does not replace or mutate the protected checkpoints:

- `ghostlab/implementation@189f0c6`: original pairwise-linear champion;
- `ghostlab/integration@ec4906a`: validated guarded-constraint GBDT candidate;
- `main@55b3d55`: first baseline.

The submission runtime remains `ghostlab.runtime.agent.GhostLabRuntime`. Unified
research configuration is intentionally separate in
`ghostlab/research/technique_suite.py`, preventing additional experimental
defaults from changing the compiled candidate's configuration hash or behavior.

The default reproducible candidate is
`configs/suites/champion_guarded.json`. Treat every other suite as a challenger,
including presets that combine individually parked techniques.

The reason for consolidating parked code is future retesting. A technique that
lost against today's retrieval head, state representation, ranker, or question
policy may become useful after that dependency improves. “Parked” therefore means
disabled by default and evidence-preserved—not deleted, stubbed, or permanently
rejected. Every material future baseline change should trigger the dependency-based
retest review in Section 11.

The retention contract for each technique is: preserve executable source (either a
unified switch or its dedicated historical runner), a human description, dependency
group, originating branch and commit, prior evidence, test coverage, and a concrete
retest trigger. A losing result changes the default switch to off; it does not remove
the implementation.

## 2. Quick start

### 2.1 Requirements

- macOS or Linux;
- CPython 3.10 through 3.13; Python 3.12 is recommended;
- `uv` for the locked environment;
- the released `data/catalog.jsonl`;
- no API key or external service is required for the champion.

Install the small core and developer tools:

```bash
cd /path/to/tiktok-techjam
uv sync --group dev
```

Install one optional family:

```bash
uv sync --extra gbdt --group dev
uv sync --extra dense --group dev
uv sync --extra neural --group dev
```

Install everything needed for every imported technique:

```bash
uv sync --all-extras --group dev
```

The lock file pins the full transitive environment. Do not use the untracked
`.venv` from another worktree.

### 2.2 Catalog setup

Download the organizer's released `catalog.jsonl.gz`, verify it against the
release `SHA256SUMS`, and then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The catalog is intentionally ignored by Git. Historical report-integrity tests
verify its digest when it is present and otherwise continue verifying tracked
source and manifest hashes.

### 2.3 Validate the repository

```bash
uv run ruff check ghostlab scripts tests
uv run mypy ghostlab
uv run pytest -q
```

Validate every unified preset without loading a neural model:

```bash
uv run pytest -q tests/test_technique_suite.py
```

### 2.4 Run the selected candidate

```bash
uv run python -m scripts.run_unified_preset \
  --config configs/suites/champion_guarded.json \
  --output artifacts/reports/local_unified_champion.json
```

The command evaluates all 200 public sessions unless a different `--dataset`
is provided. Do not tune after inspecting a protected holdout.

## 3. Folder map

| Path | Responsibility |
|---|---|
| `baseline/` | Minimal keyword, dense, hybrid, and state baselines. |
| `starter/` | Organizer-compatible submission entry point. |
| `evaluator/` | Official local evaluator; do not change for score reporting. |
| `ghostlab/competition/` | Agent contract and competition boundary types. |
| `ghostlab/state/` | Conversation memory, invalidation, and query construction. |
| `ghostlab/retrieval/` | Sparse, dense, fusion, priors, filters, linear, GBDT, and neural ranking. |
| `ghostlab/policy/` | Fixed, heuristic, adaptive, and learned question policies plus runtime schema. |
| `ghostlab/runtime/` | Submission runtime, compiled winner, immutable historical agent, unified research agent, and normalization. |
| `ghostlab/research/` | Leakage firewall, replay, counterfactual training, technique suite, and research policies. |
| `ghostlab/evaluation/` | Frozen splits, paired statistics, and deployment audits. |
| `ghostlab/optimization/` | Bounded search, racing, evidence, patches, and meta-search. |
| `configs/techniques/` | Production technique configs and machine-readable catalog. |
| `configs/suites/` | Complete unified presets that teammates can run directly. |
| `configs/experiments/` | Immutable historical manifests declared before evaluation. |
| `configs/assets/` | Pinned local-model identities and verification information. |
| `configs/search/` | Finite, declared combination spaces. |
| `scripts/` | Asset preparation, individual challengers, unified planning, evaluation, and audits. |
| `tests/` | Contract, parity, technique, evidence, and composition tests. |
| `artifacts/models/` | Small deployable JSON models tracked in Git. |
| `artifacts/reports/` | Historical and current evaluation evidence. |
| `artifacts/cache/` | Generated/downloaded large assets; ignored by Git. |
| `docs/` | Checkpoints, technique decisions, reports, and this guide. |

## 4. Dependency groups

| Group | Direct dependencies | Techniques |
|---|---|---|
| core | `numpy`, `pydantic` | Sparse retrieval, state, queries, fixed/heuristic/learned questions, priors, filters, compiled JSON inference. |
| gbdt | core + `scikit-learn` | LambdaMART/GBDT fitting, nested refits, deployment audits. |
| dense | core + `sentence-transformers`, `huggingface-hub` | MiniLM/E5 encoding and dense indexes; PyTorch/Transformers arrive transitively. |
| neural | core + `sentence-transformers`, `torch`, `huggingface-hub` | Cross-encoder scoring and neural ranking. |
| all | gbdt + dense + neural | Dense/deep GBDT, neural GBDT, and broad combination work. |
| dev | `pytest`, `ruff`, `mypy` | Tests, formatting/lint, and static analysis. |

The champion uses JSON tree inference and does not need `scikit-learn` at
runtime. Training or refitting GBDT models does.

Optional imports occur only while building a selected component. Parsing configs,
listing techniques, and running keyword-only presets must not import
`sentence_transformers` or `torch`.

## 5. Model assets

Large models and indexes must remain under ignored `artifacts/cache/`. Never
commit `.venv`, Hugging Face cache folders, `.npy` indexes, or downloaded
catalog data.

| Asset | Revision | Local destination | Approximate size |
|---|---|---|---:|
| MiniLM dense control | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | `artifacts/cache/models/all-MiniLM-L6-v2` | model-dependent |
| E5-small-v2 | `ffb93f3bd4047442299a41ebb6fa998a38507c52` | `artifacts/cache/models/e5-small-v2` | 134,478,697 bytes |
| E5 embedding index | content-addressed by catalog and model | `artifacts/cache/dense/` | about 73 MiB |
| MiniLM embedding index | content-addressed by catalog and model | `artifacts/cache/dense/` | about 73 MiB |
| Cross-encoder MiniLM | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | `artifacts/cache/models/ms-marco-MiniLM-L6-v2` | about 87 MiB observed |
| GBDT JSON models | hashes in production config | `artifacts/models/` | about 76 KiB each |

Fetch a pinned optional asset:

```bash
uv run python -m scripts.fetch_optional_assets minilm
uv run python -m scripts.fetch_optional_assets e5
uv run python -m scripts.fetch_optional_assets cross_encoder
```

Verify already-downloaded assets without network access:

```bash
uv run python -m scripts.fetch_optional_assets e5 --verify-only
uv run python -m scripts.fetch_optional_assets cross_encoder --verify-only
```

For an alternate Hugging Face endpoint:

```bash
uv run python -m scripts.fetch_optional_assets e5 \
  --endpoint https://your-approved-endpoint.example
```

Set the following during offline evaluation:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The E5 manifest verifies every required file by size and SHA-256. MiniLM and the
cross-encoder use an exact revision plus a local acquisition receipt. If stricter
packaging is required, freeze per-file hashes before submission.

## 6. Configuration model and switches

Production configs continue using `ghostlab.policy.models.RuntimeConfig`.
Research combinations use
`ghostlab.research.technique_suite.UnifiedTechniqueConfig`.
The composition runtime is `ghostlab/runtime/unified_experimental.py`; the
historically hashed `ghostlab/runtime/experimental.py` remains unchanged.

### 6.1 Mutually exclusive selections

| Field | Values |
|---|---|
| `engine` | `compiled`, `experimental` |
| `state_variant` | `current`, `raw_history`, `single`, `multi`, `compressed` |
| `query_variant` | null, `raw_history`, `structured_active`, `category_constraints`, `raw_plus_active`, `compressed_raw`, `negation_safe_hybrid` |
| `question_variant` | `none`, `fixed`, `sequence`, `missing_priority`, `feature_first`, `uncertainty`, `other_always`, `adaptive`, `learned` |
| `retrieval_route` | `keyword`, `dense`, `rrf`, `weighted`, `sparse_first_union` |
| `dense_backend` | `off`, `minilm_control`, `e5_small_v2` |
| `reranker` | `none`, `linear`, `metadata_gbdt` |

### 6.2 Additive switches and parameters

| Field | Meaning |
|---|---|
| `negative_evidence` | Retain explicit exclusions in structured state. |
| `provenance` | Record where each preference was observed. |
| `override_invalidation` | Invalidate stale preferences after observable corrections. |
| `structured_filter` | Apply the coverage-aware constraint filter. |
| `profile_prior_weight` | Add the optional profile prior when greater than zero. |
| `quality_prior_weight` | Blend catalog quality into the candidate head. |
| `cross_encoder_enabled` | Lazily enable the pinned neural Top-K reranker. |
| `sparse_weights` | Title/category/features/details/store/description field weights. |
| `sparse_weight`, `dense_weight` | Weighted-fusion coefficients; must sum to one. |
| `rerank_k` | Candidate depth passed to the selected learned reranker. |
| `cross_encoder_weight` | Neural blend weight. |
| `cross_encoder_rerank_k` | Neural scoring depth. |

### 6.3 Compatibility enforcement

The schema rejects, before model loading:

- dense routes without a dense backend and local model path;
- keyword-only runs that accidentally configure a dense backend;
- weighted fusion whose weights do not sum to one;
- learned questions without a learned model;
- a sequence policy without a question order;
- query variants without conversation memory;
- metadata GBDT without its JSON model;
- cross-encoder fields when the cross-encoder is disabled;
- absolute paths or `..` paths escaping the repository.

A disabled neural technique requires no model, no import, no network call, and no
initialization.

## 7. Technique inventory

The machine-readable source of truth is
`configs/techniques/catalog_v1.json`. Historical results and decisions are in
`docs/technique_registry_and_retest_guide.md` and
`artifacts/evidence/technique_decisions.jsonl`.

### 7.1 State and query

| Technique | Switch | Source | Status |
|---|---|---|---|
| Current turn only | `state_variant=current` | `runtime/experimental.py` | Negative control |
| Single state | `single` | `state/memory.py` | Control |
| Multi-value state | `multi` | `state/memory.py` | Available |
| Raw history | `raw_history` | `state/memory.py` | Promoted |
| Compressed state | `compressed` | `state/memory.py` | Parked |
| Structured query variants | `query_variant=...` | `state/query.py` | Available for retest |
| Dense-specific structured query | historical dense interaction script | `retrieval/query.py` | Parked |

### 7.2 Questions

| Technique | Switch | Extra | Status/result |
|---|---|---|---|
| No questions | `question_variant=none` | core | Essential control |
| Fixed organizer questions | `fixed` | core | Control |
| Selected sequence | `sequence` + `question_order` | core | Promoted |
| Missing/feature/uncertainty heuristics | named variants | core | Available |
| Other-always | `other_always` | core | Strong early control |
| Observable adaptive heuristic | `adaptive` | core | Available |
| Learned counterfactual policy | `learned` + asset | core | Parked; `0.808951` standalone and `0.847744` with GBDT |

The learned model is small JSON and tracked. The large raw counterfactual training
tables remain on `exp/learned-question-policy` and
`exp/interaction-gbdt-question`; they can also be regenerated by the preserved
scripts. They are not runtime dependencies.

### 7.3 Retrieval and fusion

| Technique | Switch | Extra | Status/result |
|---|---|---|---|
| Organizer keyword baseline | baseline path | core | `0.106710` |
| Field-aware FTS5 BM25 | keyword + `sparse_weights` | core | Promoted |
| MiniLM dense | `dense_backend=minilm_control` | dense | Control |
| E5-small-v2 | `e5_small_v2` | dense | Parked; recall diagnostics preserved |
| Dense-only | `retrieval_route=dense` | dense | Available |
| RRF | `rrf` | dense | Available |
| Weighted fusion | `weighted` | dense | Available |
| Sparse-first union | `sparse_first_union` | dense | Parked interaction path |
| Dense + structured-query E2E | dedicated E2E runner/preset fields | dense | Parked interaction; matched report preserved |

### 7.4 Ranking, filtering, and guards

| Technique | Switch/path | Extra | Status/result |
|---|---|---|---|
| No reranker | `reranker=none` | core | Control |
| Fixed lexical linear | `reranker=linear` | core | Available |
| Pairwise learned linear | historical compiled/config path | core | Original champion `0.817649` |
| Metadata GBDT | `metadata_gbdt` + model asset | gbdt for fitting | Fallback `0.861417` |
| Constraint GBDT + observable guard | compiled champion | core inference | Selected OOF `0.878963` |
| Deep dense-candidate GBDT | dedicated script/module | all | Parked `0.829423` |
| Cross-encoder | additive neural fields | neural | Parked standalone `0.790915` |
| Neural-score GBDT | dedicated script/module | all | Parked `0.858094` |
| Coverage-aware filter | `structured_filter=true` | core | Parked |
| Profile prior | positive weight | core | Parked |
| Catalog quality | positive weight | core | Promoted at `0.2` |
| Negative evidence/invalidation | Boolean switches | core | Retained |
| Observable override fallback | compiled champion | core | Selected safety guard |

### 7.5 Routing, search, and research infrastructure

| Technique | Source | Status and future use |
|---|---|---|
| Observable decision list | `policy/decision_list.py` | Available for state-conditioned actions. |
| Route stump | `research/route_stump.py` | Parked after collapsing to sparse; retest after route evidence changes. |
| Route table | `research/route_policy.py` | Available fold-local route learner. |
| Counterfactual evaluator | `research/counterfactual.py` | Retained for new question/action spaces. |
| Deterministic replay | `research/replay.py` | Required evaluation infrastructure. |
| Leakage firewall | `research/firewall.py` | Required runtime-feature boundary. |
| Random/grid/beam search | `optimization/search.py` | Promoted auditable search control. |
| Multi-fidelity racing | `optimization/racing.py` | Available with pruning-regret audit. |
| Evidence allocator | `optimization/meta_search.py` | Parked after losing equal-budget comparison. |
| Family UCB allocation | `optimization/evidence.py` | Parked allocation alternative. |
| Typed patches and crossover | `optimization/patches.py` | Available for compatible policy composition. |
| Evidence/decision stores | `optimization/evidence.py` | Required chronological research record. |
| Grouped splits and paired statistics | `evaluation/` | Required anti-overfitting infrastructure. |

Research-only deep/constraint/neural interactions have dedicated fold-local
training scripts because no deployable model was promoted. They remain fully
available, but the unified runtime does not pretend that an unfitted model can be
turned on.

## 8. Presets

| Preset | Purpose |
|---|---|
| `champion_guarded.json` | Exact compiled selected candidate; default reference. |
| `keyword_research.json` | Keyword/raw-history/sequence/metadata-GBDT research control. |
| `learned_questions.json` | Learned questioning with metadata GBDT. |
| `dense_e5.json` | Sparse-first E5 union with metadata GBDT. |
| `cross_encoder.json` | Metadata GBDT followed by bounded cross-encoder. |
| `all_composable.json` | Explicit unpromoted combination of learned questions, structured query, E5 union, metadata GBDT, and cross-encoder. |

The last preset exists to make interaction testing easy. It is not claimed to be
better and must not replace the champion without nested validation and safety
gates.

Run any preset:

```bash
uv run python -m scripts.run_unified_preset \
  --config configs/suites/keyword_research.json \
  --output artifacts/reports/local_keyword_research.json
```

## 9. Combination planning and execution

The declared starter space is `configs/search/unified_space_v1.json`.

Inspect legal/rejected counts:

```bash
uv run python -m scripts.plan_unified_combinations
```

Materialize every legal candidate as a separate immutable config:

```bash
uv run python -m scripts.plan_unified_combinations \
  --output-dir artifacts/campaigns/unified_space_v1
```

Evaluate generated candidates individually:

```bash
for config in artifacts/campaigns/unified_space_v1/*.json; do
  name="$(basename "$config" .json)"
  uv run python -m scripts.run_unified_preset \
    --config "$config" \
    --output "artifacts/campaigns/unified_space_v1/${name}_result.json"
done
```

Add dense, learned-question, or neural dimensions by deriving a new versioned
space from the corresponding preset. Keep fields that depend on one another
coupled: route/backend/model path, learned policy/model path, and neural
enable/model path/weight.

Do not mutate `unified_space_v1.json` after seeing results. Create
`unified_space_v2.json` with a stated reason.

## 10. Dedicated historical experiment commands

These scripts retain fold-local fitting, matched controls, and their original
manifests:

```bash
uv run python -m scripts.run_query_challenger
uv run python -m scripts.run_learned_question_challenger
uv run python -m scripts.run_dense_retrieval
uv run python -m scripts.run_dense_query_e2e
uv run python -m scripts.run_dense_query_interaction
uv run python -m scripts.run_gbdt_dense_interaction
uv run python -m scripts.run_cross_encoder_challenger
uv run python -m scripts.run_gbdt_question_interaction
uv run python -m scripts.run_neural_rank_interaction
```

Read the corresponding `configs/experiments/*.json` before running. Some jobs
take substantial time and require prepared model assets. Never rewrite the
original report after changing code; write a new manifest and report version.

## 11. Validation and anti-overfitting protocol

A candidate is not selected merely because its public aggregate score is highest.

1. Freeze a manifest before evaluating candidate outcomes.
2. Split by complete session, never by turn.
3. Fit learned components strictly inside outer-training folds.
4. Use inner folds for parameter/model selection.
5. Stitch only outer-fold predictions into the OOF result.
6. Include an exact matched control at the same candidate depth.
7. Report paired per-session deltas, five fold deltas, and scenario deltas.
8. Report Hit@10, MRR, MTTC, technical score, failures, latency, memory, and assets.
9. Require backward ablations for combinations.
10. Treat repeated public-set searching as selection pressure.
11. Preserve parked techniques for later retesting when dependencies change.
12. Keep F3/private evaluation sealed until one complete candidate is frozen.
13. Record the protected result even if it is negative.
14. Make no parameter or policy changes after protected-holdout access.

When a combination wins, rerun each component-off ablation. This distinguishes a
real interaction from a passenger technique that adds complexity without value.

### 11.1 Dependency-based retest review

After improving the baseline, do not rerun only techniques that previously won.
Compare the changed component against every catalog entry whose stated dependency
it affects:

- a better dense model reopens fusion, dense queries, deep GBDT, and route policies;
- better state parsing reopens structured queries, filtering, profile gating,
  constraint features, and learned questions;
- a better sparse head reopens candidate-depth, filter, and reranker interactions;
- a better ranker reopens dense, neural-score, profile, and question-policy
  interactions;
- new observable uncertainty features reopen adaptive questioning and routing;
- more independent sessions reopen higher-capacity learned models and interactions;
- different hardware or latency budgets reopen cross-encoder depth and neural
  combinations;
- a changed search space reopens racing and evidence allocation.

Run the old preset unchanged first, then the updated dependency, then their
combination, then both backward ablations. Record the new decision as a new version;
never overwrite the old result.

## 12. Adding a technique

1. Add the smallest cohesive module under the relevant `ghostlab/` family.
2. Avoid changing the official evaluator or contract.
3. Add a stable technique ID to `configs/techniques/catalog_v1.json` or a new
   catalog version.
4. Add a typed switch or an explicitly research-only runner.
5. Put heavy imports inside the selected factory.
6. Add a pinned asset manifest if a model is required.
7. Add an immutable experiment manifest before evaluation.
8. Add unit, compatibility, disabled-boundary, and report-integrity tests.
9. Add a preset or search-space dimension.
10. Update this guide and the chronological technique decision ledger.
11. Confirm the champion preset still has exact parity.
12. Commit code, manifest, tests, and decision together.

Prefer minimal modules and explicit functions over frameworks, dynamic plugin
discovery, global singletons, or large conditional blocks.

## 13. Historical branches and raw evidence

All technique source needed for use and retesting is consolidated here. Original
branches remain useful for forensic recovery:

| Branch | Main prototype |
|---|---|
| `exp/query-construction` | Query construction |
| `exp/learned-question-policy` | Learned question policy and raw labels |
| `exp/dense-retrieval` | E5/MiniLM retrieval and query interaction |
| `exp/gbdt-reranker` | Metadata LambdaMART |
| `exp/cross-encoder` | Standalone cross-encoder |
| `exp/interaction-gbdt-constraints` | Constraint GBDT and guard |
| `exp/interaction-gbdt-dense` | Dense/deep GBDT |
| `exp/interaction-gbdt-question` | Learned-question/GBDT interaction and raw labels |
| `exp/interaction-neural-rank` | Neural-score/GBDT interaction |
| `exp/compile-guarded-gbdt` | Compiled selected runtime |

The large raw counterfactual label tables were deliberately not duplicated in the
unified branch. They remain content-addressed by hashes in reports and recoverable
from their branches:

```bash
git show exp/learned-question-policy:artifacts/experiments/learned_question_linear_v1/counterfactual_labels.jsonl
git show exp/interaction-gbdt-question:artifacts/experiments/gbdt_question_interaction_v1/counterfactual_labels.jsonl
```

Runtime models, OOF session summaries, manifests, scripts, and tests are included.

## 14. Git and teammate workflow

Push the unified branch:

```bash
cd /path/to/tiktok-techjam
git push -u origin ghostlab/unified-techniques
```

A teammate can then clone once and check out the unified branch:

```bash
git clone https://github.com/kjcheong03/tiktok-techjam.git
cd tiktok-techjam
git switch ghostlab/unified-techniques
uv sync --all-extras --group dev
```

Do not nest old worktrees inside this repository. After the unified branch is
pushed and verified, local experimental worktrees may be removed without deleting
their branches. Confirm each is clean first.

## 15. Recovery and promotion

Recovery order:

1. selected unified default: `configs/suites/champion_guarded.json`;
2. validated integration fallback: metadata GBDT, OOF `0.861417`;
3. immutable original champion: `ghostlab/implementation@189f0c6`;
4. first baseline: `main@55b3d55`.

Promotion requires exact official-contract replay, zero parity mismatches,
dependency-boundary checks, integrity hashes, full tests, runtime gates, paired
OOF evidence, and a documented decision. A unified preset alone is not a
promotion.

## 16. Consolidation verification snapshot

The repository-level audit is reproducible with:

```bash
uv run python -m scripts.audit_unified_consolidation
```

The 2026-08-26 audit records 40 technique/infrastructure entries, 12 imported
component families, six directly runnable presets, no missing required paths, and
no runtime dependency on the two archived raw counterfactual tables. Its
machine-readable output is
`artifacts/reports/unified_consolidation_audit_v1.json`.

The finite keyword starter search contains 864 schema-valid candidates. Another
864 Cartesian products are deliberately rejected because the metadata-GBDT model
asset must be coupled to that reranker and absent for the `none` and fixed-lexical
controls. This is compatibility filtering, not score-based pruning. Dense, learned
question, and neural families remain available through their named presets and
versioned follow-on spaces.

The protected guarded runtime was replayed with zero mismatches, and the unified
champion preset completed all 200 public sessions at technical score `0.896852`.
This full-public score is a reproducibility check, not unbiased evidence for further
tuning. The selected OOF decision remains documented separately from public replay.
