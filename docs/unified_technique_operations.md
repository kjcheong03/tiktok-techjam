# Unified Technique Catalog and Operations Guide

New teammates should begin with `docs/essentials/README.md`, which provides the
curated reading order and execution checklist while linking back to this canonical
technical reference.

The teammate State Baseline V2 is now a native, opt-in state/query/history family. Its
complete mapping, exact parity hashes, presets, interaction evidence, and safe retest
commands are in `docs/state_baseline_v2_integration.md`.

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

This file is the authoritative entry point for the entire system: architecture,
lineage, active champion, inactive controls, every challenger, exact file locations,
reproduction commands, evidence, dependencies, compatibility, and retest policy.
Supporting documents and JSON reports remain immutable evidence; a teammate should
be able to start here without knowing which historical worktree created a technique.

For the implemented autonomous campaign engine, its fairness invariants, exact freeze/run/
resume/proposal commands, pruning and HPO behavior, overfitting controls, and human gates,
use `docs/autonomous_unified_system_reference.md`. Autonomous discovery starts from
`configs/suites/unfitted_keyword_search.json`: current-turn state, fixed starter questions,
keyword retrieval only, no dense/hybrid path, no reranker, zero profile/quality priors,
and no negative-evidence/provenance/override logic. Historical champions and challengers
are controls only; they do not seed the search.

It does not replace or mutate the protected checkpoints:

- `ghostlab/implementation@189f0c6`: original pairwise-linear champion;
- `ghostlab/integration@ec4906a`: validated guarded-constraint GBDT candidate;
- `main@55b3d55`: first baseline.

`starter.Agent` now enters through `ghostlab.runtime.selected.SelectedRuntime`. With no
`configs/active_candidate.json` pointer it delegates to the exact compiled
`ghostlab.runtime.agent.GhostLabRuntime`, preserving the current champion. Only the
separate hash-bound human activation command selects a validated unified preset; runtime
failure falls back to the compiled champion.

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

### 2.5 Run the complete autonomous search

After reviewing and locally committing the implementation, run:

```bash
uv run python -m scripts.run_autonomous_end_to_end --prepare-assets
```

This prepares/verifies pinned optional models, accounts for every catalog technique,
freezes the versioned `autonomous_state_v2_v1` campaign, runs or resumes the bounded
pure-baseline F0/F1/F2 search, and prints three candidate-preparation commands. It does
not activate, commit, push, or access F3. See Section 16 of
`docs/autonomous_unified_system_reference.md` for the eight user steps, exact files,
combination/pruning algorithm, overfitting controls, activation, and rollback.

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

## 17. Configuration namespaces and execution classes

Two typed configuration schemas coexist intentionally. Do not copy a field from one
schema into the other.

| Context | Schema and factory | Field examples | Canonical configuration |
|---|---|---|---|
| Official/compiled runtime | `ghostlab/policy/models.py`; `ghostlab/runtime/agent.py` | `state_mode`, `question_policy`, `sparse_field_weights`, `reranker` | `configs/compiled_policy.json` |
| Unified research composer | `ghostlab/research/technique_suite.py`; `ghostlab/runtime/unified_experimental.py` | `state_variant`, `question_variant`, `sparse_weights`, `reranker` | `configs/suites/*.json` |
| Fold-local challenger | Dedicated `scripts/run_*.py` runner | Manifest-specific fields frozen before fitting | `configs/experiments/*.json` |

The registries below use these execution classes:

- **U**: directly composable through `UnifiedTechniqueConfig` and a suite JSON;
- **P**: active or switchable through production `RuntimeConfig`;
- **H**: preserved historical/fold-local runner; retraining is required before use;
- **I**: evaluation, search, or evidence infrastructure rather than a recommendation
  component.

A technique may have more than one class. For example, learned questions are **U**
for inference with the preserved JSON model and **H** for leakage-safe refitting.

## 18. End-to-end lineage and champion decision

This is the chronological path that produced the current candidate. Scores labelled
OOF are the comparable 150-session grouped out-of-fold estimates; all-development or
200-public scores are reproducibility measurements, not independent selection
evidence.

| Stage | What changed | Exact implementation/evidence | Comparable result and decision |
|---|---|---|---|
| Organizer baseline | Weak BM25, no robust dialogue state | `baseline/official_reference.py`; `scripts/run_baselines.py`; `artifacts/baseline_results.json` | Technical score `0.106710`; retained as the floor. |
| First state/dialogue baseline | Single/multi state, negatives, provenance, invalidation, basic questioning | `ghostlab/state/memory.py`; `ghostlab/runtime/experimental.py`; `scripts/run_policy_ablations.py`; `artifacts/reports/phase4_5_summary.json` | Established that clarification and state materially help. |
| Retrieval controls | Keyword, MiniLM dense, RRF and weighted fusion | `ghostlab/retrieval/sparse.py`; `ghostlab/retrieval/dense.py`; `ghostlab/retrieval/fusion.py`; `scripts/run_retrieval_diagnostics.py`; `artifacts/reports/phase3_retrieval.json` | MiniLM/fusion lost end to end but showed some unique recall; parked. |
| Raw-history control | Preserve all normalized user turns instead of replacing them with a summary | `ghostlab/state/memory.py`; `artifacts/reports/phase13_targeted_adaptive.json` | Raw history `0.753736` beat structured alternatives; promoted. |
| Static question sequence | Search fixed, heuristic and stopping variants | `ghostlab/runtime/experimental.py`; `scripts/run_question_policy.py`; `artifacts/reports/phase9_question_policy.json` | Selected `other, other, use_case, other, size, other, other, size`; static control `0.800591` after later rechecks. |
| Field-aware sparse retrieval | Tune six BM25 fields and interactions | `ghostlab/retrieval/sparse.py`; `scripts/run_field_weights.py`; `scripts/run_field_interactions.py`; `artifacts/reports/phase16_field_weights.json`; `artifacts/reports/phase17_field_interactions.json` | Promoted Top-200 generator. |
| Quality prior | Add bounded catalog-quality tie-breaking | `ghostlab/retrieval/quality.py`; `scripts/run_quality_priors.py`; `artifacts/reports/phase18_quality_priors.json` | Weight `0.2` promoted. |
| Pairwise linear champion | Fold-safe learned lexical/quality ranking | `ghostlab/retrieval/learned.py`; `scripts/run_learned_reranker.py`; `artifacts/reports/phase19_learned_reranker.json`; `artifacts/reports/phase20_learned_features.json` | OOF `0.817649`; original champion `189f0c6`. |
| Query challenger | Raw, structured, hybrid and negation-safe queries | `ghostlab/state/query.py`; `scripts/run_query_challenger.py`; `configs/experiments/challenger_query_v1.json` | Raw and raw+active tied at `0.800591`; structured variants parked. |
| Learned-question challenger | Counterfactual linear action values over legal next questions/stop | `ghostlab/policy/learned_questions.py`; `scripts/run_learned_question_challenger.py` | `0.808951`; parked. |
| Retrieval-specialized dense challenger | Pinned E5, dense queries and sparse/dense unions | `ghostlab/retrieval/dense.py`; dense runners/manifests | Exploratory `0.835125` was confounded; matched tests later rejected promotion. |
| Metadata GBDT | Shallow fold-local nonlinear reranking | `ghostlab/retrieval/gbdt.py`; `scripts/run_gbdt_reranker.py`; `artifacts/reports/gbdt_reranker_v1.json` | OOF `0.861417`, positive in 5/5 folds; promoted fallback. |
| Cross-encoder | Pinned MiniLM neural Top-K scoring | `ghostlab/retrieval/cross_encoder.py`; `scripts/run_cross_encoder_challenger.py` | Standalone `0.790915`; parked. |
| Dense + deep GBDT | E5/deeper candidate heads with GBDT | `ghostlab/retrieval/gbdt_dense.py`; `scripts/run_gbdt_dense_interaction.py` | `0.829423`, lost to matched Top-50 GBDT in 5/5 folds; parked. |
| Learned questions + GBDT | Test whether nonlinear ranking changes question value | `ghostlab/runtime/experimental_questions.py`; `scripts/run_gbdt_question_interaction.py` | `0.847744`, negative in 5/5 folds; parked. |
| Cross-encoder + GBDT | Add neural score as a ranker feature | `ghostlab/retrieval/neural_rank.py`; `scripts/run_neural_rank_interaction.py` | `0.858094`, below GBDT and warm p95 `525.367 ms`; parked. |
| Constraint GBDT v1 | Add observable conversation-constraint features | `ghostlab/retrieval/constraint_gbdt.py`; constraint runner/report | Apparent `0.884943` invalidated by stale override state and train/runtime mismatch; never use as promotion evidence. |
| Corrected constraint GBDT v2 | Scoped invalidation and bookkeeping/runtime repairs | `ghostlab/retrieval/constraint_gbdt.py`; `configs/experiments/gbdt_constraint_interaction_v1_amendment_1.json`; `artifacts/reports/gbdt_constraint_interaction_v2.json` | `0.876283`; failed Hit@10 and intent-override gates; parked. |
| Guarded constraint GBDT | Use constraint model normally and matched base GBDT after observable invalidation | `ghostlab/runtime/guarded_gbdt.py`; `scripts/run_gbdt_constraint_override_guard.py`; `configs/experiments/gbdt_constraint_interaction_v1_amendment_2.json` | OOF `0.878963`; all frozen gates passed; selected with borderline CI caveat. |
| Compiled selected candidate | Freeze models, official runtime and starter adapter | `ghostlab/runtime/agent.py`; `configs/compiled_policy.json`; `scripts/validate_guarded_compiled.py` | Exact parity across 150 sessions/349 turns; integration checkpoint `ec4906a`. |
| Unified preservation branch | Consolidate all reusable source and historical evidence | this guide; `configs/techniques/catalog_v1.json`; consolidation audit | Does not change the selected policy; enables future controlled retesting. |

## 19. Complete component and technique registry

Every row provides the exact implementation and operational path. Prefix a runner
with `uv run python -m`, for example
`uv run python -m scripts.run_query_challenger`. Historical reports must never be
overwritten; create a versioned manifest/report for a new run.

### 19.0 Machine-ID coverage index

This index mirrors every entry in `configs/techniques/catalog_v1.json`. Detailed
behavior, activation, evidence and retest guidance follow in Sections 19.1–19.6.

| Technique ID | Exact implementation | Current status |
|---|---|---|
| `state.current` | `ghostlab/runtime/experimental.py` | Control |
| `state.raw_history` | `ghostlab/state/memory.py` | Promoted |
| `state.multi` | `ghostlab/state/memory.py` | Available |
| `state.compressed` | `ghostlab/state/memory.py` | Parked |
| `query.structured` | `ghostlab/state/query.py` | Parked/retestable |
| `question.fixed` | `ghostlab/runtime/experimental.py` | Control |
| `question.adaptive_heuristic` | `ghostlab/policy/adaptive_questions.py` | Available/parked by current evidence |
| `question.learned_linear` | `ghostlab/policy/learned_questions.py` | Parked/retestable |
| `retrieval.sparse` | `ghostlab/retrieval/sparse.py` | Promoted |
| `retrieval.minilm` | `ghostlab/retrieval/dense.py` | Dense control |
| `retrieval.e5` | `ghostlab/retrieval/dense.py` | Parked/retestable |
| `fusion.rrf` | `ghostlab/retrieval/fusion.py` | Available |
| `fusion.weighted` | `ghostlab/retrieval/fusion.py` | Available |
| `fusion.sparse_first_union` | `ghostlab/retrieval/fusion.py` | Parked/retestable |
| `ranking.fixed_lexical` | `ghostlab/retrieval/rerank.py` | Available control |
| `ranking.pairwise_linear` | `ghostlab/retrieval/learned.py` | Original champion |
| `ranking.metadata_gbdt` | `ghostlab/retrieval/gbdt.py` | Validated fallback |
| `ranking.constraint_gbdt` | `ghostlab/retrieval/constraint_gbdt.py` | Selected with guard |
| `ranking.deep_dense_gbdt` | `ghostlab/retrieval/gbdt_dense.py` | Parked research |
| `ranking.cross_encoder` | `ghostlab/retrieval/cross_encoder.py` | Parked/retestable |
| `ranking.neural_gbdt` | `ghostlab/retrieval/neural_rank.py` | Parked research |
| `filter.structured` | `ghostlab/retrieval/filters.py` | Parked/retestable |
| `prior.profile` | `ghostlab/retrieval/profile.py` | Parked/retestable |
| `prior.quality` | `ghostlab/retrieval/quality.py` | Promoted |
| `guard.override_fallback` | `ghostlab/runtime/guarded_gbdt.py` | Selected |
| `routing.decision_list` | `ghostlab/policy/decision_list.py` | Available |
| `routing.observable_stump` | `ghostlab/research/route_stump.py` | Parked/retestable |
| `routing.route_table` | `ghostlab/research/route_policy.py` | Parked/retestable |
| `research.counterfactual` | `ghostlab/research/counterfactual.py` | Available infrastructure |
| `research.replay` | `ghostlab/research/replay.py` | Required infrastructure |
| `research.leakage_firewall` | `ghostlab/research/firewall.py` | Required infrastructure |
| `search.random_grid_beam` | `ghostlab/optimization/search.py` | Promoted control |
| `search.multifidelity_racing` | `ghostlab/optimization/racing.py` | Available |
| `search.evidence_allocator` | `ghostlab/optimization/meta_search.py` | Parked/retestable |
| `search.family_ucb` | `ghostlab/optimization/evidence.py` | Parked/retestable |
| `search.typed_patches` | `ghostlab/optimization/patches.py` | Available |
| `search.crossover` | `ghostlab/optimization/patches.py` | Interaction reserve |
| `evidence.decision_store` | `ghostlab/optimization/evidence.py` | Required infrastructure |
| `evaluation.grouped_splits` | `ghostlab/evaluation/splits.py` | Required infrastructure |
| `evaluation.paired_statistics` | `ghostlab/evaluation/statistics.py` | Required infrastructure |

First-version `single` state, individual question heuristics, no-question,
quality-only and invalid constraint-v1 controls are also documented below even
though they are not separate entries in catalog v1.

### 19.1 State, evidence and query construction

| Technique | Class and activation | Purpose | Source | Runner/config/tests/evidence | Status and retest rule |
|---|---|---|---|---|---|
| Current turn only | U: `state_variant=current`; P fallback | Stateless negative control | `ghostlab/runtime/experimental.py` | `configs/techniques/baseline_v1.json`; `tests/test_runtime.py`; `scripts/run_policy_ablations.py` | Weak control; retain permanently. |
| Single-value state | U: `single`; P: `state_mode=single` | One active value per attribute | `ghostlab/state/memory.py` | `configs/techniques/baseline_v1.json`; `tests/test_state_hardening.py`; `artifacts/reports/phase4_5_summary.json` | Replaced by raw/multi; retest parser interactions. |
| Multi-value state | U: `multi`; P: `state_mode=multi` | Preserve multiple positive/negative constraints | `ghostlab/state/memory.py` | `configs/techniques/manual_strong_v1.json`; `tests/test_state_hardening.py`; `artifacts/reports/phase4_5_summary.json` | Interaction reserve for learned questions, filters and queries. |
| Raw history | U: `raw_history`; P: `state_mode=raw_history` | Preserve discriminative lexical evidence from every turn | `ghostlab/state/memory.py` | `configs/suites/keyword_research.json`; `configs/compiled_policy.json`; `tests/test_state_hardening.py`; `artifacts/reports/phase13_targeted_adaptive.json`; `artifacts/reports/phase21_cross_phase_rechecks.json` | Promoted default; rerun after normalization changes. |
| Compressed state | U: `compressed`; P: `state_mode=compressed` | Compact active constraints into a query | `ghostlab/state/memory.py` | `scripts/run_query_challenger.py`; `tests/test_query_construction.py`; `tests/test_state_hardening.py`; `artifacts/reports/challenger_query_v1.json` | Parked after losing to raw history; retest with a new parser/model. |
| Negative evidence | U/P: `negative_evidence=true` | Retain explicit exclusions instead of treating them as positive terms | `ghostlab/state/memory.py` | `scripts/run_policy_ablations.py`; `tests/test_state_hardening.py`; `artifacts/reports/phase4_5_ablations.json` | Retained; retest with filters and constraint features. |
| Provenance | U/P: `provenance=true` | Record which turn/profile supplied a preference | `ghostlab/state/memory.py` | `tests/test_state_hardening.py`; `artifacts/reports/phase4_5_summary.json`; `artifacts/reports/phase4_5_ablations.json` | Retained safety/context signal. |
| Override invalidation | U/P: `override_invalidation=true` | Invalidate stale scoped preferences after observable corrections | `ghostlab/state/memory.py` | `tests/test_state_hardening.py`; `tests/test_gbdt_constraint_override_guard_report.py`; `artifacts/reports/gbdt_constraint_override_guard_v1.json` | Required by guarded candidate; re-audit after state changes. |
| Structured query family | U: `query_variant=structured_active`, `category_constraints`, `raw_plus_active`, `compressed_raw`, or `negation_safe_hybrid` | Convert state into bounded query forms without target labels | `ghostlab/state/query.py` | `scripts/run_query_challenger.py`; `configs/experiments/challenger_query_v1.json`; `artifacts/reports/challenger_query_v1.json`; `tests/test_query_construction.py` | Mostly parked; raw+active reserve. Retest after state, dense or ranker changes. |
| Dense-specific query builder | H: dense runner configuration | Add E5 query prefixes and structured dense text | `ghostlab/retrieval/query.py` | `scripts/run_dense_query_e2e.py`; `scripts/run_dense_query_interaction.py`; `configs/experiments/dense_query_e2e_v1.json`; `configs/experiments/dense_query_interaction_v1.json`; `tests/test_dense_query.py`; `tests/test_dense_query_interaction.py` | Parked dependency path; retest with a better dense model. |

### 19.2 Question and stopping policies

| Technique | Class and activation | Purpose/decision rule | Source | Runner/config/tests/evidence | Status and retest rule |
|---|---|---|---|---|---|
| No questions | U: `question_variant=none`; P: `question_policy=none` | Stop immediately; isolates value of clarification | `ghostlab/runtime/experimental.py` | `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json`; `artifacts/reports/phase21_cross_phase_rechecks.json` | Negative control; not a submission policy. |
| Organizer fixed | U: `fixed`; P: `fixed` | Ask organizer-defined fields in fixed order | `baseline/state.py`; dispatch in `ghostlab/runtime/experimental.py` and `ghostlab/runtime/unified_experimental.py` | `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json` | Control. |
| Selected static sequence | U: `sequence` + `question_order`; P equivalent | Ask the validated eight-action sequence, bounded by available turns | `ghostlab/runtime/experimental.py` | `configs/suites/keyword_research.json`; `configs/compiled_policy.json`; `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json`; `artifacts/reports/phase21_cross_phase_rechecks.json` | Promoted dialogue controller and champion default. |
| Missing-priority | U: `missing_priority`; P equivalent | Ask the highest-priority missing attribute | `ghostlab/runtime/experimental.py` | `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json` | Available control. |
| Feature-first | U: `feature_first`; P equivalent | Prioritize product features before broader questions | `ghostlab/runtime/experimental.py` | `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json` | Parked standalone. |
| Uncertainty-limited | U: `uncertainty`; P equivalent | Ask only while observable candidate uncertainty remains | `ghostlab/runtime/experimental.py` | `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json` | Available; reopen with improved uncertainty signals. |
| Other-always | U: `other_always`; P equivalent | Repeated broad clarification control | `ghostlab/runtime/experimental.py` | `scripts/run_question_policy.py`; `tests/test_question_policy.py`; `artifacts/reports/phase9_question_policy.json` | Strong early control, superseded by sequence. |
| Heuristic adaptive | U: `question_variant=adaptive` | Score legal next attributes from missing fields, candidate uncertainty, turn budget and repeated-question state | `ghostlab/policy/adaptive_questions.py`; dispatch in `ghostlab/runtime/unified_experimental.py` | `scripts/run_targeted_adaptive.py`; `tests/test_adaptive_questions.py`; `artifacts/reports/phase13_targeted_adaptive.json`; included in `configs/search/unified_space_v1.json` | `0.740291` versus `0.753736`; parked but directly switchable. Retest after state/uncertainty improvements. |
| Learned counterfactual | U inference: `learned` + `learned_question_asset`; H refit | Fit fold-local linear action values for every legal question and stop action | `ghostlab/policy/learned_questions.py`; `ghostlab/research/learned_questions.py`; `ghostlab/runtime/unified_experimental.py` | `configs/suites/learned_questions.json`; `configs/experiments/learned_question_linear_v1.json`; `artifacts/reports/learned_question_linear_v1.json`; `scripts/run_learned_question_challenger.py`; `tests/test_learned_questions.py`; `artifacts/experiments/learned_question_linear_v1/linear_action_value_model.json` | `0.808951`; parked. Retest only with new observable value features, objective, continuation, or independent data. |
| Learned questions + GBDT | H only | Test interaction between learned dialogue actions and nonlinear ranker | `ghostlab/runtime/experimental_questions.py` | `scripts/run_gbdt_question_interaction.py`; `configs/experiments/gbdt_question_interaction_v1.json`; `artifacts/reports/gbdt_question_interaction_v1.json`; `tests/test_gbdt_question_interaction.py`; tracked GBDT-question action-value model | `0.847744` vs `0.861417`, negative 5/5 folds; dependency-gated. |

The selected champion uses the static sequence, not either adaptive policy. Both
adaptive implementations remain executable for future retesting.

### 19.3 Retrieval, dense models and fusion

| Technique | Class and activation | Purpose | Source | Runner/config/tests/evidence | Status and retest rule |
|---|---|---|---|---|---|
| Organizer keyword baseline | baseline runner | Reproduce the starter retrieval floor | `baseline/official_reference.py`; `baseline/retrieval.py` | `scripts/run_baselines.py`; `tests/test_baseline.py`; baseline report | `0.106710`; permanent floor. |
| Field-aware FTS5 BM25 | U: `retrieval_route=keyword` + `sparse_weights`; P: `sparse_field_weights` | Generate Top-200 from title/category/features/details/store/description | `ghostlab/retrieval/sparse.py` | `scripts/run_field_weights.py`; `scripts/run_field_interactions.py`; `tests/test_retrieval_contract.py`; `artifacts/reports/phase16_field_weights.json`; `artifacts/reports/phase17_field_interactions.json` | Promoted retrieval head. |
| MiniLM dense control | U: `dense_backend=minilm_control` plus local model path | Generic semantic-retrieval control | `ghostlab/retrieval/dense.py` | `configs/assets/minilm_control.json`; `scripts/fetch_optional_assets.py`; `scripts/run_dense_retrieval.py`; `tests/test_dense_retrieval.py`; `artifacts/reports/phase3_retrieval.json`; `artifacts/reports/dense_retrieval_v1.json` | Weak standalone; preserve unique-recall control. |
| E5-small-v2 | U: `dense_backend=e5_small_v2` plus local model path | Retrieval-specialized semantic embeddings and catalog index | `ghostlab/retrieval/dense.py` | `configs/assets/e5_small_v2.json`; `configs/suites/dense_e5.json`; `scripts/fetch_dense_assets.py`; `scripts/run_dense_retrieval.py`; `configs/experiments/dense_retrieval_v1.json`; `artifacts/reports/dense_retrieval_v1.json`; `tests/test_dense_retrieval.py` | Parked. Require stable unique recall beyond BM25 before promotion tests. |
| Dense-only | U: `retrieval_route=dense` | Isolate dense candidate quality | `ghostlab/retrieval/dense.py` | derive a suite from `configs/suites/dense_e5.json`; `tests/test_dense_retrieval.py`; `artifacts/reports/dense_retrieval_v1.json` | Control, not selected. |
| Reciprocal-rank fusion | U: `retrieval_route=rrf` | Combine sparse/dense ranks without score calibration | `ghostlab/retrieval/fusion.py` | `scripts/run_retrieval_diagnostics.py`; `scripts/run_route_policy.py`; `tests/test_retrieval_contract.py`; `artifacts/reports/phase10_route_policy.json`; `artifacts/reports/phase11_optional.json` | Parked until dense component improves. |
| Weighted fusion | U: `weighted` + weights summing to one | Blend normalized sparse and dense scores | `ghostlab/retrieval/fusion.py` | `scripts/run_retrieval_diagnostics.py`; `scripts/run_route_policy.py`; `tests/test_retrieval_contract.py`; `artifacts/reports/phase10_route_policy.json`; `artifacts/reports/phase11_optional.json`; validation in `ghostlab/research/technique_suite.py` | Parked; weights must be selected inside folds. |
| Sparse-first union | U: `sparse_first_union` | Preserve sparse head and fill remaining depth with unique dense candidates | `ghostlab/retrieval/fusion.py` | `configs/suites/dense_e5.json`; `scripts/run_dense_query_interaction.py`; `tests/test_dense_query_interaction.py`; `artifacts/reports/dense_query_interaction_v1.json` | Parked; preferred dense-complement retest path. |
| Dense + structured-query E2E | H/U component combination | Test query construction, dense retrieval and fusion together | `ghostlab/retrieval/dense.py`; `ghostlab/retrieval/query.py`; `ghostlab/retrieval/fusion.py` | `scripts/run_dense_query_e2e.py`; `configs/experiments/dense_query_e2e_v1.json`; `artifacts/reports/dense_query_e2e_v1.json`; `tests/test_dense_query.py`; `tests/test_dense_query_interaction.py` | Exploratory result preserved, not promotion evidence. |

### 19.4 Ranking, filtering, priors and safety guards

| Technique | Class and activation | Purpose | Source | Runner/config/tests/evidence | Status and retest rule |
|---|---|---|---|---|---|
| No reranker | U: `reranker=none`; P equivalent | Candidate-head control | `ghostlab/runtime/unified_experimental.py`; `ghostlab/runtime/experimental.py` | unified planner; retrieval/runtime tests | Required backward ablation. |
| Fixed lexical linear | U: `reranker=linear` | Hand-weighted lexical candidate reranking | `ghostlab/retrieval/rerank.py` | `scripts/run_optional_techniques.py`; `tests/test_reranker.py`; `artifacts/reports/phase11_optional.json` | Parked standalone. |
| Catalog quality prior | U/P: positive `quality_prior_weight` | Bounded catalog-quality tie-breaker | `ghostlab/retrieval/quality.py` | `scripts/run_quality_priors.py`; `tests/test_quality_reranker.py`; `artifacts/reports/phase18_quality_priors.json` | Weight `0.2` promoted. |
| Pairwise learned linear | P compiled original champion; H refit | Fit pairwise preference weights over lexical/catalog features | `ghostlab/retrieval/learned.py` | `scripts/run_learned_reranker.py`; `scripts/run_learned_feature_ablations.py`; `tests/test_learned_reranker.py`; `artifacts/reports/phase19_learned_reranker.json`; `artifacts/reports/phase20_learned_features.json`; original config in commit `189f0c6` | OOF `0.817649`; immutable original champion/control. |
| Quality-only learned ablation | H | Test whether quality alone explains learned-ranker gain | `ghostlab/retrieval/learned.py` | `scripts/run_learned_feature_ablations.py`; `artifacts/reports/phase20_learned_features.json` | Scalar `0.819207` but lower Hit@10/worse MTTC; near-champion, not selected. |
| Metadata GBDT/LambdaMART | U inference: `metadata_gbdt` + asset; H refit | Shallow nonlinear ranking from observable lexical/catalog metadata | `ghostlab/retrieval/gbdt.py` | `scripts/run_gbdt_reranker.py`; `configs/experiments/gbdt_reranker_v1.json`; `artifacts/reports/gbdt_reranker_v1.json`; `tests/test_gbdt_reranker.py`; `ghostlab/evaluation/gbdt_audit.py`; model `artifacts/models/gbdt_reranker_v2_round56.json` | OOF `0.861417`; validated fallback. |
| Constraint GBDT v1 | H only; invalid | Add conversation constraint coverage/invalidation/confidence features | `ghostlab/retrieval/constraint_gbdt.py` | `configs/experiments/gbdt_constraint_interaction_v1.json`; `scripts/run_gbdt_constraint_interaction.py`; `tests/test_gbdt_constraint_report.py`; `artifacts/reports/gbdt_constraint_interaction_v1.json` | Apparent `0.884943` invalidated; retain only as failure evidence. |
| Corrected constraint GBDT v2 | H and compiled asset | Repair stale state, bookkeeping and concurrent context | `ghostlab/retrieval/constraint_gbdt.py` | `configs/experiments/gbdt_constraint_interaction_v1_amendment_1.json`; `tests/test_gbdt_constraint_v2_report.py`; `artifacts/reports/gbdt_constraint_interaction_v2.json`; model `artifacts/models/gbdt_constraint_interaction_v2.json` | `0.876283`; parked without guard. |
| Observable override guard | P selected compiled path | Route invalidation turns to matched metadata GBDT and ordinary turns to constraint GBDT | `ghostlab/runtime/guarded_gbdt.py` | `scripts/run_gbdt_constraint_override_guard.py`; `configs/experiments/gbdt_constraint_interaction_v1_amendment_2.json`; `tests/test_gbdt_constraint_override_guard_report.py`; `artifacts/reports/gbdt_constraint_override_guard_v1.json` | Selected OOF `0.878963`; rerun routing attribution after any dependency change. |
| Deep dense-candidate GBDT | H only | Give GBDT deeper sparse/dense candidates and dense-derived features | `ghostlab/retrieval/gbdt_dense.py` | `scripts/run_gbdt_dense_interaction.py`; `configs/experiments/gbdt_dense_interaction_v1.json`; `artifacts/reports/gbdt_dense_interaction_v1.json`; `tests/test_gbdt_dense_interaction.py` | `0.829423`; parked until candidate recall changes. |
| Compact cross-encoder | U additive `cross_encoder_enabled=true` + local asset; H refit/eval | Neural Top-K pair scoring blended with current ranking | `ghostlab/retrieval/cross_encoder.py` | `configs/suites/cross_encoder.json`; `configs/assets/cross_encoder_minilm.json`; `configs/experiments/challenger_cross_encoder_v1.json`; `scripts/run_cross_encoder_challenger.py`; `tests/test_cross_encoder.py`; `tests/test_cross_encoder_report.py`; `artifacts/reports/challenger_cross_encoder_v1.json` | `0.790915`; parked. Reopen for new model/head/hardware/data. |
| Cross-encoder score + GBDT | H only | Add neural score as a fold-local GBDT feature | `ghostlab/retrieval/neural_rank.py` | `scripts/run_neural_rank_interaction.py`; `scripts/measure_neural_rank_runtime.py`; `configs/experiments/neural_rank_interaction_v1.json`; `artifacts/reports/neural_rank_interaction_v1.json`; `tests/test_neural_rank.py`; `tests/test_neural_rank_report.py` | `0.858094`; parked due score and latency. |
| Coverage-aware structured filter | U: `structured_filter=true`; P: enabled filter | Remove candidates contradicting sufficiently covered constraints with fallback | `ghostlab/retrieval/filters.py` | `scripts/run_targeted_adaptive.py`; `tests/test_filters_signals_evidence.py`; `artifacts/reports/phase13_targeted_adaptive.json` | Slightly harmful; retest only with measured parser precision/coverage. |
| Profile prior | U: positive `profile_prior_weight`; P profile switch | Add long-term profile evidence while yielding to explicit session intent | `ghostlab/retrieval/profile.py` | `scripts/run_profile_priors.py`; `tests/test_profile_reranker.py`; `artifacts/reports/phase15_profile_priors.json` | Fixed weights degraded monotonically; only retest as learned/gated evidence. |
| Runtime normalization/output guard | Always on official runtime | Normalize IDs, deduplicate, catalog-check and truncate Top-10 | `ghostlab/runtime/normalizer.py`; contract in `ghostlab/competition/contract.py` | runtime/submission tests and compiled validation | Required submission boundary; not an optimization switch. |

### 19.5 Routing and conditional execution

| Technique | Class and activation | Purpose | Source | Runner/tests/evidence | Status and retest rule |
|---|---|---|---|---|---|
| Observable decision list | P/U research component | Route/actions using explicit runtime-observable rules | `ghostlab/policy/decision_list.py` | `tests/test_route_policy.py`; `artifacts/reports/phase10_route_policy.json`; `artifacts/reports/phase14_conditional_route.json` | Available; must beat always-sparse matched control. |
| Route stump | H | Learn one shallow observable sparse/dense route | `ghostlab/research/route_stump.py` | `scripts/run_conditional_route_stump.py`; `tests/test_route_stump.py`; `artifacts/reports/phase14_conditional_route.json` | Collapsed to always sparse; dependency-gated. |
| Route table | H | Fold-local mapping from observable state buckets to retrieval route | `ghostlab/research/route_policy.py` | `scripts/run_route_policy.py`; `tests/test_route_policy.py`; `artifacts/reports/phase10_route_policy.json` | Parked; reopen only when both retrieval heads are competitive. |
| Guard route | P selected | Special-case only observable earlier-preference invalidation | `ghostlab/runtime/guarded_gbdt.py` | `scripts/run_gbdt_constraint_override_guard.py`; `configs/experiments/gbdt_constraint_interaction_v1_amendment_2.json`; `tests/test_gbdt_constraint_override_guard_report.py`; `artifacts/reports/gbdt_constraint_override_guard_v1.json` | Selected, narrowly scoped; do not broaden without a new manifest. |

### 19.6 Search, counterfactual, evidence and validation infrastructure

| Technique | Class | Purpose | Source | Runner/config/tests/evidence | Status/use rule |
|---|---|---|---|---|---|
| Deterministic replay | I | Reproduce multi-turn sessions and per-session rewards | `ghostlab/research/replay.py` | `scripts/validate_replay.py`; `tests/test_replay.py`; `artifacts/reports/phase6_replay_parity.json` | Required for every candidate. |
| Counterfactual evaluator | I/H | Evaluate legal question/action alternatives under the simulator | `ghostlab/research/counterfactual.py` | `scripts/run_counterfactuals.py`; `tests/test_counterfactual.py`; `artifacts/reports/phase7_counterfactuals.json` | Required input to learned-question refits; firewall rules apply. |
| Leakage firewall | I | Restrict training/runtime features to observable information and protect holdout | `ghostlab/research/firewall.py` | `tests/test_submission_boundary.py`; GBDT audits | Mandatory. |
| Grid/random/beam | I | Auditable bounded enumeration/control search | `ghostlab/optimization/search.py` | `scripts/run_standard_campaign.py`; `scripts/compare_standard_searchers.py`; `configs/search/standard.json`; `tests/test_optimizer.py`; `artifacts/reports/phase8_standard_campaign.json`; `artifacts/reports/phase8_standard_searchers.json` | Grid/beam control promoted for reproducibility. |
| Multi-fidelity racing | I | Stop weak candidates at declared budgets while auditing pruning regret | `ghostlab/optimization/racing.py` | `scripts/run_standard_campaign.py`; `tests/test_patches_racing.py`; `artifacts/reports/phase8_multifidelity_analysis.json` | Available; must preserve a full-budget audit sample. |
| Evidence allocator | I | Allocate search budget using accumulated technique evidence | `ghostlab/optimization/meta_search.py` | `scripts/compare_searchers.py`; `tests/test_meta_search.py`; `artifacts/reports/phase8_searchers.json` | Lost equal-budget comparison; parked until search families diversify. |
| Family UCB | I | Balance exploration/exploitation among technique families | `ghostlab/optimization/evidence.py` | `scripts/compare_searchers.py`; `tests/test_filters_signals_evidence.py`; `artifacts/reports/phase8_searchers.json` | Parked alternative. |
| Typed patches | I | Apply schema-safe technique mutations rather than arbitrary dict edits | `ghostlab/optimization/patches.py` | `scripts/run_standard_campaign.py`; `configs/search/standard.json`; `tests/test_patches_racing.py` | Available composition primitive. |
| Typed crossover | I | Combine compatible parent patches while retaining validation | `ghostlab/optimization/patches.py` | `scripts/run_standard_campaign.py`; `tests/test_patches_racing.py` | Interaction reserve; backward ablations remain mandatory. |
| Decision/evidence store | I | Preserve chronological hypotheses, results, diagnoses and retest triggers | `ghostlab/optimization/evidence.py` | `artifacts/evidence/technique_decisions.jsonl`; `scripts/validate_decision_ledger.py`; `tests/test_decision_ledger.py` | Required research record. |
| Grouped split manager | I | Keep all turns from a session in the same fold | `ghostlab/evaluation/splits.py` | `configs/splits/adaptive_v1.json`; `configs/splits/nested_v1.json`; `scripts/freeze_splits.py`; `tests/test_splits.py` | Mandatory anti-leakage boundary. |
| Paired statistics | I | Bootstrap/randomization and per-session/fold/scenario deltas | `ghostlab/evaluation/statistics.py` | `configs/validation/primary_analysis.json`; `tests/test_statistics.py`; all advanced challenger reports | Mandatory for promotion claims. |
| GBDT deployment audit | I | Verify nested fitting, refit determinism, assets, runtime and parity | `ghostlab/evaluation/gbdt_audit.py` | `scripts/resolve_gbdt_deployment_audit.py`; `artifacts/reports/gbdt_deployment_audit_v1.json`; `tests/test_gbdt_deployment_audit.py` | Mandatory for learned-tree promotion. |
| Compiled parity/runtime gates | I/P | Prove research, compiled and starter outputs agree and meet runtime contract | `ghostlab/runtime/compiled.py`; `ghostlab/runtime/agent.py` | `scripts/validate_compiled.py`; `scripts/validate_guarded_compiled.py`; `scripts/measure_guarded_compiled.py`; `tests/test_compiled.py`; `tests/test_guarded_compiled.py` | Mandatory before submission. |
| Unified combination planner | I | Enumerate only schema-valid switch combinations | `ghostlab/research/technique_suite.py`; `scripts/plan_unified_combinations.py` | `configs/search/unified_space_v1.json`; `tests/test_technique_suite.py` | 864 legal keyword candidates; compatibility filtering is not score pruning. |

## 20. Challenger provenance, complete file bundles and commands

Each bundle below names everything required to understand or reproduce the
challenger. The origin commit is preserved even though reusable files now live in
this branch.

| Challenger | Origin | Source and runtime | Manifest/preset and asset | Runner and tests | Evidence and command |
|---|---|---|---|---|---|
| Original pairwise-linear champion | `ghostlab/implementation@189f0c6` | `ghostlab/state/memory.py`; `ghostlab/retrieval/sparse.py`; `ghostlab/retrieval/quality.py`; `ghostlab/retrieval/learned.py`; `ghostlab/runtime/compiled.py` | Historical `configs/compiled_policy.json` at that commit | `scripts/run_learned_reranker.py`; `scripts/run_learned_feature_ablations.py`; `tests/test_learned_reranker.py`; `tests/test_compiled.py`; `tests/test_runtime.py` | `artifacts/reports/phase19_learned_reranker.json`; `artifacts/reports/phase20_learned_features.json`; `artifacts/reports/phase22_champion_compiled_parity.json`; `artifacts/reports/phase23_champion_checkpoint.json`; `docs/champion_checkpoint.md`; run the checkpoint validator |
| Query construction | `exp/query-construction@c2620aa` | `ghostlab/state/query.py` | `configs/experiments/challenger_query_v1.json` | `scripts/run_query_challenger.py`; `tests/test_query_construction.py` | `artifacts/reports/challenger_query_v1.json`; `docs/challenger_query_v1.md`; run the query script |
| Learned questions | `exp/learned-question-policy@cab92c0` | `ghostlab/policy/learned_questions.py`; `ghostlab/research/learned_questions.py`; `ghostlab/runtime/unified_experimental.py` | `configs/experiments/learned_question_linear_v1.json`; `configs/suites/learned_questions.json`; `artifacts/experiments/learned_question_linear_v1/linear_action_value_model.json` | `scripts/run_learned_question_challenger.py`; `tests/test_learned_questions.py` | `artifacts/reports/learned_question_linear_v1.json`; `docs/learned_question_policy_decision.md`; run the learned-question script |
| E5/dense/query | `exp/dense-retrieval@d8cf054` | `ghostlab/retrieval/dense.py`; `ghostlab/retrieval/query.py`; `ghostlab/retrieval/fusion.py` | `configs/assets/minilm_control.json`; `configs/assets/e5_small_v2.json`; `configs/experiments/dense_retrieval_v1.json`; `configs/experiments/dense_query_e2e_v1.json`; `configs/experiments/dense_query_interaction_v1.json`; `configs/suites/dense_e5.json` | `scripts/fetch_dense_assets.py`; `scripts/run_dense_retrieval.py`; `scripts/run_dense_query_e2e.py`; `scripts/run_dense_query_interaction.py`; `tests/test_dense_retrieval.py`; `tests/test_dense_query.py`; `tests/test_dense_query_interaction.py` | `artifacts/reports/dense_retrieval_v1.json`; `artifacts/reports/dense_query_e2e_v1.json`; `artifacts/reports/dense_query_interaction_v1.json`; run the corresponding dense module |
| Metadata GBDT | `exp/gbdt-reranker@cbfd7d5` | `ghostlab/retrieval/gbdt.py`; `ghostlab/evaluation/gbdt_audit.py` | `configs/experiments/gbdt_reranker_v1.json`; `configs/experiments/gbdt_reranker_v1_amendment_1.json`; `artifacts/models/gbdt_reranker_v2_round56.json` | `scripts/run_gbdt_reranker.py`; `scripts/resolve_gbdt_deployment_audit.py`; `scripts/measure_gbdt_runtime.py`; `tests/test_gbdt_reranker.py`; `tests/test_gbdt_deployment_audit.py` | `artifacts/reports/gbdt_reranker_v1.json`; `artifacts/reports/gbdt_deployment_audit_v1.json`; run the GBDT script |
| Cross-encoder | `exp/cross-encoder@071eda9` | `ghostlab/retrieval/cross_encoder.py` | `configs/experiments/challenger_cross_encoder_v1.json`; `configs/assets/cross_encoder_minilm.json`; `configs/suites/cross_encoder.json` | `scripts/run_cross_encoder_challenger.py`; `tests/test_cross_encoder.py`; `tests/test_cross_encoder_report.py` | `artifacts/reports/challenger_cross_encoder_v1.json`; `docs/cross_encoder_challenger_v1.md`; run the CE script |
| Constraint GBDT and guard | `exp/interaction-gbdt-constraints@93bf07b` | `ghostlab/retrieval/constraint_gbdt.py`; `ghostlab/runtime/guarded_gbdt.py` | `configs/experiments/gbdt_constraint_interaction_v1.json`; `configs/experiments/gbdt_constraint_interaction_v1_amendment_1.json`; `configs/experiments/gbdt_constraint_interaction_v1_amendment_2.json`; `artifacts/models/gbdt_constraint_interaction_v2.json` | `scripts/run_gbdt_constraint_interaction.py`; `scripts/run_gbdt_constraint_override_guard.py`; `scripts/measure_constraint_gbdt_runtime.py`; `scripts/measure_constraint_guard_runtime.py`; `tests/test_gbdt_constraint_report.py`; `tests/test_gbdt_constraint_v2_report.py`; `tests/test_gbdt_constraint_override_guard_report.py` | `artifacts/reports/gbdt_constraint_interaction_v1.json`; `artifacts/reports/gbdt_constraint_interaction_v2.json`; `artifacts/reports/gbdt_constraint_override_guard_v1.json`; `docs/gbdt_constraint_interaction_report.md`; `docs/gbdt_constraint_interaction_v2_report.md`; `docs/gbdt_constraint_override_guard_report.md`; run the guard script |
| Dense + GBDT | `exp/interaction-gbdt-dense@b6500ff` | `ghostlab/retrieval/gbdt_dense.py` | `configs/experiments/gbdt_dense_interaction_v1.json` | `scripts/run_gbdt_dense_interaction.py`; `tests/test_gbdt_dense_interaction.py` | `artifacts/reports/gbdt_dense_interaction_v1.json`; `docs/gbdt_dense_interaction_report.md`; run the dense-GBDT script |
| Learned questions + GBDT | `exp/interaction-gbdt-question@6711182` | `ghostlab/runtime/experimental_questions.py` | `configs/experiments/gbdt_question_interaction_v1.json`; `artifacts/experiments/gbdt_question_interaction_v1/linear_action_value_model.json` | `scripts/run_gbdt_question_interaction.py`; `tests/test_gbdt_question_interaction.py` | `artifacts/reports/gbdt_question_interaction_v1.json`; `docs/gbdt_question_interaction_decision.md`; run the GBDT-question script |
| Cross-encoder + GBDT | `exp/interaction-neural-rank@e211434` | `ghostlab/retrieval/neural_rank.py` | `configs/experiments/neural_rank_interaction_v1.json` | `scripts/run_neural_rank_interaction.py`; `scripts/measure_neural_rank_runtime.py`; `tests/test_neural_rank.py`; `tests/test_neural_rank_report.py` | `artifacts/reports/neural_rank_interaction_v1.json`; `docs/neural_rank_interaction_report.md`; run the neural interaction script |
| Compiled guarded winner | `exp/compile-guarded-gbdt@e6b3949`, integrated at `ec4906a` | `ghostlab/runtime/agent.py`; `ghostlab/runtime/guarded_gbdt.py` | `configs/techniques/guarded_constraint_gbdt_v1.json`; `configs/suites/champion_guarded.json` | `scripts/validate_guarded_compiled.py`; `scripts/measure_guarded_compiled.py`; `tests/test_guarded_compiled.py`; `tests/test_submission_boundary.py` | `artifacts/reports/guarded_compiled_parity_v1.json`; `artifacts/reports/guarded_compiled_runtime_v1.json`; `docs/final_candidate_checkpoint.md`; run the guarded validator with `--require-default` |
| Unified composer | `ghostlab/unified-techniques` | `ghostlab/research/technique_suite.py`; `ghostlab/runtime/unified_experimental.py` | `configs/suites/`; `configs/techniques/catalog_v1.json`; `configs/search/unified_space_v1.json` | `scripts/run_unified_preset.py`; `scripts/plan_unified_combinations.py`; `scripts/audit_unified_consolidation.py`; `tests/test_technique_suite.py` | `artifacts/reports/unified_champion_verification_v1.json`; `artifacts/reports/unified_consolidation_audit_v1.json`; run the audit script |

## 21. Decision-ledger index

The complete diagnoses live in `artifacts/evidence/technique_decisions.jsonl`.
This index ensures that every decision involved in the first champion is reachable
from this system reference.

| Decisions | Subject | Outcome | Primary evidence |
|---|---|---|---|
| D001 | First pairwise-linear system champion | Promoted control `0.817649` | `artifacts/reports/phase20_learned_features.json`; `artifacts/reports/phase22_champion_compiled_parity.json`; `artifacts/reports/phase23_champion_checkpoint.json`; `docs/champion_checkpoint.md` |
| D002–D004, D023 | Static sequence, heuristic adaptive, planned/then-tested learned questioning, and no-question control | Static sequence promoted; adaptive/no-question parked; learned challenger covered by advanced reports | `artifacts/reports/phase7_counterfactuals.json`; `artifacts/reports/phase9_question_policy.json`; `artifacts/reports/phase13_targeted_adaptive.json`; `artifacts/reports/phase21_cross_phase_rechecks.json`; `artifacts/reports/learned_question_linear_v1.json` |
| D005–D007, D022 | Raw history, structured multi-state, hybrid query and negative evidence | Raw history promoted; others retained as interaction channels | `artifacts/reports/phase4_5_summary.json`; `artifacts/reports/phase4_5_ablations.json`; `artifacts/reports/phase13_targeted_adaptive.json`; `artifacts/reports/phase21_cross_phase_rechecks.json`; `artifacts/reports/challenger_query_v1.json` |
| D008–D012, D024 | Field BM25, MiniLM, specialized dense, fusion, routing and sparse-semantic proposal | Field BM25 promoted; dense/fusion/routing dependency-gated; sparse semantic remains unimplemented | `artifacts/reports/phase3_retrieval.json`; `artifacts/reports/phase10_route_policy.json`; `artifacts/reports/phase11_optional.json`; `artifacts/reports/phase14_conditional_route.json`; `artifacts/reports/phase16_field_weights.json`; `artifacts/reports/phase17_field_interactions.json`; three exact dense report paths in Section 20 |
| D013–D014 | Coverage filter and fixed profile prior | Parked | `artifacts/reports/phase13_targeted_adaptive.json`; `artifacts/reports/phase15_profile_priors.json` |
| D015–D017, D025 | Quality prior, fixed lexical, pairwise linear and quality-only ablation | Quality + pairwise linear selected under robustness rule | `artifacts/reports/phase11_optional.json`; `artifacts/reports/phase18_quality_priors.json`; `artifacts/reports/phase19_learned_reranker.json`; `artifacts/reports/phase20_learned_features.json` |
| D018–D019 | Planned GBDT and cross-encoder challengers | Subsequently implemented in advanced branches | GBDT and CE manifests/reports |
| D020–D021 | Evidence allocator versus grid/beam control | Grid control promoted; allocator parked | `artifacts/reports/phase8_searchers.json`; `artifacts/reports/phase8_standard_searchers.json` |
| D026–D027 | Metadata GBDT and deployable refit | Promoted fallback; deterministic refit/runtime passed | GBDT and deployment-audit evidence |
| D028–D029 | Constraint GBDT v1 and independent audit | Numeric gain invalidated; never use for selection | v1 report and amendment-1 audit |
| D030 | Corrected constraint GBDT v2 | Parked after safety gates failed | v2 report |
| D031 | Observable override guard | Selected OOF candidate `0.878963` | `configs/experiments/gbdt_constraint_interaction_v1_amendment_2.json`; `artifacts/reports/gbdt_constraint_override_guard_v1.json` |

## 22. Current automation boundary and autonomous experiment system

The repository now contains a bounded, resumable, proposal-only autonomous development
controller. A human still chooses and freezes a declared campaign, starts/resumes the
runner, reviews proposals, and controls Gate A/F3/Gate B; the controller cannot commit,
merge, push, expose F3, or promote a champion. The full executable contract and commands
are in `docs/autonomous_unified_system_reference.md`.

The implemented components include:

- `scripts/plan_unified_combinations.py` enumerates schema-valid combinations;
- `scripts/freeze_wave2_campaign.py` freezes hashes, splits, assets and authority;
- `scripts/run_autonomous_campaign.py` runs bounded F0/F1/F2 structure/HPO search and
  resumes from atomic checkpoints;
- `scripts/materialize_campaign_top_three.py` fail-closed packages up to three distinct
  development-confirmed proposals for human review;
- `ghostlab/research/technique_suite.py` rejects incompatible configurations;
- `ghostlab/campaign/` provides typed bindings, compatibility, scheduling, paired
  evaluation, interaction-aware search, safety gates, leaderboards and proposal records;
- `ghostlab/optimization/search.py`, `ghostlab/optimization/racing.py`,
  `ghostlab/optimization/meta_search.py` and `ghostlab/optimization/evidence.py`
  provide grid/random/beam, racing and adaptive allocation primitives;
- dedicated runners perform grouped fitting and replay for their technique family;
- reports, the decision ledger and consolidation audit preserve evidence.

The controller is deliberately not an unrestricted Cartesian-product script. It:

1. read the technique catalog as a dependency/compatibility graph;
2. freezes the pure keyword baseline, folds, objective, gates, candidate budget and
   holdout policy; prior winners are `control_only`;
3. validate/install only assets required by enabled techniques;
4. cache shared sparse, dense, feature and replay outputs by content hash;
5. schedule grouped nested-CV fitting and evaluation with deterministic seeds;
6. use successive halving/racing while fully evaluating an audit sample of pruned
   candidates to estimate pruning regret;
7. expand two-, three- and higher-order combinations only when compatibility and
   dependency evidence justify them, while retaining exploration budget for
   surprising interactions;
8. run matched controls, component-off ablations, scenario safety gates, latency,
   memory and failure checks automatically;
9. append immutable manifests, reports and machine-readable evidence;
10. proposes—but never automatically promotes—up to three distinct candidates for human
    review;
11. keep the protected/private set inaccessible to the search controller.

It does not yet automatically build/fold-fit every unavailable or `fit_required` asset,
and its generic path uses prospective disjoint development confirmation rather than fully
nested fold-local training. Those techniques remain recorded and default-off until their
safe fit path exists. Blindly trying every possible combination is neither guaranteed to find the true
best policy nor statistically safe. The number of continuous parameters, fitted
models, question trajectories and conditional routes is effectively unbounded, and
repeated public-set selection will overfit. Autonomy should improve reproducibility,
coverage and scheduling while the anti-overfitting and promotion gates remain
strictly human-controlled.

## 23. Retesting and extending the whole system

The default unbiased discovery campaign must always restart from the pure unified keyword
anchor described above. When a better baseline arrives, keep that from-scratch campaign
and run the following as a separate sensitivity/retest campaign:

1. Freeze the new baseline as a versioned suite/config and reproduce the old control.
2. Use Sections 19 and 20 to identify every technique whose dependency changed.
3. Run each affected technique alone against the new matched control.
4. Reopen previously supported interactions and plausible dependency-mediated
   combinations; do not assume an old negative result remains negative.
5. Keep incompatible or fold-trained techniques in dedicated runners; never fake a
   deployable on/off toggle when a fitted asset does not exist.
6. For a winning combination, run every component-off backward ablation and matched
   depth/runtime control.
7. Fit learned pieces inside grouped outer-training folds and select parameters only
   inside inner folds.
8. Record per-session, fold, scenario, Hit@10, MRR, MTTC, technical score, latency,
   memory, failures and asset hashes.
9. Append a new decision; never rewrite the old manifest, report or diagnosis.
10. Promote only after compiled/starter parity, safety gates, integrity audit and the
    one-shot protected-evaluation rule.

Whenever a technique is added, update all of the following in the same commit:

- its implementation and tests;
- `configs/techniques/catalog_v1.json` or a new catalog version;
- a suite or dedicated experiment manifest;
- `configs/integrity/unified_consolidation_v1.json` or a new manifest version;
- Sections 18–21 of this guide;
- `artifacts/evidence/technique_decisions.jsonl` after evaluation;
- the new immutable report and explicit retest trigger.

The preservation rule is permanent: **off is a default, not deletion; parked is a
decision under known dependencies, not a universal rejection.**
