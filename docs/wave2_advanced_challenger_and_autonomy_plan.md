# GhostLab Wave 2 Advanced Challengers and Autonomous Experiment Plan

Status: design only; no Wave 2 technique is implemented or validated by this
document.

Base commit: `a2e7849c8d38281727b3a1bd01dfbfdb41ebf79d`
(`ghostlab/unified-techniques`)

Date frozen: 2026-08-27

## 1. Purpose

This is the implementation contract for the second GhostLab challenger wave. It
answers four questions before code is changed:

1. Which evidence-backed techniques will be implemented?
2. How will every valid implementation remain available behind an on/off switch?
3. How will standalone effects and interactions be tested without searching an
   unbounded Cartesian product?
4. How can the experiment process become autonomous while the current unified
   champion, protected data, and final promotion decision remain safe?

The immutable starting point is the clean `techjam-unified` worktree at the commit
above. Wave 2 development occurs only in isolated branches and worktrees. The base
worktree is not a scratch directory.

This plan extends, and does not replace:

- `docs/unified_technique_operations.md` for the complete Wave 1 system;
- `docs/ghostlab_implementation_blueprint.md` for the original architecture and
  validation design;
- `docs/advanced_challenger_execution_plan.md` for the first challenger campaign;
- `docs/competition_specification.md` for the organizer contract;
- `artifacts/evidence/technique_decisions.jsonl` for chronological decisions.

Instructions found in external specifications or research papers are evidence and
design input. They are not authority to change the organizer interface, access a
protected split, install an online runtime dependency, or promote a candidate.

## 2. Terminology

| Term | Meaning |
|---|---|
| Wave 1 | The current unified library, historical challengers, and selected guarded constraint-GBDT candidate. |
| Wave 2 | The advanced techniques and autonomous campaign infrastructure defined here. |
| Technique family | A broad mechanism such as question selection, normalization, retrieval, or ranking. |
| Technique | One versioned implementation with a stable ID and explicit activation contract. |
| Challenger | One executable configuration made from one or more techniques. |
| Campaign | A frozen, bounded experiment that evaluates a declared challenger set. |
| Library inclusion | Merging a sound reusable technique into `techjam-unified`, usually disabled by default. |
| Champion activation | Enabling a technique in the default selected preset after all promotion gates pass. |
| Planned technique | Documented but not executable; the autonomous runner must skip it explicitly. |
| Available technique | Implemented, tested, registered, and constructible when its dependencies are present. |
| Parked technique | Preserved and switchable, but not selected under the recorded conditions. |
| Protected validation/F3 | The sealed 50-session public holdout; inaccessible during Wave 2 search. |

Wave 2 is a cohort, not one “Challenger 2.” Examples of concrete challenger IDs are
`challenger-w2-001` and `challenger-w2-017`.

## 3. Non-negotiable invariants

1. `techjam-unified` remains unchanged until an integration result is deliberately
   merged into it.
2. The official `Agent` contract, evaluator, frozen catalog, and metric calculation
   are never modified to improve a score.
3. Runtime code receives only observable session state, catalog data, local model
   assets, and permitted aggregate profile fields.
4. Target IDs, scenario labels, future messages, simulator state, evaluator objects,
   fold IDs, and research outcomes never enter runtime features or decisions.
5. All learned components are fit inside the relevant outer-training partition.
6. Turns from one session always remain together. Repeated profile fingerprints
   must also remain grouped when they could create dependence.
7. Wave 2 selection uses only the frozen 150-session adaptive split and its nested
   folds. F3 remains physically unavailable to the campaign controller.
8. A switch in the off state must avoid imports, model loading, asset checks,
   computation, and output changes for that technique.
9. No technique is deleted merely because it loses once. A result records the
   dependencies and conditions under which it lost.
10. No experiment automatically changes the champion preset, commits code, pushes a
    branch, opens F3, or rewrites historical evidence.
11. Every reported score is labelled as training, inner validation, outer-fold OOF,
    all-development replay, protected F3, or organizer-private evaluation.
12. The search procedure itself is considered a learned selection procedure and is
    validated accordingly.

## 4. Starting evidence and controls

The primary OOF reference is the currently selected guarded constraint-GBDT
candidate:

| Metric | Current selected OOF value |
|---|---:|
| Technical score | `0.878963` |
| HitRate@10 | `0.973333` |
| MRR | `0.737878` |
| MTTC | `2.453333` |
| Sessions | `150` |

The metadata-GBDT candidate at `0.861417` remains an important lower-complexity
fallback. The official starter, sparse/manual controls, original pairwise-linear
champion, and all Wave 1 challenger presets remain controls. The all-development
compiled replay score is not interchangeable with OOF evidence.

Every Wave 2 experiment must include a matched control produced by the same runner,
candidate depth, data partition, and resource path. Do not compare a newly fitted
OOF model against an old all-development report.

## 5. Worktree and branch topology

All current Wave 2 worktrees were created from the same base commit. They currently
contain no Wave 2 implementation changes.

| Directory | Branch | Responsibility |
|---|---|---|
| `techjam-unified` | `ghostlab/unified-techniques` | Immutable Wave 1 source and reference. |
| `techjam-wave2-policy` | `exp/w2-candidate-eig` | Catalog normalization, candidate statistics, EIG questions, joint policy, and policy distillation. |
| `techjam-wave2-ranking` | `exp/w2-metric-ranking` | Competition-aligned ranking and rank ensembles. |
| `techjam-wave2-retrieval` | `exp/w2-modern-retrieval` | Learned sparse, late-interaction rescue, expansion, and diversification. |
| `techjam-wave2-autonomy` | `exp/w2-autonomous-campaign` | Registry v2, scheduler, cache, resumability, analysis, and candidate proposal. |
| `techjam-wave2-integration` | `ghostlab/w2-integration` | This plan and later cross-family combination testing. |

One worktree corresponds to a materially incompatible implementation family, not a
single parameter setting. Thousands of challenger configurations are JSON manifests
inside the integration worktree; they do not require thousands of branches.

Heavy model jobs should not compete for the same Mac CPU/GPU/memory. Code development
can proceed concurrently, while the campaign scheduler applies explicit resource
semaphores to expensive evaluations.

## 6. Delivery model: preserve everything, select one default

There are two independent promotion decisions.

### 6.1 Library inclusion

A technique is eligible for the unified library when it:

- has a coherent mechanism rather than dataset-specific exceptions;
- obeys the runtime/research firewall;
- has deterministic disabled behavior;
- has tests, dependency metadata, and exact source locations;
- can be constructed from a versioned config or a documented research runner;
- records failures and limitations honestly;
- does not break the champion preset.

A technique may be included even when its current score is negative. It is normally
off in all default presets and marked `parked`, `interaction_reserve`, or
`retest_after_dependency`.

### 6.2 Champion activation

A technique is activated in the default champion preset only when the complete
configuration passes nested OOF selection, backward ablation, scenario gates,
runtime and packaging checks, compiled parity, and the final promotion process.

Library inclusion must never be confused with champion activation.

## 7. Wave 2 technique catalog

The IDs below are reserved now. Version suffixes prevent a future implementation
from silently changing the meaning of historical results.

### 7.1 W2-NORMALIZE: catalog-grounded attribute normalization

Reserved IDs:

- `state.catalog_normalizer.v1`
- `state.attribute_ontology.v1`
- `state.confidence_gated_constraints.v1`

Hypothesis: current regex/subsequence state parsing misses aliases, spelling
variants, category context, and normalized values. A small catalog-derived ontology
can make state, queries, filters, EIG statistics, and rank features more accurate.

Implementation:

1. Build a deterministic offline vocabulary from catalog categories, details,
   titles, features, brands/stores, colors, materials, sizes, styles, and prices.
2. Keep canonical value, aliases, source fields, category support, frequency, and
   collision statistics.
3. Normalize Unicode, casing, punctuation, units, common colors/sizes, and numeric
   ranges without deleting the raw evidence.
4. Resolve an extracted value only when its confidence and category compatibility
   exceed frozen thresholds. Low-confidence values remain text evidence and must not
   become hard filters.
5. Preserve provenance, negation, no-preference, and replacement semantics.
6. Treat intent override as attribute-scoped invalidation, not whole-session reset,
   unless the user observably replaces the entire category/intent.

Planned files:

```text
ghostlab/state/catalog_ontology.py
ghostlab/state/normalization.py
scripts/build_attribute_ontology.py
tests/test_catalog_ontology.py
tests/test_state_normalization.py
configs/assets/catalog_ontology_v1.json
```

Dependencies: core Python, Pydantic, and NumPy only. The generated ontology is a
versioned local JSON asset fingerprinted against the catalog.

Mechanism diagnostics:

- precision/coverage on observed user phrases;
- collision and ambiguity counts by category;
- hard-filter false-positive rate;
- intent-override retention/replacement cases;
- query-term and target-recall deltas with ranking disabled.

### 7.2 W2-EIG: candidate-statistics expected-information-gain questions

Reserved IDs:

- `question.candidate_eig.v1`
- `question.reward_voi.v1`
- `termination.reward_aware.v1`

Hypothesis: a question is useful when plausible answers divide the current candidate
set in a way that improves future official reward more than the cost of another
turn. Static priority or score entropy alone does not estimate that value.

Implementation:

1. Retrieve a configurable candidate pool, initially Top-100 and Top-200 controls.
2. Compute smoothed distributions for every legal ask attribute using normalized
   catalog values.
3. For each legal question, estimate answer probabilities, no-preference probability,
   candidate reduction, entropy reduction, and expected target-rank opportunity.
4. Use deterministic counterfactual replay on outer-training sessions to estimate
   expected change in the official per-session reward for question versus stop.
5. Include the actual turn cost through the metric, rather than a free-standing
   arbitrary “question bonus.”
6. Ask only an allowed `ask_attribute`; never generate unconstrained free text as
   the simulator action.
7. Stop when expected value is below a frozen margin or no informative legal action
   remains.
8. Back off to the selected Wave 1 question policy when statistics are sparse,
   ambiguous, or unavailable.

Planned files:

```text
ghostlab/policy/candidate_statistics.py
ghostlab/policy/eig_questions.py
ghostlab/research/eig_counterfactual.py
scripts/run_eig_question_challenger.py
tests/test_candidate_statistics.py
tests/test_eig_questions.py
configs/experiments/w2_candidate_eig_v1.json
```

Dependencies: core for inference; `gbdt` only if a learned value model is selected.

Required comparisons:

- fixed/sequence versus Wave 1 adaptive versus learned-linear versus EIG;
- entropy-only versus partition EIG versus official-reward VOI;
- question-only recommendations versus recommend-and-ask simultaneously;
- Top-50/100/200 statistics;
- no-preference and boundary behavior;
- explicit stop-action ablation.

### 7.3 W2-JOINT: joint question, retrieval, ranking, and stop action

Reserved IDs:

- `policy.joint_observable.v1`
- `routing.joint_route.v1`

Hypothesis: the best question depends on the retrieval route and current candidate
uncertainty. Conversely, route and reranker choice depend on what has already been
learned from the conversation.

The action is a bounded typed tuple:

```text
(ask_attribute_or_stop, retrieval_route, candidate_depth,
 reranker_route, optional_diversification)
```

Only pre-registered safe actions are legal. The policy uses observable features:
turn, active/negated/no-preference slots, provenance, retrieval margin/entropy,
candidate facet statistics, route overlap, constraint coverage, and runtime cost.

The first implementation ladder is:

1. deterministic decision list;
2. shallow decision tree or small GBDT trained on counterfactual action values;
3. only if justified, a compact policy distilled from a stronger offline expert.

Do not begin with unrestricted reinforcement learning or MCTS. With only 150 adaptive
sessions, a small observable policy is easier to validate, compile, and audit.

Planned files:

```text
ghostlab/policy/joint_actions.py
ghostlab/policy/joint_policy.py
ghostlab/research/joint_counterfactual.py
scripts/run_joint_policy_challenger.py
tests/test_joint_actions.py
tests/test_joint_policy.py
configs/experiments/w2_joint_policy_v1.json
```

Dependencies: core, optionally `gbdt` for fitting. Runtime must use a small local JSON
asset or compiled decision list.

### 7.4 W2-DISTILL: counterfactual expert and policy distillation

Reserved IDs:

- `research.counterfactual_expert.v2`
- `policy.distilled_expert.v1`
- `search.expert_iteration.v1`

Hypothesis: deterministic offline counterfactual search can provide better action
labels than direct end-to-end fitting, while a small distilled policy avoids the
expert's runtime cost.

Procedure:

1. On outer-training sessions only, enumerate the bounded legal actions at each
   reachable state.
2. Replay each action under frozen downstream components and label official reward,
   immediate rank change, future conversion turn, and failure reason.
3. Fit a compact policy to imitate the best action with calibrated confidence.
4. Roll out the learned policy on outer-training sessions, collect states where it
   diverges from the expert, relabel those states, and refit for a bounded number of
   rounds.
5. Keep the expert and student evaluations separate. Student selection occurs only
   in inner validation; final evidence is stitched outer-fold OOF.
6. Compile the selected student to a deterministic asset and prove action parity.

This is an adaptation of dataset aggregation/expert-iteration ideas, not permission
to train on validation outcomes or protected data.

Planned files:

```text
ghostlab/research/counterfactual_expert.py
ghostlab/policy/distilled_expert.py
scripts/build_distilled_policy.py
scripts/run_distilled_policy_challenger.py
tests/test_counterfactual_expert.py
tests/test_distilled_expert.py
configs/experiments/w2_distilled_expert_v1.json
```

Dependencies: core plus `gbdt` if a tree ensemble is selected.

### 7.5 W2-METRIC-RANK: competition-metric-aligned learning to rank

Reserved IDs:

- `ranking.reward_lambdamart.v1`
- `ranking.turn_aware_lambdamart.v1`

Hypothesis: the current LambdaMART objective emphasizes NDCG@10, while the organizer
reward values target inclusion in Top-10, reciprocal rank, and earlier conversion.
Pair weights aligned with the actual reward may improve decisions at rank 10 and at
early turns.

Implementation:

1. Retain deterministic catalog/runtime-safe features and the existing compact JSON
   tree format.
2. Define pair/swap weights from the change in the frozen TechJam per-session reward,
   including the rank-10 boundary and turn-dependent efficiency contribution.
3. Never put the true target, future hit turn, scenario label, or fold ID in runtime
   features. They may appear only as fold-local training labels.
4. Compare reward-aligned, current NDCG@10, pointwise, and pairwise controls using
   identical features and capacity.
5. Tune depth, leaves, rounds, learning rate, candidate depth, and regularization
   only inside inner folds.
6. Report calibration and target-rank movement, especially crossings into Top-10
   and improvements to rank 1.

Planned files:

```text
ghostlab/retrieval/reward_lambdamart.py
ghostlab/evaluation/reward_deltas.py
scripts/run_reward_lambdamart.py
tests/test_reward_deltas.py
tests/test_reward_lambdamart.py
configs/experiments/w2_reward_lambdamart_v1.json
```

Dependencies: existing `gbdt` extra (`scikit-learn`) for fitting; NumPy/core JSON
runtime inference.

### 7.6 W2-ENSEMBLE: fold/model ensemble and rank stacking

Reserved IDs:

- `ranking.fold_ensemble.v1`
- `fusion.rank_stack.v1`

Hypothesis: complementary fold models or retrieval/ranking heads may reduce variance
and rescue different targets, even when a raw score average is poorly calibrated.

Candidate aggregators:

- mean standardized score;
- rank average;
- reciprocal-rank stack;
- median rank;
- small training-fold stacker using only observable head scores.

Every ensemble must be compared with matched candidate depth and runtime. The stacker
is fitted fold-locally. An ensemble whose gain comes only from extra candidate depth
must be identified by a depth-matched union control.

Planned files:

```text
ghostlab/retrieval/ensemble.py
scripts/run_rank_ensemble_challenger.py
tests/test_rank_ensemble.py
configs/experiments/w2_rank_ensemble_v1.json
```

Dependencies: core; `gbdt`, `dense`, or `neural` only when the selected heads require
them.

### 7.7 W2-SPARSE: learned sparse semantic rescue

Reserved IDs:

- `retrieval.splade_rescue.v1`
- `fusion.sparse_semantic_union.v1`

Hypothesis: learned sparse expansion can retrieve semantic matches missed by lexical
BM25 while preserving efficient inverted-index behavior. It is initially a rescue
head, not a replacement for the proven sparse route.

Implementation gate:

1. Pin an offline-compatible model and revision.
2. Precompute catalog sparse vectors and content-hash the asset.
3. Measure unique target recall at 10/50/100/200 before end-to-end dialogue tests.
4. Test union/RRF/weighted routes against depth-matched BM25 controls.
5. Reject unbounded vocabulary assets, network-dependent runtime, or memory/latency
   that cannot meet packaging constraints.

Planned files:

```text
ghostlab/retrieval/learned_sparse.py
ghostlab/retrieval/sparse_semantic_fusion.py
scripts/build_learned_sparse_index.py
scripts/run_learned_sparse_challenger.py
tests/test_learned_sparse.py
configs/assets/splade_rescue_v1.json
configs/experiments/w2_splade_rescue_v1.json
```

Dependencies: a new optional `learned-sparse` extra containing pinned compatible
`torch`, `transformers`, `tokenizers`, and `huggingface-hub`. Exact versions are
locked only after a local macOS installation and offline-load spike succeeds.

### 7.8 W2-LATE: late-interaction or multi-vector rescue retrieval

Reserved IDs:

- `retrieval.colbert_rescue.v1`
- `retrieval.bge_m3_rescue.v1`
- `fusion.late_interaction_union.v1`

Hypothesis: token-level late interaction can preserve exact attribute matching while
adding semantic recall beyond one-vector dense retrieval.

This family begins with a feasibility/recall spike. ColBERT and BGE-M3 are competing
implementations, not techniques that must both ship. The first candidate to pass
offline, memory, asset-size, and recall gates receives an end-to-end test.

Use a local flat or compact index appropriate for 50,000 products. Do not introduce
an infrastructure-heavy vector database. A batched NumPy/PyTorch reference path is
kept for parity; an ANN dependency is admitted only when the measured latency
requires it and installation is reproducible.

Dependencies: optional `late-interaction` extra, exact package choice determined by
the feasibility spike. Model and index manifests must include revision, checksums,
license, dimensions, preprocessing, catalog hash, and disk size.

Planned files:

```text
ghostlab/retrieval/late_interaction.py
scripts/build_late_interaction_index.py
scripts/run_late_interaction_challenger.py
tests/test_late_interaction.py
configs/assets/late_interaction_rescue_v1.json
configs/experiments/w2_late_interaction_rescue_v1.json
```

### 7.9 W2-EXPAND: corpus-grounded query expansion

Reserved IDs:

- `query.catalog_prf.v1`
- `query.query2doc_local.v1`
- `query.expansion_guard.v1`

Hypothesis: catalog-grounded pseudo-relevance feedback or local query expansion can
add missing product terminology. Expansion must be guarded because unsupported terms
can drift away from explicit constraints.

Implementation ladder:

1. deterministic pseudo-relevance feedback from high-confidence terms in the top
   sparse candidates;
2. category/attribute synonym expansion from the catalog ontology;
3. optional local Query2Doc-style generated passage only if a pinned offline model
   can be packaged and its gain exceeds the simpler controls.

Expansion terms carry source and confidence. Explicit user terms, negations, hard
constraints, and replacements always outrank generated terms. The guard disables
expansion on low agreement, contradiction, or excessive drift.

Planned files:

```text
ghostlab/state/query_expansion.py
ghostlab/retrieval/pseudo_relevance.py
scripts/run_query_expansion_challenger.py
tests/test_query_expansion.py
tests/test_pseudo_relevance.py
configs/experiments/w2_query_expansion_v1.json
```

Dependencies: core for PRF/ontology variants; optional local model dependency only
for the generated variant.

### 7.10 W2-DIVERSIFY: conditional early-turn diversification

Reserved IDs:

- `ranking.mmr_early.v1`
- `ranking.facet_diversity.v1`

Hypothesis: early browsing turns may benefit from diverse recommendations and
candidate evidence, while buying turns with hard constraints should remain
relevance-first.

Implementation:

- rerank only a bounded head with MMR or catalog-facet coverage;
- activate only when observable uncertainty/browsing evidence passes a threshold;
- preserve constraint-satisfying candidates;
- stop diversifying after intent becomes specific;
- compare against a depth-matched relevance control.

Diagnostics include Top-10 facet coverage, target-rank movement, early-turn browsing
reward, and buying/override regressions.

Planned files:

```text
ghostlab/retrieval/diversify.py
scripts/run_diversification_challenger.py
tests/test_diversify.py
configs/experiments/w2_diversification_v1.json
```

Dependencies: core for metadata-facet diversity; dense or late-interaction assets only
for semantic pairwise similarity variants.

### 7.11 W2-ROUTER: calibrated conditional execution and fallback

Reserved IDs:

- `routing.calibrated_observable.v1`
- `guard.component_fallback.v1`

Hypothesis: expensive or fragile heads should run only where observable evidence
predicts a benefit. An attribute-scoped fallback can be safer than the current
whole-session fallback after a targeted intent correction.

The router predicts benefit/regret using only runtime-safe signals and is trained
fold-locally. It must include an always-base option. Report oracle headroom, routing
precision, routing regret, fallback frequency/reason, scenario deltas, and the cost
of false routing. Confidence is calibrated inside training data; it is not inferred
from the target at runtime.

Planned files:

```text
ghostlab/policy/calibrated_router.py
ghostlab/runtime/component_fallback.py
scripts/run_calibrated_router_challenger.py
tests/test_calibrated_router.py
tests/test_component_fallback.py
configs/experiments/w2_calibrated_router_v1.json
```

Dependencies: core or `gbdt` depending on the selected compact router.

### 7.12 W2-HPO: BOHB/Hyperband inside surviving structures

Reserved IDs:

- `search.hyperband.v1`
- `search.bohb.v1`

HPO is an optimization technique, not a runtime component. It tunes numeric
parameters only after structural technique screening. It cannot choose among invalid
or leakage-prone configurations.

Initial parameters include EIG thresholds, candidate depths, fusion weights,
reranker depth/capacity, expansion limits, diversity weight, and routing threshold.
The autonomous controller owns budgets, fidelity promotion, deterministic seeds,
and immutable trial records.

Dependencies: prefer implementing the small scheduler around existing primitives.
Add a third-party HPO framework only if it materially reduces code and its storage,
resume, and determinism behavior pass tests.

Planned files:

```text
ghostlab/optimization/hyperband.py
ghostlab/optimization/bohb.py
tests/test_hyperband.py
tests/test_bohb.py
configs/search/wave2_hpo_v1.json
```

## 8. Research basis and why these techniques are credible

The techniques are grounded in published retrieval, ranking, and sequential-decision
work, then constrained to this competition's small-data and offline-runtime setting:

- ProductAgent demonstrates structured product memory, candidate-product statistics,
  strategic clarification, and hybrid retrieval in a conversational product search
  system: [ProductAgent, EMNLP Industry 2025](https://aclanthology.org/2025.emnlp-industry.25/).
- Expected-value clarification is supported by neural EVPI work:
  [Rao and Daumé III, ACL 2018](https://aclanthology.org/P18-1255/).
- Catalog-aware attribute extraction and confidence-gated dynamic filtering have
  production e-commerce evidence:
  [Wayfair explicit attribute extraction, ECNLP 2024](https://aclanthology.org/2024.ecnlp-1.13/).
- Metric-sensitive ranking follows the LambdaLoss framework:
  [Wang et al., CIKM 2018](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/).
- Learned sparse and late-interaction candidates are based on
  [SPLADE v2](https://arxiv.org/abs/2109.10086),
  [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/), and
  [BGE-M3](https://arxiv.org/abs/2402.03216).
- Grounded expansion is informed by
  [Query2Doc](https://aclanthology.org/2023.emnlp-main.585/).
- Distillation with corrective expert labels is informed by
  [DAgger](https://proceedings.mlr.press/v15/ross11a) and
  [Expert Iteration](https://arxiv.org/abs/1705.08439).
- Conditional diversification uses the MMR mechanism:
  [Carbonell and Goldstein](https://kilthub.cmu.edu/articles/journal_contribution/The_Use_of_MMR_and_Diversity-Based_Reranking_for_Reodering_Documents_and_Producing_Summaries/6610811/1).
- Multi-fidelity allocation is based on
  [Hyperband](https://www.jmlr.org/beta/papers/v18/16-558.html) and
  [BOHB](https://proceedings.mlr.press/v80/falkner18a.html).

Published success is a hypothesis prior, not proof on the TechJam simulator. Each
mechanism still needs matched local validation.

## 9. Registry v2 and true on/off behavior

The current catalog records family, dependency extra, source, and status. Wave 2
needs a backward-compatible `catalog_v2.json` with enough metadata for autonomous
planning.

Required fields per technique:

```json
{
  "id": "question.candidate_eig.v1",
  "family": "question",
  "wave": 2,
  "availability": "planned",
  "default_enabled": false,
  "source": "ghostlab/policy/eig_questions.py",
  "config_binding": "question_variant=candidate_eig",
  "execution_class": "core",
  "fit_required": false,
  "assets": [],
  "requires": ["state.catalog_normalizer.v1", "retrieval.sparse"],
  "conflicts": ["question.fixed"],
  "compatible_families": ["ranking", "retrieval", "query"],
  "mechanism_tags": ["candidate_statistics", "information_gain"],
  "retest_triggers": ["state_parser_changed", "retriever_changed"],
  "estimated_resources": {"cpu": 1, "gpu": 0, "memory_gb": 2},
  "evidence_refs": []
}
```

Availability lifecycle:

```text
planned -> implementing -> available -> evaluated
                                  \-> invalid
evaluated -> selected | parked | interaction_reserve | retest_after_dependency
```

Rules:

- `planned` entries are documentation only and are never emitted as runnable
  candidates.
- `available` requires source, factory binding, disabled-path test, dependencies,
  and a smoke-valid preset.
- Missing extras/assets produce an explicit `UNAVAILABLE` planning record, not a
  silent fallback to another technique.
- Mutually exclusive choices remain typed enums. Truly additive techniques use
  explicit booleans/weights.
- Heavy imports stay inside the selected factory.
- Config canonicalization sorts unordered technique sets and emits a stable SHA-256
  candidate identity.
- Historical v1 IDs remain resolvable; they are never renamed in place.

### 9.1 Planned typed switches

The precise field names may be amended before implementation, but once a manifest is
evaluated their meaning is immutable. The intended v2 configuration surface is:

| Field | Type/values | Off behavior |
|---|---|---|
| `normalizer` | `off`, `catalog_v1` | Existing Wave 1 parsing remains byte-for-byte unchanged. |
| `constraint_confidence` | float in `[0,1]` | Ignored when normalizer is off. |
| `question_variant` | add `candidate_eig`, `reward_voi`, `distilled_joint` | Existing selected question policy. |
| `eig_candidate_k` | integer `50..400` | No candidate statistics are constructed when EIG is off. |
| `question_value_margin` | finite float | Ignored outside value policies. |
| `joint_policy_asset` | safe relative path or null | Null unless joint/distilled policy is selected. |
| `retrieval_route` | add `learned_sparse_union`, `late_interaction_union` | Existing selected retrieval route. |
| `learned_sparse_asset` | safe relative path or null | No learned-sparse import or index open. |
| `late_interaction_asset` | safe relative path or null | No late-interaction import or index open. |
| `query_expansion` | `off`, `ontology`, `prf`, `query2doc_local` | Query exactly matches the selected Wave 1 builder. |
| `expansion_max_terms` | bounded integer | Ignored when expansion is off. |
| `reranker` | add `reward_lambdamart`, `rank_ensemble` | Existing selected reranker. |
| `reward_ranker_asset` | safe relative path or null | No asset load outside the selected reranker. |
| `ensemble_assets` | bounded tuple of safe relative paths | Empty outside ensemble mode. |
| `diversification` | `off`, `facet_mmr`, `semantic_mmr` | Ranking is not reordered. |
| `diversification_weight` | float in `[0,1]` | Ignored when diversification is off. |
| `router` | `off`, `decision_list`, `calibrated` | Selected base path always executes. |
| `router_asset` | safe relative path or null | No router load when off. |
| `fallback_scope` | `session`, `component`, `attribute` | Existing behavior unless new guard is selected. |

Compatibility validation must reject orphaned assets/parameters as well as missing
ones. An experimental config must never silently coerce an invalid combination into
a different valid technique.

## 10. Autonomous system scope

### 10.1 What becomes autonomous

The campaign controller will automate:

1. catalog and manifest validation;
2. availability and dependency discovery;
3. candidate enumeration and canonical deduplication;
4. compatibility rejection with a recorded reason;
5. cache and asset readiness checks;
6. deterministic scheduling under CPU/GPU/memory limits;
7. grouped fold-local fitting and replay;
8. multi-fidelity racing;
9. failure retries and safe resume;
10. mechanism diagnostics and paired statistics;
11. interaction expansion and backward ablations;
12. immutable manifests, checkpoints, reports, and decision proposals;
13. a final ranked shortlist with uncertainty and complexity trade-offs.

### 10.2 What remains non-autonomous

The controller will not:

- invent or edit production source code;
- install arbitrary packages or download unpinned models;
- merge branches, commit, push, or delete worktrees;
- alter the frozen split after seeing outcomes;
- read F3 or private organizer data;
- overwrite a historical report;
- automatically promote or activate a champion;
- claim that the highest reused-development score is the absolute best possible
  algorithm.

Autonomous experimentation means deterministic orchestration of declared techniques,
not autonomous self-modifying software.

## 11. How autonomy works before challengers are finished

The controller is intentionally useful before every Wave 2 technique exists.

1. It starts from the available Wave 1 registry and validates campaign plumbing,
   cache keys, scheduling, failure recovery, evidence schemas, and interaction math.
2. Wave 2 entries remain `planned`; the candidate planner reports them as skipped
   with `reason=implementation_unavailable`.
3. A technique branch is developed and smoke-tested independently.
4. Its minimal implementation commits are integrated into
   `ghostlab/w2-integration` with the switch still off by default.
5. Registry status changes to `available` only after the source, tests, config,
   assets, and factory binding coexist in that integration commit.
6. The next frozen campaign pins that new commit and includes newly legal candidates.
7. Results from earlier commits remain valid historical evidence because every run
   records the code/catalog/config/split hashes.
8. Cross-family combinations begin only after the required techniques coexist in
   the integration worktree.

The runner must never reach into another worktree and import partially written code.
Worktrees isolate development; the integration commit defines the executable
experiment universe.

## 12. Autonomous controller architecture

Planned files in `techjam-wave2-autonomy`:

```text
ghostlab/campaign/
  __init__.py
  models.py             # immutable manifests, candidates, jobs and outcomes
  catalog.py            # registry v2 loading and availability resolution
  compatibility.py      # dependency/conflict/resource validation
  planner.py            # singles, pairs, triples, beam and ablation generation
  scheduler.py          # bounded local process scheduling and resource tokens
  runner.py             # state machine, checkpoints, resume and retries
  cache.py              # content-addressed paths, locks and integrity checks
  analyze.py            # paired metrics, interactions and scenario gates
  proposal.py           # shortlist only; never promotion
scripts/
  run_autonomous_campaign.py
  inspect_campaign.py
  resume_campaign.py
  verify_campaign.py
tests/
  test_campaign_catalog.py
  test_campaign_compatibility.py
  test_campaign_planner.py
  test_campaign_scheduler.py
  test_campaign_resume.py
  test_campaign_analysis.py
configs/campaigns/
  wave2_smoke_v1.json
  wave2_standard_v1.json
```

The package extends the current `ghostlab/optimization`, `ghostlab/research`, and
`ghostlab/evaluation` modules. It does not duplicate replay, statistics, search, or
the evidence store.

### 12.1 Campaign state machine

```text
DRAFT -> FROZEN -> READY -> RUNNING -> ANALYZING -> COMPLETE -> PROPOSED
                       |         |          |
                       v         v          v
                    BLOCKED    PAUSED     FAILED
```

- `FROZEN` means the manifest and all referenced hashes are immutable.
- `READY` means every selected technique and asset is locally available.
- `PAUSED` is resumable from atomic checkpoints.
- `FAILED` retains logs and partial results; it never masquerades as a miss.
- `PROPOSED` is a machine-generated shortlist, not a champion decision.

### 12.2 Frozen campaign manifest

Each campaign records:

```json
{
  "schema_version": 1,
  "campaign_id": "wave2_standard_v1",
  "parent_commit": "exact integration commit",
  "catalog_hash": "sha256",
  "dataset_hash": "sha256",
  "adaptive_split_hash": "sha256",
  "nested_split_hash": "sha256",
  "protected_holdout_access": "forbidden",
  "baseline_presets": ["champion_guarded", "metadata_gbdt"],
  "families": ["normalization", "question", "ranking", "retrieval"],
  "technique_versions": ["exact IDs"],
  "fidelity_budgets": {"f0": 200, "f1": 80, "f2": 24},
  "exploration_fraction": 0.2,
  "seeds": [20260826],
  "max_wall_seconds": 129600,
  "resources": {"cpu_jobs": 4, "heavy_model_jobs": 1},
  "promotion_rule": "proposal_only"
}
```

Any expansion after seeing results requires a new manifest version or a prospective
amendment describing the prior diagnosis and falsifiable next hypothesis.

### 12.3 Job identity, caches, and resume

Job identity is the hash of:

```text
code commit + technique catalog + canonical config + dataset + split + outer fold
+ inner selection procedure + seed + model/input assets + evaluator version
```

Cache classes:

- catalog preprocessing and normalized ontology;
- sparse/dense/late-interaction indices;
- fold-local candidate lists;
- fold-local feature matrices;
- counterfactual state/action labels;
- fitted fold models;
- replay outcomes and per-session metrics.

Every cached item has an integrity sidecar and dependency lineage. Fold-local caches
include the training-ID hash and may not be reused across a different fold. Writes
use a temporary path, fsync/close where appropriate, then atomic rename. File locks
prevent two workers from building the same asset. A failed cache never becomes
`ready`.

### 12.4 Resource scheduling

Each technique declares CPU, GPU, memory, model-asset, and exclusivity requirements.
The scheduler may parallelize independent lightweight folds but initially permits
only one heavy neural/model-index job. It stops admitting work before estimated
memory exceeds the host budget. Deterministic job order and seeds are retained even
when completion order differs.

## 13. Candidate-generation and combination strategy

The goal is broad, auditable interaction coverage—not naive exhaustive search.

### 13.1 Factor notation

| Symbol | Family |
|---|---|
| `N` | Catalog normalization and confidence gates |
| `Q` | Candidate EIG/reward-VOI questions |
| `J` | Joint action policy |
| `T` | Expert distillation |
| `R` | Reward-aligned LambdaMART |
| `E` | Rank/model ensemble |
| `S` | Learned sparse rescue |
| `L` | Late-interaction rescue |
| `X` | Grounded expansion |
| `D` | Conditional diversification |
| `C` | Calibrated router/fallback |

`H` (HPO) changes how parameters are selected; it is not a runtime factor.

### 13.2 Stage A: controls and mechanism gates

Before end-to-end search:

- reproduce both selected and fallback controls on the current code;
- prove exact disabled-path parity for every new technique;
- validate normalized extraction and override cases for `N`;
- validate question partitions/counterfactual regret for `Q`;
- validate swap-reward derivatives for `R`;
- validate unique recall for `S`/`L`/`X`;
- validate diversity without hard-constraint loss for `D`;
- validate routing oracle headroom for `C`.

Techniques that fail their mechanism gate are not granted large end-to-end budgets,
but their implementation and evidence remain preserved when technically sound.

### 13.3 Stage B: all standalone techniques

Run every available Wave 2 technique against:

1. the selected guarded control where compatible;
2. the metadata-GBDT fallback where the selected compiled path cannot expose the
   necessary switch;
3. the most relevant Wave 1 challenger dependency.

This prevents a new technique from appearing weak merely because an incompatible
component surrounds it.

### 13.4 Stage C: matched 2x2 interaction tests

For mechanisms `A` and `B`, run:

```text
base
base + A
base + B
base + A + B
```

Then compute:

```text
interaction(A, B) = score(A+B) - score(A) - score(B) + score(base)
```

The same calculation is stored per session, fold, scenario, and aggregate. A small
standalone loss does not eliminate a technique when a dependency-mediated interaction
is plausible.

Mandatory dependency pairs include:

- `N+Q`, `N+X`, `N+D`, `N+C`;
- `Q+R`, `Q+J`, `Q+T`;
- `R+E`, `R+S`, `R+L`, `R+X`;
- `S+X`, `S+E`, `S+C`;
- `L+E`, `L+C`;
- `X+C`, `D+C`;
- each Wave 2 family with relevant Wave 1 parked question, dense, fusion,
  cross-encoder, structured-query/filter, profile, and neural-rank techniques.

All other compatible pairs enter the coverage queue unless they are behaviorally
duplicated or fail a hard gate.

### 13.5 Stage D: triples and higher-order combinations

Triples are generated from:

- positive or uncertain pairs;
- complementary pipeline stages;
- dependency chains such as `N+Q+R`, `N+X+S`, `Q+J+T`, and `S+R+E`;
- a reserved sample containing one mildly negative standalone technique;
- crossovers of Pareto-front candidates.

Higher-order candidates use bounded beam search. Keep the top candidates by paired
OOF reward, uncertainty, scenario safety, novelty, and resource cost, plus a 20%
exploration reserve. Candidate size is not capped at three: configurations of two,
four, or more techniques remain eligible when evidence supports them.

Every finalist receives complete backward ablation and add-back testing. Passenger
components are disabled in the final preset but remain in the library.

### 13.6 Pruning rules

Permanently reject a version only when it is:

- invalid or leaking;
- incompatible with the organizer contract;
- behaviorally identical to a cheaper candidate;
- catastrophically harmful under a prospectively declared bound;
- unable to meet immutable offline, memory, latency, or asset limits;
- irreproducible after a bounded retry/audit.

Otherwise use `parked`, `interaction_reserve`, or `retest_after_dependency`. Randomly
fully evaluate a declared sample of pruned candidates to estimate pruning regret.

## 14. Multi-fidelity procedure

Fidelity controls compute, not the definition of success.

| Level | Purpose | Allowed conclusion |
|---|---|---|
| F0 | Unit/mechanism checks, deterministic small-session smoke, recall and parity diagnostics | Invalid, duplicate, catastrophic, or continue. Never champion evidence. |
| F1 | Multiple scenario-balanced development subsets/folds with matched controls | Racing and uncertainty update. Exploratory only. |
| F2 | Complete frozen five-outer-fold OOF procedure with inner selection | Comparable candidate-selection evidence. |
| F3 | One frozen candidate on sealed 50-session holdout | One-shot confirmation only, outside autonomous search. |

F0/F1 samples are fixed before the campaign and rotated or stratified prospectively;
the runner cannot choose the easiest subset after observing a score. Promising and a
declared audit sample of pruned candidates advance. The final campaign shortlist is
based on F2, not extrapolated F0/F1 scores.

## 15. Anti-overfitting and statistical protocol

### 15.1 Nested selection

For each outer fold:

1. build normalization vocabularies and all data-derived assets on outer training;
2. perform structural/parameter selection using inner folds only;
3. fit the selected components on the complete outer-training partition;
4. evaluate once on the outer validation sessions;
5. stitch all outer predictions into one OOF record.

The complete automated procedure—not just one model—is what nested validation must
represent. Fold-local caches are keyed by training IDs to enforce the firewall.

### 15.2 Required evidence

For candidate versus matched control report:

- technical score, HitRate@10, MRR, MTTC, failures, and conversion-turn histogram;
- per-session paired reward delta;
- 95% paired bootstrap interval and paired randomization p-value;
- wins/ties/losses;
- five outer-fold scores, standard deviation, and worst fold;
- Buying, Browsing, Intent Override, and Boundary metrics;
- target-rank crossings into Top-10 and rank 1;
- latency p50/p95, peak memory, model/index size, token/API use, and offline status;
- mechanism-specific diagnostics and fallback/routing reasons;
- code, config, asset, dataset, split, and result hashes.

### 15.3 Multiple comparisons and selection pressure

- Predeclare finite family budgets and hypotheses.
- Treat uncorrected campaign results as discovery.
- For large comparable families, report a family-wise max-statistic/randomization
  adjustment or false-discovery analysis in addition to raw paired intervals.
- Do not increase decimal precision to imply certainty unsupported by 150 sessions.
- Use simplicity/resource cost as the tie-breaker inside a declared uncertainty band.
- Keep F3 inaccessible until one complete candidate and analysis plan are frozen.
- Report a negative F3 result and make no post-F3 tuning.

Repeated nested CV cannot fully remove researcher selection from repeatedly examining
the same 150 sessions. The private 800-session organizer evaluation remains the
strongest unseen test.

## 16. Experiment records and diagnosis

Every evaluation writes an immutable result bundle:

```text
artifacts/campaigns/<campaign_id>/<candidate_hash>/
  manifest.json
  status.json
  compatibility.json
  folds/<fold_id>/fit_manifest.json
  folds/<fold_id>/session_results.jsonl
  metrics.json
  paired_control.json
  diagnostics.json
  resources.json
  logs/
```

The decision record must explain:

- the hypothesis and mechanism;
- what changed compared with the control;
- where target recall/rank/turn changed;
- which sessions/scenarios improved or regressed;
- whether the effect is standalone or interactive;
- what evidence suggests the causal pathway;
- why the technique is selected, parked, reserved, or invalid;
- which upstream changes should trigger retesting;
- exact implementation, config, report, and commit locations.

Correlation is not causation. “Why” claims should be labelled as mechanistic evidence,
ablation-supported inference, or unverified hypothesis.

## 17. Dependencies and installation policy

The base installation remains:

```bash
uv sync --group dev
```

Existing optional groups remain `gbdt`, `dense`, `neural`, and `all`. Wave 2 may add:

| Extra | Intended techniques | Admission gate |
|---|---|---|
| `learned-sparse` | SPLADE inference/index build | Pinned model loads offline on macOS; acceptable disk/RAM/latency. |
| `late-interaction` | ColBERT or BGE-M3 spike | Reproducible install, pinned revision, viable 50k-product index. |
| `wave2-all` | All accepted Wave 2 extras | Lockfile resolves on supported Python/macOS. |

Rules:

- Do not add a dependency until the technique passes a minimal feasibility spike.
- Pin model revision and write an asset manifest; do not commit downloaded caches or
  large generated indexes.
- Runtime never downloads from the network.
- API keys remain environment variables and must not be required for the selected
  offline path.
- `uv lock --check`, offline-load tests, and a clean-environment install are required.
- Prefer existing NumPy/Pydantic/scikit-learn components when functionality and
  accuracy are equivalent.

## 18. Implementation phases

### Phase 0: freeze and prove isolation

- Confirm all Wave 2 branches point to base commit `a2e7849`.
- Confirm all worktrees and `techjam-unified` are clean.
- Reproduce selected/fallback controls.
- Fingerprint data, split, catalog, configs, assets, and evaluator.
- Confirm F3 is inaccessible.

Exit gate: zero source changes in `techjam-unified`; exact control reproduction.

### Phase 1: autonomous contracts before autonomy

- Add catalog v2, campaign models, compatibility resolver, canonical IDs, state
  machine, and dry-run planner.
- Mark all unfinished Wave 2 entries `planned`.
- Run the planner on Wave 1 techniques only.

Exit gate: no unavailable technique is executable; deterministic legal/rejected
candidate lists; champion parity unchanged.

### Phase 2: independent mechanism implementations

Run concurrently by worktree:

- policy: `N`, `Q`, `J`, `T`;
- ranking: `R`, `E`;
- retrieval: `S`, `L`, `X`, `D`;
- autonomy: controller, cache, scheduler, analysis, proposal.

Each technique progresses through contract, implementation, tests, runner/config,
mechanism diagnostics, and a local decision record. Large families use separate
commits so integration can cherry-pick only cohesive pieces.

Exit gate: disabled parity, unit/contract tests, smoke evaluation, exact files and
dependencies documented; no champion claim.

### Phase 3: library integration

- Cherry-pick sound reusable technique commits into `ghostlab/w2-integration`.
- Resolve them through typed factories; defaults remain off.
- Register `available` only after all referenced pieces coexist.
- Re-run complete existing tests and selected compiled parity after every family.

Exit gate: one integration folder contains every valid technique and no abandoned
worktree is needed to execute it.

### Phase 4: mechanism and standalone campaign

- Run Stage A diagnostics and Stage B standalone comparisons.
- Record explicit failure causes and retest dependencies.
- Grant interaction reserve to mechanism-supported mild losers.

Exit gate: every available technique has a matched control and a decision state.

### Phase 5: interaction campaign

- Run compatible pairs and matched 2x2 analyses.
- Expand evidence-supported triples and higher-order beam candidates.
- Maintain 20% exploration and pruning-regret audit allocation.
- Apply HPO only inside surviving structures.

Exit gate: finalist shortlist with component interactions, uncertainty, scenario
safety, and resource evidence.

### Phase 6: confirmation and compilation

- Run complete F2 nested procedure for finalists.
- Perform backward ablation, add-back, seed stability, parity, offline, runtime,
  memory, and integrity audits.
- Refit exactly one frozen candidate on all 150 adaptive sessions.
- Compile to a small runtime policy and prove research/compiled/starter parity.

Exit gate: one candidate proposed for human approval; F3 still untouched.

### Phase 7: unified library update

- Merge every sound Wave 2 implementation into `techjam-unified`, disabled by
  default unless selected.
- Add all configs, manifests, tests, reports, decisions, dependencies, file maps,
  and retest triggers.
- Update the complete operations guide and catalog.
- Activate only the validated winner in a new champion preset.

Exit gate: teammates need one folder to install, switch, combine, reproduce, and
retest Wave 1 and Wave 2 techniques.

### Phase 8: guarded final validation

Only after explicit human approval:

- freeze one candidate and primary analysis;
- make F3 available to the final integration worktree only;
- execute once and append the access/result record;
- make no post-result tuning;
- await organizer-private evaluation.

## 19. Testing and quality gates

### 19.1 Per-technique tests

- typed config validation and unknown-field rejection;
- enabled behavior and exact disabled parity;
- deterministic output for fixed input/seed;
- no optional import/model/asset access while disabled;
- path confinement and checksum enforcement;
- legal action/output normalization;
- edge cases, missing metadata, ambiguity, no preference, negation, and override;
- unit tests for the distinctive mathematical mechanism;
- failure and fallback behavior.

### 19.2 Autonomous runner tests

- planned techniques cannot become jobs;
- missing dependencies become explicit blocked/skipped records;
- conflicting and duplicate candidates are rejected deterministically;
- canonical hashes are order-independent where appropriate;
- interrupted jobs resume without duplicate evidence;
- corrupt/partial caches are rejected;
- fold-local artifacts cannot cross folds;
- resource limits prevent overlapping heavy jobs;
- racing and pruning are deterministic;
- an audit sample of pruned candidates is scheduled;
- all finalist component-off ablations are generated;
- controller cannot open an F3 path or promote a preset;
- synthetic campaigns recover a known single effect and pair interaction.

### 19.3 Repository gates

```bash
uv lock --check
uvx ruff format --check .
uvx ruff check .
uv run --frozen mypy ghostlab scripts
uv run --frozen python -m unittest discover -s tests -p 'test_*.py'
uv run --frozen python -m scripts.validate_guarded_compiled
uv run --frozen python -m scripts.audit_unified_consolidation
```

Commands may be adapted to the repository's final test entry points, but standards
must not be silently weakened.

## 20. Code-quality standard

Wave 2 should be advanced research without becoming an over-engineered framework.

1. Extend existing modules when ownership is clear; create a module only for one
   cohesive new responsibility.
2. Prefer immutable Pydantic models/dataclasses, explicit factories, and pure
   functions over dynamic plugin loading or global registries.
3. Keep runtime dependency direction toward small contracts; research modules may
   depend on runtime components, never the reverse.
4. Keep file I/O at orchestration boundaries. Mathematical logic accepts values and
   returns values.
5. Use one authoritative evaluator and metric implementation.
6. Avoid speculative abstractions, generic base classes with one implementation,
   duplicated runners, hidden fallbacks, and large switch statements.
7. Add the smallest code that implements the tested mechanism; delete abandoned
   code before library inclusion, while preserving evidence through Git and reports.
8. Document non-obvious invariants and public contracts, not every line.
9. Type public interfaces, validate external data, and fail with actionable errors.
10. Every change must explain why its complexity is justified by correctness,
    reuse, or measured performance.

Reject generated-looking code that duplicates helpers, invents unused layers,
silently catches exceptions, changes unrelated files, fabricates evidence, or lacks
tests for its core claim.

## 21. Runtime, storage, and campaign budget

Known historical evidence suggests:

- a full 200-session keyword preset takes roughly tens of seconds on the current
  machine;
- hundreds of naive keyword combinations take hours;
- dense/neural index construction adds significant one-time cost;
- a broad pruned Wave 2 campaign should reserve approximately 18–36 hours, with a
  48-hour safety window after all techniques are available.

These are planning estimates, not guarantees. The autonomous manifest records actual
wall time by technique, fold, asset build, and evaluation. Cache reuse and mechanism
gates should prevent repeated model/index work. Unlimited time does not justify
unlimited reuse of development labels.

Do not commit:

- virtual environments;
- downloaded model caches;
- large generated indices;
- temporary checkpoints;
- raw logs containing unnecessary session text;
- secrets or credentials.

Commit small runtime assets only when license, size, provenance, and checksums are
acceptable. Otherwise provide reproducible builders and manifests.

## 22. Failure handling and recovery

- Every worker writes heartbeat, attempt, last completed stage, and failure class.
- Transient resource/process failures receive a bounded retry.
- Deterministic code/data failures do not retry endlessly.
- A failed candidate is excluded from score comparison and reported as failure, not
  assigned an artificial miss score unless the official runtime itself failed.
- Resume reads only verified atomic checkpoints.
- Campaign cancellation leaves completed evidence intact.
- Integration regressions are fixed in a new commit; do not destructively reset the
  immutable base or rewrite reports.
- A technique branch may be deleted locally only after its commits are integrated or
  otherwise recoverable and the worktree is clean.

## 23. Human review and final decision

The autonomous proposal contains:

- Pareto frontier for technical score, uncertainty, runtime, memory, and assets;
- best simple and best unrestricted candidates;
- fold/scenario stability and paired statistics;
- mechanism diagnostics and interaction graph;
- backward ablations and passenger components;
- known risks, unsupported routes, and retest triggers;
- exact reproduction command and hashes.

Human approval means reviewing this evidence and choosing whether one frozen
candidate may proceed to compilation/F3. It does not mean manually choosing every
experiment. The reviewer may reject a proposal for leakage, instability, complexity,
runtime, weak evidence, or unclear mechanism, but may not tune it against F3.

## 24. Definition of done

Wave 2 is complete only when:

- the original `techjam-unified` commit remains recoverable and unchanged;
- every reserved technique has an explicit state: implemented/evaluated, parked,
  invalid with evidence, or not implemented with reason;
- every sound implementation exists in one unified repository behind a true switch;
- all dependencies, assets, exact file locations, commands, and retest triggers are
  documented;
- the autonomous controller can plan, execute, resume, analyze, and propose a bounded
  campaign without touching F3 or changing the champion;
- all available techniques are tested standalone and in the declared interaction
  coverage process;
- winners receive backward ablation and complete F2 validation;
- all existing and Wave 2 tests, parity, integrity, offline, runtime, and packaging
  gates pass;
- one frozen candidate is proposed with honest uncertainty;
- library inclusion and champion activation remain separate decisions.

## 25. Immediate next action after approval

Do not implement a model first. Implement and validate the registry-v2/campaign
contracts and dry-run planner in `techjam-wave2-autonomy`, while the policy, ranking,
and retrieval worktrees begin only their smallest mechanism slices. This lets every
new technique enter the same evidence pipeline as soon as it becomes available and
prevents another manually coordinated collection of incompatible reports.

Before that work begins, review and freeze this document. Any material change to the
technique set, split use, search budget, promotion rule, or protected-data policy is
recorded as a new document version or prospective amendment.
