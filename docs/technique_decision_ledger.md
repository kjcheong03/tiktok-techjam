# GhostLab Technique Decision Ledger

Date: 2026-08-26
Champion: `ghostlab_champion_linear_v1`
Protected holdout: sealed and not accessed

## Purpose and source of truth

This document explains what GhostLab has learned from its experiments, why the
current champion has its present shape, and which losing ideas remain eligible for
dependency-aware retesting. The authoritative machine-readable source is
`artifacts/evidence/technique_decisions.jsonl`; this document is its readable
companion. Raw per-phase reports remain the source for detailed sessions, metrics,
and configurations.

The ledger is intentionally append-oriented and versioned by technique ID. A later
worktree must create a new versioned decision instead of silently changing the
meaning of an old record. Run `python -m scripts.validate_decision_ledger` to check
identifiers, links, arithmetic, evidence paths, and the holdout firewall.

## Status semantics

| Status | Meaning |
|---|---|
| `PROMOTED` | Validated and active in the champion, or retained as the current research control. |
| `PARKED_STANDALONE` | Tested and not selected alone; preserve the evidence and retest only under its stated condition. |
| `INTERACTION_RESERVE` | Not strong enough to replace the control alone, but it supplies a signal required by a justified combination. |
| `RETEST_AFTER_DEPENDENCY` | The tested version lost or tied; a named upstream change could materially change that conclusion. |
| `NOT_TESTED` | A concrete hypothesis, not an implemented or validated result. |
| `TESTING` | An active run whose result is not yet a decision. |
| `INVALID` | Evidence cannot be used because of leakage, contract failure, corruption, or another validity failure. |
| `OUT_OF_SCOPE_WITH_REASON` | Deliberately excluded with the reason recorded. |

“Parked” does not mean deleted. It prevents repeatedly rerunning the same losing
configuration while preserving candidates that may become useful after a real
dependency changes.

## How the champion emerged

| Step | Candidate score | Delta from step control | Conclusion |
|---|---:|---:|---|
| Early multi-state + fixed dialogue | 0.679729 | +0.017334 over raw-history fixed at that phase | Structured state was useful in the early system. |
| Raw-history sequence | 0.753736 | +0.013045 over `multi_other_always` | Raw answers preserved more retrieval evidence than destructive summarization. |
| Feature-field BM25 weight | 0.776106 | +0.022370 | Field-aware sparse retrieval was the first large retrieval/ranking gain. |
| Three-field interaction | 0.789226 | +0.013120 | Title, category, and feature evidence were complementary. |
| Catalog quality prior | 0.800591 | +0.011365 | A small quality signal improved ordering without replacing relevance. |
| Selected pairwise learner | 0.817649 | +0.017058 | Fold-safe learning improved Hit@10 and MTTC over the fixed scorer. |

The all-development refit and compiled adapter score `0.819719`; that is a
deployment-fit measurement. The five-fold OOF score `0.817649` remains the honest
development estimate.

## Active champion components

| Technique ID | Role | Why active |
|---|---|---|
| `state.raw_history_v1` | Conversation evidence | Robustly preserved the user's wording and beat tested structured replacements. |
| `question.static_raw_sequence_v1` | Dialogue control | Remained the strongest tested complete question policy after ranking upgrades. |
| `retrieval.field_bm25_v1` | Candidate generation | Strong exact-match retrieval with validated field interactions and low runtime cost. |
| `ranking.catalog_quality_v1` | Soft prior | Added a stable `+0.011365` over the triple-field scorer. |
| `ranking.pairwise_linear_v1` | Final ordering | Added `+0.017058` OOF while remaining small, deterministic, and offline. |
| `system.champion_v1` | Compiled complete policy | Passed adapter parity, integrity, tests, and performance gates. |

The selected learned model was not simply the row with the largest scalar score.
The quality-only ablation scored `0.819207`, but had lower Hit@10 (`0.926667`) and
worse MTTC (`3.04`) than the selected two-feature model. Under the declared
within-`0.01` robustness rule, the two-feature model's Hit@10 (`0.933333`) and
feature stability took priority.

## What improved what, and why

### Conversation and state

- Asking questions is essential in this simulator: the post-ranking no-question
  control scored `0.211462` versus `0.800591` for the raw sequence.
- Raw history improved the strongest early complete policy because the catalog is
  lexical and user answers contain discriminative product language. Structured
  summaries can omit or normalize away those terms.
- Multi-value state still has causal value as an additional representation. It
  improved the early raw-history fixed control, and removing negative evidence from
  the structured policy reduced score by `0.024249`.
- Heuristic adaptive questioning did not beat the tuned sequence. The problem was
  not that adaptivity is inherently bad; the hand-authored uncertainty signals did
  not predict downstream action value well enough.

### Retrieval and routing

- Generic MiniLM dense retrieval had Recall@200 `0.62`, below sparse `0.733333`.
  However, sparse+dense union reached `0.846667`, including 17 dense-only target
  rescues. Dense therefore has complementary recall but poor standalone precision
  and ordering in the tested form.
- RRF and weighted fusion reduced complete-policy score because fusion cannot turn
  a weak dense head into a good ranker merely by searching weights.
- The observable route stump chose keyword for every OOF session. An oracle route
  score near `0.7461` showed theoretical headroom, but the tested observable
  features could not identify dense rescue cases.
- Field-aware BM25 worked because product features, categories, and titles carry
  different amounts of exact lexical evidence. Their interaction improved the
  ranking without adding a model asset.

### Ranking, filtering, and profile

- A fixed linear lexical reranker lost to the strongest raw-history control. Later,
  a fold-safe pairwise learner succeeded because it learned a small stable feature
  combination rather than imposing hand-authored weights.
- A small quality prior helped, but it remained subordinate to relevance. It should
  not be interpreted as “recommend popular products regardless of the query.”
- Hard structured filtering lost `0.006538`, consistent with false exclusions when
  parsing or catalog coverage is incomplete. Future filters must be confidence- and
  coverage-aware and fail open.
- Fixed profile priors degraded monotonically as their weight increased. Profile
  data should return only as a learned gated signal that yields to explicit current
  intent.

### Search procedure

- At equal tested budgets, deterministic grid search found `0.742330`, versus
  `0.735983` for adaptive allocation. Grid/beam remains the auditable control.
- This does not justify an unrestricted exhaustive search. Reusing 150 sessions for
  many choices creates selection overfit even if compute is unlimited.
- Interaction testing is therefore bounded by causal dependencies, then validated
  with grouped outer folds, backward ablations, and one sealed final holdout.

## Compatibility and interaction map

| Upstream technique | Meaningful partner | Reason to combine | Required control |
|---|---|---|---|
| Structured multi-state | Learned question policy | Typed missing/negative/override state provides legal action features. | Raw history retained in parallel. |
| Structured query | Strong dense retrieval | Dense models benefit from a grammatical intent channel while BM25 retains raw exact terms. | Raw-query sparse head preserved. |
| Strong dense retrieval | GBDT or cross-encoder | Dense-only rescues need a better Top-10 ordering mechanism. | Candidate recall before end-to-end testing. |
| Strong dense retrieval | Fusion or routing | Fusion/routing becomes meaningful only when both component heads are competitive. | Always-sparse and union controls. |
| GBDT | Profile prior | Nonlinear gating can suppress profile evidence when explicit session intent conflicts. | No-profile feature ablation. |
| Coverage-aware filter | Improved structured parser | High-confidence typed constraints can reduce false exclusions. | Fail-open fallback and coverage audit. |
| Learned question policy | Query construction + retrieval | A question's value is defined by how its answer changes candidates and final reward. | Fixed sequence and no-question/stop controls. |
| Catalog quality | Learned ranker | Quality can break relevance ties but must not dominate. | Quality-off backward ablation. |

Higher-order combinations are tested only after their component or dependency pair
has plausible evidence. For a winning combination, remove each component in turn;
if removing one does not reduce performance, that component is not credited.

## Worktree challenger explanations

### `exp/learned-question-policy`

This challenger learns an adaptive action, not one universal question order. At a
state such as “turn 2, material known, use case missing, candidate set diffuse,” it
compares every legal next question and a stop/no-question action. Counterfactual
replay supplies training rewards; inference sees only runtime-observable state.
Linear action values are tried before shallow trees or GBDT. The main risks are
learning a simulator quirk, leaking target-derived information, and overvaluing
questions whose benefits come only from a fixed continuation policy.

### `exp/dense-retrieval`

This challenger tests whether a retrieval-specific embedding model can convert the
17 observed dense-only rescues into stable candidate recall. It first measures
Recall@10/50/100/200, unique rescues, losses, latency, memory, and asset size. Only a
model that passes that gate proceeds to sparse union, fusion, routing, and complete
dialogue evaluation. This separates “semantic retrieval found something new” from
“the final ranker knew how to use it.”

### `exp/gbdt-reranker`

This challenger replaces only the final learned ordering model with a constrained
nonlinear ranker. It can learn interactions such as “dense similarity matters when
exact feature overlap is low” or “profile evidence matters only without an explicit
contradiction.” Training groups every turn from a session in the same fold, and
tree depth, leaves, features, and rounds are selected inside training folds. The
linear champion is the complexity control.

Outcome: `ranking.gbdt_v2` completed the minimum independent test under the frozen
nested procedure. The selected shallow metadata model scored `0.861417` OOF versus
`0.817649` for the two-feature linear control, with a positive paired bootstrap
interval and improvements in all five outer folds. It is promoted to integration,
not compiled into this worktree's champion. See `docs/gbdt_reranker_report.md`.

### `exp/cross-encoder`

This challenger jointly reads the query and each Top-20 candidate, which can resolve
phrased semantic constraints more precisely than independent embeddings. It is a
reranker, not a full-catalog vector database. Testing begins zero-shot and only
expands to Top-50 or fold-local fine-tuning after a positive result. Cold start,
p95 latency, memory, offline packaging, and licensing can veto promotion even if a
small development gain appears.

### `exp/query-construction`

This challenger keeps raw history but adds separate typed channels for positive
constraints, negatives, overrides, and a natural-language dense query. It tests
whether cleaner intent benefits sparse or semantic retrieval without deleting the
lexical terms that made raw history strong. The critical controls are raw-only,
structured-only, hybrid, negation-safe, and raw fallback.

### `ghostlab/integration`

This is not a sixth modeling idea. It receives only validated, minimal commits from
the prototype worktrees and runs the bounded interaction tournament. Standalone
winners, dependency-aware reserves, pairwise combinations, selected triples, and
backward ablations are compared under the same folds. The integration branch
produces one frozen candidate; prototype branches never access the sealed holdout.

## Record summary

| ID | Technique | Status | Core decision |
|---|---|---|---|
| D001 | `system.champion_v1` | `PROMOTED` | Recoverable complete-policy control. |
| D002 | `question.static_raw_sequence_v1` | `PROMOTED` | Strongest tested dialogue controller. |
| D003 | `question.heuristic_adaptive_v1` | `RETEST_AFTER_DEPENDENCY` | Lost; needs learned values or stronger state. |
| D004 | `question.learned_counterfactual_v1` | `NOT_TESTED` | Advanced challenger, not a current result. |
| D005 | `state.raw_history_v1` | `PROMOTED` | Preserve raw lexical evidence. |
| D006 | `state.structured_multi_v1` | `INTERACTION_RESERVE` | Additive typed channel for future interactions. |
| D007 | `query.hybrid_structured_v1` | `NOT_TESTED` | Planned raw+typed query construction. |
| D008 | `retrieval.field_bm25_v1` | `PROMOTED` | Current sparse candidate generator. |
| D009 | `retrieval.dense_minilm_v1` | `RETEST_AFTER_DEPENDENCY` | Complementary recall, poor tested end-to-end use. |
| D010 | `retrieval.dense_specialized_v1` | `NOT_TESTED` | Planned stronger semantic generator. |
| D011 | `fusion.rrf_weighted_v1` | `RETEST_AFTER_DEPENDENCY` | Retry only after dense improves. |
| D012 | `routing.observable_stump_v1` | `RETEST_AFTER_DEPENDENCY` | Tested router collapsed to always-sparse. |
| D013 | `filter.coverage_aware_v1` | `RETEST_AFTER_DEPENDENCY` | False-exclusion risk requires better parser/guards. |
| D014 | `profile.fixed_prior_v1` | `RETEST_AFTER_DEPENDENCY` | Fixed weights hurt; return only as gated feature. |
| D015 | `ranking.catalog_quality_v1` | `PROMOTED` | Small stable tie-breaking gain. |
| D016 | `ranking.linear_lexical_v1` | `PARKED_STANDALONE` | Hand weights lost. |
| D017 | `ranking.pairwise_linear_v1` | `PROMOTED` | Selected compact learned ordering model. |
| D018 | `ranking.gbdt_v1` | `NOT_TESTED` | Planned nonlinear challenger. |
| D019 | `ranking.cross_encoder_v1` | `NOT_TESTED` | Planned semantic Top-K reranker. |
| D020 | `search.evidence_allocator_v1` | `PARKED_STANDALONE` | Grid won the equal-budget comparison. |
| D021 | `search.grid_beam_control_v1` | `PROMOTED` | Auditable bounded search control. |
| D022 | `state.negative_evidence_v1` | `INTERACTION_RESERVE` | Helped structured state; isolate on future hybrid. |
| D023 | `question.no_question_v1` | `PARKED_STANDALONE` | Useful only as a stop-action control. |
| D024 | `retrieval.sparse_semantic_v1` | `NOT_TESTED` | Optional semantic alternative, not implemented. |
| D025 | `ranking.quality_only_v1` | `PARKED_STANDALONE` | Near-champion scalar score; lost robustness tie-break. |
| D026 | `ranking.gbdt_v2` | `PROMOTED` | Stable grouped OOF gain; advance to integration without changing the compiled champion. |
| D027 | `ranking.gbdt_deployable_v2` | `PROMOTED` | Audited 56-round refit passed deterministic deployment and packaging gates. |

## How future worktrees update the evidence

Each worktree should read this ledger before declaring its experiment. After a run,
it adds a new technique version or a new decision record containing:

- the parent policy and falsifiable hypothesis;
- the actual mechanism, dependencies, and compatible partners;
- the evaluation split and sample count;
- candidate, baseline, delta, Hit@10, MRR, and MTTC where applicable;
- a causal diagnosis, including failure modes and scenario regressions;
- exact report paths and retest conditions.

Raw reports are never replaced by the ledger, and the ledger never substitutes for
fold-safe validation. It is the durable map between experiments: what changed,
what happened, why the result is plausible, and what must change before trying it
again.
