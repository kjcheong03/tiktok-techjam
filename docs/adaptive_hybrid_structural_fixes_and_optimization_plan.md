# Adaptive Hybrid Structural Fixes and Optimization Plan

## Purpose

This document is the implementation plan for correcting the confirmed structural
gaps and testing the remaining bounded optimizations in the Track 4 Adaptive Hybrid
agent.

It is a companion to
[`adaptive_hybrid_1a_3b_implementation_process.md`](adaptive_hybrid_1a_3b_implementation_process.md).
It does not replace that document, change its accepted architecture diagram, or
make any of the six required capabilities optional.

The governing rule remains:

> TikTok fixes the required 1A-3B capabilities and workflow. GhostLab optimizes
> valid implementations, parameters and compatible additions inside that workflow.

The implementation will proceed phase by phase. Each phase must pass its declared
validation gate before the next phase starts. Candidate-pool changes are completed
before retraining so the 2,200-session corpus is replayed and fitted only once for
the final structural revision.

## Current baseline and preservation boundary

The current branch is `feat/adaptive-hybrid-1a-3b` in the
`techjam-adaptive-optimizer` worktree.

The following evidence is preserved as the pre-structural-change baseline:

- the 2,200-session source/scenario-stratified split;
- the fitted union and overloaded-Browsing GBDTs and fit receipts;
- `configs/adaptive_hybrid_1a_3b_2200_v1.json`;
- the completed ranker-fitting report;
- the architecture-safe campaign plan;
- 93 completed pre-change F0 evaluations; and
- the partial campaign log at
  `artifacts/logs/adaptive_hybrid_campaign_pre_structural_f0.log`.

The stopped campaign's best raw F0 score was `0.837735`. It is useful baseline
evidence only. It cannot be treated as a final winner after candidate generation,
constraint authority or union features change.

The existing guarded champion remains the frozen comparison control and complete
precision fallback. No implementation phase may overwrite its assets or activate a
new candidate automatically.

## Fixed architecture invariants

The accepted 1A-3B topology remains static throughout implementation:

1. State V2 observes the current message and handles accumulation or override.
2. Runtime adaptation distils conflict-safe current context.
3. The observable router selects Buying or Browsing.
4. Buying remains precision-primary and Browsing remains diverse-dense-primary.
5. Keyword, independent-category and vector evidence enter one bounded 1B union.
6. A union-aware ranker examines every executable merged candidate.
7. Every successful turn reaches a literal local-LLM semantic activation decision.
8. Over-generality guidance can cut expansion and select a valuable legal question.
9. The response is validated, deduplicated and capped at Top 10.
10. Only the returned action is atomically committed.
11. The session/profile overlay is updated for the next turn.

The following are never valid GhostLab optimizations:

- removing or reordering a required capability;
- making a required capability permanently inactive or decorative;
- allowing a ranker or LLM to reintroduce a confirmed hard-constraint violation;
- using scenario labels, target IDs, evaluator outcomes, future answers or rewards
  as runtime inputs;
- treating missing catalog metadata as proof of a mismatch;
- updating history from unreturned recommendations or unsent questions;
- accessing F3/private evaluation data; or
- writing an active-candidate pointer from a research campaign.

## Consolidated gap assessment

| Area | Classification | Confirmed current state | Resolution in this plan |
|---|---|---|---|
| Constraint authority across routes | Critical compulsory behavior gap | Strict filtering applies only to Buying and covers only a bounded attribute set | Add route-independent constraint authority and a final output invariant |
| Observable 1A router | Partial optimization gap | Uses browsing markers and positive-constraint count | Add richer observable specificity/confidence signals without coupling authority to route |
| Diverse dense Browsing | Partial capability and validation gap | Multi-view E5 exists; selection is maximum relevance | Race maximum relevance, view-balanced and embedding-MMR selectors with real diversity metrics |
| Proactive preview cutoff | Compulsory behavior gap | Full E5 runs before the required overload decision | Add a cheap pre-E5 preview and a genuinely reduced overload retrieval path |
| Union-aware ranking | Material implementation gap | Source evidence is preserved by merge but discarded before GBDT features | Add source-, route- and constraint-aware union features and retrain |
| Semantic LLM ranking | Current configuration/optimization gap | Qwen reranks Browsing Top 10; Buying is skipped | Tune Qwen after structural fixes, then compare a bounded local-LLM shortlist |
| Buying precision preservation | Material implementation/evidence gap | Buying is keyword-primary but a mixed-pool GBDT can replace precision order | Add a precision-dominant Buying ranker or bounded residual and route-specific gates |
| Shared immutable context | Low-severity contract gap | Some stages use `V2StateView`; guidance/profile read mutable State V2 | Introduce one frozen per-turn context for all proposal stages |
| Runtime adaptation | Requirement present; optional coverage gap | Conflict-safe final profile reranking and profile updates exist | Race ambiguity-only profile query and question signals as optional additions |
| Diversity acceptance validation | Validation gap, not another architecture row | Query-view existence and three handcrafted family checks are treated too strongly | Measure concentration, coverage, redundancy and recall on held-out folds/templates |

Limited cross-category candidate reach has been demonstrated in three handcrafted
scenarios. This does not establish general cross-category relevance quality or
actual diversity.

## Required runtime interfaces

### Frozen per-turn context

All proposal stages will consume one immutable context created immediately after
State V2 observes the message:

```python
@dataclass(frozen=True)
class AdaptiveTurnContext:
    session_id: str
    turn: int
    current_message: str
    query_text: str
    constraints: tuple[ConstraintView, ...]
    intent_epoch: int
    shown_ids: frozenset[str]
    asked_attributes: tuple[str, ...]
    no_preference_attributes: frozenset[str]
    supplied_profile_terms: frozenset[str]
    profile_overlay: ProfileUpdate | None
```

Derived values may be cached in additional frozen objects, but no proposal stage
may retain a mutable reference to `StateBaselineV2`. Mutable state is touched only
when observing the message and when committing the validated selected action.

### Route-independent constraint authority

Constraint authority is orthogonal to Buying/Browsing intent. A request may be
open-ended Browsing while still containing mandatory budget, exclusion or product
requirements.

Every candidate/constraint pair will be classified as:

```text
CONFIRMED_MATCH
CONFIRMED_VIOLATION
UNKNOWN_METADATA
SOFT_PREFERENCE
```

Rules:

- a confirmed violation is removed on both routes;
- a confirmed match ranks before an unknown value when other evidence is equal;
- missing catalog metadata is `UNKNOWN_METADATA`, never a violation;
- soft preferences remain continuous ranking features;
- a free-form feature becomes authoritative only when parsing confidence and
  catalog evidence are sufficiently explicit;
- explicit exclusions and current-epoch corrections dominate profile evidence;
- constraint decisions record attribute, normalized value, status, provenance and
  reason code; and
- the final response validator repeats the confirmed-violation check after all
  ranking, semantic and profile stages.

Example:

```text
Request: "I'm still exploring, but it must be waterproof and under $100."
Route: Browsing
Authority:
  known price > $100                  -> remove
  known contradiction to waterproof  -> remove
  known waterproof and <= $100        -> confirmed match
  missing price/waterproof metadata   -> retain after confirmed matches
```

### Retrieval preview

The required overload preview occurs after routing but before full E5 expansion.
It uses only bounded observable evidence, such as:

- State V2 specificity and confidence;
- a shallow keyword result count and score distribution;
- bounded category/facet statistics; and
- turn and unresolved legal attributes.

The preview may not use target presence or scenario labels. When it detects clear
over-generality, it selects the reduced Browsing dense budget before E5 runs.

An overload turn still:

- executes the required dense capability on a bounded reduced pool;
- constructs the three-source union when sources are available;
- reaches the union ranker and local-LLM activation decision;
- returns safe recommendations where available; and
- asks the highest-value unresolved legal question.

It does not perform ask-only deferral by default and does not claim a cutoff when
the full configured retrieval budget was already spent.

### Dense evidence and selection

Dense retrieval will retain view-level provenance rather than only the maximum
score:

```python
@dataclass(frozen=True)
class DenseCandidateEvidence:
    parent_asin: str
    view_names: frozenset[str]
    per_view_ranks: Mapping[str, int]
    per_view_scores: Mapping[str, float]
    maximum_score: float
```

The compulsory Browsing slot supports bounded selectable implementations:

1. `multiview_max_relevance` as the matched control;
2. deterministic view-balanced selection; and
3. embedding MMR using pinned product embeddings.

View balancing is not a hard category quota. MMR may reduce redundancy only when
target recall is preserved. All selectors remain deterministic for identical
inputs and preserve source provenance.

### Source-aware union features

The union feature schema will extend catalog/query features with runtime-observable
evidence aligned to each candidate:

- route indicator;
- keyword/category/vector membership flags;
- source count;
- normalized per-source rank and reciprocal rank;
- normalized per-source score;
- explicit missing indicators for every source rank/score;
- weighted/RRF aggregate merge score;
- cross-source agreement;
- confirmed constraint-match count;
- confirmed violation count, which must be zero after authority filtering;
- unknown constraint count;
- soft constraint coverage;
- intent epoch/turn where justified; and
- bounded route-source interactions.

Target presence, target rank, scenario type, reward and future-answer values are
offline labels or evaluator outputs and are forbidden features.

### Buying precision contract

Buying must remain precision-dominant after the complete union is constructed.
The following implementations may race:

1. a route-specific Buying LambdaMART trained on exact merged pools;
2. a bounded residual blend over the verified precision ranking; and
3. a guarded constraint-aware Buying scorer adapted to State V2 and the new union
   evidence schema.

The matched deterministic control is sparse-first union ordering with hard
authority enforcement. Category/vector-only rescue products may enter the final
Top 10, but supporting evidence cannot freely overturn confirmed high-precision
matches. Every implementation must pass a Buying-specific non-regression gate.

### Profile extensions

The existing conflict-safe `ProfileUpdate` remains compulsory. The following are
optional GhostLab additions:

- one compatible profile-aware Browsing query view;
- suppression or down-weighting of clarification questions for attributes already
  confidently supplied by the profile; and
- a bounded profile match feature in the union ranker.

Profile-aware routing is not part of the first implementation because preference
tags do not reliably distinguish Buying from Browsing. It may be considered later
only if observable counterfactual evidence justifies it.

## Phase 0 - Freeze and audit the baseline

### Implementation

1. Record the current commit, branch and dirty-worktree inventory.
2. Hash the 2,200 configuration, split, models, receipts and reports.
3. Preserve the stopped F0 log and generate a small machine-readable summary.
4. Confirm no campaign or training process is running.
5. Run the existing focused architecture tests before editing runtime code.

### Validation gate

- All baseline assets parse and match their recorded hashes.
- The 2,200 fit receipts remain available.
- The partial F0 log contains 93 completed evaluations and the preserved best raw
  score.
- Existing user changes and unrelated artifacts remain untouched.
- F3 remains sealed.

### Output

`artifacts/reports/adaptive_pre_structural_baseline_manifest.json`

## Phase 1 - Immutable context and constraint authority

### Implementation

1. Add `AdaptiveTurnContext` and construct it once per turn.
2. Refactor router, dense query views, guidance and profile distillation to consume
   the frozen context or a frozen projection.
3. Add a catalog-backed `ConstraintAuthority` with explicit known/unknown status.
4. Apply authority filtering to the merged pool before route-specific ranking.
5. Apply the same invariant to the final ranked response before normalization.
6. Emit counts and reason codes for confirmed matches, violations and unknowns.
7. Keep `StateBaselineV2` mutation inside observation and atomic commit only.

### Focused tests

- precise Buying constraints still work;
- open-ended Browsing with a hard budget cannot return known over-budget products;
- explicit exclusions cannot reappear after union, LLM or profile reranking;
- missing metadata is retained after confirmed matches;
- free-form features are not made hard without sufficient evidence;
- current intent defeats conflicting profile values;
- Intent Override rebuilds authority in the new epoch;
- all proposal stages receive the same immutable turn identity; and
- returned recommendations/questions are the only committed action.

### Phase gate

- Zero confirmed violations in deterministic Buying and Browsing proof cases.
- No reduction in valid response count due solely to missing metadata.
- Existing State V2 override and atomic-commit tests pass.
- Architecture audit and official response contract pass.

## Phase 2 - Observable router and genuine overload cutoff

### Router implementation

Extend the deterministic observable router with bounded signals from the frozen
turn context:

- number and strength of positive constraints;
- hard versus soft constraint count;
- category-only versus product-specific request;
- exclusions and correction/override epoch;
- explicit uncertainty/browsing markers;
- query length and attribute coverage; and
- parser confidence/provenance where available.

The router returns route, calibrated confidence, abstention status and reason
codes. Retrieval margin/entropy is not placed before the router unless it comes
from an explicitly declared cheap preview that does not alter the canonical route
order.

### Cutoff implementation

1. Add a compulsory preview decision after route selection.
2. Separate preview budgets from normal retrieval budgets.
3. Select normal or reduced dense depth before full E5 retrieval.
4. Trace requested and consumed budgets.
5. Preserve recommend-and-ask behavior.
6. Preserve the complete precision fallback on failure.

### Focused tests

- both routes remain reachable;
- browsing language plus hard constraints remains Browsing while authority holds;
- corrections and exclusions affect confidence without hidden labels;
- low confidence safely abstains to precision;
- overloaded requests consume the reduced dense budget;
- non-overloaded requests consume the normal budget;
- the cutoff occurs before full E5 expansion;
- overload still reaches merge, ranker, semantic decision and question action; and
- preview failure returns a normalized precision response.

### Phase gate

- Route decisions are deterministic and fully explained by observable reason codes.
- No scenario/target/evaluator field enters router or preview features.
- Overload traces prove fewer dense candidates/views were evaluated than the normal
  configured path.
- No `overload:cutoff` reason is emitted when full expansion already occurred.

## Phase 3 - Diverse dense implementations and honest validation

### Implementation

1. Preserve view-level scores/ranks and candidate membership.
2. Implement deterministic view-balanced selection.
3. Bind the existing embedding MMR utility to pinned E5 product embeddings.
4. Make selection strategy, relevance weight and depths typed configuration fields.
5. Register selectors as valid implementations within the compulsory Browsing
   dense slot; none may delete dense retrieval.
6. Extend traces with view contribution and concentration statistics.

### Evaluation metrics

For each route turn, outer fold and dataset source, report:

- target Recall@50, Recall@100 and Recall@200;
- targets uniquely rescued and uniquely lost versus maximum relevance;
- unique candidate count;
- per-view unique contribution;
- maximum view share;
- category/family coverage at 50/100/200;
- maximum category share and concentration/HHI;
- facet coverage;
- mean and p95 pairwise embedding similarity; and
- warm latency and memory.

The existing three handcrafted scenarios remain diagnostic examples and are
described only as limited candidate-reach evidence. Because the previous
independent-template 1,000 was consumed during training, it is not reused as an
independent claim. Final diversity claims require outer-fold evidence and new
untouched templates or another genuinely held-out source.

### Phase gate

A selector may advance only if:

- Recall@200 does not materially regress overall or on Browsing folds;
- improvements are not driven by one prompt/category/source;
- redundancy or concentration improves materially;
- category/facet coverage improves across several sessions/folds; and
- no confirmed constraint violation is introduced.

If no selector passes, maximum relevance remains the implementation and the result
is recorded honestly. Failure of MMR does not reject the required diverse-dense
architecture.

## Phase 4 - Source-aware union ranking and Buying precision

### Implementation

1. Add a source-aware feature store that consumes `CandidateEvidence` directly.
2. Preserve exact feature parity between fitting and runtime scoring.
3. Add route-conditional union scoring.
4. Implement Buying precision-dominant candidates described above.
5. Keep strict authority filtering outside learned models as an invariant.
6. Extend candidate snapshots with all permitted source/constraint evidence.
7. Add route/source contribution traces through final Top 10.

### Training preparation

No model is fitted in this phase. The phase first freezes:

- router and preview behavior;
- dense selector candidates;
- authority behavior;
- candidate budgets;
- feature names, types and normalization; and
- exact candidate-pool fingerprints.

### Focused tests

- every union candidate has aligned source evidence;
- source-only candidates receive correct missing flags;
- category-only and vector-only rescue candidates are scored by the ranker;
- route-source interactions change only with observable route/evidence;
- fit/runtime feature matrices are byte-for-byte equivalent for fixtures;
- Buying confirmed matches cannot be displaced by known violations;
- residual weight bounds cannot eliminate precision dominance; and
- no forbidden label appears in the schema or serialized snapshot.

### Phase gate

- All complete merged pools are scored.
- Feature parity and leakage scans pass.
- The deterministic control and every learned implementation satisfy the Buying
  authority invariant.
- Candidate pools are frozen and fingerprinted for the single retraining boundary.

## Phase 5 - Optional profile-aware channels

### Implementation

1. Add at most one conflict-safe profile query view for ambiguous Browsing.
2. Add known-attribute question suppression using confidence and provenance.
3. Optionally expose a bounded profile match feature to the union ranker.
4. Register each addition independently so it can race alone and in compatible
   combinations.

### Focused tests

- identical ambiguous requests differ only when compatible profile evidence exists;
- explicit constraints disable conflicting profile influence;
- current-epoch session evidence outranks supplied profile tags;
- profile-derived questions are suppressed only with confident known evidence;
- no user/session state leaks into another session; and
- disabling every optional profile channel preserves compulsory `ProfileUpdate`.

### Phase gate

- Compulsory 3A behavior remains present with all options disabled.
- Every optional channel has an observable activation and explicit suppression case.
- No profile option bypasses authority, routing, union or semantic stages.

## Phase 6 - Structural integration and pre-fit validation

### Implementation

1. Integrate Phases 1-5 into the fixed coordinator.
2. Update typed configuration and architecture audit.
3. Update GhostLab bindings and classify all new techniques.
4. Reject unclassified catalog records or illegal combinations.
5. Add failure injection for every new component boundary.
6. Add a resumable progress/checkpoint format to the long campaign runner so each
   completed evaluation records candidate mapping, score, phase and decision.

### Validation gate

- Full unit and focused interaction suites pass.
- Architecture audit proves all required components and ordering.
- Failure injection always returns a valid precision response.
- Static typing, lint and deterministic replay pass for changed modules.
- Plan-only GhostLab output includes every new compulsory/optional technique.
- Campaign checkpoints can resume without reevaluating completed candidates.

No 2,200-session fitting begins until this gate passes.

## Phase 7 - One retraining boundary on the final structural pools

### Data use

The training corpus remains:

| Source | Samples | Status after fitting |
|---|---:|---|
| Official public development | 200 | Development/model-selection data |
| Public-like synthetic | 1,000 | Synthetic development data |
| Independent-template synthetic | 1,000 | Consumed training data; no longer independent validation |
| **Total** | **2,200** | **Grouped source/scenario-stratified fitting corpus** |

Complete sessions and target IDs stay inside one outer fold. The target is used
only as an offline label after candidate generation. Runtime collection receives
no target, scenario, future answer or reward input.

### Models to fit

1. source-aware union model;
2. Buying precision-dominant route model/residual candidates;
3. Browsing/overload ranker where still justified; and
4. profile-aware model variant only if its feature option passed Phase 5.

### Required OOF reports

Report overall and separately for:

- Buying;
- Browsing;
- Intent Override;
- Boundary as `HOLD_MORE_DATA` when too sparse;
- official public;
- public-like synthetic;
- consumed independent-template synthetic; and
- each outer fold.

### Promotion gates

- Overall Hit@10 does not materially regress.
- Overall MRR improves strictly for a learned replacement.
- Buying Hit@10 and MRR do not materially regress.
- Browsing target reach and MRR do not materially regress.
- Intent Override does not materially regress.
- No source-specific collapse is hidden by the combined corpus.
- Every bound model has a disjoint-fold receipt, exact feature schema and content
  hash.

A model that fails is still saved as rejected evidence but is not bound into the
candidate configuration.

## Phase 8 - Qwen tuning and bounded local-LLM comparison

### Ordering

LLM evaluation occurs only after retrieval, authority and union ranking are frozen.
This prevents a model comparison from being confounded by changing candidate
pools.

### Qwen study

Test Browsing depths `10`, `20` and `30` first, with a bounded weight grid and the
same prompt/product passage. Depth `5` may remain a cost control and `50` an upper
resource diagnostic, not the default search target.

Buying skip remains the evidence-backed control. A Buying uncertainty gate may be
tested only as a bounded challenger; always-on Buying LLM is not restored without
new evidence.

### Model-family study

Compare Qwen against two or three genuine local causal/instruction LMs that are:

- legally redistributable or reproducibly downloadable;
- runnable on the target Mac resource envelope;
- pinned by revision and directory hash;
- evaluated with identical candidate pools, passages and prompts; and
- incapable of adding or deleting candidate IDs.

Do not perform unrestricted model search. The shortlist is frozen before results
are observed.

### Metrics

- Hit@10 and MRR after semantic ranking;
- target movement into/out of Top 10;
- pairwise wins/ties/losses;
- ordering-change rate;
- invalid-score and fallback rate;
- deterministic replay;
- mean/p95 per-turn latency;
- peak memory; and
- complete-session TechnicalScore and scenario/source safety.

### Phase gate

Select the simplest feasible model/configuration that passes quality and resource
gates. If no alternative reliably beats Qwen, retain Qwen. If no deeper setting
beats Top 10, retain Top 10. Architectural eligibility requires a genuine active
local-LLM case, not a particular model family or depth.

## Phase 9 - GhostLab racing and final system validation

### GhostLab integration

The static architecture is materialized first. GhostLab then races:

- valid implementations within required slots;
- optional profile additions;
- dense selectors;
- union implementations;
- Buying precision implementations;
- Qwen/model/depth configurations that passed isolated feasibility; and
- existing compatible catalog techniques and their higher-order combinations.

Compulsory capabilities remain present in every submission-eligible candidate.
Parameters remain conditional on the selected implementation. Mutually exclusive
rankers/selectors cannot appear in one candidate.

### Racing gates

F0/F1/F2 promotion uses paired session rewards plus explicit non-regression gates:

- official public source;
- Buying;
- Browsing;
- Intent Override;
- constraint-violation count;
- architecture behavior;
- runtime feasibility; and
- fit-receipt eligibility.

Sparse Boundary results produce `HOLD_MORE_DATA`, not permanent rejection. A
combined score may rank candidates only after mandatory source/scenario gates pass.

### Final validation levels

1. schema and architecture audit;
2. unit contracts for every 1A-3B capability;
3. component interaction tests;
4. deterministic behavioral proof cases;
5. leakage and fold audit;
6. grouped end-to-end public/synthetic evaluation;
7. offline determinism, asset and license verification;
8. failure injection and fallback validation;
9. latency/memory/resource measurement;
10. official `reset(...)`/`respond(...)` entrypoint parity; and
11. atomic selected-action commit verification.

### Final acceptance

The implementation is complete only when:

- all required architecture capabilities are genuine and observable;
- no confirmed hard violation reaches a response on either route;
- overload traces prove a real pre-expansion cutoff;
- dense diversity claims are supported by measured coverage/concentration, not
  merely query-view existence;
- the union ranker directly consumes source evidence;
- Buying precision passes route-specific gates;
- the selected LLM is justified by a bounded same-pipeline comparison;
- profile updates remain conflict-safe and explicit-intent dominant;
- all outputs satisfy the official API contract;
- the complete selected configuration has reproducible hashes and receipts;
- the guarded champion remains recoverable; and
- activation is a separate reviewed action.

## Validation command families

Exact new script names will be finalized during implementation. The completed
workflow will provide commands in these families:

```bash
# Static and focused quality gates
uv run ruff check .
uv run mypy ghostlab
uv run pytest -q

# Focused architecture/behavior validation
PYTHONPATH=. .venv/bin/python scripts/validate_adaptive_hybrid.py
PYTHONPATH=. .venv/bin/python scripts/validate_adaptive_constraints.py
PYTHONPATH=. .venv/bin/python scripts/validate_adaptive_diversity.py

# Final structural-pool fitting
PYTHONPATH=. .venv/bin/python scripts/train_adaptive_hybrid.py --plan-only
PYTHONPATH=. .venv/bin/python scripts/train_adaptive_hybrid.py

# Bounded semantic comparison
PYTHONPATH=. .venv/bin/python scripts/compare_local_llm_rankers.py

# Architecture-safe optimization
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py --plan-only
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py ...
```

No long-running fitting or campaign command is part of a phase gate until its
preflight, asset checks and dry-run plan pass.

## Expected implementation artifacts

The implementation will produce or update:

- typed runtime/configuration code;
- architecture and technique audits;
- source-aware feature schema and parity fixtures;
- constraint-authority fixtures and reports;
- dense diversity evaluation reports;
- source/scenario-stratified split receipts;
- OOF ranker reports and model receipts;
- local-LLM comparison manifest and report;
- resumable GhostLab campaign checkpoints;
- final grouped end-to-end report;
- implementation report and README reproduction commands; and
- an explicit accepted/rejected technique table.

Generated logs, large model caches and temporary evaluation outputs remain outside
ordinary source commits unless they are intentional small reproducibility evidence.

## Change-control rules

- Do not edit the accepted architecture diagram.
- Do not silently reinterpret an optimization as a required capability.
- Do not hard-code an empirically uncertain selector/model as the only valid option.
- Do not broaden scope to UI, external databases, paid APIs or persistent
  cross-session learning during this implementation.
- Do not start retraining until candidate-pool and feature schemas are frozen.
- Do not start the final GhostLab campaign until fitting receipts and all focused
  gates pass.
- Preserve user changes and unrelated worktree files.
- Record every rejected alternative with its evidence and retest trigger.

## Ready-to-implement condition

Implementation can begin when:

1. the prior campaign is stopped and its evidence preserved;
2. the current branch/worktree is confirmed;
3. the baseline asset inventory is recorded; and
4. this plan is accepted as the scope boundary.

Conditions 1 and 2 are already satisfied. Phase 0 will machine-record condition 3.

## Implementation status

Phases 0-6 are implemented. Phase 7 was validated with a successful 200-session
structural smoke fit, then the full 2,200-session attempt was stopped at the user's
request after candidate collection and before model fitting. Phases 8-9 have complete
code paths, pinned assets, tests and plan validation; their expensive model-selection
and campaign executions are deliberately deferred until after the user starts the final
training run.

Current evidence and exact deferred commands are recorded in
`docs/adaptive_hybrid_1a_3b_implementation_report.md`. A deferred experiment is not an
implementation gap and must not be described as a promoted result.
Once this plan is accepted, implementation begins with immutable context and
constraint authority, not with retraining or another campaign.
