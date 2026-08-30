# Adaptive Hybrid Final Pre-Training Fixes and Holdout Plan

## Status and purpose

This document is the authoritative plan for the last structural corrections,
optimization safeguards, data split, final GhostLab search, final selection and
champion activation for the Adaptive Hybrid Track 4 system.

**Status (2026-08-31): the structural implementation is wired and undergoing final
validation. The long 1,650-session fit, full GhostLab campaign, one-time final selection and
champion activation have not been run.**

Implemented in this pass:

- verified 400-group lineage reconstruction, exact 1,650/550 partitioning,
  group-safe development folds and cluster-aware racing;
- genuine overload cutoff, conservative route-independent constraint authority,
  observable router evidence and behavior-based validation traces;
- source-aware union training, target-survival/source/route audits, diversity
  validation and bounded local-LLM depth/weight/family comparison; and
- development Top-3 freezing, predeclared gates and tie-breaks,
  a one-time lineage-cluster-aware final-selection runner and gated activation.

Final model training must begin only after the implementation checks below pass.

This plan supplements the accepted 1A-3B implementation process. It does not change
the required architecture diagram or make any required capability optional. Where an
older plan says that an overloaded turn should continue through the normal union and
LLM stages, this document supersedes it: a genuine cutoff must avoid those expensive
full-expansion stages.

The governing rule remains:

> TikTok fixes the required architecture and capabilities. GhostLab tunes their
> valid implementations and may add compatible techniques, but cannot remove,
> bypass or replace the required workflow.

## Final methodological decisions

The following decisions are fixed before implementation or final training:

1. Correct structural behavior before making the final dataset split.
2. Use a cross-source, lineage-grouped 75/25 split rather than splitting rows.
3. Use 1,650 development sessions for every form of fitting and selection.
4. Keep 550 sessions inaccessible to fitting, LLM selection, HPO, racing, pruning
   and challenger generation.
5. GhostLab ranks development candidates and freezes exactly three complete D
   configurations before the 550-session final selection set is opened.
6. Evaluate frozen A/B references, C and all three D configurations once on the same
   550 ordered sessions. A/B are explanatory only; each D is gated against C.
7. Choose among passing D configurations only by the frozen tie-break order. If none
   passes, retain C. Do not tune or replace any system after seeing the 550 results.
8. Never access F3 or organiser-private labels during development.
9. Never activate a candidate automatically from a campaign or final-selection script.

Evaluating three challengers and selecting among them makes the 550 a one-time final
selection set, not an unbiased holdout. The organiser-private evaluation remains the
unseen generalization test.

## Fixed 1A-3B architecture boundary

The normal successful runtime path remains:

```text
User message + supplied profile
  -> 2A State V2 accumulation/override and immutable turn context
  -> 3A conflict-safe session/profile context distillation
  -> 1A observable Buying/Browsing route
  -> 2B cheap over-generality preview
  -> normal multi-route retrieval
       Buying: precision-primary keyword + category/vector support
       Browsing: diverse dense-primary + bounded keyword/category support
  -> evidence-preserving candidate union
  -> route-aware union ranking
  -> selective literal local-LLM semantic ranking decision
  -> recommendations and optional highest-value legal question
  -> 3B response validation and atomic commit
```

The genuine overload path is a required controlled exception to full expansion:

```text
2B preview declares overload before full dense expansion
  -> reduced dense budget and bounded preview evidence only
  -> conservative route-independent constraint authority
  -> bounded Browsing-safe merge/ranker
  -> normal union ranker is SKIPPED
  -> semantic LLM execution is SKIPPED with an explicit cutoff reason
  -> safe recommendations + exactly zero or one valuable clarification question
  -> final validation and atomic commit
```

The LLM semantic-ranking capability remains present and compulsory in the
architecture. Selective invocation is permitted: a cutoff turn reaches an explicit
semantic activation decision, which records `skipped:overload_cutoff` rather than
executing an expensive model after the system has already decided to defer full
retrieval.

## Architecture invariants GhostLab cannot change

GhostLab may tune implementations, models, thresholds, budgets, weights and
compatible optional additions. It may not:

- remove State V2 accumulation or override handling;
- remove the observable Buying/Browsing route;
- remove precision-primary Buying or diverse-dense-primary Browsing;
- remove keyword, category and vector capability from the normal 1B pipeline;
- make the union ranker or semantic-LLM decision decorative;
- execute prohibited full-expansion stages after a genuine overload cutoff;
- make constraint authority depend on Buying versus Browsing;
- permit any ranker, profile prior or LLM to reintroduce a confirmed violation;
- update shown products, asked attributes or action history before final output;
- use target IDs, scenario labels, evaluator outcomes, rewards, future answers or F3
  information as runtime features;
- change heldout membership or promotion gates after seeing heldout results; or
- write `configs/active_candidate.json` automatically.

The technique catalog, racing, higher-order combination search, HPO, pruning,
champion/challenger logic and GBDT alternatives remain available. There is no fixed
six-technique ceiling on the eventual champion.

## Confirmed pre-training issues and dispositions

| Area | Current assessment | Required disposition before final training |
|---|---|---|
| Overload cutoff | Implemented: the bounded safe branch skips normal union, optional full rerankers and Qwen | Retain behavior-level trace assertions and retrain only the normal development ranker |
| Free-form constraint authority | Implemented: complete literal semantic tokens or approved equivalences are required; uncertain open-world evidence remains unknown | Retain adversarial shared-token, incomplete-equivalence, missing-metadata and contradiction tests |
| Router contract | Implemented: current-query evidence is separated from discounted history and exposes category-only, coverage, query-length, provenance, exclusion and correction signals | Tune bounded weights on development; keep retrieval statistics in guidance |
| Architecture validator | Implemented: cutoff validation checks executed stages, budgets and membership rather than labels alone | Keep prohibited-stage assertions in focused and end-to-end tests |
| Immutable turn context | Mostly implemented | Verify every proposal stage consumes the same snapshot and mutable state changes only at observe/commit boundaries |
| Source-aware union | Implemented in structural-v2 code but not yet established in the final fitted artifact | Retrain on development only and hash/verify the exact feature schema at runtime |
| Buying preservation | Not sufficiently gated | Add route-specific matched-control Hit@10/MRR gates and bounded residual influence |
| Browsing diversity | Mechanisms exist; general usefulness is not proven | Compare matched candidate pools using recall, redundancy, view/category/facet coverage and concentration |
| Local LLM choice | Three families and depth options exist, but current comparison is too small and weight is not adequately tuned | Run bounded development-only depth/weight tuning and same-pipeline family comparison |
| Semantic rescue | Configurable depths exist; rescue from ranks 11-30 is not proven | Audit pre/post target ranks, strong-head disruption, latency and fallback |
| Constraint-removal audit | Aggregate counts exist; target survival by stage is missing | Record evaluator-side target status before/after authority and final output |
| Route/source promotion gates | Current campaign stores aggregate rewards and partial source/scenario gates | Report Hit@10, MRR, MTTC, fallback, violations and latency by protected slice |
| Like-for-like evidence | Some historical reports compare different scopes | Require identical IDs, config and simulator conditions for every reported delta |
| Final artifacts | Hash and activation mechanisms exist; clean-checkout parity is incomplete | Bind dataset, split, schema, model, config, report and trace hashes and reproduce offline |

The development packager now freezes all three D file and canonical hashes, control C,
A/B reference hashes, promotion-gate and tie-break contract, and lineage-manifest hash
before final-selection access. The runner verifies all of them before creating its
access receipt. The generated finalist validator references the current 1,650-development
training report rather than an obsolete 2,200-session artifact.

## Explicit coverage of the supplied review

The plan is not limited to dataset splitting or model optimization. The following
matrix maps every item in the supplied architectural and training review to a required
change and an acceptance condition in this document.

### Core design fixes required before final training

| Priority | Review issue | Required change | Binding acceptance condition | Detailed section |
|---|---|---|---|---|
| Critical | Overload cutoff is not a real cutoff | Create a bounded safe branch; stop normal expansion; connect a schema-compatible Browsing-safe ranker or remove the unused hook; return safe recommendations and at most one clarification question | An overload trace proves reduced work and shows that normal union, optional full rerankers and semantic execution did not run | Fix 1 |
| Critical | Free-form constraint authority is too brittle | Remove only metadata-confirmed contradictions; distinguish closed-world structured fields from open-world descriptions; map approved semantic equivalents and treat uncertainty as unknown | Literal, approved-equivalent, missing, ambiguous and genuine-contradiction tests pass without deleting valid products | Fix 2 |
| Important | Router implementation and 1A specification do not fully align | Use one documented observable preliminary route contract; keep retrieval margin, entropy and overload statistics in post-route guidance unless the architecture explicitly adopts two-stage routing | Configuration, documentation and runtime reason codes enumerate the same features and counterfactual tests prove their effects | Fix 3 |
| Important | Architecture validator proves labels rather than behavior | Trace and validate actual stages, selected/consumed budgets, candidate membership and semantic execution | A supposedly cut-off turn fails validation if it executes any prohibited full-expansion stage | Fix 4 |

### Training and optimization safeguards

| Review check | Required change | Binding pass condition | Detailed section |
|---|---|---|---|
| Group-safe dataset splits and folds | Split complete lineage/profile groups across sources; keep each public row with its five public-like variants; group the independent five-row families; assign outer/inner folds by whole lineage | No lineage/profile family, sample ID or target ASIN crosses development/holdout or any train/validation boundary inside development | Cross-source lineage-safe 75/25 split |
| New union feature usage | Fit and run the keyword/category/vector source-aware schema with membership, score, rank, missingness, route and constraint evidence | Saved feature names/order/hash match the runtime feature store exactly | Fix 5 |
| Buying preservation | Compare against the matched precision control and bound residual influence | Buying Hit@10 and MRR remain inside predeclared tolerances with zero confirmed output violations | Fix 5 |
| Browsing diversity versus recall | Compare max relevance, view balance and MMR on identical E5 pools and sessions | Diversity/coverage improves without unacceptable Recall@50/100/200 or target-survival loss | Fix 6 |
| LLM selection | Tune a bounded Qwen depth/weight grid, then compare Qwen, Qwen3 and SmolLM2 under the same pipeline | Selection is based on Browsing quality, rescue, latency, memory and fallback—not model identity | Fix 7 |
| Semantic candidate rescue | Evaluate depths 10, 20 and 30 after the union ranker is frozen | The chosen setting improves MRR or rescues ranks 11-30 without excessive latency or destabilizing strong head results | Fix 7 |
| Constraint-removal audit | Record evaluator-side target status before and after authority and at final output, separated by constraint type | Zero confirmed output violations and no material increase in incorrectly removed targets | Fix 2 and Fix 4 |
| Like-for-like evaluation | Evaluate control and candidate on identical IDs, candidate pools, simulator conditions and configs | Every reported delta carries matching population/config hashes | Development-only GhostLab and one-time holdout sections |
| Route-specific promotion gates | Gate overall evidence plus Buying, Browsing, Intent Override and Boundary metrics, constraints, fallback and latency | No protected slice breaches its predeclared regression tolerance | One-time 550-session final selection |
| Final artifact verification | Hash data, split, schema, models, config and reports; reproduce from a clean offline checkout | Final report, selected config, fitted models and runtime trace form one matching reproducible hash chain | Phase 6 |

## Fix 1: genuine overload cutoff

### Required runtime behavior

The cheap preview must occur after the observable route and before normal dense
expansion. It may use only bounded runtime-observable evidence:

- shallow keyword result count and score shape;
- bounded category/facet counts;
- current constraint specificity and provenance;
- unresolved legal question attributes;
- turn number; and
- explicit broadness or uncertainty language.

It may not use target presence, target rank, scenario type, reward or future turns.

When overloaded:

1. Select the overload dense budget before E5 expansion.
2. Retrieve only the bounded safe candidate pool.
3. Apply route-independent constraint authority.
4. Rank using the existing Browsing-safe ranker interface or a deterministic
   bounded safe scorer with a compatible feature schema.
5. Do not call the normal full candidate merger, normal union GBDT, optional full
   rerankers or local LLM.
6. Return safe recommendations when available and at most one highest-value legal
   clarification question.
7. Record that full retrieval is deferred until a later user turn.
8. Fall back to complete precision behavior if the safe branch fails.

The currently unused Browsing-safe ranker must either be connected to this exact
branch and trained with its own compatible schema or removed. It must not be bound
to the source-aware union model if its input schema differs.

### Required trace fields

Each turn must expose immutable execution evidence such as:

- `preview_executed` and `preview_reason`;
- normal and selected dense budgets;
- actual dense requests and returned count;
- `safe_merge_executed`;
- `safe_ranker_executed` and backend;
- `normal_union_executed`;
- `semantic_decision_reached`;
- `semantic_executed` and skip reason;
- question action and returned recommendation count; and
- fallback reason.

### Acceptance conditions

- An overload proof trace consumes the reduced budget.
- The same trace has `normal_union_executed == false`.
- The same trace has `semantic_executed == false` and the expected cutoff reason.
- The bounded safe ranker or deterministic safe scorer is exercised.
- The response is normalized, constrained, deduplicated and capped.
- A non-overloaded matched request still follows the normal complete path.
- Tests fail if a supposedly cut-off turn invokes a prohibited stage.

## Fix 2: conservative constraint authority

Constraint authority remains orthogonal to route. Every candidate/constraint pair
must be classified as one of:

```text
CONFIRMED_MATCH
CONFIRMED_VIOLATION
UNKNOWN_METADATA
SOFT_PREFERENCE
```

Only `CONFIRMED_VIOLATION` removes a product.

### Closed-world and open-world rules

- Numeric budget with a known price may prove a match or violation.
- Explicit single-valued structured fields may prove contradiction only when the
  catalog field is present, trustworthy and semantically normalized.
- Missing fields are always `UNKNOWN_METADATA`.
- Free-form features, use cases and descriptions are open-world. Positive semantic
  evidence may prove a match, but absent literal overlap cannot prove a violation.
- An explicit exclusion becomes a violation only when the excluded value or a
  high-confidence approved equivalent is present.
- Uncertain semantic equivalence remains unknown and contributes a continuous
  ranking feature instead of deletion.
- Current-turn and current-intent evidence dominates a conflicting long-term profile.

Semantic normalization must use a bounded, versioned ontology or synonym map with
recorded provenance. An unconstrained LLM judgment may not become a hard deletion.

Example:

```text
Requirement: waterproof
Catalog text: GORE-TEX shell

approved high-confidence ontology link -> CONFIRMED_MATCH
no approved link                         -> UNKNOWN_METADATA
never literal non-overlap alone          -> CONFIRMED_VIOLATION
```

### Required auditing

Runtime traces record counts and reason codes without including the target ID.
Offline evaluation joins targets only after execution and records:

- target retrieved before authority;
- target authority status;
- target removed by authority, if applicable;
- target rank before and after ranking; and
- target present in final output.

### Acceptance conditions

- Literal matches, approved equivalents, missing metadata and real contradictions
  have focused tests.
- No valid open-text product is deleted solely because wording differs.
- Known budget and explicit exclusion violations are removed on both routes.
- Unknowns remain eligible but rank after confirmed matches when evidence is equal.
- Zero confirmed violations appear in final output.
- No downstream component can reintroduce a removed candidate.

## Fix 3: observable router contract

The 1A router must remain deterministic, observable and independent of hidden
evaluation signals. Replace early marker precedence with a documented evidence score
over the immutable turn context:

- explicit exploration/uncertainty language;
- positive constraint count and strength;
- high-confidence explicit versus inferred provenance;
- category-only versus product-specific attribute coverage;
- exclusions;
- correction/intent epoch;
- current query specificity; and
- configured abstention threshold.

Hard constraints do not automatically force Buying when the user is explicitly
exploring. Route-independent authority still enforces them. For example, “I am still
exploring, but it must be waterproof and under $100” may remain Browsing while known
violations are removed.

Retrieval margin, entropy and candidate overload belong to 2B guidance after the
preliminary 1A route. They must not be documented as initial router features unless a
formal two-stage routing contract is adopted and represented in the architecture.

### Acceptance conditions

- Buying, Browsing and precision abstention are all exercised.
- Reason codes enumerate the exact evidence used.
- Documentation, configuration and traces name the same signals.
- Corrections and exclusions affect the result without scenario labels.
- Hard constraints remain enforced regardless of route.
- Counterfactual tests change one observable signal at a time.

## Fix 4: behavior-level architecture validation

The validator must prove what executed rather than infer behavior from names or flags.
It must validate:

- one immutable turn-context identity across router, retrieval, guidance and ranking;
- preview position before dense expansion;
- selected versus consumed dense budgets;
- candidate membership before and after authority;
- normal union execution only on the normal branch;
- semantic decision and actual activation/skip reason;
- final recommendation IDs, uniqueness and count;
- question actually returned;
- atomic commit contents;
- fallback behavior; and
- absence of forbidden runtime labels.

An overload trace is invalid if union or semantic execution occurred, even if
`overloaded == true`. A normal trace is invalid if required normal stages were silently
bypassed without an explicit failure/fallback reason.

## Fix 5: source-aware union and Buying preservation

The structural-v2 code already defines keyword/category/vector membership, scores,
ranks, missing flags, route indicators, agreement and constraint/profile evidence.
Final training must prove the fitted model and runtime consume that exact schema.

### Artifact requirements

- Save the ordered feature-name list in the model and fit receipt.
- Save a feature-schema hash.
- Save the data split and candidate-pool collection hashes.
- Refuse runtime loading if model names/order differ from the runtime feature store.
- Correct report checks to use the actual `profile_term_match` feature name.

### Buying preservation

Buying remains precision-primary. The following may race on development data:

- route-specific Buying LambdaMART;
- bounded residual blending over the proven precision order; and
- guarded constraint-aware scoring using the new union evidence.

Category/vector rescue evidence may influence Buying but cannot overturn confirmed
constraints. The residual influence must remain tunable and bounded.

### Acceptance conditions

- Buying Hit@10 and MRR are reported separately against the matched precision control.
- Neither breaches its predeclared regression tolerance.
- Target survival and constraint violations are reported.
- The final artifact feature schema matches runtime exactly.

## Fix 6: diverse-dense Browsing evaluation

Compare these implementations on identical E5 pools and development sessions:

1. multi-view maximum relevance;
2. deterministic view-balanced selection; and
3. embedding MMR.

For each, record:

- Recall@50, Recall@100 and Recall@200;
- targets rescued and lost versus control;
- mean candidate similarity and redundancy;
- query-view coverage and concentration;
- category/facet coverage and concentration;
- results by turn, route, source and outer fold; and
- latency and memory.

Limited reach observed in handcrafted examples must be described as limited evidence,
not proof of general cross-category quality. A selector is promotable only if diversity
or cross-category coverage improves without unacceptable target-recall loss.

## Fix 7: bounded local-LLM selection and semantic rescue

LLM experiments occur only after retrieval, authority and union ranking are frozen.
Use the same candidate pools, prompts and development sessions for every comparison.

### Bounded search

- Tune Qwen depth over 10, 20 and 30.
- Tune a small predeclared set of semantic weights.
- Compare the selected Qwen setting with two genuine local alternatives already
  represented by available assets, such as Qwen3 and SmolLM2.
- Do not conduct an unrestricted model search.
- Record model revision/tree hash, prompt version, depth, weight, timeout, memory,
  latency, fallback and invalid-score counts.

### Rescue audit

For every setting, record:

- target rank before semantic ranking;
- target rank after semantic ranking;
- targets rescued from positions 11-20 and 21-30 into the returned head;
- targets pushed out of strong head positions;
- MRR and Hit@10 changes; and
- membership preservation.

The selected setting must improve semantic quality or produce meaningful rescues
without excessive latency, fallback or destabilization. Model identity alone is not
an acceptance criterion.

## Fix 8: profile/runtime adaptation safeguards

Conflict-safe session intent and separate caller-persistable profile updates remain
required. Session intent always dominates long-term profile evidence.

GhostLab may race these bounded optional additions mainly for ambiguous Browsing:

- one profile-aware query view;
- profile-aware clarification suppression;
- a bounded profile match union feature; and
- profile prior weight and confidence threshold.

Every accepted profile update must contain values, attributes, confidence, provenance
and intent epoch. Profile signals cannot create hard constraints unless explicitly
confirmed in the current session.

## Cross-source lineage-safe 75/25 split

### Exact partitions

| Source | Development | Untouched holdout | Total |
|---|---:|---:|---:|
| Official public | 150 | 50 | 200 |
| Public-like synthetic | 750 | 250 | 1,000 |
| Independent-template synthetic | 750 | 250 | 1,000 |
| **Total** | **1,650** | **550** | **2,200** |

### Preliminary recovered lineage evidence requiring audit

Preliminary read-only inspection of the current data indicates a stronger grouping
rule than individual hashing:

- the 200 public sessions appear to define 200 public lineage groups;
- each public session is followed by exactly five public-like variants sharing the
  same scenario, difficulty and complete profile;
- all 200 candidate public-to-five-variant groups passed the exploratory equality
  check;
- the independent-template set appears to contain 200 profile/scenario families of
  exactly five sessions each; and
- the independent profile families do not overlap the public/public-like profile
  families under the current structured profile fingerprint.

These are candidate lineage groups until a versioned reconstruction script produces a
passing audit report. The plan must not describe the 400 groups as verified provenance
before that artifact exists.

After verification, the public-derived lineage unit contains one public row plus its
five public-like variants. Fifty whole units enter holdout, yielding exactly 50 public
and 250 public-like sessions. The other 150 units yield 150 public and 750 public-like
development sessions.

Fifty whole independent five-session families enter holdout, yielding 250 independent
sessions. The remaining 150 families yield 750 development sessions.

Because the JSONL files do not contain an explicit generator `template_family` field,
the audit and manifest must label passing groups as **reconstructed lineage** based on
the verified ordering and exact scenario/difficulty/profile relationships, not as
original generator metadata. The limitation must remain disclosed even after the audit
passes.

### Deterministic allocation

1. Reconstruct all candidate groups and emit a machine-readable audit containing every
   rule, mismatch and member ID.
2. Fail closed unless the audit verifies the expected 200 public/public-like groups and
   200 independent groups with complete membership.
3. Stratify verified groups by source family and scenario; include difficulty where
   available.
4. Allocate whole groups with a versioned seed and stable hash ordering.
5. Use a deterministic largest-remainder allocation to reach exactly 50 heldout groups
   in each lineage collection while approximating scenario proportions.
6. Verify no profile/lineage group crosses partitions.
7. Verify no target ASIN crosses partitions. The current loader already rejects target
   duplication, but the split audit must repeat the invariant.
8. Assign development outer folds by whole lineage group, then expand each fold to
   member sample IDs.
9. Assign every inner training/selection, early-stopping, calibration and OOF partition
   by whole lineage group using only the outer-training groups.
10. Write one immutable manifest rather than duplicate derived JSONL files.

### Split manifest

The manifest must contain:

- schema and algorithm version;
- seed;
- source paths and SHA-256 hashes;
- lineage reconstruction rule and evidence counts;
- lineage-audit status, mismatch details and audit-report hash;
- group IDs and member sample IDs;
- development and holdout sample IDs;
- per-source/scenario/difficulty/profile-family counts;
- partition and complete-manifest hashes;
- target-disjointness and group-disjointness checks;
- group-safe outer-fold assignments and the algorithm used to derive inner folds; and
- the recovered-provenance limitation.

Proposed path:

`data/splits/adaptive_hybrid_lineage_75_25_v1.json`

### Enforcement boundary

Every development command must receive the manifest and explicit `development`
partition. This includes:

- candidate-pool collection;
- outer/inner folds;
- GBDT fitting and selection;
- dense selector selection;
- LLM family/depth/weight selection;
- GhostLab F0/F1/F2 evaluation;
- HPO and higher-order combination search;
- pruning and Top-3 development ranking; and
- all reports used to choose the frozen challenger.

Lineage grouping must remain intact inside development rather than ending at the
development/holdout boundary:

- no outer fold may split a lineage between model fitting and OOF validation;
- no inner fold may split a lineage between training, HPO selection, early stopping or
  calibration;
- stacked/ensemble OOF predictions must be generated from models that saw no member of
  the validation lineage; and
- cached folds are invalid if their manifest or lineage-audit hash changes.

Related variants are correlated observations. Racing, paired confidence intervals and
final statistical comparisons must therefore use lineage-cluster resampling: sample
whole lineage groups with replacement and include all member-session paired deltas for
each selected group. Session-level resampling that treats five related variants as five
independent units is prohibited for promotion evidence.

Fit receipts and campaign checkpoints must store the manifest hash and partition.
They must reject old 2,200-session checkpoints and any mixed-partition resume.

## Development comparison hierarchy

All development reporting uses four explicitly different roles:

1. **A — official stateless BM25:** organizer reference and explanatory baseline;
2. **B — tagged-best State Baseline V2:** native exact-parity reproduction of
   `coverage_adaptive_state_with_history + fixed_other`, retained as an explanatory,
   simulator-sensitive baseline;
3. **C — fixed adaptive architecture:** the complete compulsory 1A-3B workflow after
   its core development-only fit and before GhostLab search; and
4. **D — GhostLab challengers:** optional techniques, combinations and tuned values
   materialized on top of C without changing the compulsory workflow.

A, B and C must be evaluated on identical ordered development sample IDs. The final
development comparison includes A, B, C and the available D1-D3 finalists, but A and B
cannot be promoted because they do not implement the complete compulsory workflow.
Champion selection is restricted to C versus D. The one-time final-selection report
contains A/B/C/D1-D3 on the same ground, but A and B remain reference-only.

Every comparable A/B/C/D run uses one shared research evaluator entrypoint. Its report
contract hashes the ordered session IDs, catalog, shared harness, published evaluator,
seed and fixed turn/Top-K limits, and records the common profile, exception and response
normalization behavior. The published evaluator remains unchanged and is protected by
behavioral parity tests. Missing or unequal contracts invalidate the comparison.

The reproducible outputs are:

- `artifacts/reports/adaptive_baseline_a_development_1650.json`;
- `artifacts/reports/adaptive_baseline_b_development_1650.json`;
- `artifacts/reports/adaptive_hybrid_development_1650.json` for C;
- `artifacts/reports/adaptive_hybrid_top3.json` for D1-D3; and
- `artifacts/reports/adaptive_system_comparison_1650.json` plus `.md` for the unified
  human-readable table.

## Development-only GhostLab and frozen Top-3 proposal

GhostLab uses all 1,650 development sessions through its normal progressive fidelity
process and group-safe nested fitting procedures. Every outer/inner fold is assigned at
the verified-lineage level before expanding to sessions. Out-of-fold estimates reduce
selection bias inside development but are not called holdout results. Racing confidence
uses lineage-cluster resampling rather than independent session resampling.

The development campaign applies all predeclared development gates and freezes exactly
three entries in `frozen_proposals`. Each contains:

- immutable materialized config;
- config SHA-256 and canonical hash;
- model and feature-schema hashes;
- complete technique list and tuned parameters;
- development metrics and paired deltas;
- gate decisions;
- control identity; and
- split-manifest development hash.

After packaging, D1-D3 are re-evaluated on the same ordered full-development sessions,
catalog, seed and shared evaluator contract as A/B/C. These matched runs power the
dashboard challenger dropdown and like-for-like development table. Original GhostLab
fold/racing metrics remain separate selection evidence. The package also freezes C,
A/B definitions, gates and the complete tie-break order before any 550 access.

Proposed path:

`artifacts/reports/adaptive_hybrid_top3.json`

## One-time 550-session final selection

The holdout runner is a separate command and process. It consumes only:

- the frozen Top-3 report;
- the three hash-bound D configurations;
- the frozen matched control;
- the frozen official stateless BM25 reference definition;
- the frozen tagged-best State Baseline V2 reference definition;
- the split manifest's 550-session final-selection partition; and
- a predeclared promotion-gate configuration.

It refuses to run when:

- anything other than exactly three frozen challengers is requested;
- any config/model/hash differs from the proposal;
- the proposal was generated from a different development manifest;
- any final-selection ID appears in a fit receipt or campaign checkpoint;
- promotion gates are missing or were changed after the proposal freeze; or
- a final-selection access receipt or result already exists.

### Required reports

Report all six frozen systems on identical ordered sessions, with A/B marked
reference-only, C as control and D1-D3 as challengers. Report per-system metrics and
paired B-minus-A, C-minus-B and each D-minus-C delta for:

1. official public 50;
2. public-like synthetic 250;
3. independent-template synthetic 250;
4. combined 550 using natural source weights; and
5. source macro average so synthetic volume cannot hide an official-source issue.

Paired final-selection uncertainty must use lineage-cluster resampling. The 50 public
sessions and their 250 public-like variants represent 50 related public-derived
clusters, not 300 independent observations. The 250 independent-template sessions
represent 50 additional five-session clusters.

Metrics include:

- Hit@10;
- MRR;
- MTTC;
- recommended technical score;
- paired session reward and confidence interval;
- Buying/Browsing/Intent Override/Boundary slices;
- constraint-removal and final-output violations;
- target survival by stage;
- semantic rescue/loss counts;
- fallback and invalid-score rates;
- mean and p95 turn latency; and
- peak memory where measured consistently.

### Predeclared promotion logic

Promotion thresholds must be stored and hashed before final selection runs. They cover:

- zero confirmed final-output constraint violations;
- no breach of Buying Hit@10 or MRR non-regression tolerances;
- no protected route/scenario catastrophic regression;
- acceptable fallback and invalid-score rates;
- bounded latency and memory;
- positive overall evidence under the predeclared paired comparison; and
- complete architecture/artifact validation.

The exact numeric tolerances must be chosen from development/control variability and
committed before final-selection access. They may not be adjusted after observing the 550.

Possible outcomes are only:

```text
PROMOTE
RETAIN_CONTROL
```

An invalid run fails without a selection decision. The final-selection report is not
fed back into GhostLab.

## Phase-by-phase implementation and validation

### Phase 0: freeze current evidence

Implementation:

- record branch, commit and dirty-worktree inventory;
- preserve existing 2,200-session artifacts as historical baselines;
- hash current configs, models, receipts and reports;
- confirm no training/campaign process is running; and
- keep F3 sealed.

Gate:

- baseline manifest parses and hashes match;
- no user artifacts are deleted or overwritten; and
- existing evidence is labeled historical, not final.

### Phase 1: overload, authority and router fixes

Implementation:

- implement the real overload safe branch;
- connect or remove the Browsing-safe ranker cleanly;
- implement conservative closed/open-world authority;
- finalize the observable router contract; and
- maintain one immutable turn context.

Gate:

- focused counterfactual and failure tests pass;
- overload skips prohibited stages;
- equivalent wording and missing metadata are safe;
- both routes and abstention are exercised; and
- zero known violations reach output.

### Phase 2: validator and audit fixes

Implementation:

- add executed-stage/budget traces;
- add evaluator-side target-survival audit;
- add route/source metrics;
- validate atomic commit and fallback; and
- validate forbidden-label absence.

Gate:

- intentionally lying trace labels fail;
- intentionally executing union/LLM after cutoff fails;
- intentionally reintroducing a violation fails; and
- normal complete-path proof cases pass.

### Phase 3: lineage split and leakage enforcement

Implementation:

- reconstruct and validate the 400 candidate groups;
- emit and hash the reconstruction audit before accepting the 400-group claim;
- generate the immutable 1,650/550 manifest;
- add partition-aware corpus loading;
- replace individual-ID outer/inner folds with group-safe assignments;
- bind early stopping, calibration and OOF generation to group-safe partitions;
- bind split hashes to fits/checkpoints/reports; and
- reject legacy mixed-data resume.

Gate:

- exact per-source counts match;
- group and sample intersections are empty;
- target intersections are empty;
- no public row is separated from its five public-like variants;
- no lineage is split by any outer or inner train/validation boundary; and
- all development commands demonstrably load zero holdout IDs.

### Phase 4: development-only training and optimization

Implementation after Phases 1-3 pass:

- collect final candidate pools on development;
- train and select source-aware union/Buying-safe assets;
- run diversity/recall comparison;
- run bounded LLM selection and rescue audit;
- run GhostLab racing, pruning, HPO and combinations; and
- freeze exactly three development-selected complete D configurations.

Gate:

- every fitted asset has valid receipt/schema/split hashes;
- every comparison is like-for-like;
- every OOF result is produced by a model that saw no member of that validation
  lineage;
- racing confidence intervals use lineage-cluster resampling;
- protected route gates pass on development;
- exactly three D proposals plus A/B/C dependencies and tie-breaks are frozen; and
- no final-selection access receipt exists yet.

### Phase 5: one-time final selection

Implementation:

- evaluate frozen A/B references, C and D1-D3 on identical ordered 550 sessions, one
  catalog and one evaluator contract;
- generate per-source, route, combined and macro evidence;
- apply immutable promotion gates separately to every D versus C and select among
  passers using only the frozen tie-break order; and
- write a one-time access receipt.

Gate:

- exactly six frozen systems were evaluated: two ineligible references, one control and
  three challengers;
- exactly 50/250/250 sessions were consumed;
- paired uncertainty uses the verified holdout lineage clusters;
- no training or tuning occurred;
- all hashes and IDs match the frozen Top-3 proposal; and
- the outcome is one of the three allowed statuses.

### Phase 6: activation and clean reproduction

Implementation:

- manually activate only if status is `PROMOTE_FROZEN_CHALLENGER`;
- verify the active pointer and config hash;
- run focused tests and the full suite;
- reproduce from a clean offline checkout using pinned assets; and
- generate the final submission evidence bundle.

Gate:

- README commands reproduce the same config/report hashes;
- runtime trace, report, model and selected config correspond;
- no network/model drift occurs;
- rollback remains functional; and
- the public submission never claims the 550 was used for tuning.

## Required test matrix

| Test family | Minimum proof |
|---|---|
| Overload cutoff | Reduced work and no normal union/LLM execution |
| Normal path | Required normal stages execute or declare explicit fallback |
| Constraint authority | Literal, synonym, missing, ambiguous and genuine contradiction cases |
| Route counterfactuals | One observable signal changed per paired request |
| Intent override | Old constraints/history/profile influence cannot leak into new epoch |
| Atomic commit | Only returned IDs/question update state |
| Dense diversity | Recall, redundancy and coverage on identical pools |
| Buying preservation | Matched-control Hit@10/MRR and constraints |
| Union schema | Fit/runtime feature names, order and hash identical |
| Semantic rescue | Ranks 11-30 rescue and strong-head loss audit |
| Profile safety | Ambiguity-only use, explicit conflict suppression and provenance |
| Split leakage | Cross-source lineage, profile family, sample and target disjointness across final selection and every development fold |
| Group-safe nested validation | No lineage crosses outer/inner fitting, OOF, early-stopping, calibration or HPO boundaries |
| Cluster-aware statistics | Racing and final-selection intervals resample whole lineage groups rather than individual related sessions |
| Like-for-like | Identical session IDs/configuration for every delta |
| Freeze Top 3 | Exactly three development-selected D hashes are frozen before final selection |
| Final selection one-time | Frozen A/B/C/D1-D3, per-D C gates, immutable tie-breaks and access receipt |
| Artifact integrity | Clean-checkout hash and offline runtime parity |

## Required artifacts

Planned artifacts include:

- `artifacts/reports/adaptive_pre_final_baseline_manifest.json`;
- `artifacts/reports/adaptive_lineage_reconstruction_audit_v1.json`;
- `data/splits/adaptive_hybrid_lineage_75_25_v1.json`;
- development-only outer-fold manifest;
- source-aware union and safe-ranker models with fit receipts;
- dense diversity/recall report;
- local-LLM selection and semantic-rescue report;
- development GhostLab campaign report and checkpoint;
- development Top-3 diagnostic report;
- `artifacts/reports/adaptive_hybrid_frozen_proposal_v1.json`;
- predeclared promotion-gates config;
- one-time holdout access receipt;
- per-source/combined/macro holdout report;
- active-candidate verification report; and
- clean-checkout reproduction manifest.

Generated models, checkpoints and large reports are committed only when repository
policy explicitly permits them. Source code, small configs, manifests, tests and
documentation remain reviewable in Git.

## Stop conditions

Stop before final training when any of the following is true:

- overload still executes prohibited normal stages;
- semantic uncertainty can delete a valid candidate;
- validator checks labels instead of executed behavior;
- lineage reconstruction or exact partition counts fail;
- the reconstruction audit has not machine-verified the claimed 400 groups;
- any outer/inner fold or OOF partition splits a verified lineage;
- promotion statistics treat related variants as independent observations;
- any holdout ID appears in development evidence;
- fit/runtime feature schemas differ;
- protected Buying behavior has an unresolved regression;
- LLM comparison is not like-for-like;
- the final-selection package does not contain exactly three challengers; or
- artifact hashes do not form one reproducible chain.

Stop after final selection and retain C when no frozen D passes. Do not reopen GhostLab,
retune settings or introduce another challenger after seeing the 550 results.

## Definition of done

This plan is complete only when:

1. all required runtime and validator fixes are implemented and tested;
2. the immutable lineage-safe split contains exactly 1,650 development and 550
   one-time final-selection sessions;
3. the reconstruction script and audit verify all claimed lineage groups;
4. all fitting and GhostLab selection use only development IDs and group-safe
   outer/inner partitions;
5. the normal architecture and genuine overload exception are behaviorally proven;
6. exactly three D configurations are frozen using development evidence and
   cluster-aware statistics;
7. final selection evaluates frozen A/B/C/D1-D3 once, gates every D against C and uses
   only the frozen tie-break order;
8. the predeclared gates produce an auditable promote-or-retain result;
9. any promoted champion passes the complete suite and clean offline reproduction;
10. F3 remains sealed; and
11. documentation accurately distinguishes development, out-of-fold, final-selection
    and organiser-private evidence.
