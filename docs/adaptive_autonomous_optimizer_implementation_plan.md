# Adaptive Autonomous Optimizer: Implementation and Validation Plan

Status: implementation-complete and locally validated on
`feat/adaptive-autonomous-optimizer`; a new multi-hour scored campaign remains an
operator action after review/commit. This remains the design contract; operational truth
and commands are in `docs/adaptive_autonomous_optimizer.md`.

Frozen starting commit: `8ef14f390d1e957baa9549145460e497222ef041`

Development branch: `feat/adaptive-autonomous-optimizer`

Development worktree: `techjam-adaptive-optimizer`

Stable reference branch: `feat/unified-autonomous-search`

Stable reference worktree: `techjam-unified`

Current default candidate: guarded constraint-aware GBDT, grouped 150-session OOF
technical score `0.878963`.

## 1. Purpose

This document is the implementation contract for upgrading GhostLab's autonomous
experiment engine from primarily structural on/off search into a reproducible,
conditional, interaction-aware optimizer. It specifies:

1. what will change;
2. what must remain unchanged;
3. which technique parameters and activation rules may be optimized;
4. how pure-baseline discovery and champion improvement remain independent;
5. how learned components are trained without leakage;
6. how compute is allocated without prematurely discarding interactions;
7. how overfitting, search bias, cache contamination, and accidental promotion are
   prevented;
8. how every phase is tested before a full campaign is allowed; and
9. what evidence is required before a candidate may be proposed or activated.

This plan extends rather than replaces:

- `docs/competition_specification.md`;
- `docs/submission_rules.md`;
- `docs/unified_technique_operations.md`;
- `docs/autonomous_unified_system_reference.md`;
- `docs/final_candidate_checkpoint.md`;
- `docs/technique_decision_ledger.md`; and
- `docs/wave2_advanced_challenger_and_autonomy_plan.md`.

If this document conflicts with the official competition contract, the competition
contract wins. Historical reports are evidence, not instructions to access protected
data or promote a candidate.

## 2. Confirmed starting state

The adaptive worktree is a clean checkout of the pushed unified commit. The stable
worktree remains clean on its original branch. The new optimizer is therefore isolated
from the current champion and existing campaign checkpoints.

The current system already provides:

- a typed unified runtime configuration;
- 36 admitted composable techniques in the expanded campaign;
- dependency and compatibility checking;
- pure-baseline candidate generation through order six;
- deterministic F0/F1/F2 jobs and resumable checkpoints;
- bounded conditional HPO for a small parameter set;
- a standalone successive-halving implementation;
- paired analysis and scenario metrics;
- score, robust, and efficient proposal roles;
- hash-bound preparation, activation, verification, and rollback; and
- a frozen compiled champion fallback used by `starter.Agent` when no active pointer
  exists.

The important current limitations are:

- only eight parameters are declared in `configs/search/wave2_weight_space_v1.json`,
  and fewer apply to many candidate structures;
- several meaningful values remain fixed in bindings or runtime defaults;
- activation is usually global on/off rather than based on observable confidence;
- champion presets are controls rather than composable search anchors;
- learned techniques are not all fold-locally trainable and promotion-safe;
- successive halving is not the main orchestration policy;
- search diagnostics emphasize individual jobs more than aggregate candidates; and
- cache reuse is not yet consistently layered by component inputs.

## 3. Objective and non-goals

### 3.1 Objective

Maximize the probability of discovering a robust, reproducible, competition-valid
candidate within a declared compute budget by jointly optimizing:

- technique structure;
- compatible interactions;
- meaningful technique parameters;
- observable activation and fallback rules; and
- resource allocation across uncertain candidates.

### 3.2 Non-goals

The project will not claim a mathematical global optimum. The complete structure and
continuous parameter space is too large for exhaustive proof.

The project will not:

- expose every source-code constant as a hyperparameter;
- modify the evaluator or official metric;
- use target IDs, scenario labels, future answers, evaluator state, or protected data as
  runtime features;
- tune after reading F2 confirmation results;
- access F3 during development search;
- automatically activate, commit, merge, or push a proposal;
- delete a reusable technique because it loses one campaign; or
- replace the current champion until a human runs the explicit activation command.

## 4. Non-negotiable safety invariants

1. The `techjam-unified` worktree and `feat/unified-autonomous-search` branch remain the
   recovery point throughout implementation.
2. The default guarded champion must retain exact starter-response parity until an
   explicitly reviewed activation occurs.
3. The absence of `configs/active_candidate.json` must continue to select the compiled
   champion.
4. Every experiment uses a new versioned campaign ID and its own checkpoint directory.
5. Search, confirmation, and protected partitions remain disjoint and hash-pinned.
6. Session and dependent-profile grouping is preserved in every split.
7. A learned asset is trained only on the fold-local training IDs recorded in its
   content-addressed receipt.
8. Confirmation outcomes cannot generate new parameters, structures, thresholds, or
   follow-up campaigns under the same validation claim.
9. Technique-off behavior performs no technique-specific model loading or computation
   and preserves the applicable control output.
10. Invalid and unavailable candidates are distinguished from candidates that ran and
    performed poorly.
11. All runtime activation predicates depend only on information observable at that turn.
12. Every cache key includes all inputs capable of changing the cached value.
13. All proposal creation remains proposal-only and requires human activation.

## 5. Design principle: optimize behavior, not arbitrary constants

A value is eligible for optimization only if all of the following are true:

- changing it has a clear, generalizable behavioral interpretation;
- it can be represented in a bounded typed domain;
- its dependencies and constraints can be validated before evaluation;
- it is available at runtime without forbidden information;
- its value can be chosen entirely inside search folds; and
- its default reproduces current behavior.

The following remain invariants rather than tunables:

- organizer action schema and legal `ask_attribute` values;
- maximum competition turn and Top-K contracts;
- evaluator metric implementation;
- split membership and protected-path restrictions;
- hash verification and path-safety checks;
- exception/fallback safety boundaries;
- deterministic normalization needed for identity and caching; and
- hard resource ceilings declared in the frozen campaign.

SQLite FTS5 does not expose ordinary BM25 `k1` and `b` controls through its built-in
`bm25()` ranking function. Field weights, query construction, and retrieval depth may be
tuned on the current backend. Tunable `k1`/`b` requires a separately identified sparse
backend and matched control; it must not be presented as a parameter that SQLite ignores.

## 6. Target candidate representation

An evaluated candidate will conceptually contain four layers:

```text
anchor
  + technique structure
  + observable activation policy
  + conditional parameter assignment
```

The canonical identity must include all four layers plus relevant asset hashes. Two
candidates with identical realized behavior should deduplicate even when generated by
different search paths.

Candidate scalar parameters remain simple JSON-compatible values. Coupled values are
represented through named parameter groups and materialized into structured runtime
fields. Examples:

- `fusion_sparse_share` materializes dense share as `1 - sparse_share`;
- six named sparse field weights materialize the existing weight tuple;
- a `question_order_id` resolves to a frozen legal sequence; and
- activation-policy IDs plus thresholds materialize a typed observable predicate.

This avoids introducing arbitrary untyped dictionaries into the runtime.

## 7. Parameter registry v2

Each runtime-composable technique will publish a versioned parameter contract containing:

- stable technique and parameter IDs;
- type: categorical, integer, linear float, log float, or coupled/simplex;
- bounded domain and default;
- required techniques and exclusive groups;
- cross-parameter constraints;
- whether the parameter changes training, inference, or both;
- estimated evaluation cost tier;
- observable activation features it may use;
- cache invalidation scope;
- selection-safety and fold-fitting requirements; and
- documentation and evidence references.

The registry will be loaded once and used by preflight, candidate generation, HPO,
materialization, cache hashing, reporting, and proposal packaging. Hidden duplicated
search ranges in individual scripts are prohibited.

Planned primary files:

```text
ghostlab/campaign/models.py
ghostlab/campaign/catalog.py
ghostlab/campaign/bindings.py
ghostlab/optimization/conditional.py
ghostlab/research/technique_suite.py
configs/techniques/catalog_v3.json
configs/search/adaptive_parameter_space_v1.json
tests/test_adaptive_parameter_registry.py
```

The schema may be added as v2/v3 alongside existing schemas first. Existing campaign
manifests must remain readable so historical evidence is not invalidated.

## 8. Planned tunable surface

Final numeric bounds will be frozen before a scored campaign, based on implementation
semantics and smoke diagnostics rather than F2 outcomes.

### 8.1 Sparse retrieval

Tune when supported:

- title, feature, category, detail, store/brand, and description field weights;
- query field inclusion and negation-safe construction;
- sparse candidate depth;
- per-turn candidate depth schedule; and
- alternative sparse backend choice if a correctly validated configurable BM25 backend
  is implemented.

Do not expose ineffective SQLite `k1`/`b` placeholders.

### 8.2 Dense retrieval

Tune conditionally:

- MiniLM versus E5 backend;
- dense candidate depth;
- query representation;
- model-specific prefixes, especially E5 query prefixes;
- score normalization;
- rescue confidence threshold; and
- eligible turn range.

Model choice remains conditional on verified pinned local assets.

### 8.3 Fusion

Tune:

- RRF constant;
- sparse/dense simplex share;
- sparse-first union and backfill depth;
- confidence threshold for dense rescue;
- per-turn or confidence-bucket fusion policy; and
- component fallback behavior.

Weights must satisfy their coupled constraints by construction, not by rejecting most
random proposals after generation.

### 8.4 Cross-encoder and primary rerankers

Tune:

- rerank depth;
- interpolation weight;
- activation threshold;
- eligible turns;
- feature subset;
- tree depth/leaves, learning rate, regularization, and objective for learned rankers; and
- fallback when the model or evidence is unavailable.

Learned-ranker training parameters are legal only through the fold-local trainer.

### 8.5 Priors

Tune:

- profile and quality prior weights;
- candidate head depth;
- confidence-gated application;
- turn-dependent decay; and
- normalization before combination.

Priors cannot inspect raw private history unavailable to the participant agent.

### 8.6 Conversation state and constraints

Tune:

- extraction/correction confidence;
- hard-constraint confidence;
- conflict replacement threshold;
- value decay or persistence policy;
- attribute-scoped override rules; and
- filter-versus-soft-evidence threshold.

The raw user evidence and provenance are always retained. A low-confidence interpretation
may affect ranking softly but must not silently become a hard filter.

### 8.7 Question and termination policy

Tune:

- EIG candidate depth;
- entropy, partition, downstream-rank, and question-cost terms;
- minimum value/margin for asking;
- maximum questioning horizon;
- recommend-only, ask-only, or recommend-and-ask action policy;
- no-preference probability handling;
- fallback sequence; and
- early termination confidence.

The chosen action remains one of the organizer's legal attributes. Adaptive policy inputs
are current state and current candidate evidence only.

### 8.8 Query expansion

Tune:

- feedback candidate depth;
- minimum term support;
- maximum added terms;
- maximum expansion ratio;
- eligible fields and turns; and
- weak-evidence activation threshold.

Expansion must preserve negative constraints and avoid feeding a term back solely because
it was introduced by an earlier expansion.

### 8.9 Diversity

Tune:

- MMR relevance/diversity balance;
- rerank and output depth;
- eligible turns;
- maximum active constraint count; and
- browsing-confidence activation threshold.

### 8.10 Learned question/ranking ensembles

Tune only after fold-safe training exists:

- feature family selection;
- estimator capacity and regularization;
- ranker/ensemble weights;
- calibration method;
- rerank depth; and
- observable routing thresholds.

## 9. Observable activation policies

Techniques may be globally harmful but locally useful. The optimizer will therefore
search typed activation predicates using only observable signals such as:

- sparse score margin or concentration;
- sparse/dense rank agreement;
- candidate entropy and attribute coverage;
- number and confidence of active constraints;
- current turn;
- state conflict/invalidation flags;
- query length and evidence coverage;
- top-candidate diversity; and
- model availability/failure state.

Initial activation templates will remain deliberately small and auditable:

```text
always
low_sparse_confidence(threshold)
high_candidate_entropy(threshold)
early_browsing(max_turn, max_constraints)
high_state_confidence(threshold)
override_for_attributes(attribute_set, confidence)
late_turn(min_turn)
```

Arbitrary generated Python expressions and dataset/session-specific rules are forbidden.
Activation predicates are versioned, serializable, unit-tested, and included in candidate
hashes.

## 10. Fold-safe learned-component lifecycle

Every fit-required technique will implement a common trainer contract:

1. receive explicit training sample IDs, seed, parameters, and immutable input hashes;
2. reject confirmation/F3 IDs and any unrecognized data path;
3. fit only on the supplied partition;
4. write an immutable content-addressed asset and receipt;
5. record training IDs hash, feature schema, code commit, dependencies, seed, and metrics;
6. load the asset only for the matching validation job; and
7. refit a deployment asset only after human selection under a separately recorded step.

Candidate generation may screen a fit-required technique only when its trainer is
available. It may not become F2/proposal eligible through a globally pretrained
development-label asset.

Planned primary files:

```text
ghostlab/training/protocol.py
ghostlab/training/fold_assets.py
ghostlab/campaign/evaluator.py
ghostlab/campaign/freeze.py
ghostlab/campaign/proposal_from_campaign.py
tests/test_fold_local_training.py
tests/test_training_receipts.py
tests/test_leakage_firewall.py
```

## 11. Two independent search tracks

The upgraded campaign contains two logical tracks managed by one controller. They are not
two unmanaged terminal processes.

### 11.1 Pure-baseline discovery

Starts from the minimum keyword baseline and must independently reconsider every admitted
technique or its minimum dependency bundle. Historical champions may be matched controls
but cannot seed or bias this track's generation.

### 11.2 Champion augmentation and ablation

Starts from the frozen guarded champion and evaluates:

- champion plus each compatible technique;
- champion minus each removable component;
- component replacements;
- compatible pairs around promising changes;
- higher-order additions; and
- backward ablation of successful large candidates.

### 11.3 Fairness between tracks

Each track receives:

- a minimum candidate budget;
- a minimum exploration reserve;
- its own matched control and leaderboard;
- separate cache/evidence namespaces where behavior differs; and
- equal promotion rules at common fidelity.

Unused budget may be transferred only by a frozen rule. Early strength in one track cannot
eliminate all exploration in the other.

## 12. Candidate generation and interaction search

Generation proceeds in bounded stages:

1. controls and minimum dependency bundles;
2. every compatible standalone technique;
3. one-step additions, removals, and replacements;
4. compatible pairs selected from coverage, uncertainty, and synergy evidence;
5. triples and higher orders around multiple diverse parents;
6. backward ablation of strong large candidates;
7. deterministic resurrection and pruning-audit samples; and
8. lightweight surrogate suggestions from the still-unexplored valid space.

There is no hard rule limiting final candidates to three techniques. `max_order` remains a
declared compute ceiling, initially no lower than the existing six, and backward ablation
is used to remove unnecessary components.

Invalid candidates are rejected before consuming candidate quota or scheduler time.
Exact behavioral duplicates are deduplicated. Performance pruning requires repeated
matched evidence or a confidence bound demonstrating meaningful inferiority.

Interaction analysis records, on aligned sessions:

```text
synergy(A, B) = reward(A+B) - reward(A) - reward(B) + reward(control)
```

A weak standalone with positive or uncertain interaction evidence remains eligible for
the interaction reserve.

## 13. Conditional adaptive HPO

HPO operates inside each structure's eligible parameter subspace.

The upgraded procedure will support:

- categorical, integer, linear-float, and log-float domains;
- coupled/simplex parameter groups;
- structural and activation dependencies;
- random exploration plus diversity-aware warm starts;
- proposals near strong related structures;
- true successive halving over increasing resource levels;
- early stopping only with sufficient paired evidence;
- parameter-importance and sensitivity summaries; and
- additional trials only where uncertainty or expected improvement remains material.

The optimizer alternates structure search and parameter search. It does not fully tune
thousands of parameter settings for every weak structure, nor does it judge a strong
technique solely at one arbitrary default.

## 14. Adaptive resource allocation

The scheduler will allocate resources as follows:

1. one low-cost F0 evaluation for every admitted valid standalone/dependency bundle;
2. additional F0 seeds for strong, uncertain, mechanism-diverse, and audit-reserve
   candidates;
3. F1 sessions and HPO trials for survivors;
4. more seeds/trials for candidates with high uncertainty or upside;
5. F2 only for frozen finalists and their matched controls; and
6. no new search decision after F2 is opened.

Resource limits remain explicit:

- lightweight CPU evaluations may run concurrently;
- heavy-model jobs use a separate semaphore and default to one at a time;
- GPU jobs remain zero unless a verified GPU environment is declared;
- memory estimates are checked before scheduling; and
- wall-time and candidate ceilings make every campaign finite.

## 15. Overfitting and selection-bias controls

The optimizer itself is a learned selection procedure and must be validated as such.

Required controls:

- nested fold-local fitting and parameter selection;
- multiple predeclared search seeds;
- paired session-level comparisons;
- bootstrap confidence intervals and paired randomization tests;
- fold, seed, and scenario stability reporting;
- worst-scenario regression gates;
- a predeclared practical-effect threshold;
- family-aware multiple-comparison control for promotion claims;
- complexity, latency, and failure penalties;
- one frozen confirmation event per campaign version; and
- untouched F3 until the final authorized evaluation.

F2 may contain multiple predeclared seeds as one confirmation protocol, but all seeds,
candidates, parameters, and aggregation rules must be frozen before any F2 outcome is
read. F2 cannot be used as another HPO stage.

If F2 results influence future development, that future work requires a new campaign ID
and cannot claim F2-independent confirmation using the already observed partition.

## 16. Multi-objective selection

The engine will maintain a Pareto frontier across:

- mean technical score;
- lower confidence bound;
- worst-scenario score;
- fold/seed variance;
- HitRate@10, MRR, and MTTC components;
- latency and peak memory;
- model and asset size;
- runtime failure rate; and
- configuration complexity.

Up to three proposals are produced only when independently eligible:

1. score leader;
2. robust leader; and
3. efficient alternative.

The roles should be behaviorally distinct where possible. The system never pads missing
roles with controls, ties, unsafe candidates, or duplicate behavior.

## 17. Layered caching and reproducibility

Cacheable layers include:

- normalized conversation state;
- query construction;
- sparse retrieval;
- dense embeddings and scores;
- candidate features;
- cross-encoder scores;
- fold-local learned assets; and
- final metric aggregation.

Each cache key includes the relevant subset of:

```text
code commit
catalog hash
dataset and sample-ID hash
split/fold/training-ID hash
technique implementation version
model/asset hash
query/state hash
parameter and activation-policy hash
seed
```

Checkpoint writes are atomic. Running the same frozen campaign command resumes exact
completed work. A changed manifest, code commit, split, asset, or search space creates a
new namespace rather than silently reusing incompatible evidence.

## 18. Live observability and diagnosis

The live status report will show:

- campaign ID and frozen commit;
- current track, stage, round, fold, and seed;
- structures/jobs complete, running, failed, and remaining;
- aggregate F0, F1, and F2 leaders separately;
- candidate mean, confidence interval, and scenario scores;
- technique and parameter coverage;
- cache hit rate and resource utilization;
- current best combination and activation policy;
- pruning reason counts and individual decision records;
- interaction and backward-ablation effects; and
- elapsed time and bounded ETA.

The displayed leader is an aggregate candidate. The highest single small-budget job is
diagnostic only and is never presented as the campaign winner.

## 19. One-command operator workflow

The target command is:

```bash
uv run python -m scripts.run_autonomous_end_to_end \
  --mode full \
  --prepare-assets
```

Supported modes will be:

- `discover`: pure-baseline track only;
- `augment`: champion augmentation/ablation only; and
- `full`: both tracks through one scheduler and common confirmation/proposal process.

The command will:

1. validate dependencies and pinned assets;
2. verify a clean committed worktree;
3. freeze or resume a versioned campaign;
4. run preflight coverage;
5. execute bounded search and HPO;
6. write live and final evidence;
7. materialize up to three safe proposals; and
8. print exact candidate-preparation commands.

It will not activate a proposal. Candidate preparation prints a hash-bound activation
command; activation prints verification and rollback commands.

## 20. Planned implementation phases and exit gates

### Phase 0: preserve and measure the starting point

Work:

- capture champion response/config/model hashes;
- run the complete existing test suite;
- record old-engine fixed-budget behavior; and
- verify both worktrees and campaign isolation.

Exit gate: stable champion parity is reproducible and the adaptive branch is clean.

### Phase 1: parameter registry and typed materialization

Work:

- add versioned parameter/activation schemas;
- extend catalog loading and validation;
- map parameters through typed bindings;
- keep old manifests/configs readable; and
- add reachability and default-parity tests.

Exit gate: every declared parameter either changes its intended runtime config or fails
preflight as unsupported; no silent no-op parameters exist.

### Phase 2: observable activation policies

Work:

- implement small auditable predicate templates;
- add observable confidence features;
- route techniques and fallbacks conditionally; and
- prove forbidden fields cannot enter predicates.

Exit gate: policy serialization, hashing, fallback, off behavior, and leakage tests pass.

### Phase 3: fold-local training infrastructure

Work:

- implement trainer protocol and immutable receipts;
- adapt fit-required question/ranking techniques;
- enforce fold IDs at evaluator and proposal boundaries; and
- add final selected-candidate refit procedure.

Exit gate: synthetic contamination attempts fail and eligible learned techniques can run
train/validation folds without overlap.

### Phase 4: dual-track generation and interaction search

Work:

- add pure and champion track identities/budgets;
- implement additions, removals, replacements, pair synergy, higher-order beams, backward
  ablation, alternative anchors, and exploration reserve; and
- filter invalid/duplicate candidates before quota accounting.

Exit gate: deterministic fixtures demonstrate full standalone coverage, interaction
rescue, higher-order generation, and fair track budgets.

### Phase 5: adaptive HPO and resource allocation

Work:

- integrate successive halving into orchestration;
- add conditional/log/simplex domains;
- implement warm starts and uncertainty-aware trial allocation;
- retain audit candidates; and
- record every allocation/pruning decision.

Exit gate: synthetic known-objective tests recover expected optima within tolerance and
resume produces byte-equivalent final evidence.

### Phase 6: caching, scheduler hardening, and live reporting

Work:

- introduce layered content-addressed caches;
- add atomic checkpoints and resource semaphores;
- produce aggregate leader/coverage/pruning/ETA status; and
- validate cache invalidation.

Exit gate: warm reruns reuse valid work, changed inputs miss the cache, and interrupted
runs recover without duplicate or missing outcomes.

### Phase 7: smoke and fixed-budget engine comparison

Work:

- run a small all-technique coverage campaign;
- run old and new engines under identical data, seeds, and compute budgets;
- compare false pruning, coverage, stability, runtime, and best confirmed result; and
- fix correctness issues before expanding compute.

Exit gate: zero unexplained failures, no admitted valid technique without a trial, no
champion regression, and evidence that the new allocator/search is at least as reliable as
the old engine.

### Phase 8: full frozen campaign

Work:

- commit all reviewed code/configuration;
- freeze a new campaign ID;
- run `full` discovery plus augmentation;
- complete independent F2 confirmation once; and
- materialize proposal-only outputs.

Exit gate: top proposals pass statistical, scenario, runtime, packaging, asset, and parity
gates. Otherwise the correct outcome is to retain the current champion.

### Phase 9: documentation and handoff

Work:

- update root and essential quick starts;
- document every parameter, activation predicate, dependency, and command;
- write the engine comparison and campaign decision ledger; and
- document preparation, activation, verification, rollback, and resume.

Exit gate: a teammate can reproduce smoke, full search, proposal review, champion run, and
rollback using repository documentation alone.

## 21. Validation matrix

| Area | Required validation |
|---|---|
| Style and types | Ruff, MyPy, strict Pydantic schema validation |
| Unit behavior | Parameter domains, materialization, activation, compatibility, hashing |
| Champion safety | Exact starter parity, compiled hash, fallback and rollback |
| Optimizer | Synthetic known optima, deterministic seeds, interaction rescue, halving |
| Leakage | Disjoint IDs, forbidden paths/fields, fold-local receipts, sealed F3 |
| Resume/cache | Atomic interruption, identical resume, invalidation on every relevant input |
| Coverage | Every admitted valid technique receives a traceable trial |
| Statistics | Paired deltas, intervals, scenario gates, stability, multiplicity policy |
| Resources | CPU/heavy semaphores, memory refusal, latency and failure reporting |
| Packaging | Assets exist/hash, preset materializes, official adapter parity |
| Operations | One-command smoke/full run, live status, prepare/activate/verify/rollback |

The baseline repository quality command remains:

```bash
uv run ruff check ghostlab scripts tests
uv run mypy ghostlab
uv run pytest -q
```

Additional phase-specific commands will be added only when their implementations exist.
Documentation must not advertise commands that have not been validated end to end.

## 22. Full-campaign validation protocol

The full campaign will use a newly frozen manifest with:

- a unique campaign ID;
- exact parent commit and dependency lock;
- hashed catalog, dataset, technique registry, search space, and splits;
- pure and champion track budgets;
- predeclared seeds and fidelity budgets;
- predeclared HPO and higher-order limits;
- one confirmation protocol frozen before F2;
- explicit CPU, heavy-model, memory, and wall-time ceilings; and
- `promotion_rule=proposal_only`.

Success is not defined as merely beating `0.878963` on one evaluation. A proposal must
also demonstrate:

- positive paired development-confirmation evidence against its matched anchor;
- acceptable confidence and stability;
- no material scenario regression;
- runtime and packaging compliance;
- no leakage or untracked asset dependency;
- starter-adapter parity; and
- an explainable contribution/ablation record.

Scores from different session sets or evidence classes are never compared as if they were
the same leaderboard.

## 23. Risk register

| Risk | Mitigation |
|---|---|
| Search-space explosion | Conditional domains, dependency closure, successive halving, wall/candidate ceilings, caching |
| Excessive pruning | Mandatory coverage, uncertainty and mechanism reserves, repeated evidence, resurrection audit |
| Champion fixation | Independent pure-baseline track and alternative anchors |
| Weak-baseline fixation | Independent champion augmentation/ablation track |
| Parameter overfitting | Nested search, multiple seeds, bounded semantic domains, frozen F2, multiplicity controls |
| Learned-model leakage | Fold-local trainers, training-ID receipts, evaluator/proposal enforcement |
| No-op parameters | Reachability tests and backend-aware contracts |
| Cache contamination | Content-addressed layered keys and manifest namespaces |
| Misleading live score | Aggregate stage leaders; individual-job score labelled diagnostic |
| Resource exhaustion | Scheduler semaphores, memory checks, serialized heavy models, resumable wall bounds |
| Accidental champion replacement | Proposal-only outputs, no automatic pointer, explicit human activation and rollback |
| Complexity/AI slop | Extend existing abstractions, single source of truth, small typed modules, remove duplication |

## 24. Code-quality rules

- Prefer extending existing campaign, optimization, and runtime abstractions over creating
  parallel frameworks.
- Keep one source of truth for technique and parameter metadata.
- Use typed immutable models at campaign and proposal boundaries.
- Separate pure functions from I/O and process orchestration.
- Keep functions focused; avoid generic abstraction until at least two real consumers
  require it.
- Do not add a dependency when the current stack provides the required behavior reliably.
- Avoid generated boilerplate, duplicate wrappers, speculative plugin systems, and hidden
  global state.
- Every behavior change requires a test; every experiment decision requires evidence.
- Preserve user changes and keep commits phase-scoped and reviewable.

## 25. Required artifacts

The completed work should produce:

```text
configs/techniques/catalog_v2.json
configs/search/adaptive_parameter_space_v1.json
configs/campaigns/adaptive_autonomous_discovery_v1.template.json
configs/campaigns/adaptive_autonomous_augment_v1.template.json
configs/campaigns/adaptive_autonomous_full_v1.template.json
artifacts/campaigns/<campaign_id>/manifest.json
artifacts/campaigns/<campaign_id>/plan.json
artifacts/campaigns/<campaign_id>/checkpoint.json
artifacts/campaigns/<campaign_id>/evidence.json
artifacts/campaigns/<campaign_id>/live_status.json
artifacts/proposals/<campaign_id>/proposal_manifest.json
docs/adaptive_autonomous_optimizer.md
docs/state_v2_adaptive_residual_v2_decision.md
```

Generated campaign artifacts remain ignored unless a deliberately sanitized report is
approved for version control. Code, schemas, templates, tests, essential documentation,
and small required deployment assets are tracked.

## 26. Operator and promotion flow

After the implementation is complete, the intended operator flow is:

1. install and validate dependencies;
2. run smoke preflight;
3. commit reviewed code and templates;
4. run or resume the full autonomous command;
5. inspect coverage, pruning, comparison, and top-proposal evidence;
6. select a proposal or retain the current champion;
7. run the proposal's printed preparation command;
8. review its exact techniques, parameters, activation policy, assets, scores, and hashes;
9. run the printed activation command only after approval;
10. run verification through `starter.Agent`; and
11. roll back immediately if verification fails.

Pushing the optimizer branch shares the implementation and tracked documentation. It does
not merge into the stable branch, upload ignored local caches/checkpoints, or activate a
candidate for teammates. A selected active pointer and all required small assets must be
deliberately committed before a teammate receives that selected runtime by checkout.

## 27. Definition of done

The adaptive optimizer is complete only when:

- the stable champion remains exactly reproducible;
- every executable technique is represented accurately in the registry;
- every meaningful declared parameter and activation policy is reachable and tested;
- fit-required techniques are fold-safe or explicitly excluded from promotion;
- pure and champion tracks both receive fair bounded exploration;
- real successive halving and conditional HPO operate in the main loop;
- invalid candidates do not consume search quota;
- layered caching and exact resume are validated;
- live reporting shows aggregate progress and decision reasons;
- smoke and fixed-budget engine comparisons pass;
- a full campaign can complete from one command without protected-data access;
- up to three safe proposals and exact operator commands are produced;
- activation remains human-controlled and reversible; and
- a teammate can reproduce the workflow from the committed documentation.

Until every item above passes, the existing guarded GBDT remains the default champion.
