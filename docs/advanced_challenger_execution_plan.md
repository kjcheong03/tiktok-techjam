# GhostLab Advanced Challenger Execution Plan

Date: 2026-08-26
Starting branch: `ghostlab/implementation`
Starting policy: `ghostlab_champion_linear_v1`
Protected holdout status: sealed and not accessed

## 1. Purpose

This document defines the complete execution procedure for advancing GhostLab
without damaging the current champion or turning the 150 public development
sessions into an unrestricted leaderboard.

The current champion is the permanent control for the next phase. It has:

- five-fold out-of-fold technical score `0.817649`;
- all-development refit and compiled score `0.819719`;
- exact research-versus-compiled session parity;
- 79 passing tests, Ruff, mypy, lockfile, integrity, and offline checks;
- 4.04-second cold initialization, 44.3-millisecond warm p95, and 0.93 GB peak
  process memory in the checkpoint environment;
- no external calls or bundled model asset;
- no protected-holdout access.

The objective is not to add every possible model. The objective is to determine,
with controlled evidence, whether learned questioning, semantic retrieval,
nonlinear ranking, cross-encoding, or better query construction can improve the
complete conversational-search policy beyond this champion.

## 2. Non-negotiable invariants

1. `ghostlab/implementation` remains the recoverable champion branch.
2. Incompatible prototypes are developed in separate branches and worktrees.
3. The 50-session F3 holdout is unavailable to every prototype worktree.
4. The 150-session adaptive set is discovery data, not pristine test data.
5. All learned assets are fit inside training folds and evaluated on unseen folds.
6. Candidate selection uses paired session evidence, not the largest raw decimal.
7. A technique that loses alone may still enter a bounded, justified interaction
   test; it is parked rather than deleted.
8. Heavy searches have predeclared candidate and wall-clock limits.
9. Only one frozen final candidate may access F3, exactly once.
10. No policy tuning is allowed after F3 access.
11. The official adapter and protected evaluator are never edited for experiments.
12. No branch may commit the catalog, model cache, virtual environment, secrets,
    guarded split, or holdout access log.

## 3. Repository and worktree topology

The recommended layout is:

```text
TikTok Techjam/
├── techjam/                         champion; ghostlab/implementation
├── techjam-integration/             ghostlab/integration
├── techjam-question-policy/         exp/learned-question-policy
├── techjam-dense/                   exp/dense-retrieval
├── techjam-gbdt/                    exp/gbdt-reranker
├── techjam-cross-encoder/           exp/cross-encoder
└── techjam-query/                   exp/query-construction
```

Each directory is a normal working copy with an independent branch, working tree,
`.venv`, ignored cache, and experiment outputs. All directories share the same Git
object database, commits, branches, tags, and remotes.

The same branch cannot be checked out in two worktrees. Uncommitted files are
isolated. Committed changes become visible in the shared repository history and can
be cherry-picked or merged elsewhere.

## 4. One-time checkpoint preparation

Run these checks in the existing champion directory before creating worktrees:

```bash
cd "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam"
git branch --show-current
git status --short
uv lock --check
uvx ruff format --check .
uvx ruff check .
uv run --frozen python -m unittest discover -s tests -p 'test_*.py'
uv run --frozen python -m scripts.verify_phase0
uv run --frozen python -m scripts.validate_compiled
uv run --frozen python -m scripts.validate_champion_checkpoint
```

The expected branch is `ghostlab/implementation`. The expected parity and
performance reports are:

```text
artifacts/reports/phase22_champion_compiled_parity.json
artifacts/reports/phase23_champion_checkpoint.json
```

After reviewing the staged diff, create one checkpoint commit. An optional local
annotated tag may then provide a short immutable name:

```bash
git tag -a champion-v1 -m "GhostLab validated champion v1"
```

Do not move the tag if later work improves the system. Create `champion-v2` for a
new validated checkpoint.

## 5. Creating and accessing the worktrees

After the checkpoint exists, capture its exact commit and create the worktrees from
that commit:

```bash
cd "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam"
CHAMPION_COMMIT="$(git rev-parse HEAD)"

git worktree add ../techjam-integration \
  -b ghostlab/integration "$CHAMPION_COMMIT"
git worktree add ../techjam-question-policy \
  -b exp/learned-question-policy "$CHAMPION_COMMIT"
git worktree add ../techjam-dense \
  -b exp/dense-retrieval "$CHAMPION_COMMIT"
git worktree add ../techjam-gbdt \
  -b exp/gbdt-reranker "$CHAMPION_COMMIT"
git worktree add ../techjam-cross-encoder \
  -b exp/cross-encoder "$CHAMPION_COMMIT"
git worktree add ../techjam-query \
  -b exp/query-construction "$CHAMPION_COMMIT"

git worktree list
```

Open a worktree with VS Code's **File -> Open Folder**, or from a terminal:

```bash
code "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam-question-policy"
```

Keep the champion window visually distinct and do not implement prototypes there.

## 6. Data, environment, cache, and artifact isolation

### 6.1 Catalog input

`data/catalog.jsonl` is intentionally Git-ignored, so new worktrees do not receive
it automatically. Link the same read-only input into each worktree instead of
copying it:

```bash
cd "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam-question-policy"
ln -s "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam/data/catalog.jsonl" \
  data/catalog.jsonl
```

Repeat for each worktree. The relative target is correct while the worktrees remain
sibling directories. Never commit the symlink. Confirm `git status --short` does
not include the ignored catalog.

### 6.2 Protected holdout

`artifacts/guarded/f3_v1.json` is ignored and must not be copied or linked into any
prototype worktree. Only the final integration worktree receives guarded access,
and only after the final candidate, configuration hash, analysis settings, and
commit are frozen.

### 6.3 Python environments

Use a separate environment for each incompatible dependency set:

```bash
uv sync --frozen
```

The shared `uv` download cache prevents repeated downloads, while separate `.venv`
directories prevent a cross-encoder or GBDT dependency from changing the champion.
Dependency additions belong only to their prototype branch until promotion.

### 6.4 Outputs

Each worktree writes to its own ignored `artifacts/cache`, `artifacts/campaigns`, and
prototype-named report. Do not overwrite phase 22 or phase 23. Use names such as:

```text
artifacts/reports/challenger_question_v1.json
artifacts/reports/challenger_dense_v1.json
artifacts/reports/challenger_gbdt_v1.json
artifacts/reports/challenger_cross_encoder_v1.json
artifacts/reports/challenger_query_v1.json
```

Cache keys must include the catalog hash, model/version, query representation,
feature version, training-session hash, and fold. A fitted model or label-derived
cache entry may never be reused across folds.

## 7. Common challenger contract

Every prototype must satisfy the same contract before it is compared:

- starts from the exact champion commit;
- leaves `starter.Agent` and the official evaluator behavior unchanged;
- implements a typed, switchable technique with a deterministic disabled path;
- records dependencies, configuration, seed, code hash, data hash, split hash,
  training IDs, validation IDs, wall time, and failure status;
- emits per-session results and aggregate metrics in the common report schema;
- runs exclusively on the adaptive 150 sessions and frozen nested folds;
- uses fold-local fitting for all learned parameters;
- includes an on/off ablation and a champion comparison;
- includes unit tests for its distinctive failure modes;
- reports latency, memory, and local asset size when the technique may reach runtime;
- performs no network call during the official runtime path;
- never reads research-only fields in runtime code.

Each report must label results as one of:

```text
training
inner-validation
outer-fold/out-of-fold
all-development refit
prospective F3
private organizer evaluation
```

Only out-of-fold, F3, and private results are generalization evidence. The
all-development refit exists to build the deployable model and verify parity.

## 8. Predeclared experimental manifest

Before a heavy run, create a manifest containing:

```json
{
  "experiment_id": "challenger_family_version",
  "parent_commit": "exact champion commit",
  "family": "question|dense|gbdt|cross_encoder|query",
  "hypothesis": "falsifiable expected improvement",
  "primary_metric": "recommended_technical_score",
  "secondary_metrics": ["hit_rate_at_10", "mrr", "mttc"],
  "candidate_limit": 20,
  "wall_clock_limit_seconds": 7200,
  "seeds": [20260826],
  "split": "nested_v1",
  "holdout_accessed": false,
  "promotion_rule": "declared before evaluation"
}
```

Candidate counts below are initial ceilings, not targets that must be exhausted.
Stop early when a recall gate fails, behavior duplicates an existing candidate, or
confidence shows no plausible useful gain.

## 9. Track A: learned counterfactual question policy

### 9.1 Hypothesis

A response-conditioned policy using only runtime-observable state can improve the
technical score over the fixed sequence by selecting higher-value questions and
stopping when the expected information gain is below the turn cost.

### 9.2 State features

Use only information available before choosing the action:

- current turn and turns remaining;
- asked attributes and no-preference attributes;
- active structured constraints and their provenance;
- category and query-term counts;
- sparse retrieval candidate count, score margin, entropy, and concentration;
- attribute diversity among top candidates;
- whether recent answers changed the candidate set or top ranks;
- previous question, repeat count, and response usefulness;
- recommendation stability and whether a hit-like confidence threshold is met.

Do not include target ID, target rank, scenario type, difficulty, simulator behavior,
future answer, reward, or ground-truth-derived feature in runtime inputs.

### 9.3 Counterfactual labels

At each training state, replay every legal question plus the no-question action.
Measure downstream session reward under a declared continuation policy. Keep
first-action attribution separate from full-policy attribution. Training labels may
use simulator outcomes offline, but the compiled policy receives only the observable
features above.

### 9.4 Model ladder

Test in increasing complexity:

1. fixed sequence champion;
2. deterministic value-of-information rule;
3. regularized linear action-value model;
4. shallow decision tree;
5. small GBDT only if simpler forms leave stable headroom.

Cross-fit policy learning by session. Report action distribution, repeat rate,
stop rate, and performance by scenario in addition to the main metrics.

### 9.5 Exit gate

Promote only if the out-of-fold gain is stable, no scenario has a material unexplained
regression, and the policy does not rely on one question or a simulator artifact.
Always report the fixed-sequence and `no_other` comparators.

## 10. Track B: stronger dense retrieval

### 10.1 Hypothesis

A stronger pretrained semantic generator can rescue relevant products absent from
the field-aware BM25 candidate set. Mere reshuffling of already-retrieved candidates
does not justify runtime cost.

### 10.2 Retrieval-first gate

Before end-to-end policy evaluation, measure:

- Recall@10, @50, @100, and @200;
- unique target rescues beyond champion BM25;
- targets lost relative to BM25;
- rank overlap and Jaccard by query stage;
- latency, index build time, memory, and asset size;
- recall by scenario and early versus late turn.

Test query/document formatting explicitly, including model-required prefixes.
Represent product fields with labels rather than an unstructured dump.

### 10.3 Candidate ladder

Use a small predeclared set rather than arbitrary model shopping:

1. current MiniLM control;
2. one compact retrieval-specialized embedding model;
3. one stronger model still compatible with offline packaging;
4. optional sparse semantic model only if the dense recall gate remains weak.

Compare dense alone, BM25+dense union, RRF, weighted fusion, and conditional routing.
The union must preserve the sparse head so dense cannot erase exact-match recall.

### 10.4 Exit gate

Park dense if unique recall improvement is negligible or rescued candidates cannot
be ranked into the Top-10. Promote the smallest model that creates stable, usable
recall and satisfies packaging budgets.

## 11. Track C: cross-fitted GBDT reranker

### 11.1 Hypothesis

A regularized nonlinear ranker can learn useful interactions among lexical match,
quality, constraint satisfaction, and retrieval rank beyond the two-feature linear
model without requiring a large neural asset.

### 11.2 Features

Candidate features may include:

- original rank and rank percentile;
- per-field BM25 scores and overlaps;
- dense score and dense rank when available;
- category, feature, details, description, and store matches;
- positive constraint coverage and contradiction indicators;
- Bayesian rating, popularity, and metadata completeness;
- profile compatibility only as a gated soft feature;
- query length, turn, and retrieval-confidence features.

All feature extraction must be deterministic and runtime-safe. Missing values use an
explicit missing indicator or model-native missing handling, not fabricated zeros
when zero has semantic meaning.

### 11.3 Training

Train by grouped session using pairwise ranking or LambdaMART. Never split turns from
the same session across train and validation. Cap tree depth, leaf count, boosting
rounds, and feature set before evaluation. Perform hyperparameter selection inside
the training side of each outer fold.

### 11.4 Exit gate

Compare against fixed field+quality, two-feature linear, and a rank-only control.
Promote only if gains survive outer folds and model/feature importance is not
dominated by an unstable proxy.

## 12. Track D: compact cross-encoder

### 12.1 Hypothesis

A compact pretrained cross-encoder can improve early reciprocal rank within a
high-recall candidate head, particularly for semantically phrased constraints.

### 12.2 Safe progression

1. Measure the champion's remaining MRR headroom and target presence in Top-20/50.
2. Evaluate one compact zero-shot cross-encoder on the Top-20.
3. Evaluate Top-50 only if Top-20 misses useful reranking opportunities.
4. Consider fold-local fine-tuning only if zero-shot evidence is positive and the
   public sample is sufficient; otherwise avoid fine-tuning overfit.
5. Test the cross-encoder score alone and as one feature in the GBDT.

Cache document representations by catalog/model hash, but cache query-candidate
scores by query hash and model version. Never cache label-derived fitting across
folds.

### 12.3 Exit gate

Promote only if MRR improvement survives grouped folds and cold start, p95 latency,
memory, and offline asset budgets remain acceptable. A slower model with no unique
gain is parked.

## 13. Track E: query construction

### 13.1 Hypothesis

Separating exact lexical evidence from structured and semantic intent can retain the
raw-history recall advantage while reducing conversational noise and invalidated
preferences.

### 13.2 Variants

Test a bounded set:

1. raw-history champion;
2. structured active constraints;
3. raw category plus structured constraints;
4. compressed raw history with per-slot term limits;
5. separate lexical and natural-language dense queries;
6. override- and negation-safe query with raw-history fallback.

Preserve provenance, category scope, and invalidation. Do not turn negative prose
into positive query terms. Deduplicate repeated answers and prevent long filler text
from consuming the sparse query term budget.

### 13.3 Exit gate

Measure candidate recall before end-to-end score. Recheck query variants with dense
retrieval and learned questioning because these are known dependency interactions.

## 14. Concurrency schedule

Concurrency applies to development and independent light evaluations. Heavy model
runs are scheduled to avoid CPU/RAM contention and incomparable timing.

### Wave 0: setup

- create checkpoint commit;
- create integration and prototype worktrees;
- link the read-only catalog;
- create isolated environments;
- verify every worktree reproduces the champion smoke metrics;
- confirm no prototype contains the F3 file.

### Wave 1: independent challengers

Run concurrently:

- Track A question-policy data/model implementation;
- Track B dense recall diagnostics;
- Track C GBDT feature/training implementation;
- Track E query construction and recall tests.

Cross-encoder code preparation can occur concurrently, but defer its heavy scoring
job if dense embedding or GBDT jobs already saturate memory.

### Wave 2: neural and dependency interactions

- run Track D cross-encoder scoring;
- test query+dense combinations;
- test question-policy behavior on the strongest stable retrieval control;
- test GBDT with only runtime-cheap features, then with validated dense or
  cross-encoder scores.

### Wave 3: integration tournament

- cherry-pick validated implementations into `ghostlab/integration`;
- reproduce each standalone report in the integration environment;
- run bounded pairwise and selected higher-order interaction tests;
- backward-ablate winners;
- compile the leading complete policy and rerun parity/performance gates.

At most one memory-heavy neural evaluation and one CPU-heavy classical evaluation
should run simultaneously on the current machine. Development, unit tests, and
report analysis may remain concurrent.

## 15. Statistical and anti-overfitting procedure

### 15.1 Fold discipline

- Group by complete session.
- Use the frozen five outer folds.
- Fit every learned component on the complement of its outer fold.
- Use inner folds or training-only validation for hyperparameter selection.
- Stitch outer predictions into one 150-session OOF result.
- Fit the deployable asset on all 150 only after the configuration is selected.

### 15.2 Evidence

For each challenger versus champion, report:

- mean paired session-reward delta;
- 95% paired bootstrap interval;
- paired randomization p-value;
- wins, ties, and losses;
- fold scores, mean, standard deviation, and worst fold;
- Hit@10, MRR, MTTC, and technical score;
- scenario-specific metrics and regressions;
- complexity, latency, memory, assets, and failure rate.

### 15.3 Multiple comparisons

Do not claim significance from the best of hundreds of raw results. Keep each family
finite and predeclared. Use family-level correction or max-statistic/randomization
analysis for large comparable sets. Treat corrected evidence as confirmatory and
uncorrected exploration as hypothesis generation.

### 15.4 Selection rule

1. Reject leakage, contract, parity, determinism, or packaging failures.
2. Identify candidates within the declared uncertainty/tie band of the best OOF
   technical score.
3. Preserve the strongest Hit@10 unless a declared MRR/efficiency tradeoff clearly
   improves the official objective.
4. Prefer fewer models, features, branches, assets, and runtime operations.
5. Require fold and scenario stability.
6. Retain a more complex system only for a material, reproducible benefit.

Repeated use of the adaptive 150 creates researcher-selection risk that cannot be
removed by worktrees or repeated cross-validation. The one-shot F3 confirmation and
private organizer set remain essential.

## 16. Interaction and combination search

Testing standalone techniques is necessary but not sufficient. Use this bounded
procedure:

1. Evaluate every challenger against champion independently.
2. Retain competitive candidates plus a small synergy reserve of theoretically
   dependent losers, such as query+dense or dense+reranker.
3. Test all pairwise combinations within the retained set.
4. Compute interaction gain:

   ```text
   interaction(A,B) = gain(A+B) - gain(A) - gain(B)
   ```

5. Promote pairs with positive evidence or known input/output dependencies.
6. Test selected three-way combinations.
7. If at most eight binary switches remain, an exhaustive 256-configuration final
   sweep is computationally acceptable, but it must be nested as a selection
   procedure.
8. For larger spaces, use bounded beam search with canonicalization and behavioral
   deduplication.
9. Backward-ablate each component of every combination winner.
10. Re-test parked techniques when a material dependency changes.

Do not evaluate unrestricted higher-order combinations merely because time is
available. Statistical information, not compute, is the limiting resource.

## 17. Integration protocol

Each prototype branch should contain small, reviewable commits:

```text
1. contract/models
2. implementation
3. tests
4. experiment script/config
5. evidence report and decision
```

Before promotion, the prototype must be clean and pass its local gates. In the
integration worktree:

```bash
cd "/Users/kj/Desktop/Hackathon/TikTok Techjam/techjam-integration"
git cherry-pick <validated-commit>
```

Prefer cherry-picking the minimal validated commits instead of merging an entire
prototype branch containing caches, abandoned approaches, or dependency churn.

After each promotion:

```bash
uv lock --check
uvx ruff format --check .
uvx ruff check .
uv run --frozen python -m unittest discover -s tests -p 'test_*.py'
uv run --frozen python -m scripts.validate_compiled
```

If the integration candidate regresses or becomes unstable, reset by creating a new
integration branch from the immutable champion commit. Do not rewrite or destructively
reset the champion worktree.

## 18. Final freeze and one-shot F3

Only after the challenger and interaction phases are complete:

1. Select exactly one candidate from nested adaptive evidence.
2. Refit learned assets on all 150 adaptive sessions.
3. Compile the policy into the official adapter.
4. Prove exact research-versus-compiled parity.
5. Run full lint, type, test, integrity, offline, latency, memory, and asset gates.
6. Freeze the candidate ID, code commit, compiled policy hash, model hashes, catalog
   hash, split hash, primary metric, baseline, minimum material delta, confidence
   level, bootstrap count, and randomization count.
7. Update `configs/validation/primary_analysis.json` once.
8. Create a final local tag without moving `champion-v1`.
9. Make the guarded F3 file available only to the final integration worktree.
10. Run `scripts.promote_holdout` exactly once.
11. Report the result even when negative.
12. Make no post-F3 policy or weight changes.
13. Run the frozen candidate on all 200 public sessions for the final public report.
14. Await the private 800-session organizer evaluation as the genuinely unseen test.

The F3 access log is append-only. A failed run after the access record is created
still consumes the holdout unless the failure is proven to occur before any F3 data
was read. Therefore, all packaging and parity tests must pass before access.

## 19. Removing worktrees safely

List current worktrees:

```bash
git worktree list
```

After an experiment is committed or deliberately discarded and its worktree is
clean:

```bash
git worktree remove ../techjam-question-policy
git branch -d exp/learned-question-policy
git worktree prune
```

Do not delete the folder manually. Do not use a force option until valuable changes
have been committed or explicitly deemed disposable. Keep the champion worktree and
checkpoint branch until the competition is complete.

## 20. Per-track completion checklist

- [ ] Branch starts at the recorded champion commit.
- [ ] Worktree has no F3 file or access log.
- [ ] Hypothesis and budget were declared before the run.
- [ ] Technique has an explicit enabled/disabled configuration.
- [ ] Disabled path does not initialize its dependency or asset.
- [ ] Unit and integration tests pass.
- [ ] Training is grouped and fold-local.
- [ ] OOF sessions and aggregate metrics are stored.
- [ ] Paired comparison and uncertainty are reported.
- [ ] Scenario regressions are documented.
- [ ] Runtime cost is measured when applicable.
- [ ] Standalone decision is keep, park, or invalidate with evidence.
- [ ] Plausible interactions are named explicitly.
- [ ] Branch contains no cache, secret, guarded data, or private artifact.
- [ ] Promotion commits are minimal and reviewable.

## 21. Final definition of success

The advanced phase succeeds when one of the following is established with honest
evidence:

1. A more adaptive GhostLab policy materially and stably beats `champion-v1`, is
   compiled within runtime budgets, and is frozen for the one-shot holdout; or
2. none of the advanced challengers generalizes reliably, in which case
   `champion-v1` remains the final candidate and the negative experimental evidence
   is preserved rather than hidden.

Either outcome is technically valid. The work is successful when the chosen policy
is the strongest defensible system under the available evidence, not when the most
complex prototype is forced into the submission.
