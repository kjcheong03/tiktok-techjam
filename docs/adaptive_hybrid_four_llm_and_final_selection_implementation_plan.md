# Four-LLM Development Study and One-Time Final Selection Plan

## Implementation status

Implemented and implementation-validated on 2026-08-31. No 1,650-session fit, local-LLM
grid execution, GhostLab campaign, 550-session final selection or candidate activation
was started.

Validation evidence:

- the plan resolves to 45 trials: five arms × three depths × three weights;
- Qwen2.5, Qwen3 and MiniLM assets are currently runnable; Gemma 3 and
  SmolLM2-1.7B are reported as unavailable until their pinned assets are fetched;
- model availability is established from pinned manifest verification, not directory
  existence, so a partial download cannot enter the experiment;
- finalist selection remains invalid until all 45 configured trials have been attempted
  exactly once; failed and timed-out trials remain recorded but are ineligible;
- Qwen3 explicitly renders its chat template with `enable_thinking=False` because the
  ranker scores the immediate yes/no next-token logits;
- the pipeline plan resolves through split, fit, diversity, LLM study, development
  evaluation, baselines, validation, campaign, Top-3 packaging, finalist evaluation
  and comparison;
- a two-session development-only end-to-end smoke completed with zero output
  constraint violations and zero overload trace violations;
- 464 repository tests passed and one was skipped;
- Ruff, focused Mypy, JavaScript syntax checking and `git diff --check` passed.

## 1. Decision and statistical meaning

This implementation uses the practical competition protocol selected by the team:

1. Fit and tune only on the 1,650-session development partition.
2. Compare four genuine local LLM families and the MiniLM non-LLM control on
   lineage-safe development folds.
3. Freeze every eligible complete GhostLab D configuration up to three, together with A, C,
   the lineage manifest, evaluation contract, gates and tie-break order.
4. Evaluate A, C and every available frozen D configuration (one to three) exactly once on the
   550-session final selection partition.
5. Apply only the predeclared gates and tie-breaks. No post-selection tuning,
   model replacement, threshold change or second access is permitted.

The 550 sessions are therefore called the **one-time final selection set**, not an
unbiased holdout. The organiser's private evaluation remains the unseen
generalization test.

## 2. Fixed architecture and adaptable choices

The required 1A-3B workflow remains fixed. The experiment changes only the model
and bounded settings inside the compulsory LLM semantic-ranking slot.

Development model families:

| Candidate | Role |
|---|---|
| Qwen2.5-0.5B-Instruct | Genuine local LLM candidate |
| Qwen3-0.6B | Genuine local LLM candidate |
| Gemma 3 1B IT | Genuine local LLM candidate; gated asset |
| SmolLM2-1.7B-Instruct | Genuine local LLM candidate |
| MiniLM cross-encoder | Non-LLM control and runtime fallback |

Every genuine LLM receives the same allowed semantic-depth and blend-weight grid.
Each family may select a different optimum. Model-specific tokenizer chat templates
are allowed, but all render the same versioned shopping-relevance instruction and
the same yes/no scoring contract.

## 3. Development experiment contract

### 3.1 Lineage-safe samples

- Load the immutable 1,650/550 lineage manifest.
- Use only development outer folds.
- Preserve complete lineage groups inside their assigned fold.
- Use identical ordered session IDs for every model, depth and weight.
- Record the selected group IDs, fold membership and hashes in the report.

### 3.2 Paired candidate pools

Keyword, category, dense retrieval, authority filtering and union ranking are held
constant. For every trial, hash the ordered pre-semantic candidate pool by
session/turn. The comparison is invalid if successful trials do not report the same
candidate-pool hash.

### 3.3 Symmetric grid

For each genuine LLM, evaluate every Cartesian pair from the same configured:

- semantic depths, default `10, 20, 30`; and
- semantic weights, default `0.20, 0.35, 0.50`.

The MiniLM control uses the same depths. Its semantic blend is its own bounded
control parameter and is clearly identified as non-LLM.

### 3.4 Reliability and quality evidence

Record for every trial and fold:

- Hit@10, MRR, MTTC and technical score;
- output constraint violations and confirmed target removals;
- target rescues into Top 10 and demotions out of Top 10;
- semantic activation, ordering-change and fallback rates;
- failure counts by reason, including label-token incompatibility, invalid scores,
  model-load errors and timeouts;
- mean and p95 semantic latency, elapsed time and peak worker memory;
- prompt-contract hash, model revision/tree hash, ordered sample hash and
  candidate-pool hash.

Each grid trial runs in an isolated worker process with a hard wall-clock deadline.
This makes peak-memory measurements comparable and prevents one hung model from
blocking the full matrix.

Before the first worker starts, every one of the four LLM assets and the MiniLM
control must pass its pinned manifest or acquisition-receipt verification and expose
the required configuration and weight files. After execution, the expected and actual
`(model, depth, weight)` trial ledgers must match exactly. Worker failure and timeout
records count as attempts but never as eligible configurations. Missing, duplicate or
unexpected trials invalidate selection and prevent candidate configuration emission.

Qwen3 is evaluated in non-thinking mode. Its model-specific chat template receives
`enable_thinking=False`, and the effective setting is recorded in diagnostics. Other
models retain their native chat template without receiving unsupported Qwen-specific
arguments; prompt meaning remains identical across families.

### 3.5 Per-model and overall selection

Choose each model family's optimum using development evidence and predeclared
quality/safety/reliability ordering. Both Qwen variants remain independent
candidates. The report may rank the best complete configurations, but it cannot
access the 550 sessions or activate a model.

## 4. Freezing up to three D configurations

After the GhostLab development race:

- require at least one and at most three development-eligible D finalists;
- materialize each complete configuration;
- freeze each file hash, canonical configuration hash, candidate ID, techniques,
  parameters and development evidence;
- freeze A's implementation, B's configuration, C's complete configuration, the
  lineage manifest, gates and tie-break specification;
- set `final_selection_accessed=false`.

The package must expose `frozen_proposals` containing between one and three entries. A
single `frozen_proposal` is no longer a valid protocol artifact.

## 5. One-time 550-session final selection

The final-selection command must refuse to run if either its output or access
receipt already exists. Before loading the 550 sessions it verifies every frozen
path and hash.

It then evaluates, through one shared harness and in the same ordered session list:

1. A: official stateless BM25;
2. C: fixed adaptive architecture control;
3. D1, D2 and D3: the available one to three frozen GhostLab finalists.

Each D is gated against C. Among D configurations that pass every gate, select the
winner using the immutable tie-break order:

1. higher recommended technical score;
2. higher MRR;
3. higher Hit@10;
4. lower MTTC;
5. lower fallback rate;
6. lower development rank;
7. lexicographically smaller candidate ID.

If no D passes, retain C. A remains a reference and is never champion-eligible.
The report must retain every system's metrics and every D-versus-C gate result; it
must not hide losing finalists.

## 6. Activation contract

Activation remains manual. It is permitted only when:

- the requested preset is one of the available one to three frozen configurations;
- the final-selection report names that exact candidate as the winner;
- its file and canonical hashes still match;
- all gates for the selected D passed; and
- the report proves that every frozen D configuration (one to three) and one C control were
  evaluated on the same 550 ordered sessions.

If C is retained, no D activation command is valid.

## 7. Validation phases

### Phase A: static protocol validation

- validate model manifests and immutable revisions;
- validate the symmetric Cartesian grid;
- validate lineage-group/fold integrity and ordered sample hashes;
- validate prompt-meaning equivalence and model-specific rendering;
- validate Top-3 and tie-break schemas.

### Phase B: focused behavioral tests

- reject asymmetric model grids;
- reject missing, duplicate or unexpected trial-ledger entries;
- reject partial-download directories and any unverified required asset;
- record failed/time-out trials as attempted but promotion-ineligible;
- prove Qwen3 direct scoring uses `enable_thinking=False`;
- reject mismatched paired candidate-pool hashes;
- reject fewer than one or more than three frozen D configurations;
- reject changed A/C/D/gates/manifest hashes before final selection;
- reject mismatched evaluator contracts or session order;
- prove that gates are applied separately to every D;
- prove deterministic winner selection and C retention;
- prove activation is restricted to the selected D.

### Phase C: repository validation

- Ruff and Mypy on changed source;
- focused tests;
- full test suite;
- pipeline plan-only execution;
- a small development runtime smoke without model training or 550 access.

## 8. Explicit non-actions during implementation

- Do not download multi-gigabyte models automatically.
- Do not start GhostLab training.
- Do not evaluate the 550-session final selection set.
- Do not activate a finalist.
- Do not modify or delete historical 2,200-session artifacts.
