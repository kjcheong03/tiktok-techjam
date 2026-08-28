# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

## GhostLab Unified Quick Start

The unified system preserves the validated candidate and all reusable challenger
implementations behind versioned, independently switchable presets. The autonomous
campaign starts from the pure keyword baseline, evaluates compatible techniques and
combinations through disjoint F0/F1/F2 stages, and proposes candidates without activating
one automatically.

Supported platforms are macOS and Linux. On Windows, use WSL2 with Ubuntu; native Windows
is not yet supported because Unix-specific resource measurement remains in the campaign
runtime and the full workflow has not passed Windows CI. Use CPython 3.10 through 3.13
(3.12 recommended), `uv`, Git, and the released `data/catalog.jsonl`. Install `uv` first
from its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/)
if `uv --version` is unavailable. `jq` is optional and used only by the live-monitoring
command in Step 2.4.

### 1. Install and validate

From the repository root:

```bash
uv sync --all-extras --group dev
uv pip check
uv run ruff check ghostlab scripts tests
uv run mypy ghostlab
uv run pytest -q
git status
```

If the catalog is not present, complete **Download the Catalog** above first. Review and
commit intended implementation/configuration changes before starting a new campaign;
campaign freezing rejects a dirty worktree so that all inputs belong to a reproducible
commit.

### Run the current champion

The default competition-facing method is the validated guarded GBDT champion. When
`configs/active_candidate.json` is absent, `starter.Agent` intentionally uses this frozen
compiled runtime; an active-candidate pointer is only created after a later proposal is
explicitly approved.

After installing the dependencies and placing the released catalog at
`data/catalog.jsonl`, evaluate exactly what `starter.Agent` will run:

```bash
uv run python -m evaluator.local_evaluator \
  --output artifacts/reports/champion_results.json
```

To evaluate the champion's explicit research preset over the 200 public sessions:

```bash
uv run python -m scripts.run_unified_preset \
  --config configs/suites/champion_guarded.json \
  --output artifacts/reports/local_unified_champion.json
```

The selected candidate's grouped 150-session OOF technical score is `0.878963`; this is
a development estimate, not a guaranteed private-leaderboard score. See
[`docs/final_candidate_checkpoint.md`](docs/final_candidate_checkpoint.md) for its exact
pipeline, validation evidence, limitations, hashes, and recovery information.

### 2. Choose and run an autonomous workflow

Run these commands from the repository root after Step 1 passes and `git status` is clean.
The three modes use separate campaign IDs and checkpoints:

| Goal | Mode and campaign ID | Search anchors |
|---|---|---|
| Reconstruct from scratch without incumbent bias | `discover` / `adaptive_autonomous_discovery_v1` | Pure `state.current + question.fixed + retrieval.sparse` baseline only |
| Improve the strongest composable incumbent | `augment` / `adaptive_autonomous_augment_v1` | State Baseline V2; compiled guarded champion is a control only |
| Maximize coverage (recommended) | `full` / `adaptive_autonomous_full_v1` | Pure baseline and State Baseline V2 independently; compiled champion is a control only |

The compiled guarded champion cannot be patched safely because its internal techniques are
inseparable. Therefore, `augment` continues from State Baseline V2 rather than mutating the
compiled champion. `full` is the recommended comparison because it can discover a new path
from the pure baseline without losing the opportunity to improve the incumbent.

#### 2.1 Start a first fresh campaign

Pure-baseline discovery:

```bash
uv run python -m scripts.run_autonomous_end_to_end \
  --mode discover \
  --prepare-assets
```

Incumbent augmentation:

```bash
uv run python -m scripts.run_autonomous_end_to_end \
  --mode augment \
  --prepare-assets
```

Comprehensive search from both searchable anchors (recommended):

```bash
uv run python -m scripts.run_autonomous_end_to_end \
  --mode full \
  --prepare-assets
```

`--prepare-assets` downloads or builds missing pinned optional assets and verifies every
asset before admission. On macOS, optionally prevent system sleep only while the command
runs:

```bash
caffeinate -i uv run python -m scripts.run_autonomous_end_to_end \
  --mode full \
  --prepare-assets
```

The wrapper performs preflight, freezes the selected versioned campaign and its
42-parameter conditional search space, plans the bounded combinations, runs F0/F1/F2,
checkpoints every completed job, and emits zero or three independently confirmed proposal
roles. It never commits, pushes, opens F3, or activates a candidate. Initial dense-asset
preparation can take roughly 20–25 minutes on CPU; a full campaign can take several hours.

#### 2.2 Resume an interrupted campaign

Run the exact same mode command again. If its `manifest.json` exists, the wrapper verifies
the frozen hashes and reuses content-addressed completed jobs. It does not silently restart
them. `--prepare-assets` may remain present; existing assets are verified rather than
redownloaded. Add `--resume` only when you want the command to fail unless an existing
manifest is present:

```bash
uv run python -m scripts.run_autonomous_end_to_end \
  --mode full \
  --resume
```

#### 2.3 Start another genuinely fresh campaign later

Never delete, overwrite, or mix an existing campaign checkpoint. Copy the relevant
template, increment both its filename and internal `campaign_id` (for example, from
`adaptive_autonomous_full_v1` to `adaptive_autonomous_full_v2`), review and commit that
new template, then run:

```bash
uv run python -m scripts.run_autonomous_end_to_end \
  --template configs/campaigns/adaptive_autonomous_full_v2.template.json \
  --prepare-assets
```

This creates a separate manifest, checkpoint, evidence ledger, and proposal directory.

#### 2.4 Monitor progress safely

Open a second terminal in the repository root. Set the ID to the mode being run, then view
the atomically updated status every 30 seconds:

```bash
CAMPAIGN_ID=adaptive_autonomous_full_v1
while true; do
  clear
  date
  jq . "artifacts/campaigns/${CAMPAIGN_ID}/live_status.json"
  sleep 30
done
```

Press `Ctrl-C` to stop only the monitor. `total_jobs`, `complete`, and
`highest_individual_job` describe the currently executing fidelity stage, so counts may
change when the campaign advances. The final aggregate and scenario-safe decision is in
`evidence.json`, not the live single-job maximum. To confirm macOS sleep prevention while
the wrapped search is running, use `pmset -g assertions | grep -A4 caffeinate`.

### 3. Review the completed campaign and prepare one proposal

Replace `<campaign_id>` below with the chosen ID from the table:

| Path | Purpose |
|---|---|
| `artifacts/campaigns/<campaign_id>/admission.json` | Every technique's admission, dependency, asset, and minimum-trial disposition |
| `artifacts/campaigns/<campaign_id>/manifest.json` | Immutable commit, data, split, catalog, search-space, and campaign hashes |
| `artifacts/campaigns/<campaign_id>/plan.json` | Planned structures and explicit compatibility skips |
| `artifacts/campaigns/<campaign_id>/checkpoint.json` | Atomic per-job outcomes used for resume |
| `artifacts/campaigns/<campaign_id>/live_status.json` | Current-stage progress and highest individual job |
| `artifacts/campaigns/<campaign_id>/evidence.json` | Final F0/F1/F2 comparisons, receipts, safety gates, and confirmed Top 3 |
| `artifacts/proposals/<campaign_id>/proposal_manifest.json` | Runnable proposal presets, techniques, parameters, assets, scores, and commands |

The wrapper finishes by printing zero or three `prepare_candidate` commands. Zero means
that fewer than three candidates passed every confirmation gate; it deliberately retains
the current champion rather than padding unsafe results. For a proposal you approve, copy
and run exactly one printed command, for example:

```bash
uv run python -m scripts.prepare_candidate \
  --preset artifacts/proposals/<campaign_id>/<printed-preset>.json
```

Preparation copies the immutable preset into `configs/candidates/`, evaluates it on the
development split, hashes it, and prints `next_activation_command`. It still does not
activate anything.

### 4. Activate, verify, or roll back

After human review, run the exact hash-bound `next_activation_command` printed by Step 3.
It writes `configs/active_candidate.json` atomically and prints both the next verification
and rollback commands. Run the printed verification command:

```bash
uv run python -m scripts.verify_active_candidate
```

If review or verification fails, restore the guarded champion:

```bash
uv run python -m scripts.activate_candidate --rollback
```

The active pointer is the only selection state used by `starter.Agent`; campaign results
alone never change the competition-facing method. Start with `docs/essentials/README.md`
for the curated project reading order. Use
`docs/essentials/unified_technique_operations.md` for every technique, dependency, switch,
and retest rule, and `docs/essentials/autonomous_unified_system_reference.md` for the
complete campaign, overfitting, pruning, proposal, activation, and recovery specification.

## Technical Architecture and Complete Strategy Surface

The executable pipeline is:

```text
conversation state → query construction → sparse/dense retrieval → fusion
                   → ranking/filtering/priors/diversity
                   → question/termination policy → normalized Top-10 response
```

`configs/techniques/catalog_v2.json` extends the Wave 1 catalog and is the strategy
source of truth. `ghostlab/campaign/bindings.py` is the runtime truth that maps a technique
ID to a typed configuration patch. This README covers all 64 present strategies: 36
runtime-composable switches, 12 anchor/intrinsic implementations, and 16 experiment
procedures. Catalog entries without a runnable implementation are intentionally omitted.

The tables use these execution classes:

- **C — composable:** on when its ID is included in a candidate; off when omitted. All 36
  C entries are independently considered by the current campaign. Mutually exclusive
  values such as state, question, dense backend, fusion route, and primary reranker cannot
  coexist; additive techniques can.
- **A — anchor/intrinsic:** implemented and preserved, but not an independent additive
  switch. It is selected through a complete preset or is inseparable from another switch.
- **R — research-only:** invoked by the experiment engine, never emitted as submission
  runtime configuration.

For C entries, the **Enable** column shows the actual typed patch. Omission restores the
baseline/off value. Numeric weights are present only when their technique is enabled and
are tuned conditionally. A dagger (†) marks an executable historical fitted asset that can
be diagnosed but cannot become a proposal without fold-local refitting and a new freeze.

### Runtime-composable on/off switches (36)

| Technique ID | Enable setting and mechanism | Implementation |
|---|---|---|
| `state.current` | `state_variant=current`; current-message state and pure-search anchor | `ghostlab/runtime/experimental.py` |
| `state.raw_history` | `state_variant=raw_history`; lossless accumulated dialogue | `ghostlab/state/memory.py` |
| `state.multi` | `state_variant=multi`; retain multiple values per attribute | `ghostlab/state/memory.py` |
| `state.compressed` | `state_variant=compressed`; compact state summary | `ghostlab/state/memory.py` |
| `state.baseline_v2` | `state_variant=baseline_v2`; typed constraints, corrections, provenance, intent epochs | `ghostlab/state/baseline_v2.py` |
| `state.catalog_normalizer.v1` | `normalizer=catalog_v1`; local catalog-grounded ontology asset | `ghostlab/state/normalization.py` |
| `state.confidence_gated_constraints.v1` | `constraint_confidence=0.9`; requires catalog normalizer | `ghostlab/state/normalization.py` |
| `query.structured` | `query_variant=structured_active`; active structured evidence only | `ghostlab/state/query.py` |
| `query.coverage_adaptive_v2` | `query_variant=coverage_adaptive_v2`; State V2 query with low-coverage raw-history fallback | `ghostlab/state/baseline_v2.py` |
| `query.catalog_prf.v1` | `query_expansion=prf`; catalog-grounded pseudo-relevance feedback | `ghostlab/state/query_expansion.py` |
| `question.fixed` | `question_variant=fixed`; literal organizer attribute order | `ghostlab/runtime/experimental.py` |
| `question.other_always` | `question_variant=other_always`; simulator-sensitive diagnostic | `ghostlab/runtime/unified_experimental.py` |
| `question.adaptive_heuristic` | `question_variant=adaptive`; observable rule-based question selection | `ghostlab/policy/adaptive_questions.py` |
| `question.learned_linear`† | `question_variant=learned`; compiled linear action-value model | `ghostlab/policy/learned_questions.py` |
| `question.candidate_eig.v1` | `question_variant=candidate_eig`; expected information gain from retrieved-candidate facets | `ghostlab/policy/eig_questions.py` |
| `termination.reward_aware.v1` | `question_value_margin=0.02`; question-versus-stop margin, requires candidate EIG | `ghostlab/policy/eig_questions.py` |
| `policy.joint_observable.v1` | `question_variant=joint_observable`; bounded decision list jointly chooses observable actions | `ghostlab/policy/joint_policy.py` |
| `retrieval.sparse` | `retrieval_route=keyword`, `dense_backend=off`; organizer-compatible FTS5/BM25 | `ghostlab/retrieval/sparse.py` |
| `retrieval.minilm` | `dense_backend=minilm_control`; pinned offline MiniLM cosine retrieval | `ghostlab/retrieval/dense.py` |
| `retrieval.e5` | `dense_backend=e5_small_v2`; pinned offline E5 retrieval | `ghostlab/retrieval/dense.py` |
| `fusion.rrf` | `retrieval_route=rrf`; reciprocal-rank sparse/dense fusion | `ghostlab/retrieval/fusion.py` |
| `fusion.weighted` | `retrieval_route=weighted`; conditionally tuned sparse/dense weights | `ghostlab/retrieval/fusion.py` |
| `fusion.sparse_first_union` | `retrieval_route=sparse_first_union`; sparse ranking with dense backfill | `ghostlab/retrieval/fusion.py` |
| `ranking.fixed_lexical` | `reranker=linear`; deterministic lexical reranker | `ghostlab/retrieval/rerank.py` |
| `ranking.metadata_gbdt`† | `reranker=metadata_gbdt`; shallow catalog/lexical GBDT | `ghostlab/retrieval/gbdt.py` |
| `ranking.top10_residual_reranker.v2` | `residual_reranker_enabled=true`; fold-fitted, membership-preserving Top-10 reordering with adaptive model family, features and activation gates | `ghostlab/retrieval/residual.py`; `ghostlab/training/residual.py` |
| `ranking.cross_encoder` | `cross_encoder_enabled=true`; pinned top-k neural reranking with tunable depth/weight | `ghostlab/retrieval/cross_encoder.py` |
| `ranking.reward_lambdamart.v1`† | `reranker=reward_lambdamart`; metric-aligned learning-to-rank | `ghostlab/retrieval/reward_lambdamart.py` |
| `ranking.turn_aware_lambdamart.v1`† | `reranker=turn_aware_lambdamart`; ranking objective includes turn cost | `ghostlab/retrieval/reward_lambdamart.py` |
| `ranking.fold_ensemble.v1`† | `reranker=rank_ensemble`; fold-model variance reduction | `ghostlab/retrieval/ensemble.py` |
| `fusion.rank_stack.v1`† | rank-stack ensemble asset; requires fold ensemble | `ghostlab/retrieval/ensemble.py` |
| `ranking.facet_diversity.v1` | `diversification=facet_mmr`; facet-aware maximal marginal relevance | `ghostlab/retrieval/diversify.py` |
| `filter.structured` | `structured_filter=true`; coverage-aware constraint filtering with fallback | `ghostlab/retrieval/filters.py` |
| `recommendation.correction_scoped_history` | `recommendation_history=correction_scoped`; suppress repeats within the current intent epoch | `ghostlab/runtime/unified_experimental.py` |
| `prior.profile` | `profile_prior_weight>0`; bounded observable-profile prior | `ghostlab/retrieval/profile.py` |
| `prior.quality` | `quality_prior_weight>0`; bounded catalog-quality tie-breaker | `ghostlab/retrieval/quality.py` |

### Implemented anchor or intrinsic strategies (12)

These are off when their owning preset/switch is absent; they do not have a safe
independent C toggle.

| Technique ID | Role | Implementation |
|---|---|---|
| `state.attribute_ontology.v1` | Builds the ontology asset consumed by catalog normalization | `ghostlab/state/catalog_ontology.py` |
| `query.expansion_guard.v1` | Safety guard intrinsic to catalog PRF | `ghostlab/state/query_expansion.py` |
| `ranking.mmr_early.v1` | Early-turn gate intrinsic to facet MMR | `ghostlab/retrieval/diversify.py` |
| `routing.joint_route.v1` | Routing behavior intrinsic to the joint-policy asset | `ghostlab/policy/joint_policy.py` |
| `ranking.pairwise_linear` | Historical first-champion preset | `ghostlab/retrieval/learned.py` |
| `ranking.constraint_gbdt` | Selected guarded-GBDT preset | `ghostlab/retrieval/constraint_gbdt.py` |
| `ranking.deep_dense_gbdt` | Historical dense/GBDT challenger | `ghostlab/retrieval/gbdt_dense.py` |
| `ranking.neural_gbdt` | Historical neural-score/GBDT challenger | `ghostlab/retrieval/neural_rank.py` |
| `guard.override_fallback` | Override guard and fallback inside the compiled champion | `ghostlab/runtime/guarded_gbdt.py` |
| `routing.decision_list` | Supporting decision-list mechanism selected through joint policy | `ghostlab/policy/decision_list.py` |
| `routing.observable_stump` | Historical observable route-stump control | `ghostlab/research/route_stump.py` |
| `routing.route_table` | Historical conditional route-table control | `ghostlab/research/route_policy.py` |

### Research, search, and validation strategies (16)

These procedures are available to, or used by, the experiment engine as declared by the
frozen campaign; they are not Agent runtime switches. In particular, conditional BOHB is
active in the current runner while Hyperband remains an implemented optional scheduler.

| Technique ID | Engine role | Implementation |
|---|---|---|
| `research.replay` | Deterministic multi-turn replay and reward traces | `ghostlab/research/replay.py` |
| `research.counterfactual` | Counterfactual action evaluation | `ghostlab/research/counterfactual.py` |
| `research.counterfactual_expert.v2` | Offline expert-label generation | `ghostlab/research/counterfactual_expert.py` |
| `research.leakage_firewall` | Reject protected or label-derived inputs | `ghostlab/research/firewall.py` |
| `evaluation.grouped_splits` | Keep all turns from a session in one fold | `ghostlab/evaluation/splits.py` |
| `evaluation.paired_statistics` | Paired deltas, intervals and significance diagnostics | `ghostlab/evaluation/statistics.py` |
| `evidence.decision_store` | Append-only technique/interaction decision evidence | `ghostlab/optimization/evidence.py` |
| `search.random_grid_beam` | Random/grid screening followed by bounded beam expansion | `ghostlab/optimization/search.py` |
| `search.multifidelity_racing` | Promote or prune across F0/F1/F2 budgets | `ghostlab/optimization/racing.py` |
| `search.hyperband.v1` | Active seed-budget successive halving; weak HPO variants are pruned before full multi-seed evaluation | `ghostlab/campaign/orchestrator.py` |
| `search.bohb.v1` | Conditional model-based HPO proposals | `ghostlab/optimization/bohb.py` |
| `search.typed_patches` | Type-safe configuration mutation | `ghostlab/optimization/patches.py` |
| `search.crossover` | Interaction-reserve recombination of compatible candidates | `ghostlab/optimization/patches.py` |
| `search.evidence_allocator` | Allocate budget using accumulated evidence | `ghostlab/optimization/meta_search.py` |
| `search.family_ucb` | Preserve exploration across technique families | `ghostlab/optimization/evidence.py` |
| `search.expert_iteration.v1` | Offline counterfactual dataset aggregation | `ghostlab/research/counterfactual_expert.py` |

## How the Engine Tests Combinations

The discovery anchor is intentionally minimal:

```text
state.current + question.fixed + retrieval.sparse
```

Everything else begins off in discovery mode. Full mode also searches from the strongest
composable State V2 incumbent, while the compiled guarded champion remains a matched
control because its inseparable compiled internals cannot be safely patched. The current
versioned campaign then:

1. resolves all 36 C entries through dependency and exclusivity rules;
2. plans every valid standalone/dependency closure and compatible pair across the two
   searchable anchors—currently 984 low-order structures, of which 586 materialize and
   every rejected structure remains explicit evidence;
3. evaluates small stratified F0 budgets and permanently removes only invalid, duplicate,
   or repeatedly dominated structures;
4. preserves uncertain, mildly negative, family-diverse, random-audit, and
   interaction-reserve candidates so a weak standalone may still win in combination;
5. expands evidence-supported combinations to orders 3–6 using bounded beam/crossover
   search rather than enumerating an unbounded power set;
6. applies conditional BOHB proposals over 42 real runtime parameters, including question
   ordering and horizon, six sparse field weights, retrieval depth, normalized fusion
   share, EIG controls, PRF, priors, rerank depth, cross-encoder gating, and diversity;
   log-scaled quantities are sampled in log space and uncertainty gates use target-free
   retrieval entropy; the residual ranker additionally tunes feature set, logistic/GBDT/
   ensemble implementation, regularization, depth, blending and confidence gates;
7. performs actual seed-budget successive halving: all variants receive one seed, the
   stronger half per compatible family receives the second, and only survivors receive
   the remaining frozen seeds;
8. uses backward ablation, add-back tests and paired interaction deltas to attribute gains;
9. searches on three frozen outer folds (90 sessions), confirms finalists on two disjoint
   folds (60 sessions), and never opens F3; and
10. materializes exactly three independently confirmed, behaviorally distinct roles—score
   leader, robust leader, and efficient alternative—from one matched anchor, or stops
   without padding unsafe candidates.

The exhaustive generated structures and skips live in
`artifacts/campaigns/adaptive_autonomous_full_v1/plan.json`; live progress is written to
`live_status.json`; outcomes and interactions live in
`evidence.json`; enabled techniques, tuned parameters, full resolved configuration, hashes,
and preparation commands live in
`artifacts/proposals/adaptive_autonomous_full_v1/proposal_manifest.json`.

Every adaptive default reproduces the previous runtime. A candidate changes behavior only
when its enabled technique exposes the corresponding parameter. The search-space file is
hash-frozen into schema-v2 manifests. Fit-required additions remain barred from F2 unless
they have a declared fold-safe trainer and provide complete disjoint-fold receipts; the
residual Top-10 reranker implements that path. See
[`docs/adaptive_autonomous_optimizer.md`](docs/adaptive_autonomous_optimizer.md) for the
complete design, validation, recovery, and extension contract.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
