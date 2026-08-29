# Adaptive Autonomous Optimizer

## Purpose and safety boundary

This document is the operational and technical reference for the adaptive optimizer on
branch `feat/adaptive-autonomous-optimizer`. The implementation searches both technique
structure and meaningful runtime parameters without changing the protected guarded
champion automatically.

The engine is a development proposal system. It never opens F3, edits the active-candidate
pointer, commits, pushes, or promotes a winner. A human reviews one of three distinct
independently confirmed proposals and runs its hash-bound preparation and activation
commands.

The stable `techjam-unified` worktree is not modified by this branch. Defaults are
backward-compatible: omitting the new adaptive settings reproduces the previous runtime.

## One-command interface

Install every declared optional dependency and validate the environment:

```bash
uv sync --all-extras --group dev
uv pip check
uv run ruff check ghostlab scripts tests
uv run pytest -q
```

Run the complete dual-track search:

```bash
uv run python -m scripts.run_autonomous_end_to_end --prepare-assets
```

The same entry point supports three explicit modes:

```bash
# Pure baseline; no historical method receives a search advantage.
uv run python -m scripts.run_autonomous_end_to_end --mode discover --prepare-assets

# Improve the strongest composable State V2 incumbent; guarded champion is a control.
uv run python -m scripts.run_autonomous_end_to_end --mode augment --prepare-assets

# Run both searchable anchors and compare each candidate to its own matched control.
uv run python -m scripts.run_autonomous_end_to_end --mode full --prepare-assets
```

Running the identical command again resumes the frozen manifest and reuses complete jobs.
Campaign IDs differ by mode, so checkpoints cannot collide.

## Search anchors

| Mode | Searchable anchors | Matched controls |
|---|---|---|
| `discover` | `unfitted_keyword_search.json` | its pure keyword control |
| `augment` | `state_baseline_v2_ranked.json` | State V2 control and guarded champion |
| `full` | both anchors above | both anchor controls and guarded champion |

The guarded champion is compiled and contains inseparable internal behavior. Treating it
as an additive preset would silently stop being the same champion, so it remains a fair
control. The strongest composable State V2 preset is the incumbent augmentation anchor.
Targeted augment mode intentionally omits alternative state implementations that conflict
with its fixed State V2 anchor; preflight still records those omissions. Full mode remains
the exhaustive inventory audit and admits all 36 runnable catalog entries with a trial.

## End-to-end lifecycle

1. Preflight inventories every catalog technique and verifies assets.
2. Freeze creates a schema-v2 manifest containing code commit, dataset/split hashes,
   technique catalog hash, and conditional search-space hash.
3. Planning resolves dependencies, exclusive groups, incompatible combinations, low-order
   structures, resource limits, and deterministic exploration reserve.
4. F0 screens small, stratified budgets from each anchor.
5. Higher-order search builds evidence-supported combinations up to order six and retains
   family-diverse and random-audit candidates.
6. HPO is disabled by default. When explicitly enabled, it materializes each F1 root's
   effective runtime defaults and changes one coherent retrieval, ranking, or dialogue
   block at a time within a bounded trust region. The unchanged F1 root remains eligible.
7. Seed-budget successive halving evaluates opt-in local HPO variants, retains only
   stronger family variants for additional seeds, and never forces an HPO result into F2.
8. Backward leave-one-technique ablations test whether strong higher-order combinations
   actually need each member.
9. F1 searches only the frozen search folds. F2 confirms finalists once on prospectively
   disjoint development folds.
10. Safety gates reject incomplete, fit-unsafe, scenario-regressing, non-improving, or
    behaviorally duplicate candidates.
11. Exactly three candidates from one comparable anchor are assigned score-leader,
    robust-leader, and efficient-alternative roles and packaged. If three safe,
    behaviorally distinct candidates do not exist, none are packaged; unsafe padding is
    forbidden.
12. A human prepares, reviews, activates, verifies, or rolls back a chosen proposal.

## Adaptive parameters

The source of truth is `configs/search/adaptive_parameter_space_v1.json`. It currently
contains 42 conditional parameters. Each maps to a real `UnifiedTechniqueConfig` field or
one of the explicit typed derived bindings for question order, sparse field weights, or
normalized fusion share.

| Area | Parameters |
|---|---|
| Question policy | frozen legal `question_order_id`, `question_max_turn`, `eig_candidate_k`, `question_value_margin` |
| Sparse candidate generation | six field weights and `retrieval_k` |
| Fusion | `rrf_constant`, `fusion_sparse_share` |
| Dense routing | `dense_activation`, `dense_activation_min_entropy` |
| PRF expansion | feedback depth, support, term cap, added ratio, activation mode, entropy gate |
| Priors | profile weight/horizon and quality weight |
| Learned ranking | `rerank_k` |
| Cross encoder | weight, depth, activation mode, entropy gate, first active turn |
| Residual Top-10 ranking | feature set, logistic/GBDT/ensemble variant, regularization, rerank depth, parent/model blend, expected-gain and probability-margin gates, movement cap |
| Diversity | relevance weight, rerank depth, active-turn horizon, constraint cap |

Weights with large multiplicative ranges use log sampling. Weighted fusion exposes one
sparse share and derives `dense_weight = 1 - sparse_weight`, preventing invalid sums.
Sparse search exposes all six FTS field weights as one all-or-nothing typed group; a
partial group fails materialization rather than silently falling back. Question ordering
selects only from four frozen legal sequences and materializes the complete tuple.
Parameter conditions prevent unrelated techniques from receiving meaningless knobs.

Local HPO changes at most three parameter groups per proposal. Categorical model-family
changes are isolated, and the six sparse field weights form one normalized atomic group.
The search radius defaults to 20% of each frozen domain. Broad full-space sampling is no
longer used by the campaign orchestrator.

## Observable adaptive activation

Dense retrieval, PRF, and cross-encoding support `always` and `uncertain` activation.
The uncertain policy uses normalized entropy from the current sparse ranking only. It does
not inspect the hidden target, future answers, evaluator labels, or confirmation folds.
Missing uncertainty signals fail open to preserve recall. Every decision is written to the
retrieval trace.

Question horizon and profile-prior horizon are turn-aware. Cross-encoding also has a
minimum turn. These controls alter the actual runtime path and are covered by behavioral
tests; they are not metadata-only parameters.

## Overfitting prevention

- All turns from one session stay in one fold.
- Search folds are `(0, 2, 3)`; confirmation folds are `(1, 4)`.
- F2 is not reused for tuning after confirmation.
- Opt-in HPO seeds, ranges, local center, block and trust radius are deterministic.
- The unchanged F1 parent stays in selection, so poor HPO trials cannot replace it.
- The search-space content hash is part of the campaign manifest.
- Candidates compare against the matched control from the same anchor.
- Promotion uses paired session rewards and scenario-regression limits.
- Exploration reserve prevents overly aggressive early pruning.
- Backward ablation detects passengers in apparently strong combinations.
- Behaviorally identical candidates cannot occupy multiple proposal slots.
- Fit-required additions cannot reach F2 without a safe retraining path. The residual
  ranker is cross-fitted inside the search partition, fitted only on search IDs for F2,
  and emits a hashed receipt for every fold-specific model asset.
- Protected F3 access is forbidden in the manifest and proposal tooling.

`ghostlab/training/protocol.py` defines the fit request and receipt boundary. It rejects
overlapping train/validation session IDs and hashes the exact sample sets and produced
asset. Historical fitted assets stay diagnostic until they are regenerated under this
contract.

## Checkpoint, progress, and recovery

Each job ID is content-addressed by candidate, fidelity, fold, and seed. Outcomes are
written atomically after every job. The lightweight progress file is:

```text
artifacts/campaigns/<campaign-id>/live_status.json
```

It reports total, recorded, complete, failed, and the highest individual completed job.
It also reports the current `stage` and atomic operator `control`.
The aggregate, matched candidate leaderboards remain in `evidence.json` after stage
aggregation; individual-job highs are not promotion evidence.

If execution stops, rerun the exact same mode and template. To intentionally change a
technique, parameter domain, split, or anchor, create a new campaign ID; never reuse an old
checkpoint with changed search inputs.

To skip HPO while F0/F1 is still running, or stop scheduling additional HPO waves after
the current atomic wave, run:

```bash
uv run python -m scripts.control_autonomous_campaign \
  --campaign-id <campaign-id> \
  --skip-hpo
```

Run the same command without `--skip-hpo` for read-only status. A request during F2 is a
no-op because HPO has already completed or been bypassed. The control file is atomic and
survives checkpoint resume.

## Output and human promotion

The full-mode outputs are:

```text
artifacts/campaigns/adaptive_autonomous_full_v1/admission.json
artifacts/campaigns/adaptive_autonomous_full_v1/manifest.json
artifacts/campaigns/adaptive_autonomous_full_v1/plan.json
artifacts/campaigns/adaptive_autonomous_full_v1/checkpoint.json
artifacts/campaigns/adaptive_autonomous_full_v1/live_status.json
artifacts/campaigns/adaptive_autonomous_full_v1/evidence.json
artifacts/proposals/adaptive_autonomous_full_v1/proposal_manifest.json
```

The final command prints one preparation command per proposal. Preparation prints the
exact activation, verification, and rollback commands. The active method changes only if
a human runs the activation command.

## Main implementation files

| Responsibility | File |
|---|---|
| Runtime configuration and builders | `ghostlab/research/technique_suite.py` |
| Adaptive execution gates and traces | `ghostlab/runtime/unified_experimental.py` |
| Typed technique patches and derived fusion | `ghostlab/campaign/bindings.py` |
| Conditional parameter schema | `ghostlab/optimization/conditional.py` |
| BOHB sampler and log domains | `ghostlab/optimization/bohb.py` |
| Campaign stages, halving, interactions, safety | `ghostlab/campaign/orchestrator.py` |
| Atomic checkpoints and live progress | `ghostlab/campaign/runner.py` |
| Atomic runtime operator controls | `ghostlab/campaign/control.py`; `scripts/control_autonomous_campaign.py` |
| Frozen input/search-space validation | `ghostlab/campaign/freeze.py` |
| Fold-safe fitted-asset contract | `ghostlab/training/protocol.py` |
| Residual fold trainer and dispatch | `ghostlab/training/residual.py`; `ghostlab/training/campaign.py` |
| One-command wrapper | `scripts/run_autonomous_end_to_end.py` |
| Search space | `configs/search/adaptive_parameter_space_v1.json` |
| Mode templates | `configs/campaigns/adaptive_autonomous_*_v1.template.json` |

## Validation evidence

The fair 200-session comparison using the same replay evaluator produced:

| Method | Hit@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Guarded champion | 0.980000 | 0.774839 | 2.280 | 0.896852 |
| State Baseline V2 fixed | 0.955000 | 0.587571 | 3.625 | 0.801271 |

The State V2 integration replay also preserves exact historical session hashes for raw,
fixed, and other policies after the adaptive fields were added. On its frozen 150-session
adaptive split, the ranked State V2 augmentation anchor scores `0.885391`.

The full preflight currently accounts for all 36 runnable techniques, reports zero blocked
techniques when assets are prepared, plans 984 low-order structures, and materializes 586.

The independently nested residual evaluation supports its admission as an optional
fit-required technique. On the fair State V2 parent, it changed score from `0.782154` to
`0.876700` (`+0.094546`, all five folds nonnegative); on the stronger ranked State V2
parent it changed `0.885391` to `0.912467` (`+0.027076`, all five folds nonnegative).
Hit@10 membership and MTTC were exactly preserved in both comparisons. These are public
development estimates, not protected-holdout evidence.

## Score-aware recommendation-depth research

Returning fewer than 10 recommendations unconditionally is not beneficial: depths
1/3/5/7/10 score `0.851100`, `0.884250`, `0.890200`, `0.893593`, and `0.896852` on the
same 200-session champion replay. A hindsight oracle reaches `0.930239`, showing possible
but unproven learnable headroom.

The only safe future version is a fit-required, late-stage calibrated policy choosing
depth from `{3, 5, 7, 10}` using target-free ranking/state signals. It must default to 10
under low confidence, distribution shift, stopping, or turn 10; use fold-local
counterfactual training; and pass Hit@10 non-inferiority plus scenario gates. It is not
silently enabled in the current runtime.

## Extension checklist

For any new technique or parameter:

1. implement a real runtime or research path;
2. add a typed config field with a backward-compatible default;
3. declare technique binding, dependencies, conflicts, resources, and fit safety;
4. expose parameters only when the owning technique is enabled;
5. add a behavioral test proving the runtime path changes;
6. freeze a new search-space and campaign version;
7. validate on grouped search folds and confirm once on disjoint folds;
8. update the README, technique operations guide, and decision evidence.
