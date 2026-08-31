# GhostLab optimization engine guide

This is the canonical developer guide for rebuilding, searching, monitoring, and
packaging GhostLab from a clean clone. The root `README.md` is the organizer-facing
setup and benchmark-reproduction guide; this document covers the longer offline
training and optimization workflow.

The commands below are repository-relative. They never require an external inference
API. Model and catalog downloads require network access during setup; campaign execution
runs with Hugging Face and Transformers in offline mode.

## 1. What the engine may optimize

The engine preserves the adaptive shopping workflow:

```text
State V2
  -> observable Buying/Browsing routing
  -> BM25 + category + E5 retrieval
  -> constraint authority
  -> source-aware union
  -> union GBDT
  -> bounded local-LLM semantic ranking with MiniLM fallback
  -> proactive guidance
  -> validated response and atomic state commit
```

Nineteen compulsory techniques implement those required stages. They cannot be removed
or reordered. Twenty promotable techniques may be added or used as architecture-safe
equivalent implementations:

| Family | Promotable techniques |
|---|---|
| Fusion | RRF, weighted fusion, sparse-first union, rank-stack fusion |
| Priors and state | quality prior, profile query/question/union features, catalog normalization |
| Query and retrieval | catalog PRF, MiniLM support, balanced dense views, embedding MMR |
| Ranking | fixed lexical, facet diversity, metadata GBDT, fold ensemble, reward/turn-aware LambdaMART, Top-10 residual reranker |

The complete machine-readable registry is
`configs/techniques/catalog_v2.json`. The adaptive role mapping and typed patches are in
`ghostlab/optimization/adaptive_techniques.py`. Control-only techniques remain available
for comparison or dependency evidence but cannot silently replace compulsory stages.
Research-only procedures support search and evaluation; unavailable entries are recorded
for inventory completeness and are never presented as runnable candidates.

## 2. Progressive racing

All candidates in a phase use the same ordered, lineage-safe session subset.

| Phase | Sessions | Purpose |
|---|---:|---|
| F0 | 330 | Broad screening of controls, add-one candidates, ablations, and compatible combinations |
| F1 | 825 | Stronger paired evidence plus conditional local tuning for survivors |
| F2 | 1,650 | Full-development confirmation and finalist eligibility |

Candidates are compared using TechnicalScore, Hit Rate@10, MRR, scenario and source
slices, constraint violations, fallback behavior, latency, and paired session rewards.
Clearly dominated candidates are pruned before expensive phases. Conditional parameters
are tuned only when their parent technique is active. One to three eligible F2
challengers may be packaged; three is an upper bound, not a minimum.

The 550-session final-selection partition is outside the optimizer. Do not use it for
planning, fitting, tuning, pruning, or rerunning an optional campaign.

## 3. Clean-machine setup

### 3.1 Clone and install

```bash
git clone \
  --branch feat/adaptive-hybrid-1a-3b \
  --single-branch \
  https://github.com/kjcheong03/tiktok-techjam.git

cd tiktok-techjam

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv sync --frozen --python 3.12 --all-extras
uv pip check
```

Python 3.12 is recommended. Supported hosts are macOS, Linux, and Windows through WSL2.
Allow approximately 10 GB of disk space and at least 8 GB RAM; 16 GB is recommended.

### 3.2 Download the frozen 50,000-product catalog

```bash
curl --fail --location \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS \
  --output data/SHA256SUMS

curl --fail --location \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz \
  --output data/catalog.jsonl.gz

(
  cd data
  grep '  catalog.jsonl.gz$' SHA256SUMS | shasum -a 256 -c -
)

gzip -dk data/catalog.jsonl.gz
test "$(wc -l < data/catalog.jsonl | tr -d ' ')" = "50000"
```

The expected compressed-file SHA-256 is
`07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`.

### 3.3 Build the catalog ontology

```bash
mkdir -p artifacts/assets

.venv/bin/python -m scripts.build_attribute_ontology \
  --catalog data/catalog.jsonl \
  --output artifacts/assets/catalog_ontology_v1.json

echo \
  "3821d4f6772f7bb257c27e4ae0b85937001c15cefcffb53c74dbad2c6dd408f7  artifacts/assets/catalog_ontology_v1.json" \
  | shasum -a 256 -c -
```

### 3.4 Download and verify model assets

```bash
.venv/bin/python -m scripts.fetch_optional_assets e5
.venv/bin/python -m scripts.fetch_optional_assets minilm
.venv/bin/python -m scripts.fetch_optional_assets smollm2_ranker
.venv/bin/python -m scripts.fetch_optional_assets cross_encoder

.venv/bin/python -m scripts.fetch_optional_assets e5 --verify-only
.venv/bin/python -m scripts.fetch_optional_assets minilm --verify-only
.venv/bin/python -m scripts.fetch_optional_assets smollm2_ranker --verify-only
.venv/bin/python -m scripts.fetch_optional_assets cross_encoder --verify-only
```

E5, MiniLM, and SmolLM2 are required by the current adaptive workflow. The compact
cross-encoder is fetched for the complete optimizer's registered alternative. Gemma,
Qwen2.5, and Qwen3 were historical development candidates and are not required.

If Hugging Face throttles anonymous downloads, export a read-only token:

```bash
export HF_TOKEN="YOUR_READ_TOKEN"
```

### 3.5 Download and verify dense indexes

```bash
.venv/bin/python -m scripts.fetch_dense_index_asset
.venv/bin/python -m scripts.fetch_dense_index_asset --verify-only
```

The downloader checks the release archive and extracted files, catalog hash, pinned model
revision, row count, dimensions, and dtype.

### 3.6 Verify inputs and the current frozen champion

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_active_candidate.py
PYTHONPATH=. .venv/bin/python scripts/verify_reproduction_bundle.py
```

Both commands must finish with `"verified": true`.

## 4. Validate the engine without running it

These commands validate imports, data, folds, output paths, architecture bindings,
candidate inventory, warm-start translation, and pipeline wiring. They do not fit models
or evaluate candidates.

```bash
PYTHONPATH=. .venv/bin/python scripts/train_adaptive_hybrid.py --plan-only

PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py --show-plan

PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json \
  --warm-start configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  --plan-only
```

Expected facts include 1,650 development sessions, five lineage-disjoint folds, 11
pipeline stages, 88 catalog entries, 19 compulsory techniques, 20 promotable techniques,
and translated warm-start ID `d4e040a07e6d-translated-v2`. The historical runtime is
never executed directly.

## 5. Choose a search profile

### 5.1 Exhaustive search

Use this when time is not tightly constrained. It searches broadly from the fixed
adaptive control and can expand strong survivors into higher-order combinations.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --campaign-search-profile exhaustive
```

Default campaign limits are 500 F0 candidates, beam width 24, eight higher-order rounds,
24 F1 candidates, six F2 candidates, and two local HPO trials per surviving structure.

### 5.2 Focused warm start

This uses the architecture-safe translation of the strongest historical composable
candidate as a seed while still allowing changes around it.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --campaign-search-profile focused_warm_start \
  --campaign-warm-start \
    configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json
```

The historical implementation itself is not executed. Compatible techniques and
parameters are translated into the current compulsory architecture and fitted assets are
rebuilt against the current development folds.

### 5.3 Additive warm start

This is the most time-efficient profile. It preserves the translated warm seed and tests
monotonic compatible additions rather than deleting its proven techniques.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --campaign-search-profile additive_warm_start \
  --campaign-warm-start \
    configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json
```

The focused additive inventory contains RRF, sparse-first union, the profile union
feature, facet diversity, embedding MMR, and balanced dense views. It tests up to three
compatible additions and may reuse only exact matching evaluations from the broad
checkpoint. Reuse is read-only and hash/signature checked.

### 5.4 Bounded smoke run

For operational testing only, limit the campaign's development prefix and search size:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --campaign-search-profile focused_warm_start \
  --campaign-warm-start \
    configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  --campaign-candidate-limit 4 \
  --campaign-f1-candidates 2 \
  --campaign-f2-candidates 1 \
  --campaign-hpo-trials 0 \
  --campaign-max-samples 30 \
  --through-stage campaign
```

This is a smoke test, not selection evidence and not a substitute for F2 confirmation.

## 6. Run persistently

Foreground execution prints every stage and campaign event directly.

For a persistent macOS run:

```bash
mkdir -p artifacts/logs
nohup caffeinate -dimsu env PYTHONPATH=. PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --campaign-search-profile additive_warm_start \
  --campaign-warm-start \
    configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  > artifacts/logs/adaptive_hybrid_pipeline.log 2>&1 &
echo $!
```

For Linux or WSL2, omit `caffeinate -dimsu`:

```bash
mkdir -p artifacts/logs
nohup env PYTHONPATH=. PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  --campaign-search-profile additive_warm_start \
  --campaign-warm-start \
    configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  > artifacts/logs/adaptive_hybrid_pipeline.log 2>&1 &
echo $!
```

Closing the terminal does not stop a `nohup` run. On macOS, `caffeinate` prevents system
sleep while the process exists. Display dimming is harmless; closing the laptop lid may
still suspend it.

## 7. Monitor, interrupt, and resume

Follow the pipeline and active campaign logs:

```bash
tail -f artifacts/logs/adaptive_hybrid_pipeline.log
tail -f artifacts/logs/adaptive_hybrid_pipeline/campaign.log
```

Structured monitor for exhaustive or focused warm-start campaigns:

```bash
PYTHONPATH=. .venv/bin/python scripts/monitor_adaptive_campaign.py \
  --checkpoint artifacts/campaigns/adaptive_hybrid_1650_v1/checkpoint.json \
  --log artifacts/logs/adaptive_hybrid_pipeline/campaign.log \
  --pipeline-checkpoint \
    artifacts/campaigns/adaptive_hybrid_pipeline/checkpoint.json
```

For the additive profile, use:

```bash
PYTHONPATH=. .venv/bin/python scripts/monitor_adaptive_campaign.py \
  --checkpoint \
    artifacts/campaigns/adaptive_hybrid_additive_warm_start_1650_v1/checkpoint.json \
  --log artifacts/logs/adaptive_hybrid_pipeline/campaign.log \
  --pipeline-checkpoint \
    artifacts/campaigns/adaptive_hybrid_pipeline/checkpoint.json
```

Add `--once` for a single snapshot. The monitor reports phase progress, active candidate,
the matched C control, the highest challenger in the current phase, and score delta.

Interrupt foreground work with `Ctrl+C`. For a background job, send `TERM` to the exact
PID recorded at launch. Run the identical pipeline command to resume. A stage is reused
only when its checkpoint status, command signature, script hash, and expected outputs all
match. Use `--show-plan` before resuming to see which stages will run.

Do not delete checkpoints to restart casually. Use `--force-stage STAGE` only when the
named stage intentionally needs to be recomputed.

## 8. Pipeline stages and outputs

| Stage | Work performed |
|---|---|
| `split` | Reconstruct and audit the lineage-safe 1,650/550 partition |
| `fit` | Cross-fit and refit the source-aware union GBDT |
| `diversity` | Validate dense-view diversity behavior |
| `llm` | Prepare the fixed SmolLM2 semantic-control configuration |
| `evaluate` | Evaluate C on all development sessions |
| `baselines` | Evaluate the organizer BM25 reference |
| `validate` | Run architecture, training-receipt, constraint, and metric checks |
| `campaign` | Run F0/F1/F2 technique and parameter racing |
| `package` | Materialize up to three eligible, hash-bound challengers |
| `finalists` | Evaluate packaged challengers on identical development sessions |
| `compare` | Build the unified A/C/D development comparison |

Restrict execution with `--from-stage` and `--through-stage`. Inspect the exact commands
and outputs with:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py --show-plan
```

Primary outputs are:

```text
data/splits/adaptive_hybrid_lineage_75_25_v1.json
configs/adaptive_hybrid_1a_3b_1650_final_v1.json
configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json
artifacts/models/adaptive_union_gbdt_1650_final_v1.json
artifacts/models/adaptive_union_gbdt_1650_final_v1.fit_receipt.json
artifacts/reports/adaptive_hybrid_training_1650_final_v1.json
artifacts/reports/adaptive_hybrid_campaign_1650.json
artifacts/reports/adaptive_hybrid_top3.json
artifacts/reports/adaptive_finalist_development_evaluations.json
artifacts/reports/adaptive_system_comparison_1650.json
```

Campaign execution never overwrites `configs/active_candidate.json`.

## 9. Package and activate a reviewed finalist

The normal pipeline packages finalists automatically. To rerun packaging from a complete
campaign checkpoint:

```bash
PYTHONPATH=. .venv/bin/python scripts/package_adaptive_top_three.py \
  --campaign-report artifacts/reports/adaptive_hybrid_campaign_1650.json \
  --base-config configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json \
  --campaign-checkpoint \
    artifacts/campaigns/adaptive_hybrid_1650_v1/checkpoint.json \
  --output artifacts/reports/adaptive_hybrid_top3.json
```

One to three eligible challengers are valid. Activation is deliberately manual and
requires the exact preset hash plus the frozen finalist and holdout evidence:

```bash
PYTHONPATH=. .venv/bin/python scripts/activate_adaptive_candidate.py \
  --preset PATH_TO_REVIEWED_PRESET.json \
  --expected-sha256 EXPECTED_PRESET_SHA256 \
  --top3-report artifacts/reports/adaptive_hybrid_top3.json \
  --holdout-report artifacts/reports/adaptive_final_holdout.json
```

Do not activate the largest raw score automatically. Review paired evidence, protected
slices, constraint behavior, fit receipts, latency, and architecture audit first.

## 10. Reproduce published results

Official 200-session rerun:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=. .venv/bin/python scripts/evaluate_ac_finalist_public_200.py
```

Verify the complete frozen bundle, including the recorded one-time 550 report:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_reproduction_bundle.py
```

Do not rerun or tune against the 550 partition after its result has been revealed.

## 11. Validation and troubleshooting

Run the documentation and adaptive-engine tests:

```bash
uv run pytest -q \
  tests/test_essential_docs.py \
  tests/test_adaptive_warm_start.py \
  tests/test_adaptive_campaign_integration.py \
  tests/test_adaptive_top_three.py \
  tests/test_adaptive_pipeline_runner.py
```

Common failures:

- **`.venv/bin/python` missing:** run `uv sync --frozen --python 3.12 --all-extras`
  from the repository root.
- **`No module named evaluator`:** prefix direct script calls with `PYTHONPATH=.` or
  use the pipeline wrapper, which sets it automatically for child stages.
- **Missing catalog or ontology:** repeat Sections 3.2 and 3.3.
- **Missing model/index:** repeat Sections 3.4 and 3.5 with `--verify-only` afterward.
- **Warm-start hash mismatch:** do not regenerate the source preset; verify the tracked
  file and warm-start specification before proceeding.
- **Campaign appears static:** compare only within the same fidelity. An F0 challenger
  is not directly comparable with an F1 control.
- **Highest challenger not shown:** the current phase must finish at least one challenger
  after its matched control.
- **Resume wants to rerun a stage:** inspect `--show-plan`; command, script, or output
  changes intentionally invalidate downstream reuse.

The dashboard is independent of campaign execution:

```bash
.venv/bin/python dashboard/server.py
```

Open <http://127.0.0.1:8787/dashboard/>.
