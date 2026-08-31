# GhostLab Shopping Copilot

## 1. Project overview

GhostLab is a local conversational shopping agent for targeted buying and open-ended
discovery. On every turn it updates shopper intent, retrieves candidates through
keyword, category, and semantic search, removes confirmed constraint violations, and
ranks the merged pool with learned and selectively activated local-LLM scoring. It
returns up to ten validated recommendations and, for overly broad requests, switches to
bounded retrieval plus one clarification question.

GhostLab searches the organizer-provided 50,000-product Amazon Clothing, Shoes and
Jewelry catalog. Its offline optimizer uses a lineage-grouped 2,200-session corpus: 200
official sessions, 1,000 public-scenario variations with new targets, and 1,000
independently generated scenarios with new profiles and targets.

The governing design rule is:

> The required adaptive-shopping capabilities remain fixed; GhostLab optimizes how
> each capability is implemented and may add compatible optional techniques.

### A. Architecture

| Path | When used | Behavior |
|---|---|---|
| Buying | Specific goal or firm requirements | BM25-led retrieval for precision, supported by category and E5 |
| Browsing | Open-ended scenario or discovery request | Multi-view E5 for semantic discovery, supported by BM25 and category |
| Normal retrieval | The request is sufficiently defined | Full source-aware union, learned ranking, and optional local-LLM ranking |
| Bounded cutoff | The request is overly general | Reduced retrieval, safe ranking, preliminary recommendations, and one clarification |

The runtime follows this fixed order:

```text
message + supplied profile
  -> State V2 conversation update
  -> Buying/Browsing routing
  -> BM25 + category + E5 candidate generation
  -> constraint-authority filtering
  -> source-aware candidate merge
  -> union GBDT ranking
  -> bounded local-LLM semantic ranking when eligible
  -> optional residual reranking
  -> response validation and atomic state commit
```

Duplicate candidates retain each source's score and rank so the learned ranker can
combine exact wording, catalog structure, and semantic evidence. Conversation state
accumulates compatible requirements, replaces corrected attributes, and uses intent
epochs to isolate obsolete constraints and recommendations after a change of shopping
goal. Explicit session requirements override conflicting profile preferences.

The active **GhostLab Champion** preserves this architecture and adds RRF fusion plus a
hash-bound Top-10 residual reranker. Its configuration is selected by
`configs/active_candidate.json`; `starter.Agent` resolves that pointer at runtime.

### B. Offline optimization

GhostLab keeps the runtime stages fixed while its offline optimizer compares complete
combinations of routing, retrieval, fusion, diversity, ranking, local-LLM, and
clarification settings. Grouped folds prevent related synthetic sessions from crossing
partitions. F0/F1/F2 progressive racing evaluates weak candidates cheaply before
allocating all 1,650 development sessions to the strongest survivors. Eligible
candidates must pass architecture, constraint, fit-receipt, latency, and paired-quality
checks before one to three configurations can be frozen.

The current active champion was selected from this process. Running the optimizer is
optional for reproducing the published champion and benchmark results; it is included
below for researchers who want to reproduce the complete search.

### C. Data

| Source | Development | One-time final selection | Total |
|---|---:|---:|---:|
| Official sessions | 150 | 50 | 200 |
| Public-scenario variations with new targets | 750 | 250 | 1,000 |
| Independent scenarios with new profiles and targets | 750 | 250 | 1,000 |
| **Total** | **1,650** | **550** | **2,200** |

- **Public-scenario variations:** five variants were created from each official
  session while assigning new catalog targets.
- **Independent scenarios:** 200 new shopper and information-disclosure templates
  produced five unique-target sessions each.
- **Leakage prevention:** target selection did not run GhostLab, and related session
  families remain together in the same partition and nested fold.

The official 200-session benchmark is public and overlaps the development process. It
is reproducibility and demonstration evidence, not an unseen generalization estimate.
The 550-session partition was accessed once for final selection. The organizer's private
competition evaluation remains separate and unseen.

### D. Evaluation

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

Higher is better for every metric except MTTC. Only exact `parent_asin` equality counts
as a hit.

#### Official public benchmark — 200 sessions

| System | Hit Rate@10 | MRR | MTTC | Normalized Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 Starter (A) | 0.125000 | 0.068034 | 9.810000 | 0.119000 | 0.106710 |
| Fixed Adaptive Architecture (C) | 0.970000 | 0.572325 | 2.775000 | 0.822500 | 0.821197 |
| **GhostLab Champion** | **0.975000** | **0.688401** | **2.765000** | **0.823500** | **0.858720** |

#### One-time final selection — 550 sessions

| System | Hit Rate@10 | MRR | MTTC | Normalized Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 Starter (A) | 0.190909 | 0.101387 | 9.110909 | 0.188909 | 0.163652 |
| Fixed Adaptive Architecture (C) | 0.961818 | 0.568150 | 2.723636 | 0.827636 | 0.816881 |
| **GhostLab Champion** | **0.965455** | **0.676361** | **2.689091** | **0.831091** | **0.851854** |

The A, C, and champion rows in each table use the same ordered sessions, catalog,
response contract, evaluator semantics, Top-K and turn limits. The repository retains
the comparison reports, per-system session rows, access receipt, active pointer, and
selection record as reproducibility evidence.

## 2. Setup

These instructions start from a clean machine. No untracked training output from the
authors' worktree is required. The catalog and large model/index files are deliberately
downloaded rather than committed to Git.

### 2.1 Requirements

- macOS, Linux, or Windows through WSL2
- Python 3.10–3.13; Python 3.12 is recommended
- Git, `curl`, `gzip`, and `shasum`
- [`uv`](https://docs.astral.sh/uv/)
- Internet access for the catalog, model assets, and prebuilt dense indexes
- About 10 GB of free disk space
- 8 GB RAM minimum; 16 GB recommended
- Optional `HF_TOKEN` for more reliable Hugging Face downloads
- Optional Apple Silicon MPS or CUDA GPU; CPU execution is supported

### 2.2 Clone the submission branch

```bash
git clone \
  --branch feat/adaptive-hybrid-1a-3b \
  --single-branch \
  https://github.com/kjcheong03/tiktok-techjam.git

cd tiktok-techjam
git rev-parse HEAD
```

Record the printed commit hash with reproduced results. After this branch is merged,
the explicit `--branch` option may be omitted.

### 2.3 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 2.4 Create the locked runtime environment

```bash
uv python install 3.12
uv sync --frozen --python 3.12 --extra all --no-dev
uv pip check
```

For development and the complete test suite, use `uv sync --frozen --all-extras`
instead. The runtime-only environment is sufficient for setup and result reproduction.

### 2.5 Download and verify the frozen catalog

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

The checksum command must report `catalog.jsonl.gz: OK`. The expected compressed-file
SHA-256 is:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

### 2.6 Generate and verify the catalog ontology

```bash
mkdir -p artifacts/assets

.venv/bin/python -m scripts.build_attribute_ontology \
  --catalog data/catalog.jsonl \
  --output artifacts/assets/catalog_ontology_v1.json

echo \
  "3821d4f6772f7bb257c27e4ae0b85937001c15cefcffb53c74dbad2c6dd408f7  artifacts/assets/catalog_ontology_v1.json" \
  | shasum -a 256 -c -
```

The final command must report `artifacts/assets/catalog_ontology_v1.json: OK`.

### 2.7 Download and verify the pinned model assets

The published champion needs E5-small-v2, All-MiniLM-L6-v2 as a safe fallback, and
SmolLM2-1.7B-Instruct. The cross-encoder is also fetched so the complete optimizer can
evaluate its registered semantic alternative. Gemma, Qwen, and Qwen3 are historical
experiment models and are not required.

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

Every verification command must print `"verified": true`. If Hugging Face throttles
anonymous downloads, export a read token first:

```bash
export HF_TOKEN="YOUR_READ_TOKEN"
```

### 2.8 Download and verify the 50,000-product dense indexes

```bash
.venv/bin/python -m scripts.fetch_dense_index_asset
.venv/bin/python -m scripts.fetch_dense_index_asset --verify-only
```

The versioned GitHub Release asset contains the E5 and MiniLM matrices and metadata.
The downloader verifies the archive SHA-256, every extracted file, catalog hash, model
revision, row count, dimensions, and dtype. For a private repository, authenticate with
`GH_TOKEN`, `GITHUB_TOKEN`, or `gh auth login`; no authentication is needed when the
repository and Release are public.

### 2.9 Verify the frozen champion and tracked evidence

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_active_candidate.py
PYTHONPATH=. .venv/bin/python scripts/verify_reproduction_bundle.py
```

Both commands must print `"verified": true`. Together they verify the active adaptive
configuration, architecture contract, configuration hashes, union GBDT, residual model
and fit receipt, official-200 reports, and 550-session report/access receipt.

### 2.10 No-training preflight

```bash
PYTHONPATH=. .venv/bin/python scripts/train_adaptive_hybrid.py --plan-only
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py --show-plan
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json \
  --warm-start configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  --plan-only
```

Expected preflight facts:

- 1,650 development sessions and five lineage-disjoint folds
- 11 ordered pipeline stages: `split`, `fit`, `diversity`, `llm`, `evaluate`,
  `baselines`, `validate`, `campaign`, `package`, `finalists`, `compare`
- 88 catalog techniques, 19 compulsory techniques, 20 promotable techniques, and
  196 valid initial candidates in the broad campaign plan
- translated warm-start ID `d4e040a07e6d-translated-v2`
- `"historical_runtime_executed": false`

These commands inspect inputs and plans. They do not train, access the 550-session
partition, or change the active champion.

## 3. Results reproduction

### 3.1 Re-run A, C, and the GhostLab Champion on the official 200 sessions

After completing Section 2, run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=. .venv/bin/python scripts/evaluate_ac_finalist_public_200.py
```

The command evaluates all three systems on the same ordered 200 sessions and writes:

```text
artifacts/reports/adaptive_public_200.json
artifacts/reports/adaptive_public_200_runs/reference_a.json
artifacts/reports/adaptive_public_200_runs/control_c.json
artifacts/reports/adaptive_public_200_runs/ghostlab_finalist.json
artifacts/reports/adaptive_ac_finalist_benchmark_index.json
```

Re-run the bundle verifier afterward:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_reproduction_bundle.py
```

The metric values should match the official-public table in Section 1.D to six decimal
places. Hardware-dependent latency diagnostics may differ, but Hit Rate@10, MRR, MTTC,
normalized efficiency, and TechnicalScore are evaluator outcomes and must match.

For a quick integrity check without re-executing the agents, rebuild the comparison
manifest from the tracked per-system rows:

```bash
PYTHONPATH=. .venv/bin/python \
  scripts/evaluate_ac_finalist_public_200.py --reuse-existing
```

### 3.2 Verify the recorded 550-session final-selection result

The 550-session partition was intentionally accessed once. Reproducibility therefore
means verifying the immutable report, access receipt, per-system rows, active champion,
and hashes—not spending the partition again.

```bash
echo \
  "41234240b81673826446fd649bcd953d901e3031a76a1821471c4d674de6faa9  artifacts/reports/adaptive_final_holdout.json" \
  | shasum -a 256 -c -

echo \
  "5d54299681a60ccfde198f0722013bb20362d0186dd238fbe6b84059c8fd6e10  artifacts/reports/adaptive_final_holdout.access_receipt.json" \
  | shasum -a 256 -c -

PYTHONPATH=. .venv/bin/python scripts/verify_reproduction_bundle.py
```

The two checksum commands must report `OK`, and the verifier prints the 550 metrics in
Section 1.D. The organizer-private evaluation is not present in this repository.

### 3.3 Optional: reproduce training and GhostLab optimization

This long path is not needed to use or verify the frozen champion. It retrains the
development ranker, validates the fixed architecture, searches compatible optional
techniques, freezes one to three eligible finalists, and compares them on the same
1,650 development sessions.

Review the plan:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py --show-plan
```

Run in the foreground:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py
```

Persistent macOS run:

```bash
mkdir -p artifacts/logs
nohup caffeinate -dimsu env PYTHONPATH=. PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  > artifacts/logs/adaptive_hybrid_pipeline.log 2>&1 &
echo $!
```

Persistent Linux/WSL2 run:

```bash
mkdir -p artifacts/logs
nohup env PYTHONPATH=. PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/run_adaptive_hybrid_pipeline.py \
  > artifacts/logs/adaptive_hybrid_pipeline.log 2>&1 &
echo $!
```

Monitor or resume:

```bash
tail -f artifacts/logs/adaptive_hybrid_pipeline.log
tail -f artifacts/logs/adaptive_hybrid_pipeline/campaign.log

# If interrupted, run the identical pipeline command again.
```

The checkpoint is
`artifacts/campaigns/adaptive_hybrid_pipeline/checkpoint.json`. Completed stages are
reused only when their command signatures and expected outputs still match. Campaign
results never overwrite `configs/active_candidate.json`; activation remains explicit.

Expected development outputs include:

```text
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

The finalist package contains **one to three** eligible challengers, capped at three;
three is not a minimum. Do not rerun the recorded 550-session selection after this
optional development experiment.

### 3.4 Run the results dashboard

```bash
.venv/bin/python dashboard/server.py
```

Open <http://127.0.0.1:8787/dashboard/>. The dashboard discovers compatible JSON
reports under `artifacts/reports/` and accepts imported evaluator JSON files.

## 4. Limitations and future improvements

- **Retrieval ceiling:** ranking can promote only products found by at least one
  retrieval source. More semantic query views may improve candidate recall.
- **Synthetic-data dependence:** optimization combines official and lineage-safe
  synthetic sessions. Only the organizer-private evaluation measures generalization to
  unseen organizer scenarios.
- **Local-LLM trade-off:** SmolLM2 improves semantic evidence on selected Browsing
  turns but adds latency. Its activation is bounded and falls back safely.
- **Domain scope:** the current catalog and examples are clothing-domain data; they do
  not establish performance on unrelated product domains.
- **Profile persistence:** GhostLab emits conflict-aware profile updates, but durable
  storage, consent, retention, and privacy controls belong to the hosting application.
- **Hardware variance:** model loading and latency vary by CPU, MPS, CUDA, memory, and
  operating system even when evaluator outcomes remain deterministic.

## 5. Team member contributions

| Team member | Contributions |
|---|---|
| Chloe Chua | Baseline implementation, runtime architecture planning, project documentation, and testing |
| Cheong Kang Jie | GhostLab optimization engine, adaptive runtime integration, training/evaluation pipeline, champion selection, and reproducibility packaging |
| Chew En Wei | Contribution summary to be confirmed by the team before submission |
| Lucas Sam | Results dashboard and frontend visualization |

### Important references

- [Complete optimization engine guide](docs/engine_guide.md)
- [Runtime architecture overview](docs/architecture_overview.md)
- [Offline training and optimization pipeline](docs/offline_training_optimization_pipeline.md)
- [Results dashboard](dashboard/README.md)
- [Competition specification](docs/competition_specification.md)
- [Agent API contract](docs/agent_api_contract.json)
- [Submission rules](docs/submission_rules.md)
- [Competition data layout](data/README.md)
- [Data attribution](DATA_ATTRIBUTION.md)

The catalog and development data are derived from Amazon Reviews 2023 by McAuley Lab,
UCSD. Review `DATA_ATTRIBUTION.md` before redistribution.
