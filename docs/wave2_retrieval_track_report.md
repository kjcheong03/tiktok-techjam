# Wave 2 retrieval track report

Status: implementation and local mechanism validation complete on
`exp/w2-modern-retrieval`. Nothing in this document activates a new champion.

Protected-data rule: every executed diagnostic and replay used only the 150 IDs in
`configs/splits/adaptive_v1.json`. The sealed 50-session F3 split was not read.

## Outcome

| Technique | Stable ID | State after local gates | Why |
|---|---|---|---|
| Learned sparse rescue | `retrieval.splade_rescue.v1` | unavailable | No pinned local SPLADE model/index; no offline, license, recall, latency, memory, or packaging evidence. |
| Sparse semantic union | `fusion.sparse_semantic_union.v1` | available, dependency blocked | Deterministic weighted-RRF implementation is tested, but no admitted learned-sparse head exists. |
| ColBERT rescue | `retrieval.colbert_rescue.v1` | unavailable | No pinned local model/index; the conservative flat reference exceeds the declared query-compute gate. |
| BGE-M3 rescue | `retrieval.bge_m3_rescue.v1` | unavailable | Same feasibility state; it was not silently substituted with one-vector dense retrieval. |
| Late-interaction union | `fusion.late_interaction_union.v1` | available, dependency blocked | Fusion contract exists; no admitted full-catalog late-interaction head exists. |
| Catalog PRF | `query.catalog_prf.v1` | parked | First-turn Recall@200 fell from `0.713333` to `0.693333`; 20 rescues were offset by 23 losses. |
| Expansion guard | `query.expansion_guard.v1` | available | Preserves explicit terms and deterministically bounds additions. Retain for ontology- or router-dependent retesting. |
| Early MMR | `ranking.mmr_early.v1` | interaction reserve | Small exploratory gain with no first-turn target-rank regressions under the frozen control. |
| Facet diversity | `ranking.facet_diversity.v1` | interaction reserve | Top-10 facet coverage rose from `379.046667` to `382.480000`; needs a better browsing router and matched F2 validation. |
| Local Query2Doc | `query.query2doc_local.v1` | not implemented | No pinned offline generative asset was available, and simpler PRF failed its gate. |

## Exact implementation map

### Learned sparse rescue

- `ghostlab/retrieval/learned_sparse.py`: SPLADE encoder, pickle-free sparse CSR
  index, exact inverted search, checksummed offline asset loader.
- `ghostlab/retrieval/sparse_semantic_fusion.py`: deterministic weighted-RRF union.
- `scripts/build_learned_sparse_index.py`: local-only builder interface.
- `configs/assets/splade_rescue_v1.json`: honest unavailable asset manifest.
- `configs/experiments/w2_splade_rescue_v1.json`: recall/resource gate.
- `tests/test_learned_sparse.py`: reference encoder, round-trip, ranking, manifest,
  and fusion tests.

The runtime switch is
`retrieval_route=learned_sparse_union` with `learned_sparse_asset=<manifest>` and
`semantic_rescue_weight=<0..1>`. The manifest must be marked available and provide
confined, checksummed local model/index paths. When the switch is off, no
Transformer or Torch import, asset check, or index open occurs.

### Late interaction

- `ghostlab/retrieval/late_interaction.py`: MaxSim, local-only token encoder,
  pickle-free ragged token store, exact bounded reference retrieval/reranking, and
  resource estimator.
- `scripts/build_late_interaction_index.py`: local-only reference index builder.
- `configs/assets/late_interaction_rescue_v1.json`: honest unavailable manifest.
- `configs/experiments/w2_late_interaction_rescue_v1.json`: feasibility gate.
- `tests/test_late_interaction.py`: MaxSim, persistence, retrieval, and resource
  tests.

The future runtime switch is `retrieval_route=late_interaction_union` with
`late_interaction_asset=<manifest>`. The present manifest makes construction fail
explicitly. It never falls back silently to dense or lexical retrieval.

The declared flat reference for 50,000 products, 64 document tokens, 128 dimensions,
float16 storage, and 24 query tokens estimates about `0.763 GiB` of raw embeddings
and `76.8 million` query-token/document-token dot products per query. Storage passes
the provisional 1 GiB bound, but compute exceeds the 50 million gate before Python,
model, and index overhead. A compact candidate generator or measured optimized
implementation is required before recall evaluation.

### Catalog PRF expansion

- `ghostlab/state/query_expansion.py`: provenance-bearing expansion values and
  guard.
- `ghostlab/retrieval/pseudo_relevance.py`: catalog-grounded, IDF/support/rank-
  weighted PRF.
- `scripts/run_query_expansion_challenger.py`: adaptive-split replay wrapper.
- `configs/suites/w2_prf_core.json`: runnable switch preset.
- `configs/experiments/w2_query_expansion_v1.json`: frozen mechanism settings.
- `tests/test_query_expansion.py`: grounding, support, guard, and insufficient-
  feedback behavior.

Switches: `query_expansion=off|prf`, `expansion_feedback_k`,
`expansion_min_support`, `expansion_max_terms`, and
`expansion_max_added_ratio`. The original query is retained verbatim and expansion
terms are appended. PRF uses only observable retrieved products.

### Conditional facet MMR

- `ghostlab/retrieval/diversify.py`: metadata Jaccard similarity, bounded MMR,
  activation context, and facet coverage.
- `scripts/run_diversification_challenger.py`: adaptive-split replay wrapper.
- `configs/suites/w2_facet_mmr_core.json`: runnable switch preset.
- `configs/suites/w2_prf_facet_mmr_core.json`: within-track interaction preset.
- `configs/experiments/w2_diversification_v1.json`: frozen mechanism settings.
- `tests/test_diversify.py`: deterministic activation, preservation, and MMR tests.

Switches: `diversification=off|facet_mmr`, `diversification_weight`,
`diversification_rerank_k`, `diversification_output_k`,
`diversification_max_turn`, and `diversification_max_constraints`. The weight is
the relevance share in MMR and must be retuned inside the combination's inner folds,
not copied blindly from this standalone test. The implementation only reorders a
bounded candidate head, preserves the first result, never drops candidates, and
turns off for late or constraint-specific states.

## Validation evidence

Mechanism report: `artifacts/reports/w2_retrieval_mechanism_v1.json`.

First-turn retrieval gate on 150 adaptive sessions:

| Route | Recall@10 | Recall@50 | Recall@100 | Recall@200 |
|---|---:|---:|---:|---:|
| Frozen BM25 control | 0.173333 | 0.406667 | 0.560000 | 0.713333 |
| Catalog PRF | 0.166667 | 0.413333 | 0.506667 | 0.693333 |
| Facet MMR | 0.180000 | 0.406667 | 0.560000 | 0.713333 |

The PRF mechanism does not pass. It remains switchable because normalization,
conditional routing, or a different retrieval head may change the interaction.

Exploratory 150-session adaptive replays used one all-development metadata-GBDT
asset. They are matched F1 diagnostics, **not** the fold-local nested OOF evidence
needed to beat `0.878963`:

| Configuration | Technical | Hit@10 | MRR | MTTC | Decision |
|---|---:|---:|---:|---:|---|
| Matched control | 0.857554 | 0.973333 | 0.661624 | 2.380000 | control |
| Facet MMR | 0.858054 | 0.973333 | 0.663291 | 2.380000 | reserve; +0.000500 |
| Catalog PRF | 0.830919 | 0.940000 | 0.652841 | 2.746667 | park |
| PRF + facet MMR | 0.830919 | 0.940000 | 0.652841 | 2.746667 | no positive interaction |

Replay reports are `artifacts/reports/w2_control_replay_v1.json`,
`w2_facet_mmr_replay_v1.json`, `w2_prf_replay_v1.json`, and
`w2_prf_facet_mmr_replay_v1.json`. The reproducible paired analysis is
`artifacts/reports/w2_retrieval_decision_v1.json`.

Paired session-reward diagnostics reinforce the cautious decisions. Facet MMR had
2 wins, 148 ties, and 0 losses; its mean delta was `+0.000500`, bootstrap 95%
interval `[0.000000, 0.001333]`, and paired-randomization `p=0.501250`. PRF had 30
wins, 76 ties, and 44 losses; its mean delta was `-0.026635`, bootstrap interval
`[-0.054494, -0.000575]`, and paired-randomization `p=0.054695`. These reused-
development diagnostics do not justify champion activation.

## Dependencies and assets

Core PRF and facet MMR require only the base installation:

```bash
uv sync --group dev
```

The reference model adapters use imports already covered by the existing `neural`
extra, but that does not make a model candidate available:

```bash
uv sync --group dev --extra neural
```

Do not download a model at runtime. To admit SPLADE or late interaction, first pin a
model revision and license, place the model locally under ignored
`artifacts/cache/models/`, build a content-addressed ignored index, run the 150-ID
recall/resource gate offline, and only then create a new manifest version marked
available. Large models and indices are never committed.

## Reproduction

```bash
uv run --frozen python -m scripts.run_wave2_retrieval_diagnostics \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --split configs/splits/adaptive_v1.json \
  --output artifacts/reports/w2_retrieval_mechanism_v1.json

uv run --frozen --extra gbdt python -m scripts.run_diversification_challenger \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output artifacts/reports/w2_facet_mmr_replay_v1.json
```

## Next retests

1. Retune MMR relevance weight and activation thresholds inside each combination's
   inner folds after a calibrated browsing router is integrated.
2. Retest PRF only with catalog normalization, a query-drift router, or a stronger
   candidate head; do not spend a full F2 budget on the current standalone form.
3. Run learned-sparse and late-interaction recall before any end-to-end replay only
   after exact local assets pass the manifest gates.
4. Preserve all implementations in the unified library with defaults off. Only a
   complete, matched F2 configuration may alter the champion preset.
