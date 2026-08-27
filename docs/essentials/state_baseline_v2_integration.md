# State Baseline V2 Native Integration

Date: 2026-08-27  
Integration branch: `feat/integrate-state-baseline-v2`  
Teammate source: `origin/feat/state-baseline-v2@7b78dd4`  
Unified parent: `ghostlab/unified-techniques@48fb987`

## 1. Outcome and boundary

The teammate's retained State Baseline V2 is implemented as native GhostLab
components. It is not copied as a second `baseline/` runtime and the old branch is not
merged wholesale. This avoids duplicating the evaluator, retriever, API contract, and
session runtime while preserving the teammate implementation's observable behavior.

The integration reproduces the teammate branch exactly on the frozen 150-session
`adaptive_v1` development split: aggregate scores and canonical per-session hashes match
for the raw control, fixed-order State V2, and fixed-`other` State V2 variants.

This does **not** change the protected compiled champion or its default configuration.
All new features are opt-in research switches. The protected F3 holdout was not read.

## 2. What was integrated

### 2.1 Structured conversation state

`ghostlab/state/baseline_v2.py` implements:

- typed constraints with attribute, values, relation, polarity, strength, operator,
  source turn, source text, provenance, status, and supersession metadata;
- compatible multi-value accumulation and deterministic deduplication;
- targeted corrections that supersede the affected attribute without deleting
  unrelated evidence;
- ambiguous-correction preservation;
- no-preference preservation and later reactivation;
- an append-only audit trail of active and superseded constraints;
- a compatibility adapter around the teammate branch's frozen deterministic parser;
- a native `ConversationState.values` mirror so existing GhostLab question policies,
  filters, and rankers can consume State V2 without knowing its legacy representation.

The state is selected with:

```json
{"state_variant": "baseline_v2"}
```

### 2.2 Coverage-adaptive query construction

`ghostlab/state/query.py` exposes `coverage_adaptive_v2`. It normally emits cleaned,
active structured evidence. When a correction has superseded evidence and coverage is
low (three or fewer active constraints), it falls back to lossless raw user history.
This is the frozen teammate rule; it does not inspect scenario labels or hidden intent.

Select it with:

```json
{
  "state_variant": "baseline_v2",
  "query_variant": "coverage_adaptive_v2"
}
```

The schema rejects this query variant with another state implementation.

### 2.3 Correction-scoped recommendation history

`ghostlab/runtime/unified_experimental.py` can suppress products already returned during
the current stable intent epoch. Only actually returned valid product IDs are recorded.
An accepted correction increments `ConversationState.intent_epoch`, clears the effective
seen set, and permits products from the previous intent to appear again. Ambiguous text
that changes no state does not reset history.

Select it with:

```json
{"recommendation_history": "correction_scoped"}
```

It is applied after retrieval, reranking, and diversification and before response
truncation. Consequently it composes with all native retrieval and ranking routes rather
than being hard-coded inside BM25. The schema rejects it with states that do not observe
conversation corrections.

### 2.4 Question controls

Two exact teammate controls are available:

- `question.fixed` binds to the literal organizer sequence
  `material, color, style, use_case, feature, budget, size`;
- `question.other_always` asks `other` on every turn and remains a
  simulator-sensitive diagnostic, not a production recommendation.

The previous `question.fixed` binding accidentally used an empty sequence, which meant
"ask nothing" in the unified runtime. The binding now names the literal organizer order.

## 3. Native file and switch map

| Concern | Native implementation | Switch/binding |
|---|---|---|
| Structured State V2 | `ghostlab/state/baseline_v2.py` | `state_variant=baseline_v2`; `state.baseline_v2` |
| Intent epoch | `ghostlab/state/baseline_v2.py` | automatic inside State V2 |
| Coverage query | `ghostlab/state/query.py` | `query_variant=coverage_adaptive_v2`; `query.coverage_adaptive_v2` |
| Seen-product filtering | `ghostlab/runtime/unified_experimental.py` | `recommendation_history=correction_scoped`; `recommendation.correction_scoped_history` |
| Suite schema/factory | `ghostlab/research/technique_suite.py` | typed JSON config |
| Autonomous composition | `ghostlab/campaign/bindings.py` | four bindings above plus `question.other_always` |
| Machine catalog | `configs/techniques/catalog_v2.json` | technique metadata and dependencies |
| Campaign seed/space | `configs/campaigns/autonomous_full_v1.template.json` | baseline control and composable IDs |
| Contract tests | `tests/test_state_baseline_v2_integration.py` | state, query, history, factory tests |
| Validation runner | `scripts/validate_state_baseline_v2.py` | exact parity and paired statistics |
| Evidence | `artifacts/reports/state_baseline_v2_integration.json` | sessions, hashes, folds, intervals |

## 4. Presets

| Preset | Purpose |
|---|---|
| `configs/suites/state_baseline_v2_raw_control.json` | Lossless raw-history BM25 control, literal organizer questions, no seen filtering |
| `configs/suites/state_baseline_v2_fixed.json` | Exact retained State V2 baseline with organizer questions |
| `configs/suites/state_baseline_v2_other.json` | Exact retained teammate diagnostic with `other` every turn |
| `configs/suites/state_baseline_v2_ranked.json` | Native interaction with weighted BM25, quality prior, champion question sequence, and metadata GBDT |

Run one preset:

```bash
uv run --frozen python -m scripts.run_unified_preset \
  --config configs/suites/state_baseline_v2_fixed.json \
  --split configs/splits/adaptive_v1.json
```

Run the complete integration gate:

```bash
uv run --frozen python -m scripts.validate_state_baseline_v2
```

## 5. Validation evidence

All results below use only `adaptive_v1` (150 sessions). The 50-session F3 protected
holdout remains sealed.

| Variant | Hit@10 | MRR | MTTC | TechnicalScore | Exact teammate parity |
|---|---:|---:|---:|---:|---|
| Raw-history control | 0.793333 | 0.482206 | 4.946667 | 0.662395 | Yes, including session hash |
| State V2 + fixed order + history | 0.940000 | 0.562291 | 3.826667 | 0.782154 | Yes, including session hash |
| State V2 + `other` + history | 0.986667 | 0.600677 | 2.826667 | 0.837003 | Yes, including session hash |
| State V2 + native metadata GBDT interaction | see evidence artifact | see artifact | see artifact | 0.885391 | Native interaction, not teammate parity |

Relative to the raw-history control:

- fixed State V2 improves mean session reward by `0.119759`, has bootstrap 95%
  interval `[0.079059, 0.163503]`, and wins all five outer folds;
- fixed-`other` State V2 improves by `0.174608`, interval
  `[0.126336, 0.225652]`, and wins all five folds;
- the ranked interaction improves by `0.222996`, interval
  `[0.167410, 0.280848]`, and wins all five folds.

The ranked `0.885391` is a promising compatibility result, not a promoted champion.
`gbdt_reranker_v2_round56.json` is an existing fitted deployment asset. A fair promotion
decision after changing the state/query distribution requires the established fold-local
refit or an untouched promotion gate. Do not compare its fixed-asset adaptive replay to
the guarded candidate's `0.878963` OOF score as if the evidence classes were identical.

## 6. Overfitting safeguards

- The validation script loads only IDs in `adaptive_v1` and asserts exactly 150 rows.
- It asserts `nested_v1` exactly partitions those IDs.
- It never loads `f3_v1.json` and records `protected_holdout_accessed=false`.
- Exact teammate scores and per-session hashes are constants; integration drift fails.
- Paired deltas, bootstrap intervals, randomization p-values, and five outer-fold scores
  are reported instead of relying only on one aggregate.
- The coverage threshold remains frozen at three; it was not retuned during integration.
- The fixed-`other` result remains labeled diagnostic because the public simulator makes
  that action unusually informative.
- No new champion is promoted from this integration run.

## 7. Autonomous-system linkage

The autonomous campaign still starts from the pure keyword-only baseline. State V2 is
added as one challenger preset and as independently addressable bindings. This prevents
the search from assuming the previous champion or teammate baseline is optimal.

The following IDs are searchable and remain independently toggleable:

```text
state.baseline_v2
query.coverage_adaptive_v2
recommendation.correction_scoped_history
question.other_always
```

Compatibility validation rejects nonsensical candidates (for example, the coverage
query without State V2). Valid candidates can combine State V2 with retrieval, ranking,
question, routing, and diversification techniques. Standard F0/F1/F2 budget pruning and
human promotion gates remain unchanged.

## 8. What was deliberately not imported

The teammate branch documents an exact catalog-grounding experiment that regressed on
both fixed policies and was reverted before `7b78dd4`. Its implementation is therefore
not present in the source branch to integrate. The failure and retest rationale remain in
the teammate history; it must not be represented as retained State V2 behavior.

Likewise, State V2 does not claim to add adaptive/EIG questions, dense retrieval,
cross-encoders, or learned interpretation. Those already exist as independent unified
techniques and can now consume this state through the native adapter.

## 9. Verification checklist

```bash
uv run --frozen pytest -q tests/test_state_baseline_v2_integration.py
uv run --frozen python -m scripts.validate_state_baseline_v2
uv run --frozen ruff check ghostlab scripts tests
uv run --frozen mypy ghostlab
uv run --frozen pytest -q
```

Pass criteria:

1. all three teammate scores and session hashes match exactly;
2. all five folds improve for both retained teammate configurations over the raw control;
3. existing compiled champion tests remain unchanged;
4. F3 remains unread;
5. no technique becomes a default merely because it is integrated.
