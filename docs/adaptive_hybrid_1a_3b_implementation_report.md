# Adaptive Hybrid 1A-3B Implementation Report

## GhostLab optimizer integration update

The fixed runtime is now connected to an exhaustive adaptive technique registry
and a dedicated champion/challenger engine. The engine plans every valid
standalone and compatible pair before higher-order beam expansion, performs
conditional local BOHB tuning for surviving structures, and races candidates
through F0/F1/F2 against the matched incumbent. Required 1A-3B capabilities are
validated before evaluation and cannot be toggled by a trial.

Plan-only validation accounts for all 75 catalog records: 11 compulsory, 12
promotable, 25 control-only, 16 research-only and 11 unavailable. It produces 62
architecture-valid initial control/single/pair candidates under the current
catalog and records incompatible structures as skipped evidence. These counts are
inventory results, not fixed design limits.

No training or full campaign was started during this integration. Fit-required
challengers are explicitly prevented from promotion until their fold-safe fit
evidence is verified.

## Decision

**The 1A-3B architecture is complete and operational; score optimization remains.**

The fixed Track 4 workflow runs through the official `reset(...)` and
`respond(...)` contract. The protected F3 holdout remains sealed, every operational
check passes, and the adaptive preset is not officially activated because its
public TechnicalScore remains below the guarded champion.

“Promoted” below means selected inside the adaptive implementation. It does not
mean `configs/active_candidate.json` was created.

## Compulsory architecture versus GhostLab adaptation

| Area | Compulsory static contract GhostLab cannot remove | What GhostLab may adapt | Promoted implementation | Rejected or retained alternatives |
|---|---|---|---|---|
| 1A Routing | Both Buying and Browsing must be reachable from observable features. Buying is precision-primary; Browsing is diverse-dense-primary. | Specificity features, threshold and abstention confidence. | Conservative deterministic router with failure/low-confidence precision abstention. | A router using scenario labels or one route only is invalid, not merely low-scoring. |
| 1B Retrieval | Every normal turn constructs one bounded in-memory keyword + independent-category + vector pool with provenance. Buying remains BM25-primary; Browsing remains E5-primary. | Source depths, supporting budgets, primary shares, query views and category expansion. | Buying weights `0.90/0.05/0.05`; Browsing `0.10/0.10/0.80`; complete-pool union GBDT. | Dense-only Browsing end to end, sparse-only Buying end to end, or appended unranked candidates are architecture-invalid. MMR remains unpromoted. |
| 1B Semantic ranking | A literal local-LLM capability and mandatory semantic activation decision must remain after union ranking. Output is bounded to supplied IDs. | Activation gate, LLM model/prompt, Top-K, score weight, timeout and fallback. | Qwen2.5-0.5B-Instruct on Browsing only; deterministic pass-through for Buying. | Always-on Qwen `0.830356`; broad semantic-constraint gate `0.832516`; non-overloaded-Browsing gate activated zero times; no-LLM control `0.843037` is ineligible. |
| 2A State | State V2 accumulation, correction, intent override/epochs, exclusions, provenance and atomic selected-action history. | Parser confidence and override scope, without weakening the semantics. | State V2 with strict negative handling and intent-scoped shown history. | Restoring filtered or already-shown products merely to fill ten slots was fixed and rejected. |
| 2B Guidance | Detect over-generality, cap retrieval, return recommendations, and choose the highest-value legal unresolved question. | Overload threshold, question-value margin and discovery-turn policy. | Bounded preview plus recommend-and-ask; already collected three-source pool still reaches union and semantic decision. | Ask-only deferral and arbitrary legal-question selection remain rejected/unpromoted. |
| 3A Runtime adaptation | Distil session/profile context, suppress conflicts, keep explicit session intent dominant, and expose a profile update with confidence/provenance. | Profile confidence, ambiguity gate and ranking influence. | Session-scoped `ProfileUpdate` with values, confidence, provenance, epoch and conflicts; weight `0.02`. | Cross-session evaluator memory and profile precedence over explicit intent are forbidden. |
| 3B Orchestration | Preserve the canonical order, valid failure boundaries, reason codes, response normalization and atomic commit. | Bounded budgets and activation thresholds inside the fixed workflow. | One coordinator with complete precision fallback and per-turn strategy traces. | Removing/reordering required slots or fail-open unfiltered output is architecture-invalid. |
| Optional downstream | Nothing beyond 1A-3B is compulsory. | Evidence-gated residual reorder or other additions. | None. | Residual Top-10 reorder, quotas and ask-only deferral remain unpromoted. |

The `AdaptiveArchitectureAudit` runs before a GhostLab trial can be scored. It
requires the fixed component sequence, three active retrieval sources, literal
local LLM capability, conflict-safe profile influence and atomic commit.

## Semantic activation study

All gates used the same 200 public sessions and the same retrieval/ranking stack.
Research-only ablations remain submission-ineligible.

| LLM activation policy | Activations / 406 turns | MRR | TechnicalScore | Decision |
|---|---:|---:|---:|---|
| Never invoke Qwen | 0 | 0.571458 | 0.843037 | Control only; rejects the required active capability |
| Invoke Qwen on every turn | 406 | 0.529187 | 0.830356 | Rejected; damages deterministic Buying |
| Broad semantic-constraint gate | 239 | 0.536387 | 0.832516 | Rejected; still over-activates Buying/override |
| Browsing only after overload resolves | 0 | 0.571458 | 0.843037 | Rejected; decorative on this dataset |
| **Browsing route only** | **96** | **0.578286** | **0.845086** | **Promoted inside adaptive runtime** |

The result supports the architectural interpretation: Browsing is dense-primary
because it needs scenario/cross-category semantics, while deterministic Buying
usually benefits more from protected hard constraints and precision ranking.

The promoted policy activates Qwen on 96 Browsing turns, skips it on 310 Buying
turns, changes 91 Browsing orderings and produces zero full-run fallbacks.

## Literal local-LLM implementation

The primary model is `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0. The pinned local directory
hash is
`31b07963d699962dbbc9fdcb9d4cfaa496f5e56abc29c8e597e18195b87ebe77`;
the receipt is `artifacts/models/qwen2.5_0.5b_instruct.receipt.json`.

The model receives the distilled request and bounded catalogue passage for each
supplied ID. Its next-token `yes`/`no` logit difference produces a deterministic
relevance score. It can reorder only the supplied Top-10 and cannot generate a
catalogue ID. Runtime is local-only, uses Apple MPS when available and supports
CPU. MiniLM is accurately retained only as an invalid-score/model fallback.

## Cross-category Browsing proof

The category-blind validator uses no target IDs or official scenario labels. The
dense Top-200 recovered all declared families:

| Prompt | Recovered family counts |
|---|---|
| Warm-weather wedding | clothing 108, footwear 1, accessory 1 |
| Sunny beach holiday | swimwear 76, footwear 9, sun accessory 22 |
| Rainy road running | running 33, outerwear 41, accessory 11 |

This proves candidate reach, not official relevance quality.

## Selected public result

| Agent | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Guarded champion | 0.980000 | 0.774839 | 2.280000 | 0.896852 |
| State V2 precision control | 0.990000 | 0.746895 | 2.185000 | 0.895369 |
| Adaptive Hybrid, selective Qwen | 0.985000 | 0.578286 | 2.045000 | 0.845086 |

Selected scenario metrics:

| Scenario | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 1.000000 | 0.582063 | 2.500000 |
| Browsing | 1.000000 | 0.477847 | 1.625000 |
| Buying | 0.962500 | 0.604365 | 1.725000 |
| Intent override | 1.000000 | 0.775317 | 3.866667 |

The candidate improves Hit@10 and MTTC relative to the champion but remains behind
on MRR. It is the architecture-complete optimization base, not the active winner.

## Validation status

- All focused 1A-3B schema, behavioral, interaction and failure checks pass.
- Qwen has both observed activations and justified deterministic skips.
- Keyword, category and vector evidence contribute on both route implementations.
- Cross-category dense reach passes all three independent templates.
- Deterministic replay, offline loading, model hashes and fit receipts pass.
- Factory/config and hash-bound official-entrypoint construction pass.
- Response normalization, strict filtering and atomic action history pass.
- Full public run: 406 turns, 96 Qwen activations, 310 skips, zero fallbacks.
- F3 was not accessed and the active-candidate pointer remains absent.

## Next optimization work

Keep the architecture and promoted activation gate fixed for the next experiment.
Retrain/recalibrate the union ranker on the exact three-source pools and select by
complete-session MRR. Then tune Qwen weight/depth within Browsing only. Do not
re-enable Qwen on deterministic Buying unless a future isolated study reverses the
current evidence.

Machine-readable evidence:

- `artifacts/reports/adaptive_hybrid_qwen_selective_v3.json`
- `artifacts/reports/adaptive_hybrid_validation_v2.json`
- `artifacts/reports/semantic_activation_never_v1.json`
- `artifacts/reports/semantic_activation_always_v1.json`
- `artifacts/reports/semantic_activation_browsing_all_v1.json`
- `artifacts/reports/semantic_activation_browsing_refined_v1.json`
- `artifacts/reports/adaptive_hybrid_campaign_selective_smoke_v3.json`
- `artifacts/reports/cross_category_browsing_v1.json`
