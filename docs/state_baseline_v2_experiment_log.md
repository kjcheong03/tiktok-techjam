# State Baseline V2 Experiment Log

## Evaluation input

- Catalog: organizer `participant-kit` release, 50,000 rows
- `catalog.jsonl.gz` SHA-256: `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`
- Dataset: `data/public_set.jsonl`, 200 sessions
- Retrieval: unchanged organizer BM25

## State and interpreter candidates

| Cumulative variant | Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V1 | current order | 0.595 | 0.377867 | 6.375 | 0.503360 | — |
| V2 state only | current order | 0.595 | 0.391891 | 6.335 | 0.508367 | +0.005007 |
| V2 interpreted | current order | 0.570 | 0.317313 | 6.675 | 0.466694 | -0.041673 |
| V1 | fixed `other` | 0.630 | 0.404040 | 5.545 | 0.545312 | — |
| V2 state only | fixed `other` | 0.710 | 0.465732 | 4.930 | 0.616120 | +0.070808 |
| V2 interpreted | fixed `other` | 0.710 | 0.423925 | 4.920 | 0.603777 | -0.012343 |

The state-only candidate passed within the original parsed-query family: neither policy
regressed and current order improved by `0.005007`, meeting the `0.005` TechnicalScore
gate. Under current order it also produced five earlier hits and four better target ranks
with no paired losses. Under fixed `other` it produced 16 miss-to-hit conversions and no
hit-to-miss conversions. The later raw-history factorial control superseded this as
evidence for final component retention.

The deterministic interpreter candidate failed and was reverted. Under current order it
caused 16 hit-to-miss and 11 miss-to-hit conversions; under fixed `other` it caused 16 of
each. Both TechnicalScores regressed, so scenario-specific improvements could not satisfy
the no-regression requirement.

## Raw-history fixed-`other` reference probe

The user-reported probe combines two changes: fixed `other` question selection and raw
conversation-history retrieval. Reproducing that exact joint control scored:

- Hit Rate@10: `0.875`
- MRR: `0.540002`
- MTTC: `3.455`
- Efficiency: `0.754500`
- TechnicalScore: `0.750401`

This is a reference probe rather than an isolated state delta. The per-variant fixed
`other` results above deliberately keep each variant's own state representation.

## Raw-history query candidate

This candidate retained V2 structured state for transitions and question selection, but
changed only BM25 query compilation from active parsed constraints to exact accumulated
user messages.

| Cumulative variant | Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V2 state only | current order | 0.595 | 0.391891 | 6.335 | 0.508367 | — |
| V2 + raw-history query | current order | 0.695 | 0.432111 | 5.645 | 0.584233 | +0.075866 |
| V2 state only | fixed `other` | 0.710 | 0.465732 | 4.930 | 0.616120 | — |
| V2 + raw-history query | fixed `other` | 0.875 | 0.540002 | 3.455 | 0.750401 | +0.134281 |

The raw-history query change passed relative to V2's lossy parsed query under both
policies. Current order produced 28 miss-to-hit and 8 hit-to-miss conversions; fixed
`other` produced 34 miss-to-hit and one hit-to-miss conversion. This established the
value of raw history, but did not establish that managed state contributed.

## State/no-state factorial control

The final control held raw-history retrieval constant and explicitly separated the literal
fixed turn order from the state-aware order.

| Raw-history implementation | Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore | State delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No managed state | literal fixed turn order | 0.800 | 0.517585 | 4.795 | 0.679376 | — |
| V2 state stored | literal fixed turn order | 0.800 | 0.517585 | 4.795 | 0.679376 | 0.000000 |
| No managed state | fixed `other` | 0.875 | 0.540002 | 3.455 | 0.750401 | — |
| V2 state stored | fixed `other` | 0.875 | 0.540002 | 3.455 | 0.750401 | 0.000000 |
| V2 state drives skipping | state-aware order | 0.695 | 0.432111 | 5.645 | 0.584233 | -0.095143 |

Under both policies that ignore managed state, all 200 paired sessions were identical.
The current state-aware order caused 21 hit-to-miss conversions and no miss-to-hit
conversions relative to raw history with the literal order. Consequently:

- raw history is retained as the proven retrieval improvement;
- fixed `other` remains a diagnostic rather than the product policy;
- the literal fixed turn order is the strongest non-diagnostic control at `0.679376`; and
- structured state remains experimental until a state consumer beats that control.

## State-consumed raw-history candidate

This candidate kept raw-history lexical evidence but consumed structured state during
query compilation. It removed terms belonging to superseded, excluded, or no-preference
constraints and retained or appended active positive terms. Question policy and BM25 were
held fixed.

| Variant | Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | --- | ---: | ---: | ---: | ---: |
| V2 state only, no raw history | literal fixed turn order | 0.725 | 0.481913 | 5.310 | 0.620874 |
| Raw history, no managed state | literal fixed turn order | 0.800 | 0.517585 | 4.795 | 0.679376 |
| State-consumed raw history | literal fixed turn order | 0.715 | 0.463591 | 5.390 | 0.608777 |
| V2 state only, no raw history | fixed `other` | 0.710 | 0.465732 | 4.930 | 0.616120 |
| Raw history, no managed state | fixed `other` | 0.875 | 0.540002 | 3.455 | 0.750401 |
| State-consumed raw history | fixed `other` | 0.715 | 0.450607 | 4.885 | 0.614982 |

The candidate failed the contribution gate under both policies. Relative to raw history
without managed state, it regressed by `0.070599` under literal order, with 3 miss-to-hit
and 20 hit-to-miss conversions. Under fixed `other`, it regressed by `0.135419`, with no
miss-to-hit and 32 hit-to-miss conversions. It also failed to beat the previous state-only
query under either policy.

This proves that state was consumed, but the tested term-removal strategy discarded
lexical evidence that OR-style BM25 was using successfully. The candidate was reverted.

## State-query root-cause investigation

Session traces showed that the weak state-only scores were partly implementation defects,
not evidence that structured state was intrinsically unhelpful:

- the V1 adapter reused single-slot replacement semantics inside one composite answer, so
  compatible evidence such as `96% Nylon, 4% Spandex` lost all but the final value; and
- a no-preference answer for an attribute hid useful positive evidence already collected
  for that attribute.

The state reducer now preserves all values extracted from one answer and applies targeted
supersession across messages. No-preference state controls future questioning without
deleting earlier positive query evidence.

| Variant | Policy | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw history, no state | literal fixed turn order | 0.800 | 0.517585 | 4.795 | 0.6205 | 0.679376 |
| Corrected state only | literal fixed turn order | 0.795 | 0.527571 | 4.945 | 0.6055 | 0.676871 |
| Raw history, no state | fixed `other` | 0.875 | 0.540002 | 3.455 | 0.7545 | 0.750401 |
| Corrected state only | fixed `other` | 0.870 | 0.559954 | 3.480 | 0.7520 | 0.753386 |

Relative to the original state-only implementation, the corrected state produced 15
miss-to-hit and one hit-to-miss conversions under literal order, and 32 miss-to-hit with
no hit-to-miss conversions under fixed `other`. State-only was therefore close to raw
history, but its remaining weakness was concentrated in low-coverage correction sessions.

## Retained coverage-adaptive state query

A state prefix followed by the complete raw history was tested first and produced exactly
the raw-history result in all 200 sessions under both policies. The active terms were
already present in the raw query, so duplication did not change the BM25 outcome.

The retained query consumer uses cleaned active state by default. If a correction has
superseded evidence and three or fewer active constraints remain, it uses exact raw history
because the parser does not yet cover enough of the user's intent safely. This is a
session-state confidence rule, not question-policy or evaluator-scenario branching.

| Variant | Policy | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore | Delta vs raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw history, no state | literal fixed turn order | 0.800 | 0.517585 | 4.795 | 0.6205 | 0.679376 | — |
| Corrected state only | literal fixed turn order | 0.795 | 0.527571 | 4.945 | 0.6055 | 0.676871 | -0.002505 |
| Coverage-adaptive state | literal fixed turn order | 0.820 | 0.549690 | 4.705 | 0.6295 | **0.700807** | **+0.021431** |
| Raw history, no state | fixed `other` | 0.875 | 0.540002 | 3.455 | 0.7545 | 0.750401 | — |
| Corrected state only | fixed `other` | 0.870 | 0.559954 | 3.480 | 0.7520 | 0.753386 | +0.002985 |
| Coverage-adaptive state | fixed `other` | 0.870 | 0.559954 | 3.480 | 0.7520 | **0.753386** | **+0.002985** |

Under literal order, adaptive state versus corrected state-only had five miss-to-hit, zero
hit-to-miss, five earlier-hit, and zero later-hit conversions. Versus raw history it had
14 miss-to-hit and 10 hit-to-miss conversions, a net gain of four hits. Under fixed
`other`, adaptive state intentionally matched corrected state-only in all 200 sessions;
versus raw it had three miss-to-hit and four hit-to-miss conversions but improved MRR
enough to raise the aggregate score.

This candidate passes the contribution gate: it does not regress either fixed policy,
improves literal-order TechnicalScore by `0.021431`, and beats the raw-history/no-state
control under both policies. The state preservation fixes and coverage-adaptive query are
retained; the naive term-removal query remains rejected. Because the coverage threshold
was selected after diagnosing the same public set, these numbers establish the retained
public baseline. The threshold is frozen; further public-set work should use paired
ablations and resampling rather than choosing another cutoff from the aggregate result.

## Retained recommendation history

The first history candidate filtered every product previously shown in the session. It
improved Buying and Browsing substantially but failed overall because Intent Override
sessions do not count a target shown before the correction. All 25 literal-order and all
26 fixed-`other` hit-to-miss conversions were Intent Override sessions.

The retained design scopes history to an intent epoch. It filters and fills from unseen
candidates while intent is stable, then clears history when state accepts an explicit
correction. Ambiguous corrections that change no state do not clear it.

### Cumulative score breakdown

| Variant / policy | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Raw history — literal fixed turn order | 0.800 | 0.517585 | 4.795 | 0.679376 |
| Raw history — fixed `other` | 0.875 | 0.540002 | 3.455 | 0.750401 |
| Corrected state — literal fixed turn order | 0.795 | 0.527571 | 4.945 | 0.676871 |
| Corrected state — fixed `other` | 0.870 | 0.559954 | 3.480 | 0.753386 |
| Coverage-adaptive state — literal fixed turn order | 0.820 | 0.549690 | 4.705 | 0.700807 |
| Coverage-adaptive state — fixed `other` | 0.870 | 0.559954 | 3.480 | 0.753386 |
| + recommendation history — literal fixed turn order | 0.955 | 0.587571 | 3.625 | 0.801271 |
| + recommendation history — fixed `other` | 0.990 | 0.610714 | 2.715 | 0.843914 |

| Variant | Policy | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coverage-adaptive state | literal fixed turn order | 0.820 | 0.549690 | 4.705 | 0.6295 | 0.700807 | — |
| + correction-scoped history | literal fixed turn order | **0.955** | **0.587571** | **3.625** | **0.7375** | **0.801271** | **+0.100464** |
| Coverage-adaptive state | fixed `other` | 0.870 | 0.559954 | 3.480 | 0.7520 | 0.753386 | — |
| + correction-scoped history | fixed `other` | **0.990** | **0.610714** | **2.715** | **0.8285** | **0.843914** | **+0.090528** |

Under literal order, history produced 27 miss-to-hit and zero hit-to-miss conversions,
with 27 earlier and zero later hits. Under fixed `other`, it produced 24 miss-to-hit and
zero hit-to-miss conversions, with eight earlier and zero later hits. It passes the
component gate under both policies and is retained.

## Deterministic transition audit

The transition suite distinguishes semantic correctness from isolated retrieval
contribution:

| Transition | Transcript/unit evidence | End-to-end contribution evidence |
| --- | --- | --- |
| Compatible-value accumulation and deduplication | Pass | Included in the corrected-state bundle; not individually ablated |
| Targeted and ambiguous correction handling | Pass | Included in state and Intent Override behavior; not individually ablated |
| No-preference preservation and later explicit reactivation | Pass through the real message adapter | No-preference query preservation was bundled with the corrected-state fix; reactivation is not individually ablated |
| Intent override and explicit category replacement | Pass through real correction phrases | Intent Override scenario is covered, but transition logic is not separately ablated |
| Recommendation history and correction-scoped epochs | Pass at state and agent boundaries | Isolated: `+0.100464` literal order and `+0.090528` fixed `other` |

An epoch is the current stable-intent period, not a separate model or persisted object.
The state stores products shown during that period. An accepted explicit correction clears
the set and begins a new period; an ambiguous correction that changes no state preserves
it. The full repository suite contains 50 passing tests after adding transcript coverage
for preference reactivation and explicit category replacement.

### Disable-one transition ablations

Each ablation retains coverage-adaptive querying, correction-scoped recommendation
history, unchanged BM25, and the same question policy. The delta is `full retained -
ablated`, so a positive value favors the transition.

| Disabled transition | Literal ablated | Literal delta | Fixed-`other` ablated | Fixed-`other` delta | Strict gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Compatible-value accumulation | 0.754617 | **+0.046654** | 0.813179 | **+0.030735** | Pass |
| Targeted correction | 0.798357 | +0.002914 | 0.844914 | -0.001000 | Fail: small fixed-`other` regression |
| Ambiguous-correction preservation | 0.801271 | 0.000000 | 0.843914 | 0.000000 | No public-session coverage |
| No-preference evidence preservation | 0.802821 | -0.001550 | 0.792844 | **+0.051070** | Fail: small literal regression |
| **Full retained state** | **0.801271** | — | **0.843914** | — | — |

Compatible-value accumulation independently passes. Targeted correction adds one net hit
under literal order and improves its Intent Override Hit Rate from `0.866667` to `0.900`,
but loses `0.001000` TechnicalScore under fixed `other`. Ambiguous-correction preservation
changes no public session because the evaluator emits no matching ambiguous correction.
No-preference preservation adds 15 net hits under fixed `other`, but loses `0.001550`
under literal order through small rank tradeoffs. Therefore the ablation does not establish
that all four satisfy the existing no-regression gate; the latter three remain explicit
semantic-versus-score retention decisions rather than claimed isolated wins.
The repository contains 57 passing tests including the seven ablation-isolation tests.

#### Provisional retention decision and use cases

All four transitions remain in the baseline for now. The mixed results remain visible and
are not reclassified as strict-gate passes.

| Transition | Customer use case | Evidence and retention reason |
| --- | --- | --- |
| Compatible-value accumulation | The customer accepts alternatives or supplies multiple requirements in one answer, such as `nylon; spandex` or `imported; wrap closure`. | Clear isolated win: `+0.046654` literal and `+0.030735` fixed `other`; retain. |
| Targeted correction | The customer changes one attribute, such as black to navy, while material, budget, and category should remain valid. | Mixed score: `+0.002914` literal and `-0.001000` fixed `other`; adds one net literal-order hit and improves Intent Override Hit Rate from `0.866667` to `0.900`; retain for correction correctness. |
| Ambiguous-correction preservation | The customer says something vague such as `something different`; uncertain text must not erase known constraints. | Exact `0.000000` delta because no public session exercises this case; retain as a small guard against destructive state loss. |
| No-preference preservation and reactivation | The customer has no additional preference for an attribute, but earlier explicit evidence remains valid; a later concrete preference reactivates that attribute. | Policy-dependent: `-0.001550` literal and `+0.051070` fixed `other`, including 15 net fixed-`other` hits; retain because the strong gain and semantics outweigh the small literal tradeoff. |

This is a provisional architecture decision, not proof that every transition independently
improves every fixed policy. Later changes must continue reporting the two policies
separately so these interactions remain observable.

## Deferred exact catalog normalization and grounding

This isolated candidate built a read-only, in-memory vocabulary from all 50,000
catalog rows. It contained `19,855` distinct non-empty participant-visible `store`
values and `863` `categories` values. Keys used only case/whitespace normalization.
Grounding reassigned a constraint to `brand` or `category` only when its normalized
value identified one raw catalog value in one field; normalized collisions, cross-field
collisions, and unmatched values retained their prior interpretation. Source text and
constraint values were never rewritten. No catalog or derived index was persisted.

The minimal consumer applied this grounding before the retained state reducer; BM25,
fixed question policies, coverage-adaptive query selection, and recommendation-history
epochs were unchanged. The candidate failed both required comparisons and was removed,
including its runtime vocabulary and tests, rather than retained as unused complexity.
The complete session-level evidence is `artifacts/state_catalog_grounding_results.json`.

| Variant | Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore | Delta vs retained |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full retained | literal fixed turn order | 0.955 | 0.587571 | 3.625 | 0.801271 | — |
| + exact catalog grounding | literal fixed turn order | 0.950 | 0.585905 | 3.660 | 0.797571 | -0.003700 |
| Full retained | fixed `other` | 0.990 | 0.610714 | 2.715 | 0.843914 | — |
| + exact catalog grounding | fixed `other` | 0.985 | 0.606048 | 2.775 | 0.838814 | -0.005100 |

The regression was entirely in Intent Override; Boundary, Browsing, and Buying scenario
metrics were identical under each policy.

| Policy | Intent Override: retained (Hit/MRR/MTTC) | Grounded (Hit/MRR/MTTC) |
| --- | --- | --- |
| literal fixed turn order | 0.900000 / 0.712540 / 4.633333 | 0.866667 / 0.701429 / 4.866667 |
| fixed `other` | 0.966667 / 0.701243 / 4.200000 | 0.933333 / 0.670132 / 4.600000 |

Paired outcomes were also unfavorable. Under literal order there were zero miss-to-hit
and one hit-to-miss conversion (`public_0103`). Under fixed `other` there were zero
miss-to-hit and one hit-to-miss (`public_0103`), zero earlier and two later hits
(`public_0064`, `public_0080`), one better rank (`public_0177`), and two worse ranks
(`public_0064`, `public_0080`).
