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
public baseline but do not by themselves prove that the threshold generalizes. Freeze it
before evaluating on any unseen set.
