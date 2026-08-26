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

The state-only candidate passed: neither policy regressed and current order improved by
`0.005007`, meeting the `0.005` TechnicalScore gate. Under current order it also produced
five earlier hits and four better target ranks with no paired losses. Under fixed `other`
it produced 16 miss-to-hit conversions and no hit-to-miss conversions.

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
