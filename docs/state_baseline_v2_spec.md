# State Baseline V2 Specification

## Objective

Strengthen the deterministic conversational state baseline before evaluating adaptive
question selection, advanced retrieval, or LLM components.

State Baseline V2 must understand and retain customer constraints more faithfully while
keeping the organizer's SQLite FTS5/BM25 retrieval implementation unchanged. It must also
make the effect of state changes measurable independently from question-policy changes.

## Evaluation Status

The original specification below describes candidate components, not assumed wins. The
factorial evaluation established the following retained control:

- lossless raw user history is the BM25 query;
- the literal fixed turn order is the primary non-diagnostic question control; and
- fixed `other` remains a simulator-sensitive diagnostic.

Structured state is not yet a retained performance component. With raw-history retrieval,
storing V2 state while holding the literal fixed order or fixed `other` constant changed
zero session outcomes. Letting the current state-aware order skip known or rejected
attributes reduced TechnicalScore from `0.679376` to `0.584233` and caused 21 hit-to-miss
conversions with no miss-to-hit conversions. The deterministic interpreter was also
rejected after regressing both policies. A state-consumed raw-history query was then
tested under identical fixed policies and rejected: it scored `0.608777` versus the
`0.679376` raw-history control under literal order, and `0.614982` versus `0.750401`
under fixed `other`.

Future state work must therefore beat `raw history + no managed state` under the same
question policy. A comparison only against the lossy parsed-query baseline is insufficient.

## Current Baseline

The current stateful baseline stores:

- every customer message;
- a list of typed `SlotValue` records containing attribute, raw value, source turn,
  source text, provenance, and active status;
- previously asked attributes;
- attributes for which the customer reported no preference; and
- the last asked attribute.

It builds a BM25 query by concatenating active slot values and chooses questions in the
fixed order `material`, `color`, `style`, `use_case`, `feature`, `budget`, then `size`,
skipping attributes already known, asked, or rejected.

The current implementation has these limitations:

- extraction depends on phrases emitted by the public simulator;
- a new value deactivates the previous active value for the same attribute, so alternatives
  such as "black or navy" cannot be retained;
- polarity, requirement strength, conjunctions, and numeric operators are not represented;
- an intent-override phrase deactivates every non-category slot instead of only the
  corrected attribute;
- returned values can be assigned to the last asked attribute without independently
  interpreting their contents; and
- products recommended on previous turns are not recorded.

## Scope

State Baseline V2 candidate experiments include:

- a shared attribute schema and normalization vocabulary;
- deterministic extraction of structured constraints from customer messages;
- multi-value, negative, hard, soft, numeric, and superseded constraint state;
- targeted correction and no-preference handling;
- lossless raw-history BM25 queries, with structured state evaluated as a separate sidecar;
- per-session recommendation history and unseen-product filtering;
- the unchanged keyword retriever; and
- controlled evaluation under the state-aware order, literal fixed turn order, and
  fixed-`other` probe.

State Baseline V2 does not include:

- adaptive question selection or information-gain policies;
- candidate-belief models;
- structured facet filtering, boosting, or reranking;
- changes to the BM25 implementation or its field weights;
- dense-model, cross-encoder, or LLM changes;
- learned classifiers, contextual bandits, or reinforcement learning; or
- personalization using `user_profile`.

## Frozen Catalog Boundary

`data/catalog.jsonl` remains immutable. State Baseline V2 must not rewrite product
records, introduce identifiers, enrich products with external data, or use public or
private ground-truth labels to derive product attributes.

A deterministic, read-only normalization vocabulary may be built in memory from
participant-visible catalog fields. This is analogous to building the existing FTS5 index,
not modifying the catalog. Its only V2 purpose is to ground customer-language
normalization, not to change product ranking.

The vocabulary may contain:

- exact brands from `store`;
- normalized category terms from `categories`;
- conservative aliases for colors, materials, sizes, and common feature terms found in
  `title`, `features`, `details`, and `description`; and
- numeric parsing rules for `price` and customer budget expressions.

An unrecognized or ambiguous value remains unknown. The normalizer must not invent a
value. V2 does not persist a second catalog, build per-product facet filters or postings,
or create derived ranking features.

## Constraint Contract

The message interpreter emits one or more constraint groups. A group contains:

```text
attribute       one allowed agent attribute
values          one or more normalized values
relation        any | all
polarity        include | exclude
strength        hard | soft | unspecified
operator        equals | at_most | at_least | none
source_turn     turn on which the evidence appeared
source_text     unchanged customer message
provenance      explicit | simulator_answer | inferred
status          active | superseded
```

Examples:

```text
"Black or navy is fine"
  attribute=color, values=[black, navy], relation=any, polarity=include

"Waterproof and lightweight"
  attribute=feature, values=[waterproof, lightweight], relation=all, polarity=include

"Not leather"
  attribute=material, values=[leather], polarity=exclude

"It must be under $80"
  attribute=budget, values=[80], operator=at_most, strength=hard
```

The raw text is always preserved. Normalization must never remove the evidence needed to
re-interpret a message later.

## Conversation Interpretation

The deterministic interpreter must:

- recognize the existing public-simulator templates;
- recognize equivalent paraphrases without requiring one exact prefix;
- extract multiple constraints from one message;
- distinguish `and` from `or` when explicitly stated;
- detect common negation forms such as "not", "avoid", and "anything but";
- detect hard language such as "must", "required", and explicit maximum budgets;
- detect soft language such as "prefer" and "would be nice";
- classify and normalize returned values independently of the last requested attribute;
- retain the last requested attribute only as fallback evidence when the value cannot be
  classified independently; and
- return no structured constraint when the evidence is ambiguous rather than guessing.

This interpreter is the baseline constraint classifier. It remains deterministic and
catalog-grounded; an embedding or LLM implementation can replace it later through the
same contract.

## State Transitions

### Accumulation

Multiple compatible values may remain active for the same attribute. Repeating the same
normalized evidence must not create a duplicate.

### Targeted correction

A correction supersedes active constraints only for the corrected attribute. For example,
after "black leather shoes under $80", the message "actually, navy instead" supersedes
black, activates navy, and preserves leather and the budget.

If a correction cannot be assigned confidently to an attribute, V2 preserves the current
state and raw message instead of clearing all non-category constraints.

### No preference

When the customer reports no preference for an attribute, that attribute is marked
unavailable for further questions. Existing hard constraints are not silently removed.
An explicit later preference reactivates the attribute.

### Intent override

Intent-override wording is treated as correction evidence, not as permission to erase all
non-category state. Only attributes identified in the replacement content are superseded.
Category is replaced only when the customer explicitly supplies a new category.

### Recommendation history

State records each catalog-valid `parent_asin` returned to the customer. Products already
shown in the same session are filtered from later recommendations while unseen retrieved
candidates remain. `reset()` creates an empty history for the new session.

## BM25 Query Compilation

The initial active-state query compiler was useful as an ablation but was not retained as
the primary BM25 query. It discarded lexical evidence that the unchanged OR-style BM25
retriever uses effectively. The retained query is the exact accumulated user-message
history in turn order.

Structured state remains a sidecar representation. If a later retrieval experiment uses
it, the compiler must preserve the raw evidence and separately demonstrate the value of:

- placing category terms first;
- ordering active positive values by source turn;
- emitting duplicate normalized terms once;
- omitting superseded and no-preference values;
- retaining negative constraints in state while omitting them from the positive BM25 query,
  because the unchanged organizer retriever does not implement structured exclusion; and
- retaining hard and soft strength without altering BM25 field weights.

These restrictions keep retrieval unchanged and prevent a phrase such as "not leather"
from becoming a positive search for leather. Structured exclusion and preference weighting
belong to a later retrieval experiment.

## Question Policies Used for Evaluation

Question policy is held separate from state behavior.

The state-aware `current_order` condition uses the existing order:

```text
material -> color -> style -> use_case -> feature -> budget -> size
```

It skips attributes that state reports as known, previously asked, or rejected. Evaluation
showed that this behavior currently regresses the raw-history control, so it is an
experimental state policy rather than the primary control.

The `fixed_turn_order` condition asks the same literal sequence by turn without reading
managed state. This is the primary non-diagnostic control for state comparisons.

The fixed-`other` condition returns `ask_attribute="other"` on every turn. It is a
simulator-sensitive diagnostic probe, not the product policy or an adaptive controller.
The user-reported raw-history fixed-`other` TechnicalScore of approximately `0.76` must be
reproduced and recorded before it is used as a non-regression benchmark. The exact
reproduced value is `0.750401`.

No new question order is designed in this work. Adaptive question selection is evaluated
separately after State Baseline V2.

## Implementation Units

1. Preserve V1 in the evaluation harness so V1 and V2 can run through the same retriever,
   evaluator, and question-policy implementations. V1 does not need to remain a production
   runtime option.
2. Add the shared constraint types, attribute normalization, and in-memory read-only
   catalog vocabulary.
3. Add the deterministic V2 message interpreter and state-transition implementation.
4. Add deterministic state-to-query compilation and recommendation-history filtering.
5. Add explicit current-order and fixed-`other` question-policy options without adding an
   adaptive policy.
6. Extend the baseline runner to emit the complete V1/V2 comparison and component
   ablations.

Likely implementation locations are:

```text
baseline/constraints.py          new contract, interpreter, and normalization
baseline/state_v2.py             new state transitions and query compilation
baseline/question_policy.py      new fixed evaluation policies
baseline/agent.py                state/policy selection and seen-result filtering
scripts/run_baselines.py         comparison variants and reporting
tests/test_state_v2.py           focused state and interpreter tests
tests/test_baseline.py           agent integration and compatibility tests
```

The exact file split may be reduced during implementation if the types remain independently
testable and V1 behavior stays executable.

## Testing Strategy

### Unit tests

Cover:

- public-template and paraphrased category extraction;
- multiple values joined by `and` and `or`;
- positive and negative constraints;
- hard and soft preferences;
- numeric budget operators;
- repeated evidence without duplication;
- targeted correction while unrelated constraints remain active;
- ambiguous correction without destructive state clearing;
- no-preference followed by a later explicit preference;
- exact intent-override templates;
- deterministic query ordering;
- omission of negative and superseded values from the BM25 query; and
- recommendation history reset and unseen-result filtering.

### Transcript replay

Run identical recorded message sequences through V1 and V2 without retrieval. Assert the
final structured state and compiled query for Buying, Browsing, Intent Override, Boundary,
negation, multi-value, and paraphrased sessions. These tests measure state semantics without
customer-policy or ranking differences.

### End-to-end comparison

Run the unchanged public evaluator over the full 200-session public set using the same
keyword retriever and produce this matrix:

| State | Current-order policy | Fixed-`other` policy |
|---|---:|---:|
| V1 | TechnicalScore and scenario metrics | TechnicalScore and scenario metrics |
| V2 | TechnicalScore and scenario metrics | TechnicalScore and scenario metrics |

Report Hit Rate@10, MRR, MTTC, Efficiency, and TechnicalScore overall and for Buying,
Browsing, Intent Override, and Boundary sessions. Do not tune against individual public
ground-truth ASINs.

### Contribution isolation

Run these variants under the state-aware order, literal fixed turn order, and fixed-`other`
conditions where applicable:

| Variant | BM25 query | Interpreter | Catalog normalization | State transitions | Recommendation filtering |
|---|---|---|---|---|---|
| V1 | V1 active slots | V1 | Off | V1 | Off |
| V2 state only | V2 active constraints | V1 adapter | Off | V2 | Off |
| Raw-history no state | Raw messages | None | Off | None | Off |
| V2 raw-history query | Raw messages | V1 adapter | Off | V2 | Off |
| V2 interpreted | Raw messages | V2 | Off | V2 | Off |
| V2 normalized | Raw messages plus candidate normalized evidence | V2 | On | V2 | Off |
| V2 full | Raw messages plus retained evidence | V2 | On | V2 | On |

Use the successive deltas to attribute outcomes:

- `V2 state only - V1` measures the new state transitions and query compilation;
- `V2 raw-history query - V2 state only` measures lossless query representation while
  holding V2 state constant;
- `V2 raw-history query - raw-history no state` measures the state contribution while
  holding raw-history retrieval and question policy constant;
- `V2 interpreted - V2 raw-history query` measures conversation interpretation;
- `V2 normalized - V2 interpreted` measures catalog-backed normalization; and
- `V2 full - V2 normalized` measures recommendation-history filtering.

For every delta, report aggregate and scenario metrics plus paired session outcomes:

- miss converted to hit and hit converted to miss;
- earlier and later first-hit turns; and
- better and worse target ranks.

Unit tests and transcript replay prove semantic correctness. They do not count as evidence
that a component improves retrieval outcomes. The end-to-end deltas provide that evidence.

### Component retention

Keep a component in State Baseline V2 only when its isolated variant does one of the
following without reducing TechnicalScore under either fixed question policy:

- improves TechnicalScore by at least `0.005` under one policy, or produces at least two
  net miss-to-hit session conversions; or
- fixes an evaluator-relevant state behavior, such as targeted Intent Override or Boundary
  handling, demonstrated by transcript replay and improved scenario metrics.

If a component only adds general hardening or future flexibility and shows neither outcome,
remove it from V2 and record it as deferred work. Do not retain abstractions solely because
a later adaptive or LLM system might use them.

## Acceptance Criteria

State Baseline V2 is complete when:

- all unit, transcript-replay, agent-contract, and existing evaluator tests pass;
- V1 reproduces its committed current-order metrics within deterministic equality;
- the raw-history fixed-`other` probe is reproduced and its exact metrics are stored as an
  artifact;
- any retained state behavior meets or exceeds the raw-history/no-state control under the
  identical question policy;
- the retained raw-history implementation reproduces the exact fixed-`other` probe
  TechnicalScore of `0.750401`;
- scenario-level, successive-delta, and paired-session results are stored with the overall
  results;
- every retained V2 component satisfies the component-retention rule; and
- no adaptive policy, advanced retriever, LLM, external enrichment, or ground-truth-derived
  catalog feature enters the implementation.

## Follow-up Work

After State Baseline V2 is accepted, advanced retrieval, LLM-based interpretation, and
adaptive question-policy experiments may each use the same constraint and state contracts.
Their order is not decided by this specification, and none is required to validate V2.
