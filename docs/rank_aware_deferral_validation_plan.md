# Rank-Aware Deferral Validation Plan

Status: documented for future research; not implemented or engine-registered.

## Decision summary

Do not implement rank-aware deferral until a trace-based oracle audit establishes
that useful, observable, and statistically testable opportunities exist.

Deferral is a separate sequential policy, not a reranker setting. It chooses
between:

1. returning the current normalized recommendations now; and
2. returning one legal high-value question with no recommendations, then rebuilding
   the query, retrieval, and ranking after the answer.

The first audit uses frozen State Baseline V2. A later interaction audit may use a
frozen State V2 plus residual-reranker parent, but the two techniques must initially
remain separately trained and separately attributable.

## Why this remains a research proposal

Same-turn, membership-preserving reranking can improve MRR without changing
Hit@10 or MTTC. Deferral intentionally gives up a possible immediate hit and can
change all three metrics. It therefore has materially greater downside.

Previous learned-question experiments do not constitute a true deferral test. They
selected questions or an absorbing stop while the runtime continued to return
recommendations on the same turn. Those experiments also failed to beat their
controls:

- learned linear question policy: `0.808951` versus `0.819719`;
- learned-question and GBDT interaction: `0.847744` versus `0.861417`, with
  negative deltas in all five folds.

These results require an audit-first approach. They do not prove that ask-only
deferral cannot work after State V2, but they show that question value has been
difficult to predict from observable features.

## Architectural prerequisite: finalize before committing history

The current runtime is internally consistent: it builds a response containing the
ranked products and records the products in that returned response as shown. A
future after-response wrapper must not replace or suppress those recommendations,
because the parent would already have committed products that the user never saw.

The generally sound response transaction is:

```text
update state and build candidate ranking
        -> choose the final action
        -> construct and normalize the final response
        -> validate the final response
        -> atomically record only recommendations actually returned
```

Required invariants:

- an ask-only response records zero shown product IDs;
- a recommend response records the normalized IDs actually returned, not an
  upstream candidate list;
- invalid, missing, added, duplicated, or truncated items are resolved before the
  history commit;
- an intent-epoch change applies the existing history reset rule before filtering;
- failure before finalization returns the validated current recommendation fallback
  and records that fallback exactly once;
- failure after history commit must not be recoverable by silently changing the
  returned membership.

This boundary is an architectural requirement independent of the deferral outcome.
It should be addressed before any runtime component is allowed to suppress or
replace recommendation membership. The State V2 data object should remain focused
on state mutation and query evidence; a thin conversation controller should own
final action selection, response assembly, and history commit.

## Exact evaluator objective

For a first hit at rank `r <= 10` on turn `t`, the per-session reward is:

```text
0.50 + 0.30 / r + 0.20 * clip((11 - t) / 10, 0, 1)
```

A one-turn delay costs `0.02` when hit membership and rank otherwise remain equal.
The actual deferral comparison must nevertheless use the complete continuation
distribution, because the target may improve, remain present, fall in rank, leave
Top-10, recover on a later turn, or never recover.

The policy objective is:

```text
defer only when

lower_confidence_bound(
    expected_full_continuation_reward_after_best_question
    - expected_reward_from_recommending_now
) > frozen_safety_margin
```

This calculation, rather than a fixed rank rule, already accounts for the later-turn
penalty.

## Correct activation hypothesis

High confidence in one particular top-ranked product is a reason to recommend now,
not defer. The proposed opportunity is narrower:

- cumulative probability that the target is somewhere in the current Top-10 is
  sufficiently high;
- probability is diffuse across several plausible products, so Top-1 or Top-3
  confidence remains low;
- one legal unanswered question is expected to sharply separate those products or
  recover a missing target;
- the ambiguity comes from missing user information, not from a reranker failing to
  use information already present;
- enough turns remain to justify the continuation risk.

If available evidence already identifies the correct product but the ranker leaves
it low, the correct intervention is reranking, not deferral.

## Receding-horizon action policy

Do not maintain or consume a fixed queue of questions. At every eligible turn:

1. update State V2 from the new user message;
2. produce the current validated candidate ranking;
3. enumerate all currently legal, unanswered questions;
4. estimate the full continuation value of each question;
5. select only the highest-value question;
6. compare it with recommending now;
7. either recommend now or return that one question with an empty recommendation
   list;
8. after the answer, rebuild state, query, retrieval, and ranking and recompute from
   scratch.

The first implementation, if justified, should allow at most one ask-only deferral
per intent epoch. It must never defer on the final turn. A correction that creates a
new intent epoch resets eligibility, but the policy must still pass the full value
gate again.

## Stage 1: trace and oracle audit

### Frozen parents

Run the audit first against a hash-pinned State Baseline V2 parent. After any
residual merger is complete and independently validated, run a separate interaction
audit against a hash-pinned State V2 plus residual parent.

Do not modify or jointly train either parent during the audit.

### Eligible trace states

Audit every pre-terminal turn at which:

- at least one legal unanswered question exists;
- the turn is earlier than the final turn;
- the session has not exceeded the initial per-intent deferral allowance; and
- the current response contains at least one valid recommendation.

For each state, preserve a complete clone of conversation state, intent epoch,
asked attributes, recommendation history, stopped-policy state, and any query or
retrieval inputs needed for deterministic replay. Counterfactual branches must not
mutate one another or the control trajectory.

### Required branches

For every eligible state, evaluate:

1. **Recommend now control:** emit the current normalized recommendations.
2. **Current ask-and-recommend behavior:** emit the current list and current question,
   where applicable.
3. **One-question ask-only branches:** for each legal unanswered question, emit no
   recommendations, apply its simulated answer, and resume the frozen parent at the
   next absolute turn.
4. **Full legal hindsight oracle:** choose the action producing the highest complete
   session reward, solely to measure an unattainable upper bound.

At minimum, report both next-turn outcomes and full-continuation outcomes. A target
that misses next turn but recovers later is not equivalent to a permanent miss.

### Oracle levels

Report three distinct quantities:

1. **Clairvoyant action-oracle upper bound:** best legal action at each eligible state.
2. **One-question deferral-oracle bound:** at most one ask-only action per intent
   epoch followed by the frozen continuation.
3. **Observable-policy headroom:** gain achievable using only legal runtime features
   under grouped out-of-fold evaluation.

The first two use hidden target outcomes only for offline measurement and labels.
They are not runtime policies and cannot be promoted.

### Audit outputs

For each parent and overall scenario, report:

- number and fraction of eligible states and sessions;
- number and fraction with positive one-question oracle gain;
- recommend-now, ask-and-recommend, and ask-only technical scores;
- Hit@10, MRR, MTTC, and complete reward deltas;
- target retention, next-turn loss, later recovery, and permanent-miss rates;
- current and future target-rank transitions;
- winning question distribution by attribute and turn;
- gain distribution, not only the mean;
- scenario-specific deltas;
- oracle activation frequency and maximum defensible policy frequency;
- paired session outcomes and confidence intervals;
- deterministic replay and state-isolation failures.

If the one-question oracle has negligible aggregate gain, stop without training a
model. If it has gain but too few opportunities or no observable separation, retain
the plan for additional independent data rather than implementing runtime deferral.

## Stage 2: observable learnability audit

Proceed only after the oracle audit passes a predeclared headroom gate.

### Allowed runtime features

Candidate features may include:

- turn and turns remaining;
- current normalized list cardinality;
- calibrated per-candidate probabilities and cumulative Top-10/Top-3 mass;
- score margins, entropy, concentration, and disagreement among rankers;
- active constraints and their provenance or confidence;
- intent epoch, correction signals, and already-asked attributes;
- candidate-facet distributions and estimated information gain for each legal
  question;
- retrieval/ranking stability under observable perturbations;
- prior deferral count in the current intent epoch.

### Forbidden runtime features

Do not expose:

- target ID, target rank, hit status, or reciprocal rank;
- scenario label;
- future simulated answers;
- future candidate lists or rewards;
- fold membership or evaluation outcomes;
- any globally fitted statistic containing the held-out session.

### Learning target and calibration

Train a compact action-value or treatment-effect model against the paired difference
between full continuation reward after a question and reward from recommending now.
Do not train a generic relevance classifier and interpret its uncertainty as
question value.

Fit the parent, any residual component, the deferral model, and its calibration only
inside the appropriate training partitions. Freeze the activation threshold and
safety margin on inner validation sessions. Measure once on excluded outer-fold
sessions.

Candidate rows from one conversation must never cross folds. Model families and
feature subsets must remain small relative to the number of independent sessions.

## Runtime fallback and failure accounting

The current normalized recommendation response is the same-turn fallback when:

- the deferral asset or fit receipt is missing or invalid;
- feature extraction or inference fails;
- calibration is unavailable;
- no legal question exists;
- the confidence bound does not exceed the frozen margin;
- the final turn or per-epoch deferral limit has been reached.

Fallback is a safety mechanism, not evidence of effectiveness. Trace and report
separately:

- deliberate `recommend_now` decisions;
- confidence-gated non-activations;
- missing-asset fallbacks;
- inference exceptions;
- illegal-question or invariant failures;
- response-finalization failures.

A candidate that appears safe only because operational failures repeatedly return
the parent must not be promoted. After an answer, a previously saved list may be
used for diagnosis, but it must not be returned blindly; it must be revalidated
against the updated intent and catalog constraints.

## Validation and promotion requirements

Use grouped nested OOF development evidence:

1. fit all required components on outer-training sessions;
2. select model, calibration, threshold, and margin on inner validation sessions;
3. evaluate exactly once on the excluded outer fold;
4. stitch outer-fold predictions by session;
5. keep deployment refits separate from OOF evidence;
6. record immutable dataset, split, parent, feature, model, calibration, and policy
   hashes in fit receipts.

Predeclare numerical promotion gates after the oracle audit establishes realistic
effect and sample sizes. At minimum, require:

- positive grouped OOF technical-score delta;
- stable fold direction rather than one-fold concentration;
- no material overall or scenario Hit@10 regression;
- no material scenario technical-score regression;
- acceptable MTTC and permanent-miss behavior;
- deterministic runtime/research parity;
- zero target/future leakage;
- zero history-commit and response-membership invariant failures;
- operational fallback rates below a predeclared ceiling;
- confidence intervals and paired tests reported without treating them as runtime
  calibration.

Observed confirmation sessions are internal campaign evidence, not a globally
untouched holdout. Only genuinely new private evaluation can support a new external
generalization claim.

## Data sufficiency

Count independent sessions and activated sessions, not candidate rows. A policy
activating on five percent of 150 sessions yields only about seven or eight observed
activations and cannot support a flexible learned policy or a strong calibration
claim.

More independent, representative sessions may make the policy viable. Synthetic or
counterfactual rows help only when the simulator and frozen continuation represent
the deployment distribution; they do not create independent evidence. Use the
oracle audit's observed opportunity rate and effect distribution for a formal power
and data-collection plan.

## Relationship to residual reranking

Residual reranking and deferral remain separate techniques:

- residual reranking acts on the current turn and preserves normalized membership;
- deferral suppresses the current list, changes timing, and may change future
  membership;
- reranker confidence may be an observable deferral feature, but the reranker must
  not decide deferral by itself;
- residual failure despite sufficient existing evidence should be repaired as a
  ranking problem;
- deferral is relevant when a specific answer is expected to add missing
  discriminating information.

Validate State V2 deferral first. Only after both techniques independently pass
their gates should champion augmentation and interaction testing be considered.

## Future implementation sequence

1. Finish and freeze the current adaptive-optimizer work.
2. Create an isolated audit worktree from the exact clean parent commit.
3. Implement and test the action-before-history-commit architectural boundary.
4. Build the state-cloning and deterministic counterfactual audit harness.
5. Run the State V2 oracle trace audit.
6. Stop, collect more data, or proceed according to the predeclared oracle gate.
7. If justified, run the observable learnability audit with a compact model.
8. If OOF gates pass, implement the minimal runtime policy, initially disabled and
   fit-required.
9. Audit State V2 plus residual as a separate parent interaction.
10. Consider autonomous-engine registration only after fit-receipt, history,
    calibration, runtime-parity, and promotion gates are implemented.

## Durable decision rule

Do not implement deferral merely because the evaluator values high ranks or because
the current Top-10 appears uncertain. Implement it only if the trace audit shows
that one legal question creates meaningful full-continuation reward headroom and a
grouped OOF policy can identify those opportunities using observable information.
