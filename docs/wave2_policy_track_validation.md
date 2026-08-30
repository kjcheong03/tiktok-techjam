# Wave 2 policy-track implementation and validation

Status: complete mechanism implementation; disabled by default; no champion
promotion claim.

Protected F3 access: **none**.

## Delivered techniques

| Reserved ID | Switch/binding | Source | Current classification |
|---|---|---|---|
| `state.attribute_ontology.v1` | `normalizer=catalog_v1` | `ghostlab/state/catalog_ontology.py` | available, unvalidated end to end |
| `state.catalog_normalizer.v1` | `normalizer=catalog_v1` | `ghostlab/state/normalization.py` | available, unvalidated end to end |
| `state.confidence_gated_constraints.v1` | `catalog_normalizer.confidence_threshold` | `ghostlab/state/normalization.py` | available, unvalidated end to end |
| `question.candidate_eig.v1` | `question_variant=candidate_eig` | `ghostlab/policy/eig_questions.py` | interaction reserve |
| `question.reward_voi.v1` | `question_variant=reward_voi` | `ghostlab/research/eig_counterfactual.py` | available; fold fit required |
| `termination.reward_aware.v1` | `question_value_margin` | `ghostlab/policy/eig_questions.py` | available, unvalidated end to end |
| `policy.joint_observable.v1` | `question_variant=joint_observable` | `ghostlab/policy/joint_policy.py` | available, unvalidated end to end |
| `routing.joint_route.v1` | `joint_policy` | `ghostlab/policy/joint_actions.py` | available; fold fit required |
| `research.counterfactual_expert.v2` | research only | `ghostlab/research/counterfactual_expert.py` | available; fold fit required |
| `policy.distilled_expert.v1` | `question_variant=distilled_joint` | `ghostlab/policy/distilled_expert.py` | available; fold fit required |
| `search.expert_iteration.v1` | `maximum_expert_iterations` | `ghostlab/research/counterfactual_expert.py` | available; bounded to three rounds |
| `routing.calibrated_observable.v1` | `routing_variant=calibrated` | `ghostlab/policy/calibrated_router.py` | available; fold fit required |
| `guard.component_fallback.v1` | `component_fallback` | `ghostlab/runtime/component_fallback.py` | available, unvalidated end to end |

The machine-readable map is `configs/techniques/wave2_policy_v1.json`. Every
runtime technique is opt-in. Missing fitted assets fail explicitly; the runtime
does not silently substitute a different learned technique.

## Implementation boundaries

- Catalog ontology construction is deterministic and content-addressed. It uses
  catalog metadata only. The generated ontology stays outside Git and is rebuilt
  with `python -m scripts.build_attribute_ontology`.
- Normalization uses an adapter subclass, leaving the historical
  `ConversationState` file byte-for-byte unchanged. Raw evidence is retained and
  canonicalization is separately traced.
- Candidate EIG uses only current retrieved candidate facets and conversation
  state. Its `0.02` turn cost is the organizer efficiency weight (`0.20`) divided
  by ten turns; it is not a fitted data constant.
- Reward-VOI offsets, joint action tables, expert labels, distilled trees, and
  calibrated routers must be fit inside outer-training data. The router requires
  disjoint fit/calibration session IDs inside that training partition.
- Joint and distilled actions are a bounded typed tuple. Unknown routes/depths
  fail closed and declined/known question attributes cannot be emitted.
- The calibrated router always includes the base route. The component fallback
  replaces only a failed component output; it does not clear conversation state.
- No runtime feature contains target, reward, scenario, future answer, fold ID, or
  evaluator state.

## Verification completed

| Gate | Result |
|---|---|
| Policy/state targeted unit tests | passed |
| Complete repository test suite with `gbdt` extra | **204 passed, 1 skipped** |
| Ruff on all new/modified policy files | passed |
| Mypy on all new/modified policy files (`--follow-imports=skip`) | passed |
| Guarded research/compiled parity | **passed, 0 mismatches, 150 sessions** |
| Unified consolidation audit | passed |
| Historical `ghostlab/state/memory.py` hash | preserved: `104ba1d...6323c6` |
| Protected F3 | not accessed |

The one skipped test is the repository's existing optional neural-cache test, not
a Wave 2 policy test.

## Exploratory fixed EIG comparison

`artifacts/reports/w2_candidate_eig_f1_150.json` evaluates a fixed, non-fitted EIG
policy and its matched sequence control on the 150 adaptive development sessions.
This is an all-development mechanism comparison, **not nested OOF selection
evidence** and not comparable to the selected `0.878963` OOF champion.

| Variant | HitRate@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| matched sequence control | 0.913333 | 0.630860 | 3.266667 | 0.800591 |
| fixed candidate EIG | 0.906667 | 0.629693 | 3.333333 | 0.795575 |
| delta | -0.006666 | -0.001167 | +0.066666 | **-0.005016** |

Decision: preserve `question.candidate_eig.v1` as `interaction_reserve`. The result
does not justify activation, but the candidate-statistics mechanism is sound and
must be retested with catalog normalization, reward-VOI calibration, improved
retrieval, and combination-specific inner-fold tuning. It must not be tuned against
this report.

## Commands

```bash
uv run --frozen --extra gbdt python -m unittest discover -s tests -p 'test_*.py'
uv run --frozen --extra gbdt python -m scripts.validate_guarded_compiled
uv run --frozen --extra gbdt python -m scripts.audit_unified_consolidation
uv run --frozen python -m scripts.run_eig_question_challenger --limit 150
```

Builders and compiled-replay runners:

```text
scripts/build_attribute_ontology.py
scripts/build_distilled_policy.py
scripts/build_calibrated_router.py
scripts/run_joint_policy_challenger.py
scripts/run_distilled_policy_challenger.py
scripts/run_calibrated_router_challenger.py
```

The joint control, distilled policy, and calibrated router have passed their
synthetic mechanism and compiled-contract tests. They have not received nested OOF
TechJam scores on this branch because their counterfactual route/action labels must
first be generated by the integration campaign using fold-local downstream
components. Reporting a score before that would mislabel evidence.
