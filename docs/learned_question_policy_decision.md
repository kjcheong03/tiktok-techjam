# Learned Question Policy Challenger Decision

Date: 2026-08-26

Experiment: `learned_question_linear_v1`

Parent: commit `189f0c6`, policy `ghostlab_champion_linear_v1`

Decision: `PARKED_STANDALONE`

## Outcome

The minimum independent learned-question challenger is valid but does not replace
the fixed-sequence champion. Its five-fold stitched OOF technical score is
`0.808951`, versus `0.819719` for the frozen parent control, a paired mean session
reward delta of `-0.010768`. The 95% paired bootstrap interval is
`[-0.029205, 0.005465]`; the paired randomization p-value is `0.247975` (10 wins,
117 ties, 23 losses).

| Policy | Hit Rate@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Fixed-sequence champion | 0.933333 | 0.639508 | 2.940000 | 0.819719 |
| Learned linear, OOF | 0.926667 | 0.628947 | 3.153333 | 0.808951 |
| Heuristic adaptive control | 0.920000 | 0.635222 | 3.040000 | 0.809767 |
| Fixed no-`other` control | 0.833333 | 0.533722 | 4.333333 | 0.710116 |
| Absorbing-stop control | 0.313333 | 0.167508 | 8.173333 | 0.263452 |

Fold scores were `0.812631`, `0.794850`, `0.804984`, `0.813448`, and `0.819696`
(population standard deviation `0.008528`). Scenario reward deltas against the
champion were `+0.017143` Boundary, `-0.020048` Browsing, `-0.013992` Buying, and
`+0.013182` Intent Override. The Browsing regression crosses the predeclared
material-regression threshold and is not offset by an aggregate gain.

## Mechanism and validity

The policy considers every currently informative official typed attribute,
repeatable broad `other`, and an absorbing stop. Its features are available before
the action: turn budget, parsed active/asked/declined slots and provenance,
response usefulness, query counts, and sparse-retrieval margin, entropy, and
concentration. Target ID/rank, scenario, future answers, rewards, and simulator
state are absent from runtime inputs.

Counterfactual labels cover 431 states on fixed champion trajectories. A question
is applied for one turn and then the champion resumes at the next absolute turn;
stop remains `null` thereafter. The oracle first-action advantage was `0.023106`,
but much of the apparent action headroom was tied or not predictable from the
observable linear features. Complete-policy OOF evaluation, rather than the oracle
label statistic, determined the decision. The parent ranking asset was held frozen
for both candidate and control; only question-policy parameters were fitted inside
each grouped outer fold.

OOF behavior was legal and deterministic: 186 `other`, 78 feature, 53 use-case,
31 material, 61 stop, and smaller counts for the other actions over 462 turns. The
policy stopped in 30/150 sessions and had a non-stop consecutive-repeat rate of
`0.092949`. The all-development refit scored `0.805162`, confirming that a deployable
refit does not reverse the OOF conclusion.

## Feasibility and next decision

The challenger passed feasibility gates: 2.503925 s incremental agent
initialization, 49.476875 ms warm-turn p95, 2252.672 MB observed process peak,
0.009019 MB model asset, zero tokens, and zero external calls. These measurements
cannot rescue a negative effectiveness result.

No shallow tree or GBDT is justified: the linear model did not leave positive,
stable OOF headroom. Retest only after a material dependency changes, such as a
stronger structured query/retrieval representation that changes question value,
or after substantially richer non-target observable state is independently
validated. The fixed sequence remains the champion.

Evidence: `configs/experiments/learned_question_linear_v1.json`,
`artifacts/reports/learned_question_linear_v1.json`, and
`artifacts/experiments/learned_question_linear_v1/`.
