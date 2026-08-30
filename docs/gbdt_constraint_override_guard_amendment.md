# Observable override-guard amendment

This final bounded follow-up was frozen after corrected v2 was committed as parked
and before guarded outcomes were observed. It tests one causal hypothesis only:
constraint-aware ranking helps ordinary turns, while the matched base GBDT is safer
after an observable override invalidation.

The guard reads only current `ConversationState.invalidated_reason` values. It does
not inspect scenario, target, profile, future turns, or evaluator state. Both routes
reuse the same fitted fold models, and no feature, round, tree, threshold, or
hyperparameter is added. Failure of any frozen gate ends this constraint-search
line.
