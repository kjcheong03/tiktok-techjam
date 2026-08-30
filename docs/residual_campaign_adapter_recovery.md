# Residual Campaign Adapter Recovery

## Decision

`adaptive_autonomous_augment_v1` is retired and must not be resumed. Its completed
residual-reranker outcomes are invalid because the runtime adapter converted structured
recommendation objects into string representations rather than extracting their
`parent_asin` identifiers.

The corrected augmentation campaign is `adaptive_autonomous_augment_v2`.

## Root cause

The parent runtime correctly returned recommendations in the competition response shape:

```json
{"parent_asin": "B01EX670LS"}
```

The residual adapter previously applied `str(item)`, producing a value resembling:

```text
{'parent_asin': 'B01EX670LS'}
```

That complete string is not a catalog identifier. The evaluator therefore removed every
returned recommendation and reported zero Hit@10, zero MRR and an overall zero score. The
outcome was recorded as complete because evaluator-compatible replay intentionally contains
invalid runtime responses instead of terminating the campaign.

The earlier standalone State V2 residual evaluation was unaffected. It normalized parent
responses into valid catalog IDs before applying the membership-preserving permutation.
Therefore, its algorithmic evidence remained meaningful, but it did not validate the
campaign runtime adapter.

## Correction

The adapter now:

1. extracts the canonical `parent_asin` from each structured recommendation;
2. sends only canonical IDs into the residual feature and ranking path;
3. returns the original response exactly when the activation gate remains closed;
4. verifies that an activated result has identical membership and length; and
5. reorders the original structured recommendation objects, preserving the API schema and
   any attached scores.

## Validation

- Full automated suite: `379 passed, 1 skipped`.
- Focused residual, fold-fit, campaign-evaluator and wrapper tests: `21 passed`.
- Ruff lint and formatting checks passed.
- Static typing of the changed runtime and wrapper paths passed with optional third-party
  imports ignored. The ordinary mypy invocation still reports the pre-existing absence of
  scikit-learn type stubs in `ghostlab/training/residual.py`.
- The new `adaptive_autonomous_augment_v2` template passed admission:
  - `campaign_ready: true`;
  - `admitted_count: 30`;
  - `admitted_without_trial: []`;
  - `blocked_count: 0`;
  - `materializable_structure_count: 199`.
- A real fold-fitted asset returned ten valid recommendations and preserved exact Top-10
  membership while changing only their order.
- The exact previously-zero F0 job `job-02db3d34717eb5a9` was replayed with unchanged
  candidate, seed, sample selection and fit receipts:

| Metric | Broken adapter | Corrected adapter |
|---|---:|---:|
| Technical score | `0.000000` | `0.932917` |
| Hit@10 | `0.000000` | `0.958333` |
| MRR | `0.000000` | `0.937500` |
| MTTC | `11.000` | `2.375` |

This replay proves that fold training was not the cause of the zeros. The failure was
entirely at the runtime response boundary.

## Running the corrected campaign

Commit the correction first because campaign freezing requires a clean committed worktree.
Then run:

```bash
caffeinate -i uv run python -m scripts.run_autonomous_end_to_end \
  --mode augment \
  --prepare-assets
```

The wrapper resolves `--mode augment` to
`configs/campaigns/adaptive_autonomous_augment_v2.template.json` and writes only beneath
`artifacts/campaigns/adaptive_autonomous_augment_v2/`. It cannot reuse the invalid v1
checkpoint.
