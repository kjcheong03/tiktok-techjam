# How to reproduce the GhostLab CLI demo

This runbook reproduces the three terminal segments used in the demo video:

1. one evaluator-driven shopping session;
2. a live GhostLab optimizer smoke run followed by the completed F0/F1/F2 campaign;
3. the recorded 550-session A/C/D comparison and active champion verification.

Run every command from the repository root. These commands never rerun or modify the
550-session final-selection evaluation.

## Prerequisites

Complete the setup steps in the main [README](../README.md), including the catalog,
model assets and dense indexes. Confirm that the catalog exists:

```bash
test -f data/catalog.jsonl && echo "catalog ready"
```

Verify the active champion pointer before recording:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_active_candidate.py
```

The command must print `"verified": true`.

## 1. Replay one evaluator-driven runtime session

Resolve the active champion from the tracked pointer:

```bash
GHOSTLAB_CHAMPION_CONFIG=$(
  .venv/bin/python -c \
    'import json; print(json.load(open("configs/active_candidate.json"))["preset_path"])'
)
```

Replay the two-turn `public_0001` development session:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. .venv/bin/python scripts/replay_demo_session.py \
  --sample-id public_0001 \
  --config "$GHOSTLAB_CHAMPION_CONFIG" \
  --catalog data/catalog.jsonl \
  --output-dir artifacts/demo_replay
```

The terminal prints each turn as it happens, including the conversation-state change,
route, preview decision, retrieval contributions, ranking stages, constraint check,
question, Top 10 and evaluator result. The target should reach rank 1 on turn 2.

The replay also writes:

```text
artifacts/demo_replay/demo_replay.json
artifacts/demo_replay/demo_replay.md
```

Add `--verbose` to the replay command only when raw reason codes and low-level runtime
diagnostics are needed.

## 2. Show a real optimizer evaluation

The following bounded smoke run executes the real candidate evaluator on 30 development
sessions. F0 therefore uses six sessions per candidate. It writes to demo-only paths and
cannot access the 550-session final-selection set.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. .venv/bin/python scripts/run_adaptive_hybrid_campaign.py \
  --config configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json \
  --technique-catalog configs/techniques/catalog_v2.json \
  --warm-start configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json \
  --search-mode additive_warm_start \
  --freeze-warm-semantic \
  --candidate-limit 19 \
  --beam-width 1 \
  --higher-order-rounds 0 \
  --f1-candidates 1 \
  --f2-candidates 1 \
  --hpo-trials-per-structure 1 \
  --max-samples 30 \
  --output artifacts/demo_optimizer/campaign.json \
  --checkpoint artifacts/demo_optimizer/checkpoint.json
```

For the recording, wait until the terminal shows one `evaluation_finished` event and the
next `evaluation_started` event. This proves that GhostLab evaluated one configuration
and moved to another. Press `Ctrl+C` after capturing that sequence. Completed candidate
evaluations remain in the demo checkpoint and are resumed if the command is run again.

Model loading can print several `Loading weights` bars. Silence between
`evaluation_started` and `evaluation_finished` is normal because the current CLI logs at
candidate boundaries rather than every session. Some residual-ranker candidates take
longer because they perform fold-safe fitting.

This is a time-bounded live smoke, not the full optimizer result. Do not describe its six
F0 sessions as the production F0 budget.

## 3. Show the completed GhostLab campaign

Print the frozen full-development campaign immediately after the live smoke:

```bash
PYTHONPATH=. .venv/bin/python scripts/show_optimizer_summary.py
```

This read-only command reports the completed campaign:

- F0: 14 candidates on 330 sessions;
- F1: 5 candidates on 825 sessions;
- F2: 3 candidates on all 1,650 development sessions;
- the frozen development winner, selected techniques, metrics and verified hash.

It reads the tracked campaign and finalist reports. It does not load models or restart
optimization.

## 4. Show the final 550-session A/C/D results

Print the final comparison:

```bash
PYTHONPATH=. .venv/bin/python scripts/show_champion_results.py
```

The command prints A, C and D on the same 550 one-time final-selection sessions:

- A: organizer BM25 starter;
- C: fixed adaptive architecture;
- D: active GhostLab Champion.

It also checks that D's report configuration matches `configs/active_candidate.json` and
that the active configuration file matches its recorded SHA-256. It reads frozen evidence
only and does not spend the final-selection set again.

## Recording order

Use this order in the final video:

1. runtime session replay;
2. live optimizer smoke until one candidate finishes and another begins;
3. completed F0/F1/F2 optimizer summary;
4. final 550-session A/C/D results.

## Troubleshooting

### `data/catalog.jsonl` is missing

Follow the catalog download and checksum steps in the main README. When using a temporary
Git worktree on the same machine, linking the already-verified catalog is sufficient:

```bash
ln -s /absolute/path/to/main-repository/data/catalog.jsonl data/catalog.jsonl
```

### The optimizer appears frozen after loading weights

Wait for the matching `evaluation_finished` event. A candidate may take longer when it
fits a residual ranker. `Ctrl+C` is safe for the demo: earlier completed evaluations have
already been checkpointed.

### Re-running the optimizer smoke resumes immediately

That is expected because `artifacts/demo_optimizer/checkpoint.json` is resumable. Use a
different demo checkpoint and output path when a completely fresh recording is required.
