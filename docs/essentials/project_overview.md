# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

## GhostLab Unified Quick Start

The unified system preserves the validated candidate and all reusable challenger
implementations behind versioned, independently switchable presets. The autonomous
campaign starts from the pure keyword baseline, evaluates compatible techniques and
combinations through disjoint F0/F1/F2 stages, and proposes candidates without activating
one automatically.

Supported platforms are macOS and Linux. On Windows, use WSL2 with Ubuntu; native Windows
is not yet supported because Unix-specific resource measurement remains in the campaign
runtime and the full workflow has not passed Windows CI. Use CPython 3.10 through 3.13
(3.12 recommended), `uv`, Git, and the released `data/catalog.jsonl`.

### 1. Install and validate

From the repository root:

```bash
uv sync --all-extras --group dev
uv pip check
uv run ruff check ghostlab scripts tests
uv run mypy ghostlab
uv run pytest -q
git status
```

If the catalog is not present, complete **Download the Catalog** above first. Review and
commit intended implementation/configuration changes before starting a new campaign;
campaign freezing rejects a dirty worktree so that all inputs belong to a reproducible
commit.

### 2. Run or resume the autonomous search

```bash
uv run python -m scripts.run_autonomous_end_to_end --prepare-assets
```

The command prepares and verifies optional model assets, freezes or resumes the versioned
campaign, runs the bounded F0/F1/F2 search, and materializes three independently confirmed
safe proposals or stops without padding the result. It prints one preparation command per
proposal. The first dense index can take roughly
20–25 minutes on CPU and the full campaign can take several hours. If interrupted, run
the same command again; completed checkpoint jobs are reused.

### 3. Review and prepare one proposal

Review:

- `artifacts/campaigns/autonomous_state_v2_v1/admission.json`;
- `artifacts/campaigns/autonomous_state_v2_v1/evidence.json`; and
- `artifacts/proposals/autonomous_state_v2_v1/proposal_manifest.json`.

Choose one proposal and run its preparation command printed by Stage 2. Preparation
revalidates and hashes the immutable preset, then prints its exact hash-bound activation
command. Candidate-specific commands cannot be written here
in advance because their IDs, paths, and hashes are campaign outputs.

### 4. Activate, verify, or roll back

Only after human review, run the activation command printed by preparation. Activation
prints the verification and rollback commands; run the verification command next. To
reject the selection or recover from a failed verification, run:

```bash
uv run python -m scripts.activate_candidate --rollback
```

The autonomous wrapper never commits, pushes, reads the protected F3 holdout, or changes
the active method by itself. Start with `docs/essentials/README.md` for the curated project
reading order. Use `docs/essentials/unified_technique_operations.md` for every technique,
dependency, switch, and retest rule, and
`docs/essentials/autonomous_unified_system_reference.md` for the complete campaign,
overfitting, pruning, proposal, activation, and recovery specification.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
