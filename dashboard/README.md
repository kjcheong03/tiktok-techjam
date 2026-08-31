# GhostLab results dashboard

A dependency-free local dashboard for evaluator, unified preset, baseline-comparison,
fair A/B/C/D system-comparison, and adaptive campaign result JSON files.

From the repository root, start it with:

```bash
uv run python dashboard/server.py
```

Then open <http://127.0.0.1:8787/dashboard/>. Stop the server with `Ctrl+C`.

The model selector intentionally exposes only four stable slots: **A: BM25**,
**B: BM25 + teammate State V2**, **C: adaptive control**, and **D: frozen GhostLab
champion / challenger**. When a fair comparison report is available, those slots resolve
to its matching system rows. D uses the selected champion or the top frozen challenger,
so a promoted challenger is never listed twice. You can still use **Import JSON** or drag
files anywhere onto the page; imported files are parsed only in the browser and are not
uploaded.

Supported result shapes include:

- direct `evaluator.local_evaluator` output;
- unified reports with a nested `metrics` object;
- mappings containing several named baseline runs; and
- fair comparison reports with a top-level `systems` list; and
- campaign reports containing metric-bearing `records`.

For a fair comparison report, all displayed A/B/C/D numbers must come from the same
ordered sample IDs, catalog and evaluator contract. The dashboard labels A/B as
reference-only and C/D as the promotion control/challenger; it does not infer champion
eligibility from the highest raw score.

A, B and C remain pinned in the leaderboard. When a development report contains D1-D3,
choose the displayed D from the challenger dropdown. A final holdout report contains
only the single frozen D, so the dropdown is fixed and cannot be used for post-holdout
selection.

Use a different port if `8787` is already occupied:

```bash
uv run python dashboard/server.py --port 9000
```
