# GhostLab results dashboard

A dependency-free local dashboard for evaluator, unified preset, baseline-comparison,
three-system comparison, and adaptive campaign result JSON files.

From the repository root, start it with:

```bash
uv run python dashboard/server.py
```

Then open <http://127.0.0.1:8787/dashboard/>. Stop the server with `Ctrl+C`.

The dashboard loads three stable systems automatically: **Organizer BM25 Starter**,
**Fixed Adaptive Architecture**, and **GhostLab Champion**. When a fair comparison
report is available, the systems resolve to its matching rows. The Champion slot follows
`configs/active_candidate.json` and its recorded comparison evidence, so refreshing the
page reflects a newly activated, evaluated champion. The leaderboard appears first and
includes TechnicalScore, Hit Rate@10, MRR, normalized efficiency, and MTTC.
Click a system name to inspect its detailed results. You can still use **Import JSON** or
drag files anywhere onto the page; imported files are parsed only in the browser and are
not uploaded.

Supported result shapes include:

- direct `evaluator.local_evaluator` output;
- unified reports with a nested `metrics` object;
- mappings containing several named baseline runs; and
- fair comparison reports with a top-level `systems` list; and
- campaign reports containing metric-bearing `records`.

For a fair comparison report, all displayed numbers must come from the same ordered
sample IDs, catalog, and evaluator contract. The dashboard presents the frozen selected
champion rather than exposing internal campaign labels or post-holdout selection controls.

Use a different port if `8787` is already occupied:

```bash
uv run python dashboard/server.py --port 9000
```
