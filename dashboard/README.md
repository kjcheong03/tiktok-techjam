# GhostLab results dashboard

A dependency-free local dashboard for evaluator, unified preset, baseline-comparison,
and adaptive campaign result JSON files.

From the repository root, start it with:

```bash
uv run python dashboard/server.py
```

Then open <http://127.0.0.1:8787/dashboard/>. Stop the server with `Ctrl+C`.

The dashboard discovers compatible JSON files in `artifacts/reports/`, plus
`artifacts/baseline_results.json` and `results.json`. Newly generated reports appear
after refreshing the page. You can also use **Import JSON** or drag files anywhere onto
the page; imported files are parsed only in the browser and are not uploaded.

Supported result shapes include:

- direct `evaluator.local_evaluator` output;
- unified reports with a nested `metrics` object;
- mappings containing several named baseline runs; and
- campaign reports containing metric-bearing `records`.

Use a different port if `8787` is already occupied:

```bash
uv run python dashboard/server.py --port 9000
```
