"use strict";

const COLORS = ["#39f0cf", "#ff4f8b", "#9f7aea", "#f5bd5b", "#5da8ff", "#ff8c5a", "#7bdd65", "#d786ff"];
const SCORE_KEYS = ["recommended_technical_score", "hit_rate_at_10", "mrr", "mttc", "efficiency"];

const state = { reports: [], runs: [], activeId: null };
const $ = (selector) => document.querySelector(selector);
const elements = {
  status: $("#server-status"), reportSelect: $("#report-select"), reportHint: $("#report-hint"),
  loadReport: $("#load-report"), fileInput: $("#file-input"), runTabs: $("#run-tabs"), runCount: $("#run-count"),
  empty: $("#empty-state"), content: $("#dashboard-content"), selectedName: $("#selected-run-name"),
  selectedSource: $("#selected-run-source"), metricGrid: $("#metric-grid"), comparison: $("#comparison-chart"),
  scenarioMetric: $("#scenario-metric"), scenarioChart: $("#scenario-chart"), rankChart: $("#rank-chart"),
  distributionLabel: $("#distribution-label"), sessionSearch: $("#session-search"), scenarioFilter: $("#scenario-filter"),
  outcomeFilter: $("#outcome-filter"), sessionTable: $("#session-table"), sessionCount: $("#session-count"),
  noSessions: $("#no-sessions"), dropOverlay: $("#drop-overlay"), toast: $("#toast"),
};

function isNumber(value) { return typeof value === "number" && Number.isFinite(value); }
function hasMetrics(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const metricSource = value.metrics && typeof value.metrics === "object" ? value.metrics : value;
  return SCORE_KEYS.some((key) => isNumber(metricSource[key]));
}
function slug(value) { return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""); }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[character]);
}
function deriveScore(metrics) {
  if (isNumber(metrics.recommended_technical_score)) return metrics.recommended_technical_score;
  if (![metrics.hit_rate_at_10, metrics.mrr, metrics.mttc].every(isNumber)) return null;
  const efficiency = Math.max(0, Math.min(1, (11 - metrics.mttc) / 10));
  return 0.5 * metrics.hit_rate_at_10 + 0.3 * metrics.mrr + 0.2 * efficiency;
}
function normalizeScenario(raw = {}) {
  const metrics = { ...raw };
  if (!isNumber(metrics.efficiency) && isNumber(metrics.mttc)) metrics.efficiency = Math.max(0, Math.min(1, (11 - metrics.mttc) / 10));
  metrics.recommended_technical_score = deriveScore(metrics);
  return metrics;
}

function normalizeRun(value, fallbackName, source, suffix = "run") {
  const metricSource = value.metrics && typeof value.metrics === "object" ? value.metrics : value;
  const metrics = normalizeScenario(metricSource);
  const sessions = Array.isArray(value.sessions) ? value.sessions : Array.isArray(metricSource.sessions) ? metricSource.sessions : [];
  const scenarioSource = metricSource.scenario_metrics || value.scenario_metrics || {};
  const scenarios = Object.fromEntries(Object.entries(scenarioSource).map(([name, data]) => [name, normalizeScenario(data)]));
  const configuredName = value.experiment_id || value.name || value.candidate_id || value.config?.experiment_id;
  const name = configuredName || fallbackName;
  const sampleCount = value.sample_count ?? metricSource.sample_count ?? sessions.length ?? null;
  return {
    id: `${source}::${suffix}`,
    name: String(name).replaceAll("_", " "), source, metrics, scenarios, sessions,
    sampleCount: isNumber(sampleCount) ? sampleCount : null,
    color: COLORS[0],
  };
}

function extractRuns(payload, fallbackName, source) {
  if (hasMetrics(payload)) return [normalizeRun(payload, fallbackName, source)];
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];
  if (Array.isArray(payload.records)) {
    const runs = payload.records.filter(hasMetrics).map((record, index) => normalizeRun(
      record,
      `${fallbackName} · candidate ${record.ordinal ?? index + 1}`,
      source,
      `record-${record.ordinal ?? index}`,
    ));
    if (runs.length) return runs;
  }
  return Object.entries(payload).filter(([, value]) => hasMetrics(value)).map(([key, value]) =>
    normalizeRun(value, key, source, `key-${key}`));
}

function addRuns(runs) {
  let added = 0;
  for (const run of runs) {
    if (state.runs.some((existing) => existing.id === run.id)) continue;
    run.color = COLORS[state.runs.length % COLORS.length];
    state.runs.push(run);
    added += 1;
  }
  if (!state.activeId && state.runs.length) state.activeId = state.runs[0].id;
  render();
  return added;
}

async function loadReport(report, quiet = false) {
  try {
    const response = await fetch(report.url, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const runs = extractRuns(payload, report.label, report.path);
    if (!runs.length) throw new Error("No evaluation metrics were found");
    const added = addRuns(runs);
    if (!quiet) showToast(added ? `Loaded ${added} run${added === 1 ? "" : "s"}` : "Those runs are already loaded");
  } catch (error) {
    showToast(`Could not load report: ${error.message}`);
  }
}

async function discoverReports() {
  try {
    const response = await fetch("/api/reports", { cache: "no-store" });
    if (!response.ok) throw new Error("dashboard server unavailable");
    const payload = await response.json();
    state.reports = payload.reports || [];
    elements.status.className = "status-pill online";
    elements.status.innerHTML = "<i></i> Local server online";
    elements.reportSelect.innerHTML = state.reports.map((report, index) =>
      `<option value="${index}">${escapeHtml(report.label)} · ${report.run_count} run${report.run_count === 1 ? "" : "s"}</option>`).join("");
    elements.reportHint.textContent = `${state.reports.length} compatible reports discovered in artifacts.`;
    const featured = state.reports.filter((report) => report.featured);
    for (const report of featured) await loadReport(report, true);
  } catch (_error) {
    elements.status.className = "status-pill offline";
    elements.status.innerHTML = "<i></i> Import-only mode";
    elements.reportSelect.innerHTML = "<option>Start the dashboard server to browse artifacts</option>";
    elements.reportSelect.disabled = true;
    elements.loadReport.disabled = true;
    elements.reportHint.textContent = "You can still import JSON files from your computer.";
  }
}

async function importFiles(files) {
  let total = 0;
  for (const file of files) {
    try {
      const payload = JSON.parse(await file.text());
      const runs = extractRuns(payload, file.name.replace(/\.json$/i, ""), file.name);
      if (!runs.length) throw new Error("no recognized evaluation metrics");
      total += addRuns(runs);
    } catch (error) {
      showToast(`${file.name}: ${error.message}`);
    }
  }
  if (total) showToast(`Imported ${total} run${total === 1 ? "" : "s"}`);
}

function activeRun() { return state.runs.find((run) => run.id === state.activeId) || state.runs[0]; }
function percent(value) { return isNumber(value) ? `${(value * 100).toFixed(1)}%` : "—"; }
function decimal(value, digits = 3) { return isNumber(value) ? value.toFixed(digits) : "—"; }
function metricDisplay(key, value) {
  if (["hit_rate_at_10", "efficiency"].includes(key)) return percent(value);
  if (key === "mttc") return isNumber(value) ? `${value.toFixed(2)}` : "—";
  return decimal(value);
}

function renderTabs() {
  elements.runCount.textContent = `${state.runs.length} run${state.runs.length === 1 ? "" : "s"}`;
  elements.runTabs.innerHTML = state.runs.map((run) => `
    <div class="run-tab ${run.id === state.activeId ? "active" : ""}" style="--run-color:${run.color}" data-run-id="${escapeHtml(run.id)}" role="button" tabindex="0">
      <i class="run-dot"></i><span class="tab-name" title="${escapeHtml(run.name)}">${escapeHtml(run.name)}</span>
      <button class="remove-run" data-remove-id="${escapeHtml(run.id)}" aria-label="Remove ${escapeHtml(run.name)}">×</button>
    </div>`).join("");
}

function renderMetrics(run) {
  const cards = [
    ["Technical score", "recommended_technical_score", "Weighted competition score", "#39f0cf", true],
    ["Hit Rate@10", "hit_rate_at_10", "Target found in Top 10", "#5da8ff"],
    ["MRR", "mrr", "Ranking quality", "#9f7aea"],
    ["Efficiency", "efficiency", "Turn efficiency", "#f5bd5b"],
    ["MTTC", "mttc", "Mean turns to completion", "#ff8c5a"],
    ["Samples", "sampleCount", "Evaluated sessions", "#ff4f8b"],
  ];
  elements.metricGrid.innerHTML = cards.map(([label, key, detail, color, primary]) => {
    const value = key === "sampleCount" ? run.sampleCount : run.metrics[key];
    const displayed = key === "sampleCount" ? (value ?? "—") : metricDisplay(key, value);
    return `<div class="metric-card ${primary ? "primary" : ""}" style="--metric-color:${color}"><span class="metric-label">${label}</span><strong class="metric-value">${displayed}</strong><span class="metric-detail">${detail}</span></div>`;
  }).join("");
}

function renderComparison() {
  const sorted = [...state.runs].sort((a, b) => (b.metrics.recommended_technical_score ?? -1) - (a.metrics.recommended_technical_score ?? -1));
  const maxScore = Math.max(.001, ...sorted.map((run) => run.metrics.recommended_technical_score || 0));
  elements.comparison.innerHTML = sorted.map((run) => {
    const score = run.metrics.recommended_technical_score;
    const width = isNumber(score) ? Math.max(1, (score / maxScore) * 100) : 0;
    return `<div class="comparison-row comparison-grid" style="--run-color:${run.color}">
      <div class="comparison-name"><i></i><button data-select-id="${escapeHtml(run.id)}" title="Select ${escapeHtml(run.name)}">${escapeHtml(run.name)}</button></div>
      <div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong class="bar-number">${decimal(score)}</strong></div>
      <span class="comparison-stat">${percent(run.metrics.hit_rate_at_10)}</span>
      <span class="comparison-stat">${decimal(run.metrics.mrr)}</span>
      <span class="comparison-stat">${decimal(run.metrics.mttc, 2)}</span>
    </div>`;
  }).join("");
}

function renderScenarios(run) {
  const key = elements.scenarioMetric.value;
  const entries = Object.entries(run.scenarios);
  if (!entries.length) {
    elements.scenarioChart.innerHTML = '<div class="table-empty">No scenario metrics in this report.</div>';
    return;
  }
  const maximum = key === "mttc" ? Math.max(11, ...entries.map(([, metrics]) => metrics[key] || 0)) : 1;
  elements.scenarioChart.innerHTML = entries.map(([name, metrics]) => {
    const value = metrics[key];
    const width = isNumber(value) ? Math.max(1, Math.min(100, value / maximum * 100)) : 0;
    return `<div class="scenario-row"><div class="scenario-label">${escapeHtml(name.replaceAll("_", " "))}<small>${metrics.sample_count ?? "—"} samples</small></div><div class="scenario-track"><div class="scenario-fill" style="width:${width}%"></div></div><strong class="scenario-value">${metricDisplay(key, value)}</strong></div>`;
  }).join("");
}

function renderRankChart(run) {
  if (!run.sessions.length) {
    elements.rankChart.innerHTML = '<div class="table-empty">No session data available.</div>';
    return;
  }
  const counts = Array.from({length: 11}, () => 0);
  run.sessions.forEach((session) => {
    const rank = Number(session.best_rank);
    if (session.hit && rank >= 1 && rank <= 10) counts[rank - 1] += 1;
    else counts[10] += 1;
  });
  const max = Math.max(1, ...counts);
  elements.rankChart.innerHTML = counts.map((count, index) => {
    const height = count ? Math.max(3, count / max * 100) : 1;
    return `<div class="rank-column ${index === 10 ? "miss" : ""}" title="${index === 10 ? "Miss" : `Rank ${index + 1}`}: ${count}"><span class="rank-count">${count}</span><div class="rank-bar-wrap"><div class="rank-bar" style="height:${height}%"></div></div><span class="rank-name">${index === 10 ? "Miss" : index + 1}</span></div>`;
  }).join("");
}

function resetScenarioFilter(run) {
  const previous = elements.scenarioFilter.value;
  const names = [...new Set(run.sessions.map((session) => session.scenario_type).filter(Boolean))].sort();
  elements.scenarioFilter.innerHTML = '<option value="all">All scenarios</option>' + names.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name.replaceAll("_", " "))}</option>`).join("");
  if (names.includes(previous)) elements.scenarioFilter.value = previous;
}

function renderSessions(run, resetFilter = false) {
  if (resetFilter) resetScenarioFilter(run);
  const query = elements.sessionSearch.value.trim().toLowerCase();
  const scenario = elements.scenarioFilter.value;
  const outcome = elements.outcomeFilter.value;
  const filtered = run.sessions.filter((session) => {
    if (query && !String(session.sample_id || "").toLowerCase().includes(query)) return false;
    if (scenario !== "all" && session.scenario_type !== scenario) return false;
    if (outcome === "hit" && !session.hit) return false;
    if (outcome === "miss" && session.hit) return false;
    return true;
  });
  const displayed = filtered.slice(0, 250);
  elements.sessionCount.textContent = run.sessions.length ? `${filtered.length} of ${run.sessions.length} sessions${filtered.length > 250 ? " · showing first 250" : ""}` : "No session detail";
  elements.sessionTable.innerHTML = displayed.map((session) => `
    <tr><td>${escapeHtml(session.sample_id || "—")}</td><td><span class="scenario-badge">${escapeHtml(String(session.scenario_type || "unknown").replaceAll("_", " "))}</span></td>
    <td><span class="outcome-badge ${session.hit ? "hit" : "miss"}">${session.hit ? "Hit" : "Miss"}</span></td><td>${session.first_hit_turn ?? "—"}</td><td>${session.best_rank ?? "—"}</td><td>${decimal(session.reciprocal_rank)}</td></tr>`).join("");
  elements.noSessions.hidden = displayed.length > 0;
}

function render() {
  renderTabs();
  const run = activeRun();
  const hasRuns = Boolean(run);
  elements.empty.hidden = hasRuns;
  elements.content.hidden = !hasRuns;
  if (!run) return;
  elements.selectedName.textContent = run.name;
  elements.selectedSource.textContent = run.source;
  renderMetrics(run);
  renderComparison();
  renderScenarios(run);
  renderRankChart(run);
  renderSessions(run, true);
}

let toastTimer;
function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 2800);
}

function chooseRun(id) { state.activeId = id; render(); }
function removeRun(id) {
  state.runs = state.runs.filter((run) => run.id !== id);
  if (state.activeId === id) state.activeId = state.runs[0]?.id || null;
  render();
}

elements.loadReport.addEventListener("click", () => {
  const report = state.reports[Number(elements.reportSelect.value)];
  if (report) loadReport(report);
});
elements.reportSelect.addEventListener("change", () => {
  const report = state.reports[Number(elements.reportSelect.value)];
  if (report) elements.reportHint.textContent = `${report.path} · ${report.run_count} compatible run${report.run_count === 1 ? "" : "s"}`;
});
$("#upload-button").addEventListener("click", () => elements.fileInput.click());
$("#empty-upload").addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => { importFiles(elements.fileInput.files); elements.fileInput.value = ""; });
elements.runTabs.addEventListener("click", (event) => {
  const remove = event.target.closest("[data-remove-id]");
  if (remove) { event.stopPropagation(); removeRun(remove.dataset.removeId); return; }
  const tab = event.target.closest("[data-run-id]");
  if (tab) chooseRun(tab.dataset.runId);
});
elements.comparison.addEventListener("click", (event) => {
  const target = event.target.closest("[data-select-id]");
  if (target) chooseRun(target.dataset.selectId);
});
elements.scenarioMetric.addEventListener("change", () => renderScenarios(activeRun()));
[elements.sessionSearch, elements.scenarioFilter, elements.outcomeFilter].forEach((element) => element.addEventListener("input", () => renderSessions(activeRun())));

let dragDepth = 0;
window.addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth += 1; elements.dropOverlay.hidden = false; });
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth -= 1; if (dragDepth <= 0) { dragDepth = 0; elements.dropOverlay.hidden = true; } });
window.addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; elements.dropOverlay.hidden = true; importFiles(event.dataTransfer.files); });

discoverReports();
