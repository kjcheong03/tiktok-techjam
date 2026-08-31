"use strict";

const COLORS = ["#39f0cf", "#ff4f8b", "#9f7aea", "#f5bd5b", "#5da8ff", "#ff8c5a", "#7bdd65", "#d786ff"];
const SCORE_KEYS = ["recommended_technical_score", "hit_rate_at_10", "mrr", "mttc", "efficiency"];
const SYSTEM_DISPLAY_NAMES = {
  A_official_stateless_bm25: "Organizer BM25 Starter",
  C_fixed_adaptive_architecture: "Fixed Adaptive Architecture",
  GhostLab_Challenger: "GhostLab Champion",
};
const SYSTEM_SLOT_NAMES = {
  A: "Organizer BM25 Starter",
  C: "Fixed Adaptive Architecture",
  D: "GhostLab Champion",
};
const SYSTEM_COLORS = { A: "#39f0cf", C: "#9f7aea", D: "#f5bd5b" };

const state = { reports: [], runs: [], activeId: null, comparisonMeta: null, selectedChallengerId: null };
const $ = (selector) => document.querySelector(selector);
const elements = {
  fileInput: $("#file-input"), content: $("#dashboard-content"), selectedName: $("#selected-run-name"),
  selectedSource: $("#selected-run-source"), metricGrid: $("#metric-grid"), comparison: $("#comparison-chart"),
  scenarioMetric: $("#scenario-metric"), scenarioChart: $("#scenario-chart"), rankChart: $("#rank-chart"),
  distributionLabel: $("#distribution-label"), dropOverlay: $("#drop-overlay"), toast: $("#toast"),
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
  if (!isNumber(metrics.efficiency) && isNumber(metrics.normalized_efficiency)) metrics.efficiency = metrics.normalized_efficiency;
  if (!isNumber(metrics.efficiency) && isNumber(metrics.mttc)) metrics.efficiency = Math.max(0, Math.min(1, (11 - metrics.mttc) / 10));
  metrics.recommended_technical_score = deriveScore(metrics);
  return metrics;
}

function normalizeRun(value, fallbackName, source, suffix = "run", context = {}) {
  const metricSource = value.metrics && typeof value.metrics === "object" ? value.metrics : value;
  const metrics = normalizeScenario(metricSource);
  const sessions = Array.isArray(value.sessions) ? value.sessions : Array.isArray(metricSource.sessions) ? metricSource.sessions : [];
  const scenarioSource = metricSource.scenario_metrics || value.scenario_metrics || {};
  const scenarios = Object.fromEntries(Object.entries(scenarioSource).map(([name, data]) => [name, normalizeScenario(data)]));
  const configuredName = value.system_id || value.experiment_id || value.name || value.candidate_id || value.config?.experiment_id;
  const name = SYSTEM_DISPLAY_NAMES[value.system_id] || configuredName || fallbackName;
  const sampleCount = value.sample_count ?? metricSource.sample_count ?? context.sampleCount ?? sessions.length ?? null;
  return {
    id: `${source}::${suffix}`,
    name: String(name).replaceAll("_", " "), source, metrics, scenarios, sessions,
    sampleCount: isNumber(sampleCount) ? sampleCount : null,
    systemId: value.system_id || null,
    role: value.role || "unclassified",
    championEligible: value.champion_eligible === true,
    partition: value.partition || context.partition || null,
    holdoutAccessed: value.holdout_accessed ?? context.holdoutAccessed ?? null,
    note: value.note || null,
    color: COLORS[0],
  };
}

function extractReport(payload, fallbackName, source) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return { runs: [], meta: null };
  if (Array.isArray(payload.systems)) {
    const semantics = payload.comparison_semantics || {};
    const context = {
      partition: payload.evaluation_partition || (String(payload.evaluation_scope || "").includes("holdout") ? "holdout" : null),
      sampleCount: payload.sample_count ?? payload.holdout_sample_count ?? null,
      holdoutAccessed: payload.final_selection_accessed ?? payload.holdout_accessed ?? String(payload.evaluation_scope || "").includes("holdout"),
    };
    const runs = payload.systems.filter(hasMetrics).map((system, index) => normalizeRun(
      system, system.system_id || `${fallbackName} · system ${index + 1}`, source,
      `system-${system.system_id || index}`, context,
    ));
    return {
      runs,
      meta: {
        fair: semantics.same_ground === true,
        partition: context.partition,
        sampleCount: context.sampleCount,
        holdoutAccessed: context.holdoutAccessed,
        championScope: semantics.champion_selection_scope || "C versus D only",
        sameEvaluator: semantics.same_evaluator_contract === true,
        sameOrderedIds: semantics.same_ordered_session_ids === true,
        decision: payload.decision || null,
        gatesPassed: payload.all_gates_passed ?? null,
        source,
        challengerIds: runs.filter((run) => run.role.includes("challenger")).map((run) => run.id),
        selectedSystemId: payload.selected_system_id || null,
      },
    };
  }
  if (hasMetrics(payload)) return { runs: [normalizeRun(payload, fallbackName, source)], meta: null };
  if (Array.isArray(payload.records)) {
    const runs = payload.records.filter(hasMetrics).map((record, index) => normalizeRun(
      record,
      `${fallbackName} · candidate ${record.ordinal ?? index + 1}`,
      source,
      `record-${record.ordinal ?? index}`,
    ));
    if (runs.length) return { runs, meta: null };
  }
  return {
    runs: Object.entries(payload).filter(([, value]) => hasMetrics(value)).map(([key, value]) =>
      normalizeRun(value, key, source, `key-${key}`)),
    meta: null,
  };
}

function addRuns(runs, meta = null) {
  if (meta) {
    state.runs = [];
    state.activeId = null;
    state.comparisonMeta = meta;
    state.selectedChallengerId = runs.find((run) => run.systemId === meta.selectedSystemId)?.id
      || meta.challengerIds?.[0] || null;
  } else if (state.comparisonMeta) {
    state.comparisonMeta = null;
    state.selectedChallengerId = null;
  }
  let added = 0;
  for (const run of runs) {
    if (state.runs.some((existing) => existing.id === run.id)) continue;
    run.color = SYSTEM_COLORS[run.systemId] || COLORS[state.runs.length % COLORS.length];
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
    const extracted = extractReport(payload, report.label, report.path);
    let runs = extracted.runs;
    if (report.system_id) runs = runs.filter((run) => run.systemId === report.system_id);
    if (report.run_key) runs = runs.filter((run) => run.id.endsWith(`::key-${report.run_key}`));
    if (!runs.length) throw new Error("No evaluation metrics were found for this model");
    if (report.model_id) {
      const stableId = `model::${report.model_id}`;
      const wasActive = state.activeId === stableId;
      state.runs = state.runs.filter((run) => run.id !== stableId);
      runs = [runs[0]].map((run) => ({
        ...run,
        id: stableId,
        name: SYSTEM_SLOT_NAMES[report.model_id] || report.label,
        systemId: report.model_id,
        role: report.role,
        championEligible: report.champion_eligible === true,
      }));
      const added = addRuns(runs);
      if (wasActive) {
        state.activeId = stableId;
        render();
      }
      if (!quiet) showToast(`Loaded ${report.label}`);
      return added;
    }
    const added = addRuns(runs, extracted.meta);
    if (!quiet) showToast(added ? `Loaded ${added} run${added === 1 ? "" : "s"}` : "Those runs are already loaded");
    return added;
  } catch (error) {
    showToast(`Could not load model: ${error.message}`);
    return 0;
  }
}

async function discoverReports() {
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) throw new Error("dashboard server unavailable");
    const payload = await response.json();
    state.reports = payload.models || [];
    const defaults = state.reports.filter((report) => Object.hasOwn(SYSTEM_SLOT_NAMES, report.model_id));
    for (const report of defaults) await loadReport(report, true);
    const champion = state.runs.find((run) => run.systemId === "D");
    if (champion) state.activeId = champion.id;
    render();
    if (!state.runs.length) showToast("No default evaluation reports were found");
  } catch (error) {
    showToast(`Could not load default systems: ${error.message}. Import JSON is still available.`);
  }
}

async function importFiles(files) {
  let total = 0;
  for (const file of files) {
    try {
      const payload = JSON.parse(await file.text());
      const extracted = extractReport(payload, file.name.replace(/\.json$/i, ""), file.name);
      if (!extracted.runs.length) throw new Error("no recognized evaluation metrics");
      total += addRuns(extracted.runs, extracted.meta);
    } catch (error) {
      showToast(`${file.name}: ${error.message}`);
    }
  }
  if (total) showToast(`Imported ${total} run${total === 1 ? "" : "s"}`);
}

function isChallenger(run) { return run.role.includes("challenger"); }
function displayedRuns() {
  if (!state.comparisonMeta) return state.runs;
  const pinned = state.runs.filter((run) => !isChallenger(run));
  const selected = state.runs.find((run) => run.id === state.selectedChallengerId);
  return selected ? [...pinned, selected] : pinned;
}
function activeRun() {
  const visible = displayedRuns();
  return visible.find((run) => run.id === state.activeId) || visible[0];
}
function percent(value) { return isNumber(value) ? `${(value * 100).toFixed(1)}%` : "—"; }
function decimal(value, digits = 3) { return isNumber(value) ? value.toFixed(digits) : "—"; }
function metricDisplay(key, value) {
  if (["hit_rate_at_10", "efficiency"].includes(key)) return percent(value);
  if (key === "mttc") return isNumber(value) ? `${value.toFixed(2)}` : "—";
  return decimal(value);
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
  const visible = displayedRuns();
  const sorted = state.comparisonMeta ? visible : [...visible].sort((a, b) => (b.metrics.recommended_technical_score ?? -1) - (a.metrics.recommended_technical_score ?? -1));
  const maxScore = Math.max(.001, ...sorted.map((run) => run.metrics.recommended_technical_score || 0));
  elements.comparison.innerHTML = sorted.map((run) => {
    const score = run.metrics.recommended_technical_score;
    const width = isNumber(score) ? Math.max(1, (score / maxScore) * 100) : 0;
    const roleClass = run.role.includes("reference") || run.role.includes("baseline") ? "reference" : run.role.includes("control") ? "control" : "challenger";
    const roleSuffix = run.championEligible ? "eligible" : "reference";
    const roleBadge = state.comparisonMeta
      ? `<span class="role-badge ${roleClass}">${escapeHtml(run.role.replaceAll("_", " "))} · ${roleSuffix}</span>`
      : "";
    return `<div class="comparison-row comparison-grid" style="--run-color:${run.color}">
      <div class="comparison-name"><i></i><div class="comparison-identity"><button data-select-id="${escapeHtml(run.id)}" title="Select ${escapeHtml(run.name)}">${escapeHtml(run.name)}</button>${roleBadge}</div></div>
      <div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong class="bar-number">${decimal(score)}</strong></div>
      <span class="comparison-stat">${percent(run.metrics.hit_rate_at_10)}</span>
      <span class="comparison-stat">${decimal(run.metrics.mrr)}</span>
      <span class="comparison-stat">${percent(run.metrics.efficiency)}</span>
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

function render() {
  const run = activeRun();
  const hasRuns = Boolean(run);
  elements.content.hidden = !hasRuns;
  if (!run) return;
  elements.selectedName.textContent = run.name;
  elements.selectedSource.textContent = run.source;
  renderMetrics(run);
  renderComparison();
  renderScenarios(run);
  renderRankChart(run);
}

let toastTimer;
function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 2800);
}

function chooseRun(id) { state.activeId = id; render(); }
$("#upload-button").addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => { importFiles(elements.fileInput.files); elements.fileInput.value = ""; });
elements.comparison.addEventListener("click", (event) => {
  const target = event.target.closest("[data-select-id]");
  if (target) chooseRun(target.dataset.selectId);
});
elements.scenarioMetric.addEventListener("change", () => renderScenarios(activeRun()));

let dragDepth = 0;
window.addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth += 1; elements.dropOverlay.hidden = false; });
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth -= 1; if (dragDepth <= 0) { dragDepth = 0; elements.dropOverlay.hidden = true; } });
window.addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; elements.dropOverlay.hidden = true; importFiles(event.dataTransfer.files); });

discoverReports();
