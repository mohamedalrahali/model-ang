const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function appendJsonDetails(container, data, summaryLabel = "JSON brut") {
  const det = document.createElement("details");
  det.className = "gmm-raw";
  const sum = document.createElement("summary");
  sum.textContent = summaryLabel;
  det.appendChild(sum);
  const pre = document.createElement("pre");
  pre.className = "gmm-json-pre";
  pre.textContent = JSON.stringify(data, null, 2);
  det.appendChild(pre);
  container.appendChild(det);
}

function showSection(id) {
  $$(".section-hidden").forEach((s) => s.classList.remove("is-active"));
  const target = document.getElementById(id);
  if (target) target.classList.add("is-active");
  $$(".nav a").forEach((a) => a.classList.remove("active"));
  const link = document.querySelector(`.nav a[data-section="${id}"]`);
  if (link) link.classList.add("active");
  if (id === "dash") loadDashboard();
  if (id === "gmm") {
    const host = $("#gmm-fields");
    const n = host?.querySelectorAll('input[id^="gmm_"]').length ?? 0;
    if (n < 8) void initGmmForm();
  }
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  const text = await r.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!r.ok) throw new Error(data.detail || data.message || text || r.statusText);
  return data;
}

async function loadDashboard() {
  try {
    const h = await api("/api/health");
    $("#stat-perf").textContent = h.performance?.loaded ? "Ready" : "—";
    $("#stat-eng").textContent = h.engagement?.loaded ? "Ready" : "—";
    $("#stat-gmm").textContent = h.gmm?.loaded ? "Ready" : "—";
    $("#stat-perf").className = "value " + (h.performance?.loaded ? "" : "muted");
    $("#badge-perf").innerHTML = h.performance?.loaded
      ? '<span class="badge badge-ok">OK</span>'
      : '<span class="badge badge-off">Missing</span>';
    $("#badge-eng").innerHTML = h.engagement?.loaded
      ? '<span class="badge badge-ok">OK</span>'
      : '<span class="badge badge-off">Missing</span>';
    $("#badge-gmm").innerHTML = h.gmm?.loaded
      ? '<span class="badge badge-ok">OK</span>'
      : '<span class="badge badge-off">Missing</span>';
    $("#dash-errors").textContent = [
      h.performance?.error,
      h.engagement?.error,
      h.gmm?.error,
    ]
      .filter(Boolean)
      .join("\n");
  } catch (e) {
    $("#dash-errors").textContent = String(e.message || e);
  }
}

function buildPerformanceForm(schema) {
  const host = $("#perf-fields");
  const targetLine = $("#perf-target-line");
  host.innerHTML = "";
  const names = schema.feature_names || [];
  const fields = schema.fields || [];

  if (!names.length) {
    targetLine.textContent = "";
    host.innerHTML =
      '<p class="empty-hint">No schema loaded. Run <code>python scripts/bootstrap_demo.py</code> or copy <code>artifacts/best_model_meta.json</code> from notebook 3.</p>';
    return;
  }

  const tgt = schema.target || "Final_Performance_Score";
  targetLine.innerHTML = `Target column: <code>${tgt}</code> — provide all <strong>${names.length}</strong> numeric features below.`;

  const byName = Object.fromEntries(fields.map((f) => [f.name, f]));
  for (const name of names) {
    const meta = byName[name] || {};
    const id = `pf_${name}`;
    const wrap = document.createElement("div");
    wrap.className = "field-block";
    const hintParts = [meta.description, meta.typical_range ? `Typical: ${meta.typical_range}` : ""]
      .filter(Boolean)
      .join(" ");
    wrap.innerHTML = `
      <span class="fname">${name}</span>
      ${hintParts ? `<small class="field-hint">${hintParts}</small>` : ""}
      <input type="number" step="any" id="${id}" value="60" />
    `;
    host.appendChild(wrap);
  }
}

async function initPerformanceForm() {
  try {
    const s = await api("/api/schema/performance");
    buildPerformanceForm(s);
  } catch {
    buildPerformanceForm({ feature_names: [] });
  }
}

const ENGAGEMENT_SCHEMA_FALLBACK_HTML = `
<strong>Required API / form fields</strong><br/>
<strong>TimeSpentOnCourse</strong> — Time on course (dataset units). <em>(number ≥ 0)</em><br/>
<strong>NumberOfVideosWatched</strong> — Count of videos watched. <em>(number ≥ 0)</em><br/>
<strong>NumberOfQuizzesTaken</strong> — Count of quizzes taken. <em>(number ≥ 0)</em><br/>
<strong>QuizScores</strong> — Quiz score. <em>(0–100)</em><br/>
<strong>CompletionRate</strong> — Completion rate. <em>(0–100 %)</em><br/>
<strong>DeviceType</strong> — Device flag. <em>(0 or 1)</em><br/>
<strong>CourseCategory</strong> — One of: Arts, Business, Health, Programming, Science.<br/>
<br/><em>If the live schema request returns 404, your uvicorn process is an old build: stop it and run <code>python serve.py</code> from the <code>ml-web</code> project folder.</em>
`;

async function initEngagementReference() {
  const el = $("#eng-field-list");
  if (!el) return;
  const url = new URL("/api/schema/engagement", window.location.origin).href;
  try {
    const r = await fetch(url);
    if (r.status === 404) {
      el.innerHTML = ENGAGEMENT_SCHEMA_FALLBACK_HTML;
      return;
    }
    if (!r.ok) throw new Error(await r.text());
    const s = await r.json();
    const lines = (s.inputs || [])
      .map((i) => `<strong>${i.name}</strong> — ${i.description} <em>(${i.constraints})</em>`)
      .join("<br/>");
    const cats = (s.allowed_course_categories || []).join(", ");
    el.innerHTML = `<strong>Required API / form fields</strong><br/>${lines}<br/><br/>Allowed <code>CourseCategory</code> values: <code>${cats}</code>`;
  } catch {
    el.innerHTML = ENGAGEMENT_SCHEMA_FALLBACK_HTML;
  }
}

function renderPerformanceResult(res) {
  const p = res.properties;
  if (!p) return `<pre>${escapeHtml(JSON.stringify(res, null, 2))}</pre>`;
  const pred = p.prediction || {};
  const rows = (p.input_profile || [])
    .map(
      (r) => `<tr>
      <td>${escapeHtml(r.feature_label_fr || r.feature)}</td>
      <td><code>${escapeHtml(String(r.value))}</code></td>
      <td>${
        r.level === "na"
          ? `<span class="field-hint">${escapeHtml(r.level_label_fr)}</span>`
          : `<span class="level-pill ${escapeHtml(r.level)}">${escapeHtml(r.level_label_fr)}</span> <span class="field-hint" style="display:inline">(${escapeHtml(r.level_label_en)})</span>`
      }</td>
      <td class="field-hint" style="font-size:0.78rem">${escapeHtml(r.scale_note_fr || "")}</td>
    </tr>`
    )
    .join("");
  return `
    <div class="gmm-summary">${escapeHtml(p.summary_fr || "")}</div>
    <p class="field-hint" style="margin:0 0 0.75rem">${escapeHtml(p.summary_en || "")}</p>
    <div class="gmm-section-title">Propriétés des variables saisies</div>
    <table class="gmm-profile-table">
      <thead><tr><th>Variable</th><th>Valeur</th><th>Niveau</th><th>Note</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="gmm-section-title">Score prédit</div>
    <p><strong>${escapeHtml(String(res.predicted_Final_Performance_Score))}</strong>
      <span class="level-pill ${escapeHtml(pred.level || "medium")}">${escapeHtml(pred.level_label_fr || "")}</span>
      <span class="field-hint">(${escapeHtml(pred.level_label_en || "")})</span>
    </p>
  `;
}

async function submitPerformance() {
  const out = $("#perf-result");
  out.className = "result-box";
  out.textContent = "Running…";
  const names = [...$$("#perf-fields input")].map((i) => i.id.replace("pf_", ""));
  const features = {};
  for (const name of names) {
    const inp = document.getElementById(`pf_${name}`);
    if (inp) features[name] = parseFloat(inp.value);
  }
  try {
    const res = await api("/api/predict/performance", {
      method: "POST",
      body: JSON.stringify({ features }),
    });
    out.className = "result-box gmm-rich";
    out.innerHTML = renderPerformanceResult(res);
    appendJsonDetails(out, res);
  } catch (e) {
    out.className = "result-box error";
    out.textContent = e.message || String(e);
  }
}

function renderEngagementResult(res) {
  const p = res.properties;
  if (!p) return `<pre>${escapeHtml(JSON.stringify(res, null, 2))}</pre>`;
  const pr = p.prediction || {};
  const rows = (p.learner_profile || [])
    .map(
      (r) => `<tr>
      <td>${escapeHtml(r.field_label_fr || r.field)}</td>
      <td><code>${escapeHtml(String(r.value))}</code></td>
      <td>${
        r.level === "na"
          ? `<span class="field-hint">${escapeHtml(r.level_label_fr)}</span>`
          : `<span class="level-pill ${escapeHtml(r.level)}">${escapeHtml(r.level_label_fr)}</span> <span class="field-hint" style="display:inline">(${escapeHtml(r.level_label_en)})</span>`
      }</td>
      <td class="field-hint" style="font-size:0.78rem">${escapeHtml(r.hint_fr || "")}</td>
    </tr>`
    )
    .join("");
  const cls = res.CourseCompletion_predicted === 1 ? "high" : "low";
  return `
    <div class="gmm-summary">${escapeHtml(p.summary_fr || "")}</div>
    <p class="field-hint" style="margin:0 0 0.75rem">${escapeHtml(p.summary_en || "")}</p>
    <div class="gmm-section-title">Propriétés du profil saisi</div>
    <table class="gmm-profile-table">
      <thead><tr><th>Indicateur</th><th>Valeur</th><th>Niveau</th><th>Info</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="gmm-section-title">Décision du modèle</div>
    <p>Classe prédite : <strong>${escapeHtml(String(res.CourseCompletion_predicted))}</strong>
      <span class="level-pill ${cls}">${res.CourseCompletion_predicted === 1 ? "complété" : "non complété"}</span>
      · P(complété)=<strong>${escapeHtml(String(pr.probability_completed_percent))}%</strong>
      · P(non complété)=<strong>${escapeHtml(String(pr.probability_not_completed_percent))}%</strong>
    </p>
    <p class="field-hint">${escapeHtml(pr.certainty_fr || "")} / ${escapeHtml(pr.certainty_en || "")}</p>
  `;
}

async function submitEngagement() {
  const out = $("#eng-result");
  out.className = "result-box";
  out.textContent = "Running…";
  const body = {
    TimeSpentOnCourse: parseFloat($("#eng_time").value),
    NumberOfVideosWatched: parseFloat($("#eng_vid").value),
    NumberOfQuizzesTaken: parseFloat($("#eng_quiz").value),
    QuizScores: parseFloat($("#eng_score").value),
    CompletionRate: parseFloat($("#eng_comp").value),
    DeviceType: parseInt($("#eng_dev").value, 10),
    CourseCategory: $("#eng_cat").value,
  };
  try {
    const res = await api("/api/predict/engagement", {
      method: "POST",
      body: JSON.stringify(body),
    });
    out.className = "result-box gmm-rich";
    out.innerHTML = renderEngagementResult(res);
    appendJsonDetails(out, res);
  } catch (e) {
    out.className = "result-box error";
    out.textContent = e.message || String(e);
  }
}

function renderRecommendResult(res) {
  const p = res.properties || {};
  const recs = res.recommendations || [];
  const rows = recs
    .map((r, i) => {
      const pct = Math.round((r.completion_probability || 0) * 10000) / 100;
      const w = Math.min(100, Math.max(0, pct));
      const role = i === 0 ? "meilleur parcours" : `rang ${i + 1}`;
      return `<div class="gmm-seg-row">
        <span style="min-width:6rem">${escapeHtml(r.category)}</span>
        <div class="gmm-seg-bar" title="${pct}%"><span style="width:${w}%"></span></div>
        <span style="min-width:3.5rem;text-align:right">${escapeHtml(String(pct))}%</span>
        <span class="field-hint">${escapeHtml(role)}</span>
      </div>`;
    })
    .join("");
  const profRows = (p.learner_profile || [])
    .map(
      (r) => `<tr>
      <td>${escapeHtml(r.field_label_fr || r.field)}</td>
      <td><code>${escapeHtml(String(r.value))}</code></td>
      <td>${
        r.level === "na"
          ? `<span class="field-hint">${escapeHtml(r.level_label_fr)}</span>`
          : `<span class="level-pill ${escapeHtml(r.level)}">${escapeHtml(r.level_label_fr)}</span>`
      }</td>
    </tr>`
    )
    .join("");
  return `
    <div class="gmm-summary">${escapeHtml(p.summary_fr || "")}</div>
    <p class="field-hint" style="margin:0 0 0.5rem">${escapeHtml(p.summary_en || "")}</p>
    <p class="field-hint" style="margin-bottom:0.75rem">${escapeHtml(p.note_fr || "")}<br/>${escapeHtml(p.note_en || "")}</p>
    <div class="gmm-section-title">Propriétés du profil (commun à toutes les catégories)</div>
    <table class="gmm-profile-table">
      <thead><tr><th>Indicateur</th><th>Valeur</th><th>Niveau</th></tr></thead>
      <tbody>${profRows}</tbody>
    </table>
    <div class="gmm-section-title">Classement des catégories (probabilité de complétion)</div>
    <div class="gmm-segments">${rows}</div>
  `;
}

async function submitRecommend() {
  const out = $("#reco-result");
  out.className = "result-box";
  out.textContent = "Running…";
  const body = {
    TimeSpentOnCourse: parseFloat($("#reco_time").value),
    NumberOfVideosWatched: parseFloat($("#reco_vid").value),
    NumberOfQuizzesTaken: parseFloat($("#reco_quiz").value),
    QuizScores: parseFloat($("#reco_score").value),
    CompletionRate: parseFloat($("#reco_comp").value),
    DeviceType: parseInt($("#reco_dev").value, 10),
    top_k: parseInt($("#reco_k").value, 10),
  };
  try {
    const res = await api("/api/recommend/categories", {
      method: "POST",
      body: JSON.stringify(body),
    });
    out.className = "result-box gmm-rich";
    out.innerHTML = renderRecommendResult(res);
    appendJsonDetails(out, res);
  } catch (e) {
    out.className = "result-box error";
    out.textContent = e.message || String(e);
  }
}

function buildGmmFormFromNames(names) {
  const host = $("#gmm-fields");
  host.innerHTML = "";
  for (const name of names) {
    const wrap = document.createElement("div");
    wrap.className = "field-block";
    wrap.innerHTML = `
      <span class="fname">${name}</span>
      <small class="field-hint">Numeric score (see GMM notebook). Typical 0–100.</small>
      <input type="number" step="any" id="gmm_${name}" value="65" />
    `;
    host.appendChild(wrap);
  }
}

async function initGmmForm() {
  const host = $("#gmm-fields");
  host.innerHTML = "";
  const url = new URL("/api/schema/gmm", window.location.origin).href;
  try {
    const r = await fetch(url);
    if (r.status === 404) {
      buildGmmFormFromNames([
        "Reading_Comprehension_Score",
        "Listening_Accuracy",
        "Writing_Score",
        "Speaking_Score",
        "Engagement_Level",
        "Confidence_Rating",
        "Task_Difficulty",
        "Reward_Signal",
      ]);
      return;
    }
    if (!r.ok) throw new Error(await r.text());
    const s = await r.json();
    const fields = Array.isArray(s.fields) ? s.fields : [];
    if (fields.length > 0) {
      for (const f of fields) {
        if (!f || !f.name) continue;
        const wrap = document.createElement("div");
        wrap.className = "field-block";
        const hint = [f.description, f.typical_range ? `Typical: ${f.typical_range}` : ""]
          .filter(Boolean)
          .join(" ");
        const safeHint = String(hint).replace(/</g, "&lt;");
        const safeName = String(f.name).replace(/</g, "&lt;");
        wrap.innerHTML = `
        <span class="fname">${safeName}</span>
        <small class="field-hint">${safeHint}</small>
        <input type="number" step="any" id="gmm_${f.name}" value="65" />
      `;
        host.appendChild(wrap);
      }
    }
    if (host.querySelectorAll('input[id^="gmm_"]').length < 8) {
      const fallbackNames = Array.isArray(s.feature_names) && s.feature_names.length ? s.feature_names : null;
      buildGmmFormFromNames(
        fallbackNames || [
          "Reading_Comprehension_Score",
          "Listening_Accuracy",
          "Writing_Score",
          "Speaking_Score",
          "Engagement_Level",
          "Confidence_Rating",
          "Task_Difficulty",
          "Reward_Signal",
        ]
      );
    }
  } catch {
    buildGmmFormFromNames([
      "Reading_Comprehension_Score",
      "Listening_Accuracy",
      "Writing_Score",
      "Speaking_Score",
      "Engagement_Level",
      "Confidence_Rating",
      "Task_Difficulty",
      "Reward_Signal",
    ]);
  }
}

function renderGmmResult(res) {
  const p = res.properties;
  if (!p) {
    return `<pre>${escapeHtml(JSON.stringify(res, null, 2))}</pre>`;
  }
  const profRows = (p.learner_profile || [])
    .map(
      (row) => `<tr>
      <td>${escapeHtml(row.feature_label_fr || row.feature)}</td>
      <td><code>${escapeHtml(String(row.value))}</code></td>
      <td><span class="level-pill ${escapeHtml(row.level)}">${escapeHtml(row.level_label_fr)}</span> <span class="field-hint" style="display:inline">(${escapeHtml(row.level_label_en)})</span></td>
    </tr>`
    )
    .join("");

  const asg = p.assignment || {};
  const segRows = (p.segments_ranked || [])
    .slice(0, 8)
    .map((s) => {
      const w = Math.min(100, Math.max(0, s.probability_percent || 0));
      return `<div class="gmm-seg-row">
        <span style="min-width:7rem">#${s.cluster}</span>
        <div class="gmm-seg-bar" title="${escapeHtml(String(s.probability_percent))}%"><span style="width:${w}%"></span></div>
        <span style="min-width:3.5rem;text-align:right">${escapeHtml(String(s.probability_percent))}%</span>
        <span class="field-hint">${escapeHtml(s.role_fr || "")}</span>
      </div>`;
    })
    .join("");

  return `
    <div class="gmm-summary">${escapeHtml(p.summary_fr || "")}</div>
    <p class="field-hint" style="margin:0 0 0.5rem">${escapeHtml(p.summary_en || "")}</p>

    <div class="gmm-section-title">Propriétés saisies (niveau par compétence)</div>
    <table class="gmm-profile-table">
      <thead><tr><th>Compétence</th><th>Valeur</th><th>Niveau</th></tr></thead>
      <tbody>${profRows}</tbody>
    </table>

    <div class="gmm-section-title">Affectation au mélange (probabilités par segment)</div>
    <div class="gmm-segments">${segRows}</div>
    <p class="field-hint" style="margin-top:0.5rem">
      Segment prédit : <strong>#${escapeHtml(String(asg.predicted_cluster))}</strong>
      · confiance max : <strong>${escapeHtml(String(asg.max_posterior_percent))}%</strong>
      · ${escapeHtml(asg.certainty_fr || "")} / ${escapeHtml(asg.certainty_en || "")}
    </p>

    <p class="gmm-geo-note">${escapeHtml(p.geometry_note_fr || "")}<br/>${escapeHtml(p.geometry_note_en || "")}</p>
  `;
}

async function submitGmm() {
  const out = $("#gmm-result");
  out.className = "result-box";
  out.textContent = "Running…";
  let feats;
  try {
    feats = await api("/api/schema/gmm");
  } catch (e) {
    out.className = "result-box error";
    out.textContent = e.message || String(e);
    return;
  }
  const names = feats.feature_names || [];
  const fromDom = {};
  for (const el of $$('#gmm-fields input[id^="gmm_"]')) {
    const name = el.id.slice(4);
    fromDom[name] = parseFloat(el.value);
  }
  const features = {};
  const missing = [];
  for (const name of names) {
    if (fromDom[name] === undefined || Number.isNaN(fromDom[name])) {
      missing.push(name);
      continue;
    }
    features[name] = fromDom[name];
  }
  if (missing.length) {
    out.className = "result-box error";
    out.textContent =
      `Missing or invalid input for: ${missing.join(", ")}. ` +
      `Open the GMM tab again to refresh the form (need ${names.length} fields).`;
    return;
  }
  try {
    const res = await api("/api/predict/gmm", {
      method: "POST",
      body: JSON.stringify({ features }),
    });
    out.className = "result-box gmm-rich";
    out.innerHTML = renderGmmResult(res);
    appendJsonDetails(out, res);
  } catch (e) {
    out.className = "result-box error";
    out.textContent = e.message || String(e);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $$(".nav a[data-section]").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      showSection(a.dataset.section);
    });
  });

  $("#btn-perf")?.addEventListener("click", submitPerformance);
  $("#btn-eng")?.addEventListener("click", submitEngagement);
  $("#btn-reco")?.addEventListener("click", submitRecommend);
  $("#btn-gmm")?.addEventListener("click", submitGmm);

  initPerformanceForm();
  initGmmForm();
  initEngagementReference();
  showSection("dash");
});
