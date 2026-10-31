// Failure-Aware RAG web UI: plain JavaScript, no build step.
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);

const INJECTIONS = { none: "None", missing: "Missing evidence", poor_ranking: "Poor ranking", noisy: "Noisy context", position: "Position effect", conflict: "Conflict" };
let MODES = {};

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

// ------------------------------------------------------------------ tabs
$$(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
    if (t.dataset.view === "docs") loadDocs();
    if (t.dataset.view === "eval") loadHistory();
  })
);

function chip(container, value, label, on) {
  const el = document.createElement("label");
  el.className = "chip" + (on ? " on" : "");
  el.innerHTML = `<input type="checkbox" value="${esc(value)}" ${on ? "checked" : ""}>${esc(label)}`;
  el.querySelector("input").addEventListener("change", (e) => el.classList.toggle("on", e.target.checked));
  container.appendChild(el);
}
const checked = (container) => $$("input:checked", container).map((i) => i.value);

// ---------------------------------------------------------------- health
async function init() {
  try {
    const h = await api("/api/health");
    MODES = h.modes;
    $("#status").innerHTML = `<b>${esc(h.llm)}</b><br>${esc(h.retriever)} · verifier: ${esc(h.verifier)} · ${h.documents} docs / ${h.chunks} chunks`;
    const sel = $("#mode");
    sel.innerHTML = Object.entries(MODES).filter(([k]) => k !== "oracle").map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    sel.value = "failure_aware";
    ["vanilla", "rerank", "crag_lite", "generic", "random", "llm_only"].forEach((m) => chip($("#compare"), m, MODES[m], m === "vanilla"));
    Object.entries(MODES).forEach(([k, v]) => chip($("#eval-modes"), k, v, ["vanilla", "failure_aware", "random"].includes(k)));
    Object.entries(INJECTIONS).forEach(([k, v]) => chip($("#eval-inj"), k, v, k === "none"));
  } catch (e) {
    $("#status").innerHTML = `<span class="no">Server not reachable: ${esc(e.message)}</span>`;
  }
}

// ------------------------------------------------------------------- ask
$$(".example").forEach((b) => b.addEventListener("click", () => { $("#question").value = b.textContent; ask(); }));
$("#ask-btn").addEventListener("click", ask);
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); });

async function ask() {
  const question = $("#question").value.trim();
  if (question.length < 3) return $("#question").focus();
  const options = {};
  $$(".mcq-grid input").forEach((i) => { if (i.value.trim()) options[i.dataset.opt] = i.value.trim(); });
  const btn = $("#ask-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Running`;
  $("#results").innerHTML = "";
  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: $("#mode").value, options: Object.keys(options).length ? options : null, compare: checked($("#compare")) }),
    });
    data.runs.forEach((r) => $("#results").appendChild(renderRun(r)));
  } catch (e) {
    $("#results").innerHTML = `<div class="error">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

function renderAnswer(text, choice) {
  let html = esc(text).replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, ids) =>
    ids.split(/\s*,\s*/).map((n) => `<span class="cite" data-n="${n}">${n}</span>`).join("")
  );
  if (choice) html += `<div style="margin-top:8px"><b>Selected option: ${esc(choice)}</b></div>`;
  return html;
}

const STEP_ICON = { retrieve: "R", generate: "G", detect: "✓", diagnose: "D", repair: "↻", answer: "✓", abstain: "!" };

function renderRun(r) {
  const node = $("#run-tpl").content.firstElementChild.cloneNode(true);
  node.classList.add(r.status);
  $(".run-mode", node).textContent = r.mode_label;
  const badge = $(".badge", node);
  badge.className = `badge ${r.status}`;
  badge.textContent = { answered: "Answered", repaired: `Repaired (${r.repairs.length})`, abstained: "Abstained" }[r.status];
  $(".run-meta", node).textContent = `${r.latency_ms} ms · ${r.llm_calls} LLM call${r.llm_calls === 1 ? "" : "s"}`;
  $(".answer", node).innerHTML = renderAnswer(r.answer, r.choice);
  $(".notes", node).textContent = r.notes.join(" ");

  // Signals from the last check. In baseline modes they are shown but not acted on.
  const s = r.signals;
  const passive = ["vanilla", "rerank", "crag_lite"].includes(r.mode) && s.failed_checks?.length;
  const sigs = [
    ["Top relevance", pct(s.top_relevance), s.top_relevance < 0.45],
    ["Relevant passages", pct(s.relevant_fraction), s.relevant_fraction < 0.5],
    ["Unsupported claims", pct(s.unsupported_fraction), s.unsupported_fraction > 0],
    ["Conflict", pct(s.conflict), s.conflict >= 0.7],
  ];
  $(".signals", node).innerHTML = sigs.map(([k, v, bad]) => `<div class="sig ${bad ? "bad" : ""}"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("") +
    (passive ? `<div class="sig bad" style="grid-column:1/-1"><div class="k">Checks flagged (not acted on in this baseline)</div><div class="v" style="font-size:13px">${esc(s.failed_checks.join(", "))}</div></div>` : "");

  // Trace timeline
  $(".trace", node).innerHTML = r.trace.map((st) => {
    const pass = st.kind === "detect" ? (st.title === "Checks passed" ? "pass" : "fail") : "";
    const icon = st.kind === "detect" && pass === "fail" ? "✕" : STEP_ICON[st.kind] || "•";
    return `<li class="k-${st.kind} ${pass}"><span class="dot">${icon}</span><div class="t">${esc(st.title)}<span class="ms">${st.t_ms} ms</span></div><div class="d">${esc(st.detail)}</div></li>`;
  }).join("");

  // Claims
  const tbody = $(".claims tbody", node);
  if (!r.claims.length) {
    $(".claims", node).remove();
    $(".claims-h", node).remove();
  } else {
    tbody.innerHTML = r.claims.map((c) => `<tr><td>${esc(c.text)} ${c.passage ? `<span class="cite" data-n="${c.passage}">${c.passage}</span>` : ""}</td>
      <td><div class="bar ${c.supported ? "" : "low"}"><i style="width:${Math.round(c.support * 100)}%"></i></div></td>
      <td>${c.supported ? '<span class="yes">✓</span>' : '<span class="no">✕</span>'}</td></tr>`).join("");
  }

  // Passages
  $(".passages", node).innerHTML = r.passages.length ? r.passages.map((p, i) => `
    <div class="passage" data-n="${i + 1}">
      <div class="passage-head"><span class="n">[${i + 1}]</span><span class="title" title="${esc(p.title)}">${esc(p.title)}</span>
        <span class="tag ${esc(p.authority)}">${esc(p.authority)}</span>
        <div class="bar ${p.relevance < 0.35 ? "low" : ""}" title="relevance ${pct(p.relevance)}"><i style="width:${Math.round(p.relevance * 100)}%"></i></div></div>
      <p>${esc(p.text)}</p>
    </div>`).join("") : `<p class="muted">No passages (closed-book baseline).</p>`;

  $$(".passage p", node).forEach((p) => p.addEventListener("click", () => p.parentElement.classList.toggle("open")));
  $$(".cite", node).forEach((c) => c.addEventListener("click", () => {
    const target = $(`.passage[data-n="${c.dataset.n}"]`, node);
    if (!target) return;
    target.classList.add("open", "flash");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => target.classList.remove("flash"), 1400);
  }));
  return node;
}

// ------------------------------------------------------------- documents
async function loadDocs() {
  const tbody = $("#docs-table tbody");
  try {
    const docs = await api("/api/documents");
    tbody.innerHTML = docs.length ? docs.map((d) => `<tr><td>${esc(d.title)}</td><td class="muted">${esc(d.file || d.doc_id)}</td>
      <td><span class="tag ${esc(d.authority)}">${esc(d.authority)}</span></td><td class="num">${d.chunks}</td><td class="num">${d.words}</td>
      <td class="num"><button class="link" data-del="${esc(d.doc_id)}">Remove</button></td></tr>`).join("")
      : `<tr><td colspan="6" class="muted">No documents yet. Upload some above.</td></tr>`;
    $$("[data-del]", tbody).forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Remove ${b.dataset.del} from the knowledge base? The file is deleted from data/corpus.`)) return;
      await api(`/api/documents/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" });
      loadDocs(); init();
    }));
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6"><div class="error">${esc(e.message)}</div></td></tr>`;
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  $("#drop").innerHTML = `<span class="spinner"></span> Uploading and indexing ${files.length} file(s)…`;
  try {
    await api("/api/documents", { method: "POST", body: fd });
  } catch (e) {
    alert(e.message);
  }
  $("#drop").innerHTML = `<input id="file-input" type="file" multiple accept=".pdf,.md,.txt" hidden /><strong>Drop files here</strong> or click to upload <span class="muted">(.pdf, .md, .txt)</span>`;
  bindFileInput();
  loadDocs();
  init();
}
function bindFileInput() { $("#file-input").addEventListener("change", (e) => uploadFiles(e.target.files)); }
bindFileInput();
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
$("#reindex-btn").addEventListener("click", async () => { await api("/api/reindex", { method: "POST" }); loadDocs(); init(); });

// ------------------------------------------------------------ evaluation
const COLS = [
  ["mode", "Pipeline"], ["n", "n"], ["accuracy_%", "Accuracy %"], ["hallucination_%", "Halluc. %"], ["abstain_recall_%", "Abstain recall %"],
  ["trigger_rate_%", "Trigger %"], ["repair_gain_%", "Repair gain %"], ["repair_regression_%", "Regression %"],
  ["diagnosis_acc_%", "Diagnosis acc %"], ["aurc", "AURC ↓"], ["median_latency_ms", "Median ms"],
];

function summaryTable(summary) {
  const byInj = {};
  summary.forEach((r) => (byInj[r.injection] ||= []).push(r));
  return Object.entries(byInj).map(([inj, rows]) => `
    <h4>Injected failure: ${esc(INJECTIONS[inj] || inj)}</h4>
    <div class="table-wrap"><table class="table"><thead><tr>${COLS.map(([, h], i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr class="${r.mode === "failure_aware" ? "ours" : ""}">${COLS.map(([k], i) =>
      `<td class="${i ? "num" : ""}">${esc(k === "mode" ? MODES[r.mode] || r.mode : r[k] ?? "–")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`).join("");
}

$("#eval-btn").addEventListener("click", async () => {
  const body = {
    dataset: $("#dataset").value.trim() || "demo",
    modes: checked($("#eval-modes")),
    injections: checked($("#eval-inj")),
    limit: $("#limit").value ? Number($("#limit").value) : null,
    mirage_path: $("#mirage").value.trim() || null,
  };
  if (!body.modes.length || !body.injections.length) return alert("Pick at least one pipeline and one injection setting.");
  const btn = $("#eval-btn"), bar = $("#eval-progress"), msg = $("#eval-msg");
  btn.disabled = true; bar.classList.remove("hidden"); msg.textContent = "Starting…";
  try {
    const { job_id } = await api("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    for (;;) {
      await new Promise((r) => setTimeout(r, 700));
      const job = await api(`/api/evaluate/${job_id}`);
      $("div", bar).style.width = `${Math.round((100 * job.done) / Math.max(1, job.total))}%`;
      msg.textContent = `${job.done} / ${job.total} runs`;
      if (job.status === "error") throw new Error(job.error);
      if (job.status === "done") {
        msg.textContent = `Saved to results/${job.file}`;
        const sig = (job.significance || []).filter((t) => t.pairs).map((t) =>
          `<li>Ours vs ${esc(MODES[t.b] || t.b)} (${esc(INJECTIONS[t.injection] || t.injection)}): ours-only correct ${t.a_only_correct}, other-only ${t.b_only_correct}, McNemar p = ${t.p_value}</li>`).join("");
        $("#eval-results").innerHTML = `<div class="panel"><h2>Results</h2>${summaryTable(job.summary)}${sig ? `<h4>Significance (exact McNemar)</h4><ul class="muted">${sig}</ul>` : ""}</div>`;
        loadHistory();
        break;
      }
    }
  } catch (e) {
    $("#eval-results").innerHTML = `<div class="error">${esc(e.message)}</div>`;
    msg.textContent = "";
  } finally {
    btn.disabled = false;
    bar.classList.add("hidden");
  }
});

async function loadHistory() {
  try {
    const runs = await api("/api/results");
    $("#history").innerHTML = runs.length ? runs.map((r) => `<div class="history-item"><span>${esc(r.file)} · ${esc(r.config.dataset || "")} · ${r.summary.length} rows</span>
      <span><a href="/api/results/${encodeURIComponent(r.file)}" download>JSON</a> · <a href="#" data-show="${esc(r.file)}">Show</a></span></div>`).join("") : "None yet.";
    $$("[data-show]").forEach((a) => a.addEventListener("click", (e) => {
      e.preventDefault();
      const run = runs.find((r) => r.file === a.dataset.show);
      $("#eval-results").innerHTML = `<div class="panel"><h2>${esc(run.file)}</h2>${summaryTable(run.summary)}</div>`;
    }));
  } catch (_) {}
}

init();
