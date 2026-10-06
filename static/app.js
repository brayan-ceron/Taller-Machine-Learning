"use strict";

// ------------------------------------------------------------- utilidades
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => (v == null || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const compact = new Intl.NumberFormat("es", { notation: "compact", maximumFractionDigits: 1 });
const COLORS = ["#2c7bb6", "#f4a836", "#d7191c", "#7b3294", "#1a9641", "#636363"];
const OK_COLOR = "#2e9e5b", BAD_COLOR = "#d7191c";
const METRICS = [["accuracy", "Accuracy"], ["f1_macro", "F1 macro"], ["mcc", "MCC"], ["auc", "AUC (OvR)"]];

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  let d = null;
  try { d = await r.json(); } catch { /* respuesta sin JSON */ }
  if (!r.ok) throw new Error((d && d.error) || `Error ${r.status}`);
  return d;
}

function toast(msg, type = "success") {
  const el = document.createElement("div");
  el.className = `toast align-items-center text-bg-${type} border-0`;
  el.innerHTML = `<div class="d-flex"><div class="toast-body">${esc(msg)}</div>
    <button class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
  $("#toasts").appendChild(el);
  el.addEventListener("hidden.bs.toast", () => el.remove());
  new bootstrap.Toast(el, { delay: 4500 }).show();
}

const modelLabel = m => `${m.dataset.toUpperCase()} · #${m.rank ?? "-"} · ${m.modelo} · ${m.escalador}+${m.reductor}`;
const byId = id => state.models.find(m => m.id === id);

// ------------------------------------------------------------------ mapa
class MapView {
  constructor(canvas, onClick) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.data = null; this.marker = null; this.mode = "pred"; this.size = 300;
    this.onClick = onClick;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener("click", e => this.handle(e));
  }
  setData(d) { this.data = d; this.marker = null; this.draw(); }
  setMode(m) { this.mode = m; this.draw(); }
  setMarker(lat, lon, nearest) { this.marker = { lat, lon, nearest }; this.draw(); }
  resize() {
    const w = Math.max(240, this.c.parentElement.clientWidth), dpr = window.devicePixelRatio || 1;
    this.c.style.width = this.c.style.height = w + "px";
    this.c.width = this.c.height = Math.round(w * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.size = w; this.draw();
  }
  proj() {
    const b = this.data.bounds, pad = 34, W = this.size;
    const dx = (b.lon_max - b.lon_min) || 1, dy = (b.lat_max - b.lat_min) || 1;
    const s = Math.min((W - 2 * pad) / dx, (W - 2 * pad) / dy);
    this.p = { s, ox: (W - dx * s) / 2, oy: (W - dy * s) / 2, b };
  }
  toXY(lat, lon) {
    const { s, ox, oy, b } = this.p;
    return [ox + (lon - b.lon_min) * s, this.size - (oy + (lat - b.lat_min) * s)];
  }
  toGeo(x, y) {
    const { s, ox, oy, b } = this.p;
    return [(this.size - y - oy) / s + b.lat_min, (x - ox) / s + b.lon_min];
  }
  color(pt) {
    if (this.mode === "error") return pt[3] === pt[4] ? OK_COLOR : BAD_COLOR;
    return COLORS[(this.mode === "real" ? pt[3] : pt[4]) % COLORS.length];
  }
  draw() {
    const g = this.ctx, W = this.size;
    g.clearRect(0, 0, W, W);
    if (!this.data) return;
    this.proj();
    const { b, s } = this.p;
    // rejilla y ejes
    g.strokeStyle = "#e3e7ee"; g.fillStyle = "#8a93a3"; g.font = "10px system-ui"; g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const lon = b.lon_min + (b.lon_max - b.lon_min) * i / 4, lat = b.lat_min + (b.lat_max - b.lat_min) * i / 4;
      const [x] = this.toXY(b.lat_min, lon), [, y] = this.toXY(lat, b.lon_min);
      g.beginPath(); g.moveTo(x, 8); g.lineTo(x, W - 22); g.stroke();
      g.beginPath(); g.moveTo(30, y); g.lineTo(W - 8, y); g.stroke();
      g.textAlign = "center"; g.fillText(compact.format(lon), x, W - 8);
      g.textAlign = "right"; g.fillText(compact.format(lat), 28, y + 3);
    }
    // puntos (los errores se dibujan al final para que no queden tapados)
    const r = Math.max(2.5, Math.min(9, 0.5 * this.data.median_nn * s));
    const pts = this.data.points;
    const order = this.mode === "error"
      ? [...pts].sort((a, b2) => (a[3] === a[4]) - (b2[3] === b2[4])).reverse() : pts;
    for (const pt of order) {
      const [x, y] = this.toXY(pt[0], pt[1]);
      g.beginPath(); g.arc(x, y, r, 0, 6.2832);
      g.fillStyle = this.color(pt); g.globalAlpha = 0.9; g.fill();
      g.globalAlpha = 1; g.strokeStyle = "rgba(255,255,255,.8)"; g.lineWidth = 0.8; g.stroke();
    }
    // marcador de la consulta
    if (this.marker) {
      const [cx, cy] = this.toXY(this.marker.lat, this.marker.lon);
      const n = this.marker.nearest, [nx, ny] = this.toXY(n.lat, n.lon);
      g.lineWidth = 2; g.strokeStyle = "#111";
      g.beginPath(); g.arc(nx, ny, r + 4, 0, 6.2832); g.stroke();
      g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(cx - 9, cy); g.lineTo(cx + 9, cy); g.moveTo(cx, cy - 9); g.lineTo(cx, cy + 9); g.stroke();
    }
  }
  handle(e) {
    if (!this.data) return;
    const rc = this.c.getBoundingClientRect();
    const [lat, lon] = this.toGeo(e.clientX - rc.left, e.clientY - rc.top);
    this.onClick(lat, lon);
  }
}

// ---------------------------------------------------------------- estado
const state = { models: [], points: {}, last: null };
let mapA, mapB;

async function getPoints(id) {
  if (!state.points[id]) state.points[id] = await api(`/api/models/${encodeURIComponent(id)}/points`);
  return state.points[id];
}

async function loadModels() {
  state.models = (await api("/api/models")).models;
  state.points = {};
  renderRepo();
  fillSelects();
  await updateComparison();
}

function fillSelects() {
  const selA = $("#selA"), selB = $("#selB");
  const prev = [selA.value, selB.value];
  const groups = {};
  state.models.forEach(m => (groups[m.dataset] ??= []).push(m));
  const html = Object.entries(groups).map(([ds, ms]) =>
    `<optgroup label="${esc(ds.toUpperCase())}">${ms.map(m =>
      `<option value="${esc(m.id)}">${esc(modelLabel(m))}</option>`).join("")}</optgroup>`).join("");
  selA.innerHTML = selB.innerHTML = html;
  const ids = state.models.map(m => m.id);
  selA.value = ids.includes(prev[0]) ? prev[0] : (ids[0] ?? "");
  selB.value = ids.includes(prev[1]) ? prev[1] : (ids[1] ?? ids[0] ?? "");
}

// ------------------------------------------------------------ comparación
async function updateComparison() {
  const ia = $("#selA").value, ib = $("#selB").value;
  const ma = byId(ia), mb = byId(ib);
  if (!ma || !mb) {
    $("#titleA").textContent = $("#titleB").textContent = "Sin modelos almacenados";
    $("#metricas").innerHTML = $("#cmA").innerHTML = $("#cmB").innerHTML = "";
    mapA.setData(null); mapB.setData(null);
    return;
  }
  try {
    const [pa, pb] = await Promise.all([getPoints(ia), getPoints(ib)]);
    mapA.setData(pa); mapB.setData(pb);
  } catch (e) { toast(e.message, "danger"); return; }
  $("#titleA").textContent = modelLabel(ma);
  $("#titleB").textContent = modelLabel(mb);
  renderLegend(ma);
  renderMetrics(ma, mb);
  $("#cmA").innerHTML = confusion(ma);
  $("#cmB").innerHTML = confusion(mb);
  if (state.last) consultar(state.last.lat, state.last.lon);
}

function renderLegend(m) {
  const mode = $('input[name="mode"]:checked').value;
  const items = mode === "error"
    ? [[OK_COLOR, "Acierto"], [BAD_COLOR, "Error"]]
    : m.etiquetas.map((e, i) => {
        const r = m.rangos_irradiancia?.[String(i)];
        return [COLORS[i % COLORS.length], r ? `${e} (${fmt(r[0], 1)}–${fmt(r[1], 1)})` : e];
      });
  $("#legend").innerHTML = items.map(([c, t]) =>
    `<span><span class="sw" style="background:${c}"></span>${esc(t)}</span>`).join("");
}

const composite = m => METRICS.reduce((a, [k]) => a + (m.metricas[k] ?? 0), 0) / METRICS.length;

function renderMetrics(a, b) {
  const rows = METRICS.map(([k, label]) => [label, a.metricas[k], b.metricas[k], a.metricas_std?.[k], b.metricas_std?.[k]]);
  rows.push(["Score compuesto", composite(a), composite(b)]);
  const cell = (v, sd, color, win) => `
    <div class="${win ? "win" : ""}">${fmt(v)}${sd != null ? `<span class="text-body-secondary fw-normal small"> ±${fmt(sd)}</span>` : ""}</div>
    <div class="metric-bar"><div style="width:${Math.max(0, Math.min(1, v)) * 100}%;background:${color}"></div></div>`;
  $("#metricas").innerHTML = `
    <table class="table align-middle mb-0">
      <thead><tr><th>Métrica</th><th><span class="badge tag-a">A</span></th><th><span class="badge tag-b">B</span></th>
        <th class="text-end">Δ (B − A)</th></tr></thead>
      <tbody>${rows.map(([l, va, vb, sa, sb]) => {
        const d = vb - va, eps = 1e-9;
        const cls = Math.abs(d) < eps ? "text-body-secondary" : d > 0 ? "text-danger" : "text-primary";
        return `<tr><td class="fw-semibold">${l}</td>
          <td>${cell(va, sa, "#0d6efd", va > vb + eps)}</td><td>${cell(vb, sb, "#d63384", vb > va + eps)}</td>
          <td class="text-end ${cls}">${d > 0 ? "+" : ""}${fmt(d)}</td></tr>`;
      }).join("")}</tbody></table>
    <div class="px-3 pb-3 small text-body-secondary">En negrita, el mejor valor de cada fila.
      Δ &gt; 0 (rojo) favorece a B; Δ &lt; 0 (azul) favorece a A.</div>`;
}

function confusion(m) {
  const cm = m.matriz_confusion || [];
  const et = m.etiquetas;
  const body = cm.map((row, i) => {
    const tot = row.reduce((a, v) => a + v, 0) || 1;
    return `<tr><th>${esc(et[i] ?? i)}</th>${row.map((v, j) => {
      const p = v / tot, good = i === j;
      const bg = good ? `rgba(46,158,91,${0.12 + 0.75 * p})` : `rgba(215,25,28,${0.06 + 0.7 * p})`;
      return `<td style="background:${bg}" title="${(p * 100).toFixed(1)}% de la fila">${v}</td>`;
    }).join("")}</tr>`;
  }).join("");
  return `<table class="cm"><thead><tr><th></th>${et.map(e => `<th>${esc(e)}</th>`).join("")}</tr></thead>
    <tbody>${body}</tbody></table>
    <div class="text-center small text-body-secondary mt-2">Filas: real · Columnas: predicho</div>`;
}

// -------------------------------------------------- clasificar una posición
async function consultar(lat, lon) {
  const ia = $("#selA").value, ib = $("#selB").value;
  if (!ia || !ib) return;
  state.last = { lat, lon };
  $("#inLat").value = Math.round(lat * 100) / 100;
  $("#inLon").value = Math.round(lon * 100) / 100;
  try {
    const post = id => api("/api/clasificar", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: id, lat, lon }),
    });
    const [ra, rb] = await Promise.all([post(ia), post(ib)]);
    mapA.setMarker(lat, lon, ra.punto);
    mapB.setMarker(lat, lon, rb.punto);
    renderResult(ra, rb);
  } catch (e) { toast(e.message, "danger"); }
}

function resCard(tag, r) {
  const m = byId(r.model_id);
  const col = COLORS[r.clase_pred % COLORS.length];
  const rango = r.rango_pred ? `${fmt(r.rango_pred[0], 1)} – ${fmt(r.rango_pred[1], 1)}` : "—";
  return `<div class="res-card">
    <div class="d-flex justify-content-between align-items-center mb-1">
      <span><span class="badge ${tag === "A" ? "tag-a" : "tag-b"}">${tag}</span>
        <span class="small text-body-secondary ms-1">${esc(m ? m.modelo : "")}</span></span>
      <span class="pill cls" style="background:${col}">${esc(r.etiqueta_pred)}</span>
    </div>
    <div class="small">Rango de la clase: <b>${rango}</b> · Punto más cercano: <b>${fmt(r.valor, 1)}</b>
      (real: ${esc(r.etiqueta_real)} ${r.acierto ? '<i class="bi bi-check-circle-fill text-success"></i>' : '<i class="bi bi-x-circle-fill text-danger"></i>'})</div>
    ${r.fuera_de_cobertura ? '<div class="small text-warning-emphasis mt-1"><i class="bi bi-exclamation-triangle me-1"></i>Posición lejos de los puntos con datos; el resultado es del vecino más cercano.</div>' : ""}
  </div>`;
}

function renderResult(ra, rb) {
  const same = ra.clase_pred === rb.clase_pred;
  $("#resultado").innerHTML = resCard("A", ra) + resCard("B", rb) +
    `<div class="alert ${same ? "alert-success" : "alert-warning"} py-2 mb-0 small">
      <i class="bi ${same ? "bi-check2-all" : "bi-shuffle"} me-1"></i>
      ${same ? `Ambos modelos coinciden: <b>${esc(ra.etiqueta_pred)}</b>.`
             : `Los modelos discrepan: A dice <b>${esc(ra.etiqueta_pred)}</b>, B dice <b>${esc(rb.etiqueta_pred)}</b>.`}
    </div>`;
}

// ----------------------------------------------------------- repositorio
function renderRepo() {
  $("#countModels").textContent = state.models.length;
  $("#tbodyModels").innerHTML = state.models.length ? state.models.map(m => `
    <tr>
      <td><span class="badge text-bg-secondary">${esc(m.dataset)}</span><div class="small text-body-secondary">#${esc(m.rank ?? "-")}</div></td>
      <td><div class="fw-semibold">${esc(m.modelo)}</div>
          <div class="small text-body-secondary">${esc(m.escalador)} + ${esc(m.reductor)} · ${esc(m.discretizador ?? "")}</div></td>
      ${METRICS.map(([k]) => `<td class="text-end">${fmt(m.metricas[k])}</td>`).join("")}
      <td class="text-end text-nowrap">
        ${m.tiene_joblib ? `<a class="btn btn-sm btn-outline-secondary" title="Descargar .joblib" href="/api/models/${encodeURIComponent(m.id)}/joblib"><i class="bi bi-download"></i></a>` : ""}
        <button class="btn btn-sm btn-outline-danger" data-del="${esc(m.id)}" title="Eliminar"><i class="bi bi-trash"></i></button>
      </td></tr>`).join("")
    : `<tr><td colspan="7" class="text-center text-body-secondary py-4">No hay modelos almacenados.</td></tr>`;
}

$("#tbodyModels").addEventListener("click", async e => {
  const btn = e.target.closest("[data-del]");
  if (!btn || !confirm("¿Eliminar este modelo?")) return;
  try {
    await api(`/api/models/${encodeURIComponent(btn.dataset.del)}`, { method: "DELETE" });
    toast("Modelo eliminado");
    await loadModels();
  } catch (err) { toast(err.message, "danger"); }
});

$("#formUpload").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = $("#btnUpload");
  btn.disabled = true;
  try {
    const fd = new FormData(e.target);
    if (!fd.get("joblib") || !fd.get("joblib").name) fd.delete("joblib");
    const r = await api("/api/models", { method: "POST", body: fd });
    toast(`Modelo guardado: ${r.id}`);
    e.target.reset();
    await loadModels();
  } catch (err) { toast(err.message, "danger"); }
  finally { btn.disabled = false; }
});

// ------------------------------------------------------------------ init
mapA = new MapView($("#mapA"), consultar);
mapB = new MapView($("#mapB"), consultar);
$("#selA").addEventListener("change", updateComparison);
$("#selB").addEventListener("change", updateComparison);
$("#modeGroup").addEventListener("change", () => {
  const mode = $('input[name="mode"]:checked').value;
  mapA.setMode(mode); mapB.setMode(mode);
  const m = byId($("#selA").value); if (m) renderLegend(m);
});
$("#formConsulta").addEventListener("submit", e => {
  e.preventDefault();
  consultar(parseFloat($("#inLat").value), parseFloat($("#inLon").value));
});
loadModels().catch(e => toast(e.message, "danger"));
