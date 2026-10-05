"use strict";
/* TubeMap dashboard. Vanilla JS, no build step, works offline. */

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const IN = 25.4;
const STATE_COLORS = () => {
  const cs = getComputedStyle(document.documentElement);
  return {
    OPEN: cs.getPropertyValue("--s-open").trim(), PLUGGED: cs.getPropertyValue("--s-plug").trim(),
    REVIEW: cs.getPropertyValue("--s-review").trim(), NO_TUBE: cs.getPropertyValue("--s-none").trim(),
    anchor: cs.getPropertyValue("--s-anchor").trim(),
    TUBE: cs.getPropertyValue("--c-tube").trim(), BLOCKED: cs.getPropertyValue("--c-blocked").trim(),
    UNKNOWN: cs.getPropertyValue("--c-unknown").trim(), NOT_TUBE: cs.getPropertyValue("--c-not").trim(),
    dFace: cs.getPropertyValue("--d-face").trim(), dRecess: cs.getPropertyValue("--d-recess").trim(),
    dConflict: cs.getPropertyValue("--d-conflict").trim(), lw: parseFloat(cs.getPropertyValue("--line-scale")) || 1,
  };
};
let COLORS = STATE_COLORS();

async function api(path, opts = {}) {
  const o = { method: "GET", headers: {}, ...opts };
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers["Content-Type"] = "application/json"; delete o.json; }
  const r = await fetch(path, o);
  const ct = r.headers.get("Content-Type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
  return data;
}
function toast(msg, bad = false, ms = 3500) {
  const d = document.createElement("div");
  d.textContent = msg; if (bad) d.className = "bad";
  $("#toast").appendChild(d); setTimeout(() => d.remove(), ms);
}
function when(ts) { if (!ts) return ""; const d = new Date(ts * 1000); return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }

// ------------------------------------------------------------------ canvas viewer
class Viewer {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.img = null; this.k = 1; this.ox = 0; this.oy = 0;
    this.draw_overlay = null; this.onbox = null; this.box = null; this.onclick = null; this.ondblclick = null; this.oncontext = null; this.onmove = null;
    const ro = new ResizeObserver(() => this.resize()); ro.observe(canvas.parentElement);
    let down = null;
    // Drag pans. With onbox set, Shift+drag or Ctrl+drag draws a selection box instead.
    canvas.addEventListener("pointerdown", e => {
      if (e.button === 2) return;
      const boxMode = !!(this.onbox && (e.shiftKey || e.ctrlKey || e.metaKey));
      down = { x: e.clientX, y: e.clientY, ox: this.ox, oy: this.oy, moved: false, boxMode, p0: this.toImage(e) };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (this.onmove) this.onmove(this.toImage(e), e);
      if (!down) return;
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) down.moved = true;
      if (!down.moved) return;
      if (down.boxMode) { const p = this.toImage(e); this.box = { x0: down.p0.x, y0: down.p0.y, x1: p.x, y1: p.y, ctrl: e.ctrlKey || e.metaKey }; this.draw(); }
      else { this.ox = down.ox + dx; this.oy = down.oy + dy; this.draw(); }
    });
    canvas.addEventListener("pointerup", e => {
      if (down && down.boxMode && down.moved && this.box) {
        const b = this.box; this.box = null;
        this.onbox({ x0: Math.min(b.x0, b.x1), y0: Math.min(b.y0, b.y1), x1: Math.max(b.x0, b.x1), y1: Math.max(b.y0, b.y1) }, e);
        this.draw();
      } else if (down && !down.moved && this.onclick && e.button === 0) this.onclick(this.toImage(e), e);
      down = null;
    });
    canvas.addEventListener("dblclick", e => { if (this.ondblclick) this.ondblclick(this.toImage(e), e); });
    canvas.addEventListener("contextmenu", e => { e.preventDefault(); if (this.oncontext) this.oncontext(this.toImage(e), e); });
    canvas.addEventListener("wheel", e => {
      e.preventDefault(); if (!this.img) return;
      const r = this.c.getBoundingClientRect(); const mx = e.clientX - r.left, my = e.clientY - r.top;
      const f = Math.exp(-e.deltaY * 0.0015); const nk = Math.min(40, Math.max(0.05, this.k * f));
      this.ox = mx - (mx - this.ox) * nk / this.k; this.oy = my - (my - this.oy) * nk / this.k; this.k = nk; this.draw();
    }, { passive: false });
  }
  resize() {
    const r = this.c.parentElement.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr; this.c.width = Math.max(1, r.width * dpr); this.c.height = Math.max(1, r.height * dpr);
    if (this.img && !this._fitted) this.fit(); else this.draw();
  }
  async load(url, w, h) {
    const img = new Image(); img.decoding = "async";
    await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(new Error("image failed to load")); img.src = url; });
    this.img = img; this.iw = w || img.naturalWidth; this.ih = h || img.naturalHeight; this._fitted = false; this.fit();
  }
  clear() { this.img = null; this.draw(); }
  fit() {
    if (!this.img) return this.draw();
    const r = this.c.parentElement.getBoundingClientRect(); if (r.width < 2) return;
    this.k = Math.min(r.width / this.iw, r.height / this.ih) * 0.98; this.ox = (r.width - this.iw * this.k) / 2; this.oy = (r.height - this.ih * this.k) / 2;
    this._fitted = true; this.draw();
  }
  toImage(e) { const r = this.c.getBoundingClientRect(); return { x: (e.clientX - r.left - this.ox) / this.k, y: (e.clientY - r.top - this.oy) / this.k }; }
  s(x, y) { return [x * this.k + this.ox, y * this.k + this.oy]; }
  draw() {
    const { ctx } = this; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.c.width, this.c.height);
    if (!this.img) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.imageSmoothingEnabled = this.k < 2;
    ctx.drawImage(this.img, this.ox, this.oy, this.iw * this.k, this.ih * this.k);
    if (this.draw_overlay) this.draw_overlay(ctx, this);
    if (this.box) {
      const b = this.box, [x0, y0] = this.s(b.x0, b.y0), [x1, y1] = this.s(b.x1, b.y1);
      ctx.save(); ctx.strokeStyle = b.ctrl ? "#ffd43b" : "#ffffff"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.fillStyle = b.ctrl ? "rgba(255,212,59,0.12)" : "rgba(255,255,255,0.10)";
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
    }
  }
  /** Centre image point (x,y) at zoom k (screen px per image px). */
  focus(x, y, k) {
    const r = this.c.parentElement.getBoundingClientRect();
    this.k = Math.min(40, Math.max(this.k, k)); this.ox = r.width / 2 - x * this.k; this.oy = r.height / 2 - y * this.k; this.draw();
  }
}

function drawPoly(ctx, v, pts, w, h, color, closed = true) {
  if (!pts || pts.length < 1) return;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2 * COLORS.lw; ctx.setLineDash([6, 4]); ctx.beginPath();
  pts.forEach(([x, y], i) => { const [sx, sy] = v.s(x * (w - 1), y * (h - 1)); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
  if (closed) ctx.closePath(); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle = color;
  pts.forEach(([x, y]) => { const [sx, sy] = v.s(x * (w - 1), y * (h - 1)); ctx.fillRect(sx - 3, sy - 3, 6, 6); });
  ctx.restore();
}

// ROI drawing helper shared by Map and Label
class RoiTool {
  constructor(viewer, onchange) { this.v = viewer; this.active = false; this.pts = null; this.onchange = onchange; }
  start() { this.active = true; this.pts = []; this.v.draw(); }
  click(p) { if (!this.active) return false; this.pts.push([p.x / (this.v.iw - 1), p.y / (this.v.ih - 1)]); this.v.draw(); return true; }
  finish() { if (!this.active) return false; this.active = false; if (!this.pts || this.pts.length < 3) this.pts = null; this.onchange && this.onchange(this.pts); this.v.draw(); return true; }
  cancel() { this.active = false; this.pts = null; this.onchange && this.onchange(null); this.v.draw(); }
}

// ------------------------------------------------------------------ app state
const S = {
  tab: "live", devices: [], drivers: [], settings: null, standards: null,
  liveDev: null, captures: [],
  map: { cid: null, data: null, roi: null, sel: null, mode: "known" },
  lab: { cid: null, meta: null, items: [], roi: null, cls: "TUBE", undo: [], filter: "all", dirty: false, focus: -1 },
  dev: { editing: null },
};

// ------------------------------------------------------------------ tabs
$$("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  S.tab = t;
  $$("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  $$(".tab").forEach(s => s.classList.toggle("on", s.id === "tab-" + t));
  if (t === "live") renderLive();
  if (t === "map") mapEnter();
  if (t === "label") labEnter();
  if (t === "devices") renderDevices();
  if (t === "model") modelEnter();
  if (t === "settings") settingsEnter();
  requestAnimationFrame(() => { mapV.resize(); labV.resize(); });
}

// ------------------------------------------------------------------ status polling
async function refreshStatus() {
  try {
    const st = await api("/api/status");
    $("#ver").textContent = "v" + st.version;
    S.devices = st.devices; S.status = st;
    const run = st.devices.filter(d => d.state === "running").length;
    const err = st.devices.filter(d => d.state === "error").length;
    $("#topstatus").innerHTML = `<span class="chip ${err ? "error" : run ? "running" : "stopped"}">${run} camera${run === 1 ? "" : "s"} running${err ? `, ${err} error` : ""}</span>` +
      `<span>${esc(st.hostname)}</span><span>${fmt(st.disk_free_gb, 1)} GB free</span>`;
    if (S.tab === "live") renderLiveList();
    if (S.tab === "devices" && !S.dev.editing) renderDevices(false);
  } catch (e) { $("#topstatus").innerHTML = `<span class="chip error">Offline</span>`; }
}
setInterval(refreshStatus, 2500);

// ------------------------------------------------------------------ LIVE
let liveImgFor = null;
function renderLiveList() {
  const running = S.devices.filter(d => d.enabled);
  if (!S.liveDev || !running.find(d => d.id === S.liveDev)) S.liveDev = (running.find(d => d.state === "running") || running[0] || {}).id || null;
  $("#live-devs").innerHTML = running.length ? running.map(d => `
    <div class="devpick ${d.id === S.liveDev ? "on" : ""}" data-id="${esc(d.id)}">
      <span class="dot ${esc(d.state)}"></span>
      <div><div><b>${esc(d.name)}</b></div><div class="hint" style="margin:0">${esc(d.state)}${d.state === "running" ? ` · ${fmt(d.fps, 1)} fps${d.has_depth ? " · depth" : ""}` : ""}${d.error ? " · " + esc(d.error).slice(0, 80) : ""}</div></div>
    </div>`).join("") : `<p class="hint">No cameras enabled.</p>`;
  $$("#live-devs .devpick").forEach(el => el.onclick = () => { S.liveDev = el.dataset.id; renderLive(); });
  const d = S.devices.find(x => x.id === S.liveDev);
  $("#live-meta").textContent = d && d.state === "running" ? `${d.name} · ${d.resolution ? d.resolution.join(" × ") : "…"} · ${fmt(d.fps, 1)} fps · depth ${d.has_depth ? "yes" : "no"} · calibration ${d.intrinsics ? "yes" : "no"}${d.info && d.info.usb_speed ? " · USB " + d.info.usb_speed : ""}` : (d && d.error ? d.error : "");
  $("#btn-capture").disabled = !(d && d.state === "running");
}
function renderLive() {
  renderLiveList();
  const d = S.devices.find(x => x.id === S.liveDev);
  const view = $("#live-view");
  if (d && d.state === "running") {
    if (liveImgFor !== d.id) { view.innerHTML = `<img alt="Live view from ${esc(d.name)}" src="/api/devices/${encodeURIComponent(d.id)}/stream.mjpg?t=${Date.now()}">`; liveImgFor = d.id; }
  } else { view.innerHTML = `<div class="empty">${d ? esc(d.name) + " is " + esc(d.state) + (d.error ? ":<br>" + esc(d.error) : "") : "No camera running. Add or enable one on the Devices tab."}</div>`; liveImgFor = null; }
  renderLiveControls(d); renderRecent();
}
function renderLiveControls(d) {
  const box = $("#live-controls");
  if (!d) { box.innerHTML = ""; return; }
  const drv = S.drivers.find(x => x.id === d.driver); if (!drv) { box.innerHTML = ""; return; }
  const fields = drv.schema.filter(f => f.live && !f.advanced);
  box.innerHTML = fields.length ? fields.map(f => fieldHtml(f, d.settings[f.key], "lc-")).join("") : `<p class="hint">No live settings for this device.</p>`;
  fields.forEach(f => { const el = $("#lc-" + f.key); if (el) el.onchange = async () => {
    try { const r = await api(`/api/devices/${encodeURIComponent(d.id)}`, { method: "PUT", json: { settings: { [f.key]: readField(f, "lc-") } } }); toast(r.applied_live ? "Applied" : "Saved; camera restarting"); refreshStatus(); } catch (e) { toast(e.message, true); } }; });
}
async function renderRecent() {
  try { S.captures = await api("/api/captures?limit=40"); } catch (e) { return; }
  $("#recent").innerHTML = S.captures.slice(0, 16).map(c => `<figure data-id="${esc(c.id)}"><img loading="lazy" alt="" src="/api/captures/${c.id}/thumb.jpg"><figcaption>${when(c.created_at)} · ${esc(c.device || c.source)}${c.has_map ? " · map" : ""}</figcaption></figure>`).join("") || `<p class="hint">No captures yet.</p>`;
  $$("#recent figure").forEach(f => f.onclick = () => { S.map.cid = f.dataset.id; showTab("map"); });
}
$("#btn-capture").onclick = async () => {
  const b = $("#btn-capture"); b.disabled = true; b.textContent = "Capturing…";
  try {
    const c = await api(`/api/devices/${encodeURIComponent(S.liveDev)}/capture`, { method: "POST" });
    const comp = (c.companions || []); const bad = comp.filter(x => x.error);
    toast(`Saved ${c.id}${c.has_depth ? " with depth" : ""}${comp.length ? ` + ${comp.length - bad.length} depth-check capture(s)` : ""}`);
    if (c.depth_note) toast(`Saved without OAK depth: ${c.depth_note}`, true, 8000);
    if (bad.length) toast(`Depth camera capture failed: ${bad[0].error}`, true, 6000);
    S.map.cid = c.id; renderRecent();
  }
  catch (e) { toast("Capture failed: " + e.message, true, 6000); }
  finally { b.disabled = false; b.textContent = "Capture"; }
};
$("#btn-capture-all").onclick = async () => {
  try { const r = await api("/api/capture_all", { method: "POST" }); toast(`Saved ${r.captures.length} capture(s)` + (Object.keys(r.errors).length ? `; ${Object.keys(r.errors).length} failed` : "")); renderRecent(); }
  catch (e) { toast(e.message, true); }
};

// ------------------------------------------------------------------ settings fields (shared by devices)
function fieldHtml(f, value, prefix = "f-") {
  const id = prefix + f.key; const v = value ?? f.default;
  const help = f.help ? `<span class="fieldhelp">${esc(f.help)}</span>` : "";
  if (f.type === "bool") return `<label class="chk"><input type="checkbox" id="${id}" ${v ? "checked" : ""}> ${esc(f.label)}</label>${help}`;
  if (f.type === "select") {
    const opts = f.options.map(o => typeof o === "object" ? o : { value: o, label: o });
    return `<label>${esc(f.label)}<select id="${id}">${opts.map(o => `<option value="${esc(o.value)}" ${String(o.value) === String(v) ? "selected" : ""}>${esc(o.label)}</option>`).join("")}</select>${help}</label>`;
  }
  const t = f.type === "text" ? "text" : "number";
  const attrs = [f.min !== undefined ? `min="${f.min}"` : "", f.max !== undefined ? `max="${f.max}"` : "", `step="${f.step ?? (f.type === "int" ? 1 : "any")}"`].join(" ");
  return `<label>${esc(f.label)}<input id="${id}" type="${t}" ${t === "number" ? attrs : ""} value="${esc(v)}">${help}</label>`;
}
function readField(f, prefix = "f-") {
  const el = $("#" + prefix + f.key); if (!el) return undefined;
  if (f.type === "bool") return el.checked;
  if (f.type === "int") return parseInt(el.value, 10);
  if (f.type === "float") return parseFloat(el.value);
  return el.value;
}

// ------------------------------------------------------------------ MAP
const mapV = new Viewer($("#map-canvas"));
const mapRoi = new RoiTool(mapV, pts => { S.map.roi = pts; $("#map-roi").textContent = "Draw ROI"; });
mapV.draw_overlay = (ctx, v) => {
  const m = S.map.data; const w = v.iw, h = v.ih;
  if (m && m.capture_id === S.map.cid) {
    const showEmpty = $("#map-show-empty").checked, showIds = $("#map-show-ids").checked && v.k > 1.2;
    const anchors = new Set((m.anchors || []).map(a => a.id));
    ctx.lineWidth = 1.5 * COLORS.lw; ctx.font = `${Math.max(9, Math.min(13, 6 * v.k))}px system-ui`; ctx.textAlign = "center";
    for (const s of m.sites) {
      if (s.state === "NO_TUBE" && !showEmpty) continue;
      const [x, y] = v.s(s.u, s.v); const r = Math.max(2.5, s.r_px * v.k);
      ctx.strokeStyle = COLORS[s.state] || "#fff"; ctx.setLineDash(s.state === "NO_TUBE" ? [3, 3] : []);
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
      if (s.state === "PLUGGED") { ctx.fillStyle = COLORS.PLUGGED + "88"; ctx.fill(); }
      if (anchors.has(s.id)) { ctx.setLineDash([]); ctx.strokeStyle = COLORS.anchor; ctx.lineWidth = 3 * COLORS.lw; ctx.strokeRect(x - r - 4, y - r - 4, 2 * r + 8, 2 * r + 8); ctx.lineWidth = 1.5 * COLORS.lw; }
      if (showIds && s.state !== "NO_TUBE") { ctx.fillStyle = "#fff"; ctx.fillText(s.id, x, y - r - 3); }
    }
    ctx.setLineDash([]);
    if ($("#map-show-depth").checked) {
      for (const s of m.sites) {
        if (!s.depth_evidence || s.depth_evidence === "LOW_RES" || s.depth_evidence === "UNCLEAR") continue;
        const [x, y] = v.s(s.u, s.v); const r = Math.max(2, Math.min(5, s.r_px * v.k * 0.35));
        if (s.depth_evidence === "FACE") { ctx.fillStyle = COLORS.dFace; ctx.fillRect(x - r, y - r, 2 * r, 2 * r); }
        else { ctx.fillStyle = COLORS.dRecess; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); }
        if (s.depth_conflict) { ctx.strokeStyle = COLORS.dConflict; ctx.lineWidth = 2.5 * COLORS.lw; ctx.beginPath(); ctx.arc(x, y, Math.max(6, s.r_px * v.k + 4), 0, 7); ctx.stroke(); ctx.lineWidth = 1.5 * COLORS.lw; }
      }
    }
    if ($("#map-show-rej").checked) { ctx.strokeStyle = "#ff00ff"; for (const r of m.rejected) { const [x, y] = v.s(r.u, r.v); ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke(); } }
    const sel = S.map.sel && m.sites.find(s => s.id === S.map.sel);
    if (sel) { const [x, y] = v.s(sel.u, sel.v); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5 * COLORS.lw; ctx.beginPath(); ctx.arc(x, y, Math.max(6, sel.r_px * v.k + 5), 0, 7); ctx.stroke(); }
  }
  const roi = mapRoi.active ? mapRoi.pts : S.map.roi;
  if (roi) drawPoly(ctx, v, roi, w, h, "#ffd43b", !mapRoi.active);
};
mapV.onclick = p => {
  if (mapRoi.click(p)) return;
  const m = S.map.data; if (!m) return;
  let best = null, bd = 1e9;
  for (const s of m.sites) { const d = Math.hypot(s.u - p.x, s.v - p.y); if (d < bd) { bd = d; best = s; } }
  if (best && bd < Math.max(8, best.r_px * 1.6)) { S.map.sel = best.id; renderSel(); mapV.draw(); }
};
mapV.ondblclick = () => { if (mapRoi.finish()) toast(S.map.roi ? "ROI set" : "ROI needs 3+ points"); };
document.addEventListener("keydown", e => { if (e.key === "Escape") { if (mapRoi.active) mapRoi.cancel(); if (labRoi.active) labRoi.cancel(); } });
$("#map-roi").onclick = () => { mapRoi.start(); $("#map-roi").textContent = "Drawing… (dbl-click)"; };
$("#map-roi-clear").onclick = () => { mapRoi.cancel(); };
$("#map-fit").onclick = () => mapV.fit();
["#map-show-ids", "#map-show-empty", "#map-show-rej", "#map-show-depth"].forEach(s => $(s).onchange = () => { $("#depth-legend").hidden = !$("#map-show-depth").checked; mapV.draw(); });
$$("#map-mode button").forEach(b => b.onclick = () => setMapMode(b.dataset.v));
function setMapMode(mode) {
  S.map.mode = mode;
  $$("#map-mode button").forEach(b => b.classList.toggle("on", b.dataset.v === mode));
  $$(".known-only").forEach(el => el.hidden = mode !== "known");
  $$(".disc-only").forEach(el => el.hidden = mode === "known");
  if (mode === "known" && $("#g-pattern").value === "AUTO") $("#g-pattern").value = "TRIANGULAR";
}
function geometryForm() {
  return { mode: S.map.mode, pattern: $("#g-pattern").value, pitch_value: parseFloat($("#g-pitch").value), pitch_unit: $("#g-pitch-unit").value,
           id_value: parseFloat($("#g-id").value), id_unit: $("#g-id-unit").value };
}
function setGeometryForm(g) {
  if (!g) return; setMapMode(g.mode || "known");
  $("#g-pattern").value = g.pattern || "TRIANGULAR"; $("#g-pitch").value = g.pitch_value ?? ""; $("#g-pitch-unit").value = g.pitch_unit || "in";
  $("#g-id").value = g.id_value ?? ""; $("#g-id-unit").value = g.id_unit || "in";
}
$("#g-tube").onchange = () => {
  const v = $("#g-tube").value; if (!v) return; const t = JSON.parse(v);
  if ($("#g-id-unit").value === "in") $("#g-id").value = t.id_in; else $("#g-id").value = t.id_mm;
};
async function mapEnter() {
  if (!S.standards) await loadStandards();
  try { S.captures = await api("/api/captures?limit=200"); } catch (e) { }
  const sel = $("#map-capture");
  sel.innerHTML = S.captures.map(c => `<option value="${esc(c.id)}">${c.name ? esc(c.name) + " · " : ""}${when(c.created_at)} · ${esc(c.device || c.source || "")}${c.has_depth ? " · depth" : ""}${c.has_map ? " · mapped" : ""}</option>`).join("");
  if (!S.map.cid && S.captures.length) S.map.cid = S.captures[0].id;
  if (S.map.cid) { sel.value = S.map.cid; await mapLoadCapture(S.map.cid); }
}
$("#map-capture").onchange = e => mapLoadCapture(e.target.value);
async function mapLoadCapture(cid) {
  S.map.cid = cid; S.map.sel = null; $("#map-err").textContent = "";
  const d = await api(`/api/captures/${cid}`);
  await mapV.load(`/api/captures/${cid}/preview.jpg`, d.meta.image.width, d.meta.image.height);
  S.map.data = d.map;
  if (d.map) { S.map.roi = d.map.roi; if (d.map.geometry_request) setGeometryForm(d.map.geometry_request); }
  else { S.map.roi = d.labels && d.labels.roi || null; if (S.settings) setGeometryForm(S.settings.geometry); }
  renderMapResult(); mapV.draw();
}
$("#map-build").onclick = async () => {
  if (!S.map.cid) return toast("Capture an image first.", true);
  const b = $("#map-build"); b.disabled = true; b.textContent = "Building…"; $("#map-err").textContent = "";
  try {
    const g = geometryForm();
    S.map.data = await api(`/api/captures/${S.map.cid}/map`, { method: "POST", json: { geometry: g, roi: S.map.roi } });
    if (S.settings) { S.settings.geometry = g; api("/api/settings", { method: "PUT", json: { geometry: g } }).catch(() => { }); }
    renderMapResult(); mapV.draw();
  } catch (e) { $("#map-err").textContent = e.message; }
  finally { b.disabled = false; b.textContent = "Build map"; }
};
function renderMapResult() {
  const m = S.map.data; const box = $("#map-result");
  $("#map-anchor-box").hidden = !m;
  if (!m) { box.innerHTML = `<p class="muted">Build a map to see results.</p>`; return; }
  const g = m.geometry; const fails = m.checks.filter(c => c.status === "FAIL");
  const pitch = g.pitch_mm ? `${fmt(g.pitch_mm, 3)} mm · ${fmt(g.pitch_mm / IN, 4)} in` : "not measured";
  const tid = g.tube_id_mm ? `${fmt(g.tube_id_mm, 2)} mm · ${fmt(g.tube_id_mm / IN, 3)} in` : "–";
  let snap = "";
  if (g.nearest_pitch) snap += `<b>Nearest std pitch</b><span>${g.nearest_pitch.nominal_in}" (${fmt(g.nearest_pitch.delta_pct, 1)}%)</span>`;
  if (g.nearest_tubes && g.nearest_tubes.length) snap += `<b>Tube candidates</b><span>${g.nearest_tubes.slice(0, 2).map(t => `${t.od_in}" ${t.bwg} BWG`).join(", ")}</span>`;
  box.innerHTML = `
    <div class="banner ${m.ready ? "ok" : "bad"}">${m.ready ? "Map ready" : "Not ready: " + esc(fails.map(c => c.label).join(", "))}</div>
    <div class="checks">${m.checks.map(c => `<div class="check"><span class="chip ${c.status}">${c.status}</span><b>${esc(c.label)}</b><span class="d">${esc(c.detail)}</span></div>`).join("")}</div>
    <div class="kv">
      <b>Pattern</b><span>${esc((g.pattern || "").toLowerCase())}</span>
      <b>Pitch (${esc(g.pitch_source)})</b><span>${pitch}</span>
      <b>Tube ID (${esc(g.tube_id_source || "–")})</b><span>${tid}</span>${snap}
      <b>Tube sites</b><span>${m.tube_sites}</span>
      ${Object.entries(m.counts).map(([k, v]) => `<b><i class="s-${k}" style="display:inline-block;width:9px;height:9px;border-radius:50%;border:2px solid"></i> ${k.replace("_", " ").toLowerCase()}</b><span>${v}</span>`).join("")}
      <b>Lattice residual</b><span>${fmt(m.lattice.residual_px_median, 2)} px (p95 ${fmt(m.lattice.residual_px_p95, 2)})</span>
      <b>Profile</b><span>${esc(m.profile)}</span>
      <b>Built in</b><span>${fmt(m.elapsed_s, 2)} s</span>
    </div>
    <div class="btnrow">
      <a class="btn small" href="/api/captures/${m.capture_id}/map.csv">CSV</a>
      <a class="btn small" href="/api/captures/${m.capture_id}/map" download="${m.capture_id}_tubemap.json">JSON</a>
      <button class="small" id="map-to-label">Label this capture</button>
      ${m.mode === "known" && m.depth_pitch ? `<button class="small" id="map-cal">Calibrate depth scale</button>` : ""}
      ${m.mode === "known" && m.companion && m.companion.pitch_status ? `<button class="small" id="map-cal-comp">Calibrate ${esc(m.companion.device)} depth</button>` : ""}
    </div>${companionHtml(m)}`;
  $("#map-to-label").onclick = () => { S.lab.cid = m.capture_id; S.lab.proposeOnOpen = true; showTab("label"); };
  const calc = $("#map-cal-comp");
  if (calc) calc.onclick = async () => {
    if (!confirmInline(calc, "Only on a sheet whose pitch you measured with a rule. Click again to apply.")) return;
    try { const r = await api(`/api/captures/${m.capture_id}/calibrate_depth`, { method: "POST", json: { target: "companion" } }); S.map.data = r.map; renderMapResult(); mapV.draw(); toast(`${m.companion.device} depth scale ${fmt(r.previous, 4)} → ${fmt(r.factor, 4)}`); }
    catch (e) { toast(e.message, true, 6000); }
  };
  const cal = $("#map-cal");
  if (cal) cal.onclick = async () => {
    if (!confirmInline(cal, "Only on a sheet whose pitch you measured with a rule. Click again to apply.")) return;
    try { const r = await api(`/api/captures/${m.capture_id}/calibrate_depth`, { method: "POST" }); S.map.data = r.map; renderMapResult(); mapV.draw(); toast(`Depth scale ${fmt(r.previous, 4)} → ${fmt(r.factor, 4)}${r.device_updated ? " (saved to device)" : ""}`); }
    catch (e) { toast(e.message, true, 6000); }
  };
  renderSel(); renderAnchors();
}
function companionHtml(m) {
  const c = m.companion; const ds = m.depth_summary;
  let h = "";
  const ev = { FACE: 0, RECESS: 0, UNCLEAR: 0, LOW_RES: 0 }; let conf = 0, any = false;
  for (const s of m.sites) { if (s.depth_evidence) { any = true; ev[s.depth_evidence]++; } if (s.depth_conflict) conf++; }
  if (any) h += `<div class="subhead">Depth at openings${c && c.match && c.match.valid ? ` (from ${esc(c.device)})` : ""}</div><div class="kv">
      <b>Surface at face</b><span>${ev.FACE}</span><b>Open / deep</b><span>${ev.RECESS}</span><b>Unclear</b><span>${ev.UNCLEAR}</span><b>Too small</b><span>${ev.LOW_RES}</span>
      <b>Disagree with image</b><span>${conf}</span></div><p class="hint">Tick “Depth evidence” above the image to see them.</p>`;
  if (c) h += `<div class="subhead">Companion: ${esc(c.device)}</div><div class="kv"><b>Capture</b><span>${esc(c.capture_id || "–")}</span>
      ${c.match ? (c.match.valid ? `<b>Tubes lined up</b><span>${c.match.matched} (${fmt(100 * c.match.overlap, 0)}%)</span><b>Rotation</b><span>${fmt(c.match.rotation_deg, 0)}°${c.match.mirrored ? " · mirrored" : ""}</span>` : `<b>Match</b><span>${esc(c.match.reason)}</span>`) : ""}
      ${c.error ? `<b>Error</b><span>${esc(c.error)}</span>` : ""}</div>${c.match && c.match.mirrored ? `<p class="hint">The companion image is mirrored: turn on “Un-mirror images” in its device settings.</p>` : ""}`;
  return h;
}
function confirmInline(btn, msg) {
  if (btn.dataset.armed) { delete btn.dataset.armed; return true; }
  btn.dataset.armed = "1"; const old = btn.textContent; btn.textContent = "Confirm"; toast(msg, false, 5000);
  setTimeout(() => { if (btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = old; } }, 5000);
  return false;
}
function renderSel() {
  const m = S.map.data; const s = m && S.map.sel && m.sites.find(x => x.id === S.map.sel);
  $("#sel-site").innerHTML = s ? `<b>Tube</b><span>${esc(s.id)}</span><b>State</b><span>${esc(s.state)} (${fmt(s.confidence, 2)})</span>
    <b>Sheet X / Y</b><span>${fmt(s.sheet_x_mm, 1)} / ${fmt(s.sheet_y_mm, 1)} mm</span>
    ${s.machine_x_mm !== undefined ? `<b>Machine X / Y</b><span>${fmt(s.machine_x_mm, 2)} / ${fmt(s.machine_y_mm, 2)}</span>` : ""}
    <b>Pixel</b><span>${fmt(s.u, 1)}, ${fmt(s.v, 1)}</span>` : `<span class="hint" style="grid-column:1/3">Click a tube on the image.</span>`;
}
function renderAnchors() {
  const m = S.map.data; if (!m) return;
  const tf = m.transform;
  $("#anc-scale").checked = !!m.anchor_allow_scale;
  $("#anc-list").innerHTML = (m.anchors || []).length ? `<table><tr><th>Tube</th><th>X</th><th>Y</th><th>Resid.</th><th></th></tr>${m.anchors.map((a, i) => `<tr><td>${esc(a.id)}</td><td>${fmt(a.machine_x_mm, 2)}</td><td>${fmt(a.machine_y_mm, 2)}</td><td>${tf && tf.valid && tf.residuals_mm[a.id] !== undefined ? fmt(tf.residuals_mm[a.id], 2) : "–"}</td><td><button class="small" data-i="${i}">×</button></td></tr>`).join("")}</table>` +
    (tf ? (tf.valid ? `<div class="kv" style="margin-top:8px"><b>RMS / max</b><span>${fmt(tf.rms_mm, 3)} / ${fmt(tf.max_mm, 3)} mm</span><b>Rotation</b><span>${fmt(tf.rotation_deg, 3)}°</span>${tf.scale_error_pct !== null && tf.scale_error_pct !== undefined ? `<b>Scale error</b><span>${fmt(tf.scale_error_pct, 3)}%</span>` : ""}${tf.mirrored ? `<b>Mirrored</b><span>yes</span>` : ""}</div>${tf.note ? `<p class="hint">${esc(tf.note)}</p>` : ""}` : `<p class="hint">${esc(tf.reason)}</p>`) : "")
    : `<p class="hint">No anchors yet.</p>`;
  $$("#anc-list button").forEach(b => b.onclick = () => { const a = [...m.anchors]; a.splice(+b.dataset.i, 1); saveAnchors(a); });
}
async function saveAnchors(anchors) {
  try { S.map.data = await api(`/api/captures/${S.map.cid}/anchors`, { method: "PUT", json: { anchors, allow_scale: $("#anc-scale").checked } }); renderMapResult(); mapV.draw(); }
  catch (e) { toast(e.message, true); }
}
$("#anc-add").onclick = () => {
  const m = S.map.data; if (!m || !S.map.sel) return toast("Select a tube first.", true);
  const x = parseFloat($("#anc-x").value), y = parseFloat($("#anc-y").value);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return toast("Enter machine X and Y in mm.", true);
  const a = (m.anchors || []).filter(a => a.id !== S.map.sel); a.push({ id: S.map.sel, machine_x_mm: x, machine_y_mm: y });
  $("#anc-x").value = ""; $("#anc-y").value = ""; saveAnchors(a);
};
$("#anc-scale").onchange = () => { if (S.map.data) saveAnchors(S.map.data.anchors || []); };
async function loadStandards() {
  S.standards = await api("/api/standards");
  $("#std-pitch").innerHTML = S.standards.pitches_in.map(p => `<option value="${p}">`).join("");
  $("#g-tube").innerHTML = `<option value="">Pick OD + BWG to fill ID…</option>` + S.standards.tubes.map(t => `<option value='${JSON.stringify(t)}'>${t.od_in}" OD · ${t.bwg} BWG → ID ${t.id_in}" (${t.id_mm} mm)</option>`).join("");
}

// ------------------------------------------------------------------ LABEL
const labV = new Viewer($("#lab-canvas"));
const labRoi = new RoiTool(labV, pts => { S.lab.roi = pts; S.lab.dirty = true; $("#lab-roi").textContent = "ROI"; });
// Quick review: dashed = proposed and not yet checked. Doubtful proposals (not "tube", or low confidence)
// get an extra ring and can be walked one by one with N; everything else is accepted in bulk.
const LAB_LS = "tubemap.lab.";
function lsGet(k, d) { try { const v = localStorage.getItem(LAB_LS + k); return v === null ? d : v; } catch (e) { return d; } }
function lsSet(k, v) { try { localStorage.setItem(LAB_LS + k, v); } catch (e) { } }
function labThr() { return parseFloat($("#lab-doubt").value) || 0; }
function isAuto(a) { return a.src === "auto"; }
function isDoubt(a) { return isAuto(a) && (a.cls !== "TUBE" || (a.conf !== null && a.conf !== undefined && a.conf < labThr())); }
labV.draw_overlay = (ctx, v) => {
  const dim = $("#lab-dim").checked, lw = 2 * COLORS.lw;
  for (let i = 0; i < S.lab.items.length; i++) {
    const a = S.lab.items[i];
    const [x, y] = v.s(a.u, a.v); const r = Math.max(3, a.r * v.k);
    const doubt = isDoubt(a);
    ctx.globalAlpha = dim && isAuto(a) && !doubt ? 0.3 : 1;
    ctx.lineWidth = lw;
    ctx.strokeStyle = COLORS[a.cls]; ctx.setLineDash(a.cls === "NOT_TUBE" ? [3, 3] : isAuto(a) ? [6, 2] : []);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    if (a.cls === "BLOCKED") { ctx.fillStyle = COLORS.BLOCKED + "66"; ctx.fill(); }
    if (doubt) { ctx.setLineDash([]); ctx.strokeStyle = COLORS.REVIEW; ctx.lineWidth = lw * 0.75; ctx.beginPath(); ctx.arc(x, y, r + Math.max(3, lw * 1.5), 0, 7); ctx.stroke(); }
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
  const f = S.lab.items[S.lab.focus];
  if (f) {
    const [x, y] = v.s(f.u, f.v); const r = Math.max(6, f.r * v.k) + 8;
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.strokeStyle = "#000000"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r + 2, 0, 7); ctx.stroke();
  }
  const roi = labRoi.active ? labRoi.pts : S.lab.roi;
  if (roi) drawPoly(ctx, v, roi, v.iw, v.ih, "#ffd43b", !labRoi.active);
};
function labNearest(p) {
  let best = -1, bd = 1e9;
  S.lab.items.forEach((a, i) => { const d = Math.hypot(a.u - p.x, a.v - p.y); if (d < bd) { bd = d; best = i; } });
  return best >= 0 && bd <= Math.max(S.lab.items[best].r * 1.4, 6 / labV.k) ? best : -1;
}
function labPush() { S.lab.undo.push(JSON.stringify(S.lab.items)); if (S.lab.undo.length > 100) S.lab.undo.shift(); S.lab.dirty = true; }
function labRefresh() { labCounts(); labV.draw(); }
labV.onclick = (p, e) => {
  if (labRoi.click(p)) return;
  if (!S.lab.cid) return;
  const i = labNearest(p);
  if (e.shiftKey || i < 0) {
    if (!e.shiftKey) return;
    if (p.x < 0 || p.y < 0 || p.x >= labV.iw || p.y >= labV.ih) return;
    labPush(); S.lab.items.push({ id: "M" + Date.now().toString(36), u: p.x, v: p.y, r: parseFloat($("#lab-radius").value) || 6, cls: S.lab.cls, src: "manual" });
  } else { labPush(); S.lab.items[i].cls = S.lab.cls; S.lab.items[i].src = "manual"; }
  labRefresh();
};
// Shift+drag: accept everything in the box as it is. Ctrl+drag: set everything in the box to the selected class.
labV.onbox = (b, e) => {
  if (!S.lab.cid || labRoi.active) return;
  const setCls = e.ctrlKey || e.metaKey;
  const inside = S.lab.items.filter(a => a.u >= b.x0 && a.u <= b.x1 && a.v >= b.y0 && a.v <= b.y1);
  if (!inside.length) return;
  labPush();
  let n = 0;
  for (const a of inside) {
    if (setCls) { if (a.cls !== S.lab.cls || isAuto(a)) n++; a.cls = S.lab.cls; a.src = "manual"; }
    else if (isAuto(a)) { a.src = "reviewed"; n++; }
  }
  labRefresh();
  toast(setCls ? `${n} mark(s) set to ${S.lab.cls.replace("_", " ").toLowerCase()}` : `${n} mark(s) accepted`);
};
labV.oncontext = p => { const i = labNearest(p); if (i >= 0) { labPush(); S.lab.items.splice(i, 1); if (S.lab.focus === i) S.lab.focus = -1; else if (S.lab.focus > i) S.lab.focus--; labRefresh(); } };
labV.ondblclick = () => { labRoi.finish(); };
$$("#lab-classes button").forEach(b => b.onclick = () => setLabClass(b.dataset.c));
function setLabClass(c) { S.lab.cls = c; $$("#lab-classes button").forEach(b => b.classList.toggle("on", b.dataset.c === c)); }

function labAcceptAll() {
  const n = S.lab.items.filter(isAuto).length;
  if (!n) { toast("Nothing left to accept."); return; }
  labPush(); S.lab.items.forEach(a => { if (isAuto(a)) a.src = "reviewed"; }); S.lab.focus = -1;
  labRefresh(); toast(`${n} mark(s) accepted`);
}
function labDoubtOrder() {
  return S.lab.items.map((a, i) => [a, i]).filter(([a]) => isDoubt(a))
    .sort((x, y) => (x[0].conf ?? 0) - (y[0].conf ?? 0)).map(([, i]) => i);
}
function labNextDoubt() {
  const order = labDoubtOrder();
  if (!order.length) { S.lab.focus = -1; labRefresh(); toast("No doubtful marks left. Accept the rest (A) or Save & mark reviewed."); return; }
  const pos = order.indexOf(S.lab.focus);
  S.lab.focus = order[pos >= 0 ? (pos + 1) % order.length : 0];
  const a = S.lab.items[S.lab.focus];
  labV.focus(a.u, a.v, Math.min(12, 30 / Math.max(1, a.r)));
  labCounts();
}
function labFocusSet(cls) {           // cls null = keep the proposed class (confirm)
  const a = S.lab.items[S.lab.focus]; if (!a) return false;
  labPush();
  if (cls) { a.cls = cls; a.src = "manual"; } else if (isAuto(a)) a.src = "reviewed";
  labNextDoubt(); return true;
}
$("#lab-accept").onclick = labAcceptAll;
$("#lab-walk").onclick = labNextDoubt;
$("#lab-dim").checked = lsGet("dim", "0") === "1";
$("#lab-dim").onchange = e => { lsSet("dim", e.target.checked ? "1" : "0"); labV.draw(); };
$("#lab-doubt").value = lsGet("doubt", "0.85");
$("#lab-doubt").oninput = e => { lsSet("doubt", e.target.value); $("#lab-doubt-v").textContent = fmt(labThr(), 2); labRefresh(); };
$("#lab-doubt-v").textContent = fmt(labThr(), 2);

document.addEventListener("keydown", e => {
  if (S.tab !== "label" || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key === "z") { e.preventDefault(); labUndo(); return; }
  if (mod && e.key === "s") { e.preventDefault(); labSave(false); return; }
  if (mod && e.key === "Enter") { e.preventDefault(); labReviewNext(); return; }
  if (mod || e.altKey) return;
  const k = { "1": "TUBE", "2": "BLOCKED", "3": "UNKNOWN", "4": "NOT_TUBE" }[e.key];
  if (k) { setLabClass(k); if (S.lab.focus >= 0) labFocusSet(k); return; }
  const key = e.key.toLowerCase();
  if (key === "n") { e.preventDefault(); labNextDoubt(); }
  else if (key === "a") { e.preventDefault(); labAcceptAll(); }
  else if ((e.key === "Enter" || e.key === " ") && S.lab.focus >= 0) { e.preventDefault(); labFocusSet(null); }
  else if ((e.key === "Delete" || e.key === "Backspace") && S.lab.focus >= 0) {
    e.preventDefault(); labPush(); S.lab.items.splice(S.lab.focus, 1); S.lab.focus = -1; labNextDoubt();
  }
  else if (e.key === "Escape" && S.lab.focus >= 0) { S.lab.focus = -1; labRefresh(); }
});
function labUndo() { const s = S.lab.undo.pop(); if (s) { S.lab.items = JSON.parse(s); S.lab.focus = -1; labCounts(); labV.draw(); } }
$("#lab-undo").onclick = labUndo;
$("#lab-fit").onclick = () => labV.fit();
$("#lab-roi").onclick = () => { labRoi.start(); $("#lab-roi").textContent = "Drawing… (dbl-click)"; };
$$("#lab-filter button").forEach(b => b.onclick = () => { S.lab.filter = b.dataset.v; $$("#lab-filter button").forEach(x => x.classList.toggle("on", x === b)); labList(); });
async function labEnter() { await labList(); if (!S.lab.cid && S.captures.length) S.lab.cid = S.captures[0].id; if (S.lab.cid) labOpen(S.lab.cid); }
async function labList() {
  const q = S.lab.filter === "all" ? "" : `&reviewed=${S.lab.filter}`;
  const caps = await api(`/api/captures?limit=500${q}`); if (S.lab.filter === "all") S.captures = caps;
  S.lab.list = caps;
  $("#lab-list").innerHTML = caps.map(c => `<div class="capitem ${c.id === S.lab.cid ? "on" : ""}" data-id="${esc(c.id)}"><img loading="lazy" alt="" src="/api/captures/${c.id}/thumb.jpg">
    <div><div class="t">${c.name ? `<b>${esc(c.name)}</b> ` : ""}${when(c.created_at)} ${c.reviewed ? '<span class="chip ok">reviewed</span>' : ""}${(c.tags || []).includes("test") ? ' <span class="chip">test set</span>' : ""}</div><div class="s">${esc(c.device || c.source || "")} · ${c.labels} labels${c.label_counts.BLOCKED ? ` · ${c.label_counts.BLOCKED} blocked` : ""}</div></div></div>`).join("") || `<p class="hint">Nothing here yet.</p>`;
  $$("#lab-list .capitem").forEach(el => el.onclick = () => labOpen(el.dataset.id));
}
async function labOpen(cid) {
  if (S.lab.dirty && S.lab.cid && S.lab.cid !== cid) { try { await labSave(false, true); } catch (e) { } }
  S.lab.cid = cid; S.lab.undo = []; S.lab.dirty = false; S.lab.focus = -1;
  $$("#lab-list .capitem").forEach(el => el.classList.toggle("on", el.dataset.id === cid));
  const d = await api(`/api/captures/${cid}`); S.lab.meta = d.meta;
  S.lab.items = d.labels.items || []; S.lab.roi = d.labels.roi || null; $("#lab-notes").value = d.labels.notes || "";
  $("#lab-test").checked = (d.meta.tags || []).includes("test");
  $("#lab-name").value = d.meta.name || d.meta.source_filename || "";
  $("#lab-title").textContent = `${cid}`; $("#lab-zip").href = `/api/captures/${cid}/export.zip`;
  const rs = S.lab.items.map(a => a.r).sort((a, b) => a - b); $("#lab-radius").value = rs.length ? fmt(rs[rs.length >> 1], 1) : 8;
  await labV.load(`/api/captures/${cid}/preview.jpg`, d.meta.image.width, d.meta.image.height);
  $("#lab-msg").textContent = d.labels.reviewed ? `Reviewed ${when(d.labels.reviewed_at)}` : "Not reviewed yet";
  labCounts();
  if (S.lab.proposeOnOpen) { S.lab.proposeOnOpen = false; if (!S.lab.items.length) labPropose(); }
}
function labCounts() {
  const c = { TUBE: 0, BLOCKED: 0, UNKNOWN: 0, NOT_TUBE: 0 }; S.lab.items.forEach(a => c[a.cls]++);
  const auto = S.lab.items.filter(isAuto).length, doubt = S.lab.items.filter(isDoubt).length;
  $("#lab-counts").innerHTML = Object.entries(c).map(([k, v]) => `<b><i class="c-${k}" style="display:inline-block;width:9px;height:9px;border-radius:50%;border:2px solid"></i> ${k.replace("_", " ").toLowerCase()}</b><span>${v}</span>`).join("");
  $("#lab-progress").innerHTML = !S.lab.items.length ? "" :
    `<div><b>${doubt}</b> doubtful · <b>${auto}</b> unchecked · ${S.lab.items.length - auto} checked</div>` +
    (S.lab.focus >= 0 && S.lab.items[S.lab.focus] ? `<div class="hint" style="margin:2px 0 0">Selected: ${esc(S.lab.items[S.lab.focus].cls.replace("_", " ").toLowerCase())}${S.lab.items[S.lab.focus].conf != null ? ` · confidence ${fmt(S.lab.items[S.lab.focus].conf, 2)}` : ""}. 1–4 relabel · Enter keep · Del remove · N skip</div>` : "");
  $("#lab-walk").textContent = doubt ? `Check doubtful (${doubt}) · N` : "Check doubtful · N";
  $("#lab-accept").textContent = `Accept the rest${auto ? ` (${auto})` : ""} · A`;
  $("#lab-accept").disabled = !auto;
}
async function labPropose() {
  if (!S.lab.cid) return;
  const b = $("#lab-propose"); b.disabled = true; b.textContent = "Proposing…";
  try {
    const r = await api(`/api/captures/${S.lab.cid}/propose`, { method: "POST", json: { roi: S.lab.roi, geometry: S.settings && S.settings.geometry } });
    labPush(); S.lab.items = r.items; S.lab.focus = -1; labCounts(); labV.draw();
    toast(r.warning || `${r.items.length} proposals from the ${r.source}. Press N to check the doubtful ones, then A to accept the rest.`, !!r.warning, 5000);
    const rs = r.items.map(a => a.r).sort((a, b) => a - b); if (rs.length) $("#lab-radius").value = fmt(rs[rs.length >> 1], 1);
  } catch (e) { toast(e.message, true); } finally { b.disabled = false; b.textContent = "Propose"; }
}
$("#lab-propose").onclick = labPropose;
async function labSave(reviewed, quiet = false) {
  if (!S.lab.cid) return;
  const items = S.lab.items.map(a => reviewed ? { ...a, src: a.src === "auto" ? "reviewed" : a.src } : a);
  const r = await api(`/api/captures/${S.lab.cid}/labels`, { method: "PUT", json: { items, roi: S.lab.roi, reviewed, notes: $("#lab-notes").value } });
  if (reviewed) S.lab.items = items;
  S.lab.dirty = false; if (!quiet) toast(reviewed ? `Saved ${r.labels} labels · reviewed` : `Saved ${r.labels} labels`);
  $("#lab-msg").textContent = reviewed ? "Reviewed just now" : "Saved (not reviewed)";
  labCounts(); labList();
}
$("#lab-save").onclick = () => labSave(false).catch(e => toast(e.message, true));
$("#lab-review").onclick = () => labSave(true).catch(e => toast(e.message, true));
$("#lab-name").onchange = async e => {
  if (!S.lab.cid) return;
  try { S.lab.meta = await api(`/api/captures/${S.lab.cid}/meta`, { method: "PUT", json: { name: e.target.value.trim() } }); toast("Name saved"); labList(); }
  catch (err) { toast(err.message, true); }
};
$("#lab-test").onchange = async e => {
  if (!S.lab.cid) return;
  const tags = new Set((S.lab.meta && S.lab.meta.tags) || []);
  if (e.target.checked) tags.add("test"); else tags.delete("test");
  try { S.lab.meta = await api(`/api/captures/${S.lab.cid}/meta`, { method: "PUT", json: { tags: [...tags] } }); toast(e.target.checked ? "In the test set: never used for training" : "Removed from the test set"); labList(); }
  catch (err) { toast(err.message, true); }
};
async function labGoNext() { const caps = await api("/api/captures?limit=500&reviewed=0"); const n = caps.find(c => c.id !== S.lab.cid); if (n) { S.lab.proposeOnOpen = true; labOpen(n.id); } else toast("Everything is reviewed."); }
async function labReviewNext() { try { await labSave(true); await labGoNext(); } catch (e) { toast(e.message, true); } }
$("#lab-next").onclick = labGoNext;
$("#lab-review-next").onclick = labReviewNext;
$("#lab-del").onclick = async e => {
  if (!confirmInline(e.target, "Moves the capture to the trash folder. Click again to confirm.")) return;
  try { await api(`/api/captures/${S.lab.cid}`, { method: "DELETE" }); S.lab.cid = null; labV.clear(); if (S.map.cid) S.map.cid = null; labEnter(); } catch (err) { toast(err.message, true); }
};
$("#imp-img").onchange = async e => {
  const files = [...e.target.files]; let n = 0;
  for (const f of files) { try { await fetch("/api/import/image", { method: "POST", headers: { "X-Filename": encodeURIComponent(f.name) }, body: f }).then(async r => { if (!r.ok) throw new Error((await r.json()).error); }); n++; } catch (err) { toast(`${f.name}: ${err.message}`, true); } }
  toast(`Imported ${n} image(s)`); e.target.value = ""; labList();
};
$("#imp-zip").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try { const r = await fetch("/api/import/zip", { method: "POST", body: f }); const d = await r.json(); if (!r.ok) throw new Error(d.error); toast(`Imported ${d.imported} capture(s) from ${d.format}${d.skipped ? `, skipped ${d.skipped} duplicates` : ""}`); }
  catch (err) { toast(err.message, true, 6000); }
  e.target.value = ""; labList();
};

// ------------------------------------------------------------------ DEVICES
async function renderDevices(full = true) {
  if (full || !S.drivers.length) S.drivers = await api("/api/drivers");
  const devs = full ? await api("/api/devices") : S.devices;
  $("#dev-list").innerHTML = devs.map(d => {
    const drv = S.drivers.find(x => x.id === d.driver) || { label: d.driver };
    const info = d.info || {};
    return `<div class="card" data-id="${esc(d.id)}">
      <div class="head"><span class="dot ${esc(d.state)}"></span><b>${esc(d.name)}</b><span class="chip ${esc(d.state)}">${esc(d.state)}</span><span class="sub" style="margin-left:auto">${esc(d.role)}</span></div>
      <div class="sub">${esc(drv.label)}${info.model ? " · " + esc(info.model) : ""}${info.serial ? " · " + esc(info.serial) : ""}${d.resolution ? " · " + d.resolution.join("×") : ""}${d.state === "running" ? ` · ${fmt(d.fps, 1)} fps` : ""}${d.has_depth ? " · depth" : ""}</div>
      ${d.stats && d.stats.rgb_frames !== undefined ? `<div class="sub">Colour frames ${d.stats.rgb_frames} · depth frames ${d.stats.depth_frames} · paired ${d.stats.paired}${d.stats.last_pair_dt_ms !== null && d.stats.last_pair_dt_ms !== undefined ? ` (${d.stats.last_pair_dt_ms} ms apart)` : ""}${info.usb_speed ? ` · USB ${esc(({ SUPER: "3", SUPER_PLUS: "3.1", HIGH: "2: too slow for colour + depth, use a blue USB 3 port", FULL: "1.1", LOW: "1.0" })[info.usb_speed] || info.usb_speed)}` : ""}</div>` : ""}
      ${info.color_note ? `<div class="sub">${esc(info.color_note)}</div>` : ""}
      ${d.error ? `<div class="errtext">${esc(d.error)}</div>` : ""}
      <div class="btnrow"><button class="small" data-act="edit">Edit</button><button class="small" data-act="toggle">${d.enabled ? "Disable" : "Enable"}</button><button class="small" data-act="restart" ${d.enabled ? "" : "disabled"}>Restart</button><button class="small danger" data-act="remove">Remove</button></div>
    </div>`;
  }).join("") || `<p class="hint">No devices. Add one.</p>`;
  $$("#dev-list .card").forEach(card => {
    const id = card.dataset.id; const d = devs.find(x => x.id === id);
    $$("button", card).forEach(b => b.onclick = async () => {
      try {
        if (b.dataset.act === "edit") return devEdit(d);
        if (b.dataset.act === "toggle") await api(`/api/devices/${id}`, { method: "PUT", json: { enabled: !d.enabled } });
        if (b.dataset.act === "restart") await api(`/api/devices/${id}/restart`, { method: "POST" });
        if (b.dataset.act === "remove") { if (!confirmInline(b, "Removes this device's settings. Captures are kept. Click again to confirm.")) return; await api(`/api/devices/${id}`, { method: "DELETE" }); }
        setTimeout(() => refreshStatus().then(() => renderDevices()), 400);
      } catch (e) { toast(e.message, true); }
    });
  });
}
$("#dev-add").onclick = () => devEdit(null);
$("#dev-cancel").onclick = () => { S.dev.editing = null; $("#dev-editor").hidden = true; };
$("#dev-adv").onchange = () => devFields();
$("#dev-driver").onchange = () => { if (S.dev.editing === "new") { const id = $("#dev-driver").value; $("#dev-role").value = id === "orbbec" ? "depth" : id === "oak" ? "overview" : "aux"; } devFields(); };
function devEdit(d) {
  S.dev.editing = d ? d.id : "new"; $("#dev-editor").hidden = false;
  $("#dev-ed-title").textContent = d ? `Edit ${d.name}` : "Add device";
  $("#dev-driver").innerHTML = S.drivers.map(x => `<option value="${x.id}">${esc(x.label)}${x.available ? "" : " (not installed)"}</option>`).join("");
  $("#dev-driver").value = d ? d.driver : "oak"; $("#dev-driver").disabled = !!d;
  $("#dev-name").value = d ? d.name : ""; $("#dev-role").value = d ? d.role : "aux"; $("#dev-enabled").checked = d ? d.enabled : true;
  S.dev.values = d ? { ...d.settings } : {};
  $("#dev-found").innerHTML = ""; $("#dev-test-out").innerHTML = ""; devFields();
}
function devDriver() { return S.drivers.find(x => x.id === $("#dev-driver").value); }
function devFields() {
  const drv = devDriver(); if (!drv) return;
  const cur = devCollect(true);
  Object.assign(S.dev.values, cur);
  $("#dev-driver-desc").innerHTML = esc(drv.description) + (drv.available ? "" : `<br><b style="color:var(--danger)">${esc(drv.availability)}</b>`) +
    (drv.id === "agent" ? `<br>On the Persee 2 run <code>python3 tubemap_agent.py</code> (see agent/README.md), then enter its IP address here.` : "");
  const adv = $("#dev-adv").checked;
  $("#dev-fields").innerHTML = drv.schema.filter(f => adv || !f.advanced).map(f => fieldHtml(f, S.dev.values[f.key], "df-")).join("");
  if (!$("#dev-name").value && S.dev.editing === "new") $("#dev-name").placeholder = drv.label;
}
function devCollect(silent = false) {
  const drv = devDriver(); if (!drv) return {};
  const out = {}; for (const f of drv.schema) { const v = readField(f, "df-"); if (v !== undefined && !(typeof v === "number" && Number.isNaN(v))) out[f.key] = v; }
  return out;
}
$("#dev-save").onclick = async () => {
  const drv = devDriver(); const settings = { ...S.dev.values, ...devCollect() };
  try {
    if (S.dev.editing === "new") await api("/api/devices", { method: "POST", json: { name: $("#dev-name").value || drv.label, driver: drv.id, role: $("#dev-role").value, enabled: $("#dev-enabled").checked, settings } });
    else { const r = await api(`/api/devices/${S.dev.editing}`, { method: "PUT", json: { name: $("#dev-name").value, role: $("#dev-role").value, enabled: $("#dev-enabled").checked, settings } }); toast(r.restarted ? "Saved; device restarting" : r.applied_live ? "Applied without restart" : "Saved"); }
    S.dev.editing = null; $("#dev-editor").hidden = true; await refreshStatus(); renderDevices();
  } catch (e) { toast(e.message, true, 6000); }
};
$("#dev-discover").onclick = async () => {
  const drv = devDriver(); $("#dev-found").innerHTML = `<p class="hint">Searching…</p>`;
  try {
    const r = await api(`/api/drivers/${drv.id}/discover`, { method: "POST" });
    $("#dev-found").innerHTML = r.found.length ? `<h3>Found</h3>` + r.found.map((f, i) => `<button class="small" data-i="${i}">${esc(f.label)}</button>`).join("") : `<p class="hint">Nothing found. Devices already in use by TubeMap may not be listed.</p>`;
    $$("#dev-found button").forEach(b => b.onclick = () => { Object.assign(S.dev.values, r.found[+b.dataset.i].settings); devFields(); toast("Filled from discovered device"); });
  } catch (e) { $("#dev-found").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
};
$("#dev-test").onclick = async () => {
  const drv = devDriver(); const out = $("#dev-test-out"); out.innerHTML = `<p class="hint">Opening device… (it must not be running already)</p>`;
  try {
    const r = await api(`/api/drivers/${drv.id}/test`, { method: "POST", json: { settings: { ...S.dev.values, ...devCollect() } } });
    out.innerHTML = `<p><span class="chip ok">Works</span> ${r.resolution.join("×")}${r.depth ? ` · depth ${r.depth.join("×")}` : ""} · calibration ${r.intrinsics ? "yes" : "no"} · ${r.seconds}s</p><img alt="Test frame" src="${r.thumbnail}" style="width:100%;border-radius:5px">`;
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
};

// ------------------------------------------------------------------ MODEL
async function modelEnter() {
  const [st, pr] = await Promise.all([api("/api/dataset/stats"), api("/api/profiles")]);
  const c = st.reviewed_label_counts;
  $("#model-stats").innerHTML = [["Captures", st.captures], ["Reviewed", st.reviewed], ["Tube labels", c.TUBE], ["Blocked", c.BLOCKED], ["Unknown", c.UNKNOWN], ["Not a tube", c.NOT_TUBE]]
    .map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
  $("#profiles").innerHTML = pr.profiles.map(p => {
    const m = p.metrics || {}; const hd = m.test_detector || m.holdout_detector || m.train_detector; const hc = m.test_classifier || m.cv_classifier || m.holdout_classifier || m.train_classifier;
    return `<div class="card"><div class="head"><b>${esc(p.name)}</b>${p.name === pr.active ? '<span class="chip ok">active</span>' : ""}<span class="sub" style="margin-left:auto">${p.created_at ? when(p.created_at) : "built in"}</span></div>
      <div class="sub">${hd ? `Detector F1 ${fmt(hd.f1, 3)} (${m.test_detector ? "test set" : m.holdout_detector ? "held-out" : "training"})` : "Default detector"}${hc ? ` · state accuracy ${fmt(hc.accuracy, 3)}${m.test_classifier ? " (test set)" : m.cv_classifier ? " (cross-validated)" : ""}` : " · rule-based states"}</div>
      <div class="btnrow">${p.metrics ? `<button class="small" data-d="${esc(p.name)}">Details</button>` : ""}${p.name !== pr.active ? `<button class="small" data-n="${esc(p.name)}">Activate</button>` : ""}</div></div>`;
  }).join("");
  $$("#profiles button[data-n]").forEach(b => b.onclick = async () => { await api(`/api/profiles/${b.dataset.n}/activate`, { method: "POST" }); toast(`Active profile: ${b.dataset.n}`); modelEnter(); });
  $$("#profiles button[data-d]").forEach(b => b.onclick = () => { const p = pr.profiles.find(x => x.name === b.dataset.d); if (p) { showTrainResult(p); $("#train-result").scrollIntoView({ behavior: "smooth" }); } });
}
$("#train-go").onclick = async () => {
  try {
    const job = await api("/api/profiles/train", { method: "POST", json: { name: $("#train-name").value.trim() || null } });
    $("#train-result").innerHTML = ""; pollJob(job.id);
  } catch (e) { toast(e.message, true); }
};
async function pollJob(id) {
  const j = await api(`/api/jobs/${id}`);
  $("#train-prog").innerHTML = `<div class="prog"><i style="width:${Math.round(j.progress * 100)}%"></i></div><div class="hint">${esc(j.message)}</div>`;
  if (j.state === "running") return setTimeout(() => pollJob(id), 800);
  if (j.state === "error") { $("#train-prog").innerHTML = `<p class="err">${esc(j.error)}</p>`; return; }
  showTrainResult(j.result); modelEnter();
}
function showTrainResult(p) {
  const m = p.metrics || {}; const t = p.training || {};
  const pct = v => v === null || v === undefined ? "–" : (100 * v).toFixed(1) + "%";
  let h = `<h3>${esc(p.name)} <span class="hint">(${esc(t.model || "")})</span></h3>`;
  const det = (lbl, a, b) => a ? `<tr><td>${lbl}</td><td>${fmt(a.f1, 3)}</td><td>${fmt(a.precision, 3)}</td><td>${fmt(a.recall, 3)}</td><td>${b ? fmt(b.f1, 3) : "–"}</td></tr>` : "";
  h += `<div class="subhead">Opening detector</div><table><tr><th>Scored on</th><th>F1</th><th>Precision</th><th>Recall</th><th>Current F1</th></tr>
    ${det("Held-out captures", m.holdout_detector, m.baseline_holdout_detector)}${det("Test set", m.test_detector, m.baseline_test_detector)}
    ${!m.holdout_detector && !m.test_detector ? det("Training captures", m.train_detector, null) : ""}</table>`;
  const cv = m.cv_classifier;
  if (cv) {
    h += `<div class="subhead">Tube state, cross-validated (every reviewed capture held out once) · accuracy ${pct(cv.accuracy)}</div>
      <table><tr><th>Class</th><th>Recall</th><th>Precision</th><th>Examples</th><th>Captures</th></tr>` +
      Object.entries(cv.per_class).map(([k, v]) => `<tr><td>${k.replace("_", " ").toLowerCase()}</td><td>${pct(v.recall)}</td><td>${pct(v.precision)}</td><td>${v.support}</td><td>${v.captures ?? "–"}</td></tr>`).join("") + `</table>
      <p class="hint">Recall: share of real examples the model gets right. Precision: how often its call is correct. Few captures behind a class means the score can swing a lot.</p>`;
  }
  if (m.test_classifier) {
    const b = m.baseline_test_classifier;
    h += `<div class="subhead">Test set (never trained on)</div><table><tr><th>Class</th><th>New recall</th><th>Current recall</th><th>Examples</th></tr>` +
      Object.entries(m.test_classifier.per_class).map(([k, v]) => `<tr><td>${k.replace("_", " ").toLowerCase()}</td><td>${pct(v.recall)}</td><td>${b ? pct(b.per_class[k].recall) : "–"}</td><td>${v.support}</td></tr>`).join("") + `</table>`;
  }
  if ((m.possibly_incomplete || []).length) {
    h += `<div class="subhead">Check these captures for unlabelled tubes</div><div class="list">` +
      m.possibly_incomplete.map(x => `<button class="small" data-cid="${esc(x.id)}" style="justify-content:space-between">${esc(x.id)}<span class="hint" style="margin:0">${x.unlabelled_detections} detections with no label · ${x.labels} labels</span></button>`).join("") + `</div>`;
  }
  h += (t.notes || []).map(n => `<p class="hint">${esc(n)}</p>`).join("");
  h += `<div class="btnrow"><button class="primary" id="train-activate">Activate ${esc(p.name)}</button></div>`;
  $("#train-result").innerHTML = h;
  $$("#train-result button[data-cid]").forEach(b => b.onclick = () => { S.lab.cid = b.dataset.cid; showTab("label"); });
  $("#train-activate").onclick = async () => { await api(`/api/profiles/${p.name}/activate`, { method: "POST" }); toast("Activated"); modelEnter(); };
}

// ------------------------------------------------------------------ SETTINGS
async function settingsEnter() {
  renderLook();
  S.settings = await api("/api/settings"); const s = S.settings;
  $("#s-customer").value = s.job.customer || ""; $("#s-site").value = s.job.site || ""; $("#s-exchanger").value = s.job.exchanger || ""; $("#s-jobnotes").value = s.job.notes || "";
  for (const [k, v] of Object.entries(s.tolerances)) { const el = $("#t-" + k); if (el) { if (el.type === "checkbox") el.checked = !!v; else el.value = v; } }
  $("#s-live_max_width").value = s.live_max_width; $("#s-live_fps").value = s.live_fps;
  const st = S.status || await api("/api/status");
  $("#sys").innerHTML = `<b>Version</b><span>${esc(st.version)}</span><b>Computer</b><span>${esc(st.hostname)}</span><b>Data folder</b><span>${esc(st.data_dir)}</span><b>Disk free</b><span>${fmt(st.disk_free_gb, 1)} of ${fmt(st.disk_total_gb, 0)} GB</span><b>Active profile</b><span>${esc(st.active_profile)}</span>`;
}
$("#s-save").onclick = async () => {
  const s = S.settings; s.job = { customer: $("#s-customer").value, site: $("#s-site").value, exchanger: $("#s-exchanger").value, notes: $("#s-jobnotes").value };
  for (const k of Object.keys(s.tolerances)) { const el = $("#t-" + k); if (!el) continue; if (el.type === "checkbox") s.tolerances[k] = el.checked; else if (el.value !== "") s.tolerances[k] = parseFloat(el.value); }
  s.live_max_width = parseInt($("#s-live_max_width").value, 10); s.live_fps = parseFloat($("#s-live_fps").value);
  try { S.settings = await api("/api/settings", { method: "PUT", json: s }); $("#s-msg").textContent = "Saved"; setTimeout(() => $("#s-msg").textContent = "", 2500); } catch (e) { toast(e.message, true); }
};

// ------------------------------------------------------------------ APPEARANCE
const LOOK_DEFAULT = { theme: "auto", accent: "", text_scale: 1, density: "normal", line_scale: 1, colors: {} };
const LOOK_COLORS = {
  map: [["--s-open", "Open"], ["--s-plug", "Plugged"], ["--s-review", "Review"], ["--s-none", "No tube"],
        ["--s-anchor", "Anchor"], ["--d-face", "Depth: surface"], ["--d-recess", "Depth: open"], ["--d-conflict", "Depth disagrees"]],
  label: [["--c-tube", "Tube"], ["--c-blocked", "Blocked"], ["--c-unknown", "Unknown"], ["--c-not", "Not a tube"]],
};
const DENSITY = { compact: 0.7, normal: 1, roomy: 1.3 };
let LOOK = JSON.parse(JSON.stringify(LOOK_DEFAULT));
function inkFor(hex) {
  const n = parseInt(hex.slice(1), 16); const r = n >> 16 & 255, g = n >> 8 & 255, b = n & 255;
  const L = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  return 0.2126 * L(r) + 0.7152 * L(g) + 0.0722 * L(b) > 0.4 ? "#000000" : "#ffffff";
}
function lookVars(look) {
  const v = { "--text-scale": String(look.text_scale || 1), "--dens": String(DENSITY[look.density] || 1), "--line-scale": String(look.line_scale || 1) };
  if (look.accent) { v["--accent"] = look.accent; v["--accent-ink"] = inkFor(look.accent); }
  for (const [k, c] of Object.entries(look.colors || {})) if (c) v[k] = c;
  return v;
}
function applyLook(look) {
  const r = document.documentElement;
  for (const k of [...r.style]) if (k.startsWith("--")) r.style.removeProperty(k);
  if (look.theme && look.theme !== "auto") r.dataset.theme = look.theme; else delete r.dataset.theme;
  const vars = lookVars(look);
  for (const [k, v] of Object.entries(vars)) r.style.setProperty(k, v);
  try { localStorage.setItem("tubemap.look", JSON.stringify({ theme: look.theme, vars })); } catch (e) { }
  COLORS = STATE_COLORS(); mapV.draw(); labV.draw();
}
function toHex(c) {
  const ctx = document.createElement("canvas").getContext("2d"); ctx.fillStyle = "#000"; ctx.fillStyle = c; return ctx.fillStyle;
}
function renderLook() {
  $$("#look-theme button").forEach(b => b.classList.toggle("on", b.dataset.v === (LOOK.theme || "auto")));
  $$("#look-accent .sw").forEach(b => b.classList.toggle("on", b.dataset.v === (LOOK.accent || "")));
  $("#look-accent-custom").value = toHex(getComputedStyle(document.documentElement).getPropertyValue("--accent").trim());
  $("#look-text").value = LOOK.text_scale || 1; $("#look-text-val").textContent = Math.round(100 * (LOOK.text_scale || 1)) + "%";
  $$("#look-density button").forEach(b => b.classList.toggle("on", b.dataset.v === (LOOK.density || "normal")));
  $("#look-line").value = LOOK.line_scale || 1; $("#look-line-val").textContent = (LOOK.line_scale || 1) + "×";
  const cs = getComputedStyle(document.documentElement);
  for (const [grid, list] of [["#look-map-colors", LOOK_COLORS.map], ["#look-label-colors", LOOK_COLORS.label]]) {
    $(grid).innerHTML = list.map(([k, lbl]) => `<label><input type="color" data-var="${k}" value="${toHex(cs.getPropertyValue(k).trim())}">${lbl}</label>`).join("");
    $$(`${grid} input`).forEach(inp => inp.oninput = () => { LOOK.colors = { ...(LOOK.colors || {}), [inp.dataset.var]: inp.value }; applyLook(LOOK); });
  }
}
function lookChanged() { applyLook(LOOK); renderLook(); $("#look-msg").textContent = "Not saved yet"; }
$$("#look-theme button").forEach(b => b.onclick = () => { LOOK.theme = b.dataset.v; lookChanged(); });
$$("#look-accent .sw").forEach(b => b.onclick = () => { LOOK.accent = b.dataset.v; lookChanged(); });
$("#look-accent-custom").oninput = e => { LOOK.accent = e.target.value; applyLook(LOOK); $("#look-msg").textContent = "Not saved yet"; };
$("#look-text").oninput = e => { LOOK.text_scale = parseFloat(e.target.value); applyLook(LOOK); $("#look-text-val").textContent = Math.round(100 * LOOK.text_scale) + "%"; $("#look-msg").textContent = "Not saved yet"; };
$$("#look-density button").forEach(b => b.onclick = () => { LOOK.density = b.dataset.v; lookChanged(); });
$("#look-line").oninput = e => { LOOK.line_scale = parseFloat(e.target.value); applyLook(LOOK); $("#look-line-val").textContent = LOOK.line_scale + "×"; $("#look-msg").textContent = "Not saved yet"; };
$("#look-save").onclick = async () => {
  try { await api("/api/settings", { method: "PUT", json: { appearance: LOOK } }); $("#look-msg").textContent = "Saved"; }
  catch (e) { toast(e.message, true); }
};
$("#look-reset").onclick = async () => {
  LOOK = JSON.parse(JSON.stringify(LOOK_DEFAULT)); applyLook(LOOK); renderLook();
  try { await api("/api/settings", { method: "PUT", json: { appearance: LOOK } }); $("#look-msg").textContent = "Defaults restored"; } catch (e) { toast(e.message, true); }
};

// ------------------------------------------------------------------ boot
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { COLORS = STATE_COLORS(); mapV.draw(); labV.draw(); });
window.addEventListener("beforeunload", e => { if (S.lab.dirty) { e.preventDefault(); e.returnValue = ""; } });
(async function boot() {
  try {
    S.settings = await api("/api/settings"); setGeometryForm(S.settings.geometry);
    LOOK = { ...JSON.parse(JSON.stringify(LOOK_DEFAULT)), ...(S.settings.appearance || {}) }; applyLook(LOOK);
  } catch (e) { }
  try { S.drivers = await api("/api/drivers"); } catch (e) { }
  await refreshStatus(); renderLive(); loadStandards().catch(() => { });
})();
