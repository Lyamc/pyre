// pyre web UI — vanilla JS, talks only to pyre's public API (+ /api/* management).
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const LS = { get: (k, d) => { try { return JSON.parse(localStorage.getItem("pyre." + k)) ?? d; } catch { return d; } }, set: (k, v) => localStorage.setItem("pyre." + k, JSON.stringify(v)) };
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const human = n => n >= 1e9 ? (n / 1e9).toFixed(2) + " GB" : n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : (n / 1e3).toFixed(0) + " KB";

// ---------- API ----------
function headers(extra = {}) { const k = LS.get("apikey", ""); return Object.assign(k ? { Authorization: "Bearer " + k } : {}, extra); }
async function api(path, body, method) {
  const r = await fetch(path, { method: method || (body ? "POST" : "GET"), headers: headers(body ? { "Content-Type": "application/json" } : {}), body: body ? JSON.stringify(body) : undefined });
  if (r.status === 401) { askKey(); throw new Error("unauthorized (set the API key with 🔑)"); }
  const t = await r.text(); let j; try { j = JSON.parse(t); } catch { j = { raw: t }; }
  if (!r.ok) throw new Error(j?.error?.message || t || r.statusText);
  return j;
}
async function* sse(path, body, signal) {
  const r = await fetch(path, { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify(body), signal });
  if (r.status === 401) { askKey(); throw new Error("unauthorized"); }
  if (!r.ok) { const t = await r.text(); let m = t; try { m = JSON.parse(t).error.message; } catch { } throw new Error(m); }
  const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await rd.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    let i; while ((i = buf.indexOf("\n\n")) >= 0) {
      const ev = buf.slice(0, i); buf = buf.slice(i + 2);
      for (const line of ev.split("\n")) if (line.startsWith("data:")) { const d = line.slice(5).trim(); if (d === "[DONE]") return; try { yield JSON.parse(d); } catch { } }
    }
  }
}
function askKey() { const k = prompt("API key for this pyre server (stored in this browser):", LS.get("apikey", "")); if (k !== null) { LS.set("apikey", k.trim()); refreshAll(); } }

function showToast(msg, ms = 8000) {
  let box = $("#pyreToasts");
  if (!box) {
    box = document.createElement("div"); box.id = "pyreToasts"; box.className = "pyre-toasts";
    document.body.appendChild(box);
  }
  const el = document.createElement("div"); el.className = "pyre-toast"; el.textContent = String(msg);
  box.appendChild(el);
  setTimeout(() => el.remove(), ms);
}
window.addEventListener("error", (e) => {
  const m = (e && e.message) || "script error";
  console.error(e);
  try { showToast("UI error: " + m); } catch {}
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e && e.reason; const m = (r && r.message) || String(r || "unhandled rejection");
  console.error(e);
  try { showToast("UI error: " + m); } catch {}
});

// ---------- availability (generic disabled-reason) ----------
// reason_kind: busy | loading | missing_model | no_permission | offline | ""
function setAvail(el, { enabled, reason_kind = "", message = "" } = {}) {
  if (!el) return;
  const on = !!enabled;
  if ("disabled" in el) el.disabled = !on;
  el.dataset.avail = on ? "1" : "0";
  el.dataset.reasonKind = on ? "" : (reason_kind || "");
  if (on) {
    if (el.dataset._reasonTitle === "1") { el.title = el.dataset.availBaseTitle || ""; delete el.dataset._reasonTitle; }
  } else {
    if (!el.dataset.availBaseTitle) el.dataset.availBaseTitle = el.title || "";
    el.title = message || reasonMessage(reason_kind) || el.dataset.availBaseTitle || "Unavailable";
    el.dataset._reasonTitle = "1";
  }
  el.setAttribute("aria-disabled", on ? "false" : "true");
}
function reasonMessage(kind) {
  return ({
    busy: "Busy - wait for the current job to finish",
    loading: "Loading...",
    missing_model: "Required model is not loaded or not downloaded",
    no_permission: "You do not have permission for this action",
    offline: "Server unreachable",
  })[kind] || "";
}
function availBusy(message) { return { enabled: false, reason_kind: "busy", message }; }
function availMissing(message) { return { enabled: false, reason_kind: "missing_model", message }; }
function availOk() { return { enabled: true, reason_kind: "", message: "" }; }

// ---------- global status pill (one shared SSE; avoids HTTP/1.1 6-conn starvation) ----------
let statusState = { state: "down", headline: "Connecting...", jobs: [], vram: {}, chat_active: [] };
let statusCtrl = null;
function renderStatusPill() {
  const pill = $("#statusPill"); const text = $("#statusPillText"); const pop = $("#statusPopJobs"); const vram = $("#statusPopVram");
  if (!pill || !text) return;
  pill.classList.remove("idle", "busy", "down");
  pill.classList.add(statusState.state === "busy" ? "busy" : statusState.state === "idle" ? "idle" : "down");
  text.textContent = statusState.headline || (statusState.state === "idle" ? "Idle" : "…");
  pill.title = statusState.headline || "Server status";
  if (pop) {
    const jobs = statusState.jobs || [];
    const chat = statusState.chat_active || [];
    if (!jobs.length && !chat.length) pop.innerHTML = `<div class="muted">Idle</div>`;
    else {
      pop.innerHTML = jobs.map(j => {
        const eta = (j.eta_s != null && j.eta_s >= 0.5) ? ` · ${Math.max(0, Math.floor(j.eta_s / 60))}:${String(Math.max(0, Math.floor(j.eta_s % 60))).padStart(2, "0")} left` : (j.eta_s != null && j.fraction != null && j.fraction >= 0.98) ? ` · 0:00 left` : (j.fraction != null ? ` · ${Math.round(j.fraction * 100)}%` : "");
        const prog = (j.current != null && j.total) ? ` ${j.current}/${j.total}` : "";
        return `<div class="spop-job"><span>${esc(j.label || j.kind)}${prog}${eta}</span>${j.cancelable ? `<button type="button" data-acancel="${esc(j.id)}">Cancel</button>` : ""}</div>`;
      }).join("") + chat.map(c => `<div class="spop-job"><span>Chat ${esc(c.phase || "generating")} · ${esc(String(c.model || "").split("/").pop())}</span></div>`).join("");
    }
  }
  if (vram) {
    const v = statusState.vram || {};
    const bits = [];
    if (v.used_mib != null && v.total_mib != null) bits.push(`VRAM ${v.used_mib}/${v.total_mib} MiB`);
    else if (v.free_mib != null) bits.push(`VRAM free ${v.free_mib} MiB`);
    if (v.util_pct != null) bits.push(`GPU ${v.util_pct}%`);
    vram.textContent = bits.join(" · ");
  }
  refreshAvailFromStatus();
}
function refreshAvailFromStatus() {
  // NEVER disable tab navigation — tabs must work offline / while Connecting.
  const down = statusState.state === "down";
  const sttBusy = (typeof sttJob !== "undefined") && !!sttJob;
  const kindsObj = (typeof kinds !== "undefined") ? kinds : {};
  if ($("#sttGo") && !sttBusy) {
    if (typeof updateSttRunAvail === "function") updateSttRunAvail();
    else if (down) setAvail($("#sttGo"), { enabled: false, reason_kind: "offline", message: statusState.headline || "Server disconnected" });
    else if ($("#sttGo").dataset.forceBusy !== "1") setAvail($("#sttGo"), availOk());
  }
  if ($("#imgGo")) {
    const hasImg = !!(kindsObj.image && kindsObj.image.length);
    if (down) setAvail($("#imgGo"), { enabled: false, reason_kind: "offline", message: statusState.headline || "Server disconnected — reconnecting..." });
    else if (!hasImg) setAvail($("#imgGo"), availMissing("No image model loaded — open Models or run a Cabin card (loads on demand)"));
    else setAvail($("#imgGo"), availOk());
  }
  if ($("#sendBtn")) {
    const chatBusy = (typeof ctrl !== "undefined") && !!ctrl;
    if (down) setAvail($("#sendBtn"), { enabled: false, reason_kind: "offline", message: statusState.headline || "Server disconnected — reconnecting..." });
    else if (chatBusy) setAvail($("#sendBtn"), availBusy("Chat is generating — stop it first or wait"));
    else setAvail($("#sendBtn"), availOk());
  }
}
async function connectStatusStream() {
  if (statusCtrl) statusCtrl.abort();
  statusCtrl = new AbortController();
  const signal = statusCtrl.signal;
  let gotEvent = false;
  const CONNECT_MS = 5000;
  const timer = setTimeout(() => {
    if (gotEvent || signal.aborted) return;
    statusState.state = "down";
    statusState.headline = "Can't reach pyre - retrying (no status yet)";
    renderStatusPill();
  }, CONNECT_MS);
  const backoff = (ms) => new Promise(r => setTimeout(r, ms));
  let delay = 1000;
  while (!signal.aborted) {
    try {
      statusState.headline = statusState.headline || "Connecting...";
      renderStatusPill();
      const r = await fetch("/api/status/stream", { headers: headers(), signal });
      if (!r.ok) throw new Error("HTTP " + r.status);
      if (!r.body) throw new Error("empty body");
      const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
      delay = 1000;
      for (;;) {
        const { value, done } = await rd.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, i); buf = buf.slice(i + 2);
          for (const line of block.split("\n")) if (line.startsWith("data:")) {
            try {
              const j = JSON.parse(line.slice(5).trim());
              gotEvent = true; clearTimeout(timer);
              statusState = Object.assign({}, statusState, j);
              if (!j.state) statusState.state = ((statusState.jobs || []).length || (statusState.chat_active || []).length) ? "busy" : "idle";
              renderStatusPill();
            } catch (parseErr) { console.warn("status parse", parseErr); }
          }
        }
      }
      throw new Error("stream closed");
    } catch (e) {
      if (signal.aborted) break;
      clearTimeout(timer);
      const reason = (e && e.name === "AbortError") ? "aborted" : ((e && e.message) || String(e));
      statusState.state = "down";
      statusState.headline = "Can't reach pyre - retrying (" + reason + ")";
      statusState.jobs = []; statusState.chat_active = [];
      renderStatusPill();
      await backoff(delay);
      delay = Math.min(delay * 1.5, 8000);
    }
  }
  clearTimeout(timer);
}
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {});
  const bootPill = () => {
    if ($("#statusPill")) {
      $("#statusPill").onclick = (e) => {
        e.stopPropagation();
        const pop = $("#statusPop"); if (!pop) return;
        const closing = !pop.classList.contains("hidden");
        pop.classList.toggle("hidden", closing);
        pop.hidden = closing;
      };
      document.addEventListener("click", (e) => {
        const pop = $("#statusPop"); if (!pop || pop.classList.contains("hidden")) return;
        if (!e.target.closest("#statusPop") && !e.target.closest("#statusPill")) { pop.classList.add("hidden"); pop.hidden = true; }
      });
      $("#statusPop") && $("#statusPop").addEventListener("click", async (e) => {
        const id = e.target.dataset && e.target.dataset.acancel; if (!id) return;
        try { await api("/api/activity/cancel", { id }); } catch (err) { console.warn(err); }
      });
    }
  };
  bootPill();
}


// ---------- tiny markdown ----------
function md(src) {
  const parts = src.split(/```/); let html = "";
  parts.forEach((p, i) => {
    if (i % 2) { const nl = p.indexOf("\n"); const lang = nl > 0 ? p.slice(0, nl).trim() : ""; const code = nl > 0 ? p.slice(nl + 1) : p;
      html += `<pre><button class="copy small">copy</button><code data-lang="${esc(lang)}">${esc(code.replace(/\n$/, ""))}</code></pre>`; return; }
    let t = esc(p);
    t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
      .replace(/^(#{1,6}) (.*)$/gm, (_, h, t) => `<h${Math.min(h.length + 2, 6)}>${t}</h${Math.min(h.length + 2, 6)}>`).replace(/^(-{3,}|\*{3,})$/gm, "<hr>").replace(/^&gt; ?(.*)$/gm, "<blockquote>$1</blockquote>").replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    const lines = t.split("\n"); let out = "", list = null;
    for (let li = 0; li < lines.length; li++) {
      const l = lines[li];
      if (/^\s*\|.*\|\s*$/.test(l) && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[li + 1] || "") && (lines[li + 1] || "").includes("-")) { // GFM table
        if (list) { out += `</${list}>`; list = null; }
        const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
        out += "<table class=\"mdt\"><thead><tr>" + cells(l).map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>"; li += 2;
        while (li < lines.length && /^\s*\|/.test(lines[li])) { out += "<tr>" + cells(lines[li]).map(c => `<td>${c}</td>`).join("") + "</tr>"; li++; }
        li--; out += "</tbody></table>"; continue;
      }
      const m = l.match(/^\s*([-*]|\d+\.)\s+(.*)/);
      if (m) { const tag = /\d/.test(m[1]) ? "ol" : "ul"; if (list !== tag) { if (list) out += `</${list}>`; out += `<${tag}>`; list = tag; } out += `<li>${m[2]}</li>`; continue; }
      if (!l.trim()) continue; // blank lines don't break a list
      if (list && /^\s{2,}/.test(l)) { out = out.replace(/<\/li>$/, `<br>${l.trim()}</li>`); continue; }
      if (list) { out += `</${list}>`; list = null; }
      out += /^<(h\d|hr|blockquote)/.test(l) ? l : `<p>${l}</p>`;
    }
    if (list) out += `</${list}>`; html += out;
  });
  return html;
}
// Qwen3-style reasoning: <think>…</think> (or the API's reasoning_content) shown collapsed.
function splitThink(text) {
  const m = text.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
  if (!m) { const k = text.indexOf("</think>"); return k >= 0 ? { think: text.slice(0, k), done: true, answer: text.slice(k + 8) } : { think: "", done: true, answer: text }; }
  return { think: m[1], done: !!m[2], answer: m[3] };
}
function renderAssistant(m) {
  const t = splitThink(m.content); const think = (m.reasoning || "") + t.think;
  let h = "";
  if (think.trim()) h += `<details class="think"><summary>${t.done && m.content.length > 0 && (t.answer.trim() || !m.streaming) ? "Thought process" : "Thinking…"} <span class="muted">(${m.rinfo ? rinfoText(m.rinfo) : think.trim().split(/\s+/).length + " words"})</span></summary><div class="thinkbody">${md(think.trim())}</div></details>`;
  h += md(t.answer);
  if (m.error) h += `<div class="err">⚠ ${esc(m.error)}</div>`;
  if (m.streaming && !t.answer && !think) h += `<span class="dots">…</span>`;
  return h;
}
function plainAnswer(m) { return splitThink(m.content).answer.trim(); }
document.addEventListener("click", e => { if (e.target.classList.contains("copy")) { navigator.clipboard.writeText(e.target.nextElementSibling.textContent); e.target.textContent = "copied"; setTimeout(() => e.target.textContent = "copy", 1000); } });

// ---------- tabs / theme ----------
// ---------- Audio section toggles (STT | STS | TTS), independent ----------
function audioLayoutIsWide() {
  const el = document.querySelector(".audio-layout");
  if (el && typeof el.clientWidth === "number" && el.clientWidth > 0) return el.clientWidth >= 1100;
  return window.matchMedia("(min-width: 1280px)").matches;
}
function defaultAudioSecs() {
  const wide = audioLayoutIsWide();
  return { stt: true, sts: !!wide, tts: !!wide };
}
function loadAudioSecs() {
  try {
    const raw = localStorage.getItem("pyre.audioSecs");
    if (raw) {
      const o = JSON.parse(raw);
      return {
        stt: o.stt !== false,
        sts: !!o.sts,
        tts: !!o.tts,
      };
    }
  } catch {}
  return defaultAudioSecs();
}
function saveAudioSecs(secs) {
  try { localStorage.setItem("pyre.audioSecs", JSON.stringify(secs)); } catch {}
}
function applyAudioSecs(secs) {
  ["stt", "sts", "tts"].forEach((k) => {
    const on = !!secs[k];
    const btn = document.querySelector('.audio-tog[data-audio-sec="' + k + '"]');
    const card = document.querySelector('.audio-sec[data-audio-sec="' + k + '"]');
    if (btn) btn.setAttribute("aria-pressed", on ? "true" : "false");
    if (card) {
      if (on) card.removeAttribute("hidden");
      else card.setAttribute("hidden", "");
    }
  });
}
function initAudioSectionToggles() {
  const bar = $("#audioToggles"); if (!bar) return;
  let secs = loadAudioSecs();
  applyAudioSecs(secs);
  bar.addEventListener("click", (e) => {
    const b = e.target.closest("[data-audio-sec]"); if (!b) return;
    const k = b.dataset.audioSec;
    secs = loadAudioSecs();
    secs[k] = !(secs[k]);
    if (!secs.stt && !secs.sts && !secs.tts) secs[k] = true;
    saveAudioSecs(secs);
    applyAudioSecs(secs);
  });
  bar.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== " " && e.key !== "Enter") return;
    const tabs = $$("#audioToggles .audio-tog"); if (!tabs.length) return;
    let i = tabs.indexOf(document.activeElement);
    if (i < 0) i = 0;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      i = e.key === "ArrowRight" ? (i + 1) % tabs.length : (i - 1 + tabs.length) % tabs.length;
      tabs[i].focus();
      return;
    }
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      tabs[i].click();
    }
  });
  try {
    if (!localStorage.getItem("pyre.audioSecs")) {
      const ro = new ResizeObserver(() => {
        if (localStorage.getItem("pyre.audioSecs")) return;
        applyAudioSecs(defaultAudioSecs());
      });
      const lay = document.querySelector(".audio-layout");
      if (lay) ro.observe(lay);
    }
  } catch {}
}
initAudioSectionToggles();

$$("#tabs button").forEach(b => b.onclick = () => { $$("#tabs button").forEach(x => x.classList.toggle("active", x === b)); $$(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab)); location.hash = b.dataset.tab; onTab(b.dataset.tab); });
function onTab(t) { /* never await ? tab switch must stay instant even when backend is busy */
  if (t === "models") loadModels(); if (t === "status") loadStatus(); if (t === "images" || t === "audio") loadKinds(); if (t === "apps") loadApps(); if (t === "cabin" && window.cabin && cabin.refreshStatus) cabin.refreshStatus(); }
document.documentElement.dataset.theme = LS.get("theme", matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
$("#themeBtn").onclick = () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = t; LS.set("theme", t); };
$("#keyBtn").onclick = askKey;

// ---------- chat ----------
let convs = LS.get("convs", []); let cur = LS.get("cur", null); let ctrl = null;
const settingsIds = ["sysPrompt", "sTemp", "sTopP", "sTopK", "sMax", "sSeed"];
const saved = LS.get("settings", {}); settingsIds.forEach(id => { if (saved[id] !== undefined) $("#" + id).value = saved[id]; $("#" + id).onchange = () => { const s = {}; settingsIds.forEach(i => s[i] = $("#" + i).value); LS.set("settings", s); paintProv(); }; });
["sCtx", "sCtxN", "sKv"].forEach(id => { const el = $("#" + id); if (el) el.addEventListener("input", paintProv); });
$("#settingsBtn").onclick = () => { $("#settings").classList.toggle("hidden"); if (!$("#settings").classList.contains("hidden")) { loadQwenCtx(); loadProfiles(); } };
async function loadQwenCtx() {
  try {
    const c = await api("/api/qwen/context");
    const sel = $("#sCtx");
    const cur = String(c.context_tokens);
    sel.innerHTML = (c.presets || []).map(p => `<option value="${esc(p.id)}" data-tokens="${p.tokens}"${p.tokens === c.context_tokens ? " selected" : ""}>${esc(p.id)} (${p.tokens}${p.yarn ? ", YaRN" : ""})</option>`).join("");
    if (![...sel.options].some(o => o.selected)) {
      const o = document.createElement("option"); o.value = cur; o.textContent = cur + " tokens"; o.selected = true; sel.appendChild(o);
    }
    $("#sKv").value = String(c.kv_bits || 4);
    const fit = c.fit || {};
    $("#ctxFit").textContent = (fit.message || "") + (c.reload_required ? " Saved size needs a model reload before YaRN changes." : "");
    paintProv();
  } catch (e) { $("#ctxFit").textContent = e.message; }
}
let profState = null;
function sameVal(a, b) {
  if (a == null && b == null) return true;
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-6;
  return String(a ?? "") === String(b ?? "");
}
function contextTokensNow() {
  const custom = $("#sCtxN").value.trim();
  if (custom) return +custom;
  const opt = $("#sCtx").selectedOptions && $("#sCtx").selectedOptions[0];
  if (opt && opt.dataset.tokens) return +opt.dataset.tokens;
  return profState && profState.active && profState.defaults ? profState.defaults.context_tokens : 131072;
}
function currentSettings() {
  return {
    temperature: +$("#sTemp").value, top_p: +$("#sTopP").value, top_k: +$("#sTopK").value,
    max_tokens: +$("#sMax").value, seed: $("#sSeed").value === "" ? null : +$("#sSeed").value,
    context_tokens: contextTokensNow(), kv_bits: +$("#sKv").value, system: $("#sysPrompt").value.trim(),
  };
}
function sourceValues(src, name) {
  if (!profState) return null;
  if (src === "default") return profState.defaults;
  if (src === "tune") return profState.tune;
  return profState.customs && profState.customs[name];
}
function paintProv() {
  if (!profState || !profState.defaults) return;
  const cur = currentSettings();
  const active = profState.active || { source: "default" };
  const base = sourceValues(active.source, active.name);
  document.querySelectorAll(".prov").forEach(el => {
    const k = el.dataset.for;
    const d = profState.defaults[k];
    const t = profState.tune ? profState.tune[k] : undefined;
    let label = "modified";
    if (profState.tune && sameVal(cur[k], t) && !sameVal(t, d)) label = "tune";
    else if (sameVal(cur[k], d)) label = "default";
    el.dataset.prov = label;
    el.textContent = label;
    el.classList.toggle("edited", !!(base && !sameVal(cur[k], base[k])));
  });
}
function applySettings(v) {
  if (!v) return;
  $("#sTemp").value = v.temperature;
  $("#sTopP").value = v.top_p;
  $("#sTopK").value = v.top_k;
  $("#sMax").value = v.max_tokens;
  $("#sSeed").value = v.seed == null ? "" : v.seed;
  $("#sysPrompt").value = v.system || "";
  $("#sKv").value = String(v.kv_bits || 4);
  const tok = v.context_tokens;
  const opt = [...$("#sCtx").options].find(o => +o.dataset.tokens === tok);
  if (opt) { $("#sCtx").value = opt.value; $("#sCtxN").value = ""; }
  else { $("#sCtxN").value = tok || ""; }
  const s = {}; settingsIds.forEach(i => s[i] = $("#" + i).value); LS.set("settings", s);
}
function markProfileSource() {
  const a = (profState && profState.active) || { source: "default" };
  $("#profDefault").classList.toggle("active", a.source === "default");
  $("#profTune").classList.toggle("active", a.source === "tune");
  const sel = $("#profCustom");
  const names = Object.keys((profState && profState.customs) || {}).sort();
  const keep = a.source === "custom" ? a.name : sel.value;
  sel.innerHTML = `<option value="">Custom…</option>` + names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (keep && names.includes(keep)) sel.value = keep;
  sel.classList.toggle("active", a.source === "custom");
  $("#profDelete").classList.toggle("hidden", !(a.source === "custom" && a.name));
  const note = $("#profNote");
  if (profState && profState.tune_stale) note.textContent = "Hardware changed since autotune was recorded. The tune snapshot is still here and was not overwritten. Each row says default, tune, or modified.";
  else note.textContent = "Each setting is marked default, tune, or modified. Default is built in. Tune is the autotune snapshot. A custom name is saved with the disk icon and does not replace either.";
  paintProv();
}
async function loadProfiles() {
  try { profState = await api("/api/settings/profiles"); markProfileSource(); }
  catch (e) { $("#profNote").textContent = e.message; }
}
async function selectProfile(source, name) {
  const r = await api("/api/settings/profiles/select", { source, name: name || undefined });
  profState = r;
  if (r.values) applySettings(r.values);
  markProfileSource();
  loadQwenCtx();
}
$("#profDefault").onclick = async () => { try { await selectProfile("default"); } catch (e) { $("#profNote").textContent = e.message; } };
$("#profTune").onclick = async () => { try { await selectProfile("tune"); } catch (e) { $("#profNote").textContent = e.message; } };
$("#profCustom").onchange = async () => {
  const name = $("#profCustom").value;
  if (!name) return;
  try { await selectProfile("custom", name); } catch (e) { $("#profNote").textContent = e.message; }
};
$("#profSave").onclick = async () => {
  const name = prompt("Name this custom preset. Default and tune stay as they are.", $("#profCustom").value || "");
  if (!name) return;
  const body = { name, values: currentSettings(), replace: false };
  try {
    profState = await api("/api/settings/profiles/custom", body);
  } catch (e) {
    if (!/already exists/.test(e.message) || !confirm(e.message)) { $("#profNote").textContent = e.message; return; }
    try { profState = await api("/api/settings/profiles/custom", Object.assign(body, { replace: true })); }
    catch (e2) { $("#profNote").textContent = e2.message; return; }
  }
  try { await selectProfile("custom", name.trim()); } catch (e) { $("#profNote").textContent = e.message; markProfileSource(); }
};
$("#profTuneSave").onclick = async () => {
  const exists = profState && profState.tune;
  if (exists && !confirm("Replace the tune snapshot with the current settings? Default and custom presets stay as they are.")) return;
  try { profState = await api("/api/settings/profiles/tune", { values: currentSettings() }); markProfileSource(); }
  catch (e) { $("#profNote").textContent = e.message; }
};
$("#profDelete").onclick = async () => {
  const name = profState && profState.active && profState.active.name;
  if (!name || !confirm(`Delete custom preset “${name}”? Default and tune stay as they are.`)) return;
  try { profState = await api("/api/settings/profiles/custom/delete", { name }); markProfileSource(); }
  catch (e) { $("#profNote").textContent = e.message; }
};
$("#ctxApply").onclick = async () => {
  const custom = $("#sCtxN").value.trim();
  const context = custom || $("#sCtx").value;
  try {
    const r = await api("/api/qwen/context", { context, kv_bits: +$("#sKv").value });
    $("#ctxFit").textContent = r.message || "saved";
    loadQwenCtx();
  } catch (e) { $("#ctxFit").textContent = e.message; }
};
async function loadTune() {
  try {
    const a = await api("/api/autotune");
    const b = $("#tuneBanner");
    if (!a.suggest || !a.message) { b.classList.add("hidden"); b.innerHTML = ""; return; }
    b.classList.remove("hidden");
    b.innerHTML = `<div>${esc(a.message)}</div><div class="row" style="margin-top:8px"><button type="button" id="tuneAck" class="ghost small">Keep these results</button><button type="button" id="tuneRe" class="small">Retune on next start</button></div>`;
    $("#tuneAck").onclick = async () => { try { await api("/api/autotune/ack", {}); loadTune(); } catch (e) { b.textContent = e.message; } };
    $("#tuneRe").onclick = async () => { try { const r = await api("/api/autotune/retune", {}); b.textContent = r.message; } catch (e) { b.textContent = e.message; } };
  } catch {}
}
function save() { LS.set("convs", convs); LS.set("cur", cur); }
function conv() { return convs.find(c => c.id === cur); }
function newConv() { const c = { id: Date.now().toString(36), title: "New chat", messages: [] }; convs.unshift(c); cur = c.id; save(); renderConvs(); renderMsgs(); $("#input").focus(); }
function renderConvs() {
  $("#convList").innerHTML = convs.map(c => `<li data-id="${c.id}" class="${c.id === cur ? "active" : ""}"><span>${esc(c.title)}</span><button data-act="ren" title="Rename">✎</button><button data-act="del" title="Delete">✕</button></li>`).join("");
}
$("#convList").onclick = e => {
  const li = e.target.closest("li"); if (!li) return; const c = convs.find(x => x.id === li.dataset.id);
  if (e.target.dataset.act === "del") { if (!confirm(`Delete "${c.title}"?`)) return; convs = convs.filter(x => x !== c); if (cur === c.id) cur = convs[0]?.id ?? null; }
  else if (e.target.dataset.act === "ren") { const t = prompt("Rename conversation", c.title); if (t) c.title = t; }
  else cur = c.id;
  save(); renderConvs(); renderMsgs();
};
$("#newConv").onclick = newConv;
function msgEl(m, i) {
  const d = document.createElement("div"); d.className = "msg " + m.role;
  d.innerHTML = `<div class="role"><span>${m.role}</span><span class="muted">${m.stats ? esc(m.stats) : ""}</span></div><div class="body">${m.role === "assistant" ? renderAssistant(m) : esc(m.content).replace(/\n/g, "<br>")}</div>
    <div class="macts"><button class="small ghost" data-mact="copy">copy</button>${m.role === "user" ? `<button class="small ghost" data-mact="edit">edit</button>` : `<button class="small ghost" data-mact="regen">regenerate</button>`}<button class="small ghost" data-mact="del">delete</button></div>`;
  d.dataset.i = i; return d;
}
function renderMsgs() {
  const c = conv(); const box = $("#messages"); box.innerHTML = "";
  if (!c) { box.innerHTML = `<div class="muted" style="margin:auto">Start a new chat.</div>`; return; }
  c.messages.forEach((m, i) => box.appendChild(msgEl(m, i))); box.scrollTop = box.scrollHeight;
}
function rinfoText(r) { let s = `${r.thinking_tokens ?? "?"} tokens · ${r.effort}`; if (r.auto && r.auto.level) s += ` → ${r.auto.level} (rating ${(+r.auto.rating).toFixed(2)})`; if (r.forced_close) s += " · budget reached"; return s; }
$("#effortPick").value = localStorage.getItem("pyre.effort") || "medium"; $("#effortPick").onchange = () => localStorage.setItem("pyre.effort", $("#effortPick").value);
async function generate() {
  const c = conv(); const model = $("#modelPick").value;
  const msgs = []; const sys = $("#sysPrompt").value.trim(); if (sys) msgs.push({ role: "system", content: sys });
  c.messages.forEach(m => { if (m.role === "assistant" && m.error && !m.content) return; msgs.push({ role: m.role, content: m.role === "assistant" ? plainAnswer(m) : m.content }); });
  const body = { model, messages: msgs, stream: true, stream_options: { include_usage: true },
    temperature: +$("#sTemp").value, top_p: +$("#sTopP").value, top_k: +$("#sTopK").value, max_tokens: +$("#sMax").value, reasoning_effort: $("#effortPick").value };
  if ($("#sSeed").value !== "") body.seed = +$("#sSeed").value;
  if (!model) { const a = { role: "assistant", content: "", error: "No text model is loaded. Load one on the Models page (or start pyre with -m <model>)." }; c.messages.push(a); save(); renderMsgs(); return; }
  const a = { role: "assistant", content: "", reasoning: "", streaming: true }; c.messages.push(a);
  const el = msgEl(a, c.messages.length - 1); $("#messages").appendChild(el); const bodyEl = $(".body", el);
  ctrl = new AbortController(); $("#stopBtn").classList.remove("hidden"); $("#sendBtn").disabled = true;
  const t0 = performance.now(); let first = 0, n = 0, timings = null, raf = 0;
  const box = $("#messages"); const stick = () => box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  const paint = () => { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; const s = stick(); const open = $("details.think", bodyEl)?.open; bodyEl.innerHTML = renderAssistant(a); if (open) $("details.think", bodyEl).open = true; if (s) box.scrollTop = 1e9; }); };
  paint();
  try {
    for await (const ch of sse("/v1/chat/completions", body, ctrl.signal)) {
      if (ch.error) throw new Error(ch.error.message);
      const dl = ch.choices?.[0]?.delta || {}; const d = dl.content, r = dl.reasoning_content;
      if (d || r) { if (!first) first = performance.now(); n++; if (d) a.content += d; if (r) a.reasoning += r; paint(); }
      if (ch.timings) timings = ch.timings; if (ch.reasoning) { a.rinfo = ch.reasoning; paint(); }
    }
  } catch (e) { if (e.name !== "AbortError") a.error = e.message || String(e); else a.stats = "stopped"; }
  finally {
    if (raf) cancelAnimationFrame(raf); delete a.streaming; bodyEl.innerHTML = renderAssistant(a);
    ctrl = null; $("#stopBtn").classList.add("hidden"); $("#sendBtn").disabled = false;
    if (timings) a.stats = `${timings.predicted_n} tok · ${timings.predicted_per_second.toFixed(1)} tok/s · prompt ${timings.prompt_per_second.toFixed(0)} tok/s`;
    else if (n && a.stats !== "stopped") a.stats = `${n} chunks · ${(n / ((performance.now() - (first || t0)) / 1000)).toFixed(1)} tok/s`;
    $("#tps").textContent = a.stats || ""; $(".role .muted", el).textContent = a.stats || "";
    save();
  }
}
$("#composer").onsubmit = e => {
  e.preventDefault(); const t = $("#input").value.trim(); if (!t || ctrl) return;
  if (!conv()) newConv(); const c = conv();
  c.messages.push({ role: "user", content: t }); if (c.title === "New chat") c.title = t.slice(0, 40);
  $("#input").value = ""; save(); renderConvs(); renderMsgs(); generate();
};
$("#input").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } };
$("#stopBtn").onclick = () => ctrl?.abort();
$("#regenBtn").onclick = () => { const c = conv(); if (!c || ctrl) return; while (c.messages.length && c.messages.at(-1).role === "assistant") c.messages.pop(); if (!c.messages.length) return; renderMsgs(); generate(); };

$("#messages").onclick = e => {
  const act = e.target.dataset?.mact; if (!act) return; const c = conv(); const el = e.target.closest(".msg"); const i = +el.dataset.i; const m = c.messages[i];
  if (act === "copy") { navigator.clipboard.writeText(m.role === "assistant" ? plainAnswer(m) : m.content); e.target.textContent = "copied"; return; }
  if (ctrl) return;
  if (act === "del") { c.messages.splice(i, 1); save(); renderMsgs(); }
  else if (act === "regen") { c.messages.splice(i); save(); renderMsgs(); generate(); }
  else if (act === "edit") {
    const body = $(".body", el); body.innerHTML = `<textarea class="editbox" rows="4"></textarea><div class="row"><button class="small primary" data-mact="editsave">Save & send</button><button class="small" data-mact="editcancel">Cancel</button></div>`;
    $("textarea", body).value = m.content; $("textarea", body).focus();
  }
  else if (act === "editcancel") renderMsgs();
  else if (act === "editsave") { m.content = $("textarea", el).value; c.messages.splice(i + 1); save(); renderMsgs(); generate(); }
};
$("#convToggle").onclick = () => document.body.classList.toggle("show-convs");
$("#convList").addEventListener("click", () => document.body.classList.remove("show-convs"));

// ---------- models ----------
let kinds = {};
function capHtml(c) {
  const s = c && c.summary;
  if (!s || s === "none") return `<span class="muted">none</span>`;
  return s.split(", ").map(t => `<span class="tag">${esc(t)}</span>`).join("");
}
async function loadKinds() {
  try {
    const r = await api("/v1/models"); kinds = {}; r.data.forEach(m => (kinds[m.kind] ??= []).push(m));
    const sel = $("#modelPick"); const prev = sel.value || LS.get("model", "");
    sel.innerHTML = (kinds.llm || []).map(m => `<option title="${esc(m.capabilities?.summary || "")}">${esc(m.id)}</option>`).join("") || `<option value="">(no text model loaded)</option>`;
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
    const picked = (kinds.llm || []).find(m => m.id === sel.value);
    sel.title = picked?.capabilities?.summary ? `${picked.id}: ${picked.capabilities.summary}` : "Model";
    const en = (k, form, note) => { const on = !!(kinds[k]?.length) && kinds[k].some(m => m.meta?.status !== "not-runnable"); $(form).disabled = !on; $(note).classList.toggle("hidden", on); };
    en("image", "#imgForm", "#imgNote"); /* stt form stays enabled for model download */ en("tts", "#ttsForm", "#ttsNote");
    if (typeof refreshSttModels === "function") refreshSttModels();
  } catch (e) { console.warn(e); }
}
$("#modelPick").onchange = () => { LS.set("model", $("#modelPick").value); const m = (kinds.llm || []).find(x => x.id === $("#modelPick").value); $("#modelPick").title = m?.capabilities?.summary ? `${m.id}: ${m.capabilities.summary}` : "Model"; };
async function loadModels() {
  let r; try { r = await api("/api/models"); } catch (e) { $("#storeInfo").textContent = e.message; return; }
  $("#storeInfo").textContent = `store: ${r.store_root} · HF cache: ${r.hf_cache}${r.offline ? " · OFFLINE" : ""}`;
  $("#loadedTbl tbody").innerHTML = r.loaded.map(m => `<tr><td>${esc(m.id)}</td><td><span class="tag ${m.kind}">${m.kind}</span></td><td class="caps">${capHtml(m.capabilities)}</td><td>${esc(m.backend)} ${m.meta?.dtype ? "· " + m.meta.dtype : ""}</td><td>${m.busy ? "busy" : "idle"}</td><td>${m.unloadable === false ? "" : `<button class="small" data-unload="${esc(m.id)}">Unload</button>`}</td></tr>`).join("") || `<tr><td colspan=6 class="muted">nothing loaded</td></tr>`;
  $("#availTbl tbody").innerHTML = r.models.map(m => `<tr><td>${esc(m.name)}${m.alias ? ` <span class="tag">${esc(m.alias)}</span>` : ""}</td><td><span class="tag ${m.kind}">${m.kind}</span></td><td class="caps">${capHtml(m.capabilities)}</td><td>${m.source}</td><td>${human(m.size)}</td><td>${esc(m.storage || "")}</td><td class="loc">${esc(String(m.location))}</td>
    <td><button class="small" data-load="${esc(m.alias || m.name)}" data-kind="${m.kind}">Load</button> <button class="small" data-alias="${esc(m.name)}">Alias</button> ${m.source === "hf-cache" ? `<button class="small" data-import="${esc(m.name)}">Link to store</button>` : ""} <button class="small" data-rm="${esc(m.name)}" data-src="${m.source}">Remove</button></td></tr>`).join("");
}
function dialog(text, extraHtml = "") { return new Promise(res => { $("#dlgText").textContent = text; $("#dlgExtra").innerHTML = extraHtml; const d = $("#dlg"); d.returnValue = ""; d.showModal(); d.onclose = () => res(d.returnValue === "ok" ? d : null); }); }
$("#tab-models").onclick = async e => {
  const b = e.target; const msg = t => { $("#impMsg").textContent = t; };
  try {
    if (b.dataset.unload) { await api("/api/models/unload", { id: b.dataset.unload }); }
    else if (b.dataset.load) {
      const d = await dialog(`Load ${b.dataset.load}`, `<label>Backend <select id="dlgBackend"><option>cuda</option><option>wgpu</option><option>cpu</option></select></label> <label>Kind <select id="dlgKind"><option>llm</option><option>stt</option><option>image</option><option>tts</option></select></label>`);
      $("#dlgKind").value = b.dataset.kind === "unknown" ? "llm" : b.dataset.kind;
      if (!d) return; b.disabled = true; b.textContent = "loading…";
      await api("/api/models/load", { model: b.dataset.load, kind: $("#dlgKind").value, backend: $("#dlgBackend").value });
    }
    else if (b.dataset.alias) { const a = prompt(`Alias for ${b.dataset.alias}`); if (!a) return; await api("/api/models/alias", { name: b.dataset.alias, alias: a }); }
    else if (b.dataset.import) { const r = await api("/api/models/import", { source: b.dataset.import, mode: "link" }); msg(`linked into ${r.dir} ${(r.notes || []).join("; ")}`); }
    else if (b.dataset.rm) {
      const hf = b.dataset.src === "hf-cache";
      const d = await dialog(hf ? `Delete ${b.dataset.rm} from the Hugging Face cache? This removes the downloaded blobs (other tools share them).` : `Remove ${b.dataset.rm} from the pyre store? Linked files only drop a reference; the HF cache is untouched.`,
        hf ? "" : `<label><input type="checkbox" id="dlgHf"> also delete from the HF cache</label>`);
      if (!d) return; await api("/api/models/rm", { name: b.dataset.rm, from_hf_cache: hf || !!$("#dlgHf")?.checked, confirm: true });
    } else return;
    loadModels(); loadKinds();
  } catch (err) { alert(err.message); loadModels(); }
};
$("#pullBtn").onclick = async () => {
  const repo = $("#pullRepo").value.trim(); if (!repo) return; const bar = $("#pullProg"); bar.classList.remove("hidden"); $("#pullBtn").disabled = true;
  try {
    for await (const ev of sse("/api/pull", { repo, store: $("#pullStore").value })) {
      if (ev.error) throw new Error(ev.error);
      if (ev.done) { $("#pullMsg").textContent = "done → " + ev.path; $("div", bar).style.width = "100%"; break; }
      $("div", bar).style.width = (100 * ev.index / Math.max(ev.total, 1)).toFixed(0) + "%"; $("#pullMsg").textContent = `[${ev.index + 1}/${ev.total}] ${ev.file}`;
    }
  } catch (e) { $("#pullMsg").textContent = "error: " + e.message; }
  $("#pullBtn").disabled = false; loadModels();
};
$("#impBtn").onclick = async () => {
  try { const r = await api("/api/models/import", { source: $("#impSrc").value.trim(), mode: $("#impMode").value, kind: $("#impKind").value || undefined }); $("#impMsg").textContent = `ok: ${r.name} → ${r.dir} ${(r.notes || []).join("; ")}`; loadModels(); }
  catch (e) { $("#impMsg").textContent = "error: " + e.message; }
};

// ---------- images (enabled once an image model is served) ----------
$("#imgGo").onclick = async e => {
  e.preventDefault(); const [w, h] = $("#imgSize").value.split("x").map(Number);
  const body = { prompt: $("#imgPrompt").value, negative_prompt: $("#imgNeg").value, size: `${w}x${h}`, n: 1, response_format: "b64_json", steps: +$("#imgSteps").value, guidance_scale: +$("#imgCfg").value };
  if ($("#imgSeed").value !== "") body.seed = +$("#imgSeed").value;
  if (w === 0) delete body.size;
  if ($("#imgRefiner").checked) body.refiner = true;
  const initF = $("#imgInit").files[0]; let path = "/v1/images/generations";
  if (initF) { body.image = await new Promise((ok, bad) => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.onerror = bad; fr.readAsDataURL(initF); }); body.strength = +$("#imgStrength").value; path = "/v1/images/edits"; }
  $("#imgGo").disabled = true; $("#imgGo").textContent = "generating…";
  try { const r = await api(path, body); r.data.forEach(d => { const im = new Image(); im.src = "data:image/png;base64," + d.b64_json; im.title = body.prompt; $("#gallery").prepend(im); }); }
  catch (err) { alert(err.message); }
  $("#imgGo").disabled = false; $("#imgGo").textContent = "Generate";
};

// ---------- audio ----------
const STT_PREF_KEY = "sttPrefs";
const STT_TX_KEY = "sttTranscript";
const QUALITY_MODELS = { fast: "tiny", balanced: "base", best: "large-v3-turbo" };
const SPK_COLORS = ["#2f7ed8", "#d8572f", "#2fa36b", "#9b4fd8", "#c9a227", "#d82f8a", "#2fb5c9", "#7a7a7a"];
const fmtT = s => { const m = Math.floor(s / 60); const sec = (s - m * 60); return m + ":" + sec.toFixed(1).padStart(4, "0"); };
const fmtTs = s => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const ss = String(Math.floor(sec)).padStart(2, "0") + "," + String(Math.floor((sec % 1) * 1000)).padStart(3, "0");
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0") + ":" + ss;
};
const fmtVtt = s => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const ss = String(Math.floor(sec)).padStart(2, "0") + "." + String(Math.floor((sec % 1) * 1000)).padStart(3, "0");
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0") + ":" + ss;
};

/** @type {{ sourceName: string, model: string, language: string, duration: number, segments: Array, names: Record<string,string>, text: string, summary: string, summaryStale: boolean }} */
let sttState = null;
let sttText = "";
let sttJob = null;
let sttSumCtrl = null;
let sttCatalog = { models: [], loaded: [] };
let recBlob = null, rec = null;
let stsRecBlob = null, stsRec = null, stsCtrl = null, stsReplyBlob = null, stsRecTimer = null, stsRecStarted = 0;

function loadSttPrefs() {
  const d = { model: "", quality: "balanced", format: "verbose_json", language: "", timestamps: true, sentences: false, diarize: false, beam: 5, maxTok: 448, task: "transcriptions", advancedOpen: false };
  return Object.assign(d, LS.get(STT_PREF_KEY, {}));
}
function saveSttPrefs(p) { LS.set(STT_PREF_KEY, p); }
function sttPrefsFromUi() {
  return {
    model: ($("#sttModel") && $("#sttModel").value) || "",
    quality: ($("#sttQuality") && $("#sttQuality").value) || "balanced",
    format: ($("#sttFormat") && $("#sttFormat").value) || "verbose_json",
    language: ($("#sttLang") && $("#sttLang").value) || "",
    timestamps: !!( $("#sttTimestamps") && $("#sttTimestamps").checked ),
    sentences: !!( $("#sttSentences") && $("#sttSentences").checked ),
    diarize: !!( $("#sttDiar") && $("#sttDiar").checked ),
    beam: +(( $("#sttBeam") && $("#sttBeam").value ) || 5),
    maxTok: +(( $("#sttMaxTok") && $("#sttMaxTok").value ) || 448),
    task: ($("#sttTask") && $("#sttTask").value) || "transcriptions",
    advancedOpen: !!( $("#sttAdvanced") && $("#sttAdvanced").open ),
  };
}
function applySttPrefs(p) {
  if ($("#sttQuality")) $("#sttQuality").value = p.quality || "balanced";
  // sttQualitySeg sync
  if ($("#sttQualitySeg")) $$("#sttQualitySeg .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.q === ($("#sttQuality").value)));
  if ($("#sttTaskSeg") && $("#sttTask")) $$("#sttTaskSeg .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.task === ($("#sttTask").value)));

  if ($("#sttFormat")) $("#sttFormat").value = p.format || "verbose_json";
  if ($("#sttLang")) $("#sttLang").value = p.language || "";
  if ($("#sttTimestamps")) $("#sttTimestamps").checked = !!p.timestamps;
  if ($("#sttSentences")) $("#sttSentences").checked = !!p.sentences;
  if ($("#sttDiar") && !$("#sttDiar").disabled) $("#sttDiar").checked = !!p.diarize;
  if ($("#sttBeam")) $("#sttBeam").value = p.beam || 5;
  if ($("#sttMaxTok")) $("#sttMaxTok").value = p.maxTok || 448;
  if ($("#sttTask")) $("#sttTask").value = p.task || "transcriptions";
  if ($("#sttAdvanced")) $("#sttAdvanced").open = !!p.advancedOpen;
  if ($("#sttModel") && p.model) {
    if ([...($("#sttModel").options || [])].some(o => o.value === p.model)) $("#sttModel").value = p.model;
  }
}
function effectiveSttModel(pref) {
  const adv = $("#sttAdvanced") && $("#sttAdvanced").open;
  if (adv && pref.model) return pref.model;
  return QUALITY_MODELS[pref.quality] || QUALITY_MODELS.balanced;
}
function persistSttState() {
  if (!sttState) { try { localStorage.removeItem("pyre." + STT_TX_KEY); } catch {} return; }
  try { LS.set(STT_TX_KEY, sttState); } catch (e) { console.warn("stt persist", e); }
}
function restoreSttState() {
  const s = LS.get(STT_TX_KEY, null);
  if (!s || !Array.isArray(s.segments)) return;
  sttState = s;
  paintSttResult({ restored: true });
}
function displayName(id) {
  if (!sttState) return id || "?";
  const n = sttState.names && sttState.names[id];
  return n || String(id || "?").replace(/^SPEAKER_/, "Speaker ");
}
function speakerIds() {
  if (!sttState) return [];
  const ids = [];
  for (const s of sttState.segments) {
    const sp = s.speaker || "?";
    if (!ids.includes(sp)) ids.push(sp);
  }
  return ids;
}
function colorFor(id) {
  const ids = speakerIds();
  const i = Math.max(0, ids.indexOf(id));
  return SPK_COLORS[i % SPK_COLORS.length];
}
function labelledText() {
  if (!sttState) return sttText || "";
  if (!sttState.segments.length) return sttState.text || "";
  return sttState.segments.map(s => {
    const who = displayName(s.speaker);
    const hasSpk = !!(s.speaker || (sttState.names && Object.keys(sttState.names).length));
    const line = (s.text || "").trim();
    if (!hasSpk && speakerIds().length <= 1 && !(s.speaker)) return line;
    return who + ": " + line;
  }).filter(Boolean).join("\n");
}
function buildTxt() {
  if (!sttState) return sttText || "";
  const ts = loadSttPrefs().timestamps;
  return sttState.segments.map(s => {
    const who = displayName(s.speaker);
    const head = ts ? "[" + fmtT(s.start) + " → " + fmtT(s.end) + "] " : "";
    return head + who + ": " + (s.text || "").trim();
  }).join("\n");
}
function buildSrt() {
  if (!sttState) return "";
  return sttState.segments.map((s, i) => {
    const who = displayName(s.speaker);
    return (i + 1) + "\n" + fmtTs(s.start) + " --> " + fmtTs(s.end) + "\n" + who + ": " + (s.text || "").trim() + "\n";
  }).join("\n");
}
function buildVtt() {
  if (!sttState) return "WEBVTT\n\n";
  return "WEBVTT\n\n" + sttState.segments.map(s => {
    const who = displayName(s.speaker);
    return fmtVtt(s.start) + " --> " + fmtVtt(s.end) + "\n" + who + ": " + (s.text || "").trim() + "\n";
  }).join("\n");
}
function buildJson() {
  if (!sttState) return "{}";
  const segs = sttState.segments.map((s, i) => ({
    id: i, start: s.start, end: s.end, text: (s.text || "").trim(),
    speaker: s.speaker || null, speaker_label: displayName(s.speaker),
  }));
  return JSON.stringify({
    model: sttState.model, language: sttState.language, duration: sttState.duration,
    source: sttState.sourceName, names: sttState.names || {},
    text: labelledText(), segments: segs,
  }, null, 2);
}
function buildMd() {
  if (!sttState) return sttText || "";
  let md = "# Transcript";
  if (sttState.sourceName) md += " — " + sttState.sourceName;
  md += "\n\n";
  if (sttState.model) md += "_Model: `" + sttState.model + "`";
  if (sttState.language) md += " · Language: " + sttState.language;
  md += "_\n\n";
  for (const s of sttState.segments) {
    md += "**" + displayName(s.speaker) + "** [" + fmtT(s.start) + "–" + fmtT(s.end) + "]\n\n" + (s.text || "").trim() + "\n\n";
  }
  return md;
}
function sttBaseName() {
  const n = (sttState && sttState.sourceName) || "transcript";
  return n.replace(/\.[^.]+$/, "").replace(/[^\w\-]+/g, "_").slice(0, 80) || "transcript";
}
function downloadBlob(name, text, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime || "text/plain" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function setSttProg(show, pct, label) {
  const box = $("#sttProg"); if (!box) return;
  box.classList.toggle("hidden", !show);
  if (pct != null && $("#sttBar")) $("#sttBar").style.width = Math.max(0, Math.min(100, pct)) + "%";
  if (label != null && $("#sttProgLabel")) $("#sttProgLabel").textContent = label;
}
function setSumProg(show, pct, label) {
  const box = $("#sttSumProg"); if (!box) return;
  box.classList.toggle("hidden", !show);
  if (pct != null && $("#sttSumBar")) $("#sttSumBar").style.width = Math.max(0, Math.min(100, pct)) + "%";
  if (label != null && $("#sttSumProgLabel")) $("#sttSumProgLabel").textContent = label;
}
async function refreshSttModels() {
  try {
    sttCatalog = await api("/api/audio/models");
    const sel = $("#sttModel"); if (!sel) return;
    const pref = loadSttPrefs();
    const cur = pref.model || (kinds.stt && kinds.stt[0] && kinds.stt[0].id) || (sttCatalog.defaults && sttCatalog.defaults.model) || "base";
    sel.innerHTML = "";
    for (const m of (sttCatalog.models || [])) {
      const o = document.createElement("option");
      o.value = m.id;
      const cap = (m.capabilities && m.capabilities.summary) || "";
      o.textContent = m.id + (m.loaded ? " *" : (m.present ? "" : " (download)")) + " — " + (m.tagline || m.description || "");
      o.title = cap;
      o.dataset.present = m.present ? "1" : "0";
      sel.append(o);
    }
    if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
    updateSttDlBtn();
    if ($("#sttForm")) $("#sttForm").removeAttribute("disabled");
    if ((sttCatalog.loaded || []).length) $("#sttNote").classList.add("hidden");
    else $("#sttNote").classList.remove("hidden");
  } catch (e) { console.warn("stt models", e); }
}
function updateSttDlBtn() {
  const sel = $("#sttModel"); const btn = $("#sttModelDl"); if (!sel || !btn) return;
  const opt = sel.selectedOptions[0];
  const need = opt && opt.dataset.present === "0";
  btn.classList.toggle("hidden", !need);
  btn.textContent = need ? ("Download " + sel.value) : "Download model";
}
async function* audioJobSse(fd, signal) {
  const r = await fetch("/api/audio/jobs", { method: "POST", headers: headers(), body: fd, signal });
  if (!r.ok) { const t = await r.text(); let m = t; try { m = JSON.parse(t).error.message || t; } catch {} throw new Error(m || r.statusText); }
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop();
    for (const block of parts) {
      const data = block.split("\n").filter(l => l.startsWith("data:")).map(l => l.slice(5).trim()).join("");
      if (!data) continue;
      try { yield JSON.parse(data); } catch {}
    }
  }
}
let sttBarFloor = null;
function resetSttBar() { sttBarFloor = null; }
function sttStagePct(ev) {
  sttBarFloor = advanceSttBar(sttBarFloor, ev);
  return sttBarFloor;
}
function paintSttResult(opts) {
  opts = opts || {};
  if (!sttState) return;
  sttText = labelledText();
  const meta = $("#sttMeta");
  if (meta) {
    meta.classList.remove("hidden");
    const bits = [];
    if (sttState.model) bits.push("model " + sttState.model);
    if (sttState.routing && sttState.routing.reason) bits.push(sttState.routing.reason);
    if (sttState.language) bits.push("language " + sttState.language);
    if (sttState.duration) bits.push(sttState.duration.toFixed(1) + "s audio");
    if (opts.elapsed) bits.push(opts.elapsed.toFixed(2) + "s wall");
    if (opts.restored) bits.push("restored");
    if (sttState.summaryStale) bits.push("summary outdated — regenerate");
    meta.textContent = bits.join(" · ");
  }
  renderSpeakerChrome();
  $("#sttOut").textContent = sttText + (opts.footer ? "\n\n(" + opts.footer + ")" : "");
  if ($("#sttActions")) $("#sttActions").classList.toggle("hidden", !sttText.trim());
  if ($("#sttSumBox")) {
    if (sttState.summary) {
      $("#sttSumBox").classList.remove("hidden");
      $("#sttSumOut").innerHTML = md(sttState.summary);
      if ($("#sttSumRegen")) $("#sttSumRegen").classList.toggle("hidden", !sttState.summaryStale);
    }
  }
}
function renderSpeakerChrome() {
  const box = $("#sttSpk"); const legend = $("#sttLegend"); const tl = $("#sttTimeline");
  if (!box || !sttState) return;
  const ids = speakerIds();
  const hasSpk = ids.some(id => id && id !== "?") || (sttState.segments.some(s => s.speaker));
  box.innerHTML = "";
  if (legend) legend.innerHTML = "";
  if (tl) tl.innerHTML = "";
  if (!hasSpk) {
    box.classList.add("hidden");
    if (legend) legend.classList.add("hidden");
    if (tl) tl.classList.add("hidden");
    return;
  }
  if (legend) {
    legend.classList.remove("hidden");
    for (const id of ids) {
      const chip = document.createElement("span"); chip.className = "spk-chip";
      const dot = document.createElement("span"); dot.className = "dot"; dot.style.background = colorFor(id);
      const lab = document.createElement("span"); lab.textContent = displayName(id);
      const ren = document.createElement("button"); ren.type = "button"; ren.textContent = "rename";
      ren.onclick = () => renameSpeaker(id);
      const mer = document.createElement("button"); mer.type = "button"; mer.textContent = "merge…";
      mer.onclick = () => mergeSpeaker(id);
      chip.append(dot, lab, ren, mer);
      legend.append(chip);
    }
  }
  if (tl && sttState.duration > 0) {
    tl.classList.remove("hidden");
    for (const s of sttState.segments) {
      const w = Math.max(0.2, 100 * ((s.end - s.start) / sttState.duration));
      const span = document.createElement("span");
      span.style.width = w + "%";
      span.style.background = colorFor(s.speaker || "?");
      span.title = displayName(s.speaker) + " " + fmtT(s.start) + "–" + fmtT(s.end);
      tl.append(span);
    }
  }
  for (let i = 0; i < sttState.segments.length; i++) {
    const s = sttState.segments[i];
    const sp = s.speaker || "?";
    const c = colorFor(sp);
    const row = document.createElement("div"); row.className = "spk-row"; row.style.borderLeftColor = c;
    const h = document.createElement("span"); h.className = "spk-who"; h.style.color = c;
    h.textContent = displayName(sp) + " · " + fmtT(s.start) + "–" + fmtT(s.end);
    h.title = "Click to rename speaker";
    h.onclick = () => renameSpeaker(sp);
    const acts = document.createElement("span"); acts.className = "spk-seg-acts";
    const reas = document.createElement("button"); reas.type = "button"; reas.textContent = "reassign";
    reas.onclick = (e) => { e.stopPropagation(); reassignSegment(i); };
    acts.append(reas);
    const tt = document.createElement("span"); tt.className = "spk-text"; tt.textContent = (s.text || "").trim();
    row.append(h, acts, tt); box.append(row);
  }
  box.classList.remove("hidden");
}
function markSummaryStale() {
  if (sttState && sttState.summary) {
    sttState.summaryStale = true;
    if ($("#sttSumRegen")) $("#sttSumRegen").classList.remove("hidden");
    if ($("#sttMeta") && !$("#sttMeta").classList.contains("hidden")) {
      paintSttResult();
    }
  }
}
function renameSpeaker(id) {
  if (!sttState) return;
  const cur = displayName(id);
  const n = prompt("Rename speaker", cur);
  if (n == null) return;
  const name = n.trim();
  if (!name) return;
  sttState.names = sttState.names || {};
  sttState.names[id] = name;
  markSummaryStale();
  persistSttState();
  paintSttResult();
}
function mergeSpeaker(fromId) {
  if (!sttState) return;
  const ids = speakerIds().filter(x => x !== fromId);
  if (!ids.length) return alert("No other speakers to merge into");
  const choices = ids.map(id => displayName(id) + " (" + id + ")").join("\n");
  const to = prompt("Merge " + displayName(fromId) + " into which speaker id?\n\n" + choices, ids[0]);
  if (to == null) return;
  const toId = to.trim();
  if (!ids.includes(toId) && !ids.some(id => displayName(id) === toId)) {
    // allow typing display name
    const hit = ids.find(id => id === toId || displayName(id) === toId);
    if (!hit) return alert("Unknown speaker");
    return mergeSpeakerInto(fromId, hit);
  }
  const hit = ids.includes(toId) ? toId : ids.find(id => displayName(id) === toId);
  mergeSpeakerInto(fromId, hit);
}
function mergeSpeakerInto(fromId, toId) {
  for (const s of sttState.segments) {
    if ((s.speaker || "?") === fromId) s.speaker = toId;
  }
  if (sttState.names) delete sttState.names[fromId];
  markSummaryStale();
  persistSttState();
  paintSttResult();
}
function reassignSegment(idx) {
  if (!sttState) return;
  const ids = speakerIds();
  const choices = ids.map(id => displayName(id) + " [" + id + "]").join("\n");
  const cur = sttState.segments[idx].speaker || "?";
  const to = prompt("Reassign this segment to speaker id:\n\n" + choices, cur);
  if (to == null) return;
  let hit = ids.find(id => id === to.trim() || displayName(id) === to.trim());
  if (!hit) {
    // create new speaker id
    hit = to.trim();
    if (!hit) return;
  }
  sttState.segments[idx].speaker = hit;
  markSummaryStale();
  persistSttState();
  paintSttResult();
}


function micAllowed() {
  try {
    const h = location.hostname;
    return !!(window.isSecureContext || h === "localhost" || h === "127.0.0.1" || h === "[::1]");
  } catch { return false; }
}
function currentSttAudio() {
  if (recBlob) return recBlob;
  const inp = $("#sttFile");
  return (inp && inp.files && inp.files[0]) || null;
}
function fmtDur(s) {
  s = Math.max(0, Math.floor(s || 0));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}
function setSttSourceMode(mode) {
  const file = mode !== "record";
  if ($("#sttModeFile")) { $("#sttModeFile").classList.toggle("active", file); $("#sttModeFile").setAttribute("aria-selected", file ? "true" : "false"); }
  if ($("#sttModeRec")) { $("#sttModeRec").classList.toggle("active", !file); $("#sttModeRec").setAttribute("aria-selected", !file ? "true" : "false"); }
  if ($("#sttFilePane")) $("#sttFilePane").classList.toggle("hidden", !file);
  if ($("#sttRecPane")) $("#sttRecPane").classList.toggle("hidden", file);
  updateMicGate();
  updateSttRunAvail();
}
function updateMicGate() {
  const ok = micAllowed();
  const hint = $("#sttMicHint");
  if (hint) {
    hint.classList.toggle("hidden", ok);
    const u = $("#sttLocalUrl");
    if (u) u.textContent = "http://127.0.0.1:" + (location.port || "8099") + "/";
  }
  if ($("#recBtn")) {
    setAvail($("#recBtn"), ok
      ? availOk()
      : { enabled: false, reason_kind: "no_permission", message: "Microphone needs HTTPS or localhost — open 127.0.0.1 on this PC" });
  }
}
function showPreview(blobOrFile) {
  if (!blobOrFile) return;
  // Source step only. Results must not repeat this file (STS/TTS reply players are separate).
  const a = $("#sttPreview");
  if (!a) return;
  a.src = URL.createObjectURL(blobOrFile);
  a.classList.remove("hidden");
}
function describeFile(f) {
  if (!f) { if ($("#sttFileMeta")) $("#sttFileMeta").textContent = "MP3, WAV, M4A, FLAC, WebM…"; return; }
  const bits = [f.name || "audio"];
  if (f.size) bits.push(human(f.size));
  if ($("#sttFileMeta")) $("#sttFileMeta").textContent = bits.join(" · ");
  const drop = $("#sttDrop"); if (drop) drop.classList.add("has-file");
  // duration via preview metadata
  const a = $("#sttPreview");
  if (a) {
    const onmeta = () => {
      if (isFinite(a.duration)) {
        if ($("#sttFileMeta")) $("#sttFileMeta").textContent = bits.concat([fmtDur(a.duration)]).join(" · ");
      }
      a.removeEventListener("loadedmetadata", onmeta);
    };
    a.addEventListener("loadedmetadata", onmeta);
  }
}
function updateSttRunAvail() {
  const btn = $("#sttGo"); if (!btn) return;
  if (btn.dataset.forceBusy === "1") return;
  const down = statusState && statusState.state === "down";
  if (down) {
    setAvail(btn, { enabled: false, reason_kind: "offline", message: statusState.headline || "Server disconnected" });
    return;
  }
  const f = currentSttAudio();
  if (!f) {
    setAvail(btn, { enabled: false, reason_kind: "", message: "Choose or record audio first" });
    return;
  }
  setAvail(btn, availOk());
  const task = ($("#sttTask") && $("#sttTask").value) || "transcriptions";
  btn.textContent = task === "translations" ? "Translate" : "Transcribe";
}
function setSttRunningUi(on) {
  if ($("#sttGo")) $("#sttGo").classList.toggle("hidden", !!on);
  if ($("#sttCancel")) $("#sttCancel").classList.toggle("hidden", !on);
}
let recTimer = null, recStarted = 0, recAnalyser = null, recRaf = 0, recStream = null;
function stopRecMeter() {
  if (recRaf) cancelAnimationFrame(recRaf); recRaf = 0;
  if ($("#sttRecMeter")) $("#sttRecMeter").classList.add("hidden");
}
function startRecMeter(stream) {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const src = ac.createMediaStreamSource(stream);
    const an = ac.createAnalyser(); an.fftSize = 256;
    src.connect(an); recAnalyser = { ac, an };
    const canvas = $("#sttRecMeter"); if (!canvas) return;
    canvas.classList.remove("hidden");
    const ctx = canvas.getContext("2d");
    const data = new Uint8Array(an.frequencyBinCount);
    const draw = () => {
      recRaf = requestAnimationFrame(draw);
      an.getByteTimeDomainData(data);
      let peak = 0;
      for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i] - 128));
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "rgba(127,127,127,.25)"; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = peak > 40 ? "#2fa36b" : "#d8a017";
      ctx.fillRect(0, 0, Math.min(w, (peak / 128) * w * 2), h);
    };
    draw();
  } catch {}
}
function tickRecTimer() {
  if ($("#sttRecTimer")) $("#sttRecTimer").textContent = fmtDur((Date.now() - recStarted) / 1000);
}
if ($("#sttModeFile")) $("#sttModeFile").onclick = () => setSttSourceMode("file");
if ($("#sttModeRec")) $("#sttModeRec").onclick = () => setSttSourceMode("record");
if ($("#sttDrop")) {
  const dz = $("#sttDrop");
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    const dt = new DataTransfer(); dt.items.add(f);
    $("#sttFile").files = dt.files;
    recBlob = null;
    describeFile(f); showPreview(f); updateSttRunAvail();
  });
}
if ($("#recBtn")) $("#recBtn").onclick = async () => {
  if (rec) { rec.stop(); return; }
  if (!micAllowed()) { updateMicGate(); return; }
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    recStream = s;
    const chunks = [];
    rec = new MediaRecorder(s);
    rec.ondataavailable = e => chunks.push(e.data);
    rec.onstop = () => {
      recBlob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      try { s.getTracks().forEach(t => t.stop()); } catch {}
      rec = null; recStream = null;
      if (recTimer) { clearInterval(recTimer); recTimer = null; }
      stopRecMeter();
      if (recAnalyser) { try { recAnalyser.ac.close(); } catch {} recAnalyser = null; }
      $("#recBtn").textContent = "Start recording";
      if ($("#sttRecAfter")) $("#sttRecAfter").classList.remove("hidden");
      if ($("#sttRecMeta")) $("#sttRecMeta").textContent = "Recording ready · " + human(recBlob.size);
      // clear file input so currentSttAudio prefers recBlob
      if ($("#sttFile")) try { $("#sttFile").value = ""; } catch {}
      describeFile(null);
      showPreview(recBlob);
      updateSttRunAvail();
    };
    rec.start();
    $("#recBtn").textContent = "Stop recording";
    if ($("#sttRecAfter")) $("#sttRecAfter").classList.add("hidden");
    recStarted = Date.now(); tickRecTimer();
    recTimer = setInterval(tickRecTimer, 250);
    startRecMeter(s);
  } catch (e) { alert("microphone: " + e.message); updateMicGate(); }
};
if ($("#sttRecAgain")) $("#sttRecAgain").onclick = () => {
  recBlob = null;
  if ($("#sttRecAfter")) $("#sttRecAfter").classList.add("hidden");
  if ($("#sttPreview")) { $("#sttPreview").removeAttribute("src"); $("#sttPreview").classList.add("hidden"); }
  if ($("#sttRecTimer")) $("#sttRecTimer").textContent = "0:00";
  updateSttRunAvail();
  if ($("#recBtn") && !$("#recBtn").disabled) $("#recBtn").click();
};
if ($("#sttFile")) $("#sttFile").onchange = () => {
  recBlob = null;
  const f = $("#sttFile").files[0];
  if (f) { describeFile(f); showPreview(f); }
  else { describeFile(null); }
  updateSttRunAvail();
};
if ($("#sttQualitySeg")) $("#sttQualitySeg").onclick = (e) => {
  const b = e.target.closest("[data-q]"); if (!b) return;
  $$("#sttQualitySeg .seg-btn").forEach(x => x.classList.toggle("active", x === b));
  if ($("#sttQuality")) $("#sttQuality").value = b.dataset.q;
  if ($("#sttQuality")) $("#sttQuality").dispatchEvent(new Event("change"));
};
if ($("#sttTaskSeg")) $("#sttTaskSeg").onclick = (e) => {
  const b = e.target.closest("[data-task]"); if (!b) return;
  $$("#sttTaskSeg .seg-btn").forEach(x => x.classList.toggle("active", x === b));
  if ($("#sttTask")) $("#sttTask").value = b.dataset.task;
  if ($("#sttTask")) $("#sttTask").dispatchEvent(new Event("change"));
  updateSttRunAvail();
};
updateMicGate();
setSttSourceMode("file");
updateSttRunAvail();
if ($("#sttModel")) $("#sttModel").onchange = () => { updateSttDlBtn(); saveSttPrefs(sttPrefsFromUi()); };
if ($("#sttModel")) $("#sttModel").onchange = () => { updateSttDlBtn(); saveSttPrefs(sttPrefsFromUi()); };
if ($("#sttQuality")) $("#sttQuality").onchange = () => {
  const p = sttPrefsFromUi();
  // sync advanced model select to the quality mapping when quality changes
  const m = QUALITY_MODELS[p.quality];
  if (m && $("#sttModel") && [...$("#sttModel").options].some(o => o.value === m)) $("#sttModel").value = m;
  saveSttPrefs(sttPrefsFromUi());
};
["sttFormat","sttLang","sttTimestamps","sttSentences","sttDiar","sttBeam","sttMaxTok","sttTask"].forEach(id => {
  const el = document.getElementById(id); if (el) el.addEventListener("change", () => saveSttPrefs(sttPrefsFromUi()));
});
if ($("#sttAdvanced")) $("#sttAdvanced").addEventListener("toggle", () => saveSttPrefs(sttPrefsFromUi()));
if ($("#sttModelDl")) $("#sttModelDl").onclick = async () => {
  const id = $("#sttModel").value; if (!id) return;
  $("#sttModelDl").disabled = true; setSttProg(true, 5, "Downloading/loading " + id + "...");
  try {
    await api("/api/models/load", { model: id, kind: "stt", backend: "cpu" });
    await refreshSttModels();
    if (typeof refreshAll === "function") await refreshAll();
    setSttProg(true, 100, id + " ready");
  } catch (e) { setSttProg(true, 0, "error: " + e.message); }
  finally { $("#sttModelDl").disabled = false; }
};
if ($("#sttCancel")) $("#sttCancel").onclick = () => {
  if (!sttJob) return;
  sttJob.ctrl.abort();
  fetch("/api/audio/jobs/cancel", { method: "POST", headers: headers({ "content-type": "application/json" }), body: JSON.stringify({ id: sttJob.id }) }).catch(() => {});
  $("#sttCancel").classList.add("hidden");
  setSttProg(true, null, "cancelling...");
};
if ($("#sttGo")) $("#sttGo").onclick = async () => {
  const f = currentSttAudio(); if (!f) return alert("Choose or record audio first");
  const pref = sttPrefsFromUi(); saveSttPrefs(pref);
  const model = effectiveSttModel(pref);
  const fd = new FormData();
  fd.append("file", f, f.name || "recording.webm");
  fd.append("model", model);
  fd.append("response_format", "verbose_json");
  fd.append("task", pref.task === "translations" ? "translations" : "transcriptions");
  if (pref.language) fd.append("language", pref.language);
  fd.append("timestamps", pref.timestamps ? "true" : "false");
  fd.append("sentences", pref.sentences ? "true" : "false");
  fd.append("beam_size", String(pref.beam || 5));
  fd.append("max_tokens", String(pref.maxTok || 448));
  const diar = pref.diarize && $("#sttDiar") && !$("#sttDiar").disabled && pref.task === "transcriptions";
  if (diar) fd.append("diarize", "true");
  $("#sttSpk").classList.add("hidden"); $("#sttSpk").innerHTML = "";
  if ($("#sttLegend")) { $("#sttLegend").classList.add("hidden"); $("#sttLegend").innerHTML = ""; }
  if ($("#sttTimeline")) { $("#sttTimeline").classList.add("hidden"); $("#sttTimeline").innerHTML = ""; }
  if ($("#sttActions")) $("#sttActions").classList.add("hidden");
  if ($("#sttSumBox")) $("#sttSumBox").classList.add("hidden");
  $("#sttOut").textContent = "";
  resetSttBar();
  setSttProg(true, 0, "uploading… · model " + model);
  setSttRunningUi(true);
  if ($("#sttGo")) { $("#sttGo").dataset.forceBusy = "1"; setAvail($("#sttGo"), availBusy((statusState && statusState.headline) || "Transcribing?")); }
  const ctrl = new AbortController();
  sttJob = { id: null, ctrl };
  const t0 = performance.now();
  let partial = "";
  try {
    for await (const ev of audioJobSse(fd, ctrl.signal)) {
      if (ev.job) sttJob.id = ev.job;
      if (ev.stage === "error" || ev.stage === "cancelled") throw new Error(ev.message || ev.stage);
      const pct = sttStagePct(ev);
      let label = ev.message || ev.stage;
      if (ev.model) label += " · " + ev.model;
      if (ev.stage === "downloading") {
        if (ev.bytes != null && ev.bytes_total) label = (ev.message || "downloading") + " · " + human(ev.bytes) + " / " + human(ev.bytes_total);
        else if (ev.fraction != null) label = (ev.message || "downloading") + " · " + Math.round(ev.fraction * 100) + "%";
      }
      if (ev.stage === "transcribing" && (ev.total || ev.fraction != null)) {
        const prog = ev.total ? (ev.current + "/" + ev.total) : "";
        const eta = (ev.eta_s != null && (ev.eta_s >= 0.5 || (ev.fraction != null && ev.fraction >= 0.98)))
          ? (" · ETA " + Math.max(0, ev.eta_s).toFixed(0) + "s")
          : (ev.fraction != null ? (" · " + Math.round(ev.fraction * 100) + "%") : "");
        label = (ev.message || "transcribing") + (prog ? (" · " + prog) : "") + eta +
          (ev.elapsed_s != null ? (" · " + ev.elapsed_s.toFixed(1) + "s") : "");
      }
      if (ev.stage !== "segment") setSttProg(true, pct, label);
      if (ev.stage === "segment") {
        if (ev.partial) partial = ev.partial;
        else if (ev.text) partial = (partial ? partial + " " : "") + ev.text;
        $("#sttOut").textContent = partial + "\n\n(...still working)";
      }
      if (ev.stage === "done") {
        const j = ev.result || {};
        const segs = Array.isArray(j.segments) ? j.segments.map(s => ({
          start: s.start || 0, end: s.end || 0, text: s.text || "", speaker: s.speaker || null,
        })) : [{ start: 0, end: j.duration || 0, text: j.text || j.rendered || "", speaker: null }];
        const names = {};
        for (const s of segs) if (s.speaker) names[s.speaker] = String(s.speaker).replace(/^SPEAKER_/, "Speaker ");
        sttState = {
          sourceName: f.name || "recording.webm",
          model: j.model || model,
          routing: j.routing || null,
          language: j.language || "",
          duration: j.duration || 0,
          segments: segs,
          names,
          text: j.text || "",
          summary: "",
          summaryStale: false,
        };
        persistSttState();
        const elapsed = (performance.now() - t0) / 1000;
        paintSttResult({
          elapsed,
          footer: elapsed.toFixed(2) + " s" + (j.model ? ", model " + j.model : "") +
            (j.routing && j.routing.redirected ? ", routed: " + j.routing.reason : "") +
            (segs.some(s => s.speaker) ? ", " + speakerIds().length + " speakers" : "") +
            (j.language ? ", language " + j.language : ""),
        });
        setSttProg(true, 100, "done in " + elapsed.toFixed(2) + "s · model " + (j.model || model));
      }
    }
  } catch (e) {
    let msg = e && e.message ? e.message : String(e);
    if (e.name === "AbortError") msg = "cancelled";
    if (/NetworkError|Failed to fetch|network error|Load failed/i.test(msg)) {
      msg += " (no usable response — server may have closed the connection, rejected a too-large upload, or crashed; check pyre logs)";
    }
    $("#sttOut").textContent = "error: " + msg;
    setSttProg(true, 0, "error");
  } finally {
    if ($("#sttGo")) { delete $("#sttGo").dataset.forceBusy; }
    setSttRunningUi(false);
    updateSttRunAvail();
    sttJob = null;
  }
};

function wireDl(id, kind) {
  const el = document.getElementById(id); if (!el) return;
  el.onclick = () => {
    if (!sttState) return alert("run a transcription first");
    const base = sttBaseName();
    if (kind === "txt") downloadBlob(base + ".txt", buildTxt(), "text/plain");
    else if (kind === "srt") downloadBlob(base + ".srt", buildSrt(), "application/x-subrip");
    else if (kind === "vtt") downloadBlob(base + ".vtt", buildVtt(), "text/vtt");
    else if (kind === "json") downloadBlob(base + ".json", buildJson(), "application/json");
    else if (kind === "md") downloadBlob(base + ".md", buildMd(), "text/markdown");
  };
}
// fix vtt mime
if ($("#sttDlVtt")) $("#sttDlVtt").onclick = () => {
  if (!sttState) return alert("run a transcription first");
  downloadBlob(sttBaseName() + ".vtt", buildVtt(), "text/vtt");
};
wireDl("sttDlTxt", "txt");
wireDl("sttDlSrt", "srt");
wireDl("sttDlJson", "json");
wireDl("sttDlMd", "md");

if ($("#sttToChat")) $("#sttToChat").onclick = () => {
  const i = $("#input"); i.value = (i.value ? i.value + " " : "") + labelledText().trim();
  const tab = document.querySelector('#tabs [data-tab="chat"]'); if (tab) tab.click(); i.focus();
};

async function summarizeTranscript() {
  if (!sttState) return alert("run a transcription first");
  const model = $("#modelPick") && $("#modelPick").value;
  if (!model) return alert("No text model is loaded. Load Qwen (or another LLM) on the Models page.");
  const transcript = labelledText().trim();
  if (!transcript) return alert("empty transcript");
  $("#sttSumBox").classList.remove("hidden");
  $("#sttSumOut").textContent = "";
  if ($("#sttSumRegen")) $("#sttSumRegen").classList.add("hidden");
  if ($("#sttSumCancel")) $("#sttSumCancel").classList.remove("hidden");
  setSumProg(true, 5, "summarizing…");
  sttSumCtrl = new AbortController();
  // Summaries run with reasoning off: measured on a ~45 min (9.4k-token) transcript, medium effort spent
  // ~1.4k reasoning tokens (~85 s) without improving the summary. Override with localStorage pyre.sumEffort.
  const effort = (localStorage.getItem("pyre.sumEffort") || "none");
  // ~6k chars per chunk keeps plenty of headroom even for smaller ctx; n_ctx here is 131k so we can go larger
  // Prefill dominates (~50 tok/s), so a single pass beats map-reduce (209 s vs 340 s for 36k chars):
  // only split very long transcripts (~>20k tokens) to keep KV growth well under the VRAM floor.
  const CHUNK = 80000;
  const parts = [];
  if (transcript.length <= CHUNK) parts.push(transcript);
  else {
    let i = 0;
    while (i < transcript.length) {
      let end = Math.min(transcript.length, i + CHUNK);
      if (end < transcript.length) {
        const nl = transcript.lastIndexOf("\n", end);
        if (nl > i + CHUNK * 0.5) end = nl;
      }
      parts.push(transcript.slice(i, end));
      i = end;
    }
  }
  const sys = "You are a precise meeting analyst. Summarize the transcript with: (1) Key points, (2) Decisions, (3) Action items (owner + due if stated), (4) Per-speaker notes when speakers are labelled. Use the speaker names exactly as given. Be concise; use markdown headings and bullets.";
  try {
    let combined = "";
    if (parts.length === 1) {
      setSumProg(true, 20, "streaming summary…");
      combined = await streamChatSummary(model, sys, "Summarize this transcript:\n\n" + parts[0], effort, sttSumCtrl.signal, (t) => {
        $("#sttSumOut").innerHTML = md(t);
        setSumProg(true, 20 + Math.min(70, t.length / 40), "streaming…");
      });
    } else {
      const partials = [];
      for (let i = 0; i < parts.length; i++) {
        setSumProg(true, 10 + 60 * (i / parts.length), "chunk " + (i + 1) + "/" + parts.length);
        const p = await streamChatSummary(model, sys,
          "This is part " + (i + 1) + " of " + parts.length + " of a longer transcript. Extract key points, decisions, action items, and per-speaker notes. Speaker names must be preserved.\n\n" + parts[i],
          effort, sttSumCtrl.signal, (t) => { $("#sttSumOut").innerHTML = md("_Chunk " + (i + 1) + "/" + parts.length + "_\n\n" + t); });
        partials.push(p);
      }
      setSumProg(true, 80, "merging chunks…");
      combined = await streamChatSummary(model, sys,
        "Merge these partial summaries of one meeting into a single clean summary with Key points, Decisions, Action items, and Per-speaker notes. Deduplicate. Keep speaker names.\n\n" +
        partials.map((p, i) => "### Part " + (i + 1) + "\n" + p).join("\n\n"),
        effort, sttSumCtrl.signal, (t) => { $("#sttSumOut").innerHTML = md(t); });
    }
    sttState.summary = combined;
    sttState.summaryStale = false;
    persistSttState();
    $("#sttSumOut").innerHTML = md(combined);
    setSumProg(true, 100, "done");
  } catch (e) {
    if (e.name === "AbortError") setSumProg(true, 0, "cancelled");
    else { setSumProg(true, 0, "error"); $("#sttSumOut").textContent = "error: " + (e.message || e); }
  } finally {
    sttSumCtrl = null;
    if ($("#sttSumCancel")) $("#sttSumCancel").classList.add("hidden");
  }
}
async function streamChatSummary(model, system, user, effort, signal, onDelta) {
  let out = "";
  const body = {
    model, stream: true, reasoning_effort: effort || "medium",
    temperature: 0.3, max_tokens: 4096,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
  };
  for await (const ch of sse("/v1/chat/completions", body, signal)) {
    if (ch.error) throw new Error(ch.error.message || "chat error");
    const d = ch.choices?.[0]?.delta?.content;
    if (d) { out += d; if (onDelta) onDelta(out); }
  }
  return plainAnswer({ content: out }) || out;
}
if ($("#sttSummarize")) $("#sttSummarize").onclick = () => summarizeTranscript();
if ($("#sttSumRegen")) $("#sttSumRegen").onclick = () => summarizeTranscript();
if ($("#sttSumCancel")) $("#sttSumCancel").onclick = () => { if (sttSumCtrl) sttSumCtrl.abort(); };
if ($("#sttSumCopy")) $("#sttSumCopy").onclick = async () => {
  const t = sttState && sttState.summary; if (!t) return;
  try { await navigator.clipboard.writeText(t); } catch { prompt("Copy summary:", t); }
};
if ($("#sttSumDlMd")) $("#sttSumDlMd").onclick = () => {
  if (!sttState || !sttState.summary) return;
  downloadBlob(sttBaseName() + "_summary.md", "# Summary — " + (sttState.sourceName || "") + "\n\n" + sttState.summary, "text/markdown");
};
if ($("#sttSumDlTxt")) $("#sttSumDlTxt").onclick = () => {
  if (!sttState || !sttState.summary) return;
  // strip light markdown for txt
  const t = sttState.summary.replace(/^#+\s+/gm, "").replace(/\*\*([^*]+)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1");
  downloadBlob(sttBaseName() + "_summary.txt", t, "text/plain");
};

fetch("/api/diarizer", { headers: headers() }).then(r => r.json()).then(j => {
  const ok = !!j.loaded; if ($("#sttDiar")) $("#sttDiar").disabled = !ok;
  if ($("#sttDiarL")) $("#sttDiarL").title = ok ? ("Who spoke when: " + j.info.model + " on " + j.info.backend) : "No diarization model loaded (start with --load diar=nvidia/Nemotron-3-Diarization)";
  applySttPrefs(loadSttPrefs());
}).catch(() => {});
refreshSttModels();
applySttPrefs(loadSttPrefs());
restoreSttState();

async function refreshVoicePicker() {
  try {
    const j = await api("/v1/audio/voices");
    const voices = (j && j.voices) || [];
    const fill = (sel) => {
      if (!sel) return;
      const cur = sel.value;
      sel.innerHTML = "";
      voices.forEach(v => {
        const o = document.createElement("option");
        o.value = v.id;
        o.textContent = v.name + (v.builtin ? "" : (v.synthesizable ? "" : " (saved — synth pending)"));
        o.disabled = !v.builtin && !v.synthesizable;
        sel.appendChild(o);
      });
      try { sel.value = cur; } catch {}
      if (!sel.value && voices[0]) sel.value = voices[0].id;
    };
    fill($("#ttsVoice"));
    fill($("#stsVoice"));
    const box = $("#ttsVoicesList");
    if (box) {
      if (!voices.length) box.textContent = "No voices yet.";
      else {
        box.innerHTML = voices.map(v => {
          const pending = !v.builtin && !v.synthesizable ? " <span class=\"muted\">(synth pending)</span>" : "";
          const del = v.builtin ? "" : (" <button type=\"button\" data-voice-del=\"" + v.id + "\" class=\"ghost small\">Delete</button>");
          const prev = v.has_reference ? (" <a href=\"/v1/audio/voices/" + encodeURIComponent(v.id) + "/reference\" target=\"_blank\" rel=\"noopener\">preview</a>") : "";
          return "<div><b>" + v.name + "</b> <code>" + v.id + "</code>" + pending + prev + del + "</div>";
        }).join("");
        box.querySelectorAll("[data-voice-del]").forEach(btn => {
          btn.onclick = async () => {
            const id = btn.getAttribute("data-voice-del");
            if (!confirm("Delete voice " + id + "?")) return;
            const r = await fetch("/v1/audio/voices/" + encodeURIComponent(id), { method: "DELETE", headers: headers() });
            if (!r.ok) { showToast(await r.text()); return; }
            refreshVoicePicker();
          };
        });
      }
      const note = $("#ttsVoiceNote");
      if (note && j.clone_engine) note.textContent = j.clone_engine.message || note.textContent;
    }
  } catch (e) { console.warn("voices", e); }
}

async function saveVoiceFromBlob(name, blob, source) {
  if (!name || !blob) { showToast("Name and audio are required"); return null; }
  const fd = new FormData();
  fd.append("name", name);
  fd.append("source", source || "ui");
  fd.append("file", blob, (blob.name || "reference.webm"));
  const r = await fetch("/v1/audio/voices", { method: "POST", headers: headers(), body: fd });
  const txt = await r.text();
  let j = null; try { j = JSON.parse(txt); } catch {}
  if (!r.ok) { showToast((j && j.error && j.error.message) || txt); return null; }
  showToast((j && j.message) || "Voice saved");
  await refreshVoicePicker();
  return j;
}

async function showDirectionsHelp() {
  const dlg = $("#ttsDirDlg"); const body = $("#ttsDirBody");
  if (!dlg || !body) return;
  body.textContent = "Loading…";
  dlg.showModal();
  try {
    const j = await api("/v1/audio/directions");
    const ev = (j.native_syntax && j.native_syntax.event_tags) || [];
    const del = (j.native_syntax && j.native_syntax.delivery_tags) || [];
    const stages = (j.engine && j.engine.stages) || [];
    body.innerHTML =
      "<p><b>Engine:</b> " + (j.engine && j.engine.id || "?") + " (" + (j.engine && j.engine.licence || "") + ") — " + (j.engine && j.engine.status || "") + "</p>" +
      "<p><b>Native example:</b> <code>" + (j.native_syntax && j.native_syntax.example || "") + "</code></p>" +
      "<p><b>Event tags:</b> " + ev.map(t => "<code>" + t + "</code>").join(" ") + "</p>" +
      "<p><b>Delivery tags:</b> " + del.map(t => "<code>" + t + "</code>").join(" ") + "</p>" +
      "<p><b>Portable timing:</b> <code>" + (j.portable_timing && j.portable_timing.example || "") + "</code></p>" +
      "<p class=\"muted small\">" + ((j.native_syntax && j.native_syntax.notes) || []).join(" ") + "</p>" +
      "<h4>Port stages</h4><ul>" + stages.map(s => "<li><code>" + s.id + "</code> — " + s.state + "</li>").join("") + "</ul>";
  } catch (e) { body.textContent = String(e); }
}

if ($("#ttsDirHelp")) $("#ttsDirHelp").onclick = () => showDirectionsHelp();
if ($("#ttsCloneSave")) $("#ttsCloneSave").onclick = async () => {
  const name = ($("#ttsCloneName") && $("#ttsCloneName").value || "").trim();
  const f = $("#ttsCloneFile") && $("#ttsCloneFile").files && $("#ttsCloneFile").files[0];
  if (!f) { showToast("Pick a reference audio file"); return; }
  await saveVoiceFromBlob(name, f, "tts-upload");
};
if ($("#sttCloneVoice")) $("#sttCloneVoice").onclick = async () => {
  const a = (typeof currentSttAudio === "function") && currentSttAudio();
  if (!a) { showToast("Transcribe or load audio first"); return; }
  const name = prompt("Name for this voice:");
  if (!name) return;
  await saveVoiceFromBlob(name.trim(), a, "stt");
};
if ($("#stsCloneVoice")) $("#stsCloneVoice").onclick = async () => {
  const f = (typeof currentStsAudio === "function") && currentStsAudio();
  if (!f) { showToast("Provide STS audio first"); return; }
  const name = prompt("Name for this voice:");
  if (!name) return;
  await saveVoiceFromBlob(name.trim(), f, "sts");
};

if ($("#ttsGo")) $("#ttsGo").onclick = async () => {
  const input = $("#ttsText").value || "";
  if (!input.trim()) { showToast("Enter text to speak"); return; }
  const voice = ($("#ttsVoice") && $("#ttsVoice").value) || "mms";
  const speed = parseFloat(($("#ttsSpeed") && $("#ttsSpeed").value) || "1") || 1;
  $("#ttsProg").classList.remove("hidden");
  $("#ttsProgLabel").textContent = "Speaking…";
  $("#ttsDirIgnored").classList.add("hidden");
  try {
    const r = await fetch("/v1/audio/speech", {
      method: "POST", headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ input, voice_id: voice, speed, response_format: "wav" })
    });
    const metaRaw = r.headers.get("x-pyre-speech-meta");
    if (metaRaw) {
      try {
        const meta = JSON.parse(metaRaw);
        if (meta.ignored && meta.ignored.length) {
          $("#ttsDirIgnored").classList.remove("hidden");
          $("#ttsDirIgnored").textContent = "Ignored directions: " + meta.ignored.map(i => i.tag + " (" + i.reason + ")").join("; ");
        }
      } catch {}
    }
    if (!r.ok) {
      const txt = await r.text();
      let msg = txt; try { msg = JSON.parse(txt).error?.message || txt; } catch {}
      showToast(msg); return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = $("#ttsAudio"); a.src = url; a.classList.remove("hidden"); a.play().catch(()=>{});
  } catch (e) { showToast(String(e)); }
  finally { $("#ttsProg").classList.add("hidden"); }
};
refreshVoicePicker();


const STS_DEFAULT_SYS = "You are a helpful voice assistant. Answer clearly and concisely in plain spoken language. Avoid markdown, bullet lists, and URLs unless asked.";
function loadStsPrefs() {
  return LS.get("stsPrefs", { quality: "balanced", voice: "default", speed: 1.0, system: STS_DEFAULT_SYS });
}
function saveStsPrefs(p) { LS.set("stsPrefs", p); }
function stsPrefsFromUi() {
  return {
    quality: ($("#stsQuality") && $("#stsQuality").value) || "balanced",
    voice: ($("#stsVoice") && $("#stsVoice").value) || "default",
    speed: ($("#stsSpeed") && +$("#stsSpeed").value) || 1.0,
    system: ($("#stsSystem") && $("#stsSystem").value) || STS_DEFAULT_SYS,
  };
}
function applyStsPrefs(p) {
  if ($("#stsQuality")) $("#stsQuality").value = p.quality || "balanced";
  if ($("#stsQualitySeg")) $$("#stsQualitySeg .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.stsQ === ($("#stsQuality").value)));
  if ($("#stsVoice") && p.voice) try { $("#stsVoice").value = p.voice; } catch {}
  if ($("#stsSpeed") && p.speed != null) $("#stsSpeed").value = p.speed;
  if ($("#stsSystem")) $("#stsSystem").value = p.system || STS_DEFAULT_SYS;
}
function setStsSourceMode(mode) {
  const file = mode !== "record";
  if ($("#stsModeFile")) $("#stsModeFile").classList.toggle("active", file);
  if ($("#stsModeRec")) $("#stsModeRec").classList.toggle("active", !file);
  if ($("#stsFilePane")) $("#stsFilePane").classList.toggle("hidden", !file);
  if ($("#stsRecPane")) $("#stsRecPane").classList.toggle("hidden", file);
  if ($("#stsMicHint")) {
    const ok = micAllowed();
    $("#stsMicHint").classList.toggle("hidden", ok);
    if ($("#stsLocalUrl")) $("#stsLocalUrl").textContent = "http://127.0.0.1:" + (location.port || "8099") + "/";
  }
  if ($("#stsRecBtn")) setAvail($("#stsRecBtn"), micAllowed() ? availOk() : { enabled: false, reason_kind: "no_permission", message: "Microphone needs HTTPS or localhost" });
}
function currentStsAudio() {
  if (stsRecBlob) return stsRecBlob;
  const inp = $("#stsFile");
  return (inp && inp.files && inp.files[0]) || null;
}
function setStsProg(show, pct, label) {
  const box = $("#stsProg"); if (!box) return;
  box.classList.toggle("hidden", !show);
  if (pct != null && $("#stsBar")) $("#stsBar").style.width = Math.max(0, Math.min(100, pct)) + "%";
  if (label != null && $("#stsProgLabel")) $("#stsProgLabel").textContent = label;
}
function syncStsVoicesFromTts() {
  const src = $("#ttsVoice"); const dst = $("#stsVoice"); if (!src || !dst) return;
  const cur = dst.value;
  dst.innerHTML = src.innerHTML;
  if ([...dst.options].some(o => o.value === cur)) dst.value = cur;
}
if ($("#stsModeFile")) $("#stsModeFile").onclick = () => setStsSourceMode("file");
if ($("#stsModeRec")) $("#stsModeRec").onclick = () => setStsSourceMode("record");
if ($("#stsDrop")) {
  const dz = $("#stsDrop");
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (!f) return;
    const dt = new DataTransfer(); dt.items.add(f);
    $("#stsFile").files = dt.files; stsRecBlob = null;
    if ($("#stsFileMeta")) $("#stsFileMeta").textContent = (f.name || "audio") + (f.size ? (" · " + human(f.size)) : "");
    dz.classList.add("has-file");
    if ($("#stsPreview")) { $("#stsPreview").src = URL.createObjectURL(f); $("#stsPreview").classList.remove("hidden"); }
  });
}
if ($("#stsFile")) $("#stsFile").onchange = () => {
  stsRecBlob = null;
  const f = $("#stsFile").files[0];
  if (f) {
    if ($("#stsFileMeta")) $("#stsFileMeta").textContent = (f.name || "audio") + (f.size ? (" · " + human(f.size)) : "");
    if ($("#stsDrop")) $("#stsDrop").classList.add("has-file");
    if ($("#stsPreview")) { $("#stsPreview").src = URL.createObjectURL(f); $("#stsPreview").classList.remove("hidden"); }
  }
};
if ($("#stsQualitySeg")) $("#stsQualitySeg").onclick = (e) => {
  const b = e.target.closest("[data-sts-q]"); if (!b) return;
  $$("#stsQualitySeg .seg-btn").forEach(x => x.classList.toggle("active", x === b));
  if ($("#stsQuality")) { $("#stsQuality").value = b.dataset.stsQ; saveStsPrefs(stsPrefsFromUi()); }
};
["stsVoice", "stsSpeed", "stsSystem"].forEach(id => {
  const el = document.getElementById(id); if (el) el.addEventListener("change", () => saveStsPrefs(stsPrefsFromUi()));
});
if ($("#stsRecBtn")) $("#stsRecBtn").onclick = async () => {
  if (stsRec) { stsRec.stop(); return; }
  if (!micAllowed()) { setStsSourceMode("record"); return; }
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    stsRec = new MediaRecorder(s);
    stsRec.ondataavailable = e => chunks.push(e.data);
    stsRec.onstop = () => {
      stsRecBlob = new Blob(chunks, { type: stsRec.mimeType || "audio/webm" });
      try { s.getTracks().forEach(t => t.stop()); } catch {}
      stsRec = null;
      if (stsRecTimer) { clearInterval(stsRecTimer); stsRecTimer = null; }
      $("#stsRecBtn").textContent = "Start recording";
      if ($("#stsRecAfter")) $("#stsRecAfter").classList.remove("hidden");
      if ($("#stsRecMeta")) $("#stsRecMeta").textContent = "Recording ready · " + human(stsRecBlob.size);
      if ($("#stsFile")) try { $("#stsFile").value = ""; } catch {}
      if ($("#stsPreview")) { $("#stsPreview").src = URL.createObjectURL(stsRecBlob); $("#stsPreview").classList.remove("hidden"); }
    };
    stsRec.start();
    $("#stsRecBtn").textContent = "Stop recording";
    if ($("#stsRecAfter")) $("#stsRecAfter").classList.add("hidden");
    stsRecStarted = Date.now();
    if ($("#stsRecTimer")) $("#stsRecTimer").textContent = "0:00";
    stsRecTimer = setInterval(() => {
      if ($("#stsRecTimer")) $("#stsRecTimer").textContent = fmtDur((Date.now() - stsRecStarted) / 1000);
    }, 250);
  } catch (e) { alert("microphone: " + e.message); }
};
if ($("#stsRecAgain")) $("#stsRecAgain").onclick = () => {
  stsRecBlob = null;
  if ($("#stsRecAfter")) $("#stsRecAfter").classList.add("hidden");
  if ($("#stsPreview")) { $("#stsPreview").removeAttribute("src"); $("#stsPreview").classList.add("hidden"); }
  if ($("#stsRecBtn") && !$("#stsRecBtn").disabled) $("#stsRecBtn").click();
};
if ($("#stsCancel")) $("#stsCancel").onclick = () => { if (stsCtrl) stsCtrl.abort(); };
if ($("#stsDlWav")) $("#stsDlWav").onclick = () => {
  if (!stsReplyBlob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(stsReplyBlob);
  a.download = "sts-reply.wav";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
if ($("#stsToChat")) $("#stsToChat").onclick = () => {
  const t = ($("#stsReply") && $("#stsReply").textContent) || "";
  if (!t.trim()) return;
  const c = conv(); if (!c) return;
  c.messages.push({ role: "assistant", content: t });
  renderMessages(); save();
  onTab("chat");
};
if ($("#stsGo")) $("#stsGo").onclick = async () => {
  const f = currentStsAudio(); if (!f) return alert("Choose or record audio first");
  const pref = stsPrefsFromUi(); saveStsPrefs(pref);
  const model = QUALITY_MODELS[pref.quality] || "base";
  const llm = (kinds.llm && kinds.llm[0] && kinds.llm[0].id) || ($("#model") && $("#model").value) || "";
  if (!llm) return alert("No chat model loaded");
  const ttsModel = kinds.tts && kinds.tts[0] && kinds.tts[0].id;
  if (!ttsModel) return alert("No TTS model loaded");
  stsCtrl = new AbortController();
  stsReplyBlob = null;
  if ($("#stsDlWav")) $("#stsDlWav").classList.add("hidden");
  if ($("#stsToChat")) $("#stsToChat").classList.add("hidden");
  if ($("#stsTranscript")) $("#stsTranscript").textContent = "";
  if ($("#stsReply")) $("#stsReply").textContent = "";
  if ($("#stsAudio")) { $("#stsAudio").classList.add("hidden"); $("#stsAudio").removeAttribute("src"); }
  $("#stsGo").classList.add("hidden");
  $("#stsCancel").classList.remove("hidden");
  try {
    resetSttBar();
    setStsProg(true, 0, "Transcribing…");
    const fd = new FormData();
    fd.append("file", f, f.name || "recording.webm");
    fd.append("model", model);
    fd.append("response_format", "verbose_json");
    fd.append("task", "transcriptions");
    let transcript = "";
    for await (const ev of audioJobSse(fd, stsCtrl.signal)) {
      if (ev.stage === "error" || ev.stage === "cancelled") throw new Error(ev.message || ev.stage);
      const pct = sttStagePct(ev);
      setStsProg(true, pct != null ? Math.min(40, pct * 0.4) : null, ev.message || ev.stage || "transcribing");
      if (ev.stage === "done") {
        const j = ev.result || {};
        transcript = (j.text || j.rendered || "").trim();
      }
    }
    if (!transcript) throw new Error("empty transcript");
    if ($("#stsTranscript")) $("#stsTranscript").textContent = transcript;
    setStsProg(true, 45, "Thinking…");
    let reply = "";
    const body = {
      model: llm,
      stream: true,
      reasoning_effort: "medium",
      messages: [
        { role: "system", content: pref.system || STS_DEFAULT_SYS },
        { role: "user", content: transcript },
      ],
    };
    for await (const ch of sse("/v1/chat/completions", body, stsCtrl.signal)) {
      if (ch.error) throw new Error(ch.error.message || "chat error");
      const d = ch.choices && ch.choices[0] && ch.choices[0].delta && ch.choices[0].delta.content;
      if (d) {
        reply += d;
        if ($("#stsReply")) $("#stsReply").textContent = reply;
        setStsProg(true, 55 + Math.min(25, reply.length / 20), "Writing reply…");
      }
    }
    reply = (reply || "").trim();
    if (!reply) throw new Error("empty chat reply");
    const spoken = reply.replace(/^#+\s+/gm, "").replace(/\*\*([^*]+)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1").replace(/^\s*[-*]\s+/gm, "");
    setStsProg(true, 85, "Synthesizing speech…");
    // One-shot TTS for the full reply. Sentence-by-sentence streaming would need
    // N extra /v1/audio/speech round-trips (load+encode per sentence) and is not
    // wired here; expect full-reply latency ~seconds after chat finishes.
    const r = await fetch("/v1/audio/speech", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        model: ttsModel,
        input: spoken,
        voice_id: (pref.voice && pref.voice !== "default") ? pref.voice : "mms", voice: pref.voice || "mms",
        speed: pref.speed || 1.0,
        response_format: "wav",
      }),
      signal: stsCtrl.signal,
    });
    if (!r.ok) {
      let m = r.statusText;
      try { m = (await r.json()).error.message || m; } catch {}
      throw new Error(m);
    }
    stsReplyBlob = await r.blob();
    if ($("#stsAudio")) {
      $("#stsAudio").src = URL.createObjectURL(stsReplyBlob);
      $("#stsAudio").classList.remove("hidden");
      try { $("#stsAudio").play(); } catch {}
    }
    if ($("#stsDlWav")) $("#stsDlWav").classList.remove("hidden");
    if ($("#stsToChat")) $("#stsToChat").classList.remove("hidden");
    setStsProg(true, 100, "done");
  } catch (e) {
    const msg = (e && e.name === "AbortError") ? "cancelled" : ((e && e.message) || String(e));
    setStsProg(true, 0, "error: " + msg);
    if ($("#stsReply") && !$("#stsReply").textContent) $("#stsReply").textContent = "error: " + msg;
  } finally {
    $("#stsGo").classList.remove("hidden");
    $("#stsCancel").classList.add("hidden");
    stsCtrl = null;
  }
};
setStsSourceMode("file");
applyStsPrefs(loadStsPrefs());
{
  const _obsTtsVoice = $("#ttsVoice");
  if (_obsTtsVoice) {
    const mo = new MutationObserver(() => syncStsVoicesFromTts());
    mo.observe(_obsTtsVoice, { childList: true });
    syncStsVoicesFromTts();
  }
}

// ---------- apps ----------
async function loadApps() {
  let r; try { r = await api("/api/apps"); } catch (e) { $("#appsInfo").textContent = e.message; return; }
  $("#appsInfo").textContent = `folder: ${r.apps_dir} · apps are configured with ${r.server}`;
  $("#appsList").innerHTML = r.apps.map(a => { const m = a.manifest?.app || {}; const inst = a.installed; const ents = a.manifest?.entry || [];
    const models = (a.manifest?.models || []).map(x => `${x.kind}: ${x.models[0]}`).join(", ");
    return `<div class="appcard"><div><b>${esc(a.name)}</b> <span class="muted small">${esc(m.version || "")}</span>${inst ? ' <span class="tag llm">installed</span>' : ""}<div class="small">${esc(m.description || "")}</div>
      <div class="muted small">${models ? "needs " + esc(models) : ""}${inst ? " · " + esc(inst.dir) : ""}${inst?.warnings?.length ? " · ⚠ " + esc(inst.warnings.join("; ")) : ""}</div></div>
      <div class="row">${inst ? ents.map(e => `<button class="small" data-apprun="${esc(a.name)}" data-entry="${esc(e.name)}" title="${esc(e.description || "")}">Run ${esc(e.name)}</button>`).join("") + `<button class="small" data-appinst="${esc(a.name)}">Update</button><button class="small danger" data-apprm="${esc(a.name)}">Remove</button>` : `<button class="small primary" data-appinst="${esc(a.name)}">Install</button>`}
      ${m.web_ui ? `<a class="small" href="${esc(m.web_ui)}" target="_blank">Open</a>` : ""}</div></div>`; }).join("");
}
async function appInstall(name) {
  const lg = $("#appLog"); lg.classList.remove("hidden"); lg.textContent = `installing ${name} (building can take a few minutes)…`;
  try { const r = await api("/api/apps/install", { name }); lg.textContent = r.log.join("\n") + (r.models?.length ? "\n\nmodels: " + r.models.map(m => `${m.kind} → ${m.models.join(", ")}`).join("; ") + " (load them on the Models page)" : ""); }
  catch (e) { lg.textContent = "error: " + e.message; }
  loadApps();
}
$("#tab-apps").onclick = async e => {
  const b = e.target;
  try {
    if (b.dataset.appinst) { b.disabled = true; await appInstall(b.dataset.appinst); }
    else if (b.dataset.apprm) { if (!confirm(`Remove ${b.dataset.apprm}? (deletes its folder in pyre's apps dir)`)) return; await api("/api/apps/remove", { name: b.dataset.apprm }); loadApps(); }
    else if (b.dataset.apprun) { const r = await api("/api/apps/run", { name: b.dataset.apprun, entry: b.dataset.entry }); b.textContent = `started (pid ${r.pid})`; }
    else if (b.id === "appAdd") { const p = $("#appSrc").value.trim(); if (p) await appInstall(p); }
  } catch (err) { alert(err.message); }
};

// ---------- status ----------
function limitsText(l) {
  if (!l) return "n/a";
  const v = l.vram_enforced ? `reserve ${l.vram_reserve_mib} MiB · free ${l.vram_free_mib}/${l.vram_total_mib} MiB · headroom ${l.vram_headroom_mib} MiB` : `VRAM reserve off or not measurable (${l.vram_reserve_mib} MiB)`;
  const u = l.gpu_util_max_pct >= 100 ? "no duty-cycle cap" : `GPU util cap ${l.gpu_util_max_pct}% (now ${l.gpu_util_now_pct ?? "?"}%, throttled ${l.throttled_s.toFixed(1)} s)`;
  return v + "<br>" + u + (l.low_priority ? " · low process priority" : "");
}
function metricsText(m) {
  if (!m) return "n/a";
  const act = m.active.map(a => `#${a.id} ${esc(a.model)} ${a.phase} ${a.ctx_used}/${a.n_ctx} ctx, ${a.decode_tps.toFixed(1)} tok/s`).join("<br>");
  const rec = m.recent.slice(0, 5).map(r => `#${r.id} ${esc(r.model)} ${r.outcome}: ${r.prompt_n}+${r.gen_n} tok, prefill ${r.prefill_tps.toFixed(0)} tok/s, decode ${r.decode_tps.toFixed(1)} tok/s`).join("<br>");
  const hr = m.expert_cache.hit_rate == null ? "n/a" : (m.expert_cache.hit_rate * 100).toFixed(1) + "%";
  return `${m.active.length} active, ${m.totals.requests} done (${m.totals.prompt_tokens} prompt / ${m.totals.generated_tokens} generated tok) · expert cache ${hr} · CPU experts ${(m.compute.cpu_expert_ms / 1e3).toFixed(1)} s / GPU ${(m.compute.gpu_ms / 1e3).toFixed(1)} s`
    + (act ? "<br>" + act : "") + (rec ? "<br><i>recent</i><br>" + rec : "");
}
async function loadStatus() {
  try {
    const s = await api("/api/status");
    const g = s.gpus.map(g => `${g.name} — ${g.memory_used_mib}/${g.memory_total_mib} MiB VRAM, ${g.utilization_pct}% util`).join("<br>") || "n/a";
    $("#statusBox").innerHTML = [["version", `pyre ${s.version} (Burn ${s.burn})`], ["default backend", s.default_backend], ["compiled backends", s.backends_compiled.join(", ")], ["GPU", g],
      ["RAM", `process ${human(s.ram.process_rss)} · system ${human(s.ram.used)} / ${human(s.ram.total)}`], ["uptime", s.uptime_s + " s"], ["auth", s.auth ? "API key required" : "open"],
      ["store", s.store_root], ["HF cache", s.hf_cache], ["GPU limits", limitsText(s.limits)],
      ["Qwen context", s.qwen_context ? `${s.qwen_context.context_label} · KV ${s.qwen_context.kv_bits}-bit · ${esc(s.qwen_context.fit?.message || "")}` : "n/a"],
      ["Autotune", s.autotune?.suggest ? esc(s.autotune.message) : (s.autotune?.message || "matches this GPU")],
      ["Settings source", s.profiles ? `${esc(s.profiles.active?.source || "default")}${s.profiles.active?.name ? " · " + esc(s.profiles.active.name) : ""}${s.profiles.tune ? " · tune saved" : " · no tune yet"}${s.profiles.tune_stale ? " · hardware changed, tune kept" : ""}` : "n/a"],
      ["batching", s.batching], ["requests", metricsText(s.metrics)]].map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
    $("#slotsTbl tbody").innerHTML = s.models.map(m => `<tr><td>${esc(m.id)}</td><td><span class="tag ${m.kind}">${m.kind}</span></td><td class="caps">${capHtml(m.capabilities)}</td><td>${m.busy}</td><td>${m.queued}</td></tr>`).join("");
    const base = location.origin; const model = s.models.find(m => m.kind === "llm")?.id || "model"; const key = s.auth ? "$PYRE_API_KEY" : "none";
    $("#apiSnippet").textContent = `# curl
curl ${base}/v1/chat/completions -H "Content-Type: application/json" ${s.auth ? '-H "Authorization: Bearer $PYRE_API_KEY" ' : ""}\\
  -d '{"model":"${model}","messages":[{"role":"user","content":"Hello!"}],"stream":false}'

# Python (openai >= 1.0)
from openai import OpenAI
client = OpenAI(base_url="${base}/v1", api_key="${key}")
r = client.chat.completions.create(model="${model}", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)

# llama.cpp-native
curl ${base}/completion -d '{"prompt":"Once upon a time","n_predict":32}'`;
    const l = await api("/api/logs"); $("#logs").textContent = l.lines.join("\n"); $("#logs").scrollTop = 1e9;
  } catch (e) { $("#statusBox").textContent = e.message; }
}
$("#logRefresh").onclick = loadStatus;

function refreshAll() { loadKinds(); const t = location.hash.slice(1); if (t) onTab(t); }
renderConvs(); renderMsgs(); loadKinds(); loadQwenCtx(); loadTune(); loadProfiles();
// Status pill: paint immediately, then connect (tabs already bound above — never wait on this).
try { renderStatusPill(); connectStatusStream(); } catch (e) { console.error(e); showToast("Status pill failed: " + (e.message || e)); }
const h = location.hash.slice(1); if (h) $(`#tabs button[data-tab="${h}"]`)?.click();
setInterval(() => { if ($("#tab-status").classList.contains("active")) loadStatus(); }, 5000);
