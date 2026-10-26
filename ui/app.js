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

// ---------- tiny markdown ----------
function md(src) {
  const parts = src.split(/```/); let html = "";
  parts.forEach((p, i) => {
    if (i % 2) { const nl = p.indexOf("\n"); const lang = nl > 0 ? p.slice(0, nl).trim() : ""; const code = nl > 0 ? p.slice(nl + 1) : p;
      html += `<pre><button class="copy small">copy</button><code data-lang="${esc(lang)}">${esc(code.replace(/\n$/, ""))}</code></pre>`; return; }
    let t = esc(p);
    t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
      .replace(/^#{1,6} (.*)$/gm, "<b>$1</b>").replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    const lines = t.split("\n"); let out = "", list = null;
    for (const l of lines) {
      const m = l.match(/^\s*([-*]|\d+\.)\s+(.*)/);
      if (m) { const tag = /\d/.test(m[1]) ? "ol" : "ul"; if (list !== tag) { if (list) out += `</${list}>`; out += `<${tag}>`; list = tag; } out += `<li>${m[2]}</li>`; continue; }
      if (!l.trim()) continue; // blank lines don't break a list
      if (list && /^\s{2,}/.test(l)) { out = out.replace(/<\/li>$/, `<br>${l.trim()}</li>`); continue; }
      if (list) { out += `</${list}>`; list = null; }
      out += `<p>${l}</p>`;
    }
    if (list) out += `</${list}>`; html += out;
  });
  return html;
}
document.addEventListener("click", e => { if (e.target.classList.contains("copy")) { navigator.clipboard.writeText(e.target.nextElementSibling.textContent); e.target.textContent = "copied"; setTimeout(() => e.target.textContent = "copy", 1000); } });

// ---------- tabs / theme ----------
$$("#tabs button").forEach(b => b.onclick = () => { $$("#tabs button").forEach(x => x.classList.toggle("active", x === b)); $$(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab)); location.hash = b.dataset.tab; onTab(b.dataset.tab); });
function onTab(t) { if (t === "models") loadModels(); if (t === "status") loadStatus(); if (t === "images" || t === "audio") loadKinds(); }
document.documentElement.dataset.theme = LS.get("theme", matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
$("#themeBtn").onclick = () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = t; LS.set("theme", t); };
$("#keyBtn").onclick = askKey;

// ---------- chat ----------
let convs = LS.get("convs", []); let cur = LS.get("cur", null); let ctrl = null;
const settingsIds = ["sysPrompt", "sTemp", "sTopP", "sTopK", "sMax", "sSeed"];
const saved = LS.get("settings", {}); settingsIds.forEach(id => { if (saved[id] !== undefined) $("#" + id).value = saved[id]; $("#" + id).onchange = () => { const s = {}; settingsIds.forEach(i => s[i] = $("#" + i).value); LS.set("settings", s); }; });
$("#settingsBtn").onclick = () => $("#settings").classList.toggle("hidden");
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
  d.innerHTML = `<div class="role"><span>${m.role}</span><span class="muted">${m.stats ? esc(m.stats) : ""}</span></div><div class="body">${m.role === "assistant" ? md(m.content) : esc(m.content).replace(/\n/g, "<br>")}</div>`;
  d.dataset.i = i; return d;
}
function renderMsgs() {
  const c = conv(); const box = $("#messages"); box.innerHTML = "";
  if (!c) { box.innerHTML = `<div class="muted" style="margin:auto">Start a new chat.</div>`; return; }
  c.messages.forEach((m, i) => box.appendChild(msgEl(m, i))); box.scrollTop = box.scrollHeight;
}
async function generate() {
  const c = conv(); const model = $("#modelPick").value;
  const msgs = []; const sys = $("#sysPrompt").value.trim(); if (sys) msgs.push({ role: "system", content: sys });
  c.messages.forEach(m => msgs.push({ role: m.role, content: m.content }));
  const body = { model, messages: msgs, stream: true, stream_options: { include_usage: true },
    temperature: +$("#sTemp").value, top_p: +$("#sTopP").value, top_k: +$("#sTopK").value, max_tokens: +$("#sMax").value };
  if ($("#sSeed").value !== "") body.seed = +$("#sSeed").value;
  const a = { role: "assistant", content: "" }; c.messages.push(a);
  const el = msgEl(a, c.messages.length - 1); $("#messages").appendChild(el); const bodyEl = $(".body", el);
  ctrl = new AbortController(); $("#stopBtn").classList.remove("hidden"); $("#sendBtn").disabled = true;
  const t0 = performance.now(); let first = 0, n = 0, timings = null;
  try {
    for await (const ch of sse("/v1/chat/completions", body, ctrl.signal)) {
      if (ch.error) throw new Error(ch.error.message);
      const d = ch.choices?.[0]?.delta?.content; if (d) { if (!first) first = performance.now(); n++; a.content += d; bodyEl.innerHTML = md(a.content); $("#messages").scrollTop = 1e9; }
      if (ch.timings) timings = ch.timings;
    }
  } catch (e) { if (e.name !== "AbortError") a.content += `\n\n**error:** ${e.message}`; bodyEl.innerHTML = md(a.content); }
  finally {
    ctrl = null; $("#stopBtn").classList.add("hidden"); $("#sendBtn").disabled = false;
    if (timings) a.stats = `${timings.predicted_n} tok · ${timings.predicted_per_second.toFixed(1)} tok/s · prompt ${timings.prompt_per_second.toFixed(0)} tok/s`;
    else if (n) a.stats = `${n} chunks · ${(n / ((performance.now() - (first || t0)) / 1000)).toFixed(1)} tok/s`;
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

// ---------- models ----------
let kinds = {};
async function loadKinds() {
  try {
    const r = await api("/v1/models"); kinds = {}; r.data.forEach(m => (kinds[m.kind] ??= []).push(m));
    const sel = $("#modelPick"); const prev = sel.value || LS.get("model", "");
    sel.innerHTML = (kinds.llm || []).map(m => `<option>${esc(m.id)}</option>`).join("") || `<option value="">(no text model loaded)</option>`;
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
    const en = (k, form, note) => { const on = !!(kinds[k]?.length) && kinds[k].some(m => m.meta?.status !== "not-runnable"); $(form).disabled = !on; $(note).classList.toggle("hidden", on); };
    en("image", "#imgForm", "#imgNote"); en("stt", "#sttForm", "#sttNote"); en("tts", "#ttsForm", "#ttsNote");
  } catch (e) { console.warn(e); }
}
$("#modelPick").onchange = () => LS.set("model", $("#modelPick").value);
async function loadModels() {
  let r; try { r = await api("/api/models"); } catch (e) { $("#storeInfo").textContent = e.message; return; }
  $("#storeInfo").textContent = `store: ${r.store_root} · HF cache: ${r.hf_cache}${r.offline ? " · OFFLINE" : ""}`;
  $("#loadedTbl tbody").innerHTML = r.loaded.map(m => `<tr><td>${esc(m.id)}</td><td><span class="tag ${m.kind}">${m.kind}</span></td><td>${esc(m.backend)} ${m.meta?.dtype ? "· " + m.meta.dtype : ""}</td><td>${m.busy ? "busy" : "idle"}</td><td><button class="small" data-unload="${esc(m.id)}">Unload</button></td></tr>`).join("") || `<tr><td colspan=5 class="muted">nothing loaded</td></tr>`;
  $("#availTbl tbody").innerHTML = r.models.map(m => `<tr><td>${esc(m.name)}${m.alias ? ` <span class="tag">${esc(m.alias)}</span>` : ""}</td><td><span class="tag ${m.kind}">${m.kind}</span></td><td>${m.source}</td><td>${human(m.size)}</td><td>${esc(m.storage || "")}</td><td class="loc">${esc(String(m.location))}</td>
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
  $("#imgGo").disabled = true; $("#imgGo").textContent = "generating…";
  try { const r = await api("/v1/images/generations", body); r.data.forEach(d => { const im = new Image(); im.src = "data:image/png;base64," + d.b64_json; im.title = body.prompt; $("#gallery").prepend(im); }); }
  catch (err) { alert(err.message); }
  $("#imgGo").disabled = false; $("#imgGo").textContent = "Generate";
};

// ---------- audio ----------
let recBlob = null, rec = null;
$("#recBtn").onclick = async () => {
  if (rec) { rec.stop(); return; }
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true }); const chunks = []; rec = new MediaRecorder(s);
    rec.ondataavailable = e => chunks.push(e.data); rec.onstop = () => { recBlob = new Blob(chunks, { type: rec.mimeType }); s.getTracks().forEach(t => t.stop()); rec = null; $("#recBtn").textContent = "● Record"; $("#sttAudio").src = URL.createObjectURL(recBlob); $("#sttAudio").classList.remove("hidden"); };
    rec.start(); $("#recBtn").textContent = "■ Stop";
  } catch (e) { alert("microphone: " + e.message); }
};
$("#sttFile").onchange = () => { recBlob = null; const f = $("#sttFile").files[0]; if (f) { $("#sttAudio").src = URL.createObjectURL(f); $("#sttAudio").classList.remove("hidden"); } };
$("#sttGo").onclick = async () => {
  const f = recBlob || $("#sttFile").files[0]; if (!f) return alert("choose or record audio first");
  const fd = new FormData(); fd.append("file", f, f.name || "recording.webm"); fd.append("model", (kinds.stt?.[0]?.id) || "whisper"); fd.append("response_format", "verbose_json");
  if ($("#sttLang").value) fd.append("language", $("#sttLang").value);
  $("#sttOut").textContent = "…"; const t0 = performance.now();
  try {
    const r = await fetch("/v1/audio/" + $("#sttTask").value, { method: "POST", headers: headers(), body: fd }); const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message || r.statusText);
    $("#sttOut").textContent = `${j.text}\n\n(${((performance.now() - t0) / 1000).toFixed(2)} s${j.language ? ", language " + j.language : ""})`;
  } catch (e) { $("#sttOut").textContent = "error: " + e.message; }
};
$("#ttsGo").onclick = async () => {
  try {
    const r = await fetch("/v1/audio/speech", { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify({ model: kinds.tts?.[0]?.id, input: $("#ttsText").value, voice: $("#ttsVoice").value, speed: +$("#ttsSpeed").value, response_format: "wav" }) });
    if (!r.ok) throw new Error((await r.json()).error?.message);
    $("#ttsAudio").src = URL.createObjectURL(await r.blob()); $("#ttsAudio").classList.remove("hidden"); $("#ttsAudio").play();
  } catch (e) { alert(e.message); }
};

// ---------- status ----------
async function loadStatus() {
  try {
    const s = await api("/api/status");
    const g = s.gpus.map(g => `${g.name} — ${g.memory_used_mib}/${g.memory_total_mib} MiB VRAM, ${g.utilization_pct}% util`).join("<br>") || "n/a";
    $("#statusBox").innerHTML = [["version", `pyre ${s.version} (Burn ${s.burn})`], ["default backend", s.default_backend], ["compiled backends", s.backends_compiled.join(", ")], ["GPU", g],
      ["RAM", `process ${human(s.ram.process_rss)} · system ${human(s.ram.used)} / ${human(s.ram.total)}`], ["uptime", s.uptime_s + " s"], ["auth", s.auth ? "API key required" : "open"],
      ["store", s.store_root], ["HF cache", s.hf_cache], ["batching", s.batching]].map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
    $("#slotsTbl tbody").innerHTML = s.models.map(m => `<tr><td>${esc(m.id)}</td><td><span class="tag ${m.kind}">${m.kind}</span></td><td>${m.busy}</td><td>${m.queued}</td></tr>`).join("");
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
renderConvs(); renderMsgs(); loadKinds();
const h = location.hash.slice(1); if (h) $(`#tabs button[data-tab="${h}"]`)?.click();
setInterval(() => { if ($("#tab-status").classList.contains("active")) loadStatus(); }, 5000);
