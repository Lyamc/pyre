// pyre Cabin: one-click image workflows. Talks to /api/cabin/* (on-demand models, styles, vectorize),
// /v1/images/generations|edits (SD 1.5 txt2img / img2img / inpaint) and /v1/images/tags (Kagami tagger).
(() => {
"use strict";
const $c = s => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k.startsWith("on")) e[k] = v; else if (k === "html") e.innerHTML = v; else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v); } for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k)); return e; };
const hdr = (x = {}) => (typeof headers === "function" ? headers(x) : x);
async function api(path, body) {
  const r = await fetch(path, body === undefined ? { headers: hdr() } : { method: "POST", headers: hdr({ "content-type": "application/json" }), body: JSON.stringify(body) });
  const t = await r.text(); let j; try { j = JSON.parse(t); } catch { j = { error: { message: t } }; }
  if (!r.ok) throw new Error(j.error?.message || j.error || t || r.statusText);
  return j;
}
// ---------- state ----------
let WF = {}, STYLES = [], gallery = [];   // gallery: [{url, name, from}]
const cur = { wf: null };
function addResult(url, name, from) { gallery.unshift({ url, name, from, t: Date.now() }); if (gallery.length > 40) gallery.pop(); renderGallery(); }
// ---------- image helpers ----------
const loadImg = src => new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error("could not load image")); i.src = src; });
const fileToUrl = f => new Promise(ok => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(f); });
/** fit to max side (multiple of 8), returns PNG data URL */
async function prep(src, max = 768, min = 256) {
  const i = await loadImg(src); let w = i.naturalWidth, h = i.naturalHeight; const s = Math.min(1, max / Math.max(w, h)); w = Math.max(min, Math.round(w * s / 8) * 8); h = Math.max(min, Math.round(h * s / 8) * 8);
  const c = el("canvas", { width: w, height: h }); c.getContext("2d").drawImage(i, 0, 0, w, h); return c.toDataURL("image/png");
}
const b64 = url => url.split("base64,")[1];
function styled(prompt, neg, sid) { const s = STYLES.find(x => x.id === sid) || STYLES[0] || { positive: "{prompt}", negative: "" }; return { prompt: s.positive.replace("{prompt}", prompt || "").replace(/^, /, ""), negative: [neg, s.negative].filter(Boolean).join(", ") }; }
// ---------- shared form widgets ----------
function field(label, input, hint) { return el("label", { class: "cf" }, el("span", {}, label), input, hint ? el("small", {}, hint) : null); }
function num(v, min, max, step = 1) { return el("input", { type: "number", value: v, min, max, step }); }
function range(v, min, max, step, fmt = x => x) { const i = el("input", { type: "range", min, max, step }); i.value = v; /* value last: set before min/step it snaps to the default step grid (1 became 1.3 with min 0.3) */ const o = el("output", {}, fmt(v)); i.oninput = () => o.textContent = fmt(i.value); const w = el("span", { class: "rng" }, i, o); w.input = i; return w; }
function sel(opts, v) { const s = el("select", {}); for (const o of opts) { const [val, lab] = Array.isArray(o) ? o : [o, o]; s.append(el("option", { value: val, selected: val === v }, lab)); } return s; }
function stylePicker(def = "none") {
  const box = el("div", { class: "styles" }); box.value = def;
  const draw = () => { box.innerHTML = ""; for (const s of STYLES) { const b = el("button", { type: "button", class: "sty" + (s.id === box.value ? " on" : ""), title: s.positive.replace("{prompt}", "…"), onclick: () => { box.value = s.id; draw(); box.onchange?.(); } }, el("i", { style: `background:${s.swatch || "#888"}` }), s.name); if (s.thumb) b.prepend(el("img", { src: s.thumb })); box.append(b); } };
  draw(); return box;
}
/** image source: upload | from results | sketch canvas */
function imageSource(opts = {}) {
  const wrap = el("div", { class: "isrc" }); wrap.url = null;
  const prev = el("img", { class: "prev hidden" });
  const set = u => { wrap.url = u; prev.src = u || ""; prev.classList.toggle("hidden", !u); wrap.onchange?.(u); };
  wrap.set = set;
  const file = el("input", { type: "file", accept: "image/*", onchange: async () => { const f = file.files[0]; if (f) set(await fileToUrl(f)); } });
  const pick = el("button", { type: "button", onclick: () => pickFromGallery(set) }, "From results");
  const row = el("div", { class: "row" }, file, pick);
  let sk = null;
  if (opts.sketch) {
    sk = sketchCanvas(512, 512);
    const tog = el("button", { type: "button", onclick: () => { sk.classList.toggle("hidden"); if (!sk.classList.contains("hidden")) sk.useAsSource = u => set(u); } }, "✎ Draw");
    row.append(tog);
    sk.classList.add("hidden");
    sk.onupdate = u => set(u);
  }
  wrap.append(row, prev); if (sk) wrap.append(sk);
  return wrap;
}
function pickFromGallery(cb) {
  if (!gallery.length) return alert("no results yet");
  const d = el("div", { class: "cpick" }, el("div", { class: "cpick-in" }, el("b", {}, "Pick a result"), el("div", { class: "cgrid" }, gallery.map(g => el("img", { src: g.url, title: g.name, onclick: () => { d.remove(); cb(g.url); } }))), el("button", { onclick: () => d.remove() }, "Cancel")));
  document.body.append(d);
}
/** simple paint app: brush, colors, eraser, undo, clear, size */
function sketchCanvas(w, h) {
  const c = el("canvas", { width: w, height: h, class: "sketch" }); const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, w, h); g.lineCap = g.lineJoin = "round";
  const undo = []; const snap = () => { undo.push(g.getImageData(0, 0, w, h)); if (undo.length > 40) undo.shift(); };
  const color = el("input", { type: "color", value: "#222222" }); const size = range(8, 1, 64, 1); let erase = false;
  const pal = el("span", { class: "pal" }, ["#000000", "#ffffff", "#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5", "#8e24aa", "#6d4c41", "#f8bbd0", "#90caf9", "#a5d6a7"].map(x => el("i", { style: `background:${x}`, onclick: () => { color.value = x; erase = false; er.classList.remove("on"); } })));
  const er = el("button", { type: "button", onclick: () => { erase = !erase; er.classList.toggle("on", erase); } }, "Eraser");
  const emit = () => box.onupdate?.(c.toDataURL("image/png"));
  const bar = el("div", { class: "row skbar" }, color, pal, field("size", size), er,
    el("button", { type: "button", onclick: () => { if (undo.length) { g.putImageData(undo.pop(), 0, 0); emit(); } } }, "Undo"),
    el("button", { type: "button", onclick: () => { snap(); g.fillStyle = "#fff"; g.fillRect(0, 0, w, h); emit(); } }, "Clear"),
    el("button", { type: "button", onclick: () => pickFromGallery(async u => { snap(); const i = await loadImg(u); g.drawImage(i, 0, 0, w, h); emit(); }) }, "Trace over result"));
  let down = false, lx = 0, ly = 0;
  const pos = e => { const r = c.getBoundingClientRect(); return [(e.clientX - r.left) * w / r.width, (e.clientY - r.top) * h / r.height]; };
  c.onpointerdown = e => { down = true; snap(); [lx, ly] = pos(e); c.setPointerCapture(e.pointerId); c.onpointermove(e); };
  c.onpointermove = e => { if (!down) return; const [x, y] = pos(e); g.strokeStyle = erase ? "#fff" : color.value; g.lineWidth = +size.input.value; g.beginPath(); g.moveTo(lx, ly); g.lineTo(x, y); g.stroke(); lx = x; ly = y; };
  c.onpointerup = () => { if (down) { down = false; emit(); } };
  const box = el("div", { class: "skwrap" }, bar, c); box.canvas = c; return box;
}
/** mask painter over an image: brush, eraser, invert, clear; mask() -> PNG (white = repaint) */
function maskEditor() {
  const box = el("div", { class: "mwrap hidden" }); const base = el("canvas"); const m = el("canvas", { class: "mlayer" }); const stack = el("div", { class: "mstack" }, base, m);
  const gm = m.getContext("2d"); let W = 0, H = 0; const undo = []; let erase = false;
  const size = range(32, 4, 160, 2);
  const er = el("button", { type: "button", onclick: () => { erase = !erase; er.classList.toggle("on", erase); } }, "Eraser");
  const bar = el("div", { class: "row skbar" }, field("brush", size), er,
    el("button", { type: "button", onclick: () => { if (undo.length) gm.putImageData(undo.pop(), 0, 0); } }, "Undo"),
    el("button", { type: "button", onclick: () => { snap(); const d = gm.getImageData(0, 0, W, H); for (let i = 3; i < d.data.length; i += 4) { const on = d.data[i] > 0; d.data[i - 3] = 255; d.data[i - 2] = 40; d.data[i - 1] = 90; d.data[i] = on ? 0 : 150; } gm.putImageData(d, 0, 0); } }, "Invert"),
    el("button", { type: "button", onclick: () => { snap(); gm.clearRect(0, 0, W, H); } }, "Clear"),
    el("button", { type: "button", onclick: () => { snap(); gm.fillStyle = "rgba(255,40,90,.6)"; gm.fillRect(0, 0, W, H); } }, "All"));
  const snap = () => { undo.push(gm.getImageData(0, 0, W, H)); if (undo.length > 30) undo.shift(); };
  box.append(bar, stack);
  box.setImage = async url => { const i = await loadImg(url); W = i.naturalWidth; H = i.naturalHeight; for (const c of [base, m]) { c.width = W; c.height = H; } base.getContext("2d").drawImage(i, 0, 0); gm.clearRect(0, 0, W, H); undo.length = 0; box.classList.remove("hidden"); };
  box.setMaskRect = (x, y, w, h) => { gm.fillStyle = "rgba(255,40,90,.6)"; gm.fillRect(x, y, w, h); };
  let down = false, lx = 0, ly = 0;
  const pos = e => { const r = m.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]; };
  m.onpointerdown = e => { down = true; snap(); [lx, ly] = pos(e); m.setPointerCapture(e.pointerId); m.onpointermove(e); };
  m.onpointermove = e => { if (!down) return; const [x, y] = pos(e); gm.globalCompositeOperation = erase ? "destination-out" : "source-over"; gm.strokeStyle = "rgba(255,40,90,1)"; gm.lineCap = gm.lineJoin = "round"; gm.lineWidth = +size.input.value * W / 512; gm.beginPath(); gm.moveTo(lx, ly); gm.lineTo(x, y); gm.stroke(); gm.globalCompositeOperation = "source-over"; lx = x; ly = y; };
  m.onpointerup = () => down = false;
  box.mask = () => { const o = el("canvas", { width: W, height: H }); const g = o.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, W, H); const d = gm.getImageData(0, 0, W, H); const od = g.getImageData(0, 0, W, H); let any = 0; for (let i = 3; i < d.data.length; i += 4) { if (d.data[i] > 10) { od.data[i - 3] = od.data[i - 2] = od.data[i - 1] = 255; any++; } } g.putImageData(od, 0, 0); return any ? o.toDataURL("image/png") : null; };
  return box;
}
// ---------- result panel ----------
function resultBox() {
  const box = el("div", { class: "cres" }); box.show = (url, name, meta, kind = "image") => {
    box.innerHTML = "";
    const view = kind === "svg" ? el("div", { class: "svgview", html: meta.svg }) : el("img", { src: url });
    const dl = el("a", { href: url, download: name, class: "btnlink" }, "Download");
    const send = sel([["", "Send to…"], ...Object.entries(WF).filter(([k, v]) => v.ready && TAKES_IMAGE.includes(k)).map(([k]) => [k, CARDS.find(c => c.id === k)?.title || k])], "");
    send.onchange = () => { if (send.value) openWorkflow(send.value, url); };
    box.append(view, el("div", { class: "row" }, dl, kind === "image" ? send : null, el("small", {}, meta.info || "")));
    if (kind === "image") addResult(url, name, cur.wf);
  };
  box.busy = msg => { box.innerHTML = ""; box.append(el("div", { class: "cbusy" }, el("span", { class: "spin" }), msg)); };
  box.fail = e => { box.innerHTML = ""; box.append(el("pre", { class: "err" }, "error: " + (e.message || e))); };
  return box;
}
// ---------- generation calls ----------
async function ensureImage(res) { if (!loadedImage()) res.busy("loading Stable Diffusion 1.5 on the GPU (on demand, ~20-40 s; the LLM's expert cache shrinks to make room)…"); const j = await api("/api/cabin/ensure", { what: "image" }); markLoaded("image"); return j; }
let loadedKinds = new Set(); const loadedImage = () => loadedKinds.has("image"); const markLoaded = k => loadedKinds.add(k);
// ⚡ fast mode: LCM-LoRA (SD 1.5), ~6 steps at guidance 1.5 instead of 20-30 steps
const fastMode = () => localStorage.getItem("cabin.fast") === "1" && !!WF.txt2img?.lcm;
async function sd(body, res, label = "generating") {
  await ensureImage(res); res.busy(label + "…"); const t0 = performance.now();
  if (fastMode() && body.fast !== false) body = Object.assign({}, body, { fast: true, steps: Math.min(body.steps || 6, 6), guidance_scale: 1.5 });
  const path = body.image ? "/v1/images/edits" : "/v1/images/generations";
  const j = await api(path, Object.assign({ model: "stable-diffusion-v1-5/stable-diffusion-v1-5", n: 1 }, body));
  const tm = j.timings?.[0] || {}; return { url: "data:image/png;base64," + j.data[0].b64_json, seed: tm.seed, secs: (performance.now() - t0) / 1000, gen: tm.seconds };
}
const seedBox = () => { const s = num(-1, -1, 4294967295); const lock = el("input", { type: "checkbox" }); const w = el("span", { class: "rng" }, s, el("label", { class: "inl" }, lock, "lock")); w.get = () => +s.value; w.after = v => { if (!lock.checked) s.value = -1; else s.value = v; w.last = v; }; w.lockTo = v => { lock.checked = true; s.value = v; }; return w; };
// ---------- workflow definitions ----------
const TAKES_IMAGE = ["img2img", "restyle", "inpaint", "outpaint", "tagger", "svg", "upscale", "denoise", "rmbg", "layers", "to3d", "clothes", "pose", "animate"];
const CARDS = [
  { id: "txt2img", title: "Text to image", icon: "✦", desc: "Prompt + style → image" },
  { id: "img2img", title: "Image + text", icon: "◐", desc: "Upload or sketch, then guide it with a prompt" },
  { id: "restyle", title: "Restyle image", icon: "🎨", desc: "Ghibli, pixel art, crayon, anime… keep the composition" },
  { id: "inpaint", title: "Inpaint", icon: "▣", desc: "Paint a mask, regenerate just that part" },
  { id: "outpaint", title: "Outpaint", icon: "⤢", desc: "Extend the canvas in any direction" },
  { id: "character", title: "Character creator", icon: "☺", desc: "Build a character, tweak one aspect at a time" },
  { id: "tagger", title: "Tagger", icon: "#", desc: "Danbooru tags with scores (Kagami-24k)" },
  { id: "svg", title: "Image to SVG", icon: "◇", desc: "Vectorize: logo, illustration, pixel art, line art" },
  { id: "upscale", title: "Upscale", icon: "⇱", desc: "2x / 4x super-resolution" },
  { id: "denoise", title: "Denoise / restore", icon: "✧", desc: "Remove noise and JPEG artifacts" },
  { id: "rmbg", title: "Remove background", icon: "◌", desc: "Transparent PNG cut-out" },
  { id: "layers", title: "Separate layers", icon: "≣", desc: "Subject / background layers (zip of PNGs)" },
  { id: "animate", title: "Animate between images", icon: "▶", desc: "Keyframe interpolation → GIF/video" },
  { id: "to3d", title: "2D → 3D model", icon: "⬡", desc: "Image to GLB mesh + viewer" },
  { id: "clothes", title: "Swap clothes", icon: "👕", desc: "Change outfit from text or a garment image" },
  { id: "pose", title: "Reposition character", icon: "🤸", desc: "Change pose with a pose editor" },
];
const BUILD = {};
// --- text to image
BUILD.txt2img = (f, res) => {
  const p = el("textarea", { rows: 3, placeholder: "a cozy cabin in a snowy forest at dusk" }); const n = el("input", { placeholder: "negative (optional)" });
  const st = stylePicker("none"); const size = sel([["512x512", "512×512"], ["512x768", "512×768 portrait"], ["768x512", "768×512 landscape"], ["640x640", "640×640"], ["768x768", "768×768"]], "512x512");
  const steps = num(25, 1, 60), cfg = num(7, 1, 20, 0.5), seed = seedBox();
  f.append(field("Prompt", p), field("Style", st), el("div", { class: "row" }, field("Size", size), field("Steps", steps), field("CFG", cfg), field("Seed", seed)), field("Negative", n));
  return async () => { const s = styled(p.value, n.value, st.value); const r = await sd({ prompt: s.prompt, negative_prompt: s.negative, size: size.value, steps: +steps.value, guidance_scale: +cfg.value, seed: seed.get() }, res); seed.after(r.seed); res.show(r.url, `txt2img_${r.seed}.png`, { info: `seed ${r.seed} · ${r.secs.toFixed(1)} s (${(r.gen || 0).toFixed(1)} s on GPU)` }); };
};
// --- image + text (upload or sketch)
BUILD.img2img = (f, res, input) => {
  const src = imageSource({ sketch: true }); if (input) src.set(input);
  const p = el("textarea", { rows: 2, placeholder: "what it should become" }); const st = stylePicker("none"); const s = range(0.65, 0.1, 1, 0.05);
  const steps = num(30, 1, 60), seed = seedBox();
  f.append(field("Input image (upload, pick a result, or draw)", src), field("Prompt", p), field("Style", st), el("div", { class: "row" }, field("Strength", s, "low = keep more of the input"), field("Steps", steps), field("Seed", seed)));
  return async () => { if (!src.url) throw new Error("choose, pick or draw an input image"); const sp = styled(p.value, "", st.value); const r = await sd({ image: await prep(src.url), prompt: sp.prompt, negative_prompt: sp.negative, strength: +s.input.value, steps: +steps.value, seed: seed.get() }, res); seed.after(r.seed); res.show(r.url, `img2img_${r.seed}.png`, { info: `seed ${r.seed} · ${r.secs.toFixed(1)} s` }); };
};
// --- restyle
BUILD.restyle = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input);
  const st = stylePicker("ghibli"); const p = el("input", { placeholder: "describe the picture (optional; 'Describe' fills it from the tagger)" });
  const s = range(0.55, 0.2, 0.9, 0.05); const seed = seedBox();
  const desc = el("button", { type: "button", onclick: async () => { if (!src.url) return; desc.disabled = true; try { p.value = await describe(src.url); } catch (e) { alert(e.message); } desc.disabled = false; } }, "Describe");
  f.append(field("Image", src), field("Target style", st), field("Content", el("div", { class: "row" }, p, desc)), el("div", { class: "row" }, field("Strength", s, "how far to move toward the style"), field("Seed", seed)));
  return async () => { if (!src.url) throw new Error("choose an image"); const sp = styled(p.value || "the same scene", "", st.value); const r = await sd({ image: await prep(src.url), prompt: sp.prompt, negative_prompt: sp.negative, strength: +s.input.value, steps: 30, seed: seed.get() }, res, "restyling"); seed.after(r.seed); res.show(r.url, `restyle_${st.value}_${r.seed}.png`, { info: `${st.value} · seed ${r.seed} · ${r.secs.toFixed(1)} s` }); };
};
async function describe(url) {
  const j = await tagImage(url, 0.4); const tags = (j.data?.[0]?.tags || j.data?.[0]?.general || []);
  const names = (Array.isArray(tags) ? tags : Object.entries(tags)).map(t => Array.isArray(t) ? t[0] : (t.name || t.tag)).filter(Boolean);
  return names.slice(0, 20).map(x => x.replace(/_/g, " ")).join(", ");
}
async function tagImage(url, thr = 0.35) { await api("/api/cabin/ensure", { what: "tagger" }); return api("/v1/images/tags", { image: await prep(url, 1024, 64), threshold: thr, top_k: 80 }); }
// --- inpaint
BUILD.inpaint = (f, res, input) => {
  const src = imageSource(); const me = maskEditor(); src.onchange = async u => { if (u) me.setImage(await prep(u)); }; if (input) src.set(input);
  const p = el("textarea", { rows: 2, placeholder: "what to paint in the masked area" }); const st = stylePicker("none");
  const s = range(1, 0.3, 1, 0.05), blur = range(8, 0, 32, 1), steps = num(30, 1, 60), seed = seedBox();
  f.append(field("Image", src), field("Mask (paint the area to change)", me), field("Prompt", p), field("Style", st), el("div", { class: "row" }, field("Strength", s, "1 = fully new content"), field("Feather px", blur), field("Steps", steps), field("Seed", seed)));
  const ipm = el("input", { type: "checkbox", checked: true }); f.append(el("label", { class: "inl" }, ipm, " use the SD 1.5 inpainting model (better blending; off = base model + latent blending)"));
  return async () => { if (!src.url) throw new Error("choose an image"); const m = me.mask(); if (!m) throw new Error("paint a mask first"); const sp = styled(p.value, "", st.value); const r = await sd({ image: await prep(src.url), mask: m, mask_blur: +blur.input.value, prompt: sp.prompt, negative_prompt: sp.negative, strength: +s.input.value, steps: +steps.value, seed: seed.get(), inpaint_model: ipm.checked }, res, "inpainting"); seed.after(r.seed); res.show(r.url, `inpaint_${r.seed}.png`, { info: `seed ${r.seed} · ${r.secs.toFixed(1)} s` }); };
};
// --- outpaint
BUILD.outpaint = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input);
  const L = num(0, 0, 512, 8), R = num(128, 0, 512, 8), T = num(0, 0, 512, 8), B = num(0, 0, 512, 8);
  const ar = sel([["", "custom"], ["1:1", "1:1"], ["4:3", "4:3"], ["3:4", "3:4"], ["16:9", "16:9"], ["9:16", "9:16"]], "");
  ar.onchange = async () => { if (!ar.value || !src.url) return; const i = await loadImg(await prep(src.url)); const [a, b] = ar.value.split(":").map(Number); let w = i.naturalWidth, h = i.naturalHeight; if (w / h < a / b) { const nw = Math.round(h * a / b / 8) * 8; L.value = R.value = Math.round((nw - w) / 16) * 8; T.value = B.value = 0; } else { const nh = Math.round(w * b / a / 8) * 8; T.value = B.value = Math.round((nh - h) / 16) * 8; L.value = R.value = 0; } };
  const p = el("textarea", { rows: 2, placeholder: "describe the whole extended scene" }); const st = stylePicker("none"); const seed = seedBox();
  f.append(field("Image", src), el("div", { class: "row" }, field("Left", L), field("Right", R), field("Top", T), field("Bottom", B), field("Aspect", ar)), field("Prompt", p), field("Style", st), field("Seed", seed));
  return async () => {
    if (!src.url) throw new Error("choose an image"); const i = await loadImg(await prep(src.url, 640));
    let [l, r, t, b] = [L, R, T, B].map(x => Math.round(+x.value / 8) * 8); const w = i.naturalWidth + l + r, h = i.naturalHeight + t + b;
    if (w > 1024 || h > 1024) throw new Error(`result would be ${w}x${h}; max 1024 per side (extend less)`);
    const c = el("canvas", { width: w, height: h }); const g = c.getContext("2d");
    // fill: stretched + blurred edges so the new area starts with plausible colors
    g.filter = "blur(24px)"; g.drawImage(i, 0, 0, w, h); g.filter = "none"; g.drawImage(i, l, t);
    const mc = el("canvas", { width: w, height: h }); const mg = mc.getContext("2d"); mg.fillStyle = "#fff"; mg.fillRect(0, 0, w, h); const ov = 16; mg.fillStyle = "#000"; mg.fillRect(l + (l ? ov : 0), t + (t ? ov : 0), i.naturalWidth - (l ? ov : 0) - (r ? ov : 0), i.naturalHeight - (t ? ov : 0) - (b ? ov : 0));
    const sp = styled(p.value, "", st.value);
    const rr = await sd({ image: c.toDataURL("image/png"), mask: mc.toDataURL("image/png"), mask_blur: 12, prompt: sp.prompt, negative_prompt: sp.negative, strength: 1, steps: 30, seed: seed.get() }, res, "outpainting"); seed.after(rr.seed);
    res.show(rr.url, `outpaint_${rr.seed}.png`, { info: `${w}×${h} · seed ${rr.seed} · ${rr.secs.toFixed(1)} s` });
  };
};
// --- tagger
BUILD.tagger = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input); const thr = range(0.35, 0.05, 0.95, 0.05);
  f.append(field("Image", src), field("Threshold", thr));
  return async () => {
    if (!src.url) throw new Error("choose an image"); res.busy("loading the tagger (CPU) and tagging…"); const t0 = performance.now();
    const j = await tagImage(src.url, +thr.input.value); const d = j.data[0];
    const rows = []; const push = (arr, cat) => { for (const t of arr || []) { const [n, s] = Array.isArray(t) ? t : [t.name || t.tag, t.score ?? t.prob]; rows.push({ n, s: +s, cat }); } };
    if (Array.isArray(d.tags)) push(d.tags, "tag"); else { push(d.rating, "rating"); push(d.character, "character"); push(d.general, "general"); if (d.tags && !Array.isArray(d.tags)) push(Object.entries(d.tags), "tag"); }
    rows.sort((a, b) => b.s - a.s);
    res.innerHTML = ""; const prompt = rows.filter(r => r.cat !== "rating").map(r => r.n.replace(/_/g, " ")).join(", ");
    res.append(el("div", { class: "row" }, el("button", { type: "button", onclick: () => navigator.clipboard.writeText(prompt) }, "Copy as prompt"), el("button", { type: "button", onclick: () => openWorkflow("txt2img", null, prompt) }, "Use in Text to image"), el("small", {}, `${rows.length} tags · ${((performance.now() - t0) / 1000).toFixed(1)} s`)),
      el("div", { class: "tags" }, rows.map(r => el("div", { class: "tagrow " + r.cat }, el("span", {}, r.n), el("b", { style: `width:${Math.round(r.s * 100)}%` }), el("small", {}, r.s.toFixed(2))))));
    if (!rows.length) res.append(el("pre", {}, JSON.stringify(d, null, 1)));
  };
};
// --- upscale / denoise (Real-ESRGAN compact)
function esrBuilder(def) {
  return (f, res, input) => {
    const src = imageSource(); if (input) src.set(input);
    const scale = sel([["2", "2x"], ["4", "4x"], ["1", "1x (restore only)"]], def.scale), model = sel([["general", "General / photo"], ["anime", "Anime / illustration"]], "general"), dn = range(def.dn, 0, 1, 0.05, v => (+v).toFixed(2));
    f.append(field("Image", src), el("div", { class: "row" }, field("Scale", scale), field("Model", model), field("Denoise strength", dn, "general model only: 0 keeps fine texture/grain, 1 removes the most noise")));
    return async () => {
      if (!src.url) throw new Error("choose an image"); res.busy("upscaling on the GPU…"); const t0 = performance.now();
      const j = await api("/api/cabin/upscale", { image: await prep(src.url, 1536, 16), scale: +scale.value, model: model.value, denoise: +dn.input.value });
      res.show(j.image, `${def.name}_${j.width}x${j.height}.png`, { info: `${j.model}, denoise ${j.denoise} · ${j.width}×${j.height} · ${((performance.now() - t0) / 1000).toFixed(1)} s` });
    };
  };
}
BUILD.upscale = esrBuilder({ scale: "2", dn: 0.5, name: "upscaled" });
BUILD.denoise = esrBuilder({ scale: "1", dn: 1.0, name: "restored" });
// --- remove background / layers (IS-Net)
const checker = "background:repeating-conic-gradient(#888 0 25%,#bbb 0 50%) 0 0/16px 16px";
async function cutout(url, model) { return api("/api/cabin/rmbg", { image: await prep(url, 2048, 16), model }); }
BUILD.rmbg = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input);
  const model = sel([["general", "General / photo (RMBG-1.4)"], ["anime", "Anime / illustration (anime-seg)"]], "general");
  f.append(field("Image", src), field("Model", model));
  return async () => {
    if (!src.url) throw new Error("choose an image"); res.busy("removing the background on the GPU…"); const t0 = performance.now();
    const j = await cutout(src.url, model.value);
    res.show(j.image, "cutout.png", { info: `${j.model} · foreground ${(j.foreground_fraction * 100).toFixed(0)}% · ${((performance.now() - t0) / 1000).toFixed(1)} s` });
    const im = res.querySelector("img"); if (im) im.style.cssText = checker;
    res.append(el("div", { class: "row" }, el("a", { href: j.mask, download: "mask.png", class: "btnlink" }, "Download mask")));
  };
};
function dilateMask(url, px) { return loadImg(url).then(i => { const c = el("canvas", { width: i.naturalWidth, height: i.naturalHeight }); const g = c.getContext("2d"); g.filter = `blur(${px}px)`; g.drawImage(i, 0, 0); const d = g.getImageData(0, 0, c.width, c.height); for (let k = 0; k < d.data.length; k += 4) { const v = d.data[k] > 8 ? 255 : 0; d.data[k] = d.data[k + 1] = d.data[k + 2] = v; d.data[k + 3] = 255; } g.filter = "none"; g.putImageData(d, 0, 0); return c.toDataURL("image/png"); }); }
// OpenPose (COCO-18) skeleton editor: drag joints, presets; rendered like controlnet_aux draw_bodypose.
const OP_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
const OP_COL = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];
const OP_BASE = [[.5, .13], [.5, .21], [.40, .22], [.37, .35], [.36, .47], [.60, .22], [.63, .35], [.64, .47], [.44, .50], [.44, .68], [.44, .86], [.56, .50], [.56, .68], [.56, .86], [.48, .115], [.52, .115], [.46, .125], [.54, .125]];
const OP_PRESETS = {
  standing: {},
  "arms up": { 3: [.36, .12], 4: [.33, .03], 6: [.64, .12], 7: [.67, .03] },
  waving: { 6: [.70, .16], 7: [.72, .05] },
  walking: { 3: [.40, .35], 4: [.45, .46], 6: [.60, .35], 7: [.55, .46], 9: [.41, .68], 10: [.36, .85], 12: [.59, .67], 13: [.63, .84] },
  "hands on hips": { 3: [.32, .33], 4: [.42, .47], 6: [.68, .33], 7: [.58, .47] },
  sitting: { 8: [.44, .55], 11: [.56, .55], 9: [.38, .62], 12: [.62, .62], 10: [.40, .82], 13: [.60, .82], 3: [.36, .37], 4: [.40, .52], 6: [.64, .37], 7: [.60, .52] },
};
function poseEditor(W = 512, H = 768) {
  let pts = OP_BASE.map(p => [...p]); const c = el("canvas", { width: W, height: H, class: "pose" }); c.style.cssText = "width:256px;height:384px;background:#000;cursor:crosshair;touch-action:none";
  const draw = () => { const g = c.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
    OP_LIMBS.forEach(([a, b], i) => { const [x1, y1] = [pts[a][0] * W, pts[a][1] * H], [x2, y2] = [pts[b][0] * W, pts[b][1] * H]; const L = Math.hypot(x2 - x1, y2 - y1); g.save(); g.translate((x1 + x2) / 2, (y1 + y2) / 2); g.rotate(Math.atan2(y2 - y1, x2 - x1)); g.globalAlpha = 0.6; g.fillStyle = `rgb(${OP_COL[i]})`; g.beginPath(); g.ellipse(0, 0, L / 2, 4, 0, 0, Math.PI * 2); g.fill(); g.restore(); });
    pts.forEach(([x, y], i) => { g.fillStyle = `rgb(${OP_COL[i]})`; g.beginPath(); g.arc(x * W, y * H, 4, 0, Math.PI * 2); g.fill(); }); };
  let drag = -1; const at = e => { const r = c.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
  c.onpointerdown = e => { const [x, y] = at(e); let best = -1, bd = 0.03; pts.forEach((p, i) => { const d = Math.hypot((p[0] - x) * W / H, p[1] - y); if (d < bd) { bd = d; best = i; } }); drag = best; if (drag >= 0) c.setPointerCapture(e.pointerId); };
  c.onpointermove = e => { if (drag < 0) return; const [x, y] = at(e); pts[drag] = [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))]; draw(); };
  c.onpointerup = () => { drag = -1; };
  const preset = sel(Object.keys(OP_PRESETS).map(k => [k, k]), "standing"); preset.onchange = () => { pts = OP_BASE.map((p, i) => [...(OP_PRESETS[preset.value][i] || p)]); draw(); };
  const flip = el("button", { type: "button", onclick: () => { pts = pts.map(([x, y]) => [1 - x, y]); const sw = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13], [14, 15], [16, 17]]; for (const [a, b] of sw) [pts[a], pts[b]] = [pts[b], pts[a]]; draw(); } }, "Mirror");
  const w = el("div", {}, el("div", { class: "row" }, preset, flip, el("small", {}, "drag the dots to pose")), c); draw();
  w.image = () => c.toDataURL("image/png"); w.set = name => { preset.value = name; preset.onchange(); }; return w;
}
BUILD.pose = (f, res, input) => {
  const pe = poseEditor(); const src = imageSource(); if (input) src.set(input);
  const p = el("textarea", { rows: 2, placeholder: "who / what, e.g. young woman adventurer, red hair, leather jacket" }); const st = stylePicker("anime");
  const scale = range(1, 0.3, 1.5, 0.05), keep = range(1, 0.5, 1, 0.05), steps = num(25, 1, 60), seed = seedBox();
  f.append(el("div", { class: "row" }, field("Pose", pe), field("Character image (optional: keeps colours/look)", src)), field("Prompt", p), field("Style", st), el("div", { class: "row" }, field("Pose strength", scale), field("Change amount", keep, "1 = follow the pose (look from the prompt); lower keeps more of the character image but the pose follows less"), field("Steps", steps), field("Seed", seed)));
  return async () => {
    if (!p.value.trim()) throw new Error("describe the character"); const sp = styled(p.value + ", full body", "lowres, bad anatomy, extra limbs, deformed", st.value);
    const body = { prompt: sp.prompt, negative_prompt: sp.negative, size: "512x768", steps: +steps.value, seed: seed.get(), control_image: pe.image(), control_scale: +scale.input.value, fast: false };
    if (src.url && +keep.input.value < 0.99) Object.assign(body, { image: await prep(src.url), strength: +keep.input.value });
    const r = await sd(body, res, "posing"); seed.after(r.seed);
    res.show(r.url, `pose_${r.seed}.png`, { info: `seed ${r.seed} · ${r.secs.toFixed(1)} s` });
    res.append(el("div", { class: "side" }, el("figure", {}, el("img", { src: body.control_image }), el("figcaption", {}, "pose"))));
  };
};
BUILD.animate = (f, res, input) => {
  const a = imageSource(), b = imageSource(); if (input) a.set(input);
  const n = num(7, 1, 31), fps = num(12, 1, 60); const pp = el("input", { type: "checkbox", checked: true });
  f.append(el("div", { class: "row" }, field("Start image", a), field("End image", b)), el("div", { class: "row" }, field("In-between frames", n), field("FPS", fps), el("label", { class: "inl" }, pp, " ping-pong loop")));
  return async () => {
    if (!a.url || !b.url) throw new Error("choose both images"); res.busy("interpolating frames on the GPU…");
    const j = await api("/api/cabin/animate", { a: await prep(a.url), b: await prep(b.url), frames: +n.value, fps: +fps.value, pingpong: pp.checked });
    res.show(j.gif, "animation.gif", { info: `${j.model} · ${j.frames.length} frames ${j.width}x${j.height} @ ${j.fps} fps · ${j.seconds.toFixed(1)} s` });
    const z = await api("/api/cabin/pack", { files: j.frames.map((u, i) => ({ name: `frame_${String(i).padStart(3, "0")}.png`, image: u })) });
    const zurl = URL.createObjectURL(new Blob([Uint8Array.from(atob(z.zip), c => c.charCodeAt(0))], { type: "application/zip" }));
    res.append(el("div", { class: "row" }, el("a", { href: zurl, download: "frames.zip", class: "btnlink" }, `Download PNG frames (${j.frames.length})`)));
  };
};
// Colour pre-fill for masked edits: SD 1.5 at strength 1 keeps drifting back to the surrounding colours
// (a "red hoodie" came out white). Painting the masked area in the asked colour first and denoising from
// there (strength ~0.9) fixes the colour while the shape and shading are still re-drawn.
const COLOURS = { red: [200, 30, 40], crimson: [160, 20, 40], scarlet: [220, 40, 30], pink: [240, 140, 180], orange: [240, 130, 30], yellow: [240, 210, 50], gold: [210, 170, 60], blonde: [235, 205, 130], green: [40, 150, 60], teal: [30, 140, 140], blue: [40, 80, 210], navy: [25, 35, 90], cyan: [60, 200, 230], purple: [120, 50, 170], violet: [140, 80, 200], brown: [110, 70, 40], black: [25, 25, 28], grey: [128, 128, 128], gray: [128, 128, 128], silver: [190, 190, 200], white: [240, 240, 240], beige: [215, 195, 160] };
function colourOf(text) { const m = (text || "").toLowerCase().match(new RegExp("\\b(" + Object.keys(COLOURS).join("|") + ")\\b")); return m ? COLOURS[m[1]] : null; }
async function colourFill(url, maskUrl, rgb, maxY = 1) {
  const [i, m] = await Promise.all([loadImg(url), loadImg(maskUrl)]); const W = i.naturalWidth, H = i.naturalHeight;
  const c = el("canvas", { width: W, height: H }); const g = c.getContext("2d"); g.drawImage(i, 0, 0); const d = g.getImageData(0, 0, W, H);
  const mc = el("canvas", { width: W, height: H }); const mg = mc.getContext("2d"); mg.drawImage(m, 0, 0, W, H); const md = mg.getImageData(0, 0, W, H).data;
  const yMax = maxY * H; for (let k = 0; k < W * H; k++) { const a = md[k * 4] / 255; if (a <= 0.02 || k / W > yMax) continue; const lum = (d.data[k * 4] + d.data[k * 4 + 1] + d.data[k * 4 + 2]) / 765; const shade = 0.6 + 0.6 * lum; for (let ch = 0; ch < 3; ch++) { const v = Math.min(255, rgb[ch] * shade + (Math.random() - 0.5) * 24); d.data[k * 4 + ch] = d.data[k * 4 + ch] * (1 - 0.85 * a) + v * 0.85 * a; } }
  g.putImageData(d, 0, 0); return c.toDataURL("image/png");
}
BUILD.clothes = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input);
  const part = sel([["top", "Top / shirt / jacket"], ["sleeves", "Top incl. sleeves (arms)"], ["bottom", "Bottom (pants / skirt)"], ["dress", "Whole outfit / dress"], ["shoes", "Shoes"], ["hair", "Hair"], ["hat", "Hat"], ["acc", "Accessories / bag / scarf"]], "top");
  const p = el("textarea", { rows: 2, placeholder: "the new garment, e.g. a red hoodie" }); const st = stylePicker("none");
  const grow = range(14, 0, 48, 1), steps = num(30, 1, 60), seed = seedBox();
  f.append(field("Image", src), field("Change", part), field("New look", p), field("Style", st), el("div", { class: "row" }, field("Grow mask px", grow, "room for a different shape"), field("Steps", steps), field("Seed", seed)));
  return async () => {
    if (!src.url) throw new Error("choose an image"); if (!p.value.trim()) throw new Error("describe the new look");
    res.busy("segmenting clothes…"); const img = await prep(src.url);
    const j = await api("/api/cabin/parse", { image: img, parts: [part.value], grow: +grow.input.value });
    if (!j.mask || j.selected_fraction < 0.002) throw new Error(`no ${part.value} found in the image (found: ${Object.keys(j.coverage).join(", ")})`);
    const sp = styled(p.value + ", " + p.value, "", st.value); const rgb = colourOf(p.value);
    // pre-fill only the garment itself (un-grown parse), so the grown margin keeps its own colours
    const core = rgb ? await api("/api/cabin/parse", { image: img, parts: [part.value], grow: 0 }) : null;
    const init = rgb ? await colourFill(img, core.mask || j.mask, rgb, part.value === "hair" ? 0.32 : 1) : img;
    // base model (the 9-channel inpainting UNet copies the old garment's colours) + colour pre-fill
    const r = await sd({ image: init, mask: j.mask, mask_blur: 8, prompt: sp.prompt, negative_prompt: sp.negative, strength: rgb ? 0.9 : 1, steps: +steps.value, guidance_scale: 8.5, seed: seed.get(), inpaint_model: false }, res, "dressing"); seed.after(r.seed);
    res.show(r.url, `clothes_${r.seed}.png`, { info: `${part.value} → ${p.value} · seed ${r.seed} · ${r.secs.toFixed(1)} s` });
    res.append(el("div", { class: "side" }, el("figure", {}, el("img", { src: j.preview }), el("figcaption", {}, "parsing")), el("figure", {}, el("img", { src: j.mask }), el("figcaption", {}, "mask"))));
  };
};
// Bounding box (padded) of a white-on-black mask: a silhouette-shaped hole makes SD paint the subject back in.
function boxMask(url, pad) { return loadImg(url).then(i => { const W = i.naturalWidth, H = i.naturalHeight; const c = el("canvas", { width: W, height: H }); const g = c.getContext("2d"); g.drawImage(i, 0, 0); const d = g.getImageData(0, 0, W, H).data; let x0 = W, y0 = H, x1 = -1, y1 = -1; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (d[(y * W + x) * 4] > 127) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } g.fillStyle = "#000"; g.fillRect(0, 0, W, H); if (x1 >= 0) { g.fillStyle = "#fff"; g.fillRect(Math.max(0, x0 - pad), Math.max(0, y0 - pad), Math.min(W, x1 + pad) - Math.max(0, x0 - pad) + 1, Math.min(H, y1 + pad) - Math.max(0, y0 - pad) + 1); } return c.toDataURL("image/png"); }); }
const NO_PEOPLE = "person, people, girl, boy, man, woman, character, human, figure, face, body, silhouette, clothes";
BUILD.layers = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input);
  const model = sel([["general", "General / photo"], ["anime", "Anime / illustration"]], "general");
  const bgp = el("textarea", { rows: 2 }, "empty background scenery, nothing in the foreground");
  f.append(field("Image", src), el("div", { class: "row" }, field("Cut-out model", model)), field("Background fill prompt", bgp, "the hole left by the subject is inpainted with SD 1.5"));
  const hole = sel([["box", "Box around the subject (removes it reliably)"], ["shape", "Subject outline only (keeps more original background)"]], "box"); f.append(field("Fill area", hole));
  return async () => {
    if (!src.url) throw new Error("choose an image"); const t0 = performance.now();
    res.busy("1/3 cutting out the subject…"); const j = await cutout(src.url, model.value);
    res.busy("2/3 growing the mask…"); const sil = await dilateMask(j.mask, 16); const m = hole.value === "box" ? await boxMask(sil, 8) : sil;
    const base = await prep(src.url, 768, 64);
    const bg = await sd({ prompt: bgp.value, negative_prompt: NO_PEOPLE, image: base, mask: m, mask_blur: 8, steps: 25, response_format: "b64_json" }, res, "3/3 filling the background");
    res.busy("packing layers…");
    const files = [{ name: "1_background.png", image: bg.url }, { name: "2_subject.png", image: j.image }, { name: "subject_mask.png", image: j.mask }, { name: "0_original.png", image: src.url }];
    const z = await api("/api/cabin/pack", { files });
    const zurl = URL.createObjectURL(new Blob([Uint8Array.from(atob(z.zip), c => c.charCodeAt(0))], { type: "application/zip" }));
    res.innerHTML = "";
    res.append(el("div", { class: "side" }, el("figure", {}, el("img", { src: bg.url }), el("figcaption", {}, "background (filled)")), el("figure", {}, el("img", { src: j.image, style: checker }), el("figcaption", {}, "subject (transparent)"))),
      el("div", { class: "row" }, el("a", { href: zurl, download: "layers.zip", class: "btnlink" }, `Download layers.zip (${(z.bytes / 1024).toFixed(0)} KB)`), el("small", {}, `${((performance.now() - t0) / 1000).toFixed(1)} s`)));
    addResult(bg.url, "background.png", "layers"); addResult(j.image, "subject.png", "layers");
  };
};
// --- image to SVG
BUILD.svg = (f, res, input) => {
  const src = imageSource(); if (input) src.set(input);
  const pre = sel([["illustration", "Illustration"], ["logo", "Logo / flat"], ["pixel", "Pixel art"], ["lineart", "Line art (B&W)"], ["photo", "Photo"]], "illustration");
  const col = range(6, 1, 8, 1, v => `${v} bits`), sp = range(4, 0, 64, 1), cor = range(60, 0, 180, 5), lay = range(16, 0, 128, 2), mode = sel([["spline", "Spline"], ["polygon", "Polygon"], ["none", "Pixel exact"]], "spline");
  pre.onchange = () => { const P = { logo: [5, 8, 60, 32, "spline"], pixel: [8, 0, 60, 0, "none"], lineart: [6, 4, 60, 16, "spline"], photo: [8, 10, 180, 48, "spline"], illustration: [6, 4, 60, 16, "spline"] }[pre.value]; [col, sp, cor, lay].forEach((x, i) => { x.input.value = P[i]; x.input.oninput(); }); mode.value = P[4]; };
  f.append(field("Image", src), el("div", { class: "row" }, field("Preset", pre), field("Curve", mode)), el("div", { class: "row" }, field("Color precision", col), field("Speckle filter", sp, "drop blobs smaller than this"), field("Corner threshold", cor), field("Layer difference", lay)));
  return async () => {
    if (!src.url) throw new Error("choose an image"); res.busy("tracing…");
    const j = await api("/api/cabin/vectorize", { image: src.url, preset: pre.value, color_precision: +col.input.value, filter_speckle: +sp.input.value, corner_threshold: +cor.input.value, layer_difference: +lay.input.value, mode: mode.value, binary: pre.value === "lineart" });
    const url = URL.createObjectURL(new Blob([j.svg], { type: "image/svg+xml" }));
    res.innerHTML = ""; res.append(el("div", { class: "side" }, el("figure", {}, el("img", { src: src.url }), el("figcaption", {}, "original")), el("figure", {}, el("img", { src: url }), el("figcaption", {}, `SVG · ${j.paths} paths · ${(j.bytes / 1024).toFixed(1)} KB · ${j.seconds.toFixed(2)} s`))),
      el("div", { class: "row" }, el("a", { href: url, download: "vector.svg", class: "btnlink" }, "Download .svg")));
  };
};
// --- character creator
const CH_FIELDS = [["base", "Base description", "young woman adventurer"], ["body", "Body / height / build", "tall, athletic build"], ["face", "Face", "freckles, sharp jawline"], ["hair", "Hair style + color", "long braided red hair"], ["eyes", "Eyes", "green eyes"], ["top", "Top", "leather jacket over white shirt"], ["bottom", "Bottom", "dark cargo pants"], ["shoes", "Shoes", "brown hiking boots"], ["acc", "Accessories", "silver pendant, backpack"], ["pose", "Pose", "standing, full body"], ["expr", "Expression", "confident smile"], ["bg", "Background", "simple white background"]];
const REGION = { hair: [0.15, 0.0, 0.7, 0.35], face: [0.3, 0.08, 0.4, 0.25], eyes: [0.32, 0.12, 0.36, 0.1], expr: [0.3, 0.08, 0.4, 0.25], top: [0.15, 0.25, 0.7, 0.35], bottom: [0.2, 0.5, 0.6, 0.35], shoes: [0.2, 0.82, 0.6, 0.18], acc: [0.1, 0.2, 0.8, 0.5] };
BUILD.character = (f, res) => {
  const ch = JSON.parse(localStorage.getItem("cabin.char") || "null") || { fields: Object.fromEntries(CH_FIELDS.map(x => [x[0], x[2]])), style: "anime", seed: Math.floor(Math.random() * 4e9), history: [] };
  const inputs = {}; const grid = el("div", { class: "chgrid" });
  for (const [k, lab] of CH_FIELDS) { const i = el("input", { value: ch.fields[k] || "" }); i.dataset.k = k; inputs[k] = i; grid.append(field(lab, i)); }
  const st = stylePicker(ch.style); const seed = num(ch.seed, 0, 4294967295); const steps = num(28, 10, 60);
  const how = sel([["auto", "Auto: region inpaint for hair/face/clothes, else same seed"], ["seed", "Same seed, full regenerate"], ["img2img", "Low-strength img2img (0.45)"], ["mask", "Inpaint my painted mask"]], "auto");
  const me = maskEditor(); const hist = el("div", { class: "hist" });
  const prompt = () => CH_FIELDS.map(([k]) => inputs[k].value.trim()).filter(Boolean).join(", ");
  const save = () => { ch.fields = Object.fromEntries(CH_FIELDS.map(([k]) => [k, inputs[k].value])); ch.style = st.value; ch.seed = +seed.value; try { localStorage.setItem("cabin.char", JSON.stringify(ch)); } catch { ch.history = ch.history.slice(-8); localStorage.setItem("cabin.char", JSON.stringify(ch)); } };
  const drawHist = () => { hist.innerHTML = ""; ch.history.forEach((h, i) => hist.append(el("figure", { class: i === ch.history.length - 1 ? "on" : "" }, el("img", { src: h.url, title: h.change, onclick: () => { res.show(h.url, `character_v${i + 1}.png`, { info: `v${i + 1}: ${h.change}` }); me.setImage(h.url); } }), el("figcaption", {}, `v${i + 1} ${h.change}`)))); };
  const dirty = new Set(); for (const i of Object.values(inputs)) i.oninput = () => dirty.add(i.dataset.k);
  f.append(el("div", { class: "row" }, el("button", { type: "button", onclick: () => { const b = new Blob([JSON.stringify(ch)], { type: "application/json" }); const a = el("a", { href: URL.createObjectURL(b), download: "character.json" }); a.click(); } }, "Save .json"),
    el("label", { class: "btnlink" }, "Load .json", el("input", { type: "file", accept: ".json", class: "hidden", onchange: async e => { const t = await e.target.files[0].text(); localStorage.setItem("cabin.char", t); openWorkflow("character"); } })),
    el("button", { type: "button", onclick: () => { if (ch.history.length > 1) { ch.history.pop(); save(); drawHist(); const h = ch.history.at(-1); res.show(h.url, "character.png", { info: "undo → v" + ch.history.length }); me.setImage(h.url); } } }, "Undo"),
    el("button", { type: "button", onclick: () => { const h = ch.history; if (h.length < 2) return; res.innerHTML = ""; res.append(el("div", { class: "side" }, el("figure", {}, el("img", { src: h.at(-2).url }), el("figcaption", {}, "previous")), el("figure", {}, el("img", { src: h.at(-1).url }), el("figcaption", {}, "current")))); } }, "Compare"),
    el("button", { type: "button", onclick: () => { if (confirm("Start a new character (history is cleared)?")) { localStorage.removeItem("cabin.char"); openWorkflow("character"); } } }, "New")),
    grid, field("Style", st), el("div", { class: "row" }, field("Seed (locked)", seed), field("Steps", steps), field("Edit method", how)), field("Optional mask (for 'Inpaint my painted mask')", me), field("Versions", hist));
  drawHist(); if (ch.history.length) me.setImage(ch.history.at(-1).url);
  return async () => {
    const sp = styled(prompt(), "lowres, bad anatomy, extra limbs, deformed", st.value); const last = ch.history.at(-1); const changed = [...dirty]; let r, change;
    const styleChanged = last && st.value !== ch.style;
    if (!last || how.value === "seed" || styleChanged || changed.includes("base") || changed.includes("pose") || changed.includes("bg") || changed.includes("body") || (how.value === "auto" && !changed.some(k => REGION[k]))) {
      r = await sd({ prompt: sp.prompt, negative_prompt: sp.negative, size: "512x768", steps: +steps.value, guidance_scale: 7, seed: +seed.value }, res, last ? "regenerating with the same seed" : "creating the character"); change = last ? "regen: " + (changed.join(", ") || "same") : "created";
    } else if (how.value === "img2img") {
      r = await sd({ image: last.url, prompt: sp.prompt, negative_prompt: sp.negative, strength: 0.45, steps: +steps.value, seed: +seed.value }, res, "adjusting (img2img 0.45)"); change = "img2img: " + changed.join(", ");
    } else {
      let m = how.value === "mask" ? me.mask() : null;
      if (!m && WF.clothes?.ready) { res.busy("finding " + changed.join(", ") + "…"); const regs = changed.filter(k => REGION[k]); const j = await api("/api/cabin/parse", { image: last.url, parts: regs, grow: regs.includes("hair") ? 24 : 14 }); if (j.mask && j.selected_fraction > 0.003) m = j.mask; }
      if (!m) { const i = await loadImg(last.url); const W = i.naturalWidth, H = i.naturalHeight; const mc = el("canvas", { width: W, height: H }); const g = mc.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, W, H); g.fillStyle = "#fff"; for (const k of changed) { const b = REGION[k]; if (b) g.fillRect(b[0] * W, b[1] * H, b[2] * W, b[3] * H); } m = mc.toDataURL("image/png"); }
      // changed attributes first and repeated (SD 1.5 has no prompt weights; 77-token limit drops the tail)
      const fp = styled([...changed.map(k => inputs[k].value.trim()), ...changed.map(k => inputs[k].value.trim()), ...CH_FIELDS.map(([k]) => k).filter(k => !changed.includes(k)).map(k => inputs[k].value.trim())].filter(Boolean).join(", "), "lowres, bad anatomy, extra limbs, deformed" + (changed.includes("hair") ? ", bald, shaved head, bald patch" : ""), st.value);
      // base model + latent blending: the 9-channel inpainting UNet copies the old colours from the context
      // (red hair stayed red); the base model follows the new description
      const rgb = changed.length === 1 && REGION[changed[0]] ? colourOf(inputs[changed[0]].value) : null; let init = last.url;
      if (rgb) { let core = null; if (WF.clothes?.ready && how.value !== "mask") { const c = await api("/api/cabin/parse", { image: last.url, parts: [changed[0]], grow: 0 }); if (c.mask && c.selected_fraction > 0.003) core = c.mask; } init = await colourFill(last.url, core || m, rgb, changed[0] === "hair" ? 0.32 : 1); }
      r = await sd({ image: init, mask: m, mask_blur: 8, prompt: fp.prompt, negative_prompt: fp.negative, strength: rgb ? 0.9 : 1, steps: +steps.value, guidance_scale: 8.5, seed: +seed.value, inpaint_model: false }, res, "editing " + (changed.join(", ") || "masked area")); change = "inpaint: " + (changed.join(", ") || "mask");
    }
    dirty.clear(); ch.history.push({ url: r.url, change, fields: Object.fromEntries(CH_FIELDS.map(([k]) => [k, inputs[k].value])) }); if (ch.history.length > 12) ch.history.shift(); save(); drawHist(); me.setImage(r.url);
    res.show(r.url, `character_v${ch.history.length}.png`, { info: `v${ch.history.length} · ${change} · seed ${seed.value} · ${r.secs.toFixed(1)} s` });
  };
};
// ---------- shell ----------
function openWorkflow(id, input, prompt) {
  cur.wf = id; const card = CARDS.find(c => c.id === id); const st = WF[id] || {};
  $$c(".ccard").forEach(c => c.classList.toggle("on", c.dataset.id === id));
  const pane = $c("#cabinPane"); pane.innerHTML = "";
  pane.append(el("h3", {}, card.icon + " " + card.title), el("p", { class: "muted" }, card.desc + (st.model ? " — " + st.model : "")));
  if (!st.ready || !BUILD[id]) { pane.append(el("div", { class: "disabled-note" }, `${st.status || "not available"}: ${st.reason || "not implemented yet"}`)); return; }
  const form = el("div", { class: "cform" }); const res = resultBox(); const run = BUILD[id](form, res, input);
  if (prompt) { const t = form.querySelector("textarea"); if (t) t.value = prompt; }
  const go = el("button", { class: "primary", type: "button", onclick: async () => { go.disabled = true; try { await run(); } catch (e) { res.fail(e); } go.disabled = false; refreshStatus(); } }, "Run");
  pane.append(form, el("div", { class: "row" }, go), res);
}
const $$c = s => [...document.querySelectorAll(s)];
function renderCards() {
  const g = $c("#cabinCards"); g.innerHTML = "";
  for (const c of CARDS) { const s = WF[c.id] || {}; g.append(el("button", { class: "ccard" + (s.ready ? "" : " off"), "data-id": c.id, title: s.ready ? (s.model || "") : (s.reason || ""), onclick: () => openWorkflow(c.id) }, el("span", { class: "ci" }, c.icon), el("b", {}, c.title), el("small", {}, s.ready ? c.desc : (s.status || "unavailable")))); }
}
function renderGallery() { const g = $c("#cabinGallery"); if (!g) return; g.innerHTML = ""; for (const x of gallery) g.append(el("img", { src: x.url, title: `${x.from}: ${x.name}`, onclick: () => { const d = el("div", { class: "cpick", onclick: () => d.remove() }, el("img", { class: "big", src: x.url })); document.body.append(d); } })); }
async function refreshStatus() {
  try { const j = await api("/api/cabin/workflows"); WF = j.workflows; loadedKinds = new Set(j.loaded.map(m => m.kind)); const v = j.vram; $c("#cabinStatus").textContent = `GPU free ${v ? v.free_mib + " MiB" : "?"} · on-demand models: ${j.loaded.map(m => m.id.split("/").pop() + " (" + m.backend + ")").join(", ") || "none"} · idle unload ${j.idle_unload_s}s`; } catch (e) { $c("#cabinStatus").textContent = "cabin status: " + e.message; }
}
async function init() {
  if (!$c("#tab-cabin")) return;
  try { STYLES = (await api("/api/cabin/styles")).styles; } catch { STYLES = [{ id: "none", name: "No style", positive: "{prompt}", negative: "" }]; }
  await refreshStatus(); renderCards();
  if (WF.txt2img?.lcm) { const cb = el("input", { type: "checkbox", id: "cabinFast" }); cb.checked = localStorage.getItem("cabin.fast") === "1"; cb.onchange = () => localStorage.setItem("cabin.fast", cb.checked ? "1" : "0"); $c("#cabinCards").before(el("label", { class: "inl", title: "LCM-LoRA: about 6 steps at guidance 1.5 instead of 20-30 steps" }, cb, " ⚡ Fast mode (LCM-LoRA, ~4x faster, slightly softer)")); }
  setInterval(() => { if ($c("#tab-cabin").classList.contains("active")) refreshStatus(); }, 10000);
}
window.cabin = { openWorkflow, refreshStatus };
init();
})();
