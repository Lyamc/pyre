// media_extract.js: pull the audio track out of a picked media file in the browser, so the
// server only receives audio (16 kHz mono 16-bit WAV, the rate every pyre speech model uses).
// 1. Mediabunny (vendored, MPL-2.0, served at MEDIABUNNY_URL): demuxes MP4/MOV/MKV/WebM/OGG/MP3/...
//    lazily from the Blob (the video track is never decoded or read into memory) and decodes the
//    audio with WebCodecs. WebCodecs needs a secure context (HTTPS or localhost).
// 2. WebAudio decodeAudioData on the whole file (works on plain-http LAN URLs; holds the file in memory).
// 3. Neither works: the caller uploads the original file and the server demuxes it (Symphonia).
const MEDIABUNNY_URL = "/ui/vendor/mediabunny-1.66.0.min.mjs";
const EXTRACT_RATE = 16000;
// decodeAudioData needs the whole file as one ArrayBuffer; past this, skip it.
const WEBAUDIO_MAX_BYTES = 1.5e9;
let mediabunnyMod = null;
function loadMediabunny() {
  if (!mediabunnyMod) mediabunnyMod = import(MEDIABUNNY_URL).catch(e => { mediabunnyMod = null; throw e; });
  return mediabunnyMod;
}
// Files worth extracting: anything with a video track, plus containers/codecs the server cannot
// decode (Opus/WebM) and large uncompressed audio, which shrinks a lot at 16 kHz mono.
function needsAudioExtract(f) {
  if (!f) return false;
  const type = f.type || "", name = f.name || "";
  if (type.startsWith("video/")) return true;
  if (/\.(mp4|mkv|mov|webm|m4v|avi|3gp|ts|mts|m2ts|ogv|wmv|flv)$/i.test(name)) return true;
  if (/opus|webm/i.test(type) || /\.opus$/i.test(name)) return true;
  if (f.size > 8e6 && (/wav|aiff/i.test(type) || /\.(wav|aiff?)$/i.test(name))) return true;
  return false;
}
function extractedName(f) { return (f.name || "media").replace(/\.[^.]+$/, "") + ".wav"; }
class ExtractCancelled extends Error { constructor() { super("cancelled"); this.name = "ExtractCancelled"; } }
async function extractViaMediabunny(f, onProgress, signal) {
  const MB = await loadMediabunny();
  const input = new MB.Input({ formats: MB.ALL_FORMATS, source: new MB.BlobSource(f) });
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track) { const e = new Error((f.name || "This file") + " has no audio track."); e.noAudio = true; throw e; }
    if (!(await track.canDecode())) throw new Error("this browser cannot decode " + (track.codec || "the audio codec") + " with WebCodecs");
    const output = new MB.Output({ format: new MB.WavOutputFormat(), target: new MB.BufferTarget() });
    const conv = await MB.Conversion.init({
      input, output, showWarnings: false,
      video: { discard: true },
      audio: { numberOfChannels: 1, sampleRate: EXTRACT_RATE, codec: "pcm-s16" },
    });
    if (!conv.isValid) throw new Error("cannot convert: " + conv.discardedTracks.map(d => d.reason).join(", "));
    conv.onProgress = p => onProgress && onProgress(p);
    const abort = () => { conv.cancel().catch(() => {}); };
    if (signal) { if (signal.aborted) throw new ExtractCancelled(); signal.addEventListener("abort", abort, { once: true }); }
    try { await conv.execute(); }
    catch (e) { if (signal && signal.aborted) throw new ExtractCancelled(); throw e; }
    finally { if (signal) signal.removeEventListener("abort", abort); }
    const buf = output.target.buffer;
    if (!buf || buf.byteLength <= 44) throw new Error("no audio samples decoded");
    return { blob: new Blob([buf], { type: "audio/wav" }), duration: (buf.byteLength - 44) / (EXTRACT_RATE * 2), method: "Mediabunny" };
  } finally {
    try { input.dispose && input.dispose(); } catch {}
  }
}
async function extractViaWebAudio(f, onProgress, signal) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx || typeof OfflineAudioContext === "undefined") throw new Error("no WebAudio");
  if (f.size > WEBAUDIO_MAX_BYTES) throw new Error("file too large for in-memory decode");
  const ctx = new Ctx();
  let decoded;
  try {
    onProgress && onProgress(0.05);
    const bytes = await f.arrayBuffer();
    if (signal && signal.aborted) throw new ExtractCancelled();
    onProgress && onProgress(0.3);
    decoded = await ctx.decodeAudioData(bytes);
  } finally { try { await ctx.close(); } catch {} }
  if (signal && signal.aborted) throw new ExtractCancelled();
  onProgress && onProgress(0.7);
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * EXTRACT_RATE)), EXTRACT_RATE);
  const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate);
  const ch = mono.getChannelData(0);
  const nch = decoded.numberOfChannels || 1;
  for (let c = 0; c < nch; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < data.length; i++) ch[i] += data[i] / nch;
  }
  const src = offline.createBufferSource();
  src.buffer = mono; src.connect(offline.destination); src.start();
  const rendered = await offline.startRendering();
  onProgress && onProgress(1);
  return { blob: new Blob([encodeWavPcm(rendered.getChannelData(0), EXTRACT_RATE)], { type: "audio/wav" }), duration: decoded.duration, method: "WebAudio" };
}
// -> { blob, name, original, duration, method } or { fallback: true, reason } (send the original file).
// Throws ExtractCancelled, or an Error with .noAudio when the file has no audio track at all.
async function extractAudioTrack(f, onProgress, signal) {
  const reasons = [];
  try {
    const r = await extractViaMediabunny(f, onProgress, signal);
    return Object.assign(r, { name: extractedName(f), original: f.size });
  } catch (e) {
    if (e instanceof ExtractCancelled || e.noAudio) throw e;
    reasons.push("Mediabunny: " + e.message);
  }
  try {
    const r = await extractViaWebAudio(f, onProgress, signal);
    return Object.assign(r, { name: extractedName(f), original: f.size });
  } catch (e) {
    if (e instanceof ExtractCancelled) throw e;
    reasons.push("WebAudio: " + e.message);
  }
  try { console.warn("pyre: browser audio extraction failed; uploading the original file", reasons); } catch {}
  return { fallback: true, reason: reasons.join("; ") };
}
function describeExtract(ex) {
  const saved = Math.max(0, ex.original - ex.blob.size);
  const pct = ex.original ? Math.round(100 * saved / ex.original) : 0;
  const bits = [ex.name, human(ex.original) + " → " + human(ex.blob.size) + " audio (16 kHz mono WAV)"];
  if (saved > 0) bits.push(pct + "% smaller upload");
  if (isFinite(ex.duration) && ex.duration > 0) bits.push(fmtDur(ex.duration));
  bits.push("extracted in the browser via " + ex.method);
  return bits.join(" · ");
}
