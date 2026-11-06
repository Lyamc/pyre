// Audio tab: the original upload/recording must play once, in the source step.
// Results keep transcripts and synthesized replies (STS/TTS), not a second copy
// of the input. Serves ui/ and drives headless Chrome over CDP (no extra deps).
//
//   node scripts/ui_audio_dup_player_test.js
"use strict";
const http = require("http");
const fs = require("fs");
const path = require("path");
const net = require("net");
const { spawn } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const failures = [];
function check(cond, msg) {
  if (cond) console.log("ok:", msg);
  else { failures.push(msg); console.error("FAIL:", msg); }
}

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => resolve(p));
    });
    s.on("error", reject);
  });
}

function startStatic(port) {
  const types = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
  };
  return http.createServer((req, res) => {
    let u = decodeURIComponent((req.url || "/").split("?")[0]);
    if (u === "/") u = "/ui/index.html";
    const file = path.normalize(path.join(ROOT, u));
    if (!file.startsWith(ROOT + path.sep) && file !== ROOT) {
      res.writeHead(403); res.end(); return;
    }
    fs.readFile(file, (err, buf) => {
      if (err) { res.writeHead(404); res.end("not found"); return; }
      res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" });
      res.end(buf);
    });
  }).listen(port, "127.0.0.1");
}

function wavB64() {
  const sr = 8000, n = 1600;
  const data = Buffer.alloc(n * 2);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sr, 24);
  header.writeUInt32LE(sr * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]).toString("base64");
}
const WAV = wavB64();

class Cdp {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.next = 1;
    this.pending = new Map();
    this.opened = new Promise((resolve, reject) => {
      this.ws.addEventListener("open", () => resolve());
      this.ws.addEventListener("error", (e) => reject(e.error || e));
    });
    this.ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) p.reject(new Error(JSON.stringify(msg.error)));
        else p.resolve(msg.result || {});
      }
    });
  }
  async send(method, params) {
    await this.opened;
    const id = this.next++;
    this.ws.send(JSON.stringify({ id, method, params: params || {} }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  close() { try { this.ws.close(); } catch {} }
}

async function newTab(debugPort) {
  let r = await fetch("http://127.0.0.1:" + debugPort + "/json/new?about:blank", { method: "PUT" });
  if (!r.ok) r = await fetch("http://127.0.0.1:" + debugPort + "/json/new?about:blank");
  if (!r.ok) throw new Error("json/new " + r.status);
  return r.json();
}

const SNAP = `(() => {
  const vis = (el) => !!(el && !el.classList.contains("hidden") && !el.hidden);
  const info = (el) => {
    if (!el) return null;
    return {
      id: el.id,
      tag: el.tagName,
      controls: el.hasAttribute("controls") && el.controls === true,
      hidden: !vis(el),
      src: el.currentSrc || "",
      attr: el.getAttribute("src") || "",
    };
  };
  const results = document.querySelector("#sttStepResults");
  const stsCard = document.querySelector("#stsCard");
  const stsResults = stsCard ? [...stsCard.querySelectorAll(".astep")].find(s => /Results/.test(s.textContent || "")) : null;
  const audios = [...document.querySelectorAll("#tab-audio audio")].map(info);
  const srcs = audios.map(a => a.src).filter(Boolean);
  const dup = srcs.filter((s, i) => srcs.indexOf(s) !== i);
  return {
    preview: info(document.querySelector("#sttPreview")),
    sttAudioGone: !document.querySelector("#sttAudio"),
    sttResultAudios: results ? [...results.querySelectorAll("audio")].map(info) : null,
    sttOut: (document.querySelector("#sttOut") || {}).textContent || "",
    sttSpkHidden: document.querySelector("#sttSpk") ? document.querySelector("#sttSpk").classList.contains("hidden") : true,
    sttSpkRows: document.querySelectorAll("#sttSpk .spk-row").length,
    timelineKids: [...(document.querySelector("#sttTimeline") ? document.querySelector("#sttTimeline").children : [])].map(n => n.tagName),
    stsPreview: info(document.querySelector("#stsPreview")),
    stsAudio: info(document.querySelector("#stsAudio")),
    stsTranscript: (document.querySelector("#stsTranscript") || {}).textContent || "",
    ttsAudio: info(document.querySelector("#ttsAudio")),
    stsResultAudios: stsResults ? [...stsResults.querySelectorAll("audio")].map(info) : [],
    dup,
    audios,
  };
})()`;

async function ev(cdp, expr) {
  const r = await cdp.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) {
    const d = r.exceptionDetails;
    throw new Error((d.exception && d.exception.description) || d.text || "eval failed");
  }
  return r.result && r.result.value;
}

async function waitFor(cdp, expr, ms) {
  const t0 = Date.now();
  let last;
  while (Date.now() - t0 < ms) {
    last = await ev(cdp, expr);
    if (last) return last;
    await new Promise(r => setTimeout(r, 50));
  }
  throw new Error("timeout: " + expr + " last=" + JSON.stringify(last));
}

function assignFile(selector, name) {
  return `(() => {
    const bin = Uint8Array.from(atob(${JSON.stringify(WAV)}), c => c.charCodeAt(0));
    const f = new File([bin], ${JSON.stringify(name)}, { type: "audio/wav" });
    const input = document.querySelector(${JSON.stringify(selector)});
    const dt = new DataTransfer();
    dt.items.add(f);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return input.files.length;
  })()`;
}

function dropFile(zoneSel, name) {
  return `(() => {
    const bin = Uint8Array.from(atob(${JSON.stringify(WAV)}), c => c.charCodeAt(0));
    const f = new File([bin], ${JSON.stringify(name)}, { type: "audio/wav" });
    const dt = new DataTransfer();
    dt.items.add(f);
    const zone = document.querySelector(${JSON.stringify(zoneSel)});
    const evn = new Event("drop", { bubbles: true });
    Object.defineProperty(evn, "dataTransfer", { value: dt });
    zone.dispatchEvent(evn);
    return true;
  })()`;
}

(async () => {
  const uiPort = await freePort();
  const debugPort = await freePort();
  const server = startStatic(uiPort);
  const chromeBin = process.env.CHROME_PATH || "/usr/local/bin/google-chrome";
  const userData = fs.mkdtempSync(path.join(require("os").tmpdir(), "pyre-audio-ui-"));
  const chrome = spawn(chromeBin, [
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
    "--remote-debugging-port=" + debugPort,
    "--user-data-dir=" + userData,
    "about:blank",
  ], { stdio: "ignore" });
  let cdp;
  try {
    let tab;
    for (let i = 0; i < 50; i++) {
      try { tab = await newTab(debugPort); break; } catch { await new Promise(r => setTimeout(r, 100)); }
    }
    if (!tab || !tab.webSocketDebuggerUrl) throw new Error("chrome debug port not ready");
    cdp = new Cdp(tab.webSocketDebuggerUrl);
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    const diar = {
      sourceName: "meeting.wav",
      model: "base",
      language: "en",
      duration: 4,
      segments: [
        { start: 0, end: 1.5, text: "Hello from the first speaker", speaker: "SPEAKER_00" },
        { start: 1.5, end: 3.2, text: "And the second speaker replies", speaker: "SPEAKER_01" },
      ],
      names: { SPEAKER_00: "Ada", SPEAKER_01: "Grace" },
      text: "Hello from the first speaker And the second speaker replies",
      summary: "",
      summaryStale: false,
    };
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `
        localStorage.setItem("pyre.audioSecs", JSON.stringify({ stt: true, sts: true, tts: true }));
        localStorage.setItem("pyre.sttTranscript", ${JSON.stringify(JSON.stringify(diar))});
      `,
    });
    await cdp.send("Page.navigate", { url: "http://127.0.0.1:" + uiPort + "/#audio" });
    await waitFor(cdp, `!!document.querySelector("#sttFile") && !!document.querySelector("#sttPreview") && (document.querySelector("#sttOut").textContent || "").includes("Ada")`, 10000);

    let snap = await ev(cdp, SNAP);
    check(snap.sttAudioGone, "results original player (#sttAudio) is not in the document");
    check(snap.sttResultAudios && snap.sttResultAudios.length === 0, "diarized restored transcript renders with no audio element in the STT result card");
    check(snap.sttSpkRows === 2, "diarization speaker rows still render (" + snap.sttSpkRows + ")");
    check(snap.timelineKids.length === 2 && snap.timelineKids.every(t => t === "SPAN"), "diarization timeline is spans, not audio");
    check(/Hello from the first speaker/.test(snap.sttOut), "transcript text is preserved");
    check(snap.preview && snap.preview.hidden, "source player stays hidden until audio is selected");
    check(snap.stsAudio && snap.stsAudio.tag === "AUDIO" && snap.stsAudio.controls && snap.stsAudio.hidden && !snap.stsAudio.src, "STS reply player stays a hidden native <audio controls>");
    check(snap.ttsAudio && snap.ttsAudio.tag === "AUDIO" && snap.ttsAudio.controls && snap.ttsAudio.hidden && !snap.ttsAudio.src, "TTS player stays a hidden native <audio controls>");

    await ev(cdp, assignFile("#sttFile", "upload.wav"));
    await waitFor(cdp, `(() => { const a = document.querySelector("#sttPreview"); return a && !a.classList.contains("hidden") && !!a.currentSrc; })()`, 5000);
    snap = await ev(cdp, SNAP);
    check(snap.preview && !snap.preview.hidden && snap.preview.controls && snap.preview.src.startsWith("blob:"), "uploaded audio keeps its native source player while selected");
    check(snap.sttResultAudios.length === 0, "upload does not echo the original into the STT result card");
    check(snap.dup.length === 0, "no two audio elements share the uploaded blob URL");
    check(snap.sttSpkRows === 2, "speaker transcript remains after upload");

    await ev(cdp, dropFile("#sttDrop", "dropped.wav"));
    await waitFor(cdp, `document.querySelector("#sttFileMeta").textContent.includes("dropped.wav")`, 3000);
    snap = await ev(cdp, SNAP);
    check(snap.preview && !snap.preview.hidden && snap.preview.controls, "dropped file still uses the source player");
    check(snap.sttResultAudios.length === 0, "drop does not echo the original into results");
    check(snap.dup.length === 0, "drop does not duplicate the blob URL");

    const diarBox = documentDiarEnable();
    await ev(cdp, diarBox);
    await ev(cdp, assignFile("#sttFile", "diar-upload.wav"));
    snap = await ev(cdp, SNAP);
    check(snap.sttResultAudios.length === 0 && snap.preview && !snap.preview.hidden, "diarize checkbox + upload still has a single source player");

    await ev(cdp, `document.querySelector("#sttModeRec").click()`);
    await ev(cdp, `document.querySelector("#recBtn").click()`);
    await waitFor(cdp, `document.querySelector("#recBtn").textContent.includes("Stop")`, 4000);
    await new Promise(r => setTimeout(r, 350));
    await ev(cdp, `document.querySelector("#recBtn").click()`);
    await waitFor(cdp, `!document.querySelector("#sttRecAfter").classList.contains("hidden") && !document.querySelector("#sttPreview").classList.contains("hidden")`, 5000);
    snap = await ev(cdp, SNAP);
    check(snap.preview && !snap.preview.hidden && snap.preview.controls && snap.preview.src.startsWith("blob:"), "recording keeps a native source player");
    check(snap.sttResultAudios.length === 0, "recording does not echo the original into the STT result card");
    check(snap.dup.length === 0, "recording blob is not attached twice");
    check(snap.stsAudio.hidden && !snap.stsAudio.src, "STT recording does not fill the STS reply player");

    await ev(cdp, `document.querySelector("#sttRecAgain").click()`);
    await waitFor(cdp, `document.querySelector("#recBtn").textContent.includes("Stop")`, 4000);
    await ev(cdp, `document.querySelector("#recBtn").click()`);
    await waitFor(cdp, `!document.querySelector("#sttPreview").classList.contains("hidden")`, 5000);
    snap = await ev(cdp, SNAP);
    check(snap.sttResultAudios.length === 0 && snap.dup.length === 0, "re-record does not leave an original player in results");

    await ev(cdp, assignFile("#stsFile", "sts-upload.wav"));
    await waitFor(cdp, `(() => { const a = document.querySelector("#stsPreview"); return a && !a.classList.contains("hidden") && !!a.currentSrc; })()`, 3000);
    snap = await ev(cdp, SNAP);
    check(snap.stsPreview && !snap.stsPreview.hidden && snap.stsPreview.controls && snap.stsPreview.src.startsWith("blob:"), "STS upload keeps its source player");
    check(snap.stsAudio.hidden && !snap.stsAudio.src && snap.stsAudio.controls, "STS results player is the reply control and is not given the original");
    check(snap.stsResultAudios.length === 1 && snap.stsResultAudios[0].id === "stsAudio", "STS result card contains only the reply player");
    check(snap.dup.length === 0, "STS original is not the same URL as another player");
    check(snap.preview && !snap.preview.hidden, "STT source player stays while its audio is selected");

    await ev(cdp, dropFile("#stsDrop", "sts-dropped.wav"));
    await waitFor(cdp, `document.querySelector("#stsFileMeta").textContent.includes("sts-dropped.wav") && !document.querySelector("#stsPreview").classList.contains("hidden")`, 3000);
    snap = await ev(cdp, SNAP);
    check(snap.stsPreview && !snap.stsPreview.hidden && snap.stsAudio.hidden && !snap.stsAudio.src, "STS drop plays only in the source step");
    check(snap.sttResultAudios.length === 0 && snap.dup.length === 0, "STS drop does not echo audio into either result card");

    await ev(cdp, `document.querySelector("#stsModeRec").click()`);
    await ev(cdp, `document.querySelector("#stsRecBtn").click()`);
    await waitFor(cdp, `document.querySelector("#stsRecBtn").textContent.includes("Stop")`, 4000);
    await new Promise(r => setTimeout(r, 300));
    await ev(cdp, `document.querySelector("#stsRecBtn").click()`);
    await waitFor(cdp, `!document.querySelector("#stsPreview").classList.contains("hidden") && document.querySelector("#stsRecMeta").textContent.includes("Recording ready")`, 5000);
    snap = await ev(cdp, SNAP);
    check(snap.stsPreview && !snap.stsPreview.hidden && snap.stsAudio.hidden && !snap.stsAudio.src, "STS recording plays in the source step only");
    check(snap.stsResultAudios.length === 1 && snap.stsResultAudios[0].id === "stsAudio" && !snap.stsResultAudios[0].src, "STS record does not copy the take into the reply player");

    await ev(cdp, `window.prompt = () => "Clone From STT"; document.querySelector("#sttCloneVoice").click(); true`);
    await new Promise(r => setTimeout(r, 200));
    snap = await ev(cdp, SNAP);
    check(snap.sttResultAudios.length === 0, "STT clone-voice action does not add an original player to results");
    check(snap.preview && !snap.preview.hidden, "source player remains after clone");

    await ev(cdp, `window.prompt = () => "Clone From STS"; document.querySelector("#stsCloneVoice").click(); true`);
    await new Promise(r => setTimeout(r, 200));
    snap = await ev(cdp, SNAP);
    check(snap.stsAudio.hidden && !snap.stsAudio.src, "STS clone-voice action does not load the original into the reply player");
    check(snap.ttsAudio.controls && snap.ttsAudio.tag === "AUDIO", "TTS native audio element is unchanged");
    check(snap.dup.length === 0, "clone actions do not duplicate a blob URL");
    const toast = await ev(cdp, `(document.querySelector("#pyreToasts") || {}).textContent || ""`);
    check(!/is not defined/.test(toast), "audio handlers do not throw on missing player state");

    const native = await ev(cdp, `[...document.querySelectorAll("#tab-audio audio")].every(a => a.tagName === "AUDIO" && a.hasAttribute("controls"))`);
    check(native === true, "every audio-tab player is a native controls element");
  } finally {
    if (cdp) cdp.close();
    if (chrome && chrome.exitCode == null && !chrome.killed) {
      await new Promise((resolve) => {
        chrome.once("exit", resolve);
        chrome.kill("SIGKILL");
        setTimeout(resolve, 800);
      });
    }
    server.close();
    try { fs.rmSync(userData, { recursive: true, force: true, maxRetries: 3 }); } catch {}
  }
  if (failures.length) {
    console.error(failures.length + " failure(s)");
    process.exit(1);
  }
  console.log("audio duplicate-player UI test passed");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

function documentDiarEnable() {
  return `(() => {
    const box = document.querySelector("#sttDiar");
    box.disabled = false;
    box.checked = true;
    box.dispatchEvent(new Event("change", { bubbles: true }));
    return box.checked;
  })()`;
}
