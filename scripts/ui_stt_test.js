const { chromium } = require("playwright-core");
(async () => {
  const b = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" }); const p = await b.newPage({ viewport: { width: 1280, height: 860 } });
  const errs = []; p.on("pageerror", e => errs.push(e.message));
  await p.goto("http://127.0.0.1:8099/"); await p.click('button[data-tab="audio"]');
  await p.waitForFunction(() => !document.querySelector("#sttForm").disabled, null, { timeout: 10000 });
  await p.setInputFiles("#sttFile", "samples/speech.wav"); await p.click("#sttGo");
  await p.waitForFunction(() => { const t = document.querySelector("#sttOut").textContent; return t.length > 5 && t !== "…"; }, null, { timeout: 60000 });
  console.log("stt ui:", JSON.stringify(await p.textContent("#sttOut")));
  await p.screenshot({ path: "samples/ui_audio_stt.png", fullPage: true });
  console.log("page errors:", errs.length); await b.close();
})();

