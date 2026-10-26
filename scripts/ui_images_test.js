const { chromium } = require("playwright-core");
(async () => {
  const b = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" });
  const p = await b.newPage({ viewport: { width: 1280, height: 900 } }); const errs = []; p.on("pageerror", e => errs.push(e.message));
  await p.goto("http://127.0.0.1:8099/"); await p.click('button[data-tab="images"]');
  await p.waitForFunction(() => !document.querySelector("#imgForm").disabled, null, { timeout: 10000 });
  await p.fill("#imgPrompt", "a cozy cabin in snowy mountains, digital art"); await p.fill("#imgSteps", "20"); await p.fill("#imgSeed", "3");
  const t0 = Date.now(); await p.click("#imgGo");
  await p.waitForSelector("#gallery img", { timeout: 300000 });
  console.log("ui image in", ((Date.now() - t0) / 1000).toFixed(1), "s");
  await p.screenshot({ path: "samples/ui_images.png", fullPage: true });
  console.log("page errors:", errs.length); await b.close();
})();
