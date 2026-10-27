// Web chat against the Qwen3.8-Flash-Next GGUF server: NODE_PATH=C:\Temp\pwtest\node_modules node scripts\ui_qn_shot.js http://127.0.0.1:8131
const { chromium } = require('playwright-core');
const base = process.argv[2] || 'http://127.0.0.1:8131';
const out = require('path').join(__dirname, '..', 'samples');
require('fs').mkdirSync(out, { recursive: true });
(async () => {
  const br = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  const p = await br.newPage({ viewport: { width: 1400, height: 900 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto(base + '/');
  await p.waitForFunction(() => document.querySelector('#modelPick option')?.value);
  await p.fill('#input', 'What is the capital of Canada? One short sentence.');
  await p.press('#input', 'Enter');
  await p.waitForSelector('#stopBtn:not(.hidden)', { timeout: 30000 });
  await p.waitForTimeout(12000); await p.screenshot({ path: out + '/ui-qwen-next-streaming.png' });
  await p.waitForSelector('#stopBtn.hidden', { state: 'attached', timeout: 400000 });
  const th = await p.$('details.think summary'); if (th) await th.click();
  await p.screenshot({ path: out + '/ui-qwen-next-done.png' });
  const last = await p.$$eval('.msg.assistant, .assistant', els => els.length ? els[els.length - 1].innerText : '');
  console.log('think block:', !!th, '| stats:', await p.$eval('#tps', e => e.textContent).catch(() => '?'));
  console.log('assistant:', JSON.stringify(last.slice(0, 600)));
  console.log('page errors:', errs);
  await br.close();
})().catch(e => { console.error(e); process.exit(1); });
