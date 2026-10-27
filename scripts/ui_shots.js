// Screenshots of the web UI (desktop + phone). NODE_PATH=C:\Temp\pwtest\node_modules node scripts\ui_shots.js http://192.168.0.30:8099
const { chromium } = require('playwright-core');
const base = process.argv[2] || 'http://127.0.0.1:8099';
const out = require('path').join(__dirname, '..', 'samples');
require('fs').mkdirSync(out, { recursive: true });
(async () => {
  const br = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  const errs = [];
  const shot = async (p, n) => { const f = require('path').join(out, n); await p.screenshot({ path: f }); console.log('saved', f); };
  const p = await br.newPage({ viewport: { width: 1400, height: 900 } });
  p.on('pageerror', e => errs.push(e.message));
  await p.goto(base + '/');
  await p.waitForFunction(() => document.querySelector('#modelPick option')?.value);
  if (!process.env.SKIP_CHAT) {
  await p.fill('#input', 'In one sentence: what is a GPU? Also show a tiny Rust hello world.');
  await p.press('#input', 'Enter');
  await p.waitForSelector('#stopBtn:not(.hidden)', { timeout: 30000 });
  await p.waitForTimeout(2500); await shot(p, 'ui-chat-streaming.png');
  await p.waitForSelector('#stopBtn.hidden', { state: 'attached', timeout: 300000 });
  await shot(p, 'ui-chat-done.png');
  const th = await p.$('details.think summary'); if (th) { await th.click(); await shot(p, 'ui-chat-think-open.png'); }
  console.log('think block:', !!th, '| stats:', await p.$eval('#tps', e => e.textContent));
  // stop button
  await p.fill('#input', 'Count from 1 to 200.'); await p.press('#input', 'Enter');
  await p.waitForSelector('#stopBtn:not(.hidden)'); await p.waitForTimeout(1500); await p.click('#stopBtn');
  await p.waitForSelector('#stopBtn.hidden', { state: 'attached' }); console.log('stopped ok'); }
  // error display (simulated server error so it doesn't depend on load)
  await p.route('**/v1/chat/completions', r => r.fulfill({ status: 400, contentType: 'application/json', body: JSON.stringify({ error: { message: 'context length exceeded: 5000 > 4096 (example error)' } }) }));
  await p.fill('#input', 'hi'); await p.press('#input', 'Enter');
  await p.waitForSelector('.err', { timeout: 30000 }); await shot(p, 'ui-chat-error.png'); await p.unroute('**/v1/chat/completions');
  console.log('error shown:', await p.$eval('.err', e => e.textContent));
  for (const t of ['models', 'images', 'audio', 'apps', 'status']) { await p.click(`#tabs button[data-tab="${t}"]`); await p.waitForTimeout(1200); await shot(p, `ui-${t}.png`); }
  // reload: conversations persisted
  await p.click('#tabs button[data-tab="chat"]'); await p.reload(); await p.waitForTimeout(800);
  console.log('conversations after reload:', await p.$$eval('#convList li', l => l.length));
  const m = await br.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  m.on('pageerror', e => errs.push(e.message));
  await m.goto(base + '/'); await m.waitForTimeout(1200); await shot(m, 'ui-phone-chat.png');
  await m.click('#convToggle'); await m.waitForTimeout(300); await shot(m, 'ui-phone-convs.png');
  console.log('page errors:', errs);
  await br.close();
})().catch(e => { console.error(e); process.exit(1); });
