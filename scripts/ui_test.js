// Headless UI check. Needs playwright-core + Chrome:
//   set NODE_PATH=C:\Temp\pwtest\node_modules && node scripts\ui_test.js http://127.0.0.1:8092 [apikey]
const { chromium } = require('playwright-core');
const base = process.argv[2] || 'http://127.0.0.1:8092';
const key = process.argv[3] || '';
const out = require('path').join(__dirname, '..', 'samples');
require('fs').mkdirSync(out, { recursive: true });
(async () => {
  const br = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  const p = await br.newPage({ viewport: { width: 1400, height: 900 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto(base + '/');
  if (key) { await p.evaluate(k => localStorage.setItem('pyre.apikey', JSON.stringify(k)), key); await p.reload(); }
  await p.waitForFunction(() => document.querySelector('#modelPick option')?.value);
  console.log('models in picker:', await p.$$eval('#modelPick option', o => o.map(x => x.value)));
  await p.fill('#input', 'Give me two short tips for learning Rust.');
  await p.press('#input', 'Enter');
  await p.waitForFunction(() => document.querySelector('.msg.assistant .body')?.textContent.length > 0, null, { timeout: 120000 });
  await p.waitForTimeout(400);
  const partial = await p.$eval('.msg.assistant .body', e => e.textContent.length);
  await p.waitForSelector('#stopBtn.hidden', { state: 'attached', timeout: 120000 });
  const reply = await p.$eval('.msg.assistant .body', e => e.textContent);
  const stats = await p.$eval('#tps', e => e.textContent);
  console.log('streamed chars after 400ms:', partial, '| final chars:', reply.length, '|', stats);
  console.log('reply:', reply.slice(0, 300).replace(/\n/g, ' '));
  await p.screenshot({ path: out + '/ui-chat.png' });
  await p.click('#tabs button[data-tab=models]');
  await p.waitForSelector('#availTbl tbody tr');
  console.log('available models:', await p.$$eval('#availTbl tbody tr', r => r.map(x => x.cells[0].textContent + ' [' + x.cells[2].textContent + ']')));
  await p.screenshot({ path: out + '/ui-models.png', fullPage: true });
  await p.click('#tabs button[data-tab=status]'); await p.waitForTimeout(1500);
  await p.screenshot({ path: out + '/ui-status.png', fullPage: true });
  await p.click('#tabs button[data-tab=audio]'); await p.waitForTimeout(500);
  await p.screenshot({ path: out + '/ui-audio.png' });
  await p.click('#themeBtn'); await p.click('#tabs button[data-tab=chat]'); await p.waitForTimeout(300);
  await p.screenshot({ path: out + '/ui-chat-light.png' });
  await p.setViewportSize({ width: 390, height: 800 }); await p.waitForTimeout(300);
  await p.screenshot({ path: out + '/ui-mobile.png' });
  console.log('page errors:', errs);
  await br.close();
})().catch(e => { console.error(e); process.exit(1); });
