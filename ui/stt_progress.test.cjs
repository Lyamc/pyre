const assert = require("assert");
const { sttStagePctRaw, advanceSttBar } = require("./stt_progress.js");

const recorded = [
  { stage: "diarizing" },
  { stage: "transcribing" },
  { stage: "transcribing", current: 0, total: 902, fraction: 0 },
  { stage: "transcribing", current: 902, total: 902, fraction: 1 },
  { stage: "rendering" },
  { stage: "done" },
];

let prev = null;
const pcts = [];
for (const ev of recorded) {
  prev = advanceSttBar(prev, ev);
  pcts.push(prev);
}

assert.deepEqual(pcts, [62.5, 75, 75, 90, 93, 100]);
for (let i = 1; i < pcts.length; i++) {
  assert.ok(pcts[i] >= pcts[i - 1], `decreased at ${i}: ${pcts}`);
}

assert.equal(sttStagePctRaw({ stage: "transcribing", fraction: 0 }), 75);
assert.notEqual(sttStagePctRaw({ stage: "transcribing", fraction: 0 }), 30);

const climb = [];
prev = null;
for (let i = 0; i <= 10; i++) {
  prev = advanceSttBar(prev, { stage: "transcribing", fraction: i / 10, current: i, total: 10 });
  climb.push(prev);
}
for (let i = 1; i < climb.length; i++) assert.ok(climb[i] >= climb[i - 1]);
assert.equal(climb[0], 75);
assert.equal(climb[climb.length - 1], 90);

assert.equal(advanceSttBar(80, { stage: "transcribing", current: 0, total: 902, fraction: 0 }), 80);
assert.equal(advanceSttBar(null, { stage: "queued" }), 0);
assert.equal(advanceSttBar(62.5, { stage: "segment", text: "hello" }), 62.5);
assert.equal(sttStagePctRaw({ stage: "rendering" }), 93);
assert.equal(sttStagePctRaw({ stage: "diarizing" }), 62.5);

console.log("stt progress sequences ok", pcts.join(" -> "));
