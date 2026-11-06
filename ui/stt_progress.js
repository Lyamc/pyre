// Stage bar for one speech job. Transcribe progress lives in [75, 90], after
// diarization at 62.5 and before rendering at 93. A missing fraction uses the
// stage constant. Turn current/total places the bar; a local fraction of 0
// does not pull it back to the bottom of the page.
function sttClamp01(x) {
  if (x == null || Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function sttStagePctRaw(ev) {
  if (!ev || !ev.stage || ev.stage === "segment") return null;
  const bands = {
    queued: [0, 0],
    resolving_model: [4, 4],
    downloading: [8, 24],
    loading_model: [24, 36],
    decoding: [36, 52],
    diarizing: [62.5, 62.5],
    transcribing: [75, 90],
    rendering: [93, 93],
    done: [100, 100],
  };
  const band = bands[ev.stage];
  if (!band) return null;
  const lo = band[0];
  const hi = band[1];
  if (ev.stage === "downloading" && ev.fraction != null) {
    return lo + (hi - lo) * sttClamp01(ev.fraction);
  }
  if (ev.stage === "transcribing") {
    let f = null;
    if (ev.total) f = (ev.current || 0) / ev.total;
    else if (ev.fraction != null) f = ev.fraction;
    if (f == null) return lo;
    return lo + (hi - lo) * sttClamp01(f);
  }
  return lo;
}

function advanceSttBar(prev, ev) {
  const raw = sttStagePctRaw(ev);
  if (raw == null) return prev;
  if (prev == null) return raw;
  return Math.max(prev, raw);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { sttStagePctRaw, advanceSttBar };
}
