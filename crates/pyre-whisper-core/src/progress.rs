//! Chunk schedule and job progress for long transcriptions.
//!
//! Window planning is pure arithmetic so the redundant-tail bug can be tested
//! without a model. The ledger turns per-call chunk events into one monotonic
//! sample/turn counter. ETA stays unknown for the first minute of transcription
//! and is never reported as zero while a non-trivial fraction of the budget remains.

/// Starts of each analysis window. Every sample of `len` is covered, and each
/// window extends coverage (no tail that the previous window already finished).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPlan {
    pub starts: Vec<usize>,
    pub window: usize,
    pub shift: usize,
}

pub fn plan_windows(len: usize, window: usize, overlap: usize) -> WindowPlan {
    let window = window.max(1);
    let overlap = overlap.min(window.saturating_sub(1));
    let shift = window - overlap;
    if len == 0 {
        return WindowPlan { starts: Vec::new(), window, shift };
    }
    let mut starts = Vec::new();
    let mut start = 0usize;
    loop {
        starts.push(start);
        let end = start.saturating_add(window);
        if end >= len {
            break;
        }
        start = start.saturating_add(shift);
        if start >= len {
            break;
        }
    }
    WindowPlan { starts, window, shift }
}

/// Previous chunk count: `(len - 1) / shift + 1`. This emits an extra window
/// whenever an earlier window already reached `len`.
pub fn legacy_window_count(len: usize, shift: usize) -> usize {
    if len == 0 || shift == 0 {
        return 0;
    }
    len.saturating_sub(1) / shift + 1
}

/// Seconds of work left.
///
/// `None` for the first minute, or when nothing has finished, or when the
/// estimate would display as 0s while more than the last 2% remains.
/// `Some(0.0)` only when the budget is finished or the remainder is that last fraction.
pub fn eta_seconds(elapsed_s: f64, done: u64, total: u64) -> Option<f64> {
    if !(elapsed_s >= 60.0) || done == 0 || total == 0 {
        return None;
    }
    if done >= total {
        return Some(0.0);
    }
    let remain = (total - done) as f64;
    let eta = elapsed_s * remain / (done as f64);
    let last_fraction = remain / (total as f64) <= 0.02;
    if eta < 0.5 && !last_fraction {
        return None;
    }
    if eta < 0.5 && last_fraction {
        return Some(0.0);
    }
    Some(eta)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// One waveform with a stable window plan.
    Chunks,
    /// Speaker turns. Inner window events must not replace the turn budget.
    Turns,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProgressView {
    pub phase: &'static str,
    pub message: String,
    pub completed: u64,
    pub total: Option<u64>,
    pub fraction: Option<f64>,
    pub eta_s: Option<f64>,
}

#[derive(Debug)]
pub struct ProgressLedger {
    phase: &'static str,
    message: String,
    mode: Mode,
    samples_done: u64,
    samples_committed: u64,
    samples_budget: Option<u64>,
    units_done: u64,
    units_total: Option<u64>,
    slice_open: bool,
    slice_len: u64,
}

impl Default for ProgressLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressLedger {
    pub fn new() -> Self {
        Self {
            phase: "transcribing",
            message: "transcribing".into(),
            mode: Mode::Chunks,
            samples_done: 0,
            samples_committed: 0,
            samples_budget: None,
            units_done: 0,
            units_total: None,
            slice_open: false,
            slice_len: 0,
        }
    }

    pub fn set_phase(&mut self, phase: &'static str, message: impl Into<String>) {
        self.phase = phase;
        self.message = message.into();
    }

    /// Switch to a known speaker-turn schedule. Later per-window reports only
    /// advance samples inside the open slice; they cannot shrink `turns`.
    pub fn begin_turns(&mut self, turns: u64, sample_budget: u64) {
        self.mode = Mode::Turns;
        self.phase = "transcribing";
        self.units_done = 0;
        self.units_total = Some(turns);
        self.samples_budget = Some(sample_budget);
        self.samples_done = 0;
        self.samples_committed = 0;
        self.slice_open = false;
        self.slice_len = 0;
        self.message = format!("speaker turn 0/{turns}");
    }

    pub fn begin_slice(&mut self, slice_len: u64) {
        self.slice_open = true;
        self.slice_len = slice_len;
    }

    pub fn finish_slice(&mut self) {
        if self.mode != Mode::Turns || !self.slice_open {
            return;
        }
        self.slice_open = false;
        self.samples_committed = self.samples_committed.saturating_add(self.slice_len);
        if self.samples_committed > self.samples_done {
            self.samples_done = self.samples_committed;
        }
        self.units_done = self.units_done.saturating_add(1);
        if let Some(t) = self.units_total {
            self.units_done = self.units_done.min(t);
        }
    }

    /// `samples_done` / `samples_total` are within the waveform currently being
    /// decoded (a whole file, or one speaker slice). Counts never move backward.
    pub fn observe_chunks(
        &mut self,
        chunks_done: u64,
        chunks_total: u64,
        samples_done: u64,
        samples_total: u64,
    ) {
        match self.mode {
            Mode::Turns => {
                if !self.slice_open {
                    return;
                }
                let add = samples_done.min(self.slice_len);
                let abs = self.samples_committed.saturating_add(add);
                if abs > self.samples_done {
                    self.samples_done = abs;
                }
            }
            Mode::Chunks => {
                if chunks_done > self.units_done {
                    self.units_done = chunks_done;
                }
                let grown = chunks_total.max(self.units_done);
                self.units_total = Some(match self.units_total {
                    Some(t) => t.max(grown),
                    None => grown,
                });
                if samples_done > self.samples_done {
                    self.samples_done = samples_done;
                }
                let budget = samples_total.max(self.samples_done);
                self.samples_budget = Some(match self.samples_budget {
                    Some(b) => b.max(budget),
                    None => budget,
                });
            }
        }
    }

    pub fn view(&self, transcribe_elapsed_s: f64) -> ProgressView {
        let scheduling = self.phase == "transcribing";
        let total = if scheduling { self.units_total } else { None };
        let fraction = if !scheduling {
            None
        } else {
            match self.samples_budget {
                Some(b) if b > 0 => Some((self.samples_done as f64 / b as f64).clamp(0.0, 1.0)),
                _ => match total {
                    Some(t) if t > 0 => Some((self.units_done as f64 / t as f64).clamp(0.0, 1.0)),
                    _ => None,
                },
            }
        };
        let eta_s = if scheduling {
            self.samples_budget
                .and_then(|b| eta_seconds(transcribe_elapsed_s, self.samples_done, b))
        } else {
            None
        };
        let message = if !scheduling {
            self.message.clone()
        } else {
            match (self.mode, self.units_total) {
                (Mode::Turns, Some(t)) => format!("speaker turn {}/{}", self.units_done, t),
                (Mode::Chunks, Some(t)) => format!("chunk {}/{}", self.units_done, t),
                _ => self.message.clone(),
            }
        };
        ProgressView {
            phase: self.phase,
            message,
            completed: if scheduling { self.units_done } else { 0 },
            total,
            fraction,
            eta_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16_000;
    /// Whisper tiny `n_audio_ctx` minus the default task padding, in samples.
    const WINDOW: usize = (1500 - 200) * 160;
    const OVERLAP: usize = 3 * SR;

    #[test]
    fn twelve_second_turn_is_one_window_not_two() {
        let len = 12 * SR;
        let shift = WINDOW - OVERLAP;
        assert!(len > shift && len <= WINDOW);
        assert_eq!(legacy_window_count(len, shift), 2);
        let plan = plan_windows(len, WINDOW, OVERLAP);
        assert_eq!(plan.starts, vec![0]);
    }

    #[test]
    fn windows_cover_without_a_redundant_tail() {
        for len in [1usize, 50, 100, 101, 160, 208, 367, 368, 480, 12 * SR, 30 * SR] {
            let plan = plan_windows(len, WINDOW, OVERLAP);
            let mut covered = 0usize;
            for &start in &plan.starts {
                assert!(start <= covered, "gap before {start} covered {covered} len {len}");
                let end = start.saturating_add(plan.window).min(len);
                assert!(end > covered, "window at {start} does not extend coverage of {len}");
                covered = end;
            }
            assert_eq!(covered, len);
        }
    }

    #[test]
    fn legacy_eta_is_zero_on_the_only_chunk_while_busy() {
        let legacy = |i: usize, n: usize, elapsed: f64| -> f64 {
            if i > 0 {
                elapsed * (n.saturating_sub(i)) as f64 / i as f64
            } else {
                0.0
            }
        };
        assert_eq!(legacy(1, 1, 1_800.0), 0.0);
        assert_eq!(eta_seconds(1_800.0, 0, 57_600_000), None);
    }

    #[test]
    fn eta_is_null_for_the_first_minute_and_not_zero_while_work_remains() {
        assert_eq!(eta_seconds(59.9, 1_000, 10_000), None);
        assert_eq!(eta_seconds(120.0, 0, 10_000), None);
        let eta = eta_seconds(90.0, 100, 1_000).unwrap();
        assert!((eta - 810.0).abs() < 1e-9);
        assert!(eta >= 0.5);
        assert_eq!(eta_seconds(90.0, 1_000, 1_000), Some(0.0));
        assert_eq!(eta_seconds(60.0, 10_000, 10_050), Some(0.0));
    }

    #[test]
    fn local_chunk_counters_do_not_reset_the_turn_schedule() {
        let mut ledger = ProgressLedger::new();
        ledger.begin_turns(902, 50_000_000);
        ledger.begin_slice(160_000);
        ledger.observe_chunks(1, 1, 0, 160_000);
        ledger.observe_chunks(1, 1, 160_000, 160_000);
        ledger.finish_slice();
        ledger.begin_slice(200_000);
        ledger.observe_chunks(1, 2, 0, 200_000);
        let early = ledger.view(30.0);
        assert_eq!(early.completed, 1);
        assert_eq!(early.total, Some(902));
        assert_eq!(early.eta_s, None);
        assert!((early.fraction.unwrap() - (160_000.0 / 50_000_000.0)).abs() < 1e-12);

        ledger.observe_chunks(1, 1, 50_000, 80_000);
        let later = ledger.view(90.0);
        assert_eq!(later.total, Some(902));
        assert!(later.completed >= 1);
        assert!(later.fraction.unwrap() > early.fraction.unwrap());
        assert!(later.eta_s.unwrap() > 0.5);
    }

    #[test]
    fn chunk_schedule_does_not_regress_when_a_later_report_is_smaller() {
        let mut ledger = ProgressLedger::new();
        ledger.observe_chunks(3, 3, 300, 300);
        ledger.observe_chunks(3, 8, 300, 800);
        ledger.observe_chunks(2, 1, 100, 100);
        let view = ledger.view(10.0);
        assert_eq!(view.completed, 3);
        assert_eq!(view.total, Some(8));
        assert!((view.fraction.unwrap() - 300.0 / 800.0).abs() < 1e-12);
        assert_eq!(view.eta_s, None);
    }

    #[test]
    fn diarizing_phase_has_no_fake_eta() {
        let mut ledger = ProgressLedger::new();
        ledger.set_phase("diarizing", "finding speakers");
        ledger.observe_chunks(1, 1, 1, 1);
        let view = ledger.view(500.0);
        assert_eq!(view.phase, "diarizing");
        assert_eq!(view.eta_s, None);
        assert_eq!(view.total, None);
        assert_eq!(view.message, "finding speakers");
    }
}
