//! Optional job hooks so pyre can report chunk progress / honour cancel during long STT runs.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Progress inside one `waveform_to_text` call.
/// `chunks_done` / `samples_done` are work already finished, not the window in progress.
#[derive(Clone, Copy, Debug)]
pub struct ChunkReport {
    pub chunks_done: usize,
    pub chunks_total: usize,
    pub samples_done: usize,
    pub samples_total: usize,
}

#[derive(Clone, Default)]
pub struct JobHooks {
    pub cancel: Arc<AtomicBool>,
    pub on_chunk: Option<Arc<dyn Fn(ChunkReport) + Send + Sync>>,
}

static HOOKS: Mutex<Option<JobHooks>> = Mutex::new(None);

pub fn install(hooks: JobHooks) {
    *HOOKS.lock().unwrap_or_else(|p| p.into_inner()) = Some(hooks);
}

pub fn clear() {
    *HOOKS.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

pub fn cancelled() -> bool {
    HOOKS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|h| h.cancel.load(Ordering::Relaxed))
        .unwrap_or(false)
}

pub fn report(report: ChunkReport) {
    if let Some(h) = HOOKS.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        if let Some(cb) = &h.on_chunk {
            cb(report);
        }
    }
}
