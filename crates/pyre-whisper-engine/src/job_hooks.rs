//! Optional job hooks so pyre can report chunk progress / honour cancel during long STT runs.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct JobHooks {
    pub cancel: Arc<AtomicBool>,
    /// Called as (chunk_1based, total_chunks).
    pub on_chunk: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
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

pub fn chunk(i: usize, n: usize) {
    if let Some(h) = HOOKS.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        if let Some(cb) = &h.on_chunk {
            cb(i, n);
        }
    }
}
