//! Durable partial generations and the SSE terminal contract for pyre chat streams.
//!
//! A generation is appended to a JSONL journal under [`default_dir`] (or `PYRE_RESUME_DIR`).
//! Every successful stream ends with a `finish_reason` of `stop`, `length`, `tool_calls`,
//! `content_filter`, or `function_call`, then `data: [DONE]`. Anything else — engine error,
//! client drop, process shutdown, or a channel that closes with no `Done` — is an explicit
//! `finish_reason: "error"` plus `pyre.status` of `aborted` or `error`, then `[DONE]`.
//! Clients treat a socket EOF without one of the success reasons as a failed stream and
//! `POST` the same id back as `resume`.
//!
//! The journal survives `SIGKILL` (the watchdog path). [`Store::recover`] marks journals
//! that have no closing record as `aborted` so the next process can continue them.

mod drain;
mod journal;
mod protocol;
pub mod serve;

pub use drain::{drain, shutdown_signal, Drain, TrackGuard};
pub use journal::{
    Continue, OpenMeta, Plan, PlanError, Record, Session, Snapshot, Status, Store, SUCCESS_FINISH,
};
pub use protocol::{data_frame, error_frame, frames_failed, frames_finished, role_frame, Envelope, Frame};

/// `PYRE_RESUME=0` turns persistence off. Terminal SSE events and request ids still apply.
pub fn enabled() -> bool {
    match std::env::var("PYRE_RESUME") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

/// Directory of `{id}.jsonl` journals. `PYRE_RESUME_DIR`, else `{temp}/pyre/generations`.
pub fn default_dir() -> std::path::PathBuf {
    match std::env::var("PYRE_RESUME_DIR") {
        Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
        _ => std::env::temp_dir().join("pyre").join("generations"),
    }
}

/// SSE comment interval. `PYRE_SSE_PING_SECS` (default 5, minimum 1).
pub fn ping_secs() -> u64 {
    std::env::var("PYRE_SSE_PING_SECS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(5).max(1)
}

/// How long a graceful stop waits for in-flight streams to emit their abort frame.
/// `PYRE_DRAIN_MS`, default 2000.
pub fn drain_ms() -> u64 {
    std::env::var("PYRE_DRAIN_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2000)
}

/// Journal retention. `PYRE_RESUME_TTL_HOURS`, default 48. `0` keeps everything.
pub fn ttl_hours() -> u64 {
    std::env::var("PYRE_RESUME_TTL_HOURS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(48)
}

/// `chatcmpl-` + 16 hex chars. Untrusted ids are rejected by the store.
pub fn new_id(prefix: &str) -> String {
    format!("{prefix}-{:016x}", rand_u64())
}

fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u64(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));
    h.write_usize(std::process::id() as usize);
    // Mix in the address of a stack slot so two calls in the same nanosecond differ.
    let slot = &h as *const _ as usize;
    h.write_usize(slot);
    h.finish()
}

/// Finish reasons a client may treat as a completed choice.
pub fn is_success_finish(reason: &str) -> bool {
    SUCCESS_FINISH.contains(&reason)
}
