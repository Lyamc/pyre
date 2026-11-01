//! Context handling for long serve requests (int47). Independent switches:
//! - ctx-tooltrim (PYRE_CTX_TOOLTRIM, on by default): when a rendered chat prompt exceeds PYRE_CTX_TRIM_AT
//!   (fraction of n_ctx if <= 1, else tokens; default 0.6), elide the middle of old large tool outputs (oldest first,
//!   the last PYRE_CTX_TRIM_KEEP messages stay verbatim) until the prompt fits. Below the threshold nothing changes.
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub static TRIM_REQS: AtomicU64 = AtomicU64::new(0);
pub static TRIM_MSGS: AtomicU64 = AtomicU64::new(0);
pub static TRIM_TOKENS: AtomicU64 = AtomicU64::new(0);

fn env_f(k: &str, d: f64) -> f64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
pub fn tooltrim_on() -> bool { std::env::var("PYRE_CTX_TOOLTRIM").map(|v| v != "0").unwrap_or(true) }
/// Token threshold for `n_ctx`.
pub fn trim_at(n_ctx: usize) -> usize { let a = env_f("PYRE_CTX_TRIM_AT", 0.6); if a <= 1.0 { (a * n_ctx as f64) as usize } else { a as usize } }
fn keep() -> usize { env_f("PYRE_CTX_TRIM_KEEP", 6.0) as usize }
const MIN_CHARS: usize = 1500; const HEAD: usize = 600; const TAIL: usize = 300;

fn is_dump(m: &Value) -> bool {
    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
    let n = m.get("content").and_then(|c| c.as_str()).map(|s| s.len()).unwrap_or(0);
    (role == "tool" || role == "function") && n > MIN_CHARS
}
fn elide(s: &str) -> String {
    let mut h = HEAD.min(s.len()); while !s.is_char_boundary(h) { h -= 1; }
    let mut t = s.len() - TAIL.min(s.len()); while !s.is_char_boundary(t) { t += 1; }
    if t <= h { return s.to_string(); }
    format!("{}\n[... {} chars of old tool output elided by pyre ctx-tooltrim ...]\n{}", &s[..h], t - h, &s[t..])
}
/// Trims `msgs` (already normalized: string contents) in place until `count(msgs) <= limit` or nothing is left to
/// trim. Returns the number of messages elided. `count` renders + tokenizes.
pub fn tooltrim(msgs: &mut Vec<Value>, limit: usize, count: &mut dyn FnMut(&[Value]) -> Option<usize>) -> usize {
    let Some(mut n) = count(msgs) else { return 0 }; if n <= limit { return 0; }
    let n0 = n; let end = msgs.len().saturating_sub(keep()); let mut k = 0;
    for i in 0..end {
        if !is_dump(&msgs[i]) { continue; }
        let s = msgs[i]["content"].as_str().unwrap_or("").to_string(); msgs[i]["content"] = Value::String(elide(&s)); k += 1;
        if k % 4 == 0 || i + 1 == end { match count(msgs) { Some(c) => { n = c; if n <= limit { break; } } None => break } }
    }
    if k > 0 { if let Some(c) = count(msgs) { n = c; }
        TRIM_REQS.fetch_add(1, Relaxed); TRIM_MSGS.fetch_add(k as u64, Relaxed); TRIM_TOKENS.fetch_add(n0.saturating_sub(n) as u64, Relaxed);
        tracing::warn!("ctx-tooltrim: elided {k} old tool outputs, prompt {n0} -> {n} tokens (limit {limit})"); }
    k
}

