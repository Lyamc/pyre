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

// ---- ctx-summarize (PYRE_CTX_SUMMARIZE, on by default; only acts above PYRE_CTX_SUM_AT, default 0.8 of n_ctx) ----
// The middle of the conversation (after the leading system messages, before the last PYRE_CTX_SUM_KEEP messages) is
// summarized once by the model; the summary is remembered by a hash of the exact messages it replaced, so later
// requests that resend the same history get the same compacted prefix (prefix cache stays valid) instead of a
// rolling re-summary. A new compaction only happens when the compacted request is again above the threshold.
pub static SUM_EVENTS: AtomicU64 = AtomicU64::new(0);
pub static SUM_REUSED: AtomicU64 = AtomicU64::new(0);
pub static SUM_TOKENS: AtomicU64 = AtomicU64::new(0);
pub fn summarize_on() -> bool { std::env::var("PYRE_CTX_SUMMARIZE").map(|v| v != "0").unwrap_or(true) }
pub fn sum_at(n_ctx: usize) -> usize { let a = env_f("PYRE_CTX_SUM_AT", 0.8); if a <= 1.0 { (a * n_ctx as f64) as usize } else { a as usize } }
fn sum_keep() -> usize { env_f("PYRE_CTX_SUM_KEEP", 6.0) as usize }
static MEMO: std::sync::Mutex<Vec<(u64, usize, Value)>> = std::sync::Mutex::new(Vec::new());
fn hash(m: &[Value]) -> u64 { use std::hash::{Hash, Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); serde_json::to_string(m).unwrap_or_default().hash(&mut h); h.finish() }
fn head(m: &[Value]) -> usize { m.iter().take_while(|x| x.get("role").and_then(|r| r.as_str()) == Some("system")).count() }
/// Substitutes remembered summaries (longest first). Returns true when something was substituted.
pub fn apply_memo(msgs: &mut Vec<Value>) -> bool {
    let h0 = head(msgs); let memo = MEMO.lock().unwrap(); let mut best: Option<(usize, Value)> = None;
    for (hh, end, s) in memo.iter() { if *end <= msgs.len() && *end > h0 && hash(&msgs[..*end]) == *hh && best.as_ref().map_or(true, |b| *end > b.0) { best = Some((*end, s.clone())); } }
    drop(memo);
    if let Some((end, s)) = best { msgs.splice(h0..end, [s]); SUM_REUSED.fetch_add(1, Relaxed); true } else { false }
}
/// The span to summarize: (start, end) of the middle, or None when it is too short.
pub fn span(msgs: &[Value]) -> Option<(usize, usize)> {
    let h0 = head(msgs); let mut end = msgs.len().saturating_sub(sum_keep());
    // never cut between an assistant tool call and its tool results
    while end > h0 && matches!(msgs[end].get("role").and_then(|r| r.as_str()), Some("tool") | Some("function")) { end -= 1; }
    if end <= h0 + 1 { None } else { Some((h0, end)) }
}
pub fn summary_request(mid: &[Value]) -> Vec<Value> {
    let mut t = String::new();
    for m in mid { let r = m.get("role").and_then(|r| r.as_str()).unwrap_or("?"); let c = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let c = if c.len() > 6000 { elide(c) } else { c.to_string() }; t.push_str(&format!("[{r}]\n{c}\n"));
        if let Some(tc) = m.get("tool_calls") { t.push_str(&format!("[tool calls] {}\n", serde_json::to_string(tc).unwrap_or_default())); } }
    vec![serde_json::json!({"role": "user", "content": format!("Summarize the following earlier part of a conversation between a user and an AI coding assistant so the assistant can continue the work without it. Keep: the user's goals and requirements, decisions made, files created or changed (with paths), commands run and their important results, errors and how they were fixed, and open tasks. Be concise but complete. Output only the summary.\n\n{t}")})]
}
/// Remembers and returns the replacement message for msgs[..end] (hash covers the system head too).
pub fn remember(msgs: &[Value], end: usize, summary: &str, ntok: usize) -> Value {
    let v = serde_json::json!({"role": "user", "content": format!("[Summary of the earlier conversation, compacted by pyre ctx-summarize]\n{summary}")});
    let mut m = MEMO.lock().unwrap(); m.push((hash(&msgs[..end]), end, v.clone())); if m.len() > 32 { m.remove(0); }
    SUM_EVENTS.fetch_add(1, Relaxed); SUM_TOKENS.fetch_add(ntok as u64, Relaxed); v
}
