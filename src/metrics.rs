//! Lightweight live metrics for the dashboard (`GET /api/metrics`): active requests with live
//! token counts, a ring of recent finished requests, and generic counters that model backends can
//! bump (CPU-expert vs GPU time, expert-cache hits). Cheap to update; read by the GPUI dashboard
//! and the web UI status tab.
//!
//! Hooks for model code (e.g. the Qwen-Next path on `qwen-perf`):
//!   `crate::metrics::add_cpu_expert_ns(ns)`, `crate::metrics::add_gpu_ns(ns)`,
//!   `crate::metrics::expert_cache(hits, misses)`.
#![allow(dead_code)]
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static NEXT: AtomicU64 = AtomicU64::new(1);
static TOTAL_REQ: AtomicU64 = AtomicU64::new(0);
static TOTAL_PROMPT: AtomicU64 = AtomicU64::new(0);
static TOTAL_GEN: AtomicU64 = AtomicU64::new(0);
static CPU_EXPERT_NS: AtomicU64 = AtomicU64::new(0);
static GPU_NS: AtomicU64 = AtomicU64::new(0);
static CACHE_HIT: AtomicU64 = AtomicU64::new(0);
static CACHE_MISS: AtomicU64 = AtomicU64::new(0);

pub fn add_cpu_expert_ns(ns: u64) { CPU_EXPERT_NS.fetch_add(ns, Relaxed); }
pub fn add_gpu_ns(ns: u64) { GPU_NS.fetch_add(ns, Relaxed); }
pub fn expert_cache(hits: u64, misses: u64) { CACHE_HIT.fetch_add(hits, Relaxed); CACHE_MISS.fetch_add(misses, Relaxed); }

struct Active { id: u64, model: String, prompt_n: usize, n_ctx: usize, max_tokens: usize, start: Instant, first: Option<Instant>, gen_n: usize }

static ACTIVE: Mutex<Vec<Active>> = Mutex::new(Vec::new());
static RECENT: Mutex<VecDeque<Value>> = Mutex::new(VecDeque::new());

fn now_s() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

/// Register a request; returns its id for `token`/`finish`.
pub fn begin(model: &str, prompt_n: usize, n_ctx: usize, max_tokens: usize) -> u64 {
    let id = NEXT.fetch_add(1, Relaxed);
    ACTIVE.lock().unwrap().push(Active { id, model: model.into(), prompt_n, n_ctx, max_tokens, start: Instant::now(), first: None, gen_n: 0 });
    id
}

pub fn token(id: u64) {
    if let Some(a) = ACTIVE.lock().unwrap().iter_mut().find(|a| a.id == id) {
        if a.first.is_none() { a.first = Some(Instant::now()); }
        a.gen_n += 1;
    }
}

/// `t` = the engine's final timings when the request completed normally.
pub fn finish(id: u64, outcome: &str, t: Option<&crate::engine::Timings>) {
    let a = { let mut v = ACTIVE.lock().unwrap(); match v.iter().position(|a| a.id == id) { Some(i) => v.remove(i), None => return } };
    let total_ms = a.start.elapsed().as_secs_f64() * 1e3;
    let (pn, pms, pps, gn, gms, gps, cached) = match t {
        Some(t) => (t.prompt_n, t.prompt_ms, t.prompt_per_second, t.predicted_n, t.predicted_ms, t.predicted_per_second, t.cached_n),
        None => {
            let pms = a.first.map(|f| (f - a.start).as_secs_f64() * 1e3).unwrap_or(total_ms);
            let gms = total_ms - pms;
            (a.prompt_n, pms, if pms > 0.0 { a.prompt_n as f64 * 1e3 / pms } else { 0.0 }, a.gen_n, gms, if gms > 0.0 { a.gen_n as f64 * 1e3 / gms } else { 0.0 }, 0)
        }
    };
    TOTAL_REQ.fetch_add(1, Relaxed);
    TOTAL_PROMPT.fetch_add(pn as u64, Relaxed);
    TOTAL_GEN.fetch_add(gn as u64, Relaxed);
    let mut r = RECENT.lock().unwrap();
    r.push_front(json!({"id": a.id, "model": a.model, "finished": now_s(), "outcome": outcome,
        "prompt_n": pn, "cached_n": cached, "gen_n": gn, "ctx_used": a.prompt_n + gn, "n_ctx": a.n_ctx,
        "prefill_ms": pms, "prefill_tps": pps, "decode_ms": gms, "decode_tps": gps, "total_ms": total_ms}));
    r.truncate(50);
}

pub fn snapshot() -> Value {
    let active: Vec<Value> = ACTIVE.lock().unwrap().iter().map(|a| {
        let el = a.start.elapsed().as_secs_f64();
        let (phase, dtps) = match a.first {
            None => ("prefill", 0.0),
            Some(f) => { let d = f.elapsed().as_secs_f64(); ("decode", if d > 0.0 { a.gen_n.saturating_sub(1) as f64 / d } else { 0.0 }) }
        };
        let pms = a.first.map(|f| (f - a.start).as_secs_f64() * 1e3);
        json!({"id": a.id, "model": a.model, "phase": phase, "elapsed_s": el, "prompt_n": a.prompt_n, "gen_n": a.gen_n,
            "max_tokens": a.max_tokens, "ctx_used": a.prompt_n + a.gen_n, "n_ctx": a.n_ctx,
            "prefill_ms": pms, "prefill_tps": pms.map(|m| if m > 0.0 { a.prompt_n as f64 * 1e3 / m } else { 0.0 }), "decode_tps": dtps})
    }).collect();
    let (h, m) = (CACHE_HIT.load(Relaxed), CACHE_MISS.load(Relaxed));
    json!({
        "active": active,
        "recent": RECENT.lock().unwrap().iter().cloned().collect::<Vec<_>>(),
        "totals": {"requests": TOTAL_REQ.load(Relaxed), "prompt_tokens": TOTAL_PROMPT.load(Relaxed), "generated_tokens": TOTAL_GEN.load(Relaxed)},
        "compute": {"cpu_expert_ms": CPU_EXPERT_NS.load(Relaxed) as f64 / 1e6, "gpu_ms": GPU_NS.load(Relaxed) as f64 / 1e6},
        "expert_cache": {"hits": h, "misses": m, "hit_rate": if h + m > 0 { Some(h as f64 / (h + m) as f64) } else { None }},
    })
}
