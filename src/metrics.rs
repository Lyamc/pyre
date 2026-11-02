//! Live server metrics: one registry read by `GET /metrics` (Prometheus text), `GET /api/metrics` (JSON),
//! `/api/status` and the dashboards. Request begin/finish take a short lock (once per request); the per-token
//! hook and every counter the model code bumps are plain atomics. GPU memory/utilization is sampled by one
//! background thread at most once per second (NVML through `crate::limits`), never on the decode path.
//!
//! Hooks for model code: `add_cpu_expert_ns(ns)`, `add_gpu_ns(ns)`, `expert_cache(hits, misses)`,
//! `xcache_slots(resident, slots, slot_bytes)`.
#![allow(dead_code)]
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static NEXT: AtomicU64 = AtomicU64::new(1);
static TOTAL_REQ: AtomicU64 = AtomicU64::new(0);
static TOTAL_PROMPT: AtomicU64 = AtomicU64::new(0);
static TOTAL_CACHED: AtomicU64 = AtomicU64::new(0);
static PREFIX_HIT_REQ: AtomicU64 = AtomicU64::new(0);
static TOTAL_GEN: AtomicU64 = AtomicU64::new(0);
static ERR_REQ: AtomicU64 = AtomicU64::new(0);
static CANCEL_REQ: AtomicU64 = AtomicU64::new(0);
static WARNINGS: AtomicU64 = AtomicU64::new(0);
static CPU_EXPERT_NS: AtomicU64 = AtomicU64::new(0);
static GPU_NS: AtomicU64 = AtomicU64::new(0);
static CACHE_HIT: AtomicU64 = AtomicU64::new(0);
static CACHE_MISS: AtomicU64 = AtomicU64::new(0);
static XC_RES: AtomicU64 = AtomicU64::new(0);
static XC_SLOTS: AtomicU64 = AtomicU64::new(0);
static XC_SLOT_B: AtomicU64 = AtomicU64::new(0);
static LAST_PPS: AtomicU64 = AtomicU64::new(0);
static LAST_DPS: AtomicU64 = AtomicU64::new(0);
// GPU sampler (bytes / percent; u64::MAX = unknown)
static G_TOTAL: AtomicU64 = AtomicU64::new(u64::MAX);
static G_FREE: AtomicU64 = AtomicU64::new(u64::MAX);
static G_USED: AtomicU64 = AtomicU64::new(u64::MAX);
static G_PROC: AtomicU64 = AtomicU64::new(u64::MAX);
static G_UTIL: AtomicU64 = AtomicU64::new(u64::MAX);
static G_SAMPLES: AtomicU64 = AtomicU64::new(0);

pub fn add_cpu_expert_ns(ns: u64) { CPU_EXPERT_NS.fetch_add(ns, Relaxed); }
pub fn add_gpu_ns(ns: u64) { GPU_NS.fetch_add(ns, Relaxed); }
/// Recent window: two buckets of ~256 decode tokens each (48 layers x top-10 lookups per token); the ratio covers the
/// previous full bucket plus the current one. Races at the rotation only blur the window slightly.
const WBUCKET: u64 = 256 * 48 * 10;
static W_H: AtomicU64 = AtomicU64::new(0); static W_M: AtomicU64 = AtomicU64::new(0);
static W_PH: AtomicU64 = AtomicU64::new(0); static W_PM: AtomicU64 = AtomicU64::new(0);
static LOG_WARN: AtomicU64 = AtomicU64::new(0);
pub fn expert_cache(hits: u64, misses: u64) { CACHE_HIT.fetch_add(hits, Relaxed); CACHE_MISS.fetch_add(misses, Relaxed);
    let h = W_H.fetch_add(hits, Relaxed) + hits; let m = W_M.fetch_add(misses, Relaxed) + misses;
    if h + m >= WBUCKET { W_PH.store(W_H.swap(0, Relaxed), Relaxed); W_PM.store(W_M.swap(0, Relaxed), Relaxed); } }
fn recent_hit() -> Option<f64> { let h = W_H.load(Relaxed) + W_PH.load(Relaxed); let m = W_M.load(Relaxed) + W_PM.load(Relaxed); if h + m > 0 { Some(h as f64 / (h + m) as f64) } else { None } }
/// Any WARN/ERROR line from tracing (separate from `warning()`, which counts real operational warnings).
pub fn log_warning() { LOG_WARN.fetch_add(1, Relaxed); }
pub fn xcache_slots(resident: usize, slots: usize, slot_bytes: usize) { XC_RES.store(resident as u64, Relaxed); XC_SLOTS.store(slots as u64, Relaxed); XC_SLOT_B.store(slot_bytes as u64, Relaxed); }
pub fn warning() { WARNINGS.fetch_add(1, Relaxed); }
static ROUTER_NAN: AtomicU64 = AtomicU64::new(0);
/// Router logits contained NaN (handled: NaN -> -inf). Logged at most once per 10 s.
pub fn router_nan(layer: usize, n: usize) {
    let c = ROUTER_NAN.fetch_add(1, Relaxed); warning();
    static LAST: AtomicU64 = AtomicU64::new(0); let t = uptime_s() as u64 + 1; let l = LAST.load(Relaxed);
    if c == 0 || t >= l + 10 { LAST.store(t, Relaxed); crate::logbuf::event("router_nan", "WARN", serde_json::json!({"layer": if layer == usize::MAX { None } else { Some(layer) }, "nan_logits": n, "total_events": c + 1})); }
}

fn epoch() -> &'static Instant { static E: OnceLock<Instant> = OnceLock::new(); E.get_or_init(Instant::now) }
fn now_s() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

/// Instance label for aggregating several nodes: PYRE_INSTANCE, else `<hostname>:<port>`.
static INSTANCE: OnceLock<String> = OnceLock::new();
pub fn set_instance(port: u16) { let _ = epoch();
    let _ = INSTANCE.set(std::env::var("PYRE_INSTANCE").unwrap_or_else(|_| format!("{}:{port}", hostname())));
    let _ = epoch();
}
pub fn instance() -> String { INSTANCE.get().cloned().unwrap_or_else(|| hostname()) }
pub fn hostname() -> String { std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).unwrap_or_else(|_| "unknown".into()) }
pub fn uptime_s() -> f64 { epoch().elapsed().as_secs_f64() }

/// Fixed-bucket histogram (cumulative counts written at export time).
struct Hist { b: &'static [f64], c: [AtomicU64; 12], sum_milli: AtomicU64, n: AtomicU64 }
impl Hist {
    const fn new(b: &'static [f64]) -> Self { Hist { b, c: [const { AtomicU64::new(0) }; 12], sum_milli: AtomicU64::new(0), n: AtomicU64::new(0) } }
    fn obs(&self, v: f64) { if !v.is_finite() || v < 0.0 { return; } let i = self.b.iter().position(|&x| v <= x).unwrap_or(self.b.len()); self.c[i].fetch_add(1, Relaxed); self.sum_milli.fetch_add((v * 1000.0) as u64, Relaxed); self.n.fetch_add(1, Relaxed); }
    fn prom(&self, o: &mut String, name: &str, help: &str, lb: &str) {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} histogram"); let mut acc = 0u64;
        for (i, x) in self.b.iter().enumerate() { acc += self.c[i].load(Relaxed); let _ = writeln!(o, "{name}_bucket{{{lb},le=\"{x}\"}} {acc}"); }
        acc += self.c[self.b.len()].load(Relaxed); let _ = writeln!(o, "{name}_bucket{{{lb},le=\"+Inf\"}} {acc}");
        let _ = writeln!(o, "{name}_sum{{{lb}}} {}\n{name}_count{{{lb}}} {}", self.sum_milli.load(Relaxed) as f64 / 1000.0, self.n.load(Relaxed));
    }
    fn json(&self) -> Value { let mut acc = 0u64; let mut v = vec![]; for (i, x) in self.b.iter().enumerate() { acc += self.c[i].load(Relaxed); v.push(json!([x, acc])); } acc += self.c[self.b.len()].load(Relaxed); v.push(json!(["+Inf", acc]));
        json!({"buckets": v, "sum": self.sum_milli.load(Relaxed) as f64 / 1000.0, "count": self.n.load(Relaxed)}) }
}
static H_TTFT: Hist = Hist::new(&[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0]);
static H_LAT: Hist = Hist::new(&[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0]);
static H_PPS: Hist = Hist::new(&[10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0]);
static H_DPS: Hist = Hist::new(&[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0]);

/// Lock-free per-request token counters, indexed by id % SLOTS (the id is stored to detect reuse).
const SLOTS: usize = 64;
struct ASlot { id: AtomicU64, gen: AtomicU64, first_us: AtomicU64 }
static AS: [ASlot; SLOTS] = [const { ASlot { id: AtomicU64::new(0), gen: AtomicU64::new(0), first_us: AtomicU64::new(0) } }; SLOTS];

struct Active { id: u64, model: String, prompt_n: usize, n_ctx: usize, max_tokens: usize, start: Instant, start_us: u64 }
static ACTIVE: Mutex<Vec<Active>> = Mutex::new(Vec::new());
static RECENT: Mutex<VecDeque<Value>> = Mutex::new(VecDeque::new());

fn now_us() -> u64 { epoch().elapsed().as_micros() as u64 }

/// Register a request; returns its id for `token`/`finish`.
pub fn begin(model: &str, prompt_n: usize, n_ctx: usize, max_tokens: usize) -> u64 {
    let id = NEXT.fetch_add(1, Relaxed); let s = &AS[id as usize % SLOTS];
    s.gen.store(0, Relaxed); s.first_us.store(0, Relaxed); s.id.store(id, Relaxed);
    ACTIVE.lock().unwrap().push(Active { id, model: model.into(), prompt_n, n_ctx, max_tokens, start: Instant::now(), start_us: now_us() });
    id
}

/// One generated token (lock-free).
pub fn token(id: u64) {
    let s = &AS[id as usize % SLOTS]; if s.id.load(Relaxed) != id { return; }
    if s.gen.fetch_add(1, Relaxed) == 0 { s.first_us.store(now_us().max(1), Relaxed); }
}

pub fn finish(id: u64, outcome: &str, t: Option<&crate::engine::Timings>) { finish_ex(id, outcome, t, None, None) }

/// `t` = the engine's final timings when the request completed normally; `reason` = finish reason; `error` = error text.
pub fn finish_ex(id: u64, outcome: &str, t: Option<&crate::engine::Timings>, reason: Option<&str>, error: Option<&str>) {
    let a = { let mut v = ACTIVE.lock().unwrap(); match v.iter().position(|a| a.id == id) { Some(i) => v.remove(i), None => return } };
    let s = &AS[id as usize % SLOTS]; let (gen_n, first_us) = if s.id.load(Relaxed) == id { (s.gen.load(Relaxed) as usize, s.first_us.load(Relaxed)) } else { (0, 0) };
    let total_ms = a.start.elapsed().as_secs_f64() * 1e3;
    let ttft_ms = if first_us > 0 { Some(first_us.saturating_sub(a.start_us) as f64 / 1e3) } else { None };
    let (pn, pms, pps, gn, gms, gps, cached) = match t {
        Some(t) => (t.prompt_n + t.cached_n, t.prompt_ms, t.prompt_per_second, t.predicted_n, t.predicted_ms, t.predicted_per_second, t.cached_n),
        None => {
            let pms = ttft_ms.unwrap_or(total_ms); let gms = total_ms - pms;
            (a.prompt_n, pms, if pms > 0.0 { a.prompt_n as f64 * 1e3 / pms } else { 0.0 }, gen_n, gms, if gms > 0.0 { gen_n as f64 * 1e3 / gms } else { 0.0 }, 0)
        }
    };
    TOTAL_REQ.fetch_add(1, Relaxed); TOTAL_PROMPT.fetch_add(pn as u64, Relaxed); TOTAL_GEN.fetch_add(gn as u64, Relaxed);
    TOTAL_CACHED.fetch_add(cached as u64, Relaxed); if cached > 0 { PREFIX_HIT_REQ.fetch_add(1, Relaxed); }
    if outcome == "error" { ERR_REQ.fetch_add(1, Relaxed); } else if outcome == "cancelled" { CANCEL_REQ.fetch_add(1, Relaxed); }
    H_LAT.obs(total_ms / 1e3); if let Some(x) = ttft_ms { H_TTFT.obs(x / 1e3); }
    if pn > cached && pps > 0.0 { H_PPS.obs(pps); LAST_PPS.store(pps.to_bits(), Relaxed); }
    if gn > 1 && gps > 0.0 { H_DPS.obs(gps); LAST_DPS.store(gps.to_bits(), Relaxed); }
    let rec = json!({"id": a.id, "model": a.model, "finished": now_s(), "outcome": outcome, "finish_reason": reason, "error": error,
        "prompt_n": pn, "cached_n": cached, "gen_n": gn, "ctx_used": a.prompt_n + gn, "n_ctx": a.n_ctx, "max_tokens": a.max_tokens,
        "ttft_ms": ttft_ms, "prefill_ms": pms, "prefill_tps": pps, "decode_ms": gms, "decode_tps": gps, "total_ms": total_ms});
    crate::logbuf::event("request", if outcome == "error" { "ERROR" } else { "INFO" }, rec.clone());
    let mut r = RECENT.lock().unwrap(); r.push_front(rec); r.truncate(50);
}

/// Background GPU sampler (NVML, once per PYRE_METRICS_GPU_MS, default 1000, min 1000) + VRAM-pressure warnings.
pub fn start_sampler() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| { let ms = std::env::var("PYRE_METRICS_GPU_MS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1000).max(1000);
        let _ = std::thread::Builder::new().name("pyre-metrics".into()).spawn(move || { let mut last_warn: Option<Instant> = None; loop {
            if let Some(v) = crate::limits::vram() { G_TOTAL.store(v.total, Relaxed); G_FREE.store(v.free, Relaxed); G_USED.store(v.used, Relaxed);
                let floor = std::env::var("PYRE_VRAM_FLOOR_MB").ok().and_then(|x| x.parse::<u64>().ok()).unwrap_or(1100) << 20;
                if floor > 0 && v.free < floor + (256 << 20) && last_warn.map(|t| t.elapsed().as_secs() >= 60).unwrap_or(true) {
                    last_warn = Some(Instant::now()); warning();
                    tracing::warn!("VRAM pressure: {} MiB free (floor {} MiB)", v.free >> 20, floor >> 20); } }
            if let Some(u) = crate::limits::gpu_util_now() { G_UTIL.store(u as u64, Relaxed); }
            if let Some(p) = crate::limits::proc_vram() { G_PROC.store(p, Relaxed); }
            G_SAMPLES.fetch_add(1, Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(ms)); } }); });
}

fn opt(a: &AtomicU64) -> Option<u64> { let v = a.load(Relaxed); if v == u64::MAX { None } else { Some(v) } }
fn gpu_wait_ns() -> u64 { crate::model::qn_gpu::GPU_NS.load(Relaxed) + GPU_NS.load(Relaxed) }

fn spec() -> (u64, u64, u64) { let s = crate::spec::status(); let g = |k: &str| s["speculative"][k].as_u64().unwrap_or(0); (g("rounds"), g("drafted"), g("accepted")) }

pub fn snapshot() -> Value {
    let active: Vec<Value> = ACTIVE.lock().unwrap().iter().map(|a| {
        let el = a.start.elapsed().as_secs_f64(); let s = &AS[a.id as usize % SLOTS];
        let (gen_n, first_us) = if s.id.load(Relaxed) == a.id { (s.gen.load(Relaxed) as usize, s.first_us.load(Relaxed)) } else { (0, 0) };
        let (phase, dtps, pms) = if first_us == 0 { ("prefill", 0.0, None) } else {
            let d = now_us().saturating_sub(first_us) as f64 / 1e6; ("decode", if d > 0.0 { gen_n.saturating_sub(1) as f64 / d } else { 0.0 }, Some(first_us.saturating_sub(a.start_us) as f64 / 1e3)) };
        json!({"id": a.id, "model": a.model, "phase": phase, "elapsed_s": el, "prompt_n": a.prompt_n, "gen_n": gen_n,
            "max_tokens": a.max_tokens, "ctx_used": a.prompt_n + gen_n, "n_ctx": a.n_ctx,
            "prefill_ms": pms, "prefill_tps": pms.map(|m| if m > 0.0 { a.prompt_n as f64 * 1e3 / m } else { 0.0 }), "decode_tps": dtps})
    }).collect();
    let (h, m) = (CACHE_HIT.load(Relaxed), CACHE_MISS.load(Relaxed)); let (rounds, drafted, accepted) = spec();
    let toks = (TOTAL_PROMPT.load(Relaxed) - TOTAL_CACHED.load(Relaxed).min(TOTAL_PROMPT.load(Relaxed))) + TOTAL_GEN.load(Relaxed);
    let (res, slots, sb) = (XC_RES.load(Relaxed), XC_SLOTS.load(Relaxed), XC_SLOT_B.load(Relaxed));
    json!({
        "instance": instance(), "host": hostname(), "version": env!("CARGO_PKG_VERSION"), "build": option_env!("PYRE_BUILD").unwrap_or("dev"), "uptime_s": uptime_s(),
        "flags": {"fast": crate::spec::fast(), "xexact": std::env::var("PYRE_QN_XEXACT").map(|v| v == "1").unwrap_or(false)},
        "active": active,
        "recent": RECENT.lock().unwrap().iter().cloned().collect::<Vec<_>>(),
        "totals": {"requests": TOTAL_REQ.load(Relaxed), "prompt_tokens": TOTAL_PROMPT.load(Relaxed), "cached_prompt_tokens": TOTAL_CACHED.load(Relaxed), "prefix_cache_hit_requests": PREFIX_HIT_REQ.load(Relaxed),
            "generated_tokens": TOTAL_GEN.load(Relaxed), "errors": ERR_REQ.load(Relaxed), "cancelled": CANCEL_REQ.load(Relaxed), "warnings": WARNINGS.load(Relaxed), "log_warnings": LOG_WARN.load(Relaxed)},
        "throughput": {"last_prefill_tps": f64::from_bits(LAST_PPS.load(Relaxed)), "last_decode_tps": f64::from_bits(LAST_DPS.load(Relaxed)),
            "ttft_s": H_TTFT.json(), "latency_s": H_LAT.json(), "prefill_tps": H_PPS.json(), "decode_tps": H_DPS.json()},
        "compute": {"cpu_expert_ms": CPU_EXPERT_NS.load(Relaxed) as f64 / 1e6, "gpu_ms": gpu_wait_ns() as f64 / 1e6, "gpu_wait_ms": gpu_wait_ns() as f64 / 1e6,
            "cpu_expert_ms_per_token": if toks > 0 { Some(CPU_EXPERT_NS.load(Relaxed) as f64 / 1e6 / toks as f64) } else { None },
            "gpu_wait_ms_per_token": if toks > 0 { Some(gpu_wait_ns() as f64 / 1e6 / toks as f64) } else { None }},
        "expert_cache": {"hits": h, "misses": m, "hit_rate": if h + m > 0 { Some(h as f64 / (h + m) as f64) } else { None }, "hit_rate_recent": recent_hit(), "recent_window_tokens": 512,
            "resident": res, "slots": slots, "slot_bytes": sb, "resident_gb": (res * sb) as f64 / 1e9, "capacity_gb": (slots * sb) as f64 / 1e9},
        "speculative": {"rounds": rounds, "drafted": drafted, "accepted": accepted, "acceptance": if drafted > 0 { Some(accepted as f64 / drafted as f64) } else { None }},
        "context": crate::spec::status()["context"],
        "gpu": {"vram_total_bytes": opt(&G_TOTAL), "vram_free_bytes": opt(&G_FREE), "vram_used_bytes": opt(&G_USED), "process_vram_bytes": opt(&G_PROC), "util_pct": opt(&G_UTIL), "samples": G_SAMPLES.load(Relaxed)},
        "governor": crate::limits::governor_status(), "router_nan_events": ROUTER_NAN.load(Relaxed),
    })
}

/// Prometheus text exposition (version 0.0.4). Every series carries `instance` and `host` labels.
pub fn prometheus() -> String {
    let v = snapshot(); let lb = format!("instance=\"{}\",host=\"{}\"", esc(&instance()), esc(&hostname()));
    let mut o = String::with_capacity(8192);
    let mut g = |o: &mut String, name: &str, ty: &str, help: &str, val: Option<f64>| { if let Some(x) = val { let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {ty}\n{name}{{{lb}}} {x}"); } };
    let f = |p: &Value| p.as_f64();
    let _ = writeln!(o, "# HELP pyre_build_info Build and flags.\n# TYPE pyre_build_info gauge\npyre_build_info{{{lb},version=\"{}\",build=\"{}\",fast=\"{}\",xexact=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION"), option_env!("PYRE_BUILD").unwrap_or("dev"), v["flags"]["fast"], v["flags"]["xexact"]);
    g(&mut o, "pyre_uptime_seconds", "gauge", "Seconds since start.", f(&v["uptime_s"]));
    let t = &v["totals"];
    g(&mut o, "pyre_requests_total", "counter", "Finished requests.", f(&t["requests"]));
    g(&mut o, "pyre_request_errors_total", "counter", "Requests that ended in an error.", f(&t["errors"]));
    g(&mut o, "pyre_requests_cancelled_total", "counter", "Requests cancelled by the client.", f(&t["cancelled"]));
    g(&mut o, "pyre_warnings_total", "counter", "Operational warnings (VRAM pressure, watchdog-style events).", f(&t["warnings"]));
    g(&mut o, "pyre_log_warnings_total", "counter", "WARN/ERROR lines in the log (any source).", f(&t["log_warnings"]));
    g(&mut o, "pyre_prompt_tokens_total", "counter", "Prompt tokens (incl. prefix-cache hits).", f(&t["prompt_tokens"]));
    g(&mut o, "pyre_prompt_cached_tokens_total", "counter", "Prompt tokens served from the prefix cache.", f(&t["cached_prompt_tokens"]));
    g(&mut o, "pyre_ctx_tooltrim_requests_total", "counter", "Requests whose old tool outputs ctx-tooltrim elided.", Some(crate::ctx::TRIM_REQS.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_ctx_tooltrim_messages_total", "counter", "Tool outputs elided by ctx-tooltrim.", Some(crate::ctx::TRIM_MSGS.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_ctx_tooltrim_tokens_total", "counter", "Prompt tokens removed by ctx-tooltrim.", Some(crate::ctx::TRIM_TOKENS.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_ctx_summarize_events_total", "counter", "Conversation compactions by ctx-summarize.", Some(crate::ctx::SUM_EVENTS.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_ctx_summarize_reused_total", "counter", "Requests that reused a remembered ctx-summarize compaction.", Some(crate::ctx::SUM_REUSED.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_ctx_summarize_tokens_total", "counter", "Summary tokens generated by ctx-summarize.", Some(crate::ctx::SUM_TOKENS.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_kv8_auto_engages_total", "counter", "Layer KV uploads switched to int8 by kv8 auto under VRAM pressure.", Some(crate::model::qwen4exp_gguf::KV8_AUTO_ENGAGES.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_kv8_fallbacks_total", "counter", "kv8 int8 regions turned back into f32 (uncovered case).", Some(crate::model::qwen4exp_gguf::KV8_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed) as f64));
    g(&mut o, "pyre_prefix_cache_hit_requests_total", "counter", "Requests that reused a cached prefix.", f(&t["prefix_cache_hit_requests"]));
    g(&mut o, "pyre_generated_tokens_total", "counter", "Generated tokens.", f(&t["generated_tokens"]));
    let n_act = v["active"].as_array().map(|a| a.len()).unwrap_or(0) as f64;
    let n_pf = v["active"].as_array().map(|a| a.iter().filter(|x| x["phase"] == "prefill").count()).unwrap_or(0) as f64;
    g(&mut o, "pyre_requests_active", "gauge", "Requests in flight.", Some(n_act));
    g(&mut o, "pyre_requests_prefill", "gauge", "Requests in flight still in prefill (or queued).", Some(n_pf));
    g(&mut o, "pyre_context_used_tokens", "gauge", "Context tokens of the current sequence.", f(&v["context"]["used"]));
    g(&mut o, "pyre_context_max_tokens", "gauge", "Context size.", f(&v["context"]["max"]));
    g(&mut o, "pyre_prefill_tokens_per_second", "gauge", "Prefill tok/s of the last request.", f(&v["throughput"]["last_prefill_tps"]));
    g(&mut o, "pyre_decode_tokens_per_second", "gauge", "Decode tok/s of the last request.", f(&v["throughput"]["last_decode_tps"]));
    H_TTFT.prom(&mut o, "pyre_ttft_seconds", "Time to first token.", &lb);
    H_LAT.prom(&mut o, "pyre_request_duration_seconds", "Request latency.", &lb);
    H_PPS.prom(&mut o, "pyre_prefill_tps", "Prefill tok/s per request.", &lb);
    H_DPS.prom(&mut o, "pyre_decode_tps", "Decode tok/s per request.", &lb);
    let x = &v["expert_cache"];
    g(&mut o, "pyre_expert_cache_hits_total", "counter", "Routed experts served from the GPU cache.", f(&x["hits"]));
    g(&mut o, "pyre_expert_cache_misses_total", "counter", "Routed experts run on the CPU.", f(&x["misses"]));
    g(&mut o, "pyre_expert_cache_hit_ratio", "gauge", "Expert cache hit ratio since start.", f(&x["hit_rate"]));
    g(&mut o, "pyre_expert_cache_hit_ratio_recent", "gauge", "Expert cache hit ratio over roughly the last 256-512 decode tokens.", f(&x["hit_rate_recent"]));
    g(&mut o, "pyre_expert_cache_slots", "gauge", "Expert cache slots.", f(&x["slots"]));
    g(&mut o, "pyre_expert_cache_resident", "gauge", "Experts resident in the cache.", f(&x["resident"]));
    g(&mut o, "pyre_expert_cache_resident_bytes", "gauge", "Bytes of resident experts.", f(&x["resident_gb"]).map(|g| g * 1e9));
    let s = &v["speculative"];
    g(&mut o, "pyre_spec_drafted_total", "counter", "Speculative draft tokens.", f(&s["drafted"]));
    g(&mut o, "pyre_spec_accepted_total", "counter", "Accepted draft tokens.", f(&s["accepted"]));
    g(&mut o, "pyre_spec_acceptance_ratio", "gauge", "Draft acceptance since start.", f(&s["acceptance"]));
    let c = &v["compute"];
    g(&mut o, "pyre_cpu_expert_seconds_total", "counter", "CPU expert pool time.", f(&c["cpu_expert_ms"]).map(|m| m / 1e3));
    g(&mut o, "pyre_gpu_wait_seconds_total", "counter", "Host time waiting on GPU readbacks.", f(&c["gpu_wait_ms"]).map(|m| m / 1e3));
    g(&mut o, "pyre_cpu_expert_ms_per_token", "gauge", "CPU expert ms per processed token since start.", f(&c["cpu_expert_ms_per_token"]));
    g(&mut o, "pyre_gpu_wait_ms_per_token", "gauge", "GPU wait ms per processed token since start.", f(&c["gpu_wait_ms_per_token"]));
    let d = &v["gpu"];
    g(&mut o, "pyre_gpu_vram_total_bytes", "gauge", "Device VRAM.", f(&d["vram_total_bytes"]));
    g(&mut o, "pyre_gpu_vram_free_bytes", "gauge", "Device VRAM free.", f(&d["vram_free_bytes"]));
    g(&mut o, "pyre_gpu_vram_used_bytes", "gauge", "Device VRAM used (all processes).", f(&d["vram_used_bytes"]));
    g(&mut o, "pyre_process_vram_bytes", "gauge", "VRAM used by this process (when the driver reports it).", f(&d["process_vram_bytes"]));
    g(&mut o, "pyre_gpu_utilization_percent", "gauge", "GPU utilization.", f(&d["util_pct"]));
    let gv = &v["governor"];
    g(&mut o, "pyre_gpu_duty_base_percent", "gauge", "Configured GPU duty limit (PYRE_GPU_DUTY; 0 = pacing off).", f(&gv["duty_base_pct"]));
    g(&mut o, "pyre_gpu_duty_target_percent", "gauge", "Current auto-tuned GPU duty target.", f(&gv["duty_target_pct"]));
    g(&mut o, "pyre_gpu_pace_ratio", "gauge", "Sleep inserted per unit of step time.", f(&gv["pace_ratio"]));
    g(&mut o, "pyre_gpu_paced_seconds_total", "counter", "Idle time inserted by GPU pacing.", f(&gv["paced_s"]));
    g(&mut o, "pyre_gpu_duty_backoffs_total", "counter", "Times the duty target was lowered (heat, power, saturation, other apps).", f(&gv["duty_backoffs"]));
    g(&mut o, "pyre_gpu_util_ema_percent", "gauge", "Smoothed GPU utilization seen by the governor.", f(&gv["util_ema_pct"]));
    g(&mut o, "pyre_gpu_temperature_celsius", "gauge", "GPU temperature.", f(&gv["temp_c"]));
    g(&mut o, "pyre_gpu_power_fraction", "gauge", "GPU power draw / enforced limit.", f(&gv["power_frac"]));
    g(&mut o, "pyre_gpu_other_app_busy", "gauge", "1 while another app keeps the GPU busy.", if gv["other_app_busy"].as_bool() == Some(true) { Some(1.0) } else { Some(0.0) });
    g(&mut o, "pyre_expert_cache_configured_bytes", "gauge", "Configured expert cache size.", f(&gv["xcache_cfg_bytes"]));
    g(&mut o, "pyre_expert_cache_target_bytes", "gauge", "Adaptive expert cache target (shrinks under VRAM pressure).", f(&gv["xcache_target_bytes"]));
    g(&mut o, "pyre_expert_cache_shrinks_total", "counter", "Expert cache shrinks under VRAM pressure.", f(&gv["xcache_shrinks"]));
    g(&mut o, "pyre_expert_cache_grows_total", "counter", "Expert cache grow-backs.", f(&gv["xcache_grows"]));
    g(&mut o, "pyre_router_nan_total", "counter", "Router logit vectors that contained NaN (handled).", f(&v["router_nan_events"]));
    o
}
fn esc(s: &str) -> String { s.replace('\\', "\\\\").replace('"', "\\\"") }