//! GPU footprint limits (on by default): a VRAM reserve kept free for other apps, and a compute
//! duty-cycle cap so pyre leaves some GPU time to the desktop.
//!
//! * `--vram-reserve 1G` / `PYRE_VRAM_RESERVE` (default: max(1 GiB, 10% of VRAM); `0` disables).
//!   Model loads that would leave less than this free are refused (or rolled back), and the LLM
//!   batch slots (each owns a full-context KV cache) are capped to fit.
//! * `--gpu-util-max 90` / `PYRE_GPU_UTIL_MAX` (default 90; `100` or `0` disables). After every
//!   decode / diffusion step the worker sleeps `busy * (100 - u) / u`, so the GPU idles at least
//!   `100 - u` percent of the time pyre is running work.
//!
//! VRAM is read through NVML (NVIDIA, for both the CUDA and the wgpu backend). Without NVML
//! (non-NVIDIA wgpu adapters) the reserve can't be enforced and status says so.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

static RESERVE: AtomicU64 = AtomicU64::new(u64::MAX); // bytes; MAX = not configured yet
static UTIL: AtomicU64 = AtomicU64::new(90);
static THROTTLED_US: AtomicU64 = AtomicU64::new(0);

fn nvml() -> Option<&'static nvml_wrapper::Nvml> {
    static N: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();
    N.get_or_init(|| nvml_wrapper::Nvml::init().ok()).as_ref()
}

#[derive(Clone, Copy, Debug)]
pub struct Vram { pub total: u64, pub free: u64, pub used: u64 }

/// Device 0 memory (the device pyre runs on).
pub fn vram() -> Option<Vram> {
    let d = nvml()?.device_by_index(0).ok()?;
    let m = d.memory_info().ok()?;
    Some(Vram { total: m.total, free: m.free, used: m.used })
}
/// VRAM used by this process (None when the driver does not report it, e.g. some WDDM setups).
pub fn proc_vram() -> Option<u64> { let me = std::process::id(); nvml()?.device_by_index(0).ok()?.running_compute_processes().ok()?.into_iter().find(|p| p.pid == me).and_then(|p| match p.used_gpu_memory { nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => Some(b), _ => None }) }
pub fn gpu_util_now() -> Option<u32> { nvml()?.device_by_index(0).ok()?.utilization_rates().ok().map(|u| u.gpu) }

/// Parse `1G`, `512M`, `1.5GiB`, `0`, or plain bytes.
pub fn parse_size(s: &str) -> Option<u64> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches("ib").trim_end_matches('b');
    let (num, mul) = match t.chars().last()? {
        'k' => (&t[..t.len() - 1], 1u64 << 10), 'm' => (&t[..t.len() - 1], 1 << 20),
        'g' => (&t[..t.len() - 1], 1 << 30), 't' => (&t[..t.len() - 1], 1 << 40), _ => (t, 1),
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mul as f64) as u64)
}

/// Configure from CLI values (falling back to env vars, then defaults). Also lowers process priority.
pub fn init(reserve: Option<&str>, util: Option<u32>) {
    let r = reserve.map(String::from).or_else(|| std::env::var("PYRE_VRAM_RESERVE").ok());
    let bytes = match r.as_deref().and_then(parse_size) {
        Some(b) => b,
        None => vram().map(|v| (v.total / 10).max(1 << 30)).unwrap_or(1 << 30),
    };
    RESERVE.store(bytes, Ordering::Relaxed);
    let u = util.or_else(|| std::env::var("PYRE_GPU_UTIL_MAX").ok().and_then(|v| v.parse().ok())).unwrap_or(90);
    UTIL.store(if u == 0 { 100 } else { u.min(100) } as u64, Ordering::Relaxed);
    if std::env::var("PYRE_LOW_PRIORITY").map(|v| v != "0").unwrap_or(true) { lower_priority(); }
    tracing::info!("GPU limits: VRAM reserve {} MiB{}, GPU util max {}%", bytes >> 20,
        if vram().is_none() { " (no NVML: not enforced)" } else { "" }, util_max());
}

pub fn reserve() -> u64 { let r = RESERVE.load(Ordering::Relaxed); if r == u64::MAX { 1 << 30 } else { r } }
pub fn util_max() -> u32 { UTIL.load(Ordering::Relaxed) as u32 }

/// Bytes pyre may still allocate before breaching the reserve (`None` = unknown / unlimited).
pub fn headroom() -> Option<u64> {
    if reserve() == 0 { return None; }
    vram().map(|v| v.free.saturating_sub(reserve()))
}

/// Error if allocating `need` more bytes would breach the reserve.
pub fn check_alloc(need: u64, what: &str) -> anyhow::Result<()> {
    let (Some(v), r) = (vram(), reserve()) else { return Ok(()) };
    if r > 0 && v.free < need.saturating_add(r) {
        anyhow::bail!("{what} needs ~{} MiB of VRAM but only {} MiB is free and {} MiB is reserved for other apps \
            (--vram-reserve / PYRE_VRAM_RESERVE; 0 disables)", need >> 20, v.free >> 20, r >> 20);
    }
    Ok(())
}

/// Duty-cycle pacing: call with the start time of a GPU step that just completed.
pub fn pace(step_start: Instant) {
    let u = util_max();
    if u >= 100 { return; }
    let busy = step_start.elapsed();
    let sleep = busy.mul_f64((100 - u) as f64 / u as f64).min(Duration::from_millis(200));
    if sleep >= Duration::from_micros(200) {
        THROTTLED_US.fetch_add(sleep.as_micros() as u64, Ordering::Relaxed);
        std::thread::sleep(sleep);
    }
}

pub fn status() -> serde_json::Value {
    let v = vram();
    serde_json::json!({
        "vram_reserve_mib": reserve() >> 20,
        "vram_enforced": v.is_some() && reserve() > 0,
        "vram_total_mib": v.map(|v| v.total >> 20), "vram_free_mib": v.map(|v| v.free >> 20), "vram_used_mib": v.map(|v| v.used >> 20),
        "vram_headroom_mib": headroom().map(|h| h >> 20),
        "gpu_util_max_pct": util_max(), "gpu_util_now_pct": gpu_util_now(),
        "throttled_s": THROTTLED_US.load(Ordering::Relaxed) as f64 / 1e6,
        "low_priority": std::env::var("PYRE_LOW_PRIORITY").map(|v| v != "0").unwrap_or(true),
    })
}

#[cfg(windows)]
fn lower_priority() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass, BELOW_NORMAL_PRIORITY_CLASS};
    unsafe { SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS); }
}
#[cfg(not(windows))]
fn lower_priority() { unsafe { libc::nice(5); } }

#[cfg(test)]
mod tests {
    #[test]
    fn sizes() {
        assert_eq!(super::parse_size("1G"), Some(1 << 30));
        assert_eq!(super::parse_size("512MiB"), Some(512 << 20));
        assert_eq!(super::parse_size("0"), Some(0));
        assert_eq!(super::parse_size("1.5g"), Some(3 << 29));
    }
}

/// PYRE_VRAMLOG=1: print NVML used VRAM at a load/run stage (for the VRAM audit).
pub fn vlog(tag: &str) { if std::env::var("PYRE_VRAMLOG").is_ok() { if let Some(v) = vram() { eprintln!("vram {tag}: used {} MiB, free {} MiB", v.used >> 20, v.free >> 20); } } }

/// PYRE_VRAM_FLOOR_MB (default 1100, 0 = off): free-VRAM floor. A watcher thread polls NVML every 25 ms and
/// big allocations (>= 8 MiB) check before allocating; on a breach pyre prints the reason and exits (code 86),
/// so the driver frees its memory before the device runs dry.
pub fn vram_floor_mb() -> u64 { static F: std::sync::OnceLock<u64> = std::sync::OnceLock::new(); *F.get_or_init(|| std::env::var("PYRE_VRAM_FLOOR_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1100)) }
fn vram_breach(free: u64, need: u64, why: &str) -> ! {
    eprintln!("pyre: VRAM FLOOR BREACH ({why}): free {} MiB, about to allocate {} MiB, floor {} MiB (PYRE_VRAM_FLOOR_MB); aborting", free >> 20, need >> 20, vram_floor_mb());
    pyre_resume::drain().fire_and_wait(std::time::Duration::from_millis(400));
    std::process::exit(86)
}
/// Check before allocating `bytes` on the GPU (only allocations >= 8 MiB pay for an NVML query).
pub fn vram_check(bytes: u64) {
    let fl = vram_floor_mb(); if fl == 0 || bytes < (8 << 20) { return; }
    if let Some(v) = vram() { if v.free < bytes + (fl << 20) {
        // int45b: with an adaptive expert cache, first shrink it (applied at the next step) and allow the allocation
        // while free stays above max(1024 MiB, floor / 2); the hard abort is the last resort
        let hard = (fl / 2).max(768).min(fl) << 20; // int45e: 768 MiB hard minimum at floor 1024 (free 1127, 126 MiB KV growth aborted the server at 22k context)
        if v.free >= bytes + hard && emergency_shrink(v.free) { return; }
        vram_breach(v.free, bytes, "allocation"); } }
}
/// Start the 25 ms floor watcher once (call when the GPU is first used).
pub fn vram_watch() {
    static W: std::sync::Once = std::sync::Once::new();
    W.call_once(|| { let fl = vram_floor_mb(); if fl == 0 || vram().is_none() { return; }
        let _ = std::thread::Builder::new().name("vram-floor".into()).spawn(move || loop {
            // int45: a watcher breach first asks for a graceful stop (drain at the next step boundary, exit 87); only if
            // still below the floor 3 s later does it exit hard
            if let Some(v) = vram() { if v.free < (fl << 20) {
                if emergency_shrink(v.free) { eprintln!("pyre: VRAM below floor (watcher): free {} MiB < floor {} MiB; shrinking the expert cache", v.free >> 20, fl);
                    std::thread::sleep(std::time::Duration::from_secs(3)); }
                if vram().map_or(false, |v1| v1.free < (fl << 20)) {
                eprintln!("pyre: VRAM floor breach (watcher): free {} MiB < floor {} MiB; requesting graceful stop", v.free >> 20, fl);
                request_stop(); std::thread::sleep(std::time::Duration::from_secs(3)); }
                if let Some(v2) = vram() { if v2.free < (fl << 20) { vram_breach(v2.free, 0, "watcher"); } } } }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }); });
}
// ======================= adaptive GPU governor (int45) =======================
// One background thread (250 ms) watches NVML utilisation, temperature, power and device free VRAM and drives
//  * compute pacing: PYRE_GPU_DUTY (default 85 %, `off`/0/100 = disabled). The model thread calls gpu_pace() once per
//    step / prefill chunk and sleeps `busy * r`; r rises while the smoothed utilisation is above the current duty
//    target and decays when below, so light work (decode at ~60-75 %) is never slowed. The target itself auto-tunes:
//    -5 points (every 5 s, floor PYRE_GPU_DUTY_MIN, default 50) while the GPU is hot (>= 80 C), near its power limit
//    (>= 95 %) or saturated (>= 97 % for 20 s); down to base-15 while another app uses the GPU (util > 30 % while
//    pyre is idle); +1 point every 5 s back toward the base after 30 s calm.
//  * adaptive VRAM: when device free VRAM drops below floor + 512 MiB (other apps allocating), the expert-cache
//    target shrinks so free returns to floor + 768 MiB (256 MiB steps); after 30 s with free > floor + 832 MiB it grows
//    back by up to 512 MiB at a time (never below floor + 768 free), up to the configured size. The model applies the target at its next step (xc_target).
//  * graceful stop: a stop request (file <temp>/pyre/stop.<pid>, or request_stop()) is seen at the next step; the
//    model drains the GPU and exits 87. If no step comes within PYRE_STOP_GRACE_S (default 10) the thread exits.
use std::sync::atomic::AtomicBool;
static DUTY_BASE: AtomicU64 = AtomicU64::new(0); // 0 = pacing off
static DUTY_CUR: AtomicU64 = AtomicU64::new(0);
static PACE_R_PPM: AtomicU64 = AtomicU64::new(0);
static PACE_US: AtomicU64 = AtomicU64::new(0);
static DUTY_DOWN: AtomicU64 = AtomicU64::new(0);
static G_UTIL_EMA: AtomicU64 = AtomicU64::new(0); // x100
static G_TEMP: AtomicU64 = AtomicU64::new(0);
static G_PWR: AtomicU64 = AtomicU64::new(0); // permille of the power limit
static OTHERS_BUSY: AtomicBool = AtomicBool::new(false);
static LAST_STEP_MS: AtomicU64 = AtomicU64::new(0);
static XC_CFG: AtomicU64 = AtomicU64::new(0);   // configured expert-cache bytes (0 = no adaptive cache)
static XC_TGT: AtomicU64 = AtomicU64::new(0);   // current target bytes
static XC_CUR: AtomicU64 = AtomicU64::new(0);   // bytes the model has applied
static XC_SHRINKS: AtomicU64 = AtomicU64::new(0);
static XC_GROWS: AtomicU64 = AtomicU64::new(0);
static STOP: AtomicBool = AtomicBool::new(false);
static STOP_AT_MS: AtomicU64 = AtomicU64::new(0);
fn now_ms() -> u64 { static T0: OnceLock<Instant> = OnceLock::new(); T0.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1 }

fn duty_cfg() -> u64 {
    static D: OnceLock<u64> = OnceLock::new();
    *D.get_or_init(|| match std::env::var("PYRE_GPU_DUTY").ok().as_deref().map(str::trim) {
        Some("off") | Some("0") | Some("100") => 0,
        Some(v) => v.parse::<u64>().map(|x| x.clamp(30, 99)).unwrap_or(85),
        None => 85 })
}
fn duty_min() -> u64 { std::env::var("PYRE_GPU_DUTY_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(50u64).clamp(20, 95) }
pub fn duty_target() -> u64 { DUTY_CUR.load(Ordering::Relaxed) }

/// Pace the GPU: call once per model step (decode step, verify window, prefill chunk) on the model thread.
pub fn gpu_pace() {
    governor_start();
    thread_local! { static LAST: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) }; }
    let now = Instant::now(); LAST_STEP_MS.store(now_ms(), Ordering::Relaxed);
    let r = PACE_R_PPM.load(Ordering::Relaxed);
    let busy = LAST.with(|c| c.get()).map(|t| now - t).unwrap_or_default();
    let mut end = now;
    if r > 0 && busy < Duration::from_secs(10) && DUTY_BASE.load(Ordering::Relaxed) > 0 {
        let s = busy.mul_f64(r as f64 / 1e6).min(Duration::from_secs(2));
        if s >= Duration::from_micros(300) { std::thread::sleep(s); PACE_US.fetch_add(s.as_micros() as u64, Ordering::Relaxed); end = Instant::now(); }
    }
    LAST.with(|c| c.set(Some(end)));
}

/// The model registers its configured expert-cache size (bytes) once it exists.
pub fn xc_register(bytes: u64) { XC_CFG.store(bytes, Ordering::Relaxed); XC_TGT.store(bytes, Ordering::Relaxed); XC_CUR.store(bytes, Ordering::Relaxed); }
/// Target expert-cache bytes (the model resizes when this differs from what it applied; then calls xc_applied).
pub fn xc_target() -> u64 { XC_TGT.load(Ordering::Relaxed) }
pub fn xc_applied(bytes: u64) { XC_CUR.store(bytes, Ordering::Relaxed); }
/// PYRE_VRAM_ADAPT (default 1, 0 = off): adaptive expert cache (dedicated, returnable arenas).
/// Below the floor: cut the expert-cache target so free returns to floor + 768 MiB (applied at the next step). false = nothing to shrink.
fn emergency_shrink(free: u64) -> bool {
    let (cfg, tgt) = (XC_CFG.load(Ordering::Relaxed), XC_TGT.load(Ordering::Relaxed));
    if cfg == 0 || !adaptive_vram() || tgt <= (128 << 20) { return false; }
    let fl = vram_floor_mb().max(512) << 20; let deficit = (fl + (768 << 20)).saturating_sub(free).next_multiple_of(256 << 20);
    let nt = tgt.saturating_sub(deficit).max(128 << 20); if nt >= tgt { return false; }
    if XC_TGT.compare_exchange(tgt, nt, Ordering::Relaxed, Ordering::Relaxed).is_err() { return true; }
    XC_SHRINKS.fetch_add(1, Ordering::Relaxed); crate::metrics::warning();
    crate::logbuf::event("vram_pressure", "WARN", serde_json::json!({"action": "emergency_shrink_expert_cache", "free_mib": free >> 20, "floor_mib": fl >> 20, "from_mib": tgt >> 20, "to_mib": nt >> 20}));
    true
}
/// Bytes the expert cache will release at the next step (decided shrink not yet applied).
// ---- VRAM leases (cabin on-demand models): borrow VRAM from the expert cache while a side model is resident.
static LEASES: AtomicU64 = AtomicU64::new(0);
static LEASED: AtomicU64 = AtomicU64::new(0);
/// Make `need` bytes allocatable on top of the reserve (default 1 GiB+) by shrinking the expert-cache target,
/// then wait (<= 20 s) until the model has applied it (at its next step or idle tick). While any lease is held the
/// governor does not grow the cache back; when the last lease ends it regrows on its normal 30 s schedule.
/// Each successful call must be paired with `lease_end(need)`.
pub fn lease(need: u64, what: &str) -> anyhow::Result<()> {
    let Some(v) = vram() else { LEASES.fetch_add(1, Ordering::SeqCst); return Ok(()) };
    let want = need + reserve().max(vram_floor_mb() << 20) + (128 << 20);
    LEASES.fetch_add(1, Ordering::SeqCst);
    if v.free >= want { LEASED.fetch_add(need, Ordering::SeqCst); return Ok(()); }
    let (cfg, tgt) = (XC_CFG.load(Ordering::Relaxed), XC_TGT.load(Ordering::Relaxed));
    let deficit = (want - v.free).next_multiple_of(64 << 20);
    if cfg == 0 || tgt < deficit + (128 << 20) {
        LEASES.fetch_sub(1, Ordering::SeqCst);
        anyhow::bail!("{what} needs ~{} MiB of VRAM; {} MiB is free and the LLM expert cache can only give back {} MiB", need >> 20, v.free >> 20, tgt.saturating_sub(128 << 20) >> 20);
    }
    let nt = tgt - deficit;
    XC_TGT.store(nt, Ordering::SeqCst); XC_SHRINKS.fetch_add(1, Ordering::Relaxed);
    crate::logbuf::event("vram_lease", "INFO", serde_json::json!({"what": what, "need_mib": need >> 20, "free_mib": v.free >> 20, "xcache_from_mib": tgt >> 20, "xcache_to_mib": nt >> 20}));
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(20) {
        if XC_CUR.load(Ordering::SeqCst) <= nt && vram().map_or(true, |v| v.free >= want) { LEASED.fetch_add(need, Ordering::SeqCst); return Ok(()); }
        std::thread::sleep(Duration::from_millis(100));
    }
    LEASES.fetch_sub(1, Ordering::SeqCst);
    anyhow::bail!("{what}: the LLM did not release its expert cache within 20 s (busy?); try again when the chat is idle")
}
pub fn lease_end(need: u64) {
    if LEASES.load(Ordering::SeqCst) > 0 { LEASES.fetch_sub(1, Ordering::SeqCst); }
    let _ = LEASED.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(x.saturating_sub(need)));
}
pub fn leases() -> (u64, u64) { (LEASES.load(Ordering::SeqCst), LEASED.load(Ordering::SeqCst)) }
pub fn xc_pending_bytes() -> u64 { XC_CUR.load(Ordering::Relaxed).saturating_sub(XC_TGT.load(Ordering::Relaxed)) }
pub fn vram_adapt_on() -> bool { std::env::var("PYRE_VRAM_ADAPT").map(|v| v != "0").unwrap_or(true) }
fn adaptive_vram() -> bool { vram_adapt_on() }
/// a step within this window counts as pyre being busy (prefill chunks run ~3 s apart)
const ACTIVE_MS: u64 = 6000;
pub fn is_active() -> bool { now_ms().saturating_sub(LAST_STEP_MS.load(Ordering::Relaxed)) < ACTIVE_MS }

/// Ask pyre to stop at its next step (drain GPU work, exit 87).
pub fn request_stop() { if !STOP.swap(true, Ordering::Relaxed) { STOP_AT_MS.store(now_ms(), Ordering::Relaxed); eprintln!("pyre: stop requested; draining at the next step"); } }
pub fn stop_requested() -> bool { STOP.load(Ordering::Relaxed) }
pub fn stop_file() -> std::path::PathBuf { std::env::temp_dir().join("pyre").join(format!("stop.{}", std::process::id())) }
/// Called by the model at a step boundary after its GPU work is drained.
pub fn stop_now() -> ! {
    let _ = std::fs::remove_file(stop_file());
    eprintln!("pyre: stopped gracefully (GPU drained)");
    crate::logbuf::event("stop", "WARN", serde_json::json!({"graceful": true}));
    pyre_resume::drain().fire_and_wait(std::time::Duration::from_millis(pyre_resume::drain_ms()));
    std::process::exit(87)
}

pub fn governor_start() {
    static S: std::sync::Once = std::sync::Once::new();
    S.call_once(|| {
        let base = duty_cfg(); DUTY_BASE.store(base, Ordering::Relaxed); DUTY_CUR.store(base, Ordering::Relaxed);
        let _ = std::fs::create_dir_all(std::env::temp_dir().join("pyre"));
        let _ = std::thread::Builder::new().name("pyre-governor".into()).spawn(move || {
            let dev = nvml().and_then(|n| n.device_by_index(0).ok());
            let (mut ema, mut sat_since, mut calm_since, mut last_adj, mut last_grow, mut others_until) = (0.0f64, 0u64, 0u64, 0u64, 0u64, 0u64);
            let mut r = 0.0f64; let mut high_since = 0u64; let mut low_n = 0u32;
            let grace = std::env::var("PYRE_STOP_GRACE_S").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(10) * 1000;
            loop {
                std::thread::sleep(Duration::from_millis(250));
                let t = now_ms();
                // graceful stop
                if !STOP.load(Ordering::Relaxed) && stop_file().exists() { request_stop(); }
                // int45d: idle (no step for ACTIVE_MS) = nothing in flight on the GPU: stop now instead of waiting out the grace
                if STOP.load(Ordering::Relaxed) && !is_active() { stop_now(); }
                if STOP.load(Ordering::Relaxed) && t.saturating_sub(STOP_AT_MS.load(Ordering::Relaxed)) > grace {
                    eprintln!("pyre: stop grace ({} s) elapsed without a step boundary; exiting", grace / 1000); let _ = std::fs::remove_file(stop_file()); pyre_resume::drain().fire_and_wait(std::time::Duration::from_millis(400)); std::process::exit(88);
                }
                let Some(d) = dev.as_ref() else { continue };
                let util = d.utilization_rates().map(|u| u.gpu as f64).unwrap_or(0.0);
                let temp = d.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu).unwrap_or(0) as u64;
                let pwr = match (d.power_usage(), d.enforced_power_limit()) { (Ok(p), Ok(l)) if l > 0 => p as u64 * 1000 / l as u64, _ => 0 };
                ema = 0.7 * ema + 0.3 * util; G_UTIL_EMA.store((ema * 100.0) as u64, Ordering::Relaxed); G_TEMP.store(temp, Ordering::Relaxed); G_PWR.store(pwr, Ordering::Relaxed);
                let seen = LAST_STEP_MS.load(Ordering::Relaxed) > 0; let active = seen && t.saturating_sub(LAST_STEP_MS.load(Ordering::Relaxed)) < ACTIVE_MS;
                if seen && !active && util > 30.0 { others_until = t + 30_000; }
                OTHERS_BUSY.store(t < others_until, Ordering::Relaxed);
                // ---- duty target auto-tune
                if base > 0 {
                    let cur = DUTY_CUR.load(Ordering::Relaxed); let dmin = duty_min().min(base);
                    if active && ema >= 97.0 { if sat_since == 0 { sat_since = t; } } else { sat_since = 0; }
                    let hot = temp >= 80 || pwr >= 950 || (sat_since > 0 && t - sat_since > 20_000);
                    let mut nt = cur;
                    if hot { calm_since = 0; if t - last_adj > 5000 && cur > dmin { nt = cur.saturating_sub(5).max(dmin); } }
                    else if t < others_until && cur > base.saturating_sub(15).max(dmin) { nt = base.saturating_sub(15).max(dmin); calm_since = 0; }
                    else if temp < 75 && pwr < 850 && t >= others_until && (!active || ema < cur as f64 - 5.0) { if calm_since == 0 { calm_since = t; } if t - calm_since > 30_000 && t - last_adj > 5000 && cur < base { nt = cur + 1; } }
                    else { calm_since = 0; }
                    if nt != cur { DUTY_CUR.store(nt, Ordering::Relaxed); last_adj = t;
                        if nt < cur { DUTY_DOWN.fetch_add(1, Ordering::Relaxed); crate::metrics::warning();
                            crate::logbuf::event("gpu_duty", "WARN", serde_json::json!({"from": cur, "to": nt, "util_ema": (ema * 10.0).round() / 10.0, "temp_c": temp, "power_permille": pwr, "other_app": t < others_until})); }
                        else { crate::logbuf::event("gpu_duty", "INFO", serde_json::json!({"from": cur, "to": nt})); } }
                    // ---- pacing controller (only while pyre is stepping)
                    let tgt = DUTY_CUR.load(Ordering::Relaxed) as f64;
                    if active { let e = ema - tgt; r = if e > 0.0 { r + 0.01 * e } else { r + 0.004 * e }; r = r.clamp(0.0, 1.5); } else { r = (r - 0.02).max(0.0); }
                    PACE_R_PPM.store((r * 1e6) as u64, Ordering::Relaxed);
                }
                // ---- adaptive VRAM (expert cache)
                let cfg = XC_CFG.load(Ordering::Relaxed);
                if cfg > 0 && adaptive_vram() {
                    if let Ok(m) = d.memory_info() {
                        let fl = vram_floor_mb().max(512) << 20; let (free, tgt) = (m.free, XC_TGT.load(Ordering::Relaxed));
                        let applied = XC_CUR.load(Ordering::Relaxed) == tgt;
                        if free < fl + (512 << 20) { low_n += 1; } else { low_n = 0; }
                        if free < fl + (512 << 20) && low_n >= 3 && seen && applied && tgt > 0 { low_n = 0;
                            let deficit = (fl + (768 << 20)).saturating_sub(free).next_multiple_of(256 << 20);
                            let nt = tgt.saturating_sub(deficit);
                            XC_TGT.store(nt, Ordering::Relaxed); XC_SHRINKS.fetch_add(1, Ordering::Relaxed); crate::metrics::warning(); high_since = 0; last_grow = t;
                            crate::logbuf::event("vram_pressure", "WARN", serde_json::json!({"action": "shrink_expert_cache", "free_mib": free >> 20, "floor_mib": fl >> 20, "from_mib": tgt >> 20, "to_mib": nt >> 20}));
                        } else if free > fl + (832 << 20) && applied && tgt < cfg && LEASES.load(Ordering::SeqCst) == 0 {
                            if high_since == 0 { high_since = t; }
                            if t - high_since > 30_000 && t - last_grow > 30_000 {
                                let room = free - (fl + (768 << 20)); let step = room.min(512 << 20).min(cfg - tgt) & !((64u64 << 20) - 1);
                                if step > 0 { let nt = tgt + step; XC_TGT.store(nt, Ordering::Relaxed); XC_GROWS.fetch_add(1, Ordering::Relaxed); last_grow = t; high_since = t;
                                    crate::logbuf::event("vram_pressure", "INFO", serde_json::json!({"action": "grow_expert_cache", "free_mib": free >> 20, "from_mib": tgt >> 20, "to_mib": nt >> 20})); }
                            }
                        } else if free <= fl + (832 << 20) { high_since = 0; }
                    }
                }
            }
        });
    });
}

pub fn governor_status() -> serde_json::Value {
    serde_json::json!({
        "duty_base_pct": DUTY_BASE.load(Ordering::Relaxed), "duty_target_pct": DUTY_CUR.load(Ordering::Relaxed),
        "pace_ratio": PACE_R_PPM.load(Ordering::Relaxed) as f64 / 1e6, "paced_s": PACE_US.load(Ordering::Relaxed) as f64 / 1e6,
        "duty_backoffs": DUTY_DOWN.load(Ordering::Relaxed), "util_ema_pct": G_UTIL_EMA.load(Ordering::Relaxed) as f64 / 100.0,
        "temp_c": G_TEMP.load(Ordering::Relaxed), "power_frac": G_PWR.load(Ordering::Relaxed) as f64 / 1000.0, "other_app_busy": OTHERS_BUSY.load(Ordering::Relaxed),
        "xcache_cfg_bytes": XC_CFG.load(Ordering::Relaxed), "xcache_target_bytes": XC_TGT.load(Ordering::Relaxed), "xcache_applied_bytes": XC_CUR.load(Ordering::Relaxed),
        "xcache_shrinks": XC_SHRINKS.load(Ordering::Relaxed), "xcache_grows": XC_GROWS.load(Ordering::Relaxed), "stop_requested": STOP.load(Ordering::Relaxed),
    })
}