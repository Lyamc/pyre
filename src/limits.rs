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
