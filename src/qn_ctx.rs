//! Qwen-Next context window and KV precision, chosen the way Strata's setup does
//! (github.com/Niko1221/Strata, MIT): named sizes from 8K through 512K, plus a raw token
//! count when those are not enough. Past the GGUF's trained length (usually 262144) the
//! rope table is YaRN-scaled. Nothing is hard-capped at 256K; the ceiling is 1,048,576
//! tokens, and a size that will not leave the VRAM floor free is refused with a smaller
//! window or 4-bit storage offered instead.
//!
//! KV precision is a user choice. 4-bit (absmax nibbles, see `model::kv4`) is the default
//! and is on, not only under VRAM pressure. 8-bit (`model::kv8`) stays available.
//! `PYRE_QN_KV8=0` keeps the cache f32. `PYRE_QN_KV8=auto` restores the old pressure-only
//! engage. `PYRE_QN_CTX` and `PYRE_QN_KV_BITS` override the saved file for this process.
//!
//! The saved file is `%LOCALAPPDATA%/pyre/qwen-context.json` (or `$XDG_CONFIG_HOME/pyre/`).
//! `PYRE_QN_CTX_FILE` points somewhere else (tests).

use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

/// Hard ceiling so a typo cannot ask for a multi-million-token rope table. Not a 256K cap.
pub const ABS_MAX: usize = 1 << 20;
/// Trained length used when no GGUF has been loaded yet. Real models overwrite this.
pub const DEFAULT_TRAIN: usize = 262144;

/// Strata-style presets. `Nk` means `N * 1024` tokens, so 256k is 262144 and 384k is 393216.
pub const PRESETS: &[(&str, usize)] = &[
    ("8k", 8 * 1024),
    ("16k", 16 * 1024),
    ("32k", 32 * 1024),
    ("64k", 64 * 1024),
    ("128k", 128 * 1024),
    ("256k", 256 * 1024),
    ("384k", 384 * 1024),
    ("512k", 512 * 1024),
];

/// Shape the KV estimate is computed from. Filled from the loaded GGUF; otherwise the
/// public Qwen3-Next full-attention shape (48 layers, full attention every 4, 2 KV heads,
/// head dim 256, indexer dim 128).
#[derive(Clone, Debug)]
pub struct Geom {
    pub n_attn: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub idx_hd: usize,
    pub ctx_train: usize,
    pub recent: usize,
    pub source: &'static str,
}

impl Geom {
    pub fn fallback() -> Self {
        Self { n_attn: 12, n_kv: 2, hd: 256, idx_hd: 128, ctx_train: DEFAULT_TRAIN, recent: 1024, source: "qwen3-next-fallback" }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RopeTables {
    pub mscale: f32,
    pub scale: f32,
}

static BITS: AtomicU8 = AtomicU8::new(4);
static MODE: AtomicU8 = AtomicU8::new(1); // 0 f32, 1 quantize, 2 quantize only under VRAM pressure
static CAP: AtomicU64 = AtomicU64::new(0);
static LOADED_TRAIN: AtomicU64 = AtomicU64::new(0);
static LOADED_KEY: AtomicU64 = AtomicU64::new(u64::MAX);
static MODEL: AtomicBool = AtomicBool::new(false);
static GEOM: Mutex<Option<Geom>> = Mutex::new(None);

fn prefs_path() -> PathBuf {
    if let Ok(p) = std::env::var("PYRE_QN_CTX_FILE") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(p).join("pyre").join("qwen-context.json");
    }
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| crate::store::hfcache::home().join(".config"));
    base.join("pyre").join("qwen-context.json")
}

#[derive(Clone, Debug)]
struct Saved {
    context: Option<usize>,
    bits: Option<u32>,
}

fn read_saved() -> Saved {
    let s = std::fs::read_to_string(prefs_path()).unwrap_or_default();
    let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
    Saved {
        context: v.get("context_tokens").and_then(|n| n.as_u64()).map(|n| n as usize),
        bits: v.get("kv_bits").and_then(|n| n.as_u64()).map(|n| n as u32),
    }
}

fn write_saved(context: Option<usize>, bits: u32) {
    let p = prefs_path();
    if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
    let body = json!({"context_tokens": context, "kv_bits": bits});
    let _ = std::fs::write(&p, serde_json::to_string_pretty(&body).unwrap_or_default());
}

/// Load env + saved file into the process settings. Safe to call more than once.
pub fn init() {
    let saved = read_saved();
    let bits = std::env::var("PYRE_QN_KV_BITS").ok().and_then(|s| s.parse().ok()).or(saved.bits).unwrap_or(4);
    BITS.store(if bits <= 4 { 4 } else { 8 }, Ordering::Relaxed);
    let mode = match std::env::var("PYRE_QN_KV8").ok().as_deref() {
        Some("0") => 0,
        Some("auto") => 2,
        Some("1") => 1,
        _ => 1,
    };
    MODE.store(mode, Ordering::Relaxed);
}

pub fn kv_bits() -> u32 { let b = BITS.load(Ordering::Relaxed); if b == 8 { 8 } else { 4 } }
/// 0 = f32 cache, 1 = store the older KV at [`kv_bits`] (default), 2 = only when VRAM is tight.
pub fn quant_mode() -> u8 { MODE.load(Ordering::Relaxed) }

pub fn parse_context(s: &str) -> Result<usize, String> {
    let t = s.trim();
    if t.is_empty() { return Err("context size is empty".into()); }
    if let Some((_, n)) = PRESETS.iter().find(|(k, _)| k.eq_ignore_ascii_case(t)) { return Ok(*n); }
    let lower = t.to_ascii_lowercase();
    let tok = if let Some(n) = lower.strip_suffix('k') {
        let v: f64 = n.trim().parse().map_err(|_| format!("'{s}' is not a context size (try 128k, 256k, 512k, or a token count)"))?;
        if !(v.is_finite() && v > 0.0) { return Err(format!("'{s}' is not a context size")); }
        (v * 1024.0).round() as usize
    } else {
        t.parse::<usize>().map_err(|_| format!("'{s}' is not a context size (try 128k, 256k, 512k, or a token count)"))?
    };
    validate_tokens(tok)
}

pub fn validate_tokens(tok: usize) -> Result<usize, String> {
    if tok < 16 { return Err(format!("{tok} tokens is below the 16-token minimum")); }
    if tok > ABS_MAX { return Err(format!("{tok} tokens is above the {ABS_MAX}-token ceiling")); }
    Ok(tok)
}

pub fn label_for(tokens: usize) -> String {
    PRESETS.iter().find(|(_, n)| *n == tokens).map(|(k, _)| (*k).to_string()).unwrap_or_else(|| tokens.to_string())
}

/// Context length to build the model with. Env wins, then the saved file, then 131072.
/// Past `ctx_train` is kept (YaRN); there is no 256K clamp.
pub fn startup_tokens(ctx_train: usize, has_idx: bool, idx_budget: usize) -> Result<usize, String> {
    if !has_idx { return Ok(idx_budget.max(16)); }
    if let Ok(s) = std::env::var("PYRE_QN_CTX") {
        return parse_context(&s).map_err(|e| format!("PYRE_QN_CTX: {e}"));
    }
    let tok = read_saved().context.unwrap_or(131072);
    let _ = ctx_train;
    validate_tokens(tok)
}

/// Inverse rope frequencies. `scale` is 1 at or under the trained length. Above it, YaRN
/// (llama.cpp / transformers defaults: beta_fast 32, beta_slow 1, mscale `1 + 0.1 ln(scale)`).
/// `mscale` is applied to cos and sin by the caller, and only when it is not 1, so an
/// unscaled run keeps the previous bit pattern.
pub fn rope_inv(n_rot: usize, base: f32, ctx_train: usize, ctx: usize) -> (Vec<f32>, RopeTables) {
    let half = n_rot / 2;
    let scale = if ctx_train > 0 && ctx > ctx_train { ctx as f32 / ctx_train as f32 } else { 1.0 };
    if scale <= 1.0 {
        let inv = (0..half).map(|i| base.powf(-(2.0 * i as f32) / n_rot as f32)).collect();
        return (inv, RopeTables { mscale: 1.0, scale: 1.0 });
    }
    let dim = n_rot as f32;
    let corr = |rot: f32| (dim * (ctx_train as f32 / (rot * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * base.ln());
    let low = corr(32.0).floor().max(0.0);
    let high = corr(1.0).ceil().min((half.saturating_sub(1)) as f32);
    let mut inv = Vec::with_capacity(half);
    for i in 0..half {
        let extrap = base.powf(-(2.0 * i as f32) / dim);
        let interp = extrap / scale;
        let ramp = if (high - low).abs() < 1e-6 { 1.0 } else { ((i as f32 - low) / (high - low)).clamp(0.0, 1.0) };
        // Low indices (high frequency) stay interpolated; high indices keep the trained frequency.
        inv.push(interp * (1.0 - ramp) + extrap * ramp);
    }
    let mscale = 1.0 + 0.1 * scale.ln();
    (inv, RopeTables { mscale, scale })
}

/// Rope identity: every unscaled context shares key 0. A scaled context's key is its token
/// count, because the YaRN factor is `tokens / trained`.
pub fn rope_key(tokens: usize, ctx_train: usize) -> u64 {
    if ctx_train == 0 || tokens <= ctx_train { 0 } else { tokens as u64 }
}

pub fn note_loaded(g: Geom, tokens: usize) {
    let train = g.ctx_train;
    *GEOM.lock().unwrap() = Some(g);
    CAP.store(tokens as u64, Ordering::Relaxed);
    LOADED_TRAIN.store(train as u64, Ordering::Relaxed);
    LOADED_KEY.store(rope_key(tokens, train), Ordering::Relaxed);
    MODEL.store(true, Ordering::Relaxed);
}

pub fn live_cap(model_max: usize) -> usize {
    let c = CAP.load(Ordering::Relaxed);
    if c == 0 { model_max } else { c as usize }
}

pub fn geom() -> Geom { GEOM.lock().unwrap().clone().unwrap_or_else(Geom::fallback) }

/// Bytes of GPU KV for `tokens` at `bits` (4 or 8) across every full-attention layer.
/// Older positions are packed; the last `recent` positions and the indexer stay f32.
/// The cap matches the upload (`need + max(need/8, 1024)`, rounded up to 1024).
/// `quant == false` is the all-f32 layout.
pub fn estimate_kv_bytes(g: &Geom, tokens: usize, bits: u32, quant: bool) -> u64 {
    let layers = g.n_attn.max(1) as u64;
    let nn = tokens.max(1) as u64;
    let cap = (nn + (nn / 8).max(1024)).next_multiple_of(1024);
    let rowkv = (g.n_kv.max(1) * g.hd.max(1)) as u64;
    let idx = g.idx_hd as u64;
    if !quant {
        let row = (g.n_kv.max(1) * g.hd.max(1) * 2 + g.idx_hd) as u64 * 4;
        return cap * row * layers;
    }
    let recent = g.recent.max(64) as u64;
    let qn = if nn > recent { ((nn - recent) / 64) * 64 } else { 0 };
    let qn = qn.min(cap);
    let per = if bits <= 4 { 5u64 } else { 9u64 }; // 0.625 or 1.125 bytes, in eighths
    let window = cap.saturating_sub(qn) * rowkv * 8; // K and V, f32
    let packed = qn * rowkv * 2 * per / 8;
    let ik = cap * idx * 4;
    (window + packed + ik) * layers
}

#[derive(Clone, Debug)]
pub struct Fit {
    pub tokens: usize,
    pub bits: u32,
    pub kv_bytes: u64,
    pub fits: Option<bool>,
    pub message: String,
    pub suggest_bits: Option<u32>,
    pub suggest_tokens: Option<usize>,
}

pub fn fit_at(tokens: usize, bits: u32, free: Option<u64>, floor_mib: u64, g: &Geom) -> Fit {
    let quant = quant_mode() != 0;
    let kv = estimate_kv_bytes(g, tokens, bits, quant);
    let need = kv.saturating_add((floor_mib.saturating_add(256)) << 20);
    let fits = free.map(|f| f >= need);
    let label = label_for(tokens);
    let kv_mib = kv >> 20;
    let yarn = g.ctx_train > 0 && tokens > g.ctx_train;
    let mut suggest_bits = None;
    let mut suggest_tokens = None;
    let message = match fits {
        None => format!("{label} at {bits}-bit needs about {kv_mib} MiB of KV cache. Free VRAM could not be read, so this was not checked against the floor."),
        Some(true) => {
            let mut m = format!("{label} at {bits}-bit needs about {kv_mib} MiB of KV cache and leaves the {floor_mib} MiB floor.");
            if bits <= 4 { m.push_str(" 4-bit storage is lossy (absmax per 32 values); 8-bit is closer to f32 and uses more memory."); }
            if yarn {
                let scale = tokens as f32 / g.ctx_train as f32;
                m.push_str(&format!(" This is past the trained {} positions, so YaRN rope scaling (x{scale:.3}) is used for the whole run.", g.ctx_train));
            }
            m
        }
        Some(false) => {
            let free_mib = free.unwrap_or(0) >> 20;
            let alt_bits = if bits >= 8 { 4 } else { 8 };
            let alt = fit_bytes_ok(tokens, alt_bits, free, floor_mib, g);
            if bits >= 8 && alt {
                suggest_bits = Some(4);
            }
            let smaller = PRESETS.iter().rev().map(|(_, n)| *n).filter(|n| *n < tokens).find(|n| fit_bytes_ok(*n, bits, free, floor_mib, g))
                .or_else(|| PRESETS.iter().rev().map(|(_, n)| *n).filter(|n| *n < tokens).find(|n| fit_bytes_ok(*n, 4, free, floor_mib, g)));
            suggest_tokens = smaller;
            let mut m = format!("{label} at {bits}-bit needs about {kv_mib} MiB of KV cache; {free_mib} MiB is free and {floor_mib} MiB must stay free. It will not be applied.");
            if suggest_bits == Some(4) { m.push_str(" 4-bit storage fits this window."); }
            if let Some(n) = smaller { m.push_str(&format!(" A {} window fits at {}-bit.", label_for(n), if fit_bytes_ok(n, bits, free, floor_mib, g) { bits } else { 4 })); }
            if suggest_bits.is_none() && smaller.is_none() { m.push_str(" No preset fits at 4-bit or 8-bit with the current free VRAM."); }
            m
        }
    };
    Fit { tokens, bits, kv_bytes: kv, fits, message, suggest_bits, suggest_tokens }
}

fn fit_bytes_ok(tokens: usize, bits: u32, free: Option<u64>, floor_mib: u64, g: &Geom) -> bool {
    let Some(free) = free else { return false };
    let kv = estimate_kv_bytes(g, tokens, bits, quant_mode() != 0);
    free >= kv.saturating_add((floor_mib.saturating_add(256)) << 20)
}

fn vram_now() -> (Option<u64>, u64) {
    (crate::limits::vram().map(|v| v.free), crate::limits::vram_floor_mb())
}

pub fn status_value() -> Value {
    let g = geom();
    let (free, floor) = vram_now();
    let active = current_active();
    let saved = read_saved();
    let bits = kv_bits();
    let fit = fit_at(active, bits, free, floor, &g);
    let presets: Vec<Value> = PRESETS.iter().map(|(k, n)| {
        let f4 = fit_at(*n, 4, free, floor, &g);
        let f8 = fit_at(*n, 8, free, floor, &g);
        json!({"id": k, "tokens": n, "fits_4bit": f4.fits, "fits_8bit": f8.fits, "kv_mib_4bit": f4.kv_bytes >> 20, "kv_mib_8bit": f8.kv_bytes >> 20,
            "yarn": g.ctx_train > 0 && *n > g.ctx_train})
    }).collect();
    let loaded_key = LOADED_KEY.load(Ordering::Relaxed);
    let saved_tok = saved.context;
    let reload = MODEL.load(Ordering::Relaxed) && saved_tok.map(|t| rope_key(t, g.ctx_train) != loaded_key).unwrap_or(false);
    json!({
        "context_tokens": active,
        "context_label": label_for(active),
        "saved_tokens": saved_tok,
        "kv_bits": bits,
        "kv_bits_default": 4,
        "quant_mode": match quant_mode() { 0 => "off", 2 => "auto", _ => "on" },
        "kv_note": "4-bit context storage is lossy (one absmax scale per 32 values, 15 levels). It is the default. 8-bit uses 127 levels and about 1.8x the packed bytes. Neither matches f32.",
        "presets": presets,
        "custom_max": ABS_MAX,
        "ctx_train": g.ctx_train,
        "yarn": g.ctx_train > 0 && active > g.ctx_train,
        "yarn_scale": if g.ctx_train > 0 && active > g.ctx_train { active as f64 / g.ctx_train as f64 } else { 1.0 },
        "reload_required": reload,
        "geometry": {"source": g.source, "attn_layers": g.n_attn, "kv_heads": g.n_kv, "head_dim": g.hd, "indexer_dim": g.idx_hd, "recent_f32": g.recent},
        "fit": {"fits": fit.fits, "kv_mib": fit.kv_bytes >> 20, "free_mib": free.map(|f| f >> 20), "floor_mib": floor, "message": fit.message,
            "suggest_bits": fit.suggest_bits, "suggest_tokens": fit.suggest_tokens},
        "vram_floor_mib": floor,
    })
}

fn current_active() -> usize {
    let c = CAP.load(Ordering::Relaxed);
    if c > 0 { return c as usize; }
    read_saved().context.unwrap_or(131072).clamp(16, ABS_MAX)
}

/// Apply a context size and/or KV precision. Refuses a window that does not leave the VRAM
/// floor unless `force` is set. A YaRN factor different from the loaded one is saved and
/// reported as `reload_required` (the running rope table is not rewritten under a live cache).
pub fn apply(context: Option<&str>, bits: Option<u32>, force: bool) -> Result<Value, String> {
    init();
    let g = geom();
    let mut tokens = current_active();
    if let Some(s) = context {
        tokens = parse_context(s)?;
    }
    let mut b = kv_bits();
    if let Some(bits) = bits {
        if bits != 4 && bits != 8 { return Err("kv_bits must be 4 or 8".into()); }
        b = bits;
        // Choosing a width turns packing on for this process, even if PYRE_QN_KV8=0.
        // The fit check below has to see that layout, not the f32 one.
        MODE.store(1, Ordering::Relaxed);
    }
    let (free, floor) = vram_now();
    let fit = fit_at(tokens, b, free, floor, &g);
    if fit.fits == Some(false) && !force {
        return Err(fit.message);
    }
    BITS.store(b as u8, Ordering::Relaxed);
    write_saved(Some(tokens), b);
    let train = if MODEL.load(Ordering::Relaxed) { LOADED_TRAIN.load(Ordering::Relaxed) as usize } else { g.ctx_train };
    let key = rope_key(tokens, train);
    let loaded = LOADED_KEY.load(Ordering::Relaxed);
    let reload = MODEL.load(Ordering::Relaxed) && loaded != u64::MAX && key != loaded;
    if !reload {
        CAP.store(tokens as u64, Ordering::Relaxed);
    }
    let mut message = fit.message;
    if reload {
        message = format!("Saved {tokens} tokens at {b}-bit. YaRN for this window differs from the loaded rope table, so it starts on the next model load. The running window stays {}.", current_active());
    } else if !MODEL.load(Ordering::Relaxed) {
        message = format!("Saved {tokens} tokens at {b}-bit. It applies when Qwen-Next loads. {message}");
    }
    Ok(json!({
        "ok": true,
        "applied": !reload,
        "reload_required": reload,
        "context_tokens": if reload { current_active() } else { tokens },
        "saved_tokens": tokens,
        "kv_bits": b,
        "message": message,
        "fit": {"fits": fit.fits, "kv_mib": fit.kv_bytes >> 20, "suggest_bits": fit.suggest_bits, "suggest_tokens": fit.suggest_tokens},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_strata_and_pass_256k() {
        assert_eq!(parse_context("256k").unwrap(), 262144);
        assert_eq!(parse_context("256K").unwrap(), 262144);
        assert_eq!(parse_context("128k").unwrap(), 131072);
        assert_eq!(parse_context("384k").unwrap(), 393216);
        assert_eq!(parse_context("512k").unwrap(), 524288);
        assert_eq!(parse_context("262144").unwrap(), 262144);
        assert_eq!(parse_context("393216").unwrap(), 393216);
        // A raw count above the trained 262144 is accepted, up to the 1M ceiling.
        assert_eq!(parse_context("300000").unwrap(), 300000);
        assert!(parse_context("1048577").is_err());
        assert!(parse_context("0").is_err());
        assert!(parse_context("nope").is_err());
        let (inv, meta) = rope_inv(64, 1.0e7, 262144, 131072);
        assert_eq!(meta.scale, 1.0);
        assert_eq!(meta.mscale, 1.0);
        assert_eq!(inv[0], 1.0); // base^0
        let plain: Vec<f32> = (0..32).map(|i| 1.0e7f32.powf(-(2.0 * i as f32) / 64.0)).collect();
        assert_eq!(inv, plain);
        let (y, ym) = rope_inv(64, 1.0e7, 262144, 524288);
        assert!((ym.scale - 2.0).abs() < 1e-6);
        assert!(ym.mscale > 1.0);
        assert!((ym.mscale - (1.0 + 0.1 * 2.0f32.ln())).abs() < 1e-5);
        // High frequencies are pulled down by the scale; the lowest stays near the trained frequency.
        assert!(y[0] < plain[0]);
        assert!((y[31] - plain[31]).abs() / plain[31] < 0.05 || y[31] <= plain[31]);
        assert_eq!(rope_key(262144, 262144), 0);
        assert_eq!(rope_key(131072, 262144), 0);
        assert_eq!(rope_key(524288, 262144), 524288);
    }

    #[test]
    fn fit_refuses_a_window_the_floor_cannot_hold() {
        let g = Geom { n_attn: 12, n_kv: 2, hd: 256, idx_hd: 128, ctx_train: 262144, recent: 1024, source: "test" };
        let b4 = estimate_kv_bytes(&g, 262144, 4, true);
        let b8 = estimate_kv_bytes(&g, 262144, 8, true);
        let f32b = estimate_kv_bytes(&g, 262144, 8, false);
        assert!(b4 < b8, "4-bit KV {b4} should be under 8-bit {b8}");
        assert!(b8 < f32b, "8-bit KV {b8} should be under f32 {f32b}");
        // 512k at 8-bit against a 12 GB card with only 3 GB free must fail and point at 4-bit or a shorter window.
        let free = 3u64 << 30;
        let bad = fit_at(524288, 8, Some(free), 1100, &g);
        assert_eq!(bad.fits, Some(false));
        assert!(bad.message.contains("4-bit") || bad.suggest_tokens.is_some(), "{}", bad.message);
        assert!(bad.suggest_bits == Some(4) || bad.suggest_tokens.is_some());
        // The same free memory can still hold a short 4-bit window.
        let small = fit_at(8192, 4, Some(free), 1100, &g);
        assert_eq!(small.fits, Some(true), "{}", small.message);
        // Unknown VRAM does not pretend the window fits.
        assert_eq!(fit_at(262144, 4, None, 1100, &g).fits, None);
    }

    #[test]
    fn startup_keeps_a_saved_window_above_256k() {
        assert_eq!(validate_tokens(393216).unwrap(), 393216);
        assert!(validate_tokens(262144).is_ok());
        // env-less default path is covered by parse; the clamp itself must not fold 384k back to 256k.
        assert_ne!(validate_tokens(393216).unwrap(), 262144);
    }
}
