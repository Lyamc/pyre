//! Hardware fingerprint for CubeCL's persisted autotune cache.
//!
//! CubeCL stores autotune results (and, in pyre, compiled kernels) in
//! `%LOCALAPPDATA%\cubecl\default.db` or `$XDG_CACHE_HOME/cubecl/default.db`
//! (`kernel_cache_config` pins that global directory). The database is not tagged with
//! the GPU it was measured on. This module records the GPU name, UUID, total VRAM,
//! compute capability and SM count next to pyre's own config the first time it sees the
//! machine. A later boot with a different card surfaces a notice. The database is not
//! deleted or rewritten unless the user asks: `POST /api/autotune/retune` only renames
//! it to `default.db.bak-<unix>` on the next start, and `POST /api/autotune/ack` keeps
//! the current file and clears the notice.
//!
//! CPU core count is not part of the fingerprint. The expert pool reads it live; CubeCL
//! autotune selects GPU kernels.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
pub struct Hw {
    pub gpu_name: String,
    pub uuid: String,
    pub vram_mib: u64,
    pub compute: String,
    pub sms: u32,
}

/// Total VRAM within 256 MiB counts as the same card (driver reports differ by a few MiB).
pub fn changed(saved: &Hw, now: &Hw) -> bool {
    if !saved.uuid.is_empty() && !now.uuid.is_empty() && saved.uuid != now.uuid { return true; }
    if saved.gpu_name != now.gpu_name { return true; }
    if saved.compute != now.compute && !(saved.compute.is_empty() || now.compute.is_empty()) { return true; }
    if saved.sms != now.sms && saved.sms != 0 && now.sms != 0 { return true; }
    saved.vram_mib.abs_diff(now.vram_mib) > 256
}

pub fn describe(h: &Hw) -> String {
    if h.gpu_name.is_empty() && h.vram_mib == 0 { return "no NVIDIA GPU reported".into(); }
    format!("{} , {} MiB, compute {}, {} SMs", h.gpu_name, h.vram_mib, if h.compute.is_empty() { "?" } else { &h.compute }, h.sms)
}

pub(crate) fn state_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PYRE_AUTOTUNE_DIR") { return PathBuf::from(p); }
    if let Some(p) = std::env::var_os("LOCALAPPDATA") { return PathBuf::from(p).join("pyre"); }
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| crate::store::hfcache::home().join(".config"));
    base.join("pyre")
}

fn state_path() -> PathBuf { state_dir().join("autotune-hw.json") }
fn marker_path() -> PathBuf { state_dir().join("autotune-retune") }

pub fn cubecl_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PYRE_CUBECL_CACHE") { return PathBuf::from(p); }
    if let Some(p) = std::env::var_os("LOCALAPPDATA") { return PathBuf::from(p).join("cubecl"); }
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| crate::store::hfcache::home().join(".cache"));
    base.join("cubecl")
}

pub fn cubecl_db() -> PathBuf { cubecl_cache_dir().join("default.db") }

fn read_hw() -> Option<Hw> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(state_path()).unwrap_or_default()).unwrap_or(Value::Null);
    if v.is_null() { return None; }
    Some(Hw {
        gpu_name: v.get("gpu_name").and_then(|x| x.as_str()).unwrap_or("").into(),
        uuid: v.get("uuid").and_then(|x| x.as_str()).unwrap_or("").into(),
        vram_mib: v.get("vram_mib").and_then(|x| x.as_u64()).unwrap_or(0),
        compute: v.get("compute").and_then(|x| x.as_str()).unwrap_or("").into(),
        sms: v.get("sms").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
    })
}

fn write_hw(h: &Hw) {
    let p = state_path();
    if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
    let body = json!({"gpu_name": h.gpu_name, "uuid": h.uuid, "vram_mib": h.vram_mib, "compute": h.compute, "sms": h.sms,
        "cache_db": cubecl_db(), "recorded_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)});
    let _ = std::fs::write(p, serde_json::to_string_pretty(&body).unwrap_or_default());
}

pub fn current_hw() -> Hw {
    crate::limits::gpu_identity().map(|g| Hw { gpu_name: g.name, uuid: g.uuid, vram_mib: g.vram_mib, compute: g.compute, sms: g.sms })
        .unwrap_or(Hw { gpu_name: String::new(), uuid: String::new(), vram_mib: 0, compute: String::new(), sms: 0 })
}

static NOTICE: Mutex<String> = Mutex::new(String::new());

fn set_notice(s: String) { *NOTICE.lock().unwrap() = s; }

/// Compare the recorded card with this one. On the first run, record the card and do not
/// warn (there is no previous fingerprint to disagree with). Never touches `default.db`
/// except when a retune was requested, and then only by renaming it aside.
pub fn startup() {
    if marker_path().is_file() {
        let db = cubecl_db();
        if db.is_file() {
            let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let bak = db.with_file_name(format!("default.db.bak-{stamp}"));
            match std::fs::rename(&db, &bak) {
                Ok(()) => {
                    let _ = std::fs::remove_file(marker_path());
                    let now = current_hw();
                    write_hw(&now);
                    set_notice(format!("Autotune cache moved to {}. The next kernels measure this GPU ({}); the previous results were kept.", bak.display(), describe(&now)));
                    crate::logbuf::event("autotune", "INFO", json!({"action": "backup", "to": bak.display().to_string()}));
                    return;
                }
                Err(e) => {
                    set_notice(format!("Autotune retune was requested but {} could not be moved ({e}). The existing results were left in place.", db.display()));
                    crate::logbuf::event("autotune", "WARN", json!({"action": "backup_failed", "error": e.to_string()}));
                    return;
                }
            }
        } else {
            let _ = std::fs::remove_file(marker_path());
        }
    }
    let now = current_hw();
    match read_hw() {
        None => {
            write_hw(&now);
            set_notice(String::new());
        }
        Some(saved) if changed(&saved, &now) => {
            let db = cubecl_db();
            let msg = format!(
                "Hardware changed since autotune was recorded (was {}; now {}). CubeCL autotune at {} is still the previous card's. Pyre will not delete or overwrite it. POST /api/autotune/retune and restart to move it aside and measure this GPU, or POST /api/autotune/ack to keep these results.",
                describe(&saved), describe(&now), db.display());
            set_notice(msg.clone());
            crate::logbuf::event("autotune", "WARN", json!({"action": "suggest_rerun", "saved": describe(&saved), "current": describe(&now), "cache": db.display().to_string()}));
        }
        Some(_) => set_notice(String::new()),
    }
}

pub fn request_retune() -> Value {
    let p = marker_path();
    if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
    let _ = std::fs::write(&p, b"retune\n");
    let db = cubecl_db();
    json!({"ok": true, "retune_on_restart": true,
        "message": format!("On the next start, {} will be renamed to default.db.bak-<time> if it exists. It is not modified now. Restart pyre to measure this GPU.", db.display()),
        "cache": db.display().to_string()})
}

pub fn ack() -> Value {
    let now = current_hw();
    write_hw(&now);
    let _ = std::fs::remove_file(marker_path());
    set_notice(String::new());
    crate::logbuf::event("autotune", "INFO", json!({"action": "ack", "gpu": describe(&now)}));
    json!({"ok": true, "message": format!("Keeping the current autotune cache for {}.", describe(&now)), "gpu": describe(&now)})
}

pub fn status_value() -> Value {
    let now = current_hw();
    let saved = read_hw();
    let notice = NOTICE.lock().unwrap().clone();
    // A finished backup is informational. A mismatch, or a backup that could not be moved, asks for a decision.
    let suggest = !notice.is_empty() && !notice.contains("cache moved");
    json!({
        "suggest": suggest,
        "message": notice,
        "pending_retune": marker_path().is_file(),
        "cache": cubecl_db().display().to_string(),
        "cache_present": cubecl_db().is_file(),
        "current": {"gpu_name": now.gpu_name, "uuid": now.uuid, "vram_mib": now.vram_mib, "compute": now.compute, "sms": now.sms},
        "saved": saved.as_ref().map(|h| json!({"gpu_name": h.gpu_name, "uuid": h.uuid, "vram_mib": h.vram_mib, "compute": h.compute, "sms": h.sms})),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(name: &str, uuid: &str, vram: u64, compute: &str, sms: u32) -> Hw {
        Hw { gpu_name: name.into(), uuid: uuid.into(), vram_mib: vram, compute: compute.into(), sms }
    }

    #[test]
    fn same_card_ignores_small_vram_noise() {
        let a = hw("NVIDIA GeForce RTX 3060", "GPU-1", 12288, "8.6", 28);
        let b = hw("NVIDIA GeForce RTX 3060", "GPU-1", 12150, "8.6", 28);
        assert!(!changed(&a, &b), "VRAM reports within 256 MiB are the same card");
        let bigger = hw("NVIDIA GeForce RTX 3060", "GPU-1", 8192, "8.6", 28);
        assert!(changed(&a, &bigger));
    }

    #[test]
    fn gpu_swap_is_a_change_even_at_the_same_vram() {
        let a = hw("NVIDIA GeForce RTX 3060", "GPU-1", 12288, "8.6", 28);
        let b = hw("NVIDIA GeForce RTX 3060", "GPU-2", 12288, "8.6", 28);
        assert!(changed(&a, &b));
        let other = hw("NVIDIA GeForce RTX 4090", "GPU-9", 24576, "8.9", 128);
        assert!(changed(&a, &other));
        // Missing SM count does not by itself fire when everything else matches.
        let no_sm = hw("NVIDIA GeForce RTX 3060", "GPU-1", 12288, "8.6", 0);
        assert!(!changed(&a, &no_sm));
    }
}
