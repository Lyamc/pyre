//! Chat settings profiles. Three sources, stored apart from each other:
//!
//! * **default** is compiled into the binary. Nothing writes it.
//! * **tune** is the autotune snapshot. Only `save_tune` replaces it.
//! * **custom** is a map of named presets. `save_custom` inserts or replaces one name.
//!
//! Saving a custom never writes `tune` or the compiled defaults. Selecting a source
//! changes which preset is active; the caller applies context and KV separately, and a
//! failed apply leaves the active source where it was.
//!
//! Each setting is labeled by [`provenance`]: `default` when it matches the compiled
//! value, `tune` when it matches a tune value that differs from that default, and
//! `modified` otherwise (a named custom, or a hand edit).

use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::Mutex;

const KEYS: &[&str] = &[
    "temperature", "top_p", "top_k", "max_tokens", "seed",
    "context_tokens", "kv_bits", "system",
];

fn defaults_map() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("temperature".into(), json!(0.7));
    m.insert("top_p".into(), json!(0.95));
    m.insert("top_k".into(), json!(40));
    m.insert("max_tokens".into(), json!(4096));
    m.insert("seed".into(), Value::Null);
    m.insert("context_tokens".into(), json!(131072));
    m.insert("kv_bits".into(), json!(4));
    m.insert("system".into(), json!(""));
    m
}

fn blank() -> Store {
    Store { active_source: "default".into(), active_name: None, tune: None, customs: Map::new() }
}

#[derive(Clone)]
struct Store {
    active_source: String,
    active_name: Option<String>,
    tune: Option<Map<String, Value>>,
    customs: Map<String, Value>,
}

fn path() -> PathBuf {
    if let Ok(p) = std::env::var("PYRE_PROFILES_FILE") { return PathBuf::from(p); }
    crate::hw_tune::state_dir().join("settings-profiles.json")
}

fn load() -> Store {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path()).unwrap_or_default()).unwrap_or(Value::Null);
    let Some(o) = v.as_object() else { return blank(); };
    let active = o.get("active").and_then(|a| a.as_object());
    let source = active.and_then(|a| a.get("source")).and_then(|s| s.as_str()).unwrap_or("default");
    let source = if matches!(source, "default" | "tune" | "custom") { source } else { "default" };
    let name = active.and_then(|a| a.get("name")).and_then(|s| s.as_str()).map(|s| s.to_string());
    let tune = o.get("tune").and_then(|t| t.as_object()).cloned();
    let customs = o.get("customs").and_then(|c| c.as_object()).cloned().unwrap_or_default();
    Store { active_source: source.into(), active_name: name, tune, customs }
}

fn save(st: &Store) {
    let p = path();
    if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
    let body = json!({
        "active": {"source": st.active_source, "name": st.active_name},
        "tune": st.tune,
        "customs": st.customs,
    });
    let _ = std::fs::write(p, serde_json::to_string_pretty(&body).unwrap_or_default());
}

static LOCK: Mutex<()> = Mutex::new(());

fn with_store<T>(f: impl FnOnce(&mut Store) -> T) -> T {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut st = load();
    let out = f(&mut st);
    save(&st);
    out
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::String(x), Value::String(y)) => x.trim() == y.trim(),
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(p), Some(q)) => (p - q).abs() < 1e-6,
            _ => false,
        },
        (Value::Bool(x), Value::Bool(y)) => x == y,
        _ => false,
    }
}

/// `default` if `current` matches the compiled value, `tune` if it matches a tune
/// value that is not that default, otherwise `modified`.
pub fn provenance(current: &Map<String, Value>, tune: Option<&Map<String, Value>>) -> Map<String, Value> {
    let d = defaults_map();
    let mut out = Map::new();
    for k in KEYS {
        let cur = current.get(*k).unwrap_or(&Value::Null);
        let def = d.get(*k).unwrap_or(&Value::Null);
        let tuned = tune.and_then(|t| t.get(*k));
        let label = if tuned.is_some_and(|t| same(cur, t) && !same(t, def)) {
            "tune"
        } else if same(cur, def) {
            "default"
        } else {
            "modified"
        };
        out.insert((*k).into(), json!(label));
    }
    out
}

fn check_name(name: &str) -> Result<String, String> {
    let n = name.trim();
    if n.is_empty() || n.chars().count() > 40 {
        return Err("Name the preset with 1 to 40 characters.".into());
    }
    if n.eq_ignore_ascii_case("default") || n.eq_ignore_ascii_case("tune") {
        return Err("Default and tune are separate sources. A custom preset cannot use those names, and saving one does not change them.".into());
    }
    if !n.chars().all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '-' || c == '_') {
        return Err("Use letters, numbers, spaces, hyphens, or underscores.".into());
    }
    Ok(n.to_string())
}

fn num(v: &Value) -> Result<f64, String> {
    v.as_f64().ok_or_else(|| "expected a number".into())
}

/// Keep only known keys, check ranges, and store context as a token count.
pub fn normalize(v: &Value) -> Result<Map<String, Value>, String> {
    let Some(o) = v.as_object() else { return Err("settings must be an object".into()); };
    let mut m = Map::new();
    let temp = num(o.get("temperature").ok_or("temperature is required")?)?;
    if !(0.0..=2.0).contains(&temp) { return Err("temperature must be from 0 to 2".into()); }
    m.insert("temperature".into(), json!(temp));
    let top_p = num(o.get("top_p").ok_or("top_p is required")?)?;
    if !(0.0..=1.0).contains(&top_p) { return Err("top_p must be from 0 to 1".into()); }
    m.insert("top_p".into(), json!(top_p));
    let top_k = num(o.get("top_k").ok_or("top_k is required")?)?;
    if !(0.0..=200.0).contains(&top_k) { return Err("top_k must be from 0 to 200".into()); }
    m.insert("top_k".into(), json!(top_k.round() as u64));
    let max_tokens = num(o.get("max_tokens").ok_or("max_tokens is required")?)?;
    if !(1.0..=1_048_576.0).contains(&max_tokens) { return Err("max_tokens must be from 1 to 1048576".into()); }
    m.insert("max_tokens".into(), json!(max_tokens.round() as u64));
    match o.get("seed") {
        None | Some(Value::Null) => m.insert("seed".into(), Value::Null),
        Some(Value::String(s)) if s.trim().is_empty() => m.insert("seed".into(), Value::Null),
        Some(v) => {
            let n = num(v)?;
            if n < 0.0 || n > u32::MAX as f64 { return Err("seed must be empty or a 32-bit value".into()); }
            m.insert("seed".into(), json!(n.round() as u64))
        }
    };
    let ctx_raw = o.get("context_tokens").or_else(|| o.get("context")).ok_or("context_tokens is required")?;
    let ctx = match ctx_raw {
        Value::String(s) => crate::qn_ctx::parse_context(s)?,
        Value::Number(_) => crate::qn_ctx::validate_tokens(num(ctx_raw)?.round() as usize)?,
        _ => return Err("context_tokens must be a count or a size like 256k".into()),
    };
    m.insert("context_tokens".into(), json!(ctx));
    let bits = num(o.get("kv_bits").ok_or("kv_bits is required")?)?;
    if bits != 4.0 && bits != 8.0 { return Err("kv_bits must be 4 or 8".into()); }
    m.insert("kv_bits".into(), json!(bits as u32));
    let system = o.get("system").and_then(|s| s.as_str()).unwrap_or("");
    if system.chars().count() > 8000 { return Err("system prompt is longer than 8000 characters".into()); }
    m.insert("system".into(), json!(system.trim()));
    Ok(m)
}

fn view(st: &Store, message: &str) -> Value {
    let tune_stale = crate::hw_tune::status_value()["suggest"].as_bool().unwrap_or(false);
    json!({
        "ok": true,
        "message": message,
        "active": {"source": st.active_source, "name": st.active_name},
        "defaults": defaults_map(),
        "tune": st.tune,
        "tune_stale": tune_stale,
        "customs": st.customs,
        "keys": KEYS,
    })
}

pub fn status_value() -> Value {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    view(&load(), "")
}

/// Values of one source. Does not write anything.
pub fn values_for(source: &str, name: Option<&str>) -> Result<Map<String, Value>, String> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let st = load();
    match source {
        "default" => Ok(defaults_map()),
        "tune" => st.tune.clone().ok_or_else(|| "No tune profile is saved yet. Save as tune stores the current settings there without changing Default or any custom preset.".into()),
        "custom" => {
            let n = check_name(name.unwrap_or(""))?;
            st.customs.get(&n).and_then(|v| v.as_object()).cloned()
                .ok_or_else(|| format!("No custom preset named {n}."))
        }
        _ => Err("source must be default, tune, or custom".into()),
    }
}

pub fn set_active(source: &str, name: Option<&str>) -> Result<Value, String> {
    if !matches!(source, "default" | "tune" | "custom") {
        return Err("source must be default, tune, or custom".into());
    }
    // Confirm the source exists before recording it as active.
    let _ = values_for(source, name)?;
    Ok(with_store(|st| {
        st.active_source = source.into();
        st.active_name = if source == "custom" { name.map(|s| s.trim().to_string()) } else { None };
        view(st, &format!("Using the {source} settings."))
    }))
}

pub fn save_tune(values: &Value) -> Result<Value, String> {
    let norm = normalize(values)?;
    Ok(with_store(|st| {
        let replaced = st.tune.is_some();
        st.tune = Some(norm);
        let msg = if replaced {
            "Replaced the tune profile. Default and custom presets were not changed."
        } else {
            "Saved the tune profile. Default and custom presets were not changed."
        };
        crate::logbuf::event("settings_profile", "INFO", json!({"action": "save_tune", "replaced": replaced}));
        view(st, msg)
    }))
}

pub fn save_custom(name: &str, values: &Value, replace: bool) -> Result<Value, String> {
    let n = check_name(name)?;
    let norm = normalize(values)?;
    with_store(|st| {
        if st.customs.contains_key(&n) && !replace {
            return Err(format!("A custom preset named {n} already exists. Saving again replaces that preset only. Default and tune are not changed."));
        }
        st.customs.insert(n.clone(), Value::Object(norm));
        crate::logbuf::event("settings_profile", "INFO", json!({"action": "save_custom", "name": n}));
        Ok(view(st, &format!("Saved custom preset {n}. Default and tune were not changed.")))
    })
}

pub fn delete_custom(name: &str) -> Result<Value, String> {
    let n = check_name(name)?;
    with_store(|st| {
        if st.customs.remove(&n).is_none() {
            return Err(format!("No custom preset named {n}."));
        }
        if st.active_source == "custom" && st.active_name.as_deref() == Some(n.as_str()) {
            st.active_source = "default".into();
            st.active_name = None;
        }
        crate::logbuf::event("settings_profile", "INFO", json!({"action": "delete_custom", "name": n}));
        Ok(view(st, &format!("Deleted custom preset {n}. Default and tune were not changed.")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as TMutex;

    static TEST: TMutex<()> = TMutex::new(());

    fn isolate() -> (std::sync::MutexGuard<'static, ()>, tempfile::TempDir) {
        let g = TEST.lock().unwrap();
        let d = tempfile::tempdir().unwrap();
        std::env::set_var("PYRE_PROFILES_FILE", d.path().join("profiles.json"));
        (g, d)
    }

    fn sample(temp: f64, ctx: usize, bits: u32) -> Value {
        json!({
            "temperature": temp, "top_p": 0.95, "top_k": 40, "max_tokens": 4096,
            "seed": null, "context_tokens": ctx, "kv_bits": bits, "system": ""
        })
    }

    #[test]
    fn provenance_marks_default_tune_and_modified() {
        let d = defaults_map();
        let mut tune = d.clone();
        tune.insert("temperature".into(), json!(0.2));
        tune.insert("kv_bits".into(), json!(8));
        let mut cur = d.clone();
        cur.insert("temperature".into(), json!(0.2));
        cur.insert("top_k".into(), json!(10));
        let p = provenance(&cur, Some(&tune));
        assert_eq!(p["temperature"].as_str(), Some("tune"));
        assert_eq!(p["kv_bits"].as_str(), Some("default"));
        assert_eq!(p["top_k"].as_str(), Some("modified"));
        assert_eq!(p["context_tokens"].as_str(), Some("default"));
        // A tune value that equals the compiled default stays "default".
        let p2 = provenance(&d, Some(&tune));
        assert_eq!(p2["context_tokens"].as_str(), Some("default"));
        assert_eq!(p2["kv_bits"].as_str(), Some("default"));
    }

    #[test]
    fn custom_save_does_not_touch_tune_or_defaults() {
        let (_g, _d) = isolate();
        let near = |v: &Value, n: f64| (v.as_f64().unwrap() - n).abs() < 1e-6;
        let tune = save_tune(&sample(0.2, 262144, 8)).unwrap();
        assert!(near(&tune["tune"]["temperature"], 0.2));
        assert!(near(&tune["defaults"]["temperature"], 0.7));
        assert_eq!(tune["defaults"]["kv_bits"].as_u64(), Some(4));
        let saved = save_custom("night desk", &sample(1.1, 65536, 4), false).unwrap();
        assert!(near(&saved["tune"]["temperature"], 0.2), "saving a custom must leave the tune snapshot");
        assert_eq!(saved["tune"]["context_tokens"].as_u64(), Some(262144));
        assert_eq!(saved["defaults"]["context_tokens"].as_u64(), Some(131072));
        assert!(near(&saved["customs"]["night desk"]["temperature"], 1.1));
        let err = save_custom("tune", &sample(0.5, 8192, 4), true).unwrap_err();
        assert!(err.contains("does not change"), "{err}");
        let again = save_custom("night desk", &sample(0.4, 8192, 4), false).unwrap_err();
        assert!(again.contains("already exists"), "{again}");
        let replaced = save_custom("night desk", &sample(0.4, 8192, 4), true).unwrap();
        assert!(near(&replaced["customs"]["night desk"]["temperature"], 0.4));
        assert!(near(&replaced["tune"]["temperature"], 0.2));
        assert_eq!(replaced["defaults"]["kv_bits"].as_u64(), Some(4));
        let still = status_value();
        assert_eq!(still["tune"]["kv_bits"].as_u64(), Some(8));
        assert!(delete_custom("night desk").is_ok());
        assert!(status_value()["customs"].as_object().unwrap().is_empty());
        assert_eq!(status_value()["tune"]["context_tokens"].as_u64(), Some(262144));
    }
}
