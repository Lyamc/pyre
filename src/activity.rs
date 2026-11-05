//! Process-wide activity bus: active jobs the UI status pill can show (chat, audio, cabin, loads).
//! One cheap snapshot + broadcast; progress producers upsert/remove without touching the HTTP stack.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use tokio::sync::broadcast;

#[derive(Clone, Debug)]
pub struct Job {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub stage: String,
    pub current: Option<u64>,
    pub total: Option<u64>,
    pub fraction: Option<f64>,
    pub eta_s: Option<f64>,
    pub bytes: Option<u64>,
    pub bytes_total: Option<u64>,
    pub cancelable: bool,
    pub started: Instant,
}

impl Job {
    pub fn new(id: impl Into<String>, kind: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(), kind: kind.into(), label: label.into(),
            stage: "queued".into(), current: None, total: None, fraction: None, eta_s: None,
            bytes: None, bytes_total: None, cancelable: false, started: Instant::now(),
        }
    }
    fn to_json(&self) -> Value {
        let mut v = json!({
            "id": self.id, "kind": self.kind, "label": self.label, "stage": self.stage,
            "cancelable": self.cancelable, "elapsed_s": self.started.elapsed().as_secs_f64(),
        });
        if let Some(x) = self.current { v["current"] = json!(x); }
        if let Some(x) = self.total { v["total"] = json!(x); }
        if let Some(x) = self.fraction { v["fraction"] = json!(x); }
        if let Some(x) = self.eta_s { v["eta_s"] = json!(x); }
        if let Some(x) = self.bytes { v["bytes"] = json!(x); }
        if let Some(x) = self.bytes_total { v["bytes_total"] = json!(x); }
        v
    }
}

fn map() -> &'static Mutex<HashMap<String, Job>> {
    static M: OnceLock<Mutex<HashMap<String, Job>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}
fn bus() -> &'static broadcast::Sender<Value> {
    static B: OnceLock<broadcast::Sender<Value>> = OnceLock::new();
    B.get_or_init(|| broadcast::channel(64).0)
}

fn notify() {
    let _ = bus().send(snapshot());
}

pub fn upsert(mut job: Job) {
    let mut g = map().lock().unwrap();
    if let Some(prev) = g.get(&job.id) { job.started = prev.started; }
    g.insert(job.id.clone(), job);
    drop(g);
    notify();
}

pub fn patch(id: &str, f: impl FnOnce(&mut Job)) {
    let mut g = map().lock().unwrap();
    if let Some(j) = g.get_mut(id) { f(j); }
    drop(g);
    notify();
}

pub fn remove(id: &str) {
    map().lock().unwrap().remove(id);
    notify();
}

pub fn snapshot() -> Value {
    let jobs: Vec<Value> = {
        let g = map().lock().unwrap();
        let mut v: Vec<_> = g.values().cloned().collect();
        v.sort_by(|a, b| a.started.cmp(&b.started));
        v.into_iter().map(|j| j.to_json()).collect()
    };
    let snap = crate::metrics::snapshot();
    let chat = snap["active"].clone();
    let g = &snap["gpu"];
    let b2m = |v: &Value| v.as_u64().map(|b| b >> 20);
    let headline = headline_from(&jobs, &chat);
    let idle = jobs.is_empty() && chat.as_array().map(|a| a.is_empty()).unwrap_or(true);
    json!({
        "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
        "state": if idle { "idle" } else { "busy" },
        "headline": headline,
        "jobs": jobs,
        "chat_active": chat,
        "vram": {
            "free_mib": b2m(&g["vram_free_bytes"]),
            "used_mib": b2m(&g["vram_used_bytes"]),
            "total_mib": b2m(&g["vram_total_bytes"]),
            "util_pct": g["util_pct"].clone(),
        },
        "limits": crate::limits::status(),
    })
}

fn headline_from(jobs: &[Value], chat: &Value) -> String {
    if let Some(j) = jobs.first() {
        let kind = j["kind"].as_str().unwrap_or("job");
        let label = j["label"].as_str().unwrap_or("Working");
        let mut s = if let (Some(c), Some(t)) = (j["current"].as_u64(), j["total"].as_u64()) {
            // Prefer clean progress; label may already include "chunk i/n".
            if label.contains('/') { label.to_string() } else { format!("{label} {c}/{t}") }
        } else {
            label.to_string()
        };
        let _ = kind;
        if let Some(eta) = j["eta_s"].as_f64() {
            if eta >= 1.0 {
                let m = (eta as u64) / 60; let sec = (eta as u64) % 60;
                s.push_str(&format!(" · {m}:{sec:02} left"));
            }
        } else if let Some(frac) = j["fraction"].as_f64() {
            s.push_str(&format!(" · {:.0}%", frac * 100.0));
        }
        return s;
    }
    if let Some(a) = chat.as_array().and_then(|a| a.first()) {
        let phase = a["phase"].as_str().unwrap_or("generating");
        return format!("Chat {phase}");
    }
    "Idle".into()
}

pub fn subscribe() -> broadcast::Receiver<Value> { bus().subscribe() }

/// Cancel a job by id when the producer registered a cancel hook.
pub fn cancel(id: &str) -> bool {
    // Audio jobs own cancel flags; cabin too. We just signal via a side table.
    cancels().lock().unwrap().insert(id.to_string(), true);
    // Also try audio_jobs cancel path is via HTTP; producers should poll canceled().
    true
}

fn cancels() -> &'static Mutex<HashMap<String, bool>> {
    static C: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn take_cancel(id: &str) -> bool {
    cancels().lock().unwrap().remove(id).unwrap_or(false)
}
