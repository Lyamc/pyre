//! Streaming client for pyre's OpenAI-compatible /v1/chat/completions (blocking ureq on a worker thread).
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

pub enum Ev {
    /// (conversation id, content delta, reasoning delta)
    Delta(u64, String, String),
    /// Final chunk extras: (conversation id, reasoning object, timings, usage); any may be Null.
    Extra(u64, Value, Value, Value),
    Done(u64, Result<(), String>),
    Json(&'static str, Result<Value, String>),
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().http_status_as_error(false).timeout_global(None).build().into()
}

pub fn get(base: String, path: &'static str, tag: &'static str, tx: Sender<Ev>) {
    std::thread::spawn(move || {
        let r = (|| {
            let mut resp = agent().get(format!("{base}{path}")).call().map_err(|e| e.to_string())?;
            let ok = resp.status().is_success();
            let mut s = String::new();
            resp.body_mut().with_config().limit(64 << 20).reader().read_to_string(&mut s).map_err(|e| e.to_string())?;
            if !ok { return Err(format!("HTTP {}", resp.status())); }
            serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())
        })();
        let _ = tx.send(Ev::Json(tag, r));
    });
}

/// Streams one chat completion. Setting `cancel` drops the connection, which makes pyre stop generating.
pub fn stream(base: String, conv: u64, body: Value, cancel: Arc<AtomicBool>, tx: Sender<Ev>) {
    std::thread::spawn(move || {
        let r = (|| -> Result<(), String> {
            let mut resp = agent()
                .post(format!("{base}/v1/chat/completions"))
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .send(body.to_string())
                .map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                let st = resp.status();
                let mut s = String::new();
                let _ = resp.body_mut().as_reader().read_to_string(&mut s);
                let msg = serde_json::from_str::<Value>(&s).ok().and_then(|v| v["error"]["message"].as_str().map(str::to_string)).unwrap_or(s);
                return Err(format!("HTTP {st}: {msg}"));
            }
            let reader = BufReader::new(resp.body_mut().with_config().limit(u64::MAX).reader());
            for line in reader.lines() {
                if cancel.load(Ordering::Relaxed) { return Ok(()); }
                let line = line.map_err(|e| e.to_string())?;
                let Some(d) = line.strip_prefix("data:") else { continue };
                let d = d.trim();
                if d == "[DONE]" { break; }
                let Ok(v) = serde_json::from_str::<Value>(d) else { continue };
                if let Some(e) = v["error"]["message"].as_str() { return Err(e.to_string()); }
                let dl = &v["choices"][0]["delta"];
                let c = dl["content"].as_str().unwrap_or("");
                let r = dl["reasoning_content"].as_str().unwrap_or("");
                if !c.is_empty() || !r.is_empty() {
                    if tx.send(Ev::Delta(conv, c.to_string(), r.to_string())).is_err() { return Ok(()); }
                }
                if !v["reasoning"].is_null() || !v["timings"].is_null() || !v["usage"].is_null() {
                    let _ = tx.send(Ev::Extra(conv, v["reasoning"].clone(), v["timings"].clone(), v["usage"].clone()));
                }
            }
            Ok(())
        })();
        let _ = tx.send(Ev::Done(conv, r));
    });
}
