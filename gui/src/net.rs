//! Blocking HTTP helpers for talking to a pyre server. Every call runs on a worker thread and
//! reports back through a channel, so the GPUI thread never blocks.
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::sync::mpsc::Sender;

pub enum Msg {
    /// Result of a JSON request, tagged with what asked for it.
    Json(&'static str, Result<Value, String>),
    /// A streamed chat delta.
    ChatDelta(String),
    /// The chat stream ended (`Err` on failure).
    ChatDone(Result<(), String>),
    /// Pull progress line.
    Pull(String),
    /// Binary body (TTS audio).
    Bytes(&'static str, Result<Vec<u8>, String>),
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(None)
        .build()
        .into()
}

fn read_json(mut resp: ureq::http::Response<ureq::Body>) -> Result<Value, String> {
    let status = resp.status();
    let mut s = String::new();
    resp.body_mut().as_reader().read_to_string(&mut s).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s.clone()));
    if status.is_success() {
        Ok(v)
    } else {
        Err(format!("HTTP {status}: {}", v.get("error").map(|e| e.to_string()).unwrap_or(s)))
    }
}

pub fn get(base: String, path: &'static str, tag: &'static str, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let r = agent().get(format!("{base}{path}")).call().map_err(|e| e.to_string()).and_then(read_json);
        let _ = tx.send(Msg::Json(tag, r));
    });
}

pub fn post(base: String, path: &'static str, body: Value, tag: &'static str, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let r = agent()
            .post(format!("{base}{path}"))
            .header("content-type", "application/json")
            .send(body.to_string())
            .map_err(|e| e.to_string())
            .and_then(read_json);
        let _ = tx.send(Msg::Json(tag, r));
    });
}

pub fn post_bytes(base: String, path: &'static str, body: Value, tag: &'static str, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let r = (|| {
            let mut resp = agent()
                .post(format!("{base}{path}"))
                .header("content-type", "application/json")
                .send(body.to_string())
                .map_err(|e| e.to_string())?;
            let status = resp.status();
            let mut b = Vec::new();
            resp.body_mut().with_config().limit(512 << 20).reader().read_to_end(&mut b).map_err(|e| e.to_string())?;
            if status.is_success() { Ok(b) } else { Err(format!("HTTP {status}: {}", String::from_utf8_lossy(&b))) }
        })();
        let _ = tx.send(Msg::Bytes(tag, r));
    });
}

/// Server-sent events: calls `on` with each `data:` payload until the stream ends.
fn sse(base: &str, path: &str, body: &Value, mut on: impl FnMut(&str) -> bool) -> Result<(), String> {
    let mut resp = agent()
        .post(format!("{base}{path}"))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .send(body.to_string())
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        let status = resp.status();
        let mut s = String::new();
        let _ = resp.body_mut().as_reader().read_to_string(&mut s);
        return Err(format!("HTTP {status}: {s}"));
    }
    let reader = BufReader::new(resp.body_mut().as_reader());
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if let Some(d) = line.strip_prefix("data:") {
            let d = d.trim();
            if d == "[DONE]" || !on(d) {
                break;
            }
        }
    }
    Ok(())
}

pub fn chat_stream(base: String, model: Option<String>, messages: Vec<(String, String)>, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let msgs: Vec<Value> = messages.iter().map(|(r, c)| serde_json::json!({"role": r, "content": c})).collect();
        let mut body = serde_json::json!({"messages": msgs, "stream": true, "max_tokens": 512});
        if let Some(m) = model { body["model"] = Value::String(m); }
        let tx2 = tx.clone();
        let r = sse(&base, "/v1/chat/completions", &body, |d| {
            if let Ok(v) = serde_json::from_str::<Value>(d) {
                if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                    return tx2.send(Msg::ChatDelta(c.to_string())).is_ok();
                }
            }
            true
        });
        let _ = tx.send(Msg::ChatDone(r));
    });
}

pub fn pull(base: String, repo: String, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let tx2 = tx.clone();
        let r = sse(&base, "/api/pull", &serde_json::json!({"repo": repo}), |d| tx2.send(Msg::Pull(d.to_string())).is_ok());
        let _ = tx.send(Msg::Json("pull-done", r.map(|_| Value::Null)));
    });
}

/// `multipart/form-data` upload of one audio file to `/v1/audio/transcriptions`.
pub fn transcribe(base: String, path: String, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let r = (|| {
            let data = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
            let name = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or("audio.wav".into());
            let boundary = "pyre-gui-boundary-7d1f";
            let mut body = Vec::new();
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n").as_bytes());
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes());
            body.extend_from_slice(&data);
            body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
            agent()
                .post(format!("{base}/v1/audio/transcriptions"))
                .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                .send(&body[..])
                .map_err(|e| e.to_string())
                .and_then(read_json)
        })();
        let _ = tx.send(Msg::Json("transcribe", r));
    });
}
