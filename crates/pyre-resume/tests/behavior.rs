//! In-process server: a dropped worker, a clean stop, an engine error, and a
//! graceful drain must all end the SSE stream with an explicit terminal event.

use pyre_resume::serve::{scripted, spawn, End, Script};
use pyre_resume::Status;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

struct Http {
    status: u16,
    headers: HashMap<String, String>,
    body: String,
}

fn exchange(addr: SocketAddr, method: &str, path: &str, body: Option<&str>, idle: Duration, stop: impl Fn(&str) -> bool) -> Http {
    let mut sock = TcpStream::connect(addr).expect("connect");
    sock.set_read_timeout(Some(idle)).unwrap();
    sock.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccept: text/event-stream\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(req.as_bytes()).unwrap();
    let mut raw = Vec::new();
    let start = Instant::now();
    loop {
        let mut buf = [0u8; 4096];
        match sock.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if let Some(text) = decoded_body(&raw) {
                    if stop(&text) {
                        break;
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {
                if start.elapsed() > idle {
                    break;
                }
            }
            Err(e) => panic!("read: {e}"),
        }
        if start.elapsed() > idle + Duration::from_secs(2) {
            break;
        }
    }
    parse_http(&raw)
}

fn decoded_body(raw: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(raw);
    let (_, body) = split_head(&text)?;
    Some(if is_chunked(&text) { decode_chunked(body.as_bytes()) } else { body.to_string() })
}

fn parse_http(raw: &[u8]) -> Http {
    let text = String::from_utf8_lossy(raw).into_owned();
    let (head, body) = split_head(&text).unwrap_or((text.as_str(), ""));
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap_or("").split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let body = if headers.get("transfer-encoding").map(|v| v.contains("chunked")).unwrap_or(false) {
        decode_chunked(body.as_bytes())
    } else {
        body.to_string()
    };
    Http { status, headers, body }
}

fn split_head(text: &str) -> Option<(&str, &str)> {
    text.split_once("\r\n\r\n").map(|(h, b)| (h, b))
}

fn is_chunked(head_and_more: &str) -> bool {
    head_and_more.split("\r\n\r\n").next().unwrap_or("").to_ascii_lowercase().contains("transfer-encoding: chunked")
}

fn decode_chunked(mut body: &[u8]) -> String {
    let mut out = Vec::new();
    while !body.is_empty() {
        let Some(nl) = body.windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let hex = std::str::from_utf8(&body[..nl]).unwrap_or("0");
        let size = usize::from_str_radix(hex.trim().split(';').next().unwrap_or("0"), 16).unwrap_or(0);
        body = &body[nl + 2..];
        if size == 0 {
            break;
        }
        if body.len() < size {
            out.extend_from_slice(body);
            break;
        }
        out.extend_from_slice(&body[..size]);
        body = &body[size..];
        if body.starts_with(b"\r\n") {
            body = &body[2..];
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn payloads(body: &str) -> Vec<Value> {
    body.lines().filter_map(|l| l.trim().strip_prefix("data:").map(str::trim)).filter(|l| *l != "[DONE]").filter_map(|l| serde_json::from_str(l).ok()).collect()
}

fn content_of(events: &[Value]) -> String {
    events.iter().filter_map(|v| v.pointer("/choices/0/delta/content").and_then(|c| c.as_str())).collect()
}

fn finish_of(events: &[Value]) -> Option<String> {
    events.iter().rev().find_map(|v| v.pointer("/choices/0/finish_reason").and_then(|c| c.as_str()).map(str::to_string))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_worker_is_aborted_not_a_clean_stop() {
    let dir = tempfile::tempdir().unwrap();
    let running = spawn(
        dir.path().to_path_buf(),
        scripted(Script {
            fresh: vec!["aa".into(), "bb".into()],
            fresh_end: End::Drop,
            resume: vec!["cc".into()],
            resume_end: End::Stop,
            gap: Duration::from_millis(5),
        }),
    )
    .await;
    let addr = running.addr;
    let http = exchange(addr, "POST", "/v1/chat/completions", Some(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":16}"#), Duration::from_secs(3), |b| b.contains("[DONE]"));
    assert_eq!(http.status, 200, "{}", http.body);
    let id = http.headers.get("x-request-id").cloned().expect("x-request-id");
    assert_eq!(http.headers.get("x-resume-token").map(String::as_str), Some(id.as_str()));
    let events = payloads(&http.body);
    assert!(events.iter().all(|e| e.get("id").and_then(|i| i.as_str()) == Some(id.as_str()) || e.get("error").is_some()), "{events:?}");
    assert_eq!(content_of(&events), "aabb");
    assert_eq!(finish_of(&events).as_deref(), Some("error"));
    assert!(events.iter().any(|e| e.pointer("/pyre/status").and_then(|s| s.as_str()) == Some("aborted")));
    assert!(http.body.contains("[DONE]"));
    assert!(!events.iter().any(|e| e.pointer("/choices/0/finish_reason").and_then(|s| s.as_str()) == Some("stop")));

    let got = exchange(addr, "GET", &format!("/v1/generations/{id}"), None, Duration::from_secs(2), |_| true);
    let snap: Value = serde_json::from_str(&got.body).unwrap();
    assert_eq!(snap["status"], "aborted");
    assert_eq!(snap["text"], "aabb");
    assert_eq!(snap["resumable"], true);

    let again = exchange(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&format!(r#"{{"resume":"{id}","stream":true}}"#)),
        Duration::from_secs(3),
        |b| b.contains("[DONE]"),
    );
    let ev2 = payloads(&again.body);
    assert!(content_of(&ev2).contains("aabb"), "{}", again.body);
    assert!(content_of(&ev2).contains("cc"), "{}", again.body);
    assert_eq!(finish_of(&ev2).as_deref(), Some("stop"));
    let got = exchange(addr, "GET", &format!("/v1/generations/{id}"), None, Duration::from_secs(2), |_| true);
    let snap: Value = serde_json::from_str(&got.body).unwrap();
    assert_eq!(snap["status"], "finished");
    running.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_emits_an_abort_frame() {
    let dir = tempfile::tempdir().unwrap();
    let running = spawn(dir.path().to_path_buf(), scripted(Script::hang_then_finish())).await;
    let addr = running.addr;
    let drain = pyre_resume::Drain::new();
    // The server's own drain is what the stream subscribed to.
    let server_drain = running.state.drain.clone();
    let _ = drain;
    let buf = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let buf2 = buf.clone();
    let thr = std::thread::spawn(move || {
        let http = exchange(addr, "POST", "/v1/chat/completions", Some(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#), Duration::from_secs(4), |b| b.contains("server shutting down") && b.contains("[DONE]"));
        *buf2.lock().unwrap() = format!("STATUS {}\n{}", http.status, http.body);
    });
    let start = Instant::now();
    loop {
        let got = buf.lock().unwrap().clone();
        if got.contains("one ") || start.elapsed() > Duration::from_secs(3) {
            break;
        }
        // The reader thread only publishes at the end. Watch the journal instead.
        if start.elapsed() > Duration::from_millis(500) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // Wait until the journal has the first token, then drain.
    let start = Instant::now();
    loop {
        let mut saw = false;
        if let Ok(rd) = std::fs::read_dir(dir.path()) {
            for ent in rd.flatten() {
                if ent.file_name().to_string_lossy().ends_with(".jsonl") {
                    if let Ok(s) = std::fs::read_to_string(ent.path()) {
                        if s.contains("one ") {
                            saw = true;
                        }
                    }
                }
            }
        }
        if saw || start.elapsed() > Duration::from_secs(3) {
            break;
        }
        std::thread::sleep(Duration::from_millis(15));
    }
    server_drain.fire_and_wait(Duration::from_secs(2));
    thr.join().unwrap();
    let body = buf.lock().unwrap().clone();
    assert!(body.contains("server shutting down"), "{body}");
    assert!(body.contains("\"finish_reason\":\"error\""), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
    assert!(body.contains("\"status\":\"aborted\""), "{body}");
    let _ = Status::Aborted;
    running.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_error_is_not_resumed() {
    let dir = tempfile::tempdir().unwrap();
    let running = spawn(
        dir.path().to_path_buf(),
        scripted(Script {
            fresh: vec!["x".into()],
            fresh_end: End::Error,
            resume: vec!["nope".into()],
            resume_end: End::Stop,
            gap: Duration::ZERO,
        }),
    )
    .await;
    let http = exchange(running.addr, "POST", "/v1/chat/completions", Some(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#), Duration::from_secs(3), |b| b.contains("[DONE]"));
    let id = http.headers.get("x-request-id").cloned().unwrap();
    assert!(http.body.contains("inference panicked"), "{}", http.body);
    assert!(http.body.contains("\"status\":\"error\""), "{}", http.body);
    let again = exchange(running.addr, "POST", "/v1/chat/completions", Some(&format!(r#"{{"resume":"{id}","stream":true,"continue":true}}"#)), Duration::from_secs(3), |b| b.contains("[DONE]"));
    assert!(!again.body.contains("nope"), "hard errors must not continue: {}", again.body);
    assert!(again.body.contains("\"status\":\"error\""), "{}", again.body);
    running.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finished_resume_replays_without_a_second_generation() {
    let dir = tempfile::tempdir().unwrap();
    let running = spawn(
        dir.path().to_path_buf(),
        scripted(Script {
            fresh: vec!["hi".into()],
            fresh_end: End::Stop,
            resume: vec!["AGAIN".into()],
            resume_end: End::Stop,
            gap: Duration::ZERO,
        }),
    )
    .await;
    let http = exchange(running.addr, "POST", "/v1/chat/completions", Some(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#), Duration::from_secs(3), |b| b.contains("[DONE]"));
    let id = http.headers.get("x-request-id").cloned().unwrap();
    assert_eq!(finish_of(&payloads(&http.body)).as_deref(), Some("stop"));
    let again = exchange(running.addr, "POST", "/v1/generations/{id}/resume".replace("{id}", &id).as_str(), Some("{}"), Duration::from_secs(3), |b| b.contains("[DONE]"));
    assert!(again.body.contains("hi"), "{}", again.body);
    assert!(!again.body.contains("AGAIN"), "{}", again.body);
    assert_eq!(finish_of(&payloads(&again.body)).as_deref(), Some("stop"));
    running.shutdown().await;
}
