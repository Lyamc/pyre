//! Kill -9 the server after two tokens have been flushed. A new process on the
//! same journal must report the generation aborted and continue it on resume.
//! Drop is not allowed to run in the child: that is the watchdog path.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn restart_kill() {
    if std::env::var("PYRE_RESUME_CHILD").as_deref() == Ok("1") {
        pyre_resume::serve::child_main();
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let port_file = dir.path().join("port");
    let log = dir.path().join("child.log");
    let mut child = spawn_child(dir.path(), &port_file, &log);
    let port = wait_port(&port_file);
    let addr = format!("127.0.0.1:{port}");
    let (headers, body) = read_until(&addr, "POST", "/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":32}"#, Duration::from_secs(4), |b| b.contains("one ") && b.contains("two "));
    assert!(headers.to_ascii_lowercase().contains("x-request-id"), "{headers}");
    assert!(body.contains("one ") && body.contains("two "), "child log:\n{}\nbody:\n{body}", std::fs::read_to_string(&log).unwrap_or_default());
    let id = header(&headers, "x-request-id").expect("request id");
    child.kill().expect("sigkill");
    let _ = child.wait();

    // The lock file and the missing close record are what a hard kill leaves behind.
    let journal = std::fs::read_dir(dir.path()).unwrap().flatten().map(|e| e.path()).find(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl")).expect("journal");
    let raw = std::fs::read_to_string(&journal).unwrap();
    assert!(raw.contains("one "), "{raw}");
    assert!(!raw.contains("\"op\":\"close\""), "kill must not flush a close via Drop: {raw}");

    let _ = std::fs::remove_file(&port_file);
    let mut child = spawn_child(dir.path(), &port_file, &log);
    let port = wait_port(&port_file);
    let addr = format!("127.0.0.1:{port}");
    let (_, snap) = read_until(&addr, "GET", &format!("/v1/generations/{id}"), "", Duration::from_secs(3), |b| b.contains("aborted"));
    assert!(snap.contains("\"status\":\"aborted\""), "{snap}");
    assert!(snap.contains("restarted"), "{snap}");
    assert!(snap.contains("one "), "{snap}");

    let (_, resumed) = read_until(
        &addr,
        "POST",
        "/v1/chat/completions",
        &format!(r#"{{"resume":"{id}","stream":true}}"#),
        Duration::from_secs(4),
        |b| b.contains("[DONE]"),
    );
    assert!(resumed.contains("one "), "{resumed}");
    assert!(resumed.contains("done"), "{resumed}");
    assert!(resumed.contains("\"finish_reason\":\"stop\""), "{resumed}");
    assert!(resumed.contains("[DONE]"), "{resumed}");
    assert!(resumed.contains("\"replay\":true"), "prefix must be marked replay so a client can dedupe: {resumed}");

    let (_, snap) = read_until(&addr, "GET", &format!("/v1/generations/{id}"), "", Duration::from_secs(3), |b| b.contains("finished"));
    assert!(snap.contains("\"status\":\"finished\""), "{snap}");
    child.kill().ok();
    let _ = child.wait();
}

fn spawn_child(dir: &Path, port_file: &Path, log: &Path) -> std::process::Child {
    let exe = std::env::current_exe().unwrap();
    let log_f = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
    let err = log_f.try_clone().unwrap();
    Command::new(exe)
        .arg("restart_kill")
        .arg("--exact")
        .env("PYRE_RESUME_CHILD", "1")
        .env("PYRE_RESUME_DIR", dir)
        .env("PYRE_PORT_FILE", port_file)
        .stdout(Stdio::from(log_f))
        .stderr(Stdio::from(err))
        .spawn()
        .expect("spawn child")
}

fn wait_port(path: &Path) -> u16 {
    let start = Instant::now();
    loop {
        if let Ok(s) = std::fs::read_to_string(path) {
            if let Ok(p) = s.trim().parse::<u16>() {
                if TcpStream::connect(("127.0.0.1", p)).is_ok() {
                    return p;
                }
            }
        }
        if start.elapsed() > Duration::from_secs(15) {
            panic!("server did not publish a port at {}", path.display());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn header(head: &str, name: &str) -> Option<String> {
    head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.eq_ignore_ascii_case(name) { Some(v.trim().to_string()) } else { None }
    })
}

fn read_until(addr: &str, method: &str, path: &str, body: &str, idle: Duration, stop: impl Fn(&str) -> bool) -> (String, String) {
    let start = Instant::now();
    let mut last = String::new();
    while start.elapsed() < Duration::from_secs(10) {
        match once(addr, method, path, body, idle, &stop) {
            Ok(v) => return v,
            Err(e) => {
                last = e;
                std::thread::sleep(Duration::from_millis(30));
            }
        }
    }
    panic!("request failed: {last}");
}

fn once(addr: &str, method: &str, path: &str, body: &str, idle: Duration, stop: &impl Fn(&str) -> bool) -> Result<(String, String), String> {
    let mut sock = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    sock.set_read_timeout(Some(idle)).map_err(|e| e.to_string())?;
    sock.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccept: text/event-stream\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    let start = Instant::now();
    loop {
        let mut buf = [0u8; 4096];
        match sock.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw);
                if let Some((_, b)) = text.split_once("\r\n\r\n") {
                    let decoded = if text.to_ascii_lowercase().contains("transfer-encoding: chunked") { decode_chunked(b.as_bytes()) } else { b.to_string() };
                    if stop(&decoded) || stop(b) {
                        return Ok((text.split("\r\n\r\n").next().unwrap_or("").to_string(), if decoded.is_empty() { b.to_string() } else { decoded }));
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => break,
            Err(e) => return Err(e.to_string()),
        }
        if start.elapsed() > idle + Duration::from_secs(1) {
            break;
        }
    }
    let text = String::from_utf8_lossy(&raw).into_owned();
    if text.is_empty() {
        return Err("empty response".into());
    }
    let (h, b) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
    let decoded = if h.to_ascii_lowercase().contains("chunked") { decode_chunked(b.as_bytes()) } else { b.to_string() };
    if stop(&decoded) || stop(b) {
        Ok((h.to_string(), decoded))
    } else {
        Err(format!("stop condition not met:\n{decoded}"))
    }
}

fn decode_chunked(mut body: &[u8]) -> String {
    let mut out = Vec::new();
    while !body.is_empty() {
        let Some(nl) = body.windows(2).position(|w| w == b"\r\n") else { break };
        let hex = std::str::from_utf8(&body[..nl]).unwrap_or("0");
        let size = usize::from_str_radix(hex.trim().split(';').next().unwrap_or("0"), 16).unwrap_or(0);
        body = &body[nl + 2..];
        if size == 0 { break; }
        if body.len() < size {
            out.extend_from_slice(body);
            break;
        }
        out.extend_from_slice(&body[..size]);
        body = &body[size..];
        if body.starts_with(b"\r\n") { body = &body[2..]; }
    }
    String::from_utf8_lossy(&out).into_owned()
}
