//! Logging: console lines stay human-readable (stderr, as before); every line also goes to an in-memory ring
//! (for `GET /api/logs?since=<seq>`) and, as JSON lines, to a rotating file (`PYRE_LOG_FILE`, default
//! `<temp>/pyre/logs/pyre.jsonl`; `off` disables; rotation at `PYRE_LOG_MAX_MB` (16) keeping `PYRE_LOG_KEEP` (3)).
//! Structured events (one per request, warnings) are written with `event(kind, level, fields)`.
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

const MAX: usize = 2000;
static SEQ: AtomicU64 = AtomicU64::new(0);
static BUF: OnceLock<Mutex<VecDeque<(u64, String)>>> = OnceLock::new();
fn buf() -> &'static Mutex<VecDeque<(u64, String)>> { BUF.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX))) }

struct LogFile { path: std::path::PathBuf, f: Option<std::fs::File>, size: u64, max: u64, keep: usize }
static FILE: OnceLock<Option<Mutex<LogFile>>> = OnceLock::new();
fn file() -> Option<&'static Mutex<LogFile>> {
    FILE.get_or_init(|| {
        let p = std::env::var("PYRE_LOG_FILE").ok();
        if p.as_deref() == Some("off") || p.as_deref() == Some("0") { return None; }
        let path = p.map(std::path::PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("pyre").join("logs").join("pyre.jsonl"));
        if let Some(d) = path.parent() { let _ = std::fs::create_dir_all(d); }
        let f = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
        let size = f.as_ref().and_then(|f| f.metadata().ok()).map(|m| m.len()).unwrap_or(0);
        let max = std::env::var("PYRE_LOG_MAX_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(16).max(1) << 20;
        let keep = std::env::var("PYRE_LOG_KEEP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3);
        Some(Mutex::new(LogFile { path, f, size, max, keep }))
    }).as_ref()
}
pub fn file_path() -> Option<String> { file().map(|m| m.lock().unwrap().path.display().to_string()) }

fn write_json(v: &Value) {
    let Some(m) = file() else { return };
    let mut lf = m.lock().unwrap();
    let mut line = v.to_string(); line.push('\n');
    if lf.size + line.len() as u64 > lf.max {
        lf.f = None;
        for i in (1..lf.keep.max(1)).rev() { let a = format!("{}.{}", lf.path.display(), i); let b = format!("{}.{}", lf.path.display(), i + 1); let _ = std::fs::rename(&a, &b); }
        if lf.keep > 0 { let _ = std::fs::rename(&lf.path, format!("{}.1", lf.path.display())); } else { let _ = std::fs::remove_file(&lf.path); }
        lf.f = std::fs::OpenOptions::new().create(true).append(true).open(&lf.path).ok(); lf.size = 0;
    }
    if let Some(f) = lf.f.as_mut() { if f.write_all(line.as_bytes()).is_ok() { lf.size += line.len() as u64; } }
}
fn ts() -> String { chrono_like() }
fn chrono_like() -> String { let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default(); format!("{}.{:03}", d.as_secs(), d.subsec_millis()) }

fn push(line: String) -> u64 {
    let s = SEQ.fetch_add(1, Relaxed) + 1; let mut b = buf().lock().unwrap();
    if b.len() >= MAX { b.pop_front(); } b.push_back((s, line)); s
}

/// One structured event: a JSON line in the file and a compact line in the ring (and on the console for warnings and errors).
pub fn event(kind: &str, level: &str, fields: Value) {
    let mut v = json!({"ts": ts(), "level": level, "kind": kind, "instance": crate::metrics::instance()});
    if let (Some(o), Some(f)) = (v.as_object_mut(), fields.as_object()) { for (k, x) in f { o.insert(k.clone(), x.clone()); } }
    write_json(&v);
    push(format!("{level} {kind}: {}", fields));
    if level != "INFO" && level != "DEBUG" { let _ = writeln!(std::io::stderr(), "{level} {kind}: {fields}"); }
}

pub fn tail(n: usize) -> Vec<String> { let b = buf().lock().unwrap(); b.iter().skip(b.len().saturating_sub(n)).map(|x| x.1.clone()).collect() }
/// Lines with sequence number > `since` (at most `n`), and the last sequence number.
pub fn since(since: u64, n: usize) -> (Vec<Value>, u64) {
    let b = buf().lock().unwrap(); let v: Vec<Value> = b.iter().filter(|x| x.0 > since).take(n).map(|x| json!({"seq": x.0, "line": x.1})).collect();
    (v, SEQ.load(Relaxed))
}

struct Tee;
impl Write for Tee {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(data);
        let s = String::from_utf8_lossy(data);
        for line in s.lines().filter(|l| !l.trim().is_empty()) {
            push(line.to_string());
            // tracing's fmt line: "<time> <LEVEL> <target>: <msg>"
            let mut it = line.split_whitespace(); let _t = it.next(); let lvl = it.next().unwrap_or("INFO");
            let (lvl, rest) = if matches!(lvl, "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR") { (lvl, line.splitn(3, char::is_whitespace).nth(2).unwrap_or("").trim_start()) } else { ("INFO", line) };
            if lvl == "WARN" || lvl == "ERROR" { crate::metrics::warning(); }
            let (target, msg) = match rest.split_once(": ") { Some((a, b)) if !a.contains(' ') => (a, b), _ => ("", rest) };
            write_json(&json!({"ts": ts(), "level": lvl, "kind": "log", "target": target, "msg": msg}));
        }
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { std::io::stderr().flush() }
}

pub fn init() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_|
        "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,cubecl=warn,cubecl_runtime=warn,pliron=warn,cubecl_llvm=warn,xet_client=warn".into());
    tracing_subscriber::fmt().with_env_filter(filter).with_ansi(false).with_writer(|| Tee).init();
}