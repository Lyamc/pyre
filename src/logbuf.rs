//! Tracing setup that tees log lines to stderr and an in-memory ring buffer (for /api/logs).
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static BUF: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();
const MAX: usize = 500;

fn buf() -> &'static Mutex<VecDeque<String>> { BUF.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX))) }

pub fn tail(n: usize) -> Vec<String> {
    let b = buf().lock().unwrap();
    b.iter().skip(b.len().saturating_sub(n)).cloned().collect()
}

struct Tee;
impl Write for Tee {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(data);
        let s = String::from_utf8_lossy(data);
        let mut b = buf().lock().unwrap();
        for line in s.lines().filter(|l| !l.trim().is_empty()) {
            if b.len() >= MAX { b.pop_front(); }
            b.push_back(line.to_string());
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
