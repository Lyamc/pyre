//! pyre-tray: notification-area icon for the pyre server.
//! * Status = HTTP GET http://<host>:<port>/health (public route, {"status":"ok"}) every 3 s; host/port from
//!   %ProgramData%\pyre\config.toml [serve] (or PYRE_CONFIG). Icons: full flame = running, medium = starting, embers = stopped.
//! * Start / Stop reuse pyre-launch.ps1 (single source of truth for config -> flags/env and graceful stop-file logic).
//!   Service mode (scheduled task "pyre-server" exists): Start = `schtasks /Run`; Stop = launcher -Stop (stop file + pid),
//!   and if the server is still up afterwards (SYSTEM process, no rights) a message explains how to stop it elevated.
//! * Quit exits the tray only; "Stop server and quit" does both. Single instance via a named mutex.
#![windows_subsystem = "windows"]
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIconBuilder};
use windows_sys::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS};
use windows_sys::Win32::System::Threading::CreateMutexW;
use windows_sys::Win32::UI::Shell::ShellExecuteW;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const TASK: &str = "pyre-server";
const UP: u8 = 1;
const DOWN: u8 = 0;

fn w(s: &str) -> Vec<u16> { s.encode_utf16().chain(Some(0)).collect() }
fn msg(text: &str) { unsafe { MessageBoxW(std::ptr::null_mut(), w(text).as_ptr(), w("pyre").as_ptr(), MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND); } }
fn here() -> PathBuf { std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default() }

struct Cfg { host: String, port: u16 }
fn config_path() -> PathBuf {
    std::env::var_os("PYRE_CONFIG").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("ProgramData").unwrap_or_else(|| "C:\\ProgramData".into())).join("pyre").join("config.toml")
    })
}
fn load_cfg() -> Cfg {
    let v: toml::Value = std::fs::read_to_string(config_path()).ok().and_then(|s| s.parse().ok()).unwrap_or(toml::Value::Table(Default::default()));
    let s = v.get("serve");
    let mut host = s.and_then(|s| s.get("host")).and_then(|h| h.as_str()).unwrap_or("127.0.0.1").to_string();
    if host.is_empty() || host == "0.0.0.0" || host == "::" { host = "127.0.0.1".into(); }
    let port = s.and_then(|s| s.get("port")).and_then(|p| p.as_integer()).unwrap_or(8099) as u16;
    Cfg { host, port }
}
fn url(c: &Cfg) -> String { format!("http://{}:{}", if c.host.contains(':') { format!("[{}]", c.host) } else { c.host.clone() }, c.port) }

/// GET /health with a short timeout; true on HTTP 200.
fn healthy(c: &Cfg) -> bool {
    let Some(addr) = (c.host.as_str(), c.port).to_socket_addrs().ok().and_then(|mut a| a.next()) else { return false };
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(800)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_millis(1500)));
    if s.write_all(format!("GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", c.host).as_bytes()).is_err() { return false; }
    let mut buf = [0u8; 64];
    matches!(s.read(&mut buf), Ok(n) if n >= 12 && buf[..n].starts_with(b"HTTP/1.") && &buf[9..12] == b"200")
}

fn powershell(args: &[&str]) -> Option<std::process::Child> {
    let launch = here().join("pyre-launch.ps1");
    let mut a = vec!["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File"];
    let l = launch.to_string_lossy().to_string();
    a.push(&l); a.extend_from_slice(args);
    Command::new("powershell.exe").args(&a).current_dir(here()).creation_flags(CREATE_NO_WINDOW).spawn().ok()
}
fn service_mode() -> bool {
    Command::new("schtasks.exe").args(["/Query", "/TN", TASK]).creation_flags(CREATE_NO_WINDOW).output().map(|o| o.status.success()).unwrap_or(false)
}
fn start() {
    if service_mode() {
        let ok = Command::new("schtasks.exe").args(["/Run", "/TN", TASK]).creation_flags(CREATE_NO_WINDOW).output().map(|o| o.status.success()).unwrap_or(false);
        if !ok { msg("Could not start the 'pyre-server' scheduled task (needs administrator rights).\nRun 'Start pyre server' as administrator, or: schtasks /Run /TN pyre-server"); }
    } else if powershell(&["-NoFirstRun", "-NoTray"]).is_none() { msg("Could not run pyre-launch.ps1."); }
}
/// Blocking: launcher -Stop does the graceful stop file + grace wait + pid kill.
fn stop(c: &Cfg) {
    if let Some(mut ch) = powershell(&["-Stop"]) { let _ = ch.wait(); }
    std::thread::sleep(Duration::from_millis(500));
    if healthy(c) {
        let svc = if service_mode() { " It is running as the 'pyre-server' scheduled task (SYSTEM)." } else { "" };
        msg(&format!("pyre is still running on port {}.{svc}\nStopping it needs administrator rights: run 'Stop pyre server' as administrator.", c.port));
    }
}
fn open(c: &Cfg) {
    let gui = here().join("pyre-gui.exe");
    let u = url(c);
    if gui.exists() && Command::new(&gui).arg(&u).spawn().is_ok() { return; }
    unsafe { ShellExecuteW(std::ptr::null_mut(), w("open").as_ptr(), w(&u).as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL); }
}

fn main() {
    unsafe {
        let _m = CreateMutexW(std::ptr::null(), 0, w("Local\\pyre-tray-single-instance").as_ptr());
        if GetLastError() == ERROR_ALREADY_EXISTS { return; }
    }
    let ico = |id: u16| Icon::from_resource(id, Some((32, 32))).or_else(|_| Icon::from_resource(id, None)).expect("icon resource");
    let (i_run, i_start, i_stop) = (ico(1), ico(2), ico(3));

    let m_open = MenuItem::new("Open pyre", true, None);
    let m_toggle = MenuItem::new("Start server", true, None);
    let m_web = MenuItem::new("Open web chat", true, None);
    let m_quit = MenuItem::new("Quit tray", true, None);
    let m_stopquit = MenuItem::new("Stop server and quit", true, None);
    let menu = Menu::new();
    let _ = menu.append_items(&[&m_open, &m_web, &m_toggle, &PredefinedMenuItem::separator(), &m_quit, &m_stopquit]);
    let tray = TrayIconBuilder::new().with_menu(Box::new(menu)).with_icon(i_stop.clone()).with_tooltip("pyre: checking...").build().expect("tray");

    // health poller
    let state = Arc::new(AtomicU8::new(DOWN));
    { let st = state.clone(); std::thread::spawn(move || loop { st.store(if healthy(&load_cfg()) { UP } else { DOWN }, Ordering::Relaxed); std::thread::sleep(Duration::from_secs(3)); }); }

    let mut starting: Option<Instant> = None; // waiting for /health after Start (max 180 s)
    let mut shown: (u8, bool) = (255, false);
    let menu_rx = MenuEvent::receiver();
    loop {
        unsafe {
            let mut m: MSG = std::mem::zeroed();
            while PeekMessageW(&mut m, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                if m.message == WM_QUIT { return; }
                TranslateMessage(&m); DispatchMessageW(&m);
            }
        }
        let up = state.load(Ordering::Relaxed);
        if up == UP || starting.is_some_and(|t| t.elapsed() > Duration::from_secs(180)) { starting = None; }
        while let Ok(ev) = menu_rx.try_recv() {
            let c = load_cfg();
            if ev.id == m_open.id() { open(&c); }
            else if ev.id == m_web.id() { unsafe { ShellExecuteW(std::ptr::null_mut(), w("open").as_ptr(), w(&url(&c)).as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL); } }
            else if ev.id == m_toggle.id() {
                if up == UP { stop(&c); state.store(if healthy(&c) { UP } else { DOWN }, Ordering::Relaxed); starting = None; }
                else if starting.is_none() { start(); starting = Some(Instant::now()); }
            }
            else if ev.id == m_quit.id() { return; }
            else if ev.id == m_stopquit.id() { if up == UP { stop(&c); } return; }
        }
        let up = state.load(Ordering::Relaxed);
        let now = (up, starting.is_some());
        if now != shown {
            let port = load_cfg().port;
            let (icon, tip, label) = match now {
                (UP, _) => (&i_run, format!("pyre: running on port {port}"), "Stop server"),
                (_, true) => (&i_start, format!("pyre: starting on port {port}..."), "Starting..."),
                _ => (&i_stop, format!("pyre: stopped (port {port})"), "Start server"),
            };
            let _ = tray.set_icon(Some(icon.clone()));
            let _ = tray.set_tooltip(Some(tip));
            m_toggle.set_text(label);
            m_toggle.set_enabled(!now.1 || up == UP);
            shown = now;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
