//! Companion apps ("pyre app add peal"): programs that use a pyre server, described by a
//! `pyre-app.toml` manifest in their repo (or a built-in registry entry), built/copied into pyre's
//! apps dir and launched with pyre's URL configured.
//!
//! Apps dir: `PYRE_APPS_DIR`, else `%LOCALAPPDATA%\pyre\apps` (`~/.local/share/pyre/apps`).
//! Layout: `<apps>/<name>/bin/*` (installed binaries), `<apps>/<name>/pyre-app.toml` (manifest as
//! installed), `<apps>/<name>/run-<entry>.cmd|.sh` (launchers with the server URL set),
//! `<apps>/installed.json` (records).
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The API version companion apps declare as `min_pyre_api`.
pub const API_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub app: AppInfo,
    #[serde(default)]
    pub source: Source,
    #[serde(default)]
    pub build: Vec<BuildStep>,
    #[serde(default)]
    pub entry: Vec<Entry>,
    #[serde(default)]
    pub pyre: PyreLink,
    #[serde(default)]
    pub models: Vec<ModelReq>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    #[serde(default)] pub description: String,
    #[serde(default)] pub version: String,
    #[serde(default)] pub license: String,
    #[serde(default)] pub homepage: Option<String>,
    /// Optional web UI of the app itself (opened by "Open" in pyre's Apps page).
    #[serde(default)] pub web_ui: Option<String>,
    #[serde(default = "one")] pub min_pyre_api: u32,
}
fn one() -> u32 { 1 }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Source {
    /// Local checkout to build from (preferred when it exists).
    pub path: Option<String>,
    /// Git URL, cloned into `<apps>/<name>/src` when there is no usable path.
    pub git: Option<String>,
    #[serde(default)] pub rev: Option<String>,
}
/// One build step. `cargo` = crate dir (relative to the source root) built with
/// `cargo build --release`, then `bins` are copied into `bin/`. `prebuilt` = a binary path to
/// copy as is. `command` = any shell command run in the source root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStep {
    #[serde(default)] pub cargo: Option<String>,
    #[serde(default)] pub bins: Vec<String>,
    #[serde(default)] pub features: Vec<String>,
    #[serde(default)] pub prebuilt: Option<String>,
    #[serde(default)] pub command: Option<String>,
    /// A failing optional step (e.g. the GUI) does not fail the install.
    #[serde(default)] pub optional: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub bin: String,
    /// cli | gui
    #[serde(default = "cli")] pub kind: String,
    #[serde(default)] pub args: Vec<String>,
    #[serde(default)] pub description: String,
}
fn cli() -> String { "cli".into() }
/// How the app is told where pyre is.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PyreLink {
    /// Env var that receives the server URL (e.g. PEAL_SERVER).
    pub url_env: Option<String>,
    /// Env var that receives the API key, if pyre has one.
    pub api_key_env: Option<String>,
    /// Argument template appended for these entry kinds, `{url}` is replaced (e.g. ["{url}"] for a GUI).
    #[serde(default)] pub gui_args: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReq {
    /// llm | stt | tts | image
    pub kind: String,
    /// Model refs pyre can resolve/pull (first = default).
    pub models: Vec<String>,
    #[serde(default)] pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Installed {
    pub name: String,
    pub version: String,
    pub dir: String,
    pub source: String,
    pub bins: Vec<String>,
    pub server: String,
    pub installed_at: u64,
    #[serde(default)] pub warnings: Vec<String>,
}

const BUILTIN: &[(&str, &str)] = &[
    ("peal", include_str!("../apps/peal.toml")),
    ("cabin", include_str!("../apps/cabin.toml")),
];

static SERVER: OnceLock<String> = OnceLock::new();
/// Called by `serve` so API-launched apps point at this server.
pub fn set_server(url: String) { let _ = SERVER.set(url); }
pub fn server_url() -> String {
    SERVER.get().cloned().or_else(|| std::env::var("PYRE_URL").ok()).unwrap_or_else(|| "http://127.0.0.1:8080".into())
}

pub fn apps_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("PYRE_APPS_DIR") { return PathBuf::from(p); }
    if let Some(p) = std::env::var_os("LOCALAPPDATA") { return PathBuf::from(p).join("pyre").join("apps"); }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into())).join(".local/share/pyre/apps")
}

pub fn builtin() -> Vec<Manifest> {
    BUILTIN.iter().filter_map(|(n, t)| toml::from_str(t).map_err(|e| tracing::warn!("builtin app {n}: {e}")).ok()).collect()
}

/// `name` (built-in), a folder with pyre-app.toml, or a path to a manifest file.
pub fn resolve(spec: &str) -> Result<Manifest> {
    let p = Path::new(spec);
    let file = if p.is_dir() { Some(p.join("pyre-app.toml")) } else if p.is_file() { Some(p.to_path_buf()) } else { None };
    if let Some(f) = file {
        let mut m: Manifest = toml::from_str(&std::fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))?)
            .with_context(|| format!("parse {}", f.display()))?;
        if m.source.path.is_none() && m.source.git.is_none() { m.source.path = Some(f.parent().unwrap().display().to_string()); }
        return Ok(m);
    }
    let mut m = builtin().into_iter().find(|m| m.app.name.eq_ignore_ascii_case(spec))
        .with_context(|| format!("unknown app '{spec}' (built-in: {})", BUILTIN.iter().map(|b| b.0).collect::<Vec<_>>().join(", ")))?;
    // a manifest in the app's own checkout wins over the built-in copy
    if let Some(sp) = &m.source.path {
        let f = Path::new(sp).join("pyre-app.toml");
        if let Ok(t) = std::fs::read_to_string(&f) {
            match toml::from_str::<Manifest>(&t) { Ok(mut own) => { own.source.path.get_or_insert(sp.clone()); m = own; } Err(e) => tracing::warn!("{}: {e}", f.display()) }
        }
    }
    Ok(m)
}

fn records_path() -> PathBuf { apps_dir().join("installed.json") }
pub fn installed() -> Vec<Installed> {
    std::fs::read_to_string(records_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}
fn save_records(v: &[Installed]) -> Result<()> {
    std::fs::create_dir_all(apps_dir())?;
    std::fs::write(records_path(), serde_json::to_string_pretty(v)?)?;
    Ok(())
}

fn exe(name: &str) -> String { if cfg!(windows) && !name.ends_with(".exe") { format!("{name}.exe") } else { name.to_string() } }

fn cargo_cmd() -> Command {
    // On Windows prefer the repo's MSVC wrapper when present (sets up link.exe), else plain cargo.
    let w = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts").join("cargo.cmd");
    if cfg!(windows) && w.exists() { let mut c = Command::new("cmd"); c.arg("/c").arg(w); c } else { Command::new("cargo") }
}

/// Build/copy the app into the apps dir and write launchers. `log` receives progress lines.
pub fn install(m: &Manifest, server: &str, log: &mut dyn FnMut(String)) -> Result<Installed> {
    if m.app.min_pyre_api > API_VERSION { bail!("{} needs pyre API {} (this pyre: {API_VERSION})", m.app.name, m.app.min_pyre_api); }
    let name = &m.app.name;
    let dir = apps_dir().join(name);
    let bin_dir = dir.join("bin");
    std::fs::create_dir_all(&bin_dir)?;
    let src: PathBuf = match (&m.source.path, &m.source.git) {
        (Some(p), _) if Path::new(p).exists() => PathBuf::from(p),
        (_, Some(g)) => {
            let s = dir.join("src");
            if s.join(".git").exists() {
                log(format!("git pull in {}", s.display()));
                run(Command::new("git").arg("-C").arg(&s).arg("pull").arg("--ff-only"), log)?;
            } else {
                log(format!("git clone {g}"));
                run(Command::new("git").arg("clone").arg(g).arg(&s), log)?;
            }
            if let Some(r) = &m.source.rev { run(Command::new("git").arg("-C").arg(&s).arg("checkout").arg(r), log)?; }
            s
        }
        (Some(p), None) => bail!("source path {p} does not exist and no git URL is given"),
        (None, None) => bail!("manifest has no source"),
    };
    let target = dir.join("target");
    let mut bins = vec![];
    let mut warnings = vec![];
    for (i, step) in m.build.iter().enumerate() {
        let r = (|| -> Result<Vec<String>> {
            let mut got = vec![];
            if let Some(c) = &step.command {
                log(format!("[{i}] {c}"));
                let mut cmd = if cfg!(windows) { let mut x = Command::new("cmd"); x.arg("/c").arg(c); x } else { let mut x = Command::new("sh"); x.arg("-c").arg(c); x };
                run(cmd.current_dir(&src), log)?;
            }
            if let Some(crate_dir) = &step.cargo {
                let cd = src.join(crate_dir);
                log(format!("[{i}] cargo build --release in {}", cd.display()));
                let mut cmd = cargo_cmd();
                cmd.arg("build").arg("--release").current_dir(&cd).env("CARGO_TARGET_DIR", &target);
                for b in &step.bins { cmd.arg("--bin").arg(b); }
                if !step.features.is_empty() { cmd.arg("--features").arg(step.features.join(",")); }
                run(&mut cmd, log)?;
                for b in &step.bins {
                    let e = exe(b);
                    let cands = [target.join("release").join(&e), target.join("x86_64-pc-windows-msvc").join("release").join(&e)];
                    let f = cands.iter().filter(|p| p.exists()).max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok())
                        .with_context(|| format!("built {b} not found under {}", target.display()))?;
                    std::fs::copy(f, bin_dir.join(&e))?;
                    got.push(e);
                }
            }
            if let Some(p) = &step.prebuilt {
                let f = src.join(p);
                let n = f.file_name().context("prebuilt name")?.to_owned();
                std::fs::copy(&f, bin_dir.join(&n)).with_context(|| format!("copy {}", f.display()))?;
                got.push(n.to_string_lossy().into());
            }
            Ok(got)
        })();
        match r {
            Ok(b) => { for x in &b { log(format!("installed {}", bin_dir.join(x).display())); } bins.extend(b) }
            Err(e) if step.optional => { let w = format!("optional step {i} failed: {e:#}"); log(w.clone()); warnings.push(w); }
            Err(e) => return Err(e),
        }
    }
    std::fs::write(dir.join("pyre-app.toml"), toml::to_string_pretty(m)?)?;
    write_launchers(m, &dir, server)?;
    let rec = Installed { name: name.clone(), version: m.app.version.clone(), dir: dir.display().to_string(), source: src.display().to_string(),
        bins, server: server.to_string(), installed_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(), warnings };
    let mut all = installed();
    all.retain(|r| r.name != *name);
    all.push(rec.clone());
    save_records(&all)?;
    log(format!("{name} installed in {} (server {server})", dir.display()));
    Ok(rec)
}

fn write_launchers(m: &Manifest, dir: &Path, server: &str) -> Result<()> {
    for e in &m.entry {
        let b = dir.join("bin").join(exe(&e.bin));
        let mut args = e.args.clone();
        if e.kind == "gui" { args.extend(m.pyre.gui_args.iter().map(|a| a.replace("{url}", server))); }
        if cfg!(windows) {
            let mut s = String::from("@echo off\r\n");
            if let Some(v) = &m.pyre.url_env { s += &format!("if not defined {v} set {v}={server}\r\n"); }
            s += &format!("\"{}\" {} %*\r\n", b.display(), args.join(" "));
            std::fs::write(dir.join(format!("{}.cmd", e.name)), s)?;
        } else {
            let mut s = String::from("#!/bin/sh\n");
            if let Some(v) = &m.pyre.url_env { s += &format!(": \"${{{v}:={server}}}\"; export {v}\n"); }
            s += &format!("exec \"{}\" {} \"$@\"\n", b.display(), args.join(" "));
            let p = dir.join(&e.name);
            std::fs::write(&p, s)?;
            #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))?; }
        }
    }
    Ok(())
}

fn run(cmd: &mut Command, log: &mut dyn FnMut(String)) -> Result<()> {
    let out = cmd.output().with_context(|| format!("spawn {cmd:?}"))?;
    let text = String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    for l in text.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev() { log(format!("  {l}")); }
    if !out.status.success() { bail!("{:?} failed ({})", cmd.get_program(), out.status); }
    Ok(())
}

pub fn remove(name: &str) -> Result<()> {
    let mut all = installed();
    let rec = all.iter().find(|r| r.name == name).cloned().with_context(|| format!("{name} is not installed"))?;
    let d = PathBuf::from(&rec.dir);
    // only ever delete inside the apps dir
    if d.starts_with(apps_dir()) && d.exists() { std::fs::remove_dir_all(&d).with_context(|| format!("remove {}", d.display()))?; }
    all.retain(|r| r.name != name);
    save_records(&all)
}

/// Start an installed app's entry (default: the first). Returns the child pid.
pub fn launch(name: &str, entry: Option<&str>, extra: &[String], server: &str, api_key: Option<&str>, wait: bool) -> Result<u32> {
    let rec = installed().into_iter().find(|r| r.name == name).with_context(|| format!("{name} is not installed (pyre app install {name})"))?;
    let m: Manifest = toml::from_str(&std::fs::read_to_string(Path::new(&rec.dir).join("pyre-app.toml"))?)?;
    let e = match entry { Some(n) => m.entry.iter().find(|e| e.name == n || e.bin == n), None => m.entry.first() }
        .with_context(|| format!("{name}: no entry {}", entry.unwrap_or("")))?;
    let b = Path::new(&rec.dir).join("bin").join(exe(&e.bin));
    if !b.exists() { bail!("{} is missing (re-run pyre app install {name})", b.display()); }
    let mut c = Command::new(&b);
    c.args(&e.args);
    if e.kind == "gui" { c.args(m.pyre.gui_args.iter().map(|a| a.replace("{url}", server))); }
    c.args(extra);
    if let Some(v) = &m.pyre.url_env { c.env(v, server); }
    if let (Some(v), Some(k)) = (&m.pyre.api_key_env, api_key) { c.env(v, k); }
    c.env("PYRE_URL", server);
    if wait {
        let st = c.status()?;
        if !st.success() { bail!("{} exited with {st}", e.bin); }
        Ok(0)
    } else { Ok(c.spawn()?.id()) }
}

/// Catalog for the CLI / API / web UI: built-ins + installed, with install state.
pub fn catalog() -> serde_json::Value {
    let inst = installed();
    let mut names: Vec<String> = builtin().into_iter().map(|m| m.app.name).collect();
    for r in &inst { if !names.contains(&r.name) { names.push(r.name.clone()); } }
    let apps: Vec<serde_json::Value> = names.iter().map(|n| {
        let rec = inst.iter().find(|r| &r.name == n);
        let m = match rec { Some(r) => std::fs::read_to_string(Path::new(&r.dir).join("pyre-app.toml")).ok().and_then(|t| toml::from_str::<Manifest>(&t).ok()), None => None }
            .or_else(|| resolve(n).ok());
        serde_json::json!({ "name": n, "manifest": m, "installed": rec, "builtin": BUILTIN.iter().any(|b| b.0 == n) })
    }).collect();
    serde_json::json!({ "apps_dir": apps_dir(), "api_version": API_VERSION, "server": server_url(), "apps": apps })
}

// ---------------- CLI ----------------
#[derive(clap::Subcommand, Debug)]
pub enum AppCmd {
    /// Known apps (built-in registry + installed)
    List { #[arg(long)] json: bool },
    /// Install (build/copy) an app: built-in name, folder with pyre-app.toml, or manifest path. Alias: add
    #[command(alias = "add")]
    Install {
        app: String,
        /// pyre URL the app is configured with (default PYRE_URL or http://127.0.0.1:8080)
        #[arg(long)] server: Option<String>,
        /// Also pull the app's default models into the HF cache
        #[arg(long)] pull_models: bool,
    },
    /// Rebuild an installed app from its source
    Update { app: String, #[arg(long)] server: Option<String> },
    /// Uninstall (deletes only <apps dir>/<name>)
    #[command(alias = "rm", alias = "uninstall")]
    Remove { app: String },
    /// Run an installed app; arguments after `--` go to the app
    Run {
        app: String,
        #[arg(long)] entry: Option<String>,
        #[arg(long)] server: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)] args: Vec<String>,
    },
    /// Show an app's manifest
    Info { app: String },
}

pub fn run_cli(ctx: &crate::storage::Ctx, cmd: AppCmd) -> Result<()> {
    let srv = |s: Option<String>| s.unwrap_or_else(server_url);
    match cmd {
        AppCmd::List { json } => {
            let c = catalog();
            if json { println!("{}", serde_json::to_string_pretty(&c)?); return Ok(()); }
            println!("apps dir: {}", apps_dir().display());
            for a in c["apps"].as_array().unwrap() {
                let st = if a["installed"].is_null() { "available".to_string() } else { format!("installed ({})", a["installed"]["bins"].as_array().map(|b| b.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default()) };
                println!("{:<10} {:<40} {}", a["name"].as_str().unwrap_or(""), st, a["manifest"]["app"]["description"].as_str().unwrap_or(""));
            }
        }
        AppCmd::Install { app, server, pull_models } => {
            let m = resolve(&app)?;
            let rec = install(&m, &srv(server), &mut |l| println!("{l}"))?;
            for w in &rec.warnings { eprintln!("warning: {w}"); }
            for r in &m.models {
                let def = r.models.first().cloned().unwrap_or_default();
                if pull_models && !def.is_empty() && r.kind == "llm" {
                    println!("pulling {def} ...");
                    if let Err(e) = crate::cli_models::run(ctx, crate::ModelsCmd::Pull { repo: def.clone(), store: "hf".into(), kind: Some(r.kind.clone()) }) { eprintln!("pull {def}: {e:#}"); }
                } else {
                    println!("needs a {} model{}: {} (load in pyre: --load {}={def}{})", r.kind, if r.optional { " (optional)" } else { "" }, r.models.join(", "), r.kind,
                        if r.kind == "llm" { ", or re-run with --pull-models" } else { "" });
                }
            }
            for e in &m.entry { println!("run: pyre app run {} --entry {}   (or {})", m.app.name, e.name, Path::new(&rec.dir).join(if cfg!(windows) { format!("{}.cmd", e.name) } else { e.name.clone() }).display()); }
        }
        AppCmd::Update { app, server } => {
            let rec = installed().into_iter().find(|r| r.name == app).with_context(|| format!("{app} is not installed"))?;
            let m = resolve(&app)?;
            install(&m, &server.unwrap_or(rec.server), &mut |l| println!("{l}"))?;
        }
        AppCmd::Remove { app } => { remove(&app)?; println!("removed {app}"); }
        AppCmd::Run { app, entry, server, args } => {
            let key = std::env::var("PYRE_API_KEY").ok();
            launch(&app, entry.as_deref(), &args, &srv(server), key.as_deref(), true)?;
        }
        AppCmd::Info { app } => { println!("{}", toml::to_string_pretty(&resolve(&app)?)?); }
    }
    Ok(())
}

// ---------------- HTTP API (/api/apps*) ----------------
use axum::{extract::State, response::{IntoResponse, Response}, routing::{get, post}, Json, Router};
use std::sync::Arc;
type St = State<Arc<crate::registry::App>>;

pub fn routes(r: Router<Arc<crate::registry::App>>) -> Router<Arc<crate::registry::App>> {
    r.route("/api/apps", get(|| async { Json(catalog()) }))
        .route("/api/apps/install", post(api_install))
        .route("/api/apps/remove", post(api_remove))
        .route("/api/apps/run", post(api_run))
}
#[derive(Deserialize)]
struct AppBody { name: String, #[serde(default)] entry: Option<String>, #[serde(default)] args: Vec<String>, #[serde(default)] server: Option<String> }
fn bad(e: anyhow::Error) -> Response { crate::server::err(axum::http::StatusCode::BAD_REQUEST, format!("{e:#}")) }
async fn api_install(Json(b): Json<AppBody>) -> Response {
    let r = tokio::task::spawn_blocking(move || {
        let mut log = vec![];
        let m = resolve(&b.name)?;
        let rec = install(&m, &b.server.unwrap_or_else(server_url), &mut |l| { tracing::info!("app {}: {l}", m.app.name); log.push(l) })?;
        anyhow::Ok(serde_json::json!({ "installed": rec, "log": log, "models": m.models }))
    }).await;
    match r { Ok(Ok(v)) => Json(v).into_response(), Ok(Err(e)) => bad(e), Err(e) => bad(e.into()) }
}
async fn api_remove(Json(b): Json<AppBody>) -> Response {
    match remove(&b.name) { Ok(()) => Json(serde_json::json!({"removed": b.name})).into_response(), Err(e) => bad(e) }
}
async fn api_run(State(app): St, Json(b): Json<AppBody>) -> Response {
    // Launches on the machine running pyre (e.g. a GUI app on this desktop).
    match launch(&b.name, b.entry.as_deref(), &b.args, &b.server.unwrap_or_else(server_url), app.api_key.as_deref(), false) {
        Ok(pid) => Json(serde_json::json!({"pid": pid})).into_response(), Err(e) => bad(e) }
}
