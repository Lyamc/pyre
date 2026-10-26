//! `/api/*` management endpoints for the web UI (mirror `pyre models ...`), status, and
//! OpenAI endpoints for modalities whose backends are not wired yet (501 with a clear message).
use crate::registry::{App, Kind, LoadReq};
use crate::server::err;
use crate::store::{hfcache, store::Mode};
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;

type S = State<Arc<App>>;

pub fn routes(r: Router<Arc<App>>) -> Router<Arc<App>> {
    r.route("/api/models", get(list))
        .route("/api/models/load", post(load))
        .route("/api/models/unload", post(unload))
        .route("/api/models/import", post(import))
        .route("/api/models/alias", post(alias))
        .route("/api/models/rm", post(rm))
        .route("/api/pull", post(pull))
        .route("/api/status", get(status))
        .route("/api/logs", get(logs))
        .route("/v1/images/generations", post(images))
        .route("/sdapi/v1/txt2img", post(images))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/translations", post(translations))
        .route("/v1/audio/speech", post(speech))
        .merge(crate::ui::routes())
}

fn bad(e: anyhow::Error) -> Response { err(StatusCode::BAD_REQUEST, format!("{e:#}")) }

async fn list(State(app): S) -> Json<Value> {
    let a = app.clone();
    let mut v = tokio::task::spawn_blocking(move || crate::cli_models::list_json(&a.storage)).await.unwrap_or(json!({}));
    v["loaded"] = json!(app.loaded().iter().map(|m| json!({"id": m.id, "kind": m.kind.as_str(), "backend": m.backend, "path": m.path, "busy": m.busy(), "meta": m.meta()})).collect::<Vec<_>>());
    v["offline"] = json!(app.storage.offline);
    Json(v)
}

#[derive(Deserialize)]
struct LoadBody { model: String, kind: Option<String>, alias: Option<String>, backend: Option<String>, dtype: Option<String>, ctx_size: Option<usize> }

async fn load(State(app): S, Json(b): Json<LoadBody>) -> Response {
    let kind = match b.kind.as_deref() { Some(k) => match Kind::parse(k) { Some(k) => k, None => return err(StatusCode::BAD_REQUEST, format!("unknown kind {k}")) }, None => Kind::Llm };
    let a = app.clone();
    let r = tokio::task::spawn_blocking(move || {
        let m = crate::registry::load(&a.storage, &a.defaults, LoadReq { kind, r: &b.model, alias: b.alias, backend: b.backend, dtype: b.dtype, ctx_size: b.ctx_size })?;
        let id = m.id.clone();
        a.insert(m)?;
        anyhow::Ok(id)
    }).await;
    match r {
        Ok(Ok(id)) => Json(json!({"ok": true, "id": id})).into_response(),
        Ok(Err(e)) => bad(e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("load panicked: {e}")),
    }
}

#[derive(Deserialize)]
struct IdBody { id: String }
async fn unload(State(app): S, Json(b): Json<IdBody>) -> Response {
    if app.unload(&b.id) { Json(json!({"ok": true})).into_response() } else { err(StatusCode::NOT_FOUND, format!("{} is not loaded", b.id)) }
}

#[derive(Deserialize)]
struct ImportBody { source: String, #[serde(default)] mode: Option<String>, name: Option<String>, kind: Option<String> }
async fn import(State(app): S, Json(b): Json<ImportBody>) -> Response {
    let a = app.clone();
    let r = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let kind = crate::cli_models::parse_kind(&b.kind)?;
        let p = std::path::Path::new(&b.source);
        let mode = b.mode.as_deref().unwrap_or("link");
        if mode == "register" {
            let e = a.storage.store.register(p, b.name.as_deref(), kind)?;
            return Ok(json!({"ok": true, "name": e.manifest.name, "dir": e.dir}));
        }
        let m = match mode { "copy" => Mode::Copy, "move" => Mode::Move, _ => Mode::Link };
        let (src, repo, rev) = if p.is_dir() { (p.to_path_buf(), None, None) } else {
            let (r, rev) = hfcache::split_rev(&b.source);
            (hfcache::find(&a.storage.hf_cache, r, rev).ok_or_else(|| anyhow::anyhow!("{} not in HF cache; pull it first", b.source))?, Some(r.to_string()), rev.map(String::from))
        };
        let name = b.name.or(repo.clone()).unwrap_or_else(|| format!("local/{}", src.file_name().unwrap().to_string_lossy()));
        let (e, notes) = a.storage.store.import(&src, &name, repo.as_deref(), rev.as_deref(), kind, m)?;
        Ok(json!({"ok": true, "name": name, "dir": e.dir, "notes": notes}))
    }).await;
    match r { Ok(Ok(v)) => Json(v).into_response(), Ok(Err(e)) => bad(e), Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()) }
}

#[derive(Deserialize)]
struct AliasBody { name: String, alias: String }
async fn alias(State(app): S, Json(b): Json<AliasBody>) -> Response {
    let a = app.clone();
    let r = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        if a.storage.store.find(&b.name).is_none() {
            let (r, rev) = hfcache::split_rev(&b.name);
            let snap = hfcache::find(&a.storage.hf_cache, r, rev).ok_or_else(|| anyhow::anyhow!("{} not found", b.name))?;
            a.storage.store.import(&snap, r, Some(r), rev, None, Mode::Link)?;
            a.storage.store.set_alias(r, &b.alias)?;
        } else { a.storage.store.set_alias(&b.name, &b.alias)?; }
        Ok(())
    }).await;
    match r { Ok(Ok(())) => Json(json!({"ok": true})).into_response(), Ok(Err(e)) => bad(e), Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()) }
}

#[derive(Deserialize)]
struct RmBody { name: String, #[serde(default)] from_hf_cache: bool, #[serde(default)] confirm: bool }
async fn rm(State(app): S, Json(b): Json<RmBody>) -> Response {
    if !b.confirm { return err(StatusCode::BAD_REQUEST, "set \"confirm\": true to delete"); }
    let a = app.clone();
    let r = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let mut removed = vec![];
        if let Some(e) = a.storage.store.find(&b.name) { a.storage.store.remove(&e)?; removed.push("store"); }
        if b.from_hf_cache {
            let p = a.storage.hf_cache.join(hfcache::repo_folder(hfcache::split_rev(&b.name).0));
            if p.exists() { std::fs::remove_dir_all(&p)?; removed.push("hf-cache"); }
        }
        if removed.is_empty() { anyhow::bail!("{} not found{}", b.name, if b.from_hf_cache { "" } else { " in store (HF-cache models are only removed with from_hf_cache)" }); }
        Ok(json!({"ok": true, "removed": removed}))
    }).await;
    match r { Ok(Ok(v)) => Json(v).into_response(), Ok(Err(e)) => bad(e), Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()) }
}

#[derive(Deserialize)]
struct PullBody { repo: String, #[serde(default)] store: Option<String> }
/// SSE progress: {"file","index","total"} per file, then {"done":true,"path"} or {"error"}.
async fn pull(State(app): S, Json(b): Json<PullBody>) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let a = app.clone();
    tokio::task::spawn_blocking(move || {
        let (r, rev) = hfcache::split_rev(&b.repo);
        let res = (|| -> anyhow::Result<std::path::PathBuf> {
            if a.storage.offline { anyhow::bail!("server is in offline mode"); }
            let p = match hfcache::find(&a.storage.hf_cache, r, rev) {
                Some(p) => { let _ = tx.send(json!({"file": "already cached", "index": 1, "total": 1})); hfcache::dedupe_snapshot(&p); p }
                None => hfcache::download(&a.storage.hf_cache, r, rev, &|f, i, n| { let _ = tx.send(json!({"file": f, "index": i, "total": n})); })?,
            };
            if b.store.as_deref() == Some("pyre") && a.storage.store.find(r).is_none() { a.storage.store.import(&p, r, Some(r), rev, None, Mode::Link)?; }
            Ok(p)
        })();
        let _ = tx.send(match res { Ok(p) => json!({"done": true, "path": p}), Err(e) => json!({"error": format!("{e:#}")}) });
    });
    let s = futures::StreamExt::map(tokio_stream::wrappers::UnboundedReceiverStream::new(rx), |v| Ok::<_, Infallible>(SseEvent::default().data(v.to_string())));
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

fn nvidia() -> Value {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version", "--format=csv,noheader,nounits"]).output();
    match out {
        Ok(o) if o.status.success() => json!(String::from_utf8_lossy(&o.stdout).lines().map(|l| {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            json!({"name": f.first(), "memory_used_mib": f.get(1).and_then(|x| x.parse::<u64>().ok()), "memory_total_mib": f.get(2).and_then(|x| x.parse::<u64>().ok()),
                "utilization_pct": f.get(3).and_then(|x| x.parse::<u64>().ok()), "driver": f.get(4)})
        }).collect::<Vec<_>>()),
        _ => json!([]),
    }
}

async fn status(State(app): S) -> Json<Value> {
    let gpus = tokio::task::spawn_blocking(nvidia).await.unwrap_or(json!([]));
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let pid = sysinfo::get_current_pid().ok();
    if let Some(p) = pid { sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[p]), true); }
    let rss = pid.and_then(|p| sys.process(p)).map(|p| p.memory()).unwrap_or(0);
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"), "burn": "0.22", "uptime_s": app.started.elapsed().as_secs(),
        "default_backend": app.defaults.backend, "backends_compiled": crate::backend::enabled(),
        "auth": app.api_key.is_some(), "offline": app.storage.offline,
        "gpus": gpus, "ram": {"process_rss": rss, "total": sys.total_memory(), "used": sys.used_memory()},
        "models": app.loaded().iter().map(|m| json!({"id": m.id, "kind": m.kind.as_str(), "backend": m.backend, "busy": m.busy(), "queued": m.queued(), "meta": m.meta()})).collect::<Vec<_>>(),
        "store_root": app.storage.store.root, "hf_cache": app.storage.hf_cache,
        "batching": "single worker per model; continuous batching is TODO",
    }))
}

async fn logs() -> Json<Value> { Json(json!({"lines": crate::logbuf::tail(200)})) }

async fn images(State(app): S, body: Option<Json<Value>>) -> Response {
    let name = body.as_ref().and_then(|b| b.get("model")).and_then(|m| m.as_str()).map(String::from);
    match app.find(Kind::Image, name.as_deref()) {
        Err(r) => r,
        Ok(m) => match &m.inner {
            crate::registry::Inner::Placeholder(r) => err(StatusCode::NOT_IMPLEMENTED, r.clone()),
            _ => err(StatusCode::NOT_IMPLEMENTED, "image generation backend not available"),
        },
    }
}

async fn speech(State(app): S, body: Option<Json<Value>>) -> Response {
    let name = body.as_ref().and_then(|b| b.get("model")).and_then(|m| m.as_str()).map(String::from);
    match app.find(Kind::Tts, name.as_deref()) {
        Err(r) => r,
        Ok(m) => match &m.inner {
            crate::registry::Inner::Placeholder(r) => err(StatusCode::NOT_IMPLEMENTED, r.clone()),
            _ => err(StatusCode::NOT_IMPLEMENTED, "text-to-speech backend not available"),
        },
    }
}

async fn transcriptions(State(app): S, mp: Multipart) -> Response { crate::stt::handle(app, mp, false).await }
async fn translations(State(app): S, mp: Multipart) -> Response { crate::stt::handle(app, mp, true).await }
