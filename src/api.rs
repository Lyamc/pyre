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
        .route("/api/metrics", get(metrics))
        .route("/api/logs", get(logs))
        .route("/metrics", get(prom))
        .route("/v1/images/generations", post(images))
        .route("/sdapi/v1/txt2img", post(txt2img))
        .route("/v1/images/edits", post(images))
        .route("/sdapi/v1/img2img", post(txt2img))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/translations", post(translations))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/diarize", post(crate::diar::handle))
        .route("/api/diarizer", get(crate::diar::status))
        .route("/v1/images/tags", post(tags).layer(axum::extract::DefaultBodyLimit::max(64 << 20)))
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
        "limits": crate::limits::status(),
        "metrics": crate::metrics::snapshot(),
        "batching": format!("continuous batching, up to PYRE_PARALLEL slots per LLM"),
        "speculative": crate::spec::status()["speculative"], "route_prior": crate::spec::status()["route_prior"], "fast": crate::spec::fast(), "context": crate::spec::status()["context"],
    }))
}

static SYS: std::sync::Mutex<Option<sysinfo::System>> = std::sync::Mutex::new(None);

/// Primary LAN address (no packets sent: a UDP connect only picks the route).
fn lan_ip() -> Option<String> {
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:80").ok()?;
    s.local_addr().ok().map(|a| a.ip().to_string())
}

/// Live dashboard metrics: requests (active/recent, prefill/decode tok/s, context), VRAM vs
/// reserve, GPU util, RAM, CPU, expert compute split and cache hit rate. See `crate::metrics`.
/// Routes served on a separate `--metrics-bind` listener (read-only observability).
pub fn metrics_routes(r: Router<Arc<App>>) -> Router<Arc<App>> {
    r.route("/metrics", get(prom)).route("/api/metrics", get(metrics)).route("/api/logs", get(logs))
}

/// PYRE_METRICS_TOKEN (optional, never logged): when set, /metrics, /api/metrics and /api/logs need `Authorization: Bearer <token>`.
fn metrics_ok(h: &axum::http::HeaderMap) -> bool {
    static T: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(t) = T.get_or_init(|| std::env::var("PYRE_METRICS_TOKEN").ok().filter(|s| !s.is_empty())) else { return true };
    h.get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).map(|v| v.trim() == t).unwrap_or(false)
}

/// Prometheus text exposition of the metrics registry.
async fn prom(h: axum::http::HeaderMap) -> Response {
    if !metrics_ok(&h) { return err(StatusCode::UNAUTHORIZED, "metrics token required"); }
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], crate::metrics::prometheus()).into_response()
}

async fn metrics(State(app): S, h: axum::http::HeaderMap) -> Response {
    if !metrics_ok(&h) { return err(StatusCode::UNAUTHORIZED, "metrics token required"); }
    metrics_json(app).await.into_response()
}
async fn metrics_json(app: Arc<App>) -> Json<Value> {
    let (cpu, used, total, rss, limits) = tokio::task::spawn_blocking(|| {
        let mut g = SYS.lock().unwrap();
        let sys = g.get_or_insert_with(sysinfo::System::new);
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        let pid = sysinfo::get_current_pid().ok();
        if let Some(p) = pid { sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[p]), true); }
        let rss = pid.and_then(|p| sys.process(p)).map(|p| p.memory()).unwrap_or(0);
        (sys.global_cpu_usage(), sys.used_memory(), sys.total_memory(), rss, crate::limits::status())
    }).await.unwrap_or((0.0, 0, 0, 0, json!({})));
    let mut v = crate::metrics::snapshot();
    v["version"] = json!(env!("CARGO_PKG_VERSION"));
    v["uptime_s"] = json!(app.started.elapsed().as_secs());
    v["lan_ip"] = json!(lan_ip());
    v["limits"] = limits;
    v["cpu_pct"] = json!(cpu);
    v["ram"] = json!({"used": used, "total": total, "process_rss": rss});
    v["models"] = json!(app.loaded().iter().map(|m| json!({"id": m.id, "kind": m.kind.as_str(), "backend": m.backend, "busy": m.busy(), "queued": m.queued(), "meta": m.meta()})).collect::<Vec<_>>());
    Json(v)
}

#[derive(Deserialize)]
struct LogQ { since: Option<u64>, n: Option<usize> }
/// `GET /api/logs` (last 200 lines) or `?since=<seq>&n=<max>` for remote tailing (returns `next` = last seq).
async fn logs(h: axum::http::HeaderMap, q: axum::extract::Query<LogQ>) -> Response {
    if !metrics_ok(&h) { return err(StatusCode::UNAUTHORIZED, "metrics token required"); }
    match q.since { Some(s) => { let (v, last) = crate::logbuf::since(s, q.n.unwrap_or(500).min(2000)); Json(json!({"entries": v, "next": last, "file": crate::logbuf::file_path()})).into_response() }
        None => Json(json!({"lines": crate::logbuf::tail(q.n.unwrap_or(200).min(2000)), "next": crate::logbuf::since(u64::MAX, 0).1, "file": crate::logbuf::file_path()})).into_response() }
}

fn image_model(app: &App, name: Option<&str>) -> Result<Arc<crate::image::ImageModel>, Response> {
    let m = app.find(Kind::Image, name)?;
    match &m.inner {
        crate::registry::Inner::Image(i) => Ok(i.clone()),
        crate::registry::Inner::Placeholder(r) => Err(err(StatusCode::NOT_IMPLEMENTED, r.clone())),
        _ => Err(err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch")),
    }
}

async fn run_images(im: Arc<crate::image::ImageModel>, ps: Vec<crate::image::GenParams>) -> Result<Vec<crate::image::Generated>, Response> {
    use std::sync::atomic::Ordering;
    im.queued.fetch_add(ps.len() as u64, Ordering::SeqCst);
    let r = tokio::task::spawn_blocking(move || {
        let mut out = vec![];
        for (i, p) in ps.iter().enumerate() {
            match im.generate(p) {
                Ok(g) => { tracing::info!("image {}: {}x{} {} steps in {:.2}s ({:.2} it/s), seed {}", i, p.width, p.height, p.steps, g.seconds, p.steps as f64 / g.seconds, g.seed); out.push(g) }
                Err(e) => { im.queued.fetch_sub((ps.len() - i - 1) as u64, Ordering::SeqCst); return Err(e) }
            }
        }
        Ok(out)
    }).await;
    match r { Ok(Ok(v)) => Ok(v), Ok(Err(e)) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))), Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())) }
}

fn seed_of(v: Option<i64>) -> u64 { match v { Some(s) if s >= 0 => s as u64, _ => rand::random::<u32>() as u64 } }

/// Shared request fields for both image endpoints. Extensions beyond OpenAI: negative_prompt,
/// steps, guidance_scale (cfg_scale), seed, sampler (sampler_name), scheduler, refiner,
/// refiner_denoise. Width/height 0 = the model's native size; defaults follow cabin (euler/normal).
fn gen_params(b: &Value, prompt: &str, w: usize, h: usize, steps: Option<u64>, guidance: Option<f64>, seed: u64) -> crate::image::GenParams {
    let s = |k: &str| b[k].as_str().filter(|x| !x.is_empty() && *x != "auto").map(String::from);
    let mut p = crate::image::GenParams::new(prompt, w, h, steps.unwrap_or(0) as usize, guidance.unwrap_or(-1.0), seed);
    p.negative = b["negative_prompt"].as_str().unwrap_or("").into();
    p.sampler = s("sampler").or(s("sampler_name")).map(|x| x.to_ascii_lowercase().replace(' ', "_").replace("dpm++", "dpmpp").replace("_a", "_ancestral").replace("ancestralncestral", "ancestral"));
    p.scheduler = s("scheduler").map(|x| x.to_ascii_lowercase());
    p.refiner = b["refiner"].as_bool().unwrap_or(false);
    p.refiner_denoise = b["refiner_denoise"].as_f64();
    // img2img (`/v1/images/edits` JSON body `image`, or A1111 `init_images[0]`): base64 PNG/JPEG, data URLs too.
    let img = b["image"].as_str().or(b["init_images"][0].as_str());
    if let Some(s) = img {
        use base64::Engine;
        let s = s.split_once("base64,").map(|(_, d)| d).unwrap_or(s);
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s.trim()) { p.init_image = Some(std::sync::Arc::new(bytes)); }
        p.strength = b["strength"].as_f64().or(b["denoising_strength"].as_f64()).unwrap_or(0.75).clamp(0.0, 1.0);
    }
    p
}

/// OpenAI `/v1/images/generations`.
async fn images(State(app): S, Json(b): Json<Value>) -> Response {
    use base64::Engine;
    let im = match image_model(&app, b["model"].as_str()) { Ok(i) => i, Err(r) => return r };
    let Some(prompt) = b["prompt"].as_str() else { return err(StatusCode::BAD_REQUEST, "missing prompt") };
    let (w, h) = match b["size"].as_str().filter(|s| *s != "auto").unwrap_or("0x0").split_once('x').and_then(|(a, c)| Some((a.parse().ok()?, c.parse().ok()?))) { Some(x) => x, None => return err(StatusCode::BAD_REQUEST, "size must look like 512x512") };
    let n = b["n"].as_u64().unwrap_or(1).clamp(1, 8) as usize;
    let seed = seed_of(b["seed"].as_i64());
    let ps = (0..n).map(|i| gen_params(&b, prompt, w, h, b["steps"].as_u64(), b["guidance_scale"].as_f64().or(b["cfg_scale"].as_f64()), seed + i as u64)).collect();
    let url = b["response_format"].as_str() == Some("url");
    match run_images(im, ps).await {
        Err(r) => r,
        Ok(v) => Json(json!({"created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            "data": v.iter().map(|g| { let s = base64::engine::general_purpose::STANDARD.encode(&g.png);
                if url { json!({"url": format!("data:image/png;base64,{s}"), "revised_prompt": prompt}) } else { json!({"b64_json": s, "revised_prompt": prompt}) } }).collect::<Vec<_>>(),
            "timings": v.iter().map(|g| json!({"seconds": g.seconds, "seed": g.seed})).collect::<Vec<_>>()})).into_response(),
    }
}

/// AUTOMATIC1111-compatible `/sdapi/v1/txt2img` (subset: prompt, negative_prompt, steps, cfg_scale, width, height, seed, batch_size, n_iter).
async fn txt2img(State(app): S, Json(b): Json<Value>) -> Response {
    use base64::Engine;
    let im = match image_model(&app, b["model"].as_str().or(b["override_settings"]["sd_model_checkpoint"].as_str())) { Ok(i) => i, Err(r) => return r };
    let n = (b["batch_size"].as_u64().unwrap_or(1) * b["n_iter"].as_u64().unwrap_or(1)).clamp(1, 8) as usize;
    let seed = seed_of(b["seed"].as_i64());
    let p = |i: usize| gen_params(&b, b["prompt"].as_str().unwrap_or(""), b["width"].as_u64().unwrap_or(0) as usize, b["height"].as_u64().unwrap_or(0) as usize, b["steps"].as_u64(), b["cfg_scale"].as_f64(), seed + i as u64);
    match run_images(im, (0..n).map(p).collect()).await {
        Err(r) => r,
        Ok(v) => Json(json!({"images": v.iter().map(|g| base64::engine::general_purpose::STANDARD.encode(&g.png)).collect::<Vec<_>>(),
            "parameters": b, "info": json!({"seeds": v.iter().map(|g| g.seed).collect::<Vec<_>>(), "seconds": v.iter().map(|g| g.seconds).collect::<Vec<_>>(), "sampler_name": b["sampler_name"].as_str().unwrap_or("euler")}).to_string()})).into_response(),
    }
}

async fn speech(State(app): S, body: Option<Json<Value>>) -> Response {
    let Some(Json(b)) = body else { return err(StatusCode::BAD_REQUEST, "expected a JSON body with 'input'") };
    let name = b.get("model").and_then(|m| m.as_str()).map(String::from);
    let m = match app.find(Kind::Tts, name.as_deref()) { Err(r) => return r, Ok(m) => m };
    let t = match &m.inner {
        crate::registry::Inner::Tts(t) => t.clone(),
        crate::registry::Inner::Placeholder(r) => return err(StatusCode::NOT_IMPLEMENTED, r.clone()),
        _ => return err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch"),
    };
    let input = b.get("input").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if input.trim().is_empty() { return err(StatusCode::BAD_REQUEST, "'input' is required"); }
    let fmt = b.get("response_format").and_then(|v| v.as_str()).unwrap_or("wav").to_string();
    if !["wav", "pcm"].contains(&fmt.as_str()) { return err(StatusCode::BAD_REQUEST, format!("response_format '{fmt}' not supported (wav, pcm)")); }
    let speed = b.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let noise = b.get("noise_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let seed = b.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
    t.queued.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let r = tokio::task::spawn_blocking(move || {
        let v = t.vits.lock().unwrap();
        t.queued.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        t.busy.store(true, std::sync::atomic::Ordering::SeqCst);
        let st = std::time::Instant::now();
        let out = v.synth(&input, speed, noise, seed).map(|s| (s, v.cfg.sampling_rate));
        t.busy.store(false, std::sync::atomic::Ordering::SeqCst);
        if let Ok((s, rate)) = &out { tracing::info!("tts: {} chars -> {:.2}s audio in {:.2}s", input.len(), s.len() as f64 / *rate as f64, st.elapsed().as_secs_f64()); }
        out
    }).await;
    match r {
        Ok(Ok((s, rate))) => {
            if fmt == "pcm" {
                let mut v = Vec::with_capacity(s.len() * 2);
                for x in &s { v.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()); }
                ([("content-type", "audio/pcm")], v).into_response()
            } else {
                ([("content-type", "audio/wav")], crate::tts::wav_bytes(&s, rate)).into_response()
            }
        }
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
async fn transcriptions(State(app): S, mp: Multipart) -> Response { crate::stt::handle(app, mp, false).await }
async fn translations(State(app): S, mp: Multipart) -> Response { crate::stt::handle(app, mp, true).await }

/// `POST /v1/images/tags`: tag one or more images with the loaded tagger.
/// JSON: `{"image": "<base64 | data URL>", "images": [...], "threshold": 0.37, "character_threshold": 0.37,
/// "top_k": 50, "watch": ["tag", ...], "rating_threshold": 0.5, "include_probs": false, "model": "..."}` or multipart
/// with one or more `image`/`file` parts plus the same fields as text parts.
async fn tags(State(app): S, req: axum::extract::Request) -> Response {
    use axum::extract::FromRequest;
    use base64::Engine;
    let ct = req.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let mut imgs: Vec<Vec<u8>> = vec![];
    let mut b = json!({});
    if ct.starts_with("multipart/") {
        let mut mp = match Multipart::from_request(req, &app).await { Ok(m) => m, Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()) };
        loop {
            match mp.next_field().await {
                Ok(Some(f)) => {
                    let name = f.name().unwrap_or("").to_string();
                    let Ok(bytes) = f.bytes().await else { return err(StatusCode::BAD_REQUEST, "reading multipart field") };
                    if matches!(name.as_str(), "image" | "file" | "images" | "image[]") { imgs.push(bytes.to_vec()); continue; }
                    let t = String::from_utf8_lossy(&bytes).to_string();
                    b[name.as_str()] = if name == "watch" { json!(t.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).collect::<Vec<_>>()) } else { t.parse::<f64>().map(|v| json!(v)).unwrap_or(json!(t)) };
                }
                Ok(None) => break,
                Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
            }
        }
    } else {
        let Json(j) = match Json::<Value>::from_request(req, &app).await { Ok(j) => j, Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()) };
        let dec = |v: &Value| -> Option<Vec<u8>> {
            let s = v.as_str()?;
            let s = s.split_once(";base64,").map(|x| x.1).unwrap_or(s);
            base64::engine::general_purpose::STANDARD.decode(s.trim()).ok()
        };
        for v in std::iter::once(&j["image"]).chain(j["images"].as_array().into_iter().flatten()) {
            if v.is_null() { continue; }
            match dec(v) { Some(x) => imgs.push(x), None => return err(StatusCode::BAD_REQUEST, "images must be base64 strings or data URLs") }
        }
        b = j;
    }
    if imgs.is_empty() { return err(StatusCode::BAD_REQUEST, "no image given ('image' / 'images' as base64 in JSON, or multipart 'image' parts)"); }
    if imgs.len() > 64 { return err(StatusCode::BAD_REQUEST, "at most 64 images per request"); }
    let m = match app.find(Kind::Tagger, b["model"].as_str()) { Err(r) => return r, Ok(m) => m };
    let t = match &m.inner {
        crate::registry::Inner::Tagger(t) => t.clone(),
        _ => return err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch"),
    };
    let mut o = crate::tagger::TagOpts::default();
    if let Some(v) = b["threshold"].as_f64() { o.threshold = v as f32; }
    if let Some(v) = b["character_threshold"].as_f64() { o.character_threshold = v as f32; }
    if let Some(v) = b["rating_threshold"].as_f64() { o.rating_threshold = v as f32; }
    if let Some(v) = b["top_k"].as_u64() { o.top_k = Some(v as usize); }
    if let Some(w) = b["watch"].as_array() { o.watch.extend(w.iter().filter_map(|x| x.as_str().map(String::from))); }
    let include = b["include_probs"].as_bool().unwrap_or(false);
    let id = m.id.clone();
    t.queued.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let r = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let st = std::time::Instant::now();
        let prepped: anyhow::Result<Vec<Vec<f32>>> = imgs.iter().map(|x| t.prep(x)).collect();
        t.queued.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        let prepped = prepped?;
        let ps = t.probs(prepped);
        let secs = st.elapsed().as_secs_f64();
        let results: Vec<Value> = ps.iter().map(|p| {
            let mut s = t.summarize(p, &o);
            if include { s["probs"] = json!(p); }
            s
        }).collect();
        tracing::info!("tagger: {} image(s) in {secs:.3}s", results.len());
        Ok(json!({"object": "list", "model": id, "threshold": o.threshold, "seconds": secs, "data": results}))
    }).await;
    match r {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
