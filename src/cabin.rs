//! Cabin workflows (web UI "Cabin" tab): one-click image tasks on top of pyre's own models.
//!
//! Models load on demand and unload after `PYRE_CABIN_IDLE_S` (default 120 s) without use. A GPU model first
//! takes a VRAM lease (`limits::lease`): the Qwen-Next expert cache shrinks so the device keeps >= the reserve
//! (1 GiB+) free, and regrows on its normal schedule once the lease ends. Small CPU models skip the lease.
use crate::registry::{App, Kind, LoadReq, Loaded};
use crate::server::err;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const SD15: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
pub const TAGGER: &str = "Redstonexs/kagami-24k";
/// SD 1.5 f16: ~2.1 GB weights + activations up to ~768x768 (measured peaks in cabin) + VAE decode.
const SD15_LEASE: u64 = 3_300 << 20;
const STYLES: &str = include_str!("../ui/cabin_styles.json");

type S = State<Arc<App>>;

struct Held { id: String, lease: u64, last: Instant, gpu: bool }
fn held() -> &'static Mutex<Vec<Held>> { static H: OnceLock<Mutex<Vec<Held>>> = OnceLock::new(); H.get_or_init(Default::default) }
static LOAD_LOCK: Mutex<()> = Mutex::new(());
fn idle() -> Duration { Duration::from_secs(std::env::var("PYRE_CABIN_IDLE_S").ok().and_then(|v| v.parse().ok()).unwrap_or(120)) }

fn find(app: &App, kind: Kind) -> Option<Arc<Loaded>> { app.loaded().into_iter().find(|m| m.kind == kind) }
fn touch(id: &str) { if let Some(h) = held().lock().unwrap().iter_mut().find(|h| h.id == id) { h.last = Instant::now(); } }

/// The loaded model of `kind`, loading `r` on `backend` first if none is (with a VRAM lease on GPU).
pub async fn ensure(app: &Arc<App>, kind: Kind, r: &str, backend: &str, lease: u64) -> Result<Arc<Loaded>, Response> {
    if let Some(m) = find(app, kind) { touch(&m.id); return Ok(m); }
    let (app2, r2, b2) = (app.clone(), r.to_string(), backend.to_string());
    let res = tokio::task::spawn_blocking(move || -> anyhow::Result<Arc<Loaded>> {
        let _g = LOAD_LOCK.lock().unwrap();
        if let Some(m) = find(&app2, kind) { return Ok(m); }
        let gpu = !matches!(b2.as_str(), "cpu" | "flex" | "ndarray");
        let lease = if gpu { lease } else { 0 };
        let v0 = crate::limits::vram().map(|v| v.free >> 20);
        if lease > 0 { crate::limits::lease(lease, &format!("loading {r2}"))?; }
        let t = Instant::now();
        let m = match crate::registry::load(&app2.storage, &app2.defaults, LoadReq { kind, r: &r2, alias: None, backend: Some(b2.clone()), dtype: None, ctx_size: None }) {
            Ok(m) => m,
            Err(e) => { if gpu { cleanup(); } if lease > 0 { crate::limits::lease_end(lease); } return Err(e); }
        };
        let id = m.id.clone();
        if let Err(e) = app2.insert(m) { if lease > 0 { crate::limits::lease_end(lease); } return Err(e); }
        tracing::info!("cabin: loaded {} '{id}' on {b2} in {:.1}s (VRAM free {:?} -> {:?} MiB, lease {} MiB)", kind.as_str(), t.elapsed().as_secs_f64(), v0, crate::limits::vram().map(|v| v.free >> 20), lease >> 20);
        held().lock().unwrap().push(Held { id, lease, last: Instant::now(), gpu });
        reaper_start(app2.clone());
        find(&app2, kind).ok_or_else(|| anyhow::anyhow!("model loaded but not registered"))
    }).await;
    match res {
        Ok(Ok(m)) => Ok(m),
        Ok(Err(e)) => Err(err(StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}"))),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

fn reaper_start(app: Arc<App>) {
    static R: std::sync::Once = std::sync::Once::new();
    R.call_once(move || { let _ = std::thread::Builder::new().name("cabin-reaper".into()).spawn(move || loop {
        std::thread::sleep(Duration::from_secs(5));
        let mut drop_ids = vec![];
        {
            let mut h = held().lock().unwrap();
            let live = app.loaded();
            h.retain(|x| { let alive = live.iter().any(|m| m.id == x.id); if !alive { crate::limits::lease_end(x.lease); } alive });
            for x in h.iter_mut() {
                let m = live.iter().find(|m| m.id == x.id).unwrap();
                if m.busy() || m.queued() > 0 { x.last = Instant::now(); continue; }
                if x.last.elapsed() > idle() { drop_ids.push((x.id.clone(), x.lease, x.gpu)); }
            }
            h.retain(|x| !drop_ids.iter().any(|d| d.0 == x.id));
        }
        for (id, lease, gpu) in drop_ids {
            app.unload(&id);
            if gpu { std::thread::sleep(Duration::from_millis(500)); cleanup(); }
            crate::limits::lease_end(lease);
            tracing::info!("cabin: unloaded idle '{id}' (VRAM free now {:?} MiB)", crate::limits::vram().map(|v| v.free >> 20));
        }
    }); });
}

fn cleanup() { if let Ok((dev, _, _)) = crate::backend::select("cuda", "f32", None) { let _ = dev.sync(); dev.memory_cleanup(); } }

fn hf_has(app: &App, repo: &str) -> bool { crate::store::hfcache::find(&app.storage.hf_cache, repo, None).is_some() }

/// Workflow cards: what is ready, what is missing (and why).
async fn workflows(State(app): S) -> Json<Value> {
    let sd = hf_has(&app, SD15);
    let tg = hf_has(&app, TAGGER);
    let sd_why = if sd { "" } else { "Stable Diffusion 1.5 is not downloaded (Models tab: pull stable-diffusion-v1-5/stable-diffusion-v1-5)" };
    let wip = |why: &str| json!({"ready": false, "status": "in progress", "reason": why});
    let ok = |m: &str| json!({"ready": true, "status": "ready", "model": m});
    let sdw = |m: &str| if sd { ok(m) } else { json!({"ready": false, "status": "missing model", "reason": sd_why}) };
    let mut w = serde_json::Map::new();
    w.insert("txt2img".into(), sdw("SD 1.5 (f16, GPU on demand)"));
    w.insert("img2img".into(), sdw("SD 1.5 img2img"));
    w.insert("restyle".into(), sdw("SD 1.5 img2img + style templates"));
    w.insert("inpaint".into(), sdw("SD 1.5, masked latent blending + feathered composite"));
    w.insert("outpaint".into(), sdw("SD 1.5, masked latent blending on an extended canvas"));
    w.insert("character".into(), sdw("SD 1.5, locked seed + region inpaint"));
    w.insert("tagger".into(), if tg { ok("Kagami-24k EVA02-L (CPU)") } else { json!({"ready": false, "status": "missing model", "reason": "Redstonexs/kagami-24k is not downloaded"}) });
    w.insert("svg".into(), ok("vtracer (CPU)"));
    w.insert("upscale".into(), wip("Real-ESRGAN (SRVGGNet / RRDBNet) Burn port in progress"));
    w.insert("denoise".into(), wip("Real-ESRGAN general-x4v3 with denoise strength: Burn port in progress"));
    w.insert("rmbg".into(), wip("IS-Net (isnet-anime / general) Burn port in progress"));
    w.insert("layers".into(), wip("needs background removal (IS-Net) + inpaint; after rmbg"));
    w.insert("animate".into(), wip("frame interpolation model (RIFE / FILM) port planned"));
    w.insert("to3d".into(), wip("image-to-mesh model (TripoSR-class) port planned"));
    w.insert("clothes".into(), wip("needs a human-parsing segmenter for the clothes mask; manual-mask version via Inpaint works now"));
    w.insert("pose".into(), wip("needs ControlNet OpenPose weights + pose editor"));
    let (n, b) = crate::limits::leases();
    Json(json!({"workflows": w, "loaded": app.loaded().iter().filter(|m| m.kind != Kind::Llm).map(|m| json!({"id": m.id, "kind": m.kind.as_str(), "backend": m.backend})).collect::<Vec<_>>(),
        "vram": crate::limits::vram().map(|v| json!({"free_mib": v.free >> 20, "used_mib": v.used >> 20})), "leases": n, "leased_mib": b >> 20, "idle_unload_s": idle().as_secs()}))
}

async fn styles(State(app): S) -> Response {
    let user = app.storage.store.root.join("cabin").join("styles.json");
    let s = std::fs::read_to_string(&user).unwrap_or_else(|_| STYLES.to_string());
    ([("content-type", "application/json")], s).into_response()
}

/// `POST /api/cabin/ensure {"what": "image"|"tagger"}`: load the model a workflow needs (on demand).
async fn ensure_ep(State(app): S, Json(b): Json<Value>) -> Response {
    let t = Instant::now();
    let r = match b["what"].as_str().unwrap_or("image") {
        "image" => ensure(&app, Kind::Image, b["model"].as_str().unwrap_or(SD15), "cuda", SD15_LEASE).await,
        "tagger" => ensure(&app, Kind::Tagger, TAGGER, b["backend"].as_str().unwrap_or("cpu"), 1_800 << 20).await,
        w => return err(StatusCode::BAD_REQUEST, format!("unknown model kind '{w}'")),
    };
    match r {
        Ok(m) => Json(json!({"id": m.id, "kind": m.kind.as_str(), "backend": m.backend, "seconds": t.elapsed().as_secs_f64(), "vram_free_mib": crate::limits::vram().map(|v| v.free >> 20)})).into_response(),
        Err(r) => r,
    }
}

/// `POST /api/cabin/vectorize`: raster -> SVG with vtracer (CPU).
async fn vectorize(Json(b): Json<Value>) -> Response {
    use base64::Engine;
    let Some(s) = b["image"].as_str() else { return err(StatusCode::BAD_REQUEST, "missing image") };
    let s = s.split_once("base64,").map(|x| x.1).unwrap_or(s);
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s.trim()) else { return err(StatusCode::BAD_REQUEST, "image must be base64") };
    let r = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let t = Instant::now();
        let img = image::load_from_memory(&bytes)?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w * h > 4096 * 4096 { anyhow::bail!("image too large ({w}x{h}); max 16 MP"); }
        let mut cfg = match b["preset"].as_str().unwrap_or("illustration") {
            "logo" | "flat" => vtracer::Config { filter_speckle: 8, color_precision: 5, layer_difference: 32, corner_threshold: 60, ..Default::default() },
            "pixel" => vtracer::Config { filter_speckle: 0, color_precision: 8, layer_difference: 0, mode: visioncortex::PathSimplifyMode::None, ..Default::default() },
            "lineart" | "bw" => vtracer::Config { color_mode: vtracer::ColorMode::Binary, filter_speckle: 4, ..Default::default() },
            "photo" => vtracer::Config { filter_speckle: 10, color_precision: 8, layer_difference: 48, corner_threshold: 180, ..Default::default() },
            _ => vtracer::Config { filter_speckle: 4, color_precision: 6, layer_difference: 16, ..Default::default() },
        };
        if let Some(v) = b["filter_speckle"].as_u64() { cfg.filter_speckle = v as usize; }
        if let Some(v) = b["color_precision"].as_i64() { cfg.color_precision = v.clamp(1, 8) as i32; }
        if let Some(v) = b["layer_difference"].as_i64() { cfg.layer_difference = v.clamp(0, 255) as i32; }
        if let Some(v) = b["corner_threshold"].as_i64() { cfg.corner_threshold = v.clamp(0, 180) as i32; }
        if let Some(v) = b["length_threshold"].as_f64() { cfg.length_threshold = v.clamp(3.5, 10.0); }
        if let Some(m) = b["mode"].as_str() { cfg.mode = match m { "polygon" => visioncortex::PathSimplifyMode::Polygon, "none" | "pixel" => visioncortex::PathSimplifyMode::None, _ => visioncortex::PathSimplifyMode::Spline }; }
        if b["binary"].as_bool() == Some(true) { cfg.color_mode = vtracer::ColorMode::Binary; }
        let ci = visioncortex::ColorImage { pixels: img.into_raw(), width: w, height: h };
        let svg = vtracer::convert(ci, cfg).map_err(|e| anyhow::anyhow!("vtracer: {e}"))?.to_string();
        let paths = svg.matches("<path").count();
        Ok(json!({"svg": svg, "paths": paths, "bytes": svg.len(), "width": w, "height": h, "seconds": t.elapsed().as_secs_f64()}))
    }).await;
    match r { Ok(Ok(v)) => Json(v).into_response(), Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")), Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()) }
}

pub fn routes(r: Router<Arc<App>>) -> Router<Arc<App>> {
    r.route("/api/cabin/workflows", get(workflows))
        .route("/api/cabin/styles", get(styles))
        .route("/api/cabin/ensure", post(ensure_ep))
        .route("/api/cabin/vectorize", post(vectorize).layer(axum::extract::DefaultBodyLimit::max(64 << 20)))
}
