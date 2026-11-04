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
    if sd && crate::lcm::lora_file(&app.storage.store.root).is_file() { if let Some(o) = w.get_mut("txt2img") { o["lcm"] = json!(true); } }
    w.insert("img2img".into(), sdw("SD 1.5 img2img"));
    w.insert("restyle".into(), sdw("SD 1.5 img2img + style templates"));
    let ipm = app.storage.store.root.join("cabin").join("sd15-inpaint").join("unet").join("diffusion_pytorch_model.fp16.safetensors").is_file();
    w.insert("inpaint".into(), sdw(if ipm { "SD 1.5 inpainting UNet (9-channel) + feathered composite" } else { "SD 1.5, masked latent blending + feathered composite" }));
    w.insert("outpaint".into(), sdw("SD 1.5, masked latent blending on an extended canvas"));
    w.insert("character".into(), sdw("SD 1.5, locked seed + region inpaint"));
    w.insert("tagger".into(), if tg { ok("Kagami-24k EVA02-L (CPU)") } else { json!({"ready": false, "status": "missing model", "reason": "Redstonexs/kagami-24k is not downloaded"}) });
    w.insert("svg".into(), ok("vtracer (CPU)"));
    let esr = crate::esr::available(&crate::esr::weights_dir(&app.storage.store.root));
    w.insert("upscale".into(), if esr { ok("Real-ESRGAN compact x4 (general / anime), GPU") } else { wip("Real-ESRGAN weights missing in <store>/cabin/realesr") });
    w.insert("denoise".into(), if esr { ok("Real-ESRGAN general-x4v3 + wdn blend (denoise strength)") } else { wip("Real-ESRGAN weights missing") });
    let isd = crate::isnet::weights_dir(&app.storage.store.root);
    let isn = crate::isnet::available(&isd, "rmbg") || crate::isnet::available(&isd, "anime");
    w.insert("rmbg".into(), if isn { ok("IS-Net: BRIA RMBG-1.4 (general, non-commercial) / anime-seg (anime), GPU") } else { wip("IS-Net weights missing in <store>/cabin/isnet") });
    w.insert("layers".into(), if isn && sd { ok("IS-Net cut-out + SD 1.5 inpainted background, zip of PNG layers") } else { wip("needs IS-Net weights and SD 1.5") });
    w.insert("animate".into(), wip("frame interpolation model (RIFE / FILM) port planned"));
    w.insert("to3d".into(), wip("image-to-mesh model (TripoSR-class) port planned"));
    let seg = crate::segformer::available(&crate::segformer::weights_dir(&app.storage.store.root));
    w.insert("clothes".into(), if seg && sd { ok("SegFormer-B2 clothes parsing (auto mask) + SD 1.5 inpainting") } else { wip("SegFormer clothes weights missing in <store>/cabin/segformer") });
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
        "image" => {
            // masked edits switch to the SD 1.5 inpainting UNet when it is in the store
            let ip = app.storage.store.root.join("cabin").join("sd15-inpaint");
            if std::env::var_os("PYRE_SD_INPAINT_DIR").is_none() && ip.join("unet").join("config.json").is_file() && std::fs::read_dir(ip.join("unet")).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))).unwrap_or(false) {
                std::env::set_var("PYRE_SD_INPAINT_DIR", &ip);
            }
            let lcm = crate::lcm::lora_file(&app.storage.store.root);
            if std::env::var_os("PYRE_SD_LCM_LORA").is_none() && lcm.is_file() { std::env::set_var("PYRE_SD_LCM_LORA", &lcm); }
            ensure(&app, Kind::Image, b["model"].as_str().unwrap_or(SD15), "cuda", SD15_LEASE).await
        }
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

fn b64_image(v: &Value) -> anyhow::Result<image::DynamicImage> {
    use base64::Engine;
    let s = v.as_str().ok_or_else(|| anyhow::anyhow!("missing image"))?;
    let s = s.split_once("base64,").map(|x| x.1).unwrap_or(s);
    Ok(image::load_from_memory(&base64::engine::general_purpose::STANDARD.decode(s.trim())?)?)
}
fn png_b64(img: &image::DynamicImage) -> anyhow::Result<String> {
    use base64::Engine;
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)?;
    Ok(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner())))
}
static GPU_SMALL: Mutex<()> = Mutex::new(());
/// Run a small GPU job under a VRAM lease (the expert cache shrinks first if needed).
fn small_gpu<T>(lease: u64, what: &str, f: impl FnOnce(&burn::tensor::Device) -> anyhow::Result<T>) -> anyhow::Result<T> {
    let _g = GPU_SMALL.lock().unwrap();
    crate::limits::lease(lease, what)?;
    let r = (|| { let (dev, _, _) = crate::backend::select("cuda", "f32", None)?; let r = f(&dev); dev.memory_cleanup(); r })();
    crate::limits::lease_end(lease);
    r
}
fn json_or_err(r: Result<anyhow::Result<Value>, tokio::task::JoinError>) -> Response {
    match r { Ok(Ok(v)) => Json(v).into_response(), Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")), Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()) }
}

/// `POST /api/cabin/upscale {image, scale: 2|4, model: general|anime, denoise: 0..1}` (Real-ESRGAN compact).
async fn upscale(State(app): S, Json(b): Json<Value>) -> Response {
    let dir = crate::esr::weights_dir(&app.storage.store.root);
    json_or_err(tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let t = Instant::now();
        let img = b64_image(&b["image"])?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w * h > 1536 * 1536 { anyhow::bail!("image too large ({w}x{h}); max 1536x1536 in"); }
        let scale = b["scale"].as_u64().unwrap_or(4).clamp(1, 4) as usize;
        let model = b["model"].as_str().unwrap_or("general").to_string();
        let dn = b["denoise"].as_f64().unwrap_or(0.5) as f32;
        let out = small_gpu(700 << 20, "Real-ESRGAN upscale", |dev| {
            let e = crate::esr::Esr::load(&dir, &model, dn, dev)?;
            e.upscale_rgb(img.as_raw(), w, h, 160, 10)
        })?;
        let mut o = image::DynamicImage::ImageRgb8(image::RgbImage::from_raw((w * 4) as u32, (h * 4) as u32, out).ok_or_else(|| anyhow::anyhow!("bad output"))?);
        if scale != 4 { o = o.resize_exact((w * scale) as u32, (h * scale) as u32, image::imageops::FilterType::Lanczos3); }
        Ok(json!({"image": png_b64(&o)?, "width": o.width(), "height": o.height(), "model": format!("realesr-{model}-x4v3"), "denoise": dn, "seconds": t.elapsed().as_secs_f64()}))
    }).await)
}

/// `POST /api/cabin/esr_check {refdir}`: compare the Burn port with PyTorch reference outputs.
async fn esr_check(State(app): S, Json(b): Json<Value>) -> Response {
    let dir = crate::esr::weights_dir(&app.storage.store.root);
    let rd = std::path::PathBuf::from(b["refdir"].as_str().unwrap_or(""));
    json_or_err(tokio::task::spawn_blocking(move || small_gpu(300 << 20, "esr check", |dev| crate::esr::check(&dir, &rd, dev)).map(|s| json!({"report": s}))).await)
}

/// Minimal stored (uncompressed) zip: PNG layers are already compressed.
fn zip_store(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    fn crc32(d: &[u8]) -> u32 { let mut c = !0u32; for &b in d { c ^= b as u32; for _ in 0..8 { c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 }; } } !c }
    let (mut out, mut cd) = (Vec::new(), Vec::new());
    for (name, data) in files {
        let (crc, off, n) = (crc32(data), out.len() as u32, name.as_bytes());
        let hdr = |sig: u32, central: bool| -> Vec<u8> {
            let mut h = Vec::new(); h.extend(sig.to_le_bytes()); if central { h.extend(20u16.to_le_bytes()); }
            h.extend(20u16.to_le_bytes()); h.extend(0u16.to_le_bytes()); h.extend(0u16.to_le_bytes()); h.extend(0u32.to_le_bytes());
            h.extend(crc.to_le_bytes()); h.extend((data.len() as u32).to_le_bytes()); h.extend((data.len() as u32).to_le_bytes());
            h.extend((n.len() as u16).to_le_bytes()); h.extend(0u16.to_le_bytes());
            if central { h.extend([0u8; 6]); h.extend(0u32.to_le_bytes()); h.extend(off.to_le_bytes()); } // comment len, disk, internal attr; external attr; offset
            h.extend(n); h };
        out.extend(hdr(0x0403_4b50, false)); out.extend(data); cd.extend(hdr(0x0201_4b50, true));
    }
    let (cdo, cdl) = (out.len() as u32, cd.len() as u32); out.extend(cd);
    out.extend(0x0605_4b50u32.to_le_bytes()); out.extend([0u8; 4]); out.extend((files.len() as u16).to_le_bytes()); out.extend((files.len() as u16).to_le_bytes());
    out.extend(cdl.to_le_bytes()); out.extend(cdo.to_le_bytes()); out.extend(0u16.to_le_bytes());
    out
}

/// `POST /api/cabin/rmbg {image, model: general|anime, side: 1024}` -> RGBA cut-out + mask (IS-Net).
async fn rmbg(State(app): S, Json(b): Json<Value>) -> Response {
    let dir = crate::isnet::weights_dir(&app.storage.store.root);
    json_or_err(tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let t = Instant::now();
        let img = b64_image(&b["image"])?.to_rgb8();
        let name = match b["model"].as_str().unwrap_or("general") { "anime" => "anime", _ => "rmbg" };
        let side = b["side"].as_u64().unwrap_or(1024).clamp(256, 1280) as u32 / 32 * 32;
        let m = small_gpu(1_600 << 20, "IS-Net background removal", |dev| crate::isnet::IsNet::load(&dir, name, dev)?.matte(&img, side, dev))?;
        let mut rgba = image::RgbaImage::new(img.width(), img.height());
        for (x, y, p) in rgba.enumerate_pixels_mut() { let c = img.get_pixel(x, y); *p = image::Rgba([c[0], c[1], c[2], m.get_pixel(x, y)[0]]); }
        let fg = m.pixels().filter(|p| p[0] > 127).count() as f64 / (m.width() * m.height()) as f64;
        Ok(json!({"image": png_b64(&image::DynamicImage::ImageRgba8(rgba))?, "mask": png_b64(&image::DynamicImage::ImageLuma8(m))?, "model": if name == "anime" { "isnet anime-seg" } else { "BRIA RMBG-1.4 (IS-Net)" }, "foreground_fraction": fg, "seconds": t.elapsed().as_secs_f64()}))
    }).await)
}

/// Label ids for a part name (character creator fields and the clothes workflow).
fn part_labels(p: &str) -> &'static [u8] {
    match p { "hair" => &[2], "hat" => &[1], "face" | "eyes" | "expr" => &[11], "top" | "upper" => &[4, 17], "sleeves" => &[4, 14, 15], "bottom" => &[5, 6, 7], "skirt" => &[5], "pants" => &[6], "dress" => &[4, 5, 6, 7], "shoes" => &[9, 10], "acc" => &[1, 3, 8, 16, 17], "legs" => &[12, 13], "arms" => &[14, 15], "person" => &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], _ => &[] }
}
/// Square max-filter (dilation) of a 0/1 mask, radius r.
fn dilate(m: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 { return m.to_vec(); }
    let mut t = vec![0u8; w * h];
    for y in 0..h { let row = &m[y * w..(y + 1) * w]; let mut last: isize = -1_000_000; for x in 0..w { if row[x] > 0 { last = x as isize; } if (x as isize - last) as usize <= r { t[y * w + x] = 1; } } let mut next: isize = 1_000_000; for x in (0..w).rev() { if row[x] > 0 { next = x as isize; } if (next - x as isize) as usize <= r { t[y * w + x] = 1; } } }
    let mut o = vec![0u8; w * h];
    for x in 0..w { let mut last: isize = -1_000_000; for y in 0..h { if t[y * w + x] > 0 { last = y as isize; } if (y as isize - last) as usize <= r { o[y * w + x] = 1; } } let mut next: isize = 1_000_000; for y in (0..h).rev() { if t[y * w + x] > 0 { next = y as isize; } if (next - y as isize) as usize <= r { o[y * w + x] = 1; } } }
    o
}

/// `POST /api/cabin/parse {image, parts?: ["hair", ...] | labels?: [ids], grow: 12, protect?: [parts]}` ->
/// SegFormer clothes parsing: label map (grey = class id), colour preview, per-label coverage, and when parts or
/// labels are given a white-on-black edit mask = dilate(selected, grow) minus protected labels (default: face
/// unless the face itself is selected).
async fn parse(State(app): S, Json(b): Json<Value>) -> Response {
    let dir = crate::segformer::weights_dir(&app.storage.store.root);
    json_or_err(tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let t = Instant::now();
        let img = b64_image(&b["image"])?.to_rgb8();
        let lab = small_gpu(1_200 << 20, "SegFormer clothes parsing", |dev| crate::segformer::SegFormer::load(&dir, dev)?.parse(&img))?;
        let (w, h) = (lab.width() as usize, lab.height() as usize);
        let mut cnt = [0usize; 18]; for p in lab.pixels() { cnt[p[0] as usize % 18] += 1; }
        let cover: serde_json::Map<String, Value> = (0..18).filter(|&i| cnt[i] > 0).map(|i| (crate::segformer::LABELS[i].to_string(), json!(cnt[i] as f64 / (w * h) as f64))).collect();
        const PAL: [[u8; 3]; 18] = [[0, 0, 0], [128, 0, 0], [0, 64, 255], [255, 255, 0], [255, 140, 100], [200, 0, 200], [0, 160, 80], [255, 0, 120], [120, 80, 0], [0, 200, 200], [0, 120, 200], [255, 220, 170], [140, 140, 0], [100, 100, 0], [255, 180, 0], [200, 140, 0], [120, 0, 255], [255, 90, 0]];
        let mut prev = image::RgbImage::new(w as u32, h as u32);
        for (x, y, p) in prev.enumerate_pixels_mut() { let c = img.get_pixel(x, y); let l = PAL[lab.get_pixel(x, y)[0] as usize % 18]; *p = image::Rgb([((c[0] as u16 + l[0] as u16 * 2) / 3) as u8, ((c[1] as u16 + l[1] as u16 * 2) / 3) as u8, ((c[2] as u16 + l[2] as u16 * 2) / 3) as u8]); }
        let mut sel: Vec<u8> = b["labels"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u8).collect()).unwrap_or_default();
        let parts: Vec<String> = b["parts"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
        for p in &parts { sel.extend_from_slice(part_labels(p)); }
        let mut out = json!({"labels": png_b64(&image::DynamicImage::ImageLuma8(lab.clone()))?, "preview": png_b64(&image::DynamicImage::ImageRgb8(prev))?, "coverage": cover, "names": crate::segformer::LABELS});
        if !sel.is_empty() {
            let grow = b["grow"].as_u64().unwrap_or(12).min(96) as usize;
            let mut prot: Vec<u8> = match b["protect"].as_array() { Some(a) => a.iter().filter_map(|v| v.as_str()).flat_map(|p| part_labels(p).iter().copied()).collect(), None => vec![11] };
            prot.retain(|l| !sel.contains(l));
            let raw: Vec<u8> = lab.pixels().map(|p| sel.contains(&p[0]) as u8).collect();
            let sel_px = raw.iter().filter(|&&v| v > 0).count();
            let d = dilate(&raw, w, h, grow);
            let mut m = image::GrayImage::new(w as u32, h as u32);
            for (i, p) in m.pixels_mut().enumerate() { let l = lab.as_raw()[i]; p.0[0] = if d[i] > 0 && !prot.contains(&l) { 255 } else { 0 }; }
            out["mask"] = json!(png_b64(&image::DynamicImage::ImageLuma8(m))?);
            out["selected_fraction"] = json!(sel_px as f64 / (w * h) as f64);
        }
        out["seconds"] = json!(t.elapsed().as_secs_f64());
        Ok(out)
    }).await)
}

/// `POST /api/cabin/seg_check {refdir}`.
async fn seg_check(State(app): S, Json(b): Json<Value>) -> Response {
    let dir = crate::segformer::weights_dir(&app.storage.store.root);
    let rd = std::path::PathBuf::from(b["refdir"].as_str().unwrap_or(""));
    json_or_err(tokio::task::spawn_blocking(move || small_gpu(1_200 << 20, "segformer check", |dev| crate::segformer::check(&dir, &rd, dev)).map(|s| json!({"report": s}))).await)
}

/// `POST /api/cabin/isnet_check {refdir}`.
async fn isnet_check(State(app): S, Json(b): Json<Value>) -> Response {
    let dir = crate::isnet::weights_dir(&app.storage.store.root);
    let rd = std::path::PathBuf::from(b["refdir"].as_str().unwrap_or(""));
    json_or_err(tokio::task::spawn_blocking(move || small_gpu(1_200 << 20, "isnet check", |dev| crate::isnet::check(&dir, &rd, dev)).map(|s| json!({"report": s}))).await)
}

/// `POST /api/cabin/pack {name, files: [{name, image}]}` -> `{zip: base64}` (stored zip of the layers).
async fn pack(Json(b): Json<Value>) -> Response {
    use base64::Engine;
    json_or_err(tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let mut files = Vec::new();
        for f in b["files"].as_array().ok_or_else(|| anyhow::anyhow!("files[] missing"))? {
            let name = f["name"].as_str().unwrap_or("layer.png").replace(['/', '\\'], "_");
            let s = f["image"].as_str().ok_or_else(|| anyhow::anyhow!("{name}: image missing"))?;
            let s = s.split_once("base64,").map(|x| x.1).unwrap_or(s);
            files.push((name, base64::engine::general_purpose::STANDARD.decode(s.trim())?));
        }
        let z = zip_store(&files);
        Ok(json!({"zip": base64::engine::general_purpose::STANDARD.encode(&z), "bytes": z.len(), "files": files.len()}))
    }).await)
}

pub fn routes(r: Router<Arc<App>>) -> Router<Arc<App>> {
    r.route("/api/cabin/workflows", get(workflows))
        .route("/api/cabin/styles", get(styles))
        .route("/api/cabin/ensure", post(ensure_ep))
        .route("/api/cabin/upscale", post(upscale).layer(axum::extract::DefaultBodyLimit::max(64 << 20)))
        .route("/api/cabin/esr_check", post(esr_check))
        .route("/api/cabin/rmbg", post(rmbg).layer(axum::extract::DefaultBodyLimit::max(64 << 20)))
        .route("/api/cabin/isnet_check", post(isnet_check))
        .route("/api/cabin/parse", post(parse))
        .route("/api/cabin/seg_check", post(seg_check))
        .route("/api/cabin/pack", post(pack).layer(axum::extract::DefaultBodyLimit::max(256 << 20)))
        .route("/api/cabin/vectorize", post(vectorize).layer(axum::extract::DefaultBodyLimit::max(64 << 20)))
}
