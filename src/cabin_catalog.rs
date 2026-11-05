//! Known-compatible models for Cabin tasks: resolve lazily from the HF cache / pyre store, and
//! offer downloads (into the HF cache or `<store>/cabin/...`) when nothing is present. No startup scan.
use crate::store::{hfcache, Kind};
use crate::registry::App;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A downloadable offer for a cabin component.
#[derive(Clone, Copy)]
pub struct Offer {
    pub id: &'static str,
    pub name: &'static str,
    pub component: &'static str,
    pub size_bytes: u64,
    pub licence: &'static str,
    pub repo: &'static str,
    pub note: &'static str,
    pub kind: OfferKind,
}

#[derive(Clone, Copy)]
pub enum OfferKind {
    /// Full HF hub snapshot via `hfcache::download` (diffusers / tagger / etc.).
    HfRepo,
    /// Specific files from an HF repo into a cabin subdirectory (optional rename).
    HfFiles { files: &'static [(&'static str, &'static str)], cabin_subdir: &'static str },
    /// Direct HTTPS URL into a cabin subdirectory.
    Url { url: &'static str, filename: &'static str, cabin_subdir: &'static str },
}

pub const OFFERS: &[Offer] = &[
    Offer { id: "sd15", name: "Stable Diffusion 1.5", component: "sd15", size_bytes: 4_300_000_000, licence: "CreativeML OpenRAIL-M",
        repo: "stable-diffusion-v1-5/stable-diffusion-v1-5", note: "base text-to-image / img2img / inpaint blending", kind: OfferKind::HfRepo },
    Offer { id: "sd15-inpaint", name: "SD 1.5 inpainting UNet (9-ch)", component: "sd15-inpaint", size_bytes: 1_720_000_000, licence: "CreativeML OpenRAIL-M",
        repo: "stable-diffusion-v1-5/stable-diffusion-inpainting", note: "optional; better masked edits when present",
        kind: OfferKind::HfFiles { files: &[("unet/config.json", "unet/config.json"), ("unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors"), ("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.safetensors")], cabin_subdir: "sd15-inpaint" } },
    Offer { id: "lcm-lora", name: "LCM-LoRA for SD 1.5 (fast mode)", component: "lcm", size_bytes: 135_000_000, licence: "openrail++",
        repo: "latent-consistency/lcm-lora-sdv1-5", note: "~4x fewer steps",
        kind: OfferKind::HfFiles { files: &[("pytorch_lora_weights.safetensors", "lcm-lora-sdv1-5.safetensors")], cabin_subdir: "lcm" } },
    Offer { id: "controlnet-openpose", name: "ControlNet OpenPose v1.1", component: "openpose", size_bytes: 723_000_000, licence: "openrail++",
        repo: "lllyasviel/control_v11p_sd15_openpose", note: "pose / reposition",
        kind: OfferKind::HfFiles { files: &[("config.json", "config.json"), ("diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.fp16.safetensors")], cabin_subdir: "controlnet-openpose" } },
    Offer { id: "kagami", name: "Kagami-24k EVA02 tagger", component: "tagger", size_bytes: 900_000_000, licence: "Apache-2.0",
        repo: "Redstonexs/kagami-24k", note: "Danbooru tags", kind: OfferKind::HfRepo },
    Offer { id: "segformer-clothes", name: "SegFormer-B2 clothes parser", component: "segformer", size_bytes: 110_000_000, licence: "MIT",
        repo: "mattmdjaga/segformer_b2_clothes", note: "swap clothes / character auto-masks",
        kind: OfferKind::HfFiles { files: &[("model.safetensors", "segformer_clothes.safetensors")], cabin_subdir: "segformer" } },
    Offer { id: "triposr", name: "TripoSR (stabilityai)", component: "triposr", size_bytes: 1_680_000_000, licence: "MIT",
        repo: "stabilityai/TripoSR", note: "2D→3D; converted to f16 on first use",
        kind: OfferKind::HfFiles { files: &[("model.ckpt", "model.ckpt")], cabin_subdir: "triposr" } },
    Offer { id: "esr-general", name: "Real-ESRGAN general-x4v3", component: "esr", size_bytes: 4_900_000, licence: "BSD-3-Clause",
        repo: "xinntao/Real-ESRGAN", note: "upscale / denoise (converted safetensors)",
        kind: OfferKind::Url { url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", filename: "realesr_general_x4v3.pth", cabin_subdir: "realesr" } },
    Offer { id: "esr-general-wdn", name: "Real-ESRGAN general-wdn-x4v3", component: "esr", size_bytes: 4_900_000, licence: "BSD-3-Clause",
        repo: "xinntao/Real-ESRGAN", note: "weak-denoise blend partner",
        kind: OfferKind::Url { url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth", filename: "realesr_general_wdn_x4v3.pth", cabin_subdir: "realesr" } },
    Offer { id: "esr-anime", name: "Real-ESRGAN animevideov3", component: "esr", size_bytes: 2_500_000, licence: "BSD-3-Clause",
        repo: "xinntao/Real-ESRGAN", note: "anime upscale (smallest cabin weight)",
        kind: OfferKind::Url { url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", filename: "realesr_anime_x4v3.pth", cabin_subdir: "realesr" } },
    Offer { id: "isnet-anime", name: "anime-seg (IS-Net)", component: "isnet", size_bytes: 204_000_000, licence: "Apache-2.0",
        repo: "skytnt/anime-seg", note: "anime / illustration cut-out",
        kind: OfferKind::HfFiles { files: &[("isnetis.ckpt", "isnetis.ckpt")], cabin_subdir: "isnet" } },
    Offer { id: "isnet-rmbg", name: "BRIA RMBG-1.4 (IS-Net)", component: "isnet", size_bytes: 176_000_000, licence: "bria-rmbg-1.4 (non-commercial)",
        repo: "briaai/RMBG-1.4", note: "general / photo cut-out — non-commercial licence",
        kind: OfferKind::HfFiles { files: &[("model.safetensors", "isnet_rmbg.safetensors")], cabin_subdir: "isnet" } },
    Offer { id: "rife46", name: "RIFE 4.6", component: "rife", size_bytes: 21_000_000, licence: "MIT (code); check weight mirror",
        repo: "hzwer/Practical-RIFE", note: "frame interpolation; converted safetensors from flownet.pkl",
        kind: OfferKind::Url { url: "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/flownet.pkl", filename: "flownet.pkl", cabin_subdir: "rife" } },
];

/// Components each workflow needs (all must be present for ready=true).
pub fn needs(wf: &str) -> &'static [&'static str] {
    match wf {
        "txt2img" | "img2img" | "restyle" | "outpaint" | "character" => &["sd15"],
        "inpaint" => &["sd15"], // sd15-inpaint is optional
        "tagger" => &["tagger"],
        "svg" => &[],
        "upscale" | "denoise" => &["esr"],
        "rmbg" => &["isnet"],
        "layers" => &["isnet", "sd15"],
        "animate" => &["rife"],
        "to3d" => &["triposr", "isnet"],
        "clothes" => &["segformer", "sd15"],
        "pose" => &["openpose", "sd15"],
        _ => &[],
    }
}

fn cabin_root(app: &App) -> PathBuf { app.storage.store.root.join("cabin") }

/// `--no-auto-models` / PYRE_NO_AUTO_MODELS=1: only explicitly loaded image/tagger models count as present
/// for sd15/tagger (cabin file weights under <store>/cabin still resolve normally).
pub fn auto_models_enabled() -> bool {
    if std::env::var_os("PYRE_NO_AUTO_MODELS").is_some_and(|v| v != "0" && v != "false") { return false; }
    true
}

/// Is this component present (HF cache or cabin folder)? Explicit `--image-model` / loaded image counts as sd15.
pub fn present(app: &App, component: &str) -> bool {
    match component {
        "sd15" => {
            if app.loaded().iter().any(|m| m.kind == Kind::Image) { return true; }
            if !auto_models_enabled() { return false; }
            if hfcache::find(&app.storage.hf_cache, crate::cabin::SD15, None).is_some() { return true; }
            app.storage.store.list().into_iter().any(|e| e.manifest.kind == Kind::Image)
                || hfcache::list(&app.storage.hf_cache).into_iter().any(|c| {
                    let p = hfcache::find(&app.storage.hf_cache, &c.repo, None);
                    p.as_ref().map(|d| crate::store::detect(d).0 == Kind::Image).unwrap_or(false)
                })
        }
        "tagger" => {
            if app.loaded().iter().any(|m| m.kind == Kind::Tagger) { return true; }
            if !auto_models_enabled() { return false; }
            hfcache::find(&app.storage.hf_cache, crate::cabin::TAGGER, None).is_some()
                || app.storage.store.list().into_iter().any(|e| e.manifest.kind == Kind::Tagger)
        }
        "sd15-inpaint" => { let u = cabin_root(app).join("sd15-inpaint").join("unet"); u.join("diffusion_pytorch_model.fp16.safetensors").is_file() || u.join("diffusion_pytorch_model.safetensors").is_file() },
        "lcm" => crate::lcm::lora_file(&app.storage.store.root).is_file(),
        "openpose" => cabin_root(app).join("controlnet-openpose").join("diffusion_pytorch_model.fp16.safetensors").is_file(),
        "esr" => crate::esr::available(&crate::esr::weights_dir(&app.storage.store.root)),
        "isnet" => {
            let d = crate::isnet::weights_dir(&app.storage.store.root);
            crate::isnet::available(&d, "rmbg") || crate::isnet::available(&d, "anime")
        }
        "rife" => crate::rife::available(&crate::rife::weights_dir(&app.storage.store.root)),
        "segformer" => crate::segformer::available(&crate::segformer::weights_dir(&app.storage.store.root)),
        "triposr" => crate::triposr::available(&crate::triposr::weights_dir(&app.storage.store.root)),
        _ => false,
    }
}

pub fn offers_for(component: &str) -> Vec<&'static Offer> {
    OFFERS.iter().filter(|o| o.component == component).collect()
}

pub fn offer(id: &str) -> Option<&'static Offer> { OFFERS.iter().find(|o| o.id == id) }

pub fn offer_json(o: &Offer) -> Value {
    json!({
        "id": o.id, "name": o.name, "component": o.component, "size_bytes": o.size_bytes,
        "size": crate::store::human(o.size_bytes), "licence": o.licence, "repo": o.repo, "note": o.note,
        "present": false, // filled by caller
    })
}

/// Resolve which SD repo id to load: explicit loaded model, else the default if cached, else any detected Image.
pub fn resolve_sd_id(app: &App) -> Option<String> {
    if let Some(m) = app.loaded().into_iter().find(|m| m.kind == Kind::Image) { return Some(m.id.clone()); }
    if hfcache::find(&app.storage.hf_cache, crate::cabin::SD15, None).is_some() { return Some(crate::cabin::SD15.into()); }
    for e in app.storage.store.list() {
        if e.manifest.kind == Kind::Image { return Some(e.manifest.name.clone()); }
    }
    for c in hfcache::list(&app.storage.hf_cache) {
        if let Some(d) = hfcache::find(&app.storage.hf_cache, &c.repo, None) {
            if crate::store::detect(&d).0 == Kind::Image { return Some(c.repo); }
        }
    }
    None
}

// ---------- downloads (cancellable) ----------
struct Active { cancel: Arc<AtomicBool> }
static ACTIVE: OnceLock<Mutex<std::collections::HashMap<String, Active>>> = OnceLock::new();
fn active_map() -> &'static Mutex<std::collections::HashMap<String, Active>> { ACTIVE.get_or_init(|| Mutex::new(Default::default())) }

pub fn cancel(id: &str) -> bool {
    if let Some(a) = active_map().lock().unwrap().get(id) { a.cancel.store(true, Ordering::SeqCst); true } else { false }
}

fn cancelled(flag: &AtomicBool) -> bool { flag.load(Ordering::SeqCst) }

/// Download one catalog offer. Sends progress events as JSON values through `tx`.
pub fn download(app: &App, id: &str, tx: &dyn Fn(Value), cancel: Arc<AtomicBool>) -> Result<PathBuf> {
    let o = offer(id).with_context(|| format!("unknown cabin model id '{id}'"))?;
    if app.storage.offline { bail!("server is in offline mode"); }
    active_map().lock().unwrap().insert(id.into(), Active { cancel: cancel.clone() });
    let res = (|| -> Result<PathBuf> {
        match o.kind {
            OfferKind::HfRepo => {
                let (r, rev) = hfcache::split_rev(o.repo);
                if let Some(p) = hfcache::find(&app.storage.hf_cache, r, rev) {
                    tx(json!({"file": "already cached", "index": 1, "total": 1}));
                    hfcache::dedupe_snapshot(&p);
                    return Ok(p);
                }
                let p = hfcache::download(&app.storage.hf_cache, r, rev, &|f, i, n| {
                    tx(json!({"file": f, "index": i, "total": n}));
                })?;
                Ok(p)
            }
            OfferKind::HfFiles { files, cabin_subdir } => {
                let dest = cabin_root(app).join(cabin_subdir);
                std::fs::create_dir_all(&dest)?;
                let (owner, name) = o.repo.split_once('/').context("repo id")?;
                let client = hf_hub::HFClient::builder().cache_dir(&app.storage.hf_cache)
                    .user_agent(concat!("pyre/", env!("CARGO_PKG_VERSION"))).build_sync()?;
                let h = client.model(owner, name);
                let n = files.len();
                let mut got_any_weight = false;
                for (i, (src, dst_name)) in files.iter().enumerate() {
                    if cancelled(&cancel) { bail!("cancelled"); }
                    tx(json!({"file": *src, "index": i, "total": n}));
                    match h.download_file().filename(*src).send() {
                        Ok(got) => {
                            let out = dest.join(dst_name);
                            if let Some(p) = out.parent() { std::fs::create_dir_all(p)?; }
                            if out.exists() { let _ = std::fs::remove_file(&out); }
                            if std::fs::hard_link(&got, &out).is_err() { std::fs::copy(&got, &out)?; }
                            if src.ends_with(".safetensors") || src.ends_with(".ckpt") || src.ends_with(".bin") || src.ends_with(".pt") { got_any_weight = true; }
                        }
                        Err(e) => {
                            // alternate weight filenames are optional; configs are nice-to-have
                            if src.ends_with("config.json") { tracing::warn!("optional {src}: {e:#}"); continue; }
                            if files.iter().any(|(s, _)| *s != *src && (s.ends_with(".safetensors") || s.ends_with(".ckpt"))) {
                                tracing::warn!("skipping {src}: {e:#}"); continue;
                            }
                            return Err(e).context(format!("downloading {src}"));
                        }
                    }
                }
                if !got_any_weight { bail!("{}: no weight file downloaded", o.repo); }
                tx(json!({"file": "done", "index": n, "total": n}));
                // TripoSR: convert ckpt -> f16 safetensors when python is around
                if cabin_subdir == "triposr" && !dest.join("triposr_f16.safetensors").is_file() {
                    tx(json!({"file": "converting TripoSR weights to f16…", "index": n, "total": n + 1}));
                    convert_triposr(&dest)?;
                }
                // Real-ESRGAN / RIFE raw weights: convert .pth/.pkl → .safetensors when needed
                post_convert(cabin_subdir, &dest, tx)?;
                Ok(dest)
            }
            OfferKind::Url { url, filename, cabin_subdir } => {
                let dest = cabin_root(app).join(cabin_subdir);
                std::fs::create_dir_all(&dest)?;
                let out = dest.join(filename);
                if !out.is_file() {
                    tx(json!({"file": filename, "index": 0, "total": 1}));
                    download_url(url, &out, &cancel, tx)?;
                } else {
                    tx(json!({"file": "already present", "index": 1, "total": 1}));
                }
                post_convert(cabin_subdir, &dest, tx)?;
                Ok(dest)
            }
        }
    })();
    active_map().lock().unwrap().remove(id);
    res
}

fn download_url(url: &str, out: &Path, cancel: &AtomicBool, tx: &dyn Fn(Value)) -> Result<()> {
    let tmp = out.with_extension("part");
    tx(json!({"file": out.file_name().and_then(|s| s.to_str()).unwrap_or(""), "index": 0, "total": 1}));
    let st = std::process::Command::new("curl").args(["-fL", "--retry", "3", "-C", "-", "-o"]).arg(&tmp).arg(url).status()
        .with_context(|| format!("spawning curl for {url}"))?;
    if cancelled(cancel) { let _ = std::fs::remove_file(&tmp); bail!("cancelled"); }
    if !st.success() { let _ = std::fs::remove_file(&tmp); bail!("curl failed downloading {url} (status {st})"); }
    std::fs::rename(&tmp, out)?;
    tx(json!({"file": "done", "index": 1, "total": 1}));
    Ok(())
}

fn post_convert(subdir: &str, dest: &Path, tx: &dyn Fn(Value)) -> Result<()> {
    match subdir {
        "realesr" => {
            for (pth, st) in [
                ("realesr_general_x4v3.pth", "realesr_general_x4v3.safetensors"),
                ("realesr_general_wdn_x4v3.pth", "realesr_general_wdn_x4v3.safetensors"),
                ("realesr_anime_x4v3.pth", "realesr_anime_x4v3.safetensors"),
            ] {
                let a = dest.join(pth); let b = dest.join(st);
                if a.is_file() && !b.is_file() {
                    tx(json!({"file": format!("converting {pth}"), "index": 0, "total": 1}));
                    convert_rrdb_pth(&a, &b)?;
                }
            }
        }
        "rife" => {
            let pkl = dest.join("flownet.pkl"); let st = dest.join("rife46.safetensors");
            if pkl.is_file() && !st.is_file() {
                tx(json!({"file": "converting RIFE flownet.pkl", "index": 0, "total": 1}));
                convert_rife_pkl(&pkl, &st)?;
            }
        }
        "isnet" => {
            // HF anime-seg / RMBG may ship keys we already understand; if only .ckpt exists, convert.
            for (src, dst) in [("isnetis.ckpt", "isnet_anime.safetensors"), ("model.pth", "isnet_rmbg.safetensors")] {
                let a = dest.join(src); let b = dest.join(dst);
                if a.is_file() && !b.is_file() {
                    tx(json!({"file": format!("converting {src}"), "index": 0, "total": 1}));
                    convert_torch_to_sft(&a, &b)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn convert_triposr(dir: &Path) -> Result<()> {
    // Prefer an existing convert.py next to the weights or under C:\Temp\qnx\int\cabin\tsr_ref
    let py = which_python();
    let script = dir.join("convert.py");
    let script = if script.is_file() { script } else {
        let alt = PathBuf::from(r"C:\Temp\qnx\int\cabin\tsr_ref\convert.py");
        if alt.is_file() { alt } else { bail!("TripoSR convert.py not found; place convert.py next to model.ckpt") }
    };
    let st = dir.join("triposr_f16.safetensors");
    let st_out = std::process::Command::new(&py).args([script.to_str().unwrap(), dir.join("model.ckpt").to_str().unwrap(), st.to_str().unwrap()]).output()
        .with_context(|| format!("running {py}"))?;
    if !st_out.status.success() { bail!("TripoSR convert failed: {}", String::from_utf8_lossy(&st_out.stderr)); }
    Ok(())
}

fn which_python() -> String {
    for c in ["python", "python3", r"C:\Users\Lyam\AppData\Local\Programs\Python\Python313\python.exe"] {
        if std::process::Command::new(c).arg("-V").output().map(|o| o.status.success()).unwrap_or(false) { return c.into(); }
    }
    "python".into()
}

/// Convert a Real-ESRGAN compact .pth (state_dict) into our flat f32 safetensors.
fn convert_rrdb_pth(pth: &Path, out: &Path) -> Result<()> {
    convert_torch_to_sft(pth, out)
}

fn convert_rife_pkl(pkl: &Path, out: &Path) -> Result<()> {
    // Practical-RIFE pickle → strip "module." prefix into safetensors (f32).
    let py = which_python();
    let code = r#"
import sys, torch
from safetensors.torch import save_file
sd=torch.load(sys.argv[1], map_location='cpu')
if isinstance(sd, dict) and 'params' in sd: sd=sd['params']
sd={k.replace('module.',''): v.float().contiguous() for k,v in sd.items() if torch.is_tensor(v)}
save_file(sd, sys.argv[2])
"#;
    let st = std::process::Command::new(&py).args(["-c", code, pkl.to_str().unwrap(), out.to_str().unwrap()]).output()?;
    if !st.status.success() { bail!("RIFE convert failed: {}", String::from_utf8_lossy(&st.stderr)); }
    Ok(())
}

fn convert_torch_to_sft(src: &Path, out: &Path) -> Result<()> {
    let py = which_python();
    let code = r#"
import sys, torch
from safetensors.torch import save_file
obj=torch.load(sys.argv[1], map_location='cpu')
if isinstance(obj, dict):
    for k in ('params_ema','params','state_dict','model','net'):
        if k in obj and isinstance(obj[k], dict): obj=obj[k]; break
sd={}
for k,v in obj.items():
    if not torch.is_tensor(v): continue
    sd[k.replace('module.','')]=v.float().contiguous()
save_file(sd, sys.argv[2])
print('wrote', len(sd), 'tensors')
"#;
    let st = std::process::Command::new(&py).args(["-c", code, src.to_str().unwrap(), out.to_str().unwrap()]).output()?;
    if !st.status.success() { bail!("convert {} failed: {}", src.display(), String::from_utf8_lossy(&st.stderr)); }
    Ok(())
}
