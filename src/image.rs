//! Image-generation model kind: Stable Diffusion 1.x / 2.x / SDXL base (+ optional SDXL refiner
//! stage) via pyre-sd, whose engine is ported from cabin. All compute runs in Burn on cuda or wgpu.
//! Weights load straight from the diffusers folder (HF cache snapshot or a local folder such as
//! cabin-models): zero extra disk.
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(feature = "image")]
use pyre_sd::pipelines::engine::{self, Engine, Family, Request};

#[derive(Clone, Debug)]
pub struct GenParams {
    pub prompt: String,
    pub negative: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    /// euler (default), euler_ancestral, ddim, dpmpp_2m
    pub sampler: Option<String>,
    /// normal (default), karras, exponential, sgm_uniform, simple, ddim_uniform
    pub scheduler: Option<String>,
    /// SDXL only: run the refiner over the base latent (needs a refiner folder).
    pub refiner: bool,
    /// Refiner denoise strength (default 0.3, as in cabin's base+refiner workflow).
    pub refiner_denoise: Option<f64>,
    /// img2img: an encoded PNG/JPEG to start from; its size (rounded down to 8 px) is the output size.
    pub init_image: Option<std::sync::Arc<Vec<u8>>>,
    /// img2img denoise strength (cabin's KSampler `denoise`), 0..=1.
    pub strength: f64,
}

impl GenParams {
    pub fn new(prompt: &str, width: usize, height: usize, steps: usize, guidance: f64, seed: u64) -> Self {
        Self { prompt: prompt.into(), negative: String::new(), width, height, steps, guidance, seed, sampler: None, scheduler: None, refiner: false, refiner_denoise: None, init_image: None, strength: 1.0 }
    }
}

pub struct Generated { pub png: Vec<u8>, pub seconds: f64, pub seed: u64 }

pub struct ImageModel {
    pub dir: PathBuf,
    backend: String,
    #[cfg(feature = "image")]
    eng: Mutex<Engine>,
    pub family: &'static str,
    family_label: &'static str,
    pub native: usize,
    pub precision: &'static str,
    pub vae: String,
    pub refiner_dir: Option<PathBuf>,
    pub busy: AtomicBool,
    pub queued: AtomicU64,
    /// Steps done / total of the image in progress (for /api/status and the UI).
    pub progress: AtomicU64,
    max_area: AtomicU64,
    pub warmup_secs: Option<f64>,
}

/// An f16-safe SDXL VAE to use instead of the checkpoint's own (which must run in f32, ~30x
/// slower on the VAE passes): `PYRE_SD_VAE=<file>`, or `madebyollin/sdxl-vae-fp16-fix` in the
/// HF cache, or a cabin-style `<models>/vae/sdxl_vae_fp16_fix.safetensors` next to the checkpoint.
fn find_sdxl_vae(dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PYRE_SD_VAE") { if p == "0" || p == "none" { return None; } return Some(PathBuf::from(p)); }
    if let Some(snap) = crate::store::hfcache::find(&crate::store::hfcache::cache_dir(), "madebyollin/sdxl-vae-fp16-fix", None) {
        for f in ["diffusion_pytorch_model.safetensors", "sdxl_vae.safetensors"] { let p = snap.join(f); if p.is_file() { return Some(p); } }
    }
    let mut d = dir.parent();
    for _ in 0..3 {
        let Some(x) = d else { break };
        let p = x.join("vae").join("sdxl_vae_fp16_fix.safetensors");
        if p.is_file() { return Some(p); }
        d = x.parent();
    }
    None
}

/// The SDXL refiner for a base: `PYRE_SD_REFINER=<dir>`, the HF cache snapshot of
/// stabilityai/stable-diffusion-xl-refiner-1.0, or a sibling folder named like it.
fn find_refiner(dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PYRE_SD_REFINER") { return Some(PathBuf::from(p)).filter(|p| p.join("unet").is_dir()); }
    if let Some(s) = crate::store::hfcache::find(&crate::store::hfcache::cache_dir(), "stabilityai/stable-diffusion-xl-refiner-1.0", None) { if s.join("unet").is_dir() { return Some(s); } }
    let p = dir.parent()?.join("stable-diffusion-xl-refiner-1.0");
    p.join("unet").is_dir().then_some(p)
}

#[cfg(feature = "image")]
fn png(rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8)?;
    Ok(out)
}

impl ImageModel {
    #[cfg(not(feature = "image"))]
    pub fn load(_dir: &Path, _backend: &str) -> Result<Self> { bail!("pyre was built without the `image` feature") }

    #[cfg(feature = "image")]
    pub fn load(dir: &Path, backend: &str) -> Result<Self> {
        let (dev, _dt, label) = match backend {
            "cuda" | "wgpu" | "vulkan" => crate::backend::select(backend, "f32", None)?,
            other => bail!("image generation supports --backend cuda, wgpu or vulkan (got '{other}'; SD on CPU would take many minutes per image)"),
        };
        pyre_sd::models::fastconv::set_im2col(match std::env::var("PYRE_SD_IM2COL").as_deref() { Ok("1") => true, Ok("0") => false, _ => backend != "cuda" });
        let family = Family::detect(dir)?;
        if family == Family::SdxlRefiner { bail!("{} is an SDXL refiner; load the SDXL base and pass \"refiner\": true (the refiner is found next to it or via PYRE_SD_REFINER)", dir.display()); }
        // Weights in f16 by default (cabin: same picture as f32 to ~0.7/255 on tensor-core paths,
        // half the memory, much faster); SDXL needs it to fit 12 GB at all.
        let precision: &'static str = match std::env::var("PYRE_SD_PRECISION").as_deref() { Ok("f32") => "f32", Ok("bf16") => "bf16", _ => "f16" };
        let dtype = match precision { "f32" => burn::tensor::DType::F32, "bf16" => burn::tensor::DType::BF16, _ => burn::tensor::DType::F16 };
        let xl = family == Family::Sdxl;
        let vae = if xl { find_sdxl_vae(dir) } else { None };
        // Rough weight footprint, checked against --vram-reserve before allocating.
        let need: u64 = match (family, dtype == burn::tensor::DType::F32) { (Family::Sdxl, _) => 7_200 << 20, (_, true) => 4_100 << 20, _ => 2_100 << 20 };
        crate::limits::check_alloc(need, &format!("loading {}", family.label()))?;
        let t = std::time::Instant::now();
        let eng = Engine::load(dir, &dev, dtype, vae.as_deref())?;
        tracing::info!("{} weights loaded in {:.1}s ({precision}, VAE {} in {:?})", family.label(), t.elapsed().as_secs_f64(), eng.vae_path.display(), eng.vae_dtype);
        let refiner_dir = if xl { find_refiner(dir) } else { None };
        let vae_desc = format!("{} ({:?})", eng.vae_path.display(), eng.vae_dtype);
        let mut me = Self { dir: dir.to_path_buf(), backend: label.into(), eng: Mutex::new(eng), family: family.name(), family_label: family.label(), native: family.native(), precision, vae: vae_desc, refiner_dir,
            busy: AtomicBool::new(false), queued: AtomicU64::new(0), progress: AtomicU64::new(0), max_area: AtomicU64::new(0), warmup_secs: None };
        // Warm-up: the first image compiles + autotunes every kernel. Doing it at load time (with
        // the persistent CubeCL cache on disk) moves that cost off the first user request.
        if std::env::var("PYRE_SD_WARMUP").map(|v| v != "0").unwrap_or(true) {
            let t = std::time::Instant::now();
            me.queued.fetch_add(1, Ordering::SeqCst);
            let steps = std::env::var("PYRE_SD_WARMUP_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            let n = me.native;
            me.generate(&GenParams::new("warm-up", n, n, steps, 7.5, 0))?;
            let s = t.elapsed().as_secs_f64();
            tracing::info!("SD warm-up ({steps} steps, {n}x{n}) took {s:.1}s");
            me.warmup_secs = Some(s);
        }
        Ok(me)
    }

    pub fn describe(&self) -> String {
        format!("{} (pyre-sd on Burn/Lyamc fork, {}) on {}{}", self.family_label, self.precision, self.backend, if self.refiner_dir.is_some() { ", refiner available" } else { "" })
    }

    /// Machine-readable capabilities for /v1/models and the UIs.
    pub fn info(&self) -> serde_json::Value {
        #[cfg(feature = "image")]
        let (samplers, schedulers): (Vec<&str>, Vec<&str>) = (engine::SAMPLERS.to_vec(), engine::SCHEDULERS.to_vec());
        #[cfg(not(feature = "image"))]
        let (samplers, schedulers): (Vec<&str>, Vec<&str>) = (vec![], vec![]);
        serde_json::json!({"family": self.family, "native_size": self.native, "precision": self.precision, "vae": self.vae,
            "refiner": self.refiner_dir.as_ref().map(|p| p.display().to_string()), "samplers": samplers, "schedulers": schedulers,
            "default_sampler": "euler", "default_scheduler": "normal", "max_size": self.max_side(), "progress": self.progress.load(Ordering::SeqCst)})
    }

    fn max_side(&self) -> usize { if self.native >= 1024 { 1536 } else { 1024 } }

    #[cfg(not(feature = "image"))]
    pub fn generate(&self, _q: &GenParams) -> Result<Generated> { bail!("built without image") }

    #[cfg(feature = "image")]
    pub fn generate(&self, q: &GenParams) -> Result<Generated> {
        let mut q = q.clone();
        let init = match &q.init_image {
            Some(bytes) => {
                let img = image::load_from_memory(bytes).context("decoding the init image")?.to_rgb8();
                let (w, h) = ((img.width() as usize) / 8 * 8, (img.height() as usize) / 8 * 8);
                let img = image::imageops::crop_imm(&img, 0, 0, w as u32, h as u32).to_image();
                q.width = w; q.height = h;
                Some(img.into_raw())
            }
            None => None,
        };
        if q.width == 0 { q.width = if q.height > 0 { q.height } else { self.native }; }
        if q.height == 0 { q.height = q.width; }
        if q.steps == 0 { q.steps = 25; }
        if q.guidance < 0.0 { q.guidance = 7.5; }
        let q = &q;
        let max = self.max_side();
        if q.width % 8 != 0 || q.height % 8 != 0 || q.width < 64 || q.height < 64 || q.width > max || q.height > max { bail!("width/height must be multiples of 8 in 64..={max}"); }
        if q.steps == 0 || q.steps > 150 { bail!("steps must be 1..=150"); }
        // Burn's memory pool reuses buffers, so only a larger image than any before needs new VRAM.
        let area = (q.width * q.height) as u64;
        let seen = self.max_area.load(Ordering::SeqCst);
        // cabin measured SDXL base 1024x1024 f16 at a 9.4 GB peak with ~7 GB of weights.
        let per = if self.native >= 1024 { 2048u64 << 20 } else { 1024u64 << 20 };
        if area > seen {
            let base = (self.native * self.native) as u64;
            crate::limits::check_alloc(if seen == 0 { per * area.max(base) / base } else { per * (area - seen) / base }, "an image generation at this size")?;
        }
        let mut eng = self.eng.lock().unwrap();
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.busy.store(true, Ordering::SeqCst);
        self.progress.store(0, Ordering::SeqCst);
        let t = std::time::Instant::now();
        let r = (|| -> Result<Vec<u8>> {
            eng.ensure_unet()?;
            let req = Request { prompt: q.prompt.clone(), negative: q.negative.clone(), width: q.width, height: q.height, steps: q.steps, guidance: q.guidance, seed: q.seed,
                sampler: q.sampler.clone().unwrap_or_else(|| "euler".into()), scheduler: q.scheduler.clone().unwrap_or_else(|| "normal".into()), denoise: 1.0 };
            let prog = &self.progress;
            let mut on_step = |i: usize, n: usize| prog.store(((i as u64) << 32) | n as u64, Ordering::SeqCst);
            let mut req = req;
            let init_lat = match &init { Some(rgb) if q.strength < 1.0 => { req.denoise = q.strength.max(0.0); Some(eng.encode_image(rgb, q.width, q.height)?) } _ => None };
            if init_lat.is_some() && !matches!(req.sampler.as_str(), "euler" | "euler_ancestral") { req.sampler = "euler".into(); }
            let mut lat = eng.sample(&req, None, init_lat.as_ref(), &mut on_step)?;
            if q.refiner {
                let dir = self.refiner_dir.clone().ok_or_else(|| anyhow!("no SDXL refiner found (set PYRE_SD_REFINER or put stable-diffusion-xl-refiner-1.0 next to the base)"))?;
                let mut rq = req.clone();
                rq.sampler = if rq.sampler == "euler_ancestral" { rq.sampler } else { "euler".into() };
                rq.denoise = q.refiner_denoise.unwrap_or(0.3);
                lat = engine::refine(&mut eng, &dir, lat, &rq, &mut on_step)?;
            }
            let (rgb, w, h) = eng.decode(lat)?;
            if q.refiner { eng.ensure_unet()?; }
            png(&rgb, w, h)
        })();
        self.busy.store(false, Ordering::SeqCst);
        let png = r.context("stable diffusion")?;
        self.max_area.fetch_max(area, Ordering::SeqCst);
        Ok(Generated { png, seconds: t.elapsed().as_secs_f64(), seed: q.seed })
    }
}
