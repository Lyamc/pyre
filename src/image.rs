//! Image-generation model kind: Stable Diffusion 1.5 via pyre-sd (vendored diffusers-burn, MIT/Apache),
//! ported to the workspace Burn. All compute (CLIP, UNet, VAE, DDIM) runs in Burn on cuda or wgpu.
//! Weights load straight from the diffusers-format HF snapshot (zero extra disk).
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(feature = "image")]
use burn::tensor::{Device, Tensor};
#[cfg(feature = "image")]
use pyre_sd::pipelines::{stable_diffusion::{generate_image_ddim, StableDiffusion, StableDiffusionConfig}, weights::{load_clip_safetensors, load_unet_safetensors, load_vae_safetensors}};
#[cfg(feature = "image")]
use pyre_sd::transformers::{SimpleTokenizer, SimpleTokenizerConfig};

pub struct GenParams { pub prompt: String, pub negative: String, pub width: usize, pub height: usize, pub steps: usize, pub guidance: f64, pub seed: u64 }
pub struct Generated { pub png: Vec<u8>, pub seconds: f64, pub seed: u64 }

/// One Burn for everything: the SD pipeline is backend-agnostic and runs on whatever `Device`
/// pyre selected (`cuda`, `wgpu`, `vulkan`).
#[cfg(feature = "image")]
struct Pipe { sd: StableDiffusion, dev: Device }

pub struct ImageModel {
    pub dir: PathBuf,
    backend: String,
    #[cfg(feature = "image")]
    pipe: Mutex<Pipe>,
    #[cfg(feature = "image")]
    tok: SimpleTokenizer,
    pub busy: AtomicBool,
    pub queued: AtomicU64,
    max_area: AtomicU64,
    /// Wall time of the load-time warm-up image (kernel compile / autotune), if one ran.
    pub warmup_secs: Option<f64>,
}

#[cfg(feature = "image")]
fn build(dir: &Path, dev: Device) -> Result<Pipe> {
    let f = |p: &str| -> Result<PathBuf> { let x = dir.join(p); if x.is_file() { Ok(x) } else { bail!("missing {} (expected a diffusers-format SD 1.5 snapshot)", x.display()) } };
    let cfg = StableDiffusionConfig::v1_5(None, Some(512), Some(512));
    let t = std::time::Instant::now();
    let clip = load_clip_safetensors(cfg.build_clip_transformer(&dev), f("text_encoder/model.safetensors")?, &dev).map_err(|e| anyhow!("clip: {e}"))?;
    let vae = load_vae_safetensors(cfg.build_vae(&dev), f("vae/diffusion_pytorch_model.safetensors")?, &dev).map_err(|e| anyhow!("vae: {e}"))?;
    let unet = load_unet_safetensors(cfg.build_unet(&dev, 4), f("unet/diffusion_pytorch_model.safetensors")?, &dev).map_err(|e| anyhow!("unet: {e}"))?;
    tracing::info!("SD 1.5 weights loaded in {:.1}s", t.elapsed().as_secs_f64());
    Ok(Pipe { sd: StableDiffusion { clip, vae, unet, width: 512, height: 512 }, dev })
}

#[cfg(feature = "image")]
fn gen(p: &mut Pipe, tok: &SimpleTokenizer, q: &GenParams) -> Result<Vec<u8>> {
    let cfg = StableDiffusionConfig::v1_5(None, Some(q.height), Some(q.width));
    p.sd.width = q.width; p.sd.height = q.height;
    let sched = cfg.build_ddim_scheduler(q.steps, &p.dev);
    let pt = tok.encode(&q.prompt).map_err(|e| anyhow!("tokenize: {e}"))?;
    let nt = tok.encode(&q.negative).map_err(|e| anyhow!("tokenize: {e}"))?;
    let img: Tensor<4> = generate_image_ddim(&p.sd, &sched, &pt, &nt, q.guidance, q.seed, &p.dev);
    let [_, _, h, w] = img.dims();
    let data: Vec<f32> = img.cast(burn::tensor::FloatDType::F32).into_data().to_vec().map_err(|e| anyhow!("{e:?}"))?;
    let hw = h * w;
    let mut rgb = Vec::with_capacity(hw * 3);
    for i in 0..hw { for c in 0..3 { rgb.push((data[c * hw + i] * 255.0).round().clamp(0.0, 255.0) as u8); } }
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(&rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8)?;
    Ok(out)
}
#[cfg(feature = "image")]
use image::ImageEncoder;

impl ImageModel {
    #[cfg(not(feature = "image"))]
    pub fn load(_dir: &Path, _backend: &str) -> Result<Self> { bail!("pyre was built without the `image` feature") }

    #[cfg(feature = "image")]
    pub fn load(dir: &Path, backend: &str) -> Result<Self> {
        let merges = dir.join("tokenizer/merges.txt");
        let tok = SimpleTokenizer::new(&merges, SimpleTokenizerConfig::v1_5()).map_err(|e| anyhow!("CLIP tokenizer from {}: {e}", merges.display()))?;
        let (dev, _dt, label) = match backend {
            "cuda" | "wgpu" | "vulkan" => crate::backend::select(backend, "f32", None)?,
            other => bail!("image generation supports --backend cuda, wgpu or vulkan (got '{other}'; SD on CPU would take many minutes per image)"),
        };
        let pipe = build(dir, dev)?;
        let mut me = Self { dir: dir.to_path_buf(), backend: label.into(), pipe: Mutex::new(pipe), tok, busy: AtomicBool::new(false), queued: AtomicU64::new(0), max_area: AtomicU64::new(0), warmup_secs: None };
        // Warm-up: the first image compiles + autotunes every kernel. Doing it at load time (with
        // the persistent CubeCL cache on disk) moves that cost off the first user request.
        if std::env::var("PYRE_SD_WARMUP").map(|v| v != "0").unwrap_or(true) {
            let t = std::time::Instant::now();
            me.queued.fetch_add(1, Ordering::SeqCst);
            let steps = std::env::var("PYRE_SD_WARMUP_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            me.generate(&GenParams { prompt: "warm-up".into(), negative: String::new(), width: 512, height: 512, steps, guidance: 7.5, seed: 0 })?;
            let s = t.elapsed().as_secs_f64();
            tracing::info!("SD warm-up ({steps} steps, 512x512) took {s:.1}s");
            me.warmup_secs = Some(s);
        }
        Ok(me)
    }

    pub fn describe(&self) -> String { format!("Stable Diffusion 1.5 (pyre-sd on Burn/Lyamc fork) on {}", self.backend) }

    #[cfg(not(feature = "image"))]
    pub fn generate(&self, _q: &GenParams) -> Result<Generated> { bail!("built without image") }

    #[cfg(feature = "image")]
    pub fn generate(&self, q: &GenParams) -> Result<Generated> {
        if q.width % 8 != 0 || q.height % 8 != 0 || q.width < 64 || q.height < 64 || q.width > 1024 || q.height > 1024 { bail!("width/height must be multiples of 8 in 64..=1024"); }
        // Burn's memory pool reuses buffers, so only a larger image than any before needs new VRAM.
        let area = (q.width * q.height) as u64;
        let seen = self.max_area.load(Ordering::SeqCst);
        if area > seen {
            crate::limits::check_alloc(if seen == 0 { 2048 << 20 } else { (2048u64 << 20) * (area - seen) / seen.max(1) }, "an image generation at this size")?;
        }
        let mut g = self.pipe.lock().unwrap();
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.busy.store(true, Ordering::SeqCst);
        let t = std::time::Instant::now();
        let r = gen(&mut g, &self.tok, q);
        self.busy.store(false, Ordering::SeqCst);
        let png = r.context("stable diffusion")?;
        self.max_area.fetch_max(area, Ordering::SeqCst);
        Ok(Generated { png, seconds: t.elapsed().as_secs_f64(), seed: q.seed })
    }
}
