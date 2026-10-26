//! Image-generation model kind: Stable Diffusion 1.5 via `diffusers-burn` (Burn 0.20, MIT/Apache),
//! pinned by git rev. All compute (CLIP, UNet, VAE, DDIM) runs in Burn on cuda or wgpu.
//! Weights load straight from the diffusers-format HF snapshot (zero extra disk).
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(feature = "image")]
use burn020 as burn;
#[cfg(feature = "image")]
use burn::tensor::{backend::Backend, Tensor};
#[cfg(feature = "image")]
use diffusers_burn::pipelines::{stable_diffusion::{generate_image_ddim, StableDiffusion, StableDiffusionConfig}, weights::{load_clip_safetensors, load_unet_safetensors, load_vae_safetensors}};
#[cfg(feature = "image")]
use diffusers_burn::transformers::{SimpleTokenizer, SimpleTokenizerConfig};

pub struct GenParams { pub prompt: String, pub negative: String, pub width: usize, pub height: usize, pub steps: usize, pub guidance: f64, pub seed: u64 }
pub struct Generated { pub png: Vec<u8>, pub seconds: f64, pub seed: u64 }

#[cfg(feature = "image")]
struct Pipe<B: Backend> { sd: StableDiffusion<B>, dev: B::Device }

#[cfg(feature = "image")]
enum AnyPipe { Wgpu(Pipe<burn::backend::Wgpu>), Cuda(Pipe<burn::backend::Cuda>) }

pub struct ImageModel {
    pub dir: PathBuf,
    backend: String,
    #[cfg(feature = "image")]
    pipe: Mutex<AnyPipe>,
    #[cfg(feature = "image")]
    tok: SimpleTokenizer,
    pub busy: AtomicBool,
    pub queued: AtomicU64,
}

#[cfg(feature = "image")]
fn build<B: Backend>(dir: &Path, dev: B::Device) -> Result<Pipe<B>> {
    let f = |p: &str| -> Result<PathBuf> { let x = dir.join(p); if x.is_file() { Ok(x) } else { bail!("missing {} (expected a diffusers-format SD 1.5 snapshot)", x.display()) } };
    let cfg = StableDiffusionConfig::v1_5(None, Some(512), Some(512));
    let t = std::time::Instant::now();
    let clip = load_clip_safetensors::<B, _, _>(cfg.build_clip_transformer::<B>(&dev), f("text_encoder/model.safetensors")?, &dev).map_err(|e| anyhow!("clip: {e}"))?;
    let vae = load_vae_safetensors::<B, _, _>(cfg.build_vae::<B>(&dev), f("vae/diffusion_pytorch_model.safetensors")?, &dev).map_err(|e| anyhow!("vae: {e}"))?;
    let unet = load_unet_safetensors::<B, _, _>(cfg.build_unet::<B>(&dev, 4), f("unet/diffusion_pytorch_model.safetensors")?, &dev).map_err(|e| anyhow!("unet: {e}"))?;
    tracing::info!("SD 1.5 weights loaded in {:.1}s", t.elapsed().as_secs_f64());
    Ok(Pipe { sd: StableDiffusion { clip, vae, unet, width: 512, height: 512 }, dev })
}

#[cfg(feature = "image")]
fn gen<B: Backend>(p: &mut Pipe<B>, tok: &SimpleTokenizer, q: &GenParams) -> Result<Vec<u8>> {
    let cfg = StableDiffusionConfig::v1_5(None, Some(q.height), Some(q.width));
    p.sd.width = q.width; p.sd.height = q.height;
    let sched = cfg.build_ddim_scheduler::<B>(q.steps, &p.dev);
    let pt = tok.encode(&q.prompt).map_err(|e| anyhow!("tokenize: {e}"))?;
    let nt = tok.encode(&q.negative).map_err(|e| anyhow!("tokenize: {e}"))?;
    let img: Tensor<B, 4> = generate_image_ddim(&p.sd, &sched, &pt, &nt, q.guidance, q.seed, &p.dev);
    let [_, _, h, w] = img.dims();
    let data: Vec<f32> = img.into_data().to_vec().map_err(|e| anyhow!("{e:?}"))?;
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
        let (pipe, backend) = match backend {
            "cuda" => (AnyPipe::Cuda(build(dir, Default::default())?), "cuda"),
            "wgpu" | "vulkan" => (AnyPipe::Wgpu(build(dir, Default::default())?), "wgpu"),
            other => bail!("image generation supports --backend cuda or wgpu (got '{other}'; SD on CPU would take many minutes per image)"),
        };
        Ok(Self { dir: dir.to_path_buf(), backend: backend.into(), pipe: Mutex::new(pipe), tok, busy: AtomicBool::new(false), queued: AtomicU64::new(0) })
    }

    pub fn describe(&self) -> String { format!("Stable Diffusion 1.5 (diffusers-burn, burn 0.20) on {}", self.backend) }

    #[cfg(not(feature = "image"))]
    pub fn generate(&self, _q: &GenParams) -> Result<Generated> { bail!("built without image") }

    #[cfg(feature = "image")]
    pub fn generate(&self, q: &GenParams) -> Result<Generated> {
        if q.width % 8 != 0 || q.height % 8 != 0 || q.width < 64 || q.height < 64 || q.width > 1024 || q.height > 1024 { bail!("width/height must be multiples of 8 in 64..=1024"); }
        let mut g = self.pipe.lock().unwrap();
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.busy.store(true, Ordering::SeqCst);
        let t = std::time::Instant::now();
        let r = match &mut *g { AnyPipe::Cuda(p) => gen(p, &self.tok, q), AnyPipe::Wgpu(p) => gen(p, &self.tok, q) };
        self.busy.store(false, Ordering::SeqCst);
        let png = r.context("stable diffusion")?;
        Ok(Generated { png, seconds: t.elapsed().as_secs_f64(), seed: q.seed })
    }
}
