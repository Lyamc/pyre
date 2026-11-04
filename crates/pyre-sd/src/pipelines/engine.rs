//! One Stable Diffusion engine for every UNet family cabin runs (SD 1.x, SD 2.x, SDXL base,
//! SDXL refiner), ported from cabin's loader / CLIP Text Encode / KSampler / VAE Decode nodes
//! (cabin-nodes/src/nodes/sd.rs) onto the workspace Burn.
//!
//! What came over from cabin, beyond the plain diffusers-burn pipeline:
//! - host-seeded noise (SplitMix64 + Box-Muller), so a seed gives the same image on every backend
//!   and the same image cabin gives;
//! - samplers euler / euler_ancestral / ddim / dpmpp_2m with schedules normal / karras /
//!   exponential / sgm_uniform / simple / ddim_uniform (`sampling::schedule`);
//! - per-model dtype: weights load in f16 (or f32) regardless of the device default, the SDXL VAE
//!   runs in f32 unless an f16-safe VAE (madebyollin/sdxl-vae-fp16-fix) is supplied;
//! - SDXL micro-conditioning (time ids, aesthetic score for the refiner), pooled text embeds;
//! - partial denoise from an init latent (the refiner stage), tiled VAE decode above 1024x1024.
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use std::path::{Path, PathBuf};

use crate::models::unet_2d::{AddedCondKwargs, BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig};
use crate::models::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use crate::pipelines::stable_diffusion::{StableDiffusionConfig, STEP_HOOK, VAE_SCALE};
use crate::pipelines::stable_diffusion_xl::{StableDiffusionXLConfig, VAE_SCALE_XL};
use crate::models::controlnet::{ControlNet, ControlNetConfig};
use crate::pipelines::weights::{load_controlnet_safetensors, load_clip_safetensors, load_clip_with_projection_safetensors, load_unet_safetensors, load_vae_safetensors_as, set_load_dtype};
use crate::sampling::{noise::gaussian, schedule, tiles};
use crate::transformers::clip::{CLIPTextModelWithProjection, ClipTextTransformer};
use crate::transformers::{SimpleTokenizer, SimpleTokenizerConfig};

pub const SAMPLERS: [&str; 5] = ["euler", "euler_ancestral", "ddim", "dpmpp_2m", "lcm"];
pub use crate::sampling::schedule::SCHEDULERS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family { Sd15, Sd2, Sdxl, SdxlRefiner }

impl Family {
    pub fn name(&self) -> &'static str {
        match self { Family::Sd15 => "sd15", Family::Sd2 => "sd2", Family::Sdxl => "sdxl", Family::SdxlRefiner => "sdxl-refiner" }
    }
    pub fn label(&self) -> &'static str {
        match self { Family::Sd15 => "Stable Diffusion 1.x", Family::Sd2 => "Stable Diffusion 2.x", Family::Sdxl => "Stable Diffusion XL", Family::SdxlRefiner => "SDXL refiner" }
    }
    /// The resolution the family was trained at.
    pub fn native(&self) -> usize { match self { Family::Sd15 => 512, Family::Sd2 => 768, _ => 1024 } }

    /// Guess the family from a diffusers folder (works without model_index.json, e.g. cabin-models).
    pub fn detect(dir: &Path) -> Result<Family> {
        if let Ok(s) = std::fs::read_to_string(dir.join("unet/config.json")) {
            if let Ok(j) = serde_json_lite(&s) { if let Some(c) = j { return Ok(c); } }
        }
        let te2 = dir.join("text_encoder_2").is_dir();
        let te1 = dir.join("text_encoder").is_dir();
        if !dir.join("unet").is_dir() { bail!("{} has no unet/ (diffusers-format Stable Diffusion folder expected)", dir.display()); }
        Ok(match (te1, te2) { (true, true) => Family::Sdxl, (false, true) => Family::SdxlRefiner, _ => {
            let big = std::fs::read_to_string(dir.join("text_encoder/config.json")).map(|s| s.contains("\"hidden_size\": 1024")).unwrap_or(false);
            if big { Family::Sd2 } else { Family::Sd15 }
        } })
    }
}

/// Read `cross_attention_dim` from unet/config.json without a JSON dependency.
fn serde_json_lite(s: &str) -> Result<Option<Family>> {
    let num = |k: &str| -> Option<usize> {
        let i = s.find(&format!("\"{k}\""))?;
        let rest = &s[i + k.len() + 2..];
        let rest = rest[rest.find(':')? + 1..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    Ok(match num("cross_attention_dim") { Some(768) => Some(Family::Sd15), Some(1024) => Some(Family::Sd2), Some(2048) => Some(Family::Sdxl), Some(1280) => Some(Family::SdxlRefiner), _ => None })
}

/// `<dir>/<sub>/<stem>.safetensors`, else the `.fp16.safetensors` variant.
pub fn weights_file(dir: &Path, sub: &str, stem: &str) -> Result<PathBuf> {
    let base = dir.join(sub);
    [format!("{stem}.safetensors"), format!("{stem}.fp16.safetensors")].iter().map(|f| base.join(f)).find(|p| p.is_file())
        .ok_or_else(|| anyhow!("no {stem}(.fp16).safetensors in {}", base.display()))
}

pub fn sdxl_refiner_config() -> StableDiffusionXLConfig {
    let mut cfg = StableDiffusionXLConfig::xl(None, None);
    cfg.unet = UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig::new(384).with_use_cross_attn(false).with_attention_head_dim(6).with_transformer_layers_per_block(4),
            BlockConfig::new(768).with_use_cross_attn(true).with_attention_head_dim(12).with_transformer_layers_per_block(4),
            BlockConfig::new(1536).with_use_cross_attn(true).with_attention_head_dim(24).with_transformer_layers_per_block(4),
            BlockConfig::new(1536).with_use_cross_attn(false).with_attention_head_dim(24).with_transformer_layers_per_block(4),
        ],
        cross_attention_dim: 1280,
        use_linear_projection: true,
        addition_time_embed_dim: Some(256),
        projection_class_embeddings_input_dim: Some(2560),
        ..cfg.unet
    };
    cfg
}

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub negative: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    pub sampler: String,
    pub scheduler: String,
    /// < 1 starts from `init` partly noised (euler samplers only).
    pub denoise: f64,
    /// Inpainting: latent-resolution mask `[h/8 * w/8]`, 1 = repaint, 0 = keep `init` (blended back at every
    /// step at the matching noise level; euler samplers). Needs `init`.
    pub mask: Option<std::sync::Arc<Vec<f32>>>,
    /// 9-channel inpainting UNet (SD 1.5 inpainting): VAE latent of the image with the masked area greyed out,
    /// `[4 * h/8 * w/8]`. Set together with `mask`; the UNet then sees `[x, mask, masked_latent]` and the
    /// per-step latent blending is skipped (the model keeps the unmasked area itself).
    pub inpaint_latent: Option<std::sync::Arc<Vec<f32>>>,
    /// ControlNet hint (SD 1.5): planar RGB `[3 * height * width]` in 0..1 and the conditioning scale; needs
    /// `Engine::set_controlnet` first.
    pub control: Option<(std::sync::Arc<Vec<f32>>, f64)>,
}

pub struct Cond { embeds: Tensor<3>, pooled: Option<Tensor<2>> }

pub struct Engine {
    pub family: Family,
    pub dir: PathBuf,
    pub dtype: DType,
    pub vae_dtype: DType,
    pub vae_path: PathBuf,
    dev: Device,
    sd: StableDiffusionConfig,
    tok: SimpleTokenizer,
    clip: Option<ClipTextTransformer>,
    penultimate: bool,
    enc2: Option<(CLIPTextModelWithProjection, SimpleTokenizer)>,
    unet: Option<UNet2DConditionModel>,
    /// The resident UNet is the 9-channel inpainting one (loaded from this folder).
    unet_inpaint: Option<PathBuf>,
    controlnet: Option<(PathBuf, ControlNet)>,
    vae: AutoEncoderKL,
    vae_scale: f64,
}

fn tokens(tok: &SimpleTokenizer, text: &str, dev: &Device) -> Result<Tensor<2, Int>> {
    let ids: Vec<i64> = tok.encode(text).map_err(|e| anyhow!("tokenizer: {e}"))?.into_iter().map(|t| t as i64).collect();
    let n = ids.len();
    Ok(Tensor::<1, Int>::from_ints(ids.as_slice(), dev).reshape([1, n]))
}

fn host_tensor<const D: usize>(data: Vec<f32>, shape: [usize; D], dev: &Device, dtype: DType) -> Tensor<D> {
    Tensor::<D>::cast(Tensor::<D>::from_data(TensorData::new(data, shape), dev), dtype)
}

/// Attention slice (batch*heads per chunk) for SD1.x/2.x: bounds the attention-matrix peak so
/// 768px inpaint/outpaint fits next to the LLM (PYRE_SD_SLICE=off disables, default 8).
fn sd_slice() -> Option<usize> { match std::env::var("PYRE_SD_SLICE").ok().as_deref() { Some("off") | Some("0") => None, Some(v) => v.parse().ok(), None => Some(8) } }
fn unet_config(family: Family) -> UNet2DConditionModelConfig {
    let mut c = unet_config_base(family);
    if matches!(family, Family::Sd15 | Family::Sd2) { c.sliced_attention_size = sd_slice(); }
    c
}
fn unet_config_base(family: Family) -> UNet2DConditionModelConfig {
    match family {
        Family::Sd15 => StableDiffusionConfig::v1_5(None, None, None).unet,
        Family::Sd2 => StableDiffusionConfig::v2_1(None, None, None).unet,
        Family::Sdxl => StableDiffusionXLConfig::xl(None, None).unet,
        Family::SdxlRefiner => sdxl_refiner_config().unet,
    }
}

pub fn load_unet(dir: &Path, family: Family, dev: &Device, dtype: DType) -> Result<UNet2DConditionModel> { load_unet_ch(dir, family, dev, dtype, 4) }
/// `in_channels` of `<dir>/unet/config.json` (4 unless it says otherwise; 9 = inpainting UNet).
pub fn unet_in_channels(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("unet").join("config.json")).ok().and_then(|s| { let i = s.find("\"in_channels\"")?; let r = &s[i + 13..]; let r = r[r.find(':')? + 1..].trim_start(); r[..r.find(|c: char| !c.is_ascii_digit())?].parse().ok() }).unwrap_or(4)
}
/// `in_ch` 9 = the SD inpainting UNet (latent + mask + masked-image latent).
pub fn load_unet_ch(dir: &Path, family: Family, dev: &Device, dtype: DType, in_ch: usize) -> Result<UNet2DConditionModel> {
    let path = weights_file(dir, "unet", "diffusion_pytorch_model")?;
    set_load_dtype(Some(dtype));
    let r = load_unet_safetensors(unet_config(family).init(in_ch, 4, dev), &path, dev).map_err(|e| anyhow!("{}: {e}", path.display()));
    set_load_dtype(None);
    r
}

impl Engine {
    /// Load a diffusers folder. `vae` substitutes another VAE file (e.g. the f16-safe SDXL VAE),
    /// which then runs in `dtype`; otherwise SDXL's own VAE runs in f32 (it overflows f16).
    pub fn load(dir: &Path, dev: &Device, dtype: DType, vae: Option<&Path>) -> Result<Engine> {
        let family = Family::detect(dir)?;
        let sd = match family { Family::Sd2 => StableDiffusionConfig::v2_1(None, None, None), _ => StableDiffusionConfig::v1_5(None, None, None) };
        set_load_dtype(Some(dtype));
        let r = (|| -> Result<Engine> {
            let tok_cfg = match family { Family::Sd15 | Family::Sdxl => SimpleTokenizerConfig::v1_5(), _ => SimpleTokenizerConfig::v2_1() };
            let tok_dir = if family == Family::SdxlRefiner { "tokenizer_2" } else { "tokenizer" };
            let merges = dir.join(tok_dir).join("merges.txt");
            let tok = SimpleTokenizer::new(&merges, tok_cfg).map_err(|e| anyhow!("{}: {e}", merges.display()))?;
            let clip = match family {
                Family::SdxlRefiner => None,
                Family::Sd2 => { let p = weights_file(dir, "text_encoder", "model")?; Some(load_clip_safetensors(sd.build_clip_transformer(dev), &p, dev).map_err(|e| anyhow!("{}: {e}", p.display()))?) }
                Family::Sd15 => { let p = weights_file(dir, "text_encoder", "model")?; Some(load_clip_safetensors(sd.build_clip_transformer(dev), &p, dev).map_err(|e| anyhow!("{}: {e}", p.display()))?) }
                Family::Sdxl => { let p = weights_file(dir, "text_encoder", "model")?; Some(load_clip_safetensors(StableDiffusionXLConfig::xl(None, None).build_clip_transformer(dev), &p, dev).map_err(|e| anyhow!("{}: {e}", p.display()))?) }
            };
            let enc2 = if matches!(family, Family::Sdxl | Family::SdxlRefiner) {
                let p = weights_file(dir, "text_encoder_2", "model")?;
                let m = load_clip_with_projection_safetensors(StableDiffusionXLConfig::xl(None, None).build_clip_with_projection(dev), &p, dev).map_err(|e| anyhow!("{}: {e}", p.display()))?;
                let merges = dir.join("tokenizer_2").join("merges.txt");
                Some((m, SimpleTokenizer::new(&merges, SimpleTokenizerConfig::v2_1()).map_err(|e| anyhow!("{}: {e}", merges.display()))?))
            } else { None };
            let xl = matches!(family, Family::Sdxl | Family::SdxlRefiner);
            let (vae_path, vae_dtype) = match vae { Some(p) => (p.to_path_buf(), dtype), None => (weights_file(dir, "vae", "diffusion_pytorch_model")?, if xl { DType::F32 } else { dtype }) };
            let vae_cfg: AutoEncoderKLConfig = if xl { StableDiffusionXLConfig::xl(None, None).vae } else { sd.vae.clone() };
            let vae_m = load_vae_safetensors_as(vae_cfg.init(3, 3, dev), &vae_path, dev, Some(vae_dtype)).map_err(|e| anyhow!("{}: {e}", vae_path.display()))?;
            let unet = Some(load_unet(dir, family, dev, dtype)?);
            Ok(Engine { family, dir: dir.to_path_buf(), dtype, vae_dtype, vae_path, dev: dev.clone(), sd, tok, clip, penultimate: family != Family::Sd15, enc2, unet, unet_inpaint: None, controlnet: None, vae: vae_m, vae_scale: if xl { VAE_SCALE_XL } else { VAE_SCALE } })
        })();
        set_load_dtype(None);
        r
    }

    /// Release the UNet (the biggest part) so another one fits, e.g. the SDXL refiner's.
    pub fn unload_unet(&mut self) { self.unet = None; }
    pub fn ensure_unet(&mut self) -> Result<()> { self.select_unet(None) }
    /// Make the base UNet (None) or the 9-channel inpainting UNet in `inpaint` resident, swapping out the
    /// other one first (only one UNet's VRAM at a time).
    pub fn select_unet(&mut self, inpaint: Option<&Path>) -> Result<()> {
        if self.unet.is_some() && self.unet_inpaint.as_deref() == inpaint { return Ok(()); }
        self.unet = None; self.unet_inpaint = None; self.release_free_memory();
        self.unet = Some(match inpaint { Some(d) => load_unet_ch(d, self.family, &self.dev, self.dtype, unet_in_channels(d))?, None => load_unet(&self.dir, self.family, &self.dev, self.dtype)? });
        self.unet_inpaint = inpaint.map(Path::to_path_buf);
        Ok(())
    }
    /// Load (or drop, with None) an SD 1.5 ControlNet from a diffusers folder / safetensors file.
    pub fn set_controlnet(&mut self, path: Option<&Path>) -> Result<()> {
        if self.controlnet.as_ref().map(|c| c.0.as_path()) == path { return Ok(()); }
        self.controlnet = None; self.release_free_memory();
        if let Some(p) = path {
            let f = if p.is_dir() { weights_file(p, "", "diffusion_pytorch_model")? } else { p.to_path_buf() };
            set_load_dtype(Some(self.dtype));
            let r = load_controlnet_safetensors(ControlNetConfig::default().init(4, &self.dev), &f, &self.dev).map_err(|e| anyhow!("{}: {e}", f.display()));
            set_load_dtype(None);
            self.controlnet = Some((p.to_path_buf(), r?));
        }
        Ok(())
    }
    pub fn inpaint_unet_active(&self) -> bool { self.unet_inpaint.as_deref().map(|d| unet_in_channels(d) == 9).unwrap_or(false) }

    /// cabin's CLIP Text Encode: SD1.5 last layer, SD2 penultimate, SDXL CLIP-L penultimate ++
    /// OpenCLIP-G penultimate with the pooled projection; the refiner uses OpenCLIP-G only.
    pub fn encode(&self, text: &str, refiner: bool) -> Result<Cond> {
        if let Some((enc2, tok2)) = &self.enc2 {
            let (hidden2, pooled) = enc2.forward(tokens(tok2, text, &self.dev)?);
            if refiner || self.clip.is_none() { return Ok(Cond { embeds: hidden2, pooled: Some(pooled) }); }
            let clip = self.clip.as_ref().unwrap();
            let hidden1 = clip.forward_penultimate(tokens(&self.tok, text, &self.dev)?);
            return Ok(Cond { embeds: Tensor::cat(vec![hidden1, hidden2], 2), pooled: Some(pooled) });
        }
        let clip = self.clip.as_ref().ok_or_else(|| anyhow!("no text encoder"))?;
        let ids = tokens(&self.tok, text, &self.dev)?;
        Ok(Cond { embeds: if self.penultimate { clip.forward_penultimate(ids) } else { clip.forward(ids) }, pooled: None })
    }

    /// Sample a latent. `unet` overrides the engine's own (the refiner stage); `init` is a latent to
    /// start from at `q.denoise` (< 1).
    pub fn sample(&self, q: &Request, unet: Option<(&UNet2DConditionModel, Family)>, init: Option<&Tensor<4>>, on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor<4>> {
        let (unet, fam) = match unet { Some(u) => u, None => (self.unet.as_ref().ok_or_else(|| anyhow!("UNet not loaded"))?, self.family) };
        let refiner = fam == Family::SdxlRefiner;
        let (h, w) = (q.height / 8, q.width / 8);
        let cfg = q.guidance;
        let use_cfg = cfg > 1.0;
        let pos = self.encode(&q.prompt, refiner)?;
        let neg = if use_cfg { Some(self.encode(&q.negative, refiner)?) } else { None };
        let cond = match &neg { Some(n) => Tensor::cat(vec![n.embeds.clone(), pos.embeds.clone()], 0), None => pos.embeds.clone() };
        let batch = if use_cfg { 2 } else { 1 };
        let added = match &pos.pooled {
            None => None,
            Some(pp) => {
                let text_embeds = match &neg { Some(n) => Tensor::cat(vec![n.pooled.clone().unwrap(), pp.clone()], 0), None => pp.clone() };
                let (ph, pw) = (q.height as f32, q.width as f32);
                let data: Vec<f32> = if refiner {
                    let row = |a: f32| vec![ph, pw, 0.0, 0.0, a];
                    if use_cfg { [row(2.5), row(6.0)].concat() } else { row(6.0) }
                } else {
                    let row = vec![ph, pw, 0.0, 0.0, ph, pw];
                    if use_cfg { [row.clone(), row].concat() } else { row }
                };
                let cols = data.len() / batch;
                Some(AddedCondKwargs { text_embeds, time_ids: host_tensor(data, [batch, cols], &self.dev, self.dtype) })
            }
        };
        let prepared = schedule::sampler_inputs(q.seed, q.steps, if q.sampler == "lcm" { "euler" } else { &q.sampler }, &q.scheduler, 4, h, w)?;
        let noise = host_tensor(prepared.noise.clone(), prepared.shape, &self.dev, self.dtype);
        let hint = match (&q.control, &self.controlnet) {
            (Some((hv, sc)), Some((_, cn))) => {
                if hv.len() != 3 * q.height * q.width { bail!("control image must be {}x{}", q.width, q.height); }
                if q.inpaint_latent.is_some() { bail!("ControlNet with the 9-channel inpainting UNet is not supported"); }
                let t = host_tensor(hv.to_vec(), [1, 3, q.height, q.width], &self.dev, self.dtype);
                Some((cn, if use_cfg { Tensor::cat(vec![t.clone(), t], 0) } else { t }, *sc))
            }
            (Some(_), None) => bail!("a control image needs a ControlNet loaded"),
            _ => None,
        };
        let guided = |x: Tensor<4>, t: f64| -> Tensor<4> {
            let x2 = if use_cfg { Tensor::cat(vec![x.clone(), x], 0) } else { x };
            let eps = match &hint {
                Some((cn, h, sc)) => { let (down, mid) = cn.forward(x2.clone(), t, cond.clone(), h.clone(), *sc); unet.forward_with_additional_residuals(x2, t, cond.clone(), Some(&down), Some(&mid), added.as_ref()) }
                None => match &added { Some(a) => unet.forward_with_additional_residuals(x2, t, cond.clone(), None, None, Some(a)), None => unet.forward(x2, t, cond.clone()) },
            };
            if !use_cfg { return eps; }
            let [u, c]: [Tensor<4>; 2] = eps.chunk(2, 0).try_into().expect("batch of 2");
            u.clone() + (c - u) * cfg
        };
        let pace = |t0: std::time::Instant| if let Some(hk) = STEP_HOOK.get() { let _ = self.dev.sync(); hk(t0); };
        let denoise = q.denoise.clamp(0.0, 1.0);
        if denoise < 1.0 && !matches!(q.sampler.as_str(), "euler" | "euler_ancestral" | "lcm") { bail!("denoise below 1 is supported for euler and euler_ancestral"); }
        if denoise < 1.0 && init.is_none() { bail!("denoise below 1 needs an init latent"); }
        if q.mask.is_some() && (init.is_none() || !matches!(q.sampler.as_str(), "euler" | "euler_ancestral" | "lcm")) { bail!("inpainting needs an init image and an euler sampler"); }
        let mask = match &q.mask { Some(m) => { if m.len() != h * w { bail!("mask must be {}x{} (latent size)", w, h); } Some(host_tensor(m.to_vec(), [1, 1, h, w], &self.dev, self.dtype)) } None => None };
        // 9-channel inpainting UNet: extra input channels [mask, masked-image latent]
        let extra = match (&q.inpaint_latent, &mask) {
            (Some(l), Some(m)) => { if l.len() != 4 * h * w { bail!("inpaint latent must be 4x{h}x{w}"); } Some(Tensor::cat(vec![m.clone(), host_tensor(l.to_vec(), [1, 4, h, w], &self.dev, self.dtype)], 1)) }
            (Some(_), None) => bail!("inpaint_latent needs a mask"),
            _ => None,
        };
        let guided = |x: Tensor<4>, t: f64| -> Tensor<4> { match &extra { Some(e) => guided(Tensor::cat(vec![x, e.clone()], 1), t), None => guided(x, t) } };
        let blend = extra.is_none();
        macro_rules! run {
            ($s:expr, $tf:expr) => {{
                let ts = $s.timesteps().to_vec();
                let n = ts.len();
                let mut x = match init { Some(l) if denoise < 1.0 => l.clone(), _ => noise.clone() * $s.init_noise_sigma() };
                for (i, &t) in ts.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    let eps = guided($s.scale_model_input(x.clone(), t), $tf(t));
                    x = $s.step(&eps, t, &x);
                    pace(t0);
                    on_step(i + 1, n);
                }
                x
            }};
        }
        macro_rules! euler {
            ($s:expr) => {{
                let mut s = $s;
                s.set_schedule(prepared.timesteps.clone(), prepared.sigmas.clone().expect("euler sigmas"));
                if denoise < 1.0 || mask.is_some() {
                    let keep = ((s.timesteps().len() as f64) * denoise).round().max(1.0) as usize;
                    let drop_n = s.timesteps().len() - keep;
                    let ts: Vec<f64> = s.timesteps().iter().copied().skip(drop_n).collect();
                    let sg: Vec<f64> = s.sigmas().iter().copied().skip(drop_n).collect();
                    s.set_schedule(ts, sg);
                    let l = init.unwrap();
                    let noised = s.add_noise(l, noise.clone(), s.timesteps()[0]);
                    let ts = s.timesteps().to_vec();
                    let n = ts.len();
                    let mut x = noised;
                    for (i, &t) in ts.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        let eps = guided(s.scale_model_input(x.clone(), t), t);
                        x = s.step(&eps, t, &x);
                        if let (Some(m), true) = (&mask, blend) {
                            // keep the unmasked area on the original latent, noised to the next step level
                            let known = if i + 1 < n { s.add_noise(l, noise.clone(), ts[i + 1]) } else { l.clone() };
                            x = x * m.clone() + known * (m.clone().neg() + 1.0);
                        }
                        pace(t0);
                        on_step(i + 1, n);
                    }
                    x
                } else { run!(s, |t: f64| t) }
            }};
        }
        let x = match q.sampler.as_str() {
            "euler" => euler!(self.sd.build_euler_discrete_scheduler(q.steps)),
            "euler_ancestral" => euler!(self.sd.build_euler_ancestral_scheduler(q.steps)),
            "ddim" => {
                use crate::schedulers::ddim::{DDIMScheduler, DDIMSchedulerConfig};
                let c = DDIMSchedulerConfig { beta_start: self.sd.beta_start, beta_end: self.sd.beta_end, beta_schedule: self.sd.beta_schedule, prediction_type: self.sd.prediction_type, train_timesteps: self.sd.train_timesteps, steps_offset: schedule::ddim_offset(&q.scheduler)?, ..DDIMSchedulerConfig::default() };
                let s = DDIMScheduler::new(q.steps, c, &self.dev);
                run!(s, |t: usize| t as f64)
            }
            "lcm" => {
                // LCM multistep sampler (diffusers LCMScheduler: 50 original steps, timestep_scaling 10,
                // sigma_data 0.5, epsilon prediction); for an LCM-LoRA merged UNet, guidance ~1-2, 4-8 steps.
                let tn = self.sd.train_timesteps;
                let (b0, b1) = (self.sd.beta_start.sqrt(), self.sd.beta_end.sqrt());
                let mut ac = Vec::with_capacity(tn); let mut prod = 1.0f64;
                for i in 0..tn { let b = b0 + (b1 - b0) * i as f64 / (tn - 1) as f64; prod *= 1.0 - b * b; ac.push(prod); }
                let k = tn / 50;
                let origin: Vec<usize> = (1..=50).rev().map(|i| i * k - 1).collect();
                let n = q.steps.clamp(1, 50);
                let mut ts: Vec<usize> = (0..n).map(|i| origin[i * 50 / n]).collect();
                let keep = if denoise < 1.0 || mask.is_some() { ((n as f64) * denoise).round().max(1.0) as usize } else { n };
                ts.drain(..n - keep);
                let mut rng = q.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5DEE_CE66;
                let mut fresh = || { rng = rng.wrapping_add(1); host_tensor(crate::sampling::noise::gaussian(rng, 4 * h * w), [1, 4, h, w], &self.dev, self.dtype) };
                let noised = |l: &Tensor<4>, t: usize, nz: Tensor<4>| l.clone() * ac[t].sqrt() + nz * (1.0 - ac[t]).sqrt();
                let mut x = match init { Some(l) if keep < n || mask.is_some() => noised(l, ts[0], noise.clone()), _ => noise.clone() };
                let m_steps = ts.len();
                for (i, &t) in ts.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    let eps = guided(x.clone(), t as f64);
                    let (a, st) = (ac[t], (t * 10) as f64);
                    let c_skip = 0.25 / (st * st + 0.25);
                    let c_out = st / (st * st + 0.25).sqrt();
                    let x0 = (x.clone() - eps * (1.0 - a).sqrt()) / a.sqrt();
                    let den = x0 * c_out + x * c_skip;
                    x = if i + 1 < m_steps { noised(&den, ts[i + 1], fresh()) } else { den };
                    if let (Some(m), true, Some(l)) = (&mask, blend, init) {
                        let known = if i + 1 < m_steps { noised(l, ts[i + 1], noise.clone()) } else { l.clone() };
                        x = x * m.clone() + known * (m.clone().neg() + 1.0);
                    }
                    pace(t0);
                    on_step(i + 1, m_steps);
                }
                x
            }
            "dpmpp_2m" => { let mut s = self.sd.build_dpm_solver_scheduler(q.steps, &self.dev); run!(s, |t: usize| t as f64) }
            other => bail!("unknown sampler `{other}` (have: {})", SAMPLERS.join(", ")),
        };
        Ok(x)
    }

    /// VAE decode to RGB8, tiled (64-latent tiles, 16 overlap) above 1024x1024.
    pub fn decode(&self, latents: Tensor<4>) -> Result<(Vec<u8>, usize, usize)> {
        let [_, c, lh, lw] = latents.dims();
        let dtype = latents.dtype();
        let decode = |l: Tensor<4>| -> Tensor<4> {
            let scaled = Tensor::<4>::cast(l / self.vae_scale, self.vae_dtype);
            (Tensor::<4>::cast(self.vae.decode(scaled), dtype) / 2.0 + 0.5).clamp(0.0, 1.0)
        };
        let host = |t: Tensor<4>| -> Result<Vec<f32>> { Tensor::<4>::cast(t, DType::F32).into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}")) };
        let forced = std::env::var("PYRE_SD_VAE_TILE").ok().and_then(|v| v.parse::<usize>().ok());
        let tile = forced.unwrap_or(64);
        // 1024x1024 (128x128 latents) and up is tiled: one untiled SDXL decode at 1024 needs ~3 GB.
        let planar = if (lh * lw >= 128 * 128 || forced.is_some()) && (lh > tile || lw > tile) {
            let overlap = tile / 4;
            let plan = tiles::plan(lh, lw, tile, overlap);
            let mut blend = tiles::Blend::new(lw * 8, lh * 8);
            for t in plan.iter() {
                let img = decode(latents.clone().slice([0..1, 0..c, t.y..t.y + t.h, t.x..t.x + t.w]));
                blend.add(t.x * 8, t.y * 8, t.w * 8, t.h * 8, &host(img)?, overlap * 8);
            }
            blend.finish()
        } else { host(decode(latents))? };
        let (w, h) = (lw * 8, lh * 8);
        let hw = w * h;
        let mut rgb = Vec::with_capacity(hw * 3);
        for i in 0..hw { for ch in 0..3 { rgb.push((planar[ch * hw + i] * 255.0).round().clamp(0.0, 255.0) as u8); } }
        Ok((rgb, w, h))
    }

    /// cabin's VAE Encode (posterior mode): RGB8 pixels -> a latent in the sampler's units, for img2img.
    pub fn encode_image(&self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor<4>> {
        if w % 8 != 0 || h % 8 != 0 { bail!("image must be 8 px aligned (got {w}x{h})"); }
        let hw = w * h;
        let mut planar = vec![0f32; hw * 3];
        for i in 0..hw { for c in 0..3 { planar[c * hw + i] = rgb[i * 3 + c] as f32 / 255.0; } }
        let x = host_tensor(planar, [1, 3, h, w], &self.dev, self.vae_dtype);
        let lat = self.vae.encode(x * 2.0 - 1.0).mode() * self.vae_scale;
        Ok(Tensor::<4>::cast(lat, self.dtype))
    }

    pub fn device(&self) -> &Device { &self.dev }

    /// Give Burn's pooled-but-unused buffers back to the driver. Sampling, VAE decode and every new
    /// image size leave differently shaped buffers in the pool; on a 12 GB card SDXL weights plus
    /// those overflow into shared system memory (CUDA sysmem fallback) and steps get 2-3x slower.
    pub fn release_free_memory(&self) {
        let _ = self.dev.sync();
        self.dev.memory_cleanup();
    }
}

/// Two-stage SDXL (cabin's sdxl_base_refiner workflow, but the latent is handed over directly as in
/// diffusers' ensemble-of-experts, which skips a VAE decode+encode round trip): the base UNet is
/// released, the refiner UNet loaded, run at `denoise` over the base latent, then released again.
pub fn refine(base: &mut Engine, refiner_dir: &Path, latent: Tensor<4>, q: &Request, on_step: &mut dyn FnMut(usize, usize)) -> Result<Tensor<4>> {
    if base.family != Family::Sdxl { bail!("a refiner stage needs an SDXL base model"); }
    // Hand the base UNet's VRAM back before the refiner's 4.5 GB arrives: dropping the module only
    // returns its buffers to Burn's pool, and on a 12 GB card base + refiner + activations spill
    // into shared system memory (WDDM) and every step crawls.
    base.unload_unet();
    let _ = base.dev.sync();
    base.dev.memory_cleanup();
    let r = load_unet(refiner_dir, Family::SdxlRefiner, &base.dev, base.dtype).context("loading the SDXL refiner UNet")?;
    let out = base.sample(q, Some((&r, Family::SdxlRefiner)), Some(&latent), on_step);
    drop(r);
    let _ = base.dev.sync();
    base.dev.memory_cleanup();
    out
}
