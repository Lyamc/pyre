//! Image-tagging kind (`tagger`, task `image-classification`): EVA02-Large Danbooru taggers
//! such as Redstonexs/kagami-24k (24,000 general tags) written natively on Burn tensors.
//!
//! Architecture = timm `eva02_large_patch14_448` (24 blocks, 1024 wide, 16 heads, 2-D RoPE on
//! the patch tokens, SwiGLU MLP with an inner LayerNorm, average pool over patch tokens, fc_norm,
//! linear head, sigmoid). Weights load in place from the HF cache: the repo's PyTorch
//! `weights.pt` (read with the Burn fork's own pytorch-reader, no conversion) or a timm
//! `model.safetensors`. Tags (and Danbooru categories, if present) come from `selected_tags.csv`.
//!
//! Preprocessing follows the model card's `predict.py`: alpha over white, white square pad
//! (centred), bicubic resize to 448, x/255 -> [-1, 1], RGB -> BGR, NCHW.
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::module::{attention, layer_norm, linear};
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorCreationOptions, TensorData};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

pub const SIDE: usize = 448;
const PATCH: usize = 14;
const GRID: usize = SIDE / PATCH; // 32
const NPATCH: usize = GRID * GRID; // 1024
const PDIM: usize = 3 * PATCH * PATCH; // 588
const HEADS: usize = 16;
const HD: usize = 64;
const EPS: f64 = 1e-6;

/// Model-card defaults (Kagami-24k): 0.37 maximises macro-F1, 0.54 micro-F1.
pub const DEFAULT_THRESHOLD: f32 = 0.37;
pub const DEFAULT_CHARACTER_THRESHOLD: f32 = 0.37;

/// Tags always reported with their score (if in the vocabulary), whatever the threshold:
/// censorship, content-rating evidence and manga-format tags, so callers can filter.
pub const WATCH: &[&str] = &[
    "censored", "uncensored", "mosaic_censoring", "bar_censor", "convenient_censoring", "heart_censor", "novelty_censor",
    "blur_censor", "light_censor", "steam_censor", "censored_nipples", "pointless_censoring", "identity_censor",
    "nude", "completely_nude", "nipples", "sex", "penis", "pussy", "hetero", "yaoi", "yuri", "loli", "shota", "child",
    "blood", "guro", "injury", "monochrome", "greyscale", "comic", "speech_bubble", "4koma", "cover", "cover_page", "manga_cover",
];

/// Danbooru-style rating evidence (used when the vocabulary has no rating tags, e.g. Kagami-24k).
const EXPLICIT: &[&str] = &["sex", "vaginal", "anal", "oral", "fellatio", "cunnilingus", "paizuri", "handjob", "footjob", "penis", "pussy",
    "clitoris", "testicles", "erection", "cum", "cum_in_pussy", "cum_on_body", "cum_in_mouth", "ejaculation", "masturbation", "fingering",
    "sex_from_behind", "cowgirl_position", "missionary", "group_sex", "spread_pussy", "pussy_juice", "anus", "rape", "imminent_penetration",
    "mosaic_censoring", "bar_censor", "censored", "uncensored", "after_sex", "cumdrip", "nipple_tweak", "breast_sucking", "sex_toy", "dildo", "vibrator"];
const QUESTIONABLE: &[&str] = &["nipples", "nude", "completely_nude", "topless", "bottomless", "areolae", "breasts_out", "pubic_hair",
    "convenient_censoring", "nipple_slip", "cameltoe", "no_panties", "heart_censor", "censored_nipples", "steam_censor", "light_censor", "groin", "covering_breasts", "covering_crotch"];
const SENSITIVE: &[&str] = &["cleavage", "swimsuit", "bikini", "underwear", "panties", "lingerie", "bra", "pantyshot", "underwear_only",
    "ass", "large_breasts", "huge_breasts", "see-through", "bdsm", "bondage", "towel", "naked_towel", "partially_undressed", "undressing", "skindentation"];

struct Lin { w: Tensor<2>, b: Option<Tensor<1>> }
impl Lin { fn fwd<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> { linear(x, self.w.clone(), self.b.clone()) } }
struct Ln { g: Tensor<1>, b: Tensor<1> }
impl Ln { fn fwd<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> { layer_norm(x, self.g.clone(), Some(self.b.clone()), EPS) } }

struct Block { n1: Ln, qkv: Lin, proj: Lin, n2: Ln, fc1: Lin, hidden: usize, mnorm: Ln, fc2: Lin }

pub struct Eva02 {
    patch: Lin,
    cls: Tensor<3>,
    pos: Tensor<3>,
    blocks: Vec<Block>,
    fc_norm: Ln,
    head: Lin,
    cos: Tensor<4>,
    sin: Tensor<4>,
    dim: usize,
    pub n_tags: usize,
    device: Device,
    dtype: DType,
}

/// Host-side weight source: PyTorch checkpoint (`weights.pt`, key `model`) or safetensors.
enum Src { Pt(HashMap<String, pytorch_reader::Tensor>), St(memmap2::Mmap) }
impl Src {
    fn open(dir: &Path) -> Result<(Self, PathBuf)> {
        let st = dir.join("model.safetensors");
        if st.is_file() {
            let f = std::fs::File::open(&st)?;
            return Ok((Src::St(unsafe { memmap2::Mmap::map(&f)? }), st));
        }
        let pt = ["weights.pt", "model.pt", "pytorch_model.bin"].iter().map(|f| dir.join(f)).find(|p| p.is_file())
            .with_context(|| format!("no model.safetensors / weights.pt in {}", dir.display()))?;
        let r = match pytorch_reader::PytorchReader::with_top_level_key(&pt, "model") {
            Ok(r) => r,
            Err(_) => pytorch_reader::PytorchReader::new(&pt).map_err(|e| anyhow!("{}: {e}", pt.display()))?,
        };
        Ok((Src::Pt(r.into_tensors()), pt))
    }
    fn has(&self, name: &str) -> bool {
        match self { Src::Pt(m) => m.contains_key(name), Src::St(m) => safetensors::SafeTensors::deserialize(m).map(|s| s.tensor(name).is_ok()).unwrap_or(false) }
    }
    /// f32 values + shape.
    fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (bytes, shape, dt): (Vec<u8>, Vec<usize>, &str) = match self {
            Src::Pt(m) => {
                let t = m.get(name).with_context(|| format!("missing tensor {name}"))?;
                let dt = match t.dtype() { pytorch_reader::DType::F32 => "f32", pytorch_reader::DType::F16 => "f16", pytorch_reader::DType::BF16 => "bf16", d => bail!("{name}: unsupported dtype {d:?}") };
                (t.read().with_context(|| format!("reading {name}"))?, t.shape().to_vec(), dt)
            }
            Src::St(m) => {
                let s = safetensors::SafeTensors::deserialize(m)?;
                let v = s.tensor(name).with_context(|| format!("missing tensor {name}"))?;
                let dt = match v.dtype() { safetensors::Dtype::F32 => "f32", safetensors::Dtype::F16 => "f16", safetensors::Dtype::BF16 => "bf16", d => bail!("{name}: unsupported dtype {d:?}") };
                (v.data().to_vec(), v.shape().to_vec(), dt)
            }
        };
        let v: Vec<f32> = match dt {
            "f32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "f16" => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        };
        Ok((v, shape))
    }
}

/// `[out, in]` row-major -> `[in, out]` row-major (Burn `linear` wants `[in, out]`).
fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for r in 0..rows { for c in 0..cols { o[c * rows + r] = v[r * cols + c]; } }
    o
}

impl Eva02 {
    fn up<const D: usize>(&self, v: Vec<f32>, shape: [usize; D]) -> Tensor<D> {
        Tensor::from_data(TensorData::new(v, shape), TensorCreationOptions::new(self.device.clone()).with_dtype(self.dtype))
    }
    /// Linear from several `[out_i, in]` weights stacked along `out` (fused q/k/v, gate/up).
    fn lin(&self, src: &Src, names: &[&str], bias: bool) -> Result<Lin> {
        let mut w = vec![]; let mut b = vec![]; let mut outs = 0; let mut inn = 0;
        for n in names {
            let (v, s) = src.f32(&format!("{n}.weight"))?;
            if s.len() != 2 { bail!("{n}.weight: expected rank 2, got {s:?}"); }
            inn = s[1]; outs += s[0];
            w.extend(v);
            if bias {
                let bn = format!("{n}.bias");
                if src.has(&bn) { b.extend(src.f32(&bn)?.0); } else { b.extend(std::iter::repeat_n(0f32, s[0])); }
            }
        }
        let wt = transpose(&w, outs, inn);
        Ok(Lin { w: self.up(wt, [inn, outs]), b: if bias { Some(self.up(b, [outs])) } else { None } })
    }
    fn ln(&self, src: &Src, n: &str) -> Result<Ln> {
        let (g, s) = src.f32(&format!("{n}.weight"))?;
        let (b, _) = src.f32(&format!("{n}.bias"))?;
        Ok(Ln { g: self.up(g, [s[0]]), b: self.up(b, [s[0]]) })
    }

    pub fn load(dir: &Path, device: &Device, dtype: DType) -> Result<(Self, PathBuf)> {
        let (src, file) = Src::open(dir)?;
        let (pw, ps) = src.f32("patch_embed.proj.weight")?;
        if ps != [ps[0], 3, PATCH, PATCH] { bail!("patch_embed.proj.weight {ps:?}: expected [D, 3, 14, 14] (EVA02 patch-14 tagger)"); }
        let dim = ps[0];
        if dim % HEADS != 0 || dim / HEADS != HD { bail!("embed dim {dim}: expected 16 heads of 64 (EVA02-Large)"); }
        let mut me = Self {
            patch: Lin { w: Tensor::zeros([1, 1], device), b: None }, cls: Tensor::zeros([1, 1, 1], device), pos: Tensor::zeros([1, 1, 1], device),
            blocks: vec![], fc_norm: Ln { g: Tensor::zeros([1], device), b: Tensor::zeros([1], device) }, head: Lin { w: Tensor::zeros([1, 1], device), b: None },
            cos: Tensor::zeros([1, 1, 1, 1], device), sin: Tensor::zeros([1, 1, 1, 1], device), dim, n_tags: 0, device: device.clone(), dtype,
        };
        // patch embedding as a matmul over host-unfolded patches (c, ky, kx order = conv weight layout)
        let (pb, _) = src.f32("patch_embed.proj.bias")?;
        me.patch = Lin { w: me.up(transpose(&pw, dim, PDIM), [PDIM, dim]), b: Some(me.up(pb, [dim])) };
        me.cls = me.up(src.f32("cls_token")?.0, [1, 1, dim]);
        let (pos, pshape) = src.f32("pos_embed")?;
        if pshape != [1, NPATCH + 1, dim] { bail!("pos_embed {pshape:?}: expected [1, 1025, {dim}] (448 px input)"); }
        me.pos = me.up(pos, [1, NPATCH + 1, dim]);
        let mut i = 0;
        while src.has(&format!("blocks.{i}.norm1.weight")) {
            let p = |s: &str| format!("blocks.{i}.{s}");
            let (fc1, hidden) = if src.has(&p("mlp.fc1_g.weight")) {
                let h = src.f32(&p("mlp.fc1_g.bias"))?.0.len();
                (me.lin(&src, &[&p("mlp.fc1_g"), &p("mlp.fc1_x")], true)?, h)
            } else { bail!("block {i}: expected a SwiGLU MLP (mlp.fc1_g / fc1_x)") };
            if !src.has(&p("attn.q_proj.weight")) { bail!("block {i}: expected separate q/k/v projections"); }
            me.blocks.push(Block {
                n1: me.ln(&src, &p("norm1"))?,
                qkv: me.lin(&src, &[&p("attn.q_proj"), &p("attn.k_proj"), &p("attn.v_proj")], true)?,
                proj: me.lin(&src, &[&p("attn.proj")], true)?,
                n2: me.ln(&src, &p("norm2"))?,
                fc1, hidden,
                mnorm: me.ln(&src, &p("mlp.norm"))?,
                fc2: me.lin(&src, &[&p("mlp.fc2")], true)?,
            });
            i += 1;
        }
        if me.blocks.is_empty() { bail!("no transformer blocks found"); }
        if src.has("norm.weight") { bail!("final `norm` (token pooling) variant is not supported; expected avg pool + fc_norm"); }
        me.fc_norm = me.ln(&src, "fc_norm")?;
        me.head = me.lin(&src, &["head"], true)?;
        me.n_tags = src.f32("head.bias")?.0.len();
        let (c, s) = rope_tables();
        me.cos = me.up(c, [1, 1, NPATCH + 1, HD]);
        me.sin = me.up(s, [1, 1, NPATCH + 1, HD]);
        Ok((me, file))
    }

    /// `rot(x)`: [x0, x1, x2, x3, ..] -> [-x1, x0, -x3, x2, ..] (timm interleaved RoPE).
    fn rope(&self, t: Tensor<4>) -> Tensor<4> {
        let [b, h, n, d] = t.dims();
        let p: Tensor<5> = t.clone().reshape([b, h, n, d / 2, 2]);
        let e = p.clone().narrow(4, 0, 1);
        let o = p.narrow(4, 1, 1);
        let r: Tensor<4> = Tensor::cat(vec![o.neg(), e], 4).reshape([b, h, n, d]);
        t * self.cos.clone() + r * self.sin.clone()
    }

    /// `patches`: `[B, 1024, 588]` normalised BGR patches. Returns `[B, n_tags]` probabilities.
    pub fn forward(&self, patches: Vec<f32>, b: usize) -> Vec<f32> {
        let d = self.dim;
        let x: Tensor<3> = self.up(patches, [b, NPATCH, PDIM]);
        let x = self.patch.fwd(x);
        let cls = self.cls.clone().expand([b, 1, d]);
        let mut x = Tensor::cat(vec![cls, x], 1) + self.pos.clone();
        let n = NPATCH + 1;
        let opts = AttentionModuleOptions { scale: None, softcap: None, is_causal: false, causal_alignment: Default::default() };
        for blk in &self.blocks {
            let h = blk.n1.fwd(x.clone());
            let qkv = blk.qkv.fwd(h);
            let split = |o: usize| -> Tensor<4> { qkv.clone().narrow(2, o, d).reshape([b, n, HEADS, HD]).swap_dims(1, 2) };
            let (q, k, v) = (self.rope(split(0)), self.rope(split(d)), split(2 * d));
            let a = attention(q, k, v, None, None, opts.clone()).swap_dims(1, 2).reshape([b, n, d]);
            x = x + blk.proj.fwd(a);
            let h = blk.fc1.fwd(blk.n2.fwd(x.clone()));
            let g = h.clone().narrow(2, 0, blk.hidden);
            let u = h.narrow(2, blk.hidden, blk.hidden);
            x = x + blk.fc2.fwd(blk.mnorm.fwd(silu(g) * u));
        }
        let pooled: Tensor<2> = x.narrow(1, 1, NPATCH).mean_dim(1).reshape([b, d]);
        let p = sigmoid(self.head.fwd(self.fc_norm.fwd(pooled))).cast(FloatDType::F32);
        p.into_data().convert::<f32>().try_into_vec::<f32>().expect("tagger probabilities")
    }
}

/// timm RotaryEmbeddingCat for eva02 @448 (dim 64, 16 freq bands per axis, temperature 1e4, 'ij'
/// indexing, interleaved layout) with an identity row prepended for the cls token. NOTE: Kagami's
/// shipped ONNX (the model card's reference) uses the raw 32x32 grid (ref_feat_shape = feat_shape),
/// not timm's default 16x16 rescale; with the rescale the outputs differ by up to 0.96.
fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let nb = HD / 4; // 16
    let bands: Vec<f64> = (0..nb).map(|i| 1.0 / 10000f64.powf(i as f64 / nb as f64)).collect();
    let mut cos = vec![1f32; HD];
    let mut sin = vec![0f32; HD];
    for y in 0..GRID {
        for x in 0..GRID {
            let (ty, tx) = (y as f32, x as f32);
            for f in 0..2 * nb {
                let pos = if f < nb { ty * bands[f] as f32 } else { tx * bands[f - nb] as f32 };
                for _ in 0..2 { cos.push(pos.cos()); sin.push(pos.sin()); }
            }
        }
    }
    (cos, sin)
}

/// Decode + preprocess one image into `[1024, 588]` BGR patches in [-1, 1].
#[cfg(feature = "image")]
pub fn preprocess(bytes: &[u8], bgr: bool) -> Result<Vec<f32>> {
    let img = image::load_from_memory(bytes).context("decoding image")?;
    Ok(preprocess_img(img, bgr))
}

#[cfg(feature = "image")]
pub fn preprocess_img(img: image::DynamicImage, bgr: bool) -> Vec<f32> {
    use image::{imageops, Rgb, RgbImage};
    let rgb: RgbImage = if img.color().has_alpha() {
        let a = img.to_rgba8();
        let mut o = RgbImage::new(a.width(), a.height());
        for (p, q) in a.pixels().zip(o.pixels_mut()) {
            let al = p[3] as u32;
            *q = Rgb([0, 1, 2].map(|c| ((p[c] as u32 * al + 255 * (255 - al) + 127) / 255) as u8));
        }
        o
    } else { img.to_rgb8() };
    let (w, h) = (rgb.width(), rgb.height());
    let s = w.max(h);
    let sq = if w == h { rgb } else {
        let mut bg = RgbImage::from_pixel(s, s, Rgb([255, 255, 255]));
        imageops::replace(&mut bg, &rgb, ((s - w) / 2) as i64, ((s - h) / 2) as i64);
        bg
    };
    let raw: Vec<u8> = if s as usize == SIDE { sq.into_raw() } else { pil_bicubic(sq.as_raw(), s as usize, s as usize, SIDE, SIDE) };
    let mut out = vec![0f32; NPATCH * PDIM];
    for py in 0..GRID { for px in 0..GRID {
        let base = (py * GRID + px) * PDIM;
        for c in 0..3 {
            let sc = if bgr { 2 - c } else { c };
            for ky in 0..PATCH { for kx in 0..PATCH {
                let (yy, xx) = (py * PATCH + ky, px * PATCH + kx);
                out[base + c * PATCH * PATCH + ky * PATCH + kx] = raw[(yy * SIDE + xx) * 3 + sc] as f32 / 127.5 - 1.0;
            }}
        }
    }}
    out
}

/// Pillow's `Image.resize(..., BICUBIC)` for 8-bit RGB, reproduced exactly (Resample.c: a = -0.5
/// cubic, support scaled by the downscale factor, 22-bit fixed-point coefficients, horizontal pass
/// then vertical, each rounded and clipped to u8). The tagger was trained on PIL-resized input;
/// a generic Catmull-Rom resize moved probabilities by up to ~0.03.
pub fn pil_bicubic(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    const PB: u32 = 22;
    fn filt(x: f64) -> f64 {
        let a = -0.5; let x = x.abs();
        if x < 1.0 { ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 } else if x < 2.0 { (((x - 5.0) * x + 8.0) * x - 4.0) * a } else { 0.0 }
    }
    fn coeffs(insz: usize, outsz: usize) -> Vec<(usize, Vec<i64>)> {
        let scale = insz as f64 / outsz as f64;
        let fs = scale.max(1.0);
        let support = 2.0 * fs;
        (0..outsz).map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / fs;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(insz as i64) as usize;
            let mut k: Vec<f64> = (xmin..xmax).map(|x| filt((x as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 { for v in &mut k { *v /= ww; } }
            (xmin, k.iter().map(|&v| (v * (1u64 << PB) as f64 + if v < 0.0 { -0.5 } else { 0.5 }) as i64).collect())
        }).collect()
    }
    let clip = |ss: i64| -> u8 { (ss >> PB).clamp(0, 255) as u8 };
    let ch = coeffs(w, ow);
    let mut tmp = vec![0u8; ow * h * 3];
    for y in 0..h { for (x, (x0, k)) in ch.iter().enumerate() { for c in 0..3 {
        let mut ss: i64 = 1 << (PB - 1);
        for (i, kv) in k.iter().enumerate() { ss += src[(y * w + x0 + i) * 3 + c] as i64 * kv; }
        tmp[(y * ow + x) * 3 + c] = clip(ss);
    }}}
    let cv = coeffs(h, oh);
    let mut out = vec![0u8; ow * oh * 3];
    for (y, (y0, k)) in cv.iter().enumerate() { for x in 0..ow { for c in 0..3 {
        let mut ss: i64 = 1 << (PB - 1);
        for (i, kv) in k.iter().enumerate() { ss += tmp[((y0 + i) * ow + x) * 3 + c] as i64 * kv; }
        out[(y * ow + x) * 3 + c] = clip(ss);
    }}}
    out
}

pub struct TagEntry { pub name: String, pub category: u8 }

pub fn read_tags(dir: &Path) -> Result<Vec<TagEntry>> {
    let p = dir.join("selected_tags.csv");
    let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    let mut lines = s.lines();
    let hdr: Vec<&str> = lines.next().context("empty selected_tags.csv")?.split(',').map(|h| h.trim()).collect();
    let ni = hdr.iter().position(|h| *h == "name").context("selected_tags.csv has no 'name' column")?;
    let ci = hdr.iter().position(|h| *h == "category");
    let mut v = vec![];
    for l in lines {
        if l.trim().is_empty() { continue; }
        let f = split_csv(l);
        v.push(TagEntry { name: f.get(ni).cloned().unwrap_or_default(), category: ci.and_then(|c| f.get(c)).and_then(|c| c.parse().ok()).unwrap_or(0) });
    }
    Ok(v)
}

fn split_csv(l: &str) -> Vec<String> {
    let (mut out, mut cur, mut q) = (vec![], String::new(), false);
    let mut it = l.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '"' if q && it.peek() == Some(&'"') => { cur.push('"'); it.next(); }
            '"' => q = !q,
            ',' if !q => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

pub fn category_name(c: u8) -> &'static str {
    match c { 0 => "general", 1 => "artist", 3 => "copyright", 4 => "character", 5 => "meta", 9 => "rating", _ => "other" }
}

#[derive(Clone, Debug)]
pub struct TagOpts { pub threshold: f32, pub character_threshold: f32, pub top_k: Option<usize>, pub watch: Vec<String>, pub rating_threshold: f32 }
impl Default for TagOpts {
    fn default() -> Self { Self { threshold: DEFAULT_THRESHOLD, character_threshold: DEFAULT_CHARACTER_THRESHOLD, top_k: None, watch: WATCH.iter().map(|s| s.to_string()).collect(), rating_threshold: 0.5 } }
}

pub struct TaggerModel {
    #[allow(dead_code)]
    pub dir: PathBuf,
    pub weights: PathBuf,
    backend: String,
    pub precision: &'static str,
    net: Mutex<Eva02>,
    pub tags: Vec<TagEntry>,
    index: HashMap<String, usize>,
    bgr: bool,
    pub busy: AtomicBool,
    pub queued: AtomicU64,
    pub max_batch: usize,
    pub warmup_secs: Option<f64>,
}

impl TaggerModel {
    pub fn load(dir: &Path, backend: &str) -> Result<Self> {
        let precision: &'static str = match std::env::var("PYRE_TAGGER_PRECISION").as_deref() { Ok("f16") => "f16", Ok("bf16") => "bf16", _ => "f32" };
        let (dev, _, label) = crate::backend::select(backend, "f32", None)?;
        let dtype = match precision { "f16" => DType::F16, "bf16" => DType::BF16, _ => DType::F32 };
        let gpu = matches!(label, "cuda" | "wgpu" | "vulkan");
        if gpu { crate::limits::check_alloc(if precision == "f32" { 1_700 << 20 } else { 1_000 << 20 }, "loading the EVA02 tagger")?; }
        if precision != "f32" { tracing::warn!("PYRE_TAGGER_PRECISION={precision}: EVA02 is NOT accurate in half precision on this engine (f16 measured up to 0.97 abs prob error vs the reference); use f32 (default) for real results"); }
        let tags = read_tags(dir)?;
        let t = std::time::Instant::now();
        let (net, weights) = Eva02::load(dir, &dev, dtype)?;
        if net.n_tags != tags.len() { bail!("head has {} outputs but selected_tags.csv lists {} tags", net.n_tags, tags.len()); }
        tracing::info!("EVA02 tagger: {} blocks, {} tags from {} in {:.1}s ({precision} on {label})", net.blocks.len(), tags.len(), weights.display(), t.elapsed().as_secs_f64());
        let index = tags.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        // Kagami (predict.py / model card) and WD's own pipeline feed BGR.
        let bgr = std::env::var("PYRE_TAGGER_RGB").map(|v| v == "0").unwrap_or(true);
        let max_batch = std::env::var("PYRE_TAGGER_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize).clamp(1, 32);
        let mut me = Self { dir: dir.to_path_buf(), weights, backend: label.into(), precision, net: Mutex::new(net), tags, index, bgr,
            busy: AtomicBool::new(false), queued: AtomicU64::new(0), max_batch, warmup_secs: None };
        if gpu && std::env::var("PYRE_TAGGER_WARMUP").map(|v| v != "0").unwrap_or(true) {
            let t = std::time::Instant::now();
            let blank = vec![1f32; NPATCH * PDIM];
            let net = me.net.lock().unwrap();
            let _ = net.forward(blank.clone(), 1);
            if me.max_batch > 1 { let _ = net.forward(blank.repeat(me.max_batch), me.max_batch); }
            drop(net);
            let s = t.elapsed().as_secs_f64();
            tracing::info!("tagger warm-up (batch 1 + {}) took {s:.1}s", me.max_batch);
            me.warmup_secs = Some(s);
        }
        Ok(me)
    }

    pub fn describe(&self) -> String { format!("EVA02-L tagger ({} tags, Burn/Lyamc fork, {}) on {}", self.tags.len(), self.precision, self.backend) }

    pub fn info(&self) -> Value {
        let mut cats: HashMap<&str, usize> = HashMap::new();
        for t in &self.tags { *cats.entry(category_name(t.category)).or_default() += 1; }
        json!({"architecture": "eva02_large_patch14_448", "input_size": SIDE, "tags": self.tags.len(), "categories": cats, "weights": self.weights.display().to_string(),
            "precision": self.precision, "default_threshold": DEFAULT_THRESHOLD, "default_character_threshold": DEFAULT_CHARACTER_THRESHOLD,
            "has_rating_tags": cats.contains_key("rating"), "max_batch": self.max_batch, "preprocess": if self.bgr { "white pad square, bicubic 448, [-1,1], BGR" } else { "white pad square, bicubic 448, [-1,1], RGB" }})
    }

    #[allow(dead_code)]
    pub fn has_tag(&self, t: &str) -> bool { self.index.contains_key(t) }

    /// Probabilities for preprocessed images, batched by `max_batch`.
    pub fn probs(&self, inputs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let net = self.net.lock().unwrap();
        self.busy.store(true, Ordering::SeqCst);
        let n = self.tags.len();
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.max_batch) {
            let st = std::time::Instant::now();
            let b = chunk.len();
            let flat: Vec<f32> = chunk.concat();
            let p = net.forward(flat, b);
            out.extend(p.chunks(n).map(|c| c.to_vec()));
            crate::limits::pace(st);
        }
        self.busy.store(false, Ordering::SeqCst);
        out
    }

    #[cfg(feature = "image")]
    pub fn prep(&self, bytes: &[u8]) -> Result<Vec<f32>> { preprocess(bytes, self.bgr) }
    #[cfg(feature = "image")]
    #[allow(dead_code)]
    pub fn prep_img(&self, img: image::DynamicImage) -> Vec<f32> { preprocess_img(img, self.bgr) }

    /// Tags grouped by category, watch-list scores and a rating (model's rating tags when the
    /// vocabulary has them, else derived from explicit/questionable/sensitive evidence tags).
    pub fn summarize(&self, p: &[f32], o: &TagOpts) -> Value {
        let mut groups: std::collections::BTreeMap<&str, Vec<(usize, f32)>> = Default::default();
        let mut rating = serde_json::Map::new();
        for (i, (t, &s)) in self.tags.iter().zip(p).enumerate() {
            let cat = category_name(t.category);
            if cat == "rating" { rating.insert(t.name.clone(), json!(round(s))); continue; }
            let thr = if cat == "character" { o.character_threshold } else { o.threshold };
            if s >= thr { groups.entry(cat).or_default().push((i, s)); }
        }
        let mut g = serde_json::Map::new();
        for (k, mut v) in groups {
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            if let Some(k) = o.top_k { v.truncate(k); }
            g.insert(k.into(), json!(v.iter().map(|(i, s)| json!({"tag": self.tags[*i].name, "score": round(*s)})).collect::<Vec<_>>()));
        }
        let watch: serde_json::Map<String, Value> = o.watch.iter().filter_map(|t| self.index.get(t).map(|&i| (t.clone(), json!(round(p[i]))))).collect();
        let ev = |l: &[&str]| -> (f32, Vec<(String, f32)>) {
            let mut v: Vec<(String, f32)> = l.iter().filter_map(|t| self.index.get(*t).map(|&i| (t.to_string(), p[i]))).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            (v.first().map(|x| x.1).unwrap_or(0.0), v.into_iter().filter(|x| x.1 >= 0.2).take(5).collect())
        };
        let (rating_v, source) = if !rating.is_empty() {
            let best = rating.iter().max_by(|a, b| a.1.as_f64().unwrap_or(0.0).total_cmp(&b.1.as_f64().unwrap_or(0.0))).map(|(k, _)| k.clone()).unwrap_or_default();
            (json!({"label": best.trim_start_matches("rating:"), "scores": rating}), "model")
        } else {
            let (e, ee) = ev(EXPLICIT);
            let (q, qe) = ev(QUESTIONABLE);
            let (s, se) = ev(SENSITIVE);
            let q = q.max(e); let s = s.max(q);
            let th = o.rating_threshold;
            let label = if e >= th { "explicit" } else if q >= th { "questionable" } else if s >= th { "sensitive" } else { "general" };
            let evid: Vec<Value> = ee.iter().chain(qe.iter()).chain(se.iter()).map(|(t, s)| json!({"tag": t, "score": round(*s)})).collect();
            (json!({"label": label, "scores": {"general": round(1.0 - s), "sensitive": round(s), "questionable": round(q), "explicit": round(e)}, "evidence": evid, "threshold": th}), "derived")
        };
        json!({"tags": g, "rating": rating_v, "rating_source": source, "watch": watch})
    }
}

fn round(x: f32) -> f64 { (x as f64 * 10000.0).round() / 10000.0 }

#[derive(clap::Args, Debug)]
pub struct TagArgs {
    /// Images and/or folders of images (folders are walked recursively, sorted)
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
    /// Tagger model: alias, store name, HF repo id[@rev] or folder
    #[arg(short = 'm', long = "model", default_value = "Redstonexs/kagami-24k")]
    pub model: String,
    /// cuda | wgpu | vulkan | cpu
    #[arg(short = 'b', long, default_value = "cuda")]
    pub backend: String,
    /// General-tag threshold (model card: 0.37 macro-F1, 0.54 micro-F1)
    #[arg(short = 't', long, default_value_t = DEFAULT_THRESHOLD)]
    pub threshold: f32,
    #[arg(long = "character-threshold", default_value_t = DEFAULT_CHARACTER_THRESHOLD)]
    pub character_threshold: f32,
    /// Keep at most N tags per category
    #[arg(long = "top-k")]
    pub top_k: Option<usize>,
    /// Images per GPU batch (default PYRE_TAGGER_BATCH or 4)
    #[arg(long)]
    pub batch: Option<usize>,
    /// Write one JSON object per image (JSON Lines) here instead of printing a summary
    #[arg(long)]
    pub jsonl: Option<PathBuf>,
    /// Also dump raw probabilities ([n_images, n_tags] f32 little-endian) for reference checks
    #[arg(long = "dump-probs")]
    pub dump_probs: Option<PathBuf>,
    /// Extra tags whose score is always reported (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub watch: Vec<String>,
    /// VRAM kept free for other apps (as for `serve`)
    #[arg(long = "vram-reserve")]
    pub vram_reserve: Option<String>,
    #[arg(long = "gpu-util-max")]
    pub gpu_util_max: Option<u32>,
    /// Debug: skip decoding; read already-preprocessed NCHW f32 input ([n, 3, 448, 448], little-endian)
    /// from this file (for model-only comparisons against a Python reference)
    #[arg(long = "raw-input", hide = true)]
    pub raw_input: Option<PathBuf>,
}

/// NCHW `[3, 448, 448]` -> `[1024, 588]` patches (same layout as [`preprocess_img`]).
pub fn nchw_to_patches(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; NPATCH * PDIM];
    for py in 0..GRID { for px in 0..GRID { for c in 0..3 { for ky in 0..PATCH { for kx in 0..PATCH {
        out[(py * GRID + px) * PDIM + c * PATCH * PATCH + ky * PATCH + kx] = x[c * SIDE * SIDE + (py * PATCH + ky) * SIDE + px * PATCH + kx];
    }}}}}
    out
}

const IMG_EXT: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "bmp"];

fn collect(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = vec![];
    for p in paths {
        if p.is_dir() {
            out.extend(crate::store::walk(p).into_iter().filter(|f| f.extension().and_then(|e| e.to_str()).map(|e| IMG_EXT.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false)));
        } else { out.push(p.clone()); }
    }
    out
}

/// `pyre tag <image|dir>...`
#[cfg(feature = "image")]
pub fn run_cli(ctx: &crate::storage::Ctx, a: TagArgs) -> Result<()> {
    use std::io::Write;
    crate::limits::init(a.vram_reserve.as_deref(), a.gpu_util_max);
    if let Some(b) = a.batch { std::env::set_var("PYRE_TAGGER_BATCH", b.to_string()); }
    let raw: Option<Vec<f32>> = match &a.raw_input {
        Some(p) => Some(std::fs::read(p)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        None => None,
    };
    let files: Vec<PathBuf> = match &raw {
        Some(v) => (0..v.len() / (3 * SIDE * SIDE)).map(|i| PathBuf::from(format!("raw#{i}"))).collect(),
        None => collect(&a.paths),
    };
    if files.is_empty() { bail!("no images found"); }
    let (dir, _) = ctx.resolver().resolve(&a.model)?;
    let vram0 = crate::limits::vram();
    let t = std::time::Instant::now();
    let m = TaggerModel::load(&dir, &a.backend)?;
    let load_s = t.elapsed().as_secs_f64();
    let mut watch: Vec<String> = WATCH.iter().map(|s| s.to_string()).collect();
    watch.extend(a.watch.iter().cloned());
    let o = TagOpts { threshold: a.threshold, character_threshold: a.character_threshold, top_k: a.top_k, watch, ..Default::default() };
    let mut jl = match &a.jsonl { Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)), None => None };
    let mut dump = match &a.dump_probs { Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)), None => None };
    let (mut prep_s, mut gpu_s, mut n_ok, mut min_free) = (0f64, 0f64, 0usize, vram0.map(|v| v.free).unwrap_or(0));
    let mut lat = vec![];
    for chunk in files.chunks(m.max_batch) {
        let tp = std::time::Instant::now();
        let mut ok = vec![];
        // decode + resize in parallel (CPU); the GPU batch below is the only device work
        use rayon::prelude::*;
        let got_all: Vec<Result<Vec<f32>>> = chunk.par_iter().map(|f| match (&raw, f.to_str().and_then(|s| s.strip_prefix("raw#")).and_then(|i| i.parse::<usize>().ok())) {
                (Some(v), Some(i)) => Ok(nchw_to_patches(&v[i * 3 * SIDE * SIDE..(i + 1) * 3 * SIDE * SIDE])),
                _ => std::fs::read(f).map_err(anyhow::Error::from).and_then(|b| m.prep(&b)),
            }).collect();
        for (f, got) in chunk.iter().zip(got_all) {
            match got {
                Ok(x) => ok.push((f, x)),
                Err(e) => { eprintln!("skip {}: {e:#}", f.display()); if let Some(w) = &mut jl { writeln!(w, "{}", json!({"file": f.display().to_string(), "error": format!("{e:#}")}))?; } }
            }
        }
        prep_s += tp.elapsed().as_secs_f64();
        if ok.is_empty() { continue; }
        let tg = std::time::Instant::now();
        let ps = m.probs(ok.iter().map(|x| x.1.clone()).collect());
        let dt = tg.elapsed().as_secs_f64();
        gpu_s += dt;
        lat.push((ok.len(), dt));
        if let Some(v) = crate::limits::vram() { min_free = min_free.min(v.free); }
        for ((f, _), p) in ok.iter().zip(&ps) {
            n_ok += 1;
            let mut s = m.summarize(p, &o);
            s["file"] = json!(f.display().to_string());
            if let Some(d) = &mut dump { for x in p { d.write_all(&x.to_le_bytes())?; } }
            match &mut jl {
                Some(w) => writeln!(w, "{s}")?,
                None => {
                    let gen: Vec<String> = s["tags"]["general"].as_array().map(|v| v.iter().take(25).map(|t| format!("{}:{:.2}", t["tag"].as_str().unwrap_or(""), t["score"].as_f64().unwrap_or(0.0))).collect()).unwrap_or_default();
                    println!("== {} | rating {} ({})\n   {}", f.display(), s["rating"]["label"].as_str().unwrap_or("?"), s["rating_source"].as_str().unwrap_or(""), gen.join(", "));
                }
            }
        }
    }
    let steady: Vec<&(usize, f64)> = lat.iter().skip(1).collect();
    let (sn, ss) = steady.iter().fold((0usize, 0f64), |a, x| (a.0 + x.0, a.1 + x.1));
    let peak = vram0.map(|v| (v.free.saturating_sub(min_free)) >> 20);
    let summary = json!({"images": n_ok, "batch": m.max_batch, "load_s": load_s, "warmup_s": m.warmup_secs, "preprocess_s": prep_s, "gpu_s": gpu_s,
        "ms_per_image_steady": if sn > 0 { Some(ss / sn as f64 * 1000.0) } else { None }, "images_per_s_steady": if ss > 0.0 { Some(sn as f64 / ss) } else { None },
        "vram_used_by_pyre_mib_approx": peak, "vram_total_mib": vram0.map(|v| v.total >> 20), "backend": m.backend, "precision": m.precision});
    eprintln!("summary: {summary}");
    Ok(())
}

#[cfg(not(feature = "image"))]
pub fn run_cli(_ctx: &crate::storage::Ctx, _a: TagArgs) -> Result<()> { bail!("pyre was built without the `image` feature (needed to decode images)") }
