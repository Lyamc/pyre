//! SegFormer (MiT-B2 encoder + all-MLP decode head), native Burn: human/clothes parsing.
//!
//! Weights: mattmdjaga/segformer_b2_clothes (`model.safetensors`, f32, MIT) as `<store>/cabin/segformer/
//! segformer_clothes.safetensors`. 18 labels (ATR): 0 background, 1 hat, 2 hair, 3 sunglasses, 4 upper-clothes,
//! 5 skirt, 6 pants, 7 dress, 8 belt, 9/10 shoes, 11 face, 12/13 legs, 14/15 arms, 16 bag, 17 scarf.
//! Graph (HF `SegformerForSemanticSegmentation`): 4 stages of overlapping patch embedding (conv k7 s4 / k3 s2,
//! LayerNorm) + transformer blocks (efficient self-attention with spatial-reduction conv + LayerNorm on K/V,
//! Mix-FFN: dense -> depthwise 3x3 -> GELU -> dense) + stage LayerNorm; decode head: per-stage linear to 768,
//! bilinear (half-pixel) upsample to 1/4, concat (deepest first), 1x1 fuse + BatchNorm + ReLU, 1x1 classifier.
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{gelu, relu, softmax};
use burn::tensor::module::{conv2d, interpolate};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const LABELS: [&str; 18] = ["background", "hat", "hair", "sunglasses", "upper-clothes", "skirt", "pants", "dress", "belt", "left-shoe", "right-shoe", "face", "left-leg", "right-leg", "left-arm", "right-arm", "bag", "scarf"];
const HID: [usize; 4] = [64, 128, 320, 512];
const DEPTH: [usize; 4] = [3, 4, 6, 3];
const HEADS: [usize; 4] = [1, 2, 5, 8];
const SR: [usize; 4] = [8, 4, 2, 1];
const PATCH: [usize; 4] = [7, 3, 3, 3];
const STRIDE: [usize; 4] = [4, 2, 2, 2];

pub fn weights_dir(store_root: &Path) -> PathBuf { std::env::var("PYRE_SEGFORMER_DIR").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("segformer")) }
fn file(dir: &Path) -> PathBuf { dir.join("segformer_clothes.safetensors") }
pub fn available(dir: &Path) -> bool { file(dir).is_file() }

pub struct SegFormer { t: HashMap<String, (Vec<usize>, Vec<f32>)>, dev: Device }

impl SegFormer {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let path = file(dir);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut t = HashMap::new();
        for (k, v) in st.tensors() {
            if v.dtype() != safetensors::Dtype::F32 { continue; }
            t.insert(k.trim_start_matches("segformer.").to_string(), (v.shape().to_vec(), v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        }
        if !t.contains_key("decode_head.classifier.weight") { bail!("{}: not a SegFormer checkpoint", path.display()); }
        Ok(Self { t, dev: dev.clone() })
    }
    fn raw(&self, k: &str) -> Result<&(Vec<usize>, Vec<f32>)> { self.t.get(k).ok_or_else(|| anyhow!("missing tensor {k}")) }
    fn t1(&self, k: &str) -> Result<Tensor<1>> { let (s, v) = self.raw(k)?; Ok(Tensor::from_data(TensorData::new(v.clone(), [s[0]]), &self.dev)) }
    fn t4(&self, k: &str) -> Result<Tensor<4>> { let (s, v) = self.raw(k)?; Ok(Tensor::from_data(TensorData::new(v.clone(), [s[0], s[1], s[2], s[3]]), &self.dev)) }
    /// x [N, in] -> [N, out] with torch Linear weight [out, in].
    fn lin(&self, k: &str, x: Tensor<2>) -> Result<Tensor<2>> {
        let (s, v) = self.raw(&format!("{k}.weight"))?;
        let w = Tensor::<2>::from_data(TensorData::new(v.clone(), [s[0], s[1]]), &self.dev).transpose();
        Ok(x.matmul(w) + self.t1(&format!("{k}.bias"))?.unsqueeze::<2>())
    }
    fn ln(&self, k: &str, x: Tensor<2>) -> Result<Tensor<2>> {
        let (var, mean) = x.clone().var_mean_bias(1);
        let y = (x - mean) / (var + 1e-5).sqrt();
        Ok(y * self.t1(&format!("{k}.weight"))?.unsqueeze::<2>() + self.t1(&format!("{k}.bias"))?.unsqueeze::<2>())
    }
    fn conv(&self, k: &str, x: Tensor<4>, stride: usize, pad: usize, groups: usize, bias: bool) -> Result<Tensor<4>> {
        let b = if bias { Some(self.t1(&format!("{k}.bias"))?) } else { None };
        Ok(conv2d(x, self.t4(&format!("{k}.weight"))?, b, ConvOptions::new([stride, stride], [pad, pad], [1, 1], groups)))
    }
    /// [N, C] tokens of an h x w map <-> [1, C, h, w]
    fn to_map(x: Tensor<2>, h: usize, w: usize) -> Tensor<4> { let c = x.dims()[1]; x.transpose().reshape([1, c, h, w]) }
    fn to_tok(x: Tensor<4>) -> Tensor<2> { let [_, c, h, w] = x.dims(); x.reshape([c, h * w]).transpose() }

    fn attention(&self, p: &str, x: Tensor<2>, h: usize, w: usize, i: usize) -> Result<Tensor<2>> {
        let (n, c) = (x.dims()[0], x.dims()[1]);
        let nh = HEADS[i]; let d = c / nh;
        let kv = if SR[i] > 1 { let m = self.conv(&format!("{p}.self.sr"), Self::to_map(x.clone(), h, w), SR[i], 0, 1, true)?; self.ln(&format!("{p}.self.layer_norm"), Self::to_tok(m))? } else { x.clone() };
        let m = kv.dims()[0];
        let q = self.lin(&format!("{p}.self.query"), x)?.reshape([n, nh, d]).swap_dims(0, 1);
        let k = self.lin(&format!("{p}.self.key"), kv.clone())?.reshape([m, nh, d]).swap_dims(0, 1);
        let v = self.lin(&format!("{p}.self.value"), kv)?.reshape([m, nh, d]).swap_dims(0, 1);
        let a = softmax(q.matmul(k.swap_dims(1, 2)) * (1.0 / (d as f64).sqrt()), 2);
        let o = a.matmul(v).swap_dims(0, 1).reshape([n, c]);
        self.lin(&format!("{p}.output.dense"), o)
    }
    /// x: [1, 3, H, W] normalised -> logits [1, 18, H/4, W/4]
    pub fn forward(&self, x: Tensor<4>) -> Result<Tensor<4>> {
        let mut feats = Vec::new();
        let mut cur = x;
        for i in 0..4 {
            let e = self.conv(&format!("encoder.patch_embeddings.{i}.proj"), cur, STRIDE[i], PATCH[i] / 2, 1, true)?;
            let [_, _, h, w] = e.dims();
            let mut t = self.ln(&format!("encoder.patch_embeddings.{i}.layer_norm"), Self::to_tok(e))?;
            for j in 0..DEPTH[i] {
                let p = format!("encoder.block.{i}.{j}");
                t = t.clone() + self.attention(&format!("{p}.attention"), self.ln(&format!("{p}.layer_norm_1"), t.clone())?, h, w, i)?;
                let f = self.lin(&format!("{p}.mlp.dense1"), self.ln(&format!("{p}.layer_norm_2"), t.clone())?)?;
                let ch = f.dims()[1];
                let f = gelu(Self::to_tok(self.conv(&format!("{p}.mlp.dwconv.dwconv"), Self::to_map(f, h, w), 1, 1, ch, true)?));
                t = t + self.lin(&format!("{p}.mlp.dense2"), f)?;
            }
            let t = self.ln(&format!("encoder.layer_norm.{i}"), t)?;
            cur = Self::to_map(t, h, w);
            feats.push(cur.clone());
        }
        let [_, _, h0, w0] = feats[0].dims();
        let mut ups = Vec::new();
        for (i, f) in feats.into_iter().enumerate().rev() {
            let [_, _, h, w] = f.dims();
            let m = Self::to_map(self.lin(&format!("decode_head.linear_c.{i}.proj"), Self::to_tok(f))?, h, w);
            ups.push(if [h, w] == [h0, w0] { m } else { interpolate(m, InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false).with_output_size([h0, w0])) });
        }
        let _ = HID;
        let fused = self.conv("decode_head.linear_fuse", Tensor::cat(ups, 1), 1, 0, 1, false)?;
        // BatchNorm (eval)
        let g = self.t1("decode_head.batch_norm.weight")?; let b = self.t1("decode_head.batch_norm.bias")?;
        let mu = self.t1("decode_head.batch_norm.running_mean")?; let var = self.t1("decode_head.batch_norm.running_var")?;
        let sc = g / (var + 1e-5).sqrt(); let sh = b - mu * sc.clone();
        let c = sc.dims()[0];
        let y = relu(fused * sc.reshape([1, c, 1, 1]) + sh.reshape([1, c, 1, 1]));
        self.conv("decode_head.classifier", y, 1, 0, 1, true)
    }
    /// Per-pixel label map (w x h, values 0..17) for an RGB8 image (stretched to 512x512 like the HF processor).
    pub fn parse(&self, rgb: &image::RgbImage) -> Result<image::GrayImage> {
        let (w0, h0) = rgb.dimensions();
        let s = 512u32;
        let r = image::imageops::resize(rgb, s, s, image::imageops::FilterType::Triangle);
        let n = (s * s) as usize;
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let mut planar = vec![0f32; 3 * n];
        for (i, p) in r.pixels().enumerate() { for c in 0..3 { planar[c * n + i] = (p[c] as f32 / 255.0 - mean[c]) / std[c]; } }
        let lo = self.forward(Tensor::<4>::from_data(TensorData::new(planar, [1, 3, s as usize, s as usize]), &self.dev))?;
        let lo = interpolate(lo, InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false).with_output_size([h0 as usize, w0 as usize]));
        let v: Vec<f32> = lo.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let m = (w0 * h0) as usize;
        let mut g = image::GrayImage::new(w0, h0);
        for (i, px) in g.pixels_mut().enumerate() {
            let mut best = 0; for c in 1..18 { if v[c * m + i] > v[best * m + i] { best = c; } }
            px.0[0] = best as u8;
        }
        Ok(g)
    }
}

/// Numeric check against the PyTorch reference (512x512 input, logits 18x128x128).
pub fn check(dir: &Path, refdir: &Path, dev: &Device) -> Result<String> {
    let rd = |f: &str| -> Result<Vec<f32>> { Ok(std::fs::read(refdir.join(f)).with_context(|| f.to_string())?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) };
    let net = SegFormer::load(dir, dev)?;
    let y: Vec<f32> = net.forward(Tensor::<4>::from_data(TensorData::new(rd("ref_in.bin")?, [1, 3, 512, 512]), dev))?.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
    let r = rd("ref_out.bin")?;
    if y.len() != r.len() { bail!("output size {} vs reference {}", y.len(), r.len()); }
    let md = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mean = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum::<f32>() / r.len() as f32;
    let m = 128 * 128; let mut agree = 0;
    for i in 0..m { let am = |v: &[f32]| (0..18).max_by(|&a, &b| v[a * m + i].total_cmp(&v[b * m + i])).unwrap(); if am(&y) == am(&r) { agree += 1; } }
    Ok(format!("segformer: max |diff| {md:.2e}, mean {mean:.2e}, argmax agreement {:.3}% over {m} pixels", 100.0 * agree as f64 / m as f64))
}
