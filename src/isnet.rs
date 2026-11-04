//! IS-Net / DIS (ISNetDIS, U^2-Net RSU blocks), native Burn: background removal.
//!
//! Weights: BRIA RMBG-1.4 (general; non-commercial license) and SkyTNT anime-seg isnet_is (Apache-2.0), both
//! converted to plain f32 safetensors (`isnet_<name>.safetensors`, BatchNorm stats kept; folded into the convs
//! at load). Graph: conv_in(3->64, s2) -> RSU7/6/5/4/4F/4F encoder with ceil-mode 2x2 max pools -> RSU decoder
//! with bilinear (half-pixel) upsampling -> side1 3x3 conv -> upsample to input -> sigmoid.
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::module::{conv2d, interpolate, max_pool2d};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions, MaxPoolOptions};
use burn::tensor::{Device, Tensor, TensorData};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct IsNet { c: HashMap<String, (Tensor<4>, Tensor<1>)>, pub name: String }

/// Checkpoint file for a model name: `rmbg` = BRIA RMBG-1.4 `model.safetensors`, `anime` = SkyTNT anime-seg
/// `model.safetensors` (keys under `net.`; its `gt_encoder.*` training branch is ignored).
fn file(dir: &Path, name: &str) -> PathBuf { dir.join(format!("isnet_{name}.safetensors")) }

pub fn weights_dir(store_root: &Path) -> PathBuf { std::env::var("PYRE_ISNET_DIR").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("isnet")) }
pub fn available(dir: &Path, name: &str) -> bool { file(dir, name).is_file() }

fn up(x: Tensor<4>, hw: [usize; 2]) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    if [h, w] == hw { return x; }
    interpolate(x, InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false).with_output_size(hw))
}
fn pool(x: Tensor<4>) -> Tensor<4> { max_pool2d(x, MaxPoolOptions::new([2, 2]).with_stride([2, 2]).with_ceil_mode(true)) }
fn hw(x: &Tensor<4>) -> [usize; 2] { let d = x.dims(); [d[2], d[3]] }

impl IsNet {
    pub fn load(dir: &Path, name: &str, dev: &Device) -> Result<Self> {
        let path = file(dir, name);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let get = |k: &str| -> Option<(Vec<usize>, Vec<f32>)> { st.tensor(k).ok().filter(|t| t.dtype() == safetensors::Dtype::F32).map(|t| (t.shape().to_vec(), t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())) };
        let mut c = HashMap::new();
        let pre = if st.tensor("net.conv_in.weight").is_ok() { "net." } else { "" };
        for (k0, _) in st.tensors() {
            let Some(k) = k0.strip_prefix(pre) else { continue };
            let k = k.to_string();
            let Some(base) = k.strip_suffix(".weight") else { continue };
            let (ws, mut w) = get(&k0).unwrap();
            if ws.len() != 4 { continue; }
            let mut b = get(&format!("{pre}{base}.bias")).map(|x| x.1).unwrap_or_else(|| vec![0.0; ws[0]]);
            // REBNCONV: <x>.conv_s1 + <x>.bn_s1 ; myrebnconv: <x>.conv + <x>.bn
            let (key, bn) = if let Some(p) = base.strip_suffix(".conv_s1") { (p.to_string(), format!("{p}.bn_s1")) } else if let Some(p) = base.strip_suffix(".conv") { (p.to_string(), format!("{p}.bn")) } else { (base.to_string(), String::new()) };
            if !bn.is_empty() {
                let (g, be, m, v) = (get(&format!("{pre}{bn}.weight")), get(&format!("{pre}{bn}.bias")), get(&format!("{pre}{bn}.running_mean")), get(&format!("{pre}{bn}.running_var")));
                let (Some(g), Some(be), Some(m), Some(v)) = (g, be, m, v) else { bail!("{bn}: BatchNorm tensors missing") };
                let per = ws[1] * ws[2] * ws[3];
                for o in 0..ws[0] {
                    let s = g.1[o] / (v.1[o] + 1e-5).sqrt();
                    for x in &mut w[o * per..(o + 1) * per] { *x *= s; }
                    b[o] = (b[o] - m.1[o]) * s + be.1[o];
                }
            }
            let wt = Tensor::<4>::from_data(TensorData::new(w, [ws[0], ws[1], ws[2], ws[3]]), dev);
            let bt = Tensor::<1>::from_data(TensorData::new(b.clone(), [b.len()]), dev);
            c.insert(key, (wt, bt));
        }
        if !c.contains_key("conv_in") || !c.contains_key("side1") { bail!("{}: not an IS-Net checkpoint", path.display()); }
        Ok(Self { c, name: name.to_string() })
    }
    fn conv(&self, k: &str, x: Tensor<4>, dil: usize, stride: usize) -> Result<Tensor<4>> {
        let (w, b) = self.c.get(k).ok_or_else(|| anyhow!("missing conv {k}"))?;
        Ok(conv2d(x, w.clone(), Some(b.clone()), ConvOptions::new([stride, stride], [dil, dil], [dil, dil], 1)))
    }
    fn rebn(&self, k: &str, x: Tensor<4>, dil: usize) -> Result<Tensor<4>> { Ok(relu(self.conv(k, x, dil, 1)?)) }
    /// RSU-L (L = 4..7): U-block with L-2 pools.
    fn rsu(&self, s: &str, l: usize, x: Tensor<4>) -> Result<Tensor<4>> {
        let hxin = self.rebn(&format!("{s}.rebnconvin"), x, 1)?;
        let mut h: Vec<Tensor<4>> = vec![self.rebn(&format!("{s}.rebnconv1"), hxin.clone(), 1)?];
        for i in 2..l { let p = pool(h.last().unwrap().clone()); h.push(self.rebn(&format!("{s}.rebnconv{i}"), p, 1)?); }
        let top = self.rebn(&format!("{s}.rebnconv{l}"), h.last().unwrap().clone(), 2)?;
        let mut d = self.rebn(&format!("{s}.rebnconv{}d", l - 1), Tensor::cat(vec![top, h[l - 2].clone()], 1), 1)?;
        for i in (1..l - 1).rev() {
            let t = h[i - 1].clone();
            d = self.rebn(&format!("{s}.rebnconv{i}d"), Tensor::cat(vec![up(d, hw(&t)), t], 1), 1)?;
        }
        Ok(d + hxin)
    }
    fn rsu4f(&self, s: &str, x: Tensor<4>) -> Result<Tensor<4>> {
        let hxin = self.rebn(&format!("{s}.rebnconvin"), x, 1)?;
        let h1 = self.rebn(&format!("{s}.rebnconv1"), hxin.clone(), 1)?;
        let h2 = self.rebn(&format!("{s}.rebnconv2"), h1.clone(), 2)?;
        let h3 = self.rebn(&format!("{s}.rebnconv3"), h2.clone(), 4)?;
        let h4 = self.rebn(&format!("{s}.rebnconv4"), h3.clone(), 8)?;
        let d3 = self.rebn(&format!("{s}.rebnconv3d"), Tensor::cat(vec![h4, h3], 1), 4)?;
        let d2 = self.rebn(&format!("{s}.rebnconv2d"), Tensor::cat(vec![d3, h2], 1), 2)?;
        let d1 = self.rebn(&format!("{s}.rebnconv1d"), Tensor::cat(vec![d2, h1], 1), 1)?;
        Ok(d1 + hxin)
    }
    /// x: [1, 3, h, w] (already normalised for the model) -> alpha [1, 1, h, w] in 0..1.
    pub fn forward(&self, x: Tensor<4>) -> Result<Tensor<4>> {
        let size = hw(&x);
        let hxin = self.conv("conv_in", x, 1, 2)?;
        let hx1 = self.rsu("stage1", 7, hxin)?;
        let hx2 = self.rsu("stage2", 6, pool(hx1.clone()))?;
        let hx3 = self.rsu("stage3", 5, pool(hx2.clone()))?;
        let hx4 = self.rsu("stage4", 4, pool(hx3.clone()))?;
        let hx5 = self.rsu4f("stage5", pool(hx4.clone()))?;
        let hx6 = self.rsu4f("stage6", pool(hx5.clone()))?;
        let d5 = self.rsu4f("stage5d", Tensor::cat(vec![up(hx6, hw(&hx5)), hx5], 1))?;
        let d4 = self.rsu("stage4d", 4, Tensor::cat(vec![up(d5, hw(&hx4)), hx4], 1))?;
        let d3 = self.rsu("stage3d", 5, Tensor::cat(vec![up(d4, hw(&hx3)), hx3], 1))?;
        let d2 = self.rsu("stage2d", 6, Tensor::cat(vec![up(d3, hw(&hx2)), hx2], 1))?;
        let d1 = self.rsu("stage1d", 7, Tensor::cat(vec![up(d2, hw(&hx1)), hx1], 1))?;
        Ok(sigmoid(up(self.conv("side1", d1, 1, 1)?, size)))
    }

    /// Alpha matte (0..255, w*h) for an RGB8 image: resized to `side` (rmbg: stretched square, anime: letterboxed),
    /// run, resized back. rmbg output is min-max normalised as in BRIA's reference post-processing.
    pub fn matte(&self, rgb: &image::RgbImage, side: u32, dev: &Device) -> Result<image::GrayImage> {
        let (w0, h0) = rgb.dimensions();
        let anime = self.name == "anime";
        let (rw, rh, ox, oy) = if anime { let s = side as f32 / w0.max(h0) as f32; let (rw, rh) = (((w0 as f32 * s) as u32).max(1), ((h0 as f32 * s) as u32).max(1)); (rw, rh, (side - rw) / 2, (side - rh) / 2) } else { (side, side, 0, 0) };
        let r = image::imageops::resize(rgb, rw, rh, image::imageops::FilterType::Triangle);
        let n = (side * side) as usize;
        let mut planar = vec![0f32; 3 * n];
        for y in 0..rh { for x in 0..rw { let p = r.get_pixel(x, y); let i = ((y + oy) * side + x + ox) as usize; for c in 0..3 { planar[c * n + i] = p[c] as f32 / 255.0 - if anime { 0.0 } else { 0.5 }; } } }
        let x = Tensor::<4>::from_data(TensorData::new(planar, [1, 3, side as usize, side as usize]), dev);
        let a: Vec<f32> = self.forward(x)?.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let (mut lo, mut hi) = (0f32, 1f32);
        if !anime { lo = a.iter().cloned().fold(f32::MAX, f32::min); hi = a.iter().cloned().fold(f32::MIN, f32::max).max(lo + 1e-6); }
        let mut g = image::GrayImage::new(rw, rh);
        for y in 0..rh { for x in 0..rw { let v = (a[((y + oy) * side + x + ox) as usize] - lo) / (hi - lo); g.put_pixel(x, y, image::Luma([(v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8])); } }
        Ok(image::imageops::resize(&g, w0, h0, image::imageops::FilterType::Triangle))
    }
}

/// Numeric check against the PyTorch reference (rmbg: 168x232 input, anime: 256x256).
pub fn check(dir: &Path, refdir: &Path, dev: &Device) -> Result<String> {
    let rd = |f: &str| -> Result<Vec<f32>> { Ok(std::fs::read(refdir.join(f)).with_context(|| f.to_string())?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) };
    let mut s = String::new();
    for (name, h, w) in [("rmbg", 168usize, 232usize), ("anime", 256, 256)] {
        if !available(dir, name) { s += &format!("{name}: weights missing\n"); continue; }
        let net = IsNet::load(dir, name, dev)?;
        let x = Tensor::<4>::from_data(TensorData::new(rd(&format!("ref_in_{name}.bin"))?, [1, 3, h, w]), dev);
        let y: Vec<f32> = net.forward(x)?.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let r = rd(&format!("ref_out_{name}.bin"))?;
        let md = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let mean = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum::<f32>() / r.len() as f32;
        s += &format!("{name}: max |diff| {md:.2e}, mean {mean:.2e} over {} alpha values\n", r.len());
    }
    Ok(s)
}
