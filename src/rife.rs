//! RIFE v4.6 (Practical-RIFE IFNet_HDv3, MIT), native Burn: frame interpolation between two images.
//!
//! Weights: `flownet.pkl` converted to plain f32 safetensors (`<store>/cabin/rife/rife46.safetensors`, 5.3 M
//! params). Graph: 4 IFBlocks at scales 8/4/2/1; each: bilinear downscale of [img0, img1, t, (warped, mask,
//! flow)] -> 2 strided convs (LeakyReLU 0.2) -> 8 ResConv (conv * beta + x, LeakyReLU) -> ConvTranspose 4x4 s2
//! -> PixelShuffle(2) -> bilinear upscale; flow += block flow, mask += block mask; backward warp with
//! grid_sample (bilinear, border, align_corners) and a sigmoid-mask blend of the two warped frames.
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{leaky_relu, sigmoid};
use burn::tensor::module::{conv2d, conv_transpose2d, interpolate};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn weights_dir(store_root: &Path) -> PathBuf { std::env::var("PYRE_RIFE_DIR").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("rife")) }
fn file(dir: &Path) -> PathBuf { dir.join("rife46.safetensors") }
pub fn available(dir: &Path) -> bool { file(dir).is_file() }

pub struct Rife { t: HashMap<String, Tensor<4>>, b: HashMap<String, Tensor<1>>, dev: Device }

fn resize(x: Tensor<4>, h: usize, w: usize) -> Tensor<4> {
    let d = x.dims(); if [d[2], d[3]] == [h, w] { return x; }
    interpolate(x, InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false).with_output_size([h, w]))
}

impl Rife {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let path = file(dir);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let (mut t, mut b) = (HashMap::new(), HashMap::new());
        for (k, v) in st.tensors() {
            if v.dtype() != safetensors::Dtype::F32 { bail!("{k}: expected f32"); }
            let data: Vec<f32> = v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let s = v.shape();
            if s.len() == 4 { t.insert(k.to_string(), Tensor::from_data(TensorData::new(data, [s[0], s[1], s[2], s[3]]), dev)); } else if s.len() == 1 { b.insert(k.to_string(), Tensor::from_data(TensorData::new(data, [s[0]]), dev)); }
        }
        if !t.contains_key("block0.conv0.0.0.weight") { bail!("{}: not a RIFE 4.6 IFNet checkpoint", path.display()); }
        Ok(Self { t, b, dev: dev.clone() })
    }
    fn w(&self, k: &str) -> Result<Tensor<4>> { self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}")) }
    fn bias(&self, k: &str) -> Result<Tensor<1>> { self.b.get(k).cloned().ok_or_else(|| anyhow!("missing {k}")) }
    fn conv(&self, k: &str, x: Tensor<4>, stride: usize) -> Result<Tensor<4>> {
        Ok(conv2d(x, self.w(&format!("{k}.weight"))?, Some(self.bias(&format!("{k}.bias"))?), ConvOptions::new([stride, stride], [1, 1], [1, 1], 1)))
    }
    fn block(&self, i: usize, x: Tensor<4>, flow: Option<Tensor<4>>, scale: usize) -> Result<(Tensor<4>, Tensor<4>)> {
        let [_, _, h, w] = x.dims();
        let (hs, ws) = (h / scale, w / scale);
        let mut x = resize(x, hs, ws);
        if let Some(f) = flow { x = Tensor::cat(vec![x, resize(f, hs, ws) * (1.0 / scale as f64)], 1); }
        let p = format!("block{i}");
        let f = leaky_relu(self.conv(&format!("{p}.conv0.0.0"), x, 2)?, 0.2);
        let mut f = leaky_relu(self.conv(&format!("{p}.conv0.1.0"), f, 2)?, 0.2);
        for j in 0..8 {
            let q = format!("{p}.convblock.{j}");
            f = leaky_relu(self.conv(&format!("{q}.conv"), f.clone(), 1)? * self.w(&format!("{q}.beta"))? + f, 0.2);
        }
        let t = conv_transpose2d(f, self.w(&format!("{p}.lastconv.0.weight"))?, Some(self.bias(&format!("{p}.lastconv.0.bias"))?), ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1));
        let [_, c, th, tw] = t.dims();
        let t = t.reshape([1, c / 4, 2, 2, th, tw]).permute([0, 1, 4, 2, 5, 3]).reshape([1, c / 4, th * 2, tw * 2]);
        let t = resize(t, h, w);
        let flow = t.clone().slice([0..1, 0..4, 0..h, 0..w]) * scale as f64;
        let mask = t.slice([0..1, 4..5, 0..h, 0..w]);
        Ok((flow, mask))
    }
    /// Backward warp = torch grid_sample(bilinear, border, align_corners=True) of `img` at pixel + flow.
    /// Done as clamped pixel coordinates mapped to an align_corners=false grid with the default sampler: the
    /// clamp gives border padding, and in-range coordinates never touch the zero padding.
    fn warp(&self, img: Tensor<4>, flow: Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = img.dims();
        let px: Vec<f32> = (0..h * w).map(|i| (i % w) as f32).collect();
        let py: Vec<f32> = (0..h * w).map(|i| (i / w) as f32).collect();
        let bx = Tensor::<4>::from_data(TensorData::new(px, [1, 1, h, w]), &self.dev);
        let by = Tensor::<4>::from_data(TensorData::new(py, [1, 1, h, w]), &self.dev);
        let sx = (bx + flow.clone().slice([0..1, 0..1, 0..h, 0..w])).clamp(0.0, w as f64 - 1.0);
        let sy = (by + flow.slice([0..1, 1..2, 0..h, 0..w])).clamp(0.0, h as f64 - 1.0);
        // Manual bilinear gather (Burn's grid_sample disagreed with torch by up to 0.68 here).
        let c = img.dims()[1];
        let flat = img.reshape([1, c, h * w]);
        let x0 = sx.clone().floor();
        let y0 = sy.clone().floor();
        let wx = (sx - x0.clone()).reshape([1, 1, h * w]);
        let wy = (sy - y0.clone()).reshape([1, 1, h * w]);
        let x1 = (x0.clone() + 1.0).clamp_max(w as f64 - 1.0);
        let y1 = (y0.clone() + 1.0).clamp_max(h as f64 - 1.0);
        let at = |yy: &Tensor<4>, xx: &Tensor<4>| {
            let idx = (yy.clone() * w as f64 + xx.clone()).reshape([1, 1, h * w]).int().repeat_dim(1, c);
            flat.clone().gather(2, idx)
        };
        let (a, b, cc, d) = (at(&y0, &x0), at(&y0, &x1), at(&y1, &x0), at(&y1, &x1));
        let top = a.clone() + (b - a) * wx.clone();
        let bot = cc.clone() + (d - cc) * wx;
        (top.clone() + (bot - top) * wy).reshape([1, c, h, w])
    }

    /// img0, img1: [1, 3, H, W] in 0..1 (H, W multiples of 64) -> the frame at `t` in 0..1.
    pub fn forward(&self, img0: Tensor<4>, img1: Tensor<4>, t: f32) -> Result<Tensor<4>> {
        let [_, _, h, w] = img0.dims();
        let ts = Tensor::<4>::from_data(TensorData::new(vec![t; h * w], [1, 1, h, w]), &self.dev);
        let (mut flow, mut mask) = self.block(0, Tensor::cat(vec![img0.clone(), img1.clone(), ts.clone()], 1), None, 8)?;
        let (mut w0, mut w1) = (self.warp(img0.clone(), flow.clone().slice([0..1, 0..2, 0..h, 0..w])), self.warp(img1.clone(), flow.clone().slice([0..1, 2..4, 0..h, 0..w])));
        for (i, s) in [(1usize, 4usize), (2, 2), (3, 1)] {
            let (f0, m0) = self.block(i, Tensor::cat(vec![w0, w1, ts.clone(), mask.clone()], 1), Some(flow.clone()), s)?;
            flow = flow + f0; mask = mask + m0;
            w0 = self.warp(img0.clone(), flow.clone().slice([0..1, 0..2, 0..h, 0..w]));
            w1 = self.warp(img1.clone(), flow.clone().slice([0..1, 2..4, 0..h, 0..w]));
        }
        let m = sigmoid(mask);
        Ok(w0 * m.clone() + w1 * (m.neg() + 1.0))
    }
    /// In-between frames of two same-size RGB8 images at the given times (padded to multiples of 64).
    pub fn frames(&self, a: &image::RgbImage, b: &image::RgbImage, times: &[f32]) -> Result<Vec<image::RgbImage>> {
        let (w, h) = a.dimensions();
        let (pw, ph) = ((w as usize).div_ceil(64) * 64, (h as usize).div_ceil(64) * 64);
        let planar = |im: &image::RgbImage| { let n = pw * ph; let mut v = vec![0f32; 3 * n]; for (x, y, p) in im.enumerate_pixels() { for c in 0..3 { v[c * n + y as usize * pw + x as usize] = p[c] as f32 / 255.0; } } Tensor::<4>::from_data(TensorData::new(v, [1, 3, ph, pw]), &self.dev) };
        let (ta, tb) = (planar(a), planar(b));
        let mut out = Vec::new();
        for &t in times {
            let y: Vec<f32> = self.forward(ta.clone(), tb.clone(), t)?.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
            let n = pw * ph;
            out.push(image::RgbImage::from_fn(w, h, |x, yy| { let i = yy as usize * pw + x as usize; image::Rgb([0, 1, 2].map(|c| (y[c * n + i] * 255.0).round().clamp(0.0, 255.0) as u8)) }));
        }
        Ok(out)
    }
}

/// Numeric check against the PyTorch reference (two 192x128 frames, t = 0.5).
pub fn check(dir: &Path, refdir: &Path, dev: &Device) -> Result<String> {
    let rd = |f: &str| -> Result<Vec<f32>> { Ok(std::fs::read(refdir.join(f)).with_context(|| f.to_string())?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) };
    let net = Rife::load(dir, dev)?;
    let (h, w) = (128usize, 192usize);
    let x = Tensor::<4>::from_data(TensorData::new(rd("ref_in.bin")?, [1, 6, h, w]), dev);
    let y: Vec<f32> = net.forward(x.clone().slice([0..1, 0..3, 0..h, 0..w]), x.clone().slice([0..1, 3..6, 0..h, 0..w]), 0.5)?.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
    let cmp = |a: &[f32], b: &[f32]| (a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max), a.len() == b.len());
    let i0 = x.clone().slice([0..1, 0..3, 0..h, 0..w]); let i1 = x.clone().slice([0..1, 3..6, 0..h, 0..w]);
    let ts = Tensor::<4>::from_data(TensorData::new(vec![0.5f32; h * w], [1, 1, h, w]), dev);
    let xin = Tensor::cat(vec![i0.clone(), i1, ts], 1);
    let host = |t: Tensor<4>| -> Result<Vec<f32>> { t.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}")) };
    let mut s = String::new();
    if let Ok(r) = rd("ref_down8.bin") { s += &format!("down8 {:?}; ", cmp(&host(resize(xin.clone(), h / 8, w / 8))?, &r)); }
    let (f0, _) = net.block(0, xin, None, 8)?;
    if let Ok(r) = rd("ref_flow0.bin") { s += &format!("flow0 {:?}; ", cmp(&host(f0.clone())?, &r)); }
    if let Ok(r) = rd("ref_warp0.bin") { let rf = Tensor::<4>::from_data(TensorData::new(rd("ref_flow0.bin")?, [1, 4, h, w]), dev); s += &format!("warp(ref flow) {:?}; ", cmp(&host(net.warp(i0, rf.slice([0..1, 0..2, 0..h, 0..w])))?, &r)); }
    let r = rd("ref_out.bin")?;
    let md = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mean = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum::<f32>() / r.len() as f32;
    Ok(format!("rife 4.6: max |diff| {md:.2e}, mean {mean:.2e} over {} pixels; stages: {s}", r.len()))
}
