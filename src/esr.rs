//! Real-ESRGAN SRVGGNetCompact (realesr-general-x4v3 / -wdn / realesr-animevideov3), native Burn.
//!
//! body = conv3x3(3->64) PReLU, N x [conv3x3(64->64) PReLU], conv3x3(64->48); pixel_shuffle(4); + nearest x4 of
//! the input. "Denoise" s blends general-x4v3 (strong denoise) with general-wdn-x4v3 (weak denoise), as Real-ESRGAN's DNI: w = s*gen + (1-s)*wdn.
//! Runs in f32 on 160 px tiles with a 10 px overlap (cabin: ~1.2 M params, tiles keep activations < 200 MB).
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::prelu;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};
use std::path::{Path, PathBuf};

pub struct Esr { layers: Vec<(Tensor<4>, Tensor<1>, Option<Tensor<1>>)>, dev: Device }

fn read(path: &Path) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut v = Vec::new();
    for (k, t) in st.tensors() {
        if t.dtype() != safetensors::Dtype::F32 { bail!("{k}: expected f32"); }
        let f: Vec<f32> = t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        v.push((k, t.shape().to_vec(), f));
    }
    Ok(v)
}

/// The weight folder: `PYRE_ESR_DIR`, else `<store>/cabin/realesr`.
pub fn weights_dir(store_root: &Path) -> PathBuf { std::env::var("PYRE_ESR_DIR").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("realesr")) }
pub fn available(dir: &Path) -> bool { dir.join("realesr_general_x4v3.safetensors").is_file() }

impl Esr {
    /// `model`: "general" (with `denoise` 0..1 blending in the wdn weights) or "anime".
    pub fn load(dir: &Path, model: &str, denoise: f32, dev: &Device) -> Result<Self> {
        let mut t = match model {
            "anime" => read(&dir.join("realesr_anime_x4v3.safetensors"))?,
            _ => read(&dir.join("realesr_general_x4v3.safetensors"))?,
        };
        if model != "anime" && denoise < 1.0 {
            let w = read(&dir.join("realesr_general_wdn_x4v3.safetensors"))?;
            let s = denoise.clamp(0.0, 1.0);
            for (k, _, d) in t.iter_mut() {
                let (_, _, e) = w.iter().find(|x| &x.0 == k).ok_or_else(|| anyhow!("wdn weights lack {k}"))?;
                for (a, b) in d.iter_mut().zip(e) { *a = s * *a + (1.0 - s) * b; }
            }
        }
        let get = |k: &str| t.iter().find(|x| x.0 == k);
        let mut layers = Vec::new();
        let mut i = 0;
        while let Some((_, ws, w)) = get(&format!("body.{i}.weight")) {
            let (_, _, b) = get(&format!("body.{i}.bias")).ok_or_else(|| anyhow!("body.{i}.bias"))?;
            let w4 = Tensor::<4>::from_data(TensorData::new(w.clone(), [ws[0], ws[1], ws[2], ws[3]]), dev);
            let b1 = Tensor::<1>::from_data(TensorData::new(b.clone(), [b.len()]), dev);
            i += 1;
            let a = match get(&format!("body.{i}.weight")) { Some((_, s, a)) if s.len() == 1 => { i += 1; Some(Tensor::<1>::from_data(TensorData::new(a.clone(), [a.len()]), dev)) } _ => None };
            layers.push((w4, b1, a));
        }
        if layers.len() < 3 { bail!("no SRVGGNet layers in {}", dir.display()); }
        Ok(Self { layers, dev: dev.clone() })
    }

    /// x: [1, 3, h, w] in 0..1 -> [1, 3, 4h, 4w].
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = x.dims();
        let mut o = x.clone();
        for (cw, cb, a) in &self.layers {
            o = conv2d(o, cw.clone(), Some(cb.clone()), ConvOptions::new([1, 1], [1, 1], [1, 1], 1));
            if let Some(a) = a { o = prelu(o, a.clone()); }
        }
        let o = o.reshape([1, 3, 4, 4, h, w]).permute([0, 1, 4, 2, 5, 3]).reshape([1, 3, 4 * h, 4 * w]);
        let up = x.reshape([1, 3, h, 1, w, 1]).expand([1, 3, h, 4, w, 4]).reshape([1, 3, 4 * h, 4 * w]);
        o + up
    }

    /// Upscale planar-free RGB8 (w*h*3) by 4 with tiling; returns RGB8 (4w*4h*3).
    pub fn upscale_rgb(&self, rgb: &[u8], w: usize, h: usize, tile: usize, pad: usize) -> Result<Vec<u8>> {
        let (ow, oh) = (w * 4, h * 4);
        let mut out = vec![0u8; ow * oh * 3];
        let mut y0 = 0;
        while y0 < h {
            let y1 = (y0 + tile).min(h);
            let mut x0 = 0;
            while x0 < w {
                let x1 = (x0 + tile).min(w);
                let (py0, py1, px0, px1) = (y0.saturating_sub(pad), (y1 + pad).min(h), x0.saturating_sub(pad), (x1 + pad).min(w));
                let (tw, th) = (px1 - px0, py1 - py0);
                let mut planar = vec![0f32; 3 * tw * th];
                for yy in 0..th { for xx in 0..tw { let s = ((py0 + yy) * w + px0 + xx) * 3; for c in 0..3 { planar[c * tw * th + yy * tw + xx] = rgb[s + c] as f32 / 255.0; } } }
                let t = Tensor::<4>::from_data(TensorData::new(planar, [1, 3, th, tw]), &self.dev);
                let y = self.forward(t).clamp(0.0, 1.0).into_data();
                let v: Vec<f32> = y.to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
                let (tw4, th4) = (tw * 4, th * 4);
                for yy in (y0 - py0) * 4..(y1 - py0) * 4 { for xx in (x0 - px0) * 4..(x1 - px0) * 4 {
                    let d = ((py0 * 4 + yy) * ow + px0 * 4 + xx) * 3;
                    for c in 0..3 { out[d + c] = (v[c * tw4 * th4 + yy * tw4 + xx] * 255.0 + 0.5) as u8; }
                } }
                x0 = x1;
            }
            y0 = y1;
        }
        Ok(out)
    }
}

/// Numeric check against the PyTorch reference (`ref_in.bin` [1,3,48,40] f32 LE, `ref_out_<model>.bin`).
pub fn check(dir: &Path, refdir: &Path, dev: &Device) -> Result<String> {
    let rd = |f: &str| -> Result<Vec<f32>> { Ok(std::fs::read(refdir.join(f))?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) };
    let x = rd("ref_in.bin")?;
    let mut s = String::new();
    for (m, dn, f) in [("general", 1.0, "ref_out_general.bin"), ("anime", 0.0, "ref_out_anime.bin"), ("general", 0.5, "ref_out_dni05.bin")] {
        let e = Esr::load(dir, m, dn, dev)?;
        let y = e.forward(Tensor::<4>::from_data(TensorData::new(x.clone(), [1, 3, 48, 40]), dev)).into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let r = rd(f)?;
        let md = y.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        s += &format!("{m} denoise={dn}: max |diff| {md:.2e} over {} values\n", r.len());
    }
    Ok(s)
}
