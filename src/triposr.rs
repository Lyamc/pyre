//! TripoSR (stabilityai/TripoSR, MIT): single image -> triplane NeRF -> marching-cubes mesh with vertex colours.
//! Native Burn port. Weights: `<store>/cabin/triposr/triposr_f16.safetensors` (converted in Rust from `model.ckpt`
//! by [`convert_ckpt`]: f16 storage, decoder f32, DINO position table pre-interpolated to the 32x32 grid of 512 px).
//! Graph: DINO ViT-B/16 (12 layers, 768, eps 1e-12, exact GELU) on the 512x512 grey-composited image ->
//! 1025 tokens; learned triplane tokens [3, 1024, 32, 32] -> Transformer1D (GroupNorm 32, proj_in, 16 x
//! [LN, self-attn, LN, cross-attn to DINO, LN, GEGLU 4096], proj_out, residual) -> ConvTranspose 2x2 to
//! [3, 40, 64, 64]; points in [-0.87, 0.87]^3 sample the xy/xz/yz planes (bilinear, align_corners=false,
//! zero padding), concat 120 -> NeRF MLP (9 x Linear 64 + SiLU, Linear 4): density exp(d - 1), colour sigmoid.
//! Weights are kept f16 on the GPU and cast to f32 per op (f32 maths, half the VRAM).
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{gelu, sigmoid, softmax};
use burn::tensor::module::{conv2d, conv_transpose2d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const RADIUS: f32 = 0.87;
pub fn weights_dir(store_root: &Path) -> PathBuf { std::env::var("PYRE_TRIPOSR_DIR").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("triposr")) }
fn file(dir: &Path) -> PathBuf { dir.join("triposr_f16.safetensors") }
pub fn available(dir: &Path) -> bool { file(dir).is_file() }

/// Convert the TripoSR `model.ckpt` PyTorch checkpoint into the runtime
/// `triposr_f16.safetensors` layout, entirely in Rust. This mirrors the old
/// reference `convert.py` step for step:
///
/// * `image_tokenizer.model.*` becomes `dino.*`;
/// * `pooler.*` and the learned DINO `position_embeddings` are dropped, except
///   that the position table is pre-interpolated from 14x14 to the 32x32 patch
///   grid of a 512 px input (bicubic, `align_corners=false`) and saved as
///   `dino_pos512` (1025 x 768);
/// * `decoder.*` stays f32 (the density head is sensitive), everything else is
///   stored f16 to halve the file and the VRAM footprint.
///
/// No Python, torch, or network access is involved.
pub fn convert_ckpt(src: &Path, dst: &Path) -> Result<()> {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let reader = pytorch_reader::PytorchReader::new(src).map_err(|e| anyhow!("{}: {e}", src.display()))?;

    let pe = reader
        .get("image_tokenizer.model.embeddings.position_embeddings")
        .ok_or_else(|| anyhow!("{}: no image_tokenizer.model.embeddings.position_embeddings", src.display()))?;
    let pe_shape = pe.shape().to_vec();
    let channels = *pe_shape.last().ok_or_else(|| anyhow!("position_embeddings has no shape"))?;
    let pe_vals = tensor_f32(pe)?;
    let pos = dino_pos_512(&pe_vals, channels)?;

    // Bytes per output tensor, with the safetensors dtype they will be written as.
    let mut tensors: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    tensors.push((
        "dino_pos512".into(),
        safetensors::Dtype::F16,
        vec![1025, channels],
        f32_to_f16_bytes(&pos),
    ));

    for (name, t) in reader.tensors() {
        if name.contains("pooler") || name.ends_with("position_embeddings") {
            continue;
        }
        let out_name = name.replace("image_tokenizer.model.", "dino.");
        let vals = tensor_f32(t)?;
        let (dtype, bytes) = if name.starts_with("decoder.") {
            (safetensors::Dtype::F32, f32_to_f32_bytes(&vals))
        } else {
            (safetensors::Dtype::F16, f32_to_f16_bytes(&vals))
        };
        tensors.push((out_name, dtype, t.shape().to_vec(), bytes));
    }

    for required in ["dino_pos512", "dino.embeddings.cls_token", "decoder.layers.18.weight"] {
        if !tensors.iter().any(|(n, ..)| n == required) {
            bail!("{}: checkpoint is missing {required}", src.display());
        }
    }

    let views: Vec<(&str, TensorView)> = tensors
        .iter()
        .map(|(n, dt, sh, data)| {
            TensorView::new(*dt, sh.clone(), data)
                .map(|v| (n.as_str(), v))
                .map_err(|e| anyhow!("{n}: {e}"))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, dst).map_err(|e| anyhow!("writing {}: {e}", dst.display()))?;
    Ok(())
}

/// Read any float tensor as f32, whatever its stored dtype.
fn tensor_f32(t: &pytorch_reader::Tensor) -> Result<Vec<f32>> {
    use pytorch_reader::DType;
    let raw = t.read().map_err(|e| anyhow!("{}: {e}", t.name))?;
    let n = t.num_elements();
    let vals: Vec<f32> = match t.dtype() {
        DType::F32 => raw.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect(),
        DType::F16 => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_ne_bytes([c[0], c[1]]))).collect(),
        DType::BF16 => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_ne_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        DType::F64 => raw.chunks_exact(8).map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32).collect(),
        d => bail!("{}: unsupported dtype {d:?}; expected a float tensor", t.name),
    };
    if vals.len() != n {
        bail!("{}: read {} floats but the shape has {n} elements", t.name, vals.len());
    }
    Ok(vals)
}

fn f32_to_f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| half::f16::from_f32(*x).to_bits().to_le_bytes()).collect()
}

/// Reference `convert.py` position-table step: split CLS from the 14x14 patch
/// grid, bicubic-resize the grid to 32x32, concat CLS back on as row 0.
fn dino_pos_512(vals: &[f32], c: usize) -> Result<Vec<f32>> {
    if c == 0 || vals.len() != c * 197 {
        bail!("position_embeddings has {} values, expected 197 x {c}", vals.len());
    }
    let grid = &vals[c..];
    let mut out = Vec::with_capacity(1025 * c);
    out.extend_from_slice(&vals[..c]);
    out.extend_from_slice(&bicubic_resize(grid, 14, 14, 32, 32, c));
    Ok(out)
}

/// Separable bicubic resize with PyTorch's `mode='bicubic', align_corners=false`
/// kernel (A = -0.75). `src` is row-major `[in_h, in_w, channels]`.
fn bicubic_resize(src: &[f32], in_h: usize, in_w: usize, out_h: usize, out_w: usize, channels: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), in_h * in_w * channels);
    let ys: Vec<_> = (0..out_h).map(|o| cubic_weights(o, in_h, out_h)).collect();
    let xs: Vec<_> = (0..out_w).map(|o| cubic_weights(o, in_w, out_w)).collect();
    let mut out = vec![0.0f32; out_h * out_w * channels];
    for (oy, (iy, wy)) in ys.iter().enumerate() {
        for (ox, (ix, wx)) in xs.iter().enumerate() {
            let base = (oy * out_w + ox) * channels;
            for (ch, slot) in out[base..base + channels].iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for ky in 0..4 {
                    for kx in 0..4 {
                        let v = src[(iy[ky] * in_w + ix[kx]) * channels + ch];
                        acc += wy[ky] * wx[kx] * v;
                    }
                }
                *slot = acc;
            }
        }
    }
    out
}

/// Source indices and weights of the four taps for one output index.
fn cubic_weights(out_idx: usize, in_size: usize, out_size: usize) -> ([usize; 4], [f32; 4]) {
    let scale = in_size as f32 / out_size as f32;
    let real = scale * (out_idx as f32 + 0.5) - 0.5;
    let floor = real.floor();
    let t = real - floor;
    let i0 = floor as isize;
    let clamp = |i: isize| i.clamp(0, in_size as isize - 1) as usize;
    const A: f32 = -0.75;
    let conv1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let conv2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    (
        [clamp(i0 - 1), clamp(i0), clamp(i0 + 1), clamp(i0 + 2)],
        [conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(2.0 - t)],
    )
}

pub struct Tsr { w: HashMap<String, (Vec<usize>, Tensor<1>)>, dev: Device }

fn f16_to_f32(h: u16) -> f32 { half::f16::from_bits(h).to_f32() }

impl Tsr {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let path = file(dir);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut w = HashMap::new();
        for (k, v) in st.tensors() {
            let (vals, keep16): (Vec<f32>, bool) = match v.dtype() {
                safetensors::Dtype::F16 => (v.data().chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(), true),
                safetensors::Dtype::F32 => (v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), false),
                d => bail!("{k}: unsupported dtype {d:?}"),
            };
            let n = vals.len();
            let t = Tensor::<1>::from_data(TensorData::new(vals, [n]), dev);
            w.insert(k.to_string(), (v.shape().to_vec(), if keep16 { t.cast(DType::F16) } else { t }));
        }
        if !w.contains_key("dino_pos512") || !w.contains_key("decoder.layers.18.weight") { bail!("{}: not a converted TripoSR checkpoint", path.display()); }
        Ok(Self { w, dev: dev.clone() })
    }
    fn raw(&self, k: &str) -> Result<&(Vec<usize>, Tensor<1>)> { self.w.get(k).ok_or_else(|| anyhow!("missing tensor {k}")) }
    fn t1(&self, k: &str) -> Result<Tensor<1>> { Ok(self.raw(k)?.1.clone().cast(DType::F32)) }
    fn t2(&self, k: &str) -> Result<Tensor<2>> { let (s, t) = self.raw(k)?; let n = t.dims()[0]; let a = s[0]; Ok(t.clone().cast(DType::F32).reshape([a, n / a])) }
    fn t4(&self, k: &str) -> Result<Tensor<4>> { let (s, t) = self.raw(k)?; Ok(t.clone().cast(DType::F32).reshape([s[0], s[1], s[2], s[3]])) }
    /// x [N, in] @ W^T (+ b), torch Linear layout.
    fn lin(&self, k: &str, x: Tensor<2>, bias: bool) -> Result<Tensor<2>> {
        let y = x.matmul(self.t2(&format!("{k}.weight"))?.transpose());
        Ok(if bias { y + self.t1(&format!("{k}.bias"))?.unsqueeze::<2>() } else { y })
    }
    fn ln(&self, k: &str, x: Tensor<2>, eps: f64) -> Result<Tensor<2>> {
        let (var, mean) = x.clone().var_mean_bias(1);
        let y = (x - mean) / (var + eps).sqrt();
        Ok(y * self.t1(&format!("{k}.weight"))?.unsqueeze::<2>() + self.t1(&format!("{k}.bias"))?.unsqueeze::<2>())
    }
    /// Multi-head attention, q [n, h*d], k/v [m, h*d]; heads processed `group` at a time to bound the scores.
    fn mha(q: Tensor<2>, k: Tensor<2>, v: Tensor<2>, heads: usize, group: usize) -> Tensor<2> {
        let (n, c) = (q.dims()[0], q.dims()[1]); let m = k.dims()[0]; let d = c / heads;
        let q = q.reshape([n, heads, d]).swap_dims(0, 1);
        let k = k.reshape([m, heads, d]).swap_dims(0, 1);
        let v = v.reshape([m, heads, d]).swap_dims(0, 1);
        let mut outs = Vec::new();
        let mut h0 = 0;
        while h0 < heads {
            let h1 = (h0 + group).min(heads);
            let qs = q.clone().slice([h0..h1, 0..n, 0..d]); let ks = k.clone().slice([h0..h1, 0..m, 0..d]); let vs = v.clone().slice([h0..h1, 0..m, 0..d]);
            let a = softmax(qs.matmul(ks.swap_dims(1, 2)) * (1.0 / (d as f64).sqrt()), 2);
            outs.push(a.matmul(vs));
            h0 = h1;
        }
        Tensor::cat(outs, 0).swap_dims(0, 1).reshape([n, c])
    }

    /// img: [512*512*3] HWC in 0..1 (grey-composited foreground) -> DINO tokens [1025, 768].
    pub fn dino(&self, img: &[f32]) -> Result<Tensor<2>> {
        let s = 512; let n = s * s;
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let mut planar = vec![0f32; 3 * n];
        for i in 0..n { for c in 0..3 { planar[c * n + i] = (img[i * 3 + c] - mean[c]) / std[c]; } }
        let x = Tensor::<4>::from_data(TensorData::new(planar, [1, 3, s, s]), &self.dev);
        let p = conv2d(x, self.t4("dino.embeddings.patch_embeddings.projection.weight")?, Some(self.t1("dino.embeddings.patch_embeddings.projection.bias")?), ConvOptions::new([16, 16], [0, 0], [1, 1], 1));
        let p = p.reshape([768, 1024]).transpose();
        let cls = self.t1("dino.embeddings.cls_token")?.reshape([1, 768]);
        let mut h = Tensor::cat(vec![cls, p], 0) + self.t2("dino_pos512")?;
        for l in 0..12 {
            let pre = format!("dino.encoder.layer.{l}");
            let x = self.ln(&format!("{pre}.layernorm_before"), h.clone(), 1e-12)?;
            let q = self.lin(&format!("{pre}.attention.attention.query"), x.clone(), true)?;
            let k = self.lin(&format!("{pre}.attention.attention.key"), x.clone(), true)?;
            let v = self.lin(&format!("{pre}.attention.attention.value"), x, true)?;
            h = h + self.lin(&format!("{pre}.attention.output.dense"), Self::mha(q, k, v, 12, 12), true)?;
            let x = self.ln(&format!("{pre}.layernorm_after"), h.clone(), 1e-12)?;
            h = h.clone() + self.lin(&format!("{pre}.output.dense"), gelu(self.lin(&format!("{pre}.intermediate.dense"), x, true)?), true)?;
        }
        self.ln("dino.layernorm", h, 1e-12)
    }
    /// DINO tokens -> backbone tokens [1024, 3072] (channels x tokens, like the reference `bb[0]`).
    pub fn backbone(&self, enc: Tensor<2>) -> Result<Tensor<2>> {
        let emb = self.t4("tokenizer.embeddings")?; // [3, 1024, 32, 32]
        let x0 = emb.swap_dims(0, 1).reshape([1024, 3072]);
        // GroupNorm(32 groups, eps 1e-6) over channels
        let g = x0.clone().reshape([32, 32 * 3072]);
        let (var, mean) = g.clone().var_mean_bias(1);
        let g = ((g - mean) / (var + 1e-6).sqrt()).reshape([1024, 3072]);
        let g = g * self.t1("backbone.norm.weight")?.unsqueeze_dim::<2>(1) + self.t1("backbone.norm.bias")?.unsqueeze_dim::<2>(1);
        let mut h = self.lin("backbone.proj_in", g.transpose(), true)?; // [3072, 1024]
        for l in 0..16 {
            let p = format!("backbone.transformer_blocks.{l}");
            let x = self.ln(&format!("{p}.norm1"), h.clone(), 1e-5)?;
            let q = self.lin(&format!("{p}.attn1.to_q"), x.clone(), false)?;
            let k = self.lin(&format!("{p}.attn1.to_k"), x.clone(), false)?;
            let v = self.lin(&format!("{p}.attn1.to_v"), x, false)?;
            h = h + self.lin(&format!("{p}.attn1.to_out.0"), Self::mha(q, k, v, 16, 4), true)?;
            let x = self.ln(&format!("{p}.norm2"), h.clone(), 1e-5)?;
            let q = self.lin(&format!("{p}.attn2.to_q"), x, false)?;
            let k = self.lin(&format!("{p}.attn2.to_k"), enc.clone(), false)?;
            let v = self.lin(&format!("{p}.attn2.to_v"), enc.clone(), false)?;
            h = h + self.lin(&format!("{p}.attn2.to_out.0"), Self::mha(q, k, v, 16, 16), true)?;
            let x = self.ln(&format!("{p}.norm3"), h.clone(), 1e-5)?;
            let pr = self.lin(&format!("{p}.ff.net.0.proj"), x, true)?; // [3072, 8192]
            let a = pr.clone().slice([0..3072, 0..4096]); let gate = pr.slice([0..3072, 4096..8192]);
            h = h.clone() + self.lin(&format!("{p}.ff.net.2"), a * gelu(gate), true)?;
        }
        let out = self.lin("backbone.proj_out", h, true)?.transpose(); // [1024, 3072]
        Ok(out + x0)
    }
    /// Backbone tokens -> triplanes [3, 40, 64, 64].
    pub fn planes(&self, bb: Tensor<2>) -> Result<Tensor<4>> {
        let t = bb.reshape([1024, 3, 32, 32]).swap_dims(0, 1); // [3, 1024, 32, 32]
        Ok(conv_transpose2d(t, self.t4("post_processor.upsample.weight")?, Some(self.t1("post_processor.upsample.bias")?), ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1)))
    }
    /// Query points p [N, 3] in [-1, 1] (normalised by the radius): (density_act [N], colour [N, 3]).
    pub fn query(&self, planes: &Tensor<4>, p: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = p.len() / 3;
        let [_, c, hh, ww] = planes.dims();
        let pt = Tensor::<2>::from_data(TensorData::new(p.to_vec(), [n, 3]), &self.dev);
        let col = |i: usize| pt.clone().slice([0..n, i..i + 1]).reshape([n]);
        let mut feats = Vec::new();
        for (pi, (a, b)) in [(0usize, 1usize), (0, 2), (1, 2)].into_iter().enumerate() {
            let plane = planes.clone().slice([pi..pi + 1, 0..c, 0..hh, 0..ww]).reshape([c, hh * ww]);
            // grid x (width) = first coordinate, y (height) = second; align_corners=false
            let ix = (col(a) + 1.0) * (ww as f64 / 2.0) - 0.5;
            let iy = (col(b) + 1.0) * (hh as f64 / 2.0) - 0.5;
            let x0 = ix.clone().floor(); let y0 = iy.clone().floor();
            let wx = ix - x0.clone(); let wy = iy - y0.clone();
            let mut acc: Option<Tensor<2>> = None;
            for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let xc = x0.clone() + dx; let yc = y0.clone() + dy;
                let valid = xc.clone().greater_equal_elem(0.0).float() * xc.clone().lower_equal_elem(ww as f64 - 1.0).float()
                    * yc.clone().greater_equal_elem(0.0).float() * yc.clone().lower_equal_elem(hh as f64 - 1.0).float();
                let wgt = (if dx == 0.0 { -wx.clone() + 1.0 } else { wx.clone() }) * (if dy == 0.0 { -wy.clone() + 1.0 } else { wy.clone() }) * valid;
                let idx: Tensor<1, Int> = (yc.clamp(0.0, hh as f64 - 1.0) * ww as f64 + xc.clamp(0.0, ww as f64 - 1.0)).int();
                let v = plane.clone().select(1, idx) * wgt.unsqueeze::<2>();
                acc = Some(match acc { Some(a) => a + v, None => v });
            }
            feats.push(acc.unwrap());
        }
        let mut x = Tensor::cat(feats, 0).transpose(); // [N, 120]
        for l in (0..=16).step_by(2) { let y = self.lin(&format!("decoder.layers.{l}"), x, true)?; x = y.clone() * sigmoid(y); }
        let o = self.lin("decoder.layers.18", x, true)?; // [N, 4]
        let dens = (o.clone().slice([0..n, 0..1]) - 1.0).exp().reshape([n]);
        let colr = sigmoid(o.slice([0..n, 1..4]));
        let d: Vec<f32> = dens.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let cl: Vec<f32> = colr.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        Ok((d, cl))
    }
    /// Density on an R^3 grid over [-r, r]^3, x-major (index i*R*R + j*R + k), in chunks.
    pub fn grid(&self, planes: &Tensor<4>, r: usize, chunk: usize) -> Result<Vec<f32>> {
        let tot = r * r * r; let mut out = Vec::with_capacity(tot);
        let lin = |i: usize| 2.0 * i as f32 / (r - 1) as f32 - 1.0;
        let mut s = 0;
        while s < tot {
            let e = (s + chunk).min(tot);
            let mut p = Vec::with_capacity((e - s) * 3);
            for n in s..e { let (i, j, k) = (n / (r * r), (n / r) % r, n % r); p.extend([lin(i), lin(j), lin(k)]); }
            out.extend(self.query(planes, &p)?.0);
            s = e;
        }
        Ok(out)
    }
}

/// Marching cubes (PyMCubes corner/edge layout), inside where density >= thr. Returns grid-space vertices
/// (i, j, k as f32) and triangles; vertices shared along edges.
pub fn marching_cubes(d: &[f32], r: usize, thr: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    use crate::mc_tables::TRI;
    let at = |i: usize, j: usize, k: usize| thr - d[i * r * r + j * r + k]; // <= 0 inside
    const C: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
    const E: [[usize; 2]; 12] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    let mut verts = Vec::new(); let mut tris = Vec::new();
    let mut cache: HashMap<(usize, usize), u32> = HashMap::new(); // (lower corner linear index, axis)
    for i in 0..r - 1 { for j in 0..r - 1 { for k in 0..r - 1 {
        let mut v = [0f32; 8]; let mut ci = 0usize;
        for m in 0..8 { v[m] = at(i + C[m][0], j + C[m][1], k + C[m][2]); if v[m] <= 0.0 { ci |= 1 << m; } }
        if ci == 0 || ci == 255 { continue; }
        let row = &TRI[ci];
        let mut ids = [0u32; 12]; let mut have = [false; 12];
        let mut t = 0;
        while t < 16 && row[t] >= 0 {
            for q in 0..3 {
                let e = row[t + q] as usize;
                if !have[e] {
                    let (a, b) = (E[e][0], E[e][1]);
                    let pa = [i + C[a][0], j + C[a][1], k + C[a][2]]; let pb = [i + C[b][0], j + C[b][1], k + C[b][2]];
                    let lo = if pa <= pb { pa } else { pb };
                    let axis = (0..3).find(|&x| pa[x] != pb[x]).unwrap();
                    let key = (lo[0] * r * r + lo[1] * r + lo[2], axis);
                    let id = *cache.entry(key).or_insert_with(|| {
                        let (fa, fb) = (v[a], v[b]);
                        let s = if (fb - fa).abs() < 1e-12 { 0.5 } else { (-fa / (fb - fa)).clamp(0.0, 1.0) };
                        verts.push([pa[0] as f32 + (pb[0] as f32 - pa[0] as f32) * s, pa[1] as f32 + (pb[1] as f32 - pa[1] as f32) * s, pa[2] as f32 + (pb[2] as f32 - pa[2] as f32) * s]);
                        (verts.len() - 1) as u32
                    });
                    ids[e] = id; have[e] = true;
                }
            }
            tris.push([ids[row[t] as usize], ids[row[t + 1] as usize], ids[row[t + 2] as usize]]);
            t += 3;
        }
    }}}
    (verts, tris)
}

pub struct Mesh { pub pos: Vec<[f32; 3]>, pub col: Vec<[f32; 3]>, pub tri: Vec<[u32; 3]> }

/// Full pipeline on a prepared 512x512 HWC image.
pub fn image_to_mesh(m: &Tsr, img: &[f32], res: usize, thr: f32) -> Result<(Mesh, String)> {
    let t0 = std::time::Instant::now();
    let enc = m.dino(img)?; let bb = m.backbone(enc)?; let planes = m.planes(bb)?;
    let t1 = t0.elapsed().as_secs_f64();
    let d = m.grid(&planes, res, 1 << 18)?;
    let t2 = t0.elapsed().as_secs_f64();
    let (gv, mut tri) = marching_cubes(&d, res, thr);
    if gv.is_empty() { bail!("no surface found (empty density field; is the subject cut out?)"); }
    let s = 2.0 / (res - 1) as f32;
    let pn: Vec<f32> = gv.iter().flat_map(|v| [v[0] * s - 1.0, v[1] * s - 1.0, v[2] * s - 1.0]).collect();
    // colour at the vertices, in fixed 2^18-point chunks (padded) so the grid kernels are reused, not recompiled
    let ch = 1usize << 18; let mut cl = Vec::with_capacity(pn.len());
    for part in pn.chunks(3 * ch) { let mut q = part.to_vec(); q.resize(3 * ch, 0.0); let (_, c) = m.query(&planes, &q)?; cl.extend_from_slice(&c[..part.len()]); }
    // TripoSR frame: x towards the camera, z up -> glTF (y up, +z towards the viewer): (y, z, x)
    let pos: Vec<[f32; 3]> = pn.chunks(3).map(|p| [p[1] * RADIUS, p[2] * RADIUS, p[0] * RADIUS]).collect();
    // outward winding (positive signed volume)
    let vol: f64 = tri.iter().map(|t| { let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]); (a[0] as f64) * ((b[1] * c[2] - b[2] * c[1]) as f64) - (a[1] as f64) * ((b[0] * c[2] - b[2] * c[0]) as f64) + (a[2] as f64) * ((b[0] * c[1] - b[1] * c[0]) as f64) }).sum();
    if vol < 0.0 { for t in tri.iter_mut() { t.swap(1, 2); } }
    let col: Vec<[f32; 3]> = cl.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let info = format!("{} vertices, {} triangles; network {:.1}s, grid {}^3 {:.1}s, mesh {:.1}s", pos.len(), tri.len(), t1, res, t2 - t1, t0.elapsed().as_secs_f64() - t2);
    Ok((Mesh { pos, col, tri }, info))
}

/// Binary glTF 2.0 writer shared by TripoSR and the mesh tools: POSITION, NORMAL,
/// optional COLOR_0 (unsigned byte normalised VEC4), optional TEXCOORD_0 (f32 VEC2),
/// u32 indices. `nor` must have the same length as `pos`.
pub fn write_glb(pos: &[[f32; 3]], nor: &[[f32; 3]], col: Option<&[[f32; 3]]>, uv: Option<&[[f32; 2]]>, tri: &[[u32; 3]]) -> Vec<u8> {
    let n = pos.len();
    // Never emit an accessor whose count disagrees with its buffer view.
    let zero_nor;
    let nor: &[[f32; 3]] = if nor.len() == n { nor } else { zero_nor = vec![[0.0, 0.0, 1.0]; n]; &zero_nor };
    let col = col.filter(|c| c.len() == n);
    let uv = uv.filter(|t| t.len() == n);
    let mut mn = [f32::MAX; 3]; let mut mx = [f32::MIN; 3];
    for p in pos { for q in 0..3 { mn[q] = mn[q].min(p[q]); mx[q] = mx[q].max(p[q]); } }
    if pos.is_empty() { mn = [0.0; 3]; mx = [0.0; 3]; }
    let mut bin = Vec::new();
    for p in pos { for q in 0..3 { bin.extend(p[q].to_le_bytes()); } }
    let pos_len = bin.len();
    for p in nor { for q in 0..3 { bin.extend(p[q].to_le_bytes()); } }
    let nor_len = bin.len() - pos_len;
    let mut views = vec![
        serde_json::json!({"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962}),
        serde_json::json!({"buffer": 0, "byteOffset": pos_len, "byteLength": nor_len, "target": 34962}),
    ];
    let mut accessors = vec![
        serde_json::json!({"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": mn, "max": mx}),
        serde_json::json!({"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"}),
    ];
    let mut attrs = serde_json::Map::new();
    attrs.insert("POSITION".into(), serde_json::json!(0));
    attrs.insert("NORMAL".into(), serde_json::json!(1));
    if let Some(col) = col {
        let off = bin.len();
        for c in col { for q in 0..3 { bin.push((c[q].clamp(0.0, 1.0) * 255.0).round() as u8); } bin.push(255); }
        let view = views.len();
        views.push(serde_json::json!({"buffer": 0, "byteOffset": off, "byteLength": bin.len() - off, "target": 34962}));
        accessors.push(serde_json::json!({"bufferView": view, "componentType": 5121, "normalized": true, "count": n, "type": "VEC4"}));
        attrs.insert("COLOR_0".into(), serde_json::json!(accessors.len() - 1));
    }
    if let Some(uv) = uv {
        let off = bin.len();
        for t in uv { for q in 0..2 { bin.extend(t[q].to_le_bytes()); } }
        let view = views.len();
        views.push(serde_json::json!({"buffer": 0, "byteOffset": off, "byteLength": bin.len() - off, "target": 34962}));
        accessors.push(serde_json::json!({"bufferView": view, "componentType": 5126, "count": n, "type": "VEC2"}));
        attrs.insert("TEXCOORD_0".into(), serde_json::json!(accessors.len() - 1));
    }
    let idx_off = bin.len();
    for t in tri { for &i in t { bin.extend(i.to_le_bytes()); } }
    let idx_view = views.len();
    views.push(serde_json::json!({"buffer": 0, "byteOffset": idx_off, "byteLength": tri.len() * 12, "target": 34963}));
    accessors.push(serde_json::json!({"bufferView": idx_view, "componentType": 5125, "count": tri.len() * 3, "type": "SCALAR"}));
    let idx_acc = accessors.len() - 1;
    while bin.len() % 4 != 0 { bin.push(0); }
    let json = serde_json::json!({
        "asset": {"version": "2.0", "generator": "pyre (native)"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attrs, "indices": idx_acc, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}, "doubleSided": true}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": views,
        "accessors": accessors
    });
    let mut js = serde_json::to_vec(&json).unwrap(); while js.len() % 4 != 0 { js.push(b' '); }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(0x4654_6C67u32.to_le_bytes()); out.extend(2u32.to_le_bytes()); out.extend((total as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes()); out.extend(0x4E4F_534Au32.to_le_bytes()); out.extend(js);
    out.extend((bin.len() as u32).to_le_bytes()); out.extend(0x004E_4942u32.to_le_bytes()); out.extend(bin);
    out
}

/// One joint in a skinned GLB. `translation` is the joint's world position in
/// the mesh's bind pose; the writer derives the node's local translation and the
/// inverse bind matrix from it.
#[derive(Clone, Debug)]
pub struct SkinJoint {
    pub name: String,
    pub parent: Option<usize>,
    pub translation: [f32; 3],
}

/// Skinned binary glTF 2.0 writer: the same POSITION/NORMAL/COLOR_0/index layout
/// as [`write_glb`] plus JOINTS_0 (u8 vec4), WEIGHTS_0 (f32 vec4), a `skins`
/// entry, joint nodes, and inverse bind matrices. Colors default to opaque white
/// when the mesh has none, so the cabin viewer always has a `COLOR_0` accessor.
pub fn write_glb_skinned(
    pos: &[[f32; 3]],
    nor: &[[f32; 3]],
    col: Option<&[[f32; 4]]>,
    tri: &[[u32; 3]],
    joint_index: &[[u8; 4]],
    weights: &[[f32; 4]],
    skeleton: &[SkinJoint],
) -> Vec<u8> {
    let n = pos.len();
    let sn = skeleton.len();
    debug_assert_eq!(nor.len(), n);
    debug_assert_eq!(joint_index.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert!(sn > 0);

    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for p in pos {
        for q in 0..3 {
            mn[q] = mn[q].min(p[q]);
            mx[q] = mx[q].max(p[q]);
        }
    }
    if pos.is_empty() {
        mn = [0.0; 3];
        mx = [0.0; 3];
    }

    let white;
    let col: &[[f32; 4]] = match col {
        Some(c) if c.len() == n => c,
        _ => {
            white = vec![[1.0, 1.0, 1.0, 1.0]; n];
            &white
        }
    };

    let mut bin = Vec::new();
    for p in pos {
        for q in 0..3 {
            bin.extend(p[q].to_le_bytes());
        }
    }
    let pos_len = bin.len();
    for p in nor {
        for q in 0..3 {
            bin.extend(p[q].to_le_bytes());
        }
    }
    let nor_len = bin.len() - pos_len;
    let col_off = bin.len();
    for c in col {
        for q in 0..4 {
            bin.push((c[q].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    let joints_off = bin.len();
    for j in joint_index {
        bin.extend_from_slice(j);
    }
    let weights_off = bin.len();
    for w in weights {
        for q in 0..4 {
            bin.extend(w[q].to_le_bytes());
        }
    }
    let idx_off = bin.len();
    for t in tri {
        for &i in t {
            bin.extend(i.to_le_bytes());
        }
    }
    let ibm_off = bin.len();
    for j in skeleton {
        let t = j.translation;
        for v in [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -t[0], -t[1], -t[2], 1.0] {
            bin.extend(v.to_le_bytes());
        }
    }
    debug_assert_eq!(bin.len() % 4, 0);

    let views = vec![
        json!({"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962}),
        json!({"buffer": 0, "byteOffset": pos_len, "byteLength": nor_len, "target": 34962}),
        json!({"buffer": 0, "byteOffset": col_off, "byteLength": n * 4, "target": 34962}),
        json!({"buffer": 0, "byteOffset": joints_off, "byteLength": n * 4, "target": 34962}),
        json!({"buffer": 0, "byteOffset": weights_off, "byteLength": n * 16, "target": 34962}),
        json!({"buffer": 0, "byteOffset": idx_off, "byteLength": tri.len() * 12, "target": 34963}),
        json!({"buffer": 0, "byteOffset": ibm_off, "byteLength": sn * 64}),
    ];
    let accessors = vec![
        json!({"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": mn, "max": mx}),
        json!({"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"}),
        json!({"bufferView": 2, "componentType": 5121, "normalized": true, "count": n, "type": "VEC4"}),
        json!({"bufferView": 3, "componentType": 5121, "count": n, "type": "VEC4"}),
        json!({"bufferView": 4, "componentType": 5126, "count": n, "type": "VEC4"}),
        json!({"bufferView": 5, "componentType": 5125, "count": tri.len() * 3, "type": "SCALAR"}),
        json!({"bufferView": 6, "componentType": 5126, "count": sn, "type": "MAT4"}),
    ];
    let mut attrs = serde_json::Map::new();
    attrs.insert("POSITION".into(), json!(0));
    attrs.insert("NORMAL".into(), json!(1));
    attrs.insert("COLOR_0".into(), json!(2));
    attrs.insert("JOINTS_0".into(), json!(3));
    attrs.insert("WEIGHTS_0".into(), json!(4));

    let mut nodes = vec![json!({"name": "mesh", "mesh": 0, "skin": 0})];
    for (i, j) in skeleton.iter().enumerate() {
        let t = if let Some(p) = j.parent {
            let w = skeleton[p].translation;
            [j.translation[0] - w[0], j.translation[1] - w[1], j.translation[2] - w[2]]
        } else {
            j.translation
        };
        nodes.push(json!({"name": j.name, "translation": t}));
    }
    for i in 0..sn {
        let kids: Vec<usize> = (0..sn).filter(|&k| skeleton[k].parent == Some(i)).map(|k| 1 + k).collect();
        if !kids.is_empty() {
            nodes[1 + i]["children"] = json!(kids);
        }
    }
    let joint_nodes: Vec<usize> = (0..sn).map(|i| 1 + i).collect();
    let json = json!({
        "asset": {"version": "2.0", "generator": "pyre (native)"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "skins": [{"joints": joint_nodes, "inverseBindMatrices": 6, "skeleton": 1}],
        "meshes": [{"primitives": [{"attributes": attrs, "indices": 5, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}, "doubleSided": true}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": views,
        "accessors": accessors
    });
    let mut js = serde_json::to_vec(&json).unwrap();
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(0x4654_6C67u32.to_le_bytes());
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes());
    out.extend(0x4E4F_534Au32.to_le_bytes());
    out.extend(js);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(0x004E_4942u32.to_le_bytes());
    out.extend(bin);
    out
}

/// TripoSR GLB: POSITION, NORMAL (area-weighted), COLOR_0 (unsigned byte normalised), u32 indices.
pub fn glb(m: &Mesh) -> Vec<u8> {
    let n = m.pos.len();
    let mut nor = vec![[0f32; 3]; n];
    for t in &m.tri {
        let (a, b, c) = (m.pos[t[0] as usize], m.pos[t[1] as usize], m.pos[t[2] as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let f = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        for &i in t { for q in 0..3 { nor[i as usize][q] += f[q]; } }
    }
    for x in nor.iter_mut() { let l = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt().max(1e-12); for q in 0..3 { x[q] /= l; } }
    write_glb(&m.pos, &nor, Some(&m.col), None, &m.tri)
}

/// Prepare an RGBA cut-out like TripoSR's run.py: crop to the alpha bbox, pad square so the subject fills
/// `ratio` (0.85), composite on grey 0.5, resize to 512. Returns HWC f32 in 0..1.
pub fn prepare(rgba: &image::RgbaImage, ratio: f32) -> Result<Vec<f32>> {
    let (w, h) = rgba.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for (x, y, p) in rgba.enumerate_pixels() { if p[3] > 127 { x0 = x0.min(x); y0 = y0.min(y); x1 = x1.max(x); y1 = y1.max(y); } }
    if x1 < x0 { bail!("the cut-out is empty (no foreground found)"); }
    let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
    let side = ((cw.max(ch) as f32) / ratio).ceil() as u32;
    let mut sq = image::RgbaImage::new(side, side);
    let (ox, oy) = ((side - cw) / 2, (side - ch) / 2);
    for y in 0..ch { for x in 0..cw { sq.put_pixel(ox + x, oy + y, *rgba.get_pixel(x0 + x, y0 + y)); } }
    let r = image::imageops::resize(&sq, 512, 512, image::imageops::FilterType::Triangle);
    let mut out = Vec::with_capacity(512 * 512 * 3);
    for p in r.pixels() { let a = p[3] as f32 / 255.0; for c in 0..3 { out.push(p[c] as f32 / 255.0 * a + 0.5 * (1.0 - a)); } }
    Ok(out)
}

/// Numeric check against the PyTorch reference dumps (ref_in [512,512,3], ref_dino [1025,768],
/// ref_backbone [1024,3072], ref_scene [3,40,64,64], ref_density64 [64^3]).
pub fn check(dir: &Path, refdir: &Path, dev: &Device) -> Result<String> {
    let rd = |f: &str| -> Result<Vec<f32>> { Ok(std::fs::read(refdir.join(f)).with_context(|| f.to_string())?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) };
    let cmp = |a: &[f32], b: &[f32]| -> (f32, f64, f64) { let mut mx = 0f32; let (mut s, mut sr) = (0f64, 0f64); for (x, y) in a.iter().zip(b) { mx = mx.max((x - y).abs()); s += (x - y).abs() as f64; sr += y.abs() as f64; } (mx, s / a.len() as f64, sr / a.len() as f64) };
    let host = |t: Tensor<2>| -> Result<Vec<f32>> { t.into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}")) };
    let m = Tsr::load(dir, dev)?;
    let img = rd("ref_in.bin")?;
    let t = std::time::Instant::now();
    let enc = m.dino(&img)?; let e = host(enc.clone())?;
    let mut s = format!("dino {:?}; ", cmp(&e, &rd("ref_dino.bin")?));
    let bb = m.backbone(enc)?; s += &format!("backbone {:?}; ", cmp(&host(bb.clone())?, &rd("ref_backbone.bin")?));
    let pl = m.planes(bb)?; let pv: Vec<f32> = pl.clone().into_data().to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
    s += &format!("triplanes {:?}; ", cmp(&pv, &rd("ref_scene.bin")?));
    let d = m.grid(&pl, 64, 1 << 18)?; let rdn = rd("ref_density64.bin")?;
    let (inside_n, inside_r) = (d.iter().filter(|&&x| x > 25.0).count(), rdn.iter().filter(|&&x| x > 25.0).count());
    let agree = d.iter().zip(&rdn).filter(|(a, b)| (**a > 25.0) == (**b > 25.0)).count() as f64 / d.len() as f64;
    let ld: Vec<f32> = d.iter().map(|x| x.max(1e-6).ln()).collect(); let lr: Vec<f32> = rdn.iter().map(|x| x.max(1e-6).ln()).collect();
    s += &format!("log-density {:?}; occupancy@25 native {inside_n} vs ref {inside_r}, agreement {:.4}%; ", cmp(&ld, &lr), agree * 100.0);
    let (v, tr) = marching_cubes(&d, 64, 25.0);
    s += &format!("mc64 {} verts {} tris; {:.1}s", v.len(), tr.len(), t.elapsed().as_secs_f64());
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_matches_pytorch_align_corners_false() {
        // Weights sum to one everywhere, including at the clamped border.
        for i in 0..32 {
            let (_, w) = cubic_weights(i, 14, 32);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "output {i}: weights sum to {s}");
        }
        // A constant plane resizes exactly.
        let src = vec![2.5f32; 14 * 14];
        let out = bicubic_resize(&src, 14, 14, 32, 32, 1);
        assert!(out.iter().all(|v| (v - 2.5).abs() < 1e-5));
        // Reference values from `F.interpolate(mode='bicubic', align_corners=False)`
        // on a 1x1x4x4 input holding 0..15 (PyTorch 2.14).
        #[rustfmt::skip]
        let expected = [
            -0.52734375, -0.23046875, 0.24609375, 0.875, 1.28125, 1.91015625, 2.38671875, 2.68359375,
             0.66015625, 0.95703125, 1.43359375, 2.0625, 2.46875, 3.09765625, 3.57421875, 3.87109375,
             2.56640625, 2.86328125, 3.33984375, 3.96875, 4.375, 5.00390625, 5.48046875, 5.77734375,
             5.08203125, 5.37890625, 5.85546875, 6.484375, 6.890625, 7.51953125, 7.99609375, 8.29296875,
             6.70703125, 7.00390625, 7.48046875, 8.109375, 8.515625, 9.14453125, 9.62109375, 9.91796875,
             9.22265625, 9.51953125, 9.99609375, 10.625, 11.03125, 11.66015625, 12.13671875, 12.43359375,
            11.12890625, 11.42578125, 11.90234375, 12.53125, 12.9375, 13.56640625, 14.04296875, 14.33984375,
            12.31640625, 12.61328125, 13.08984375, 13.71875, 14.125, 14.75390625, 15.23046875, 15.52734375,
        ];
        let src: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let out = bicubic_resize(&src, 4, 4, 8, 8, 1);
        for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
            assert!((got - want).abs() < 1e-5, "pixel {i}: got {got}, want {want}");
        }
        // Multi-channel resize keeps channels independent.
        let mut rgb = vec![0f32; 14 * 14 * 3];
        for i in 0..14 * 14 { for c in 0..3 { rgb[i * 3 + c] = c as f32 + 1.0; } }
        let out = bicubic_resize(&rgb, 14, 14, 32, 32, 3);
        for p in out.chunks_exact(3) {
            assert!((p[0] - 1.0).abs() < 1e-5 && (p[1] - 2.0).abs() < 1e-5 && (p[2] - 3.0).abs() < 1e-5);
        }
    }

    /// Parity check against the old `convert.py` output. Ignored by default: it
    /// needs the 1.6 GB `model.ckpt` and its Python-converted reference. Run with
    /// `PYRE_TRIPOSR_DIR=<folder with model.ckpt and triposr_f16.safetensors>`
    /// `cargo test --bin pyre convert_ckpt -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the TripoSR model.ckpt and its converted reference safetensors"]
    fn convert_ckpt_matches_python_reference() {
        let Ok(dir) = std::env::var("PYRE_TRIPOSR_DIR") else {
            println!("PYRE_TRIPOSR_DIR not set; skipping parity check");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let src = dir.join("model.ckpt");
        let reference = dir.join("triposr_f16.safetensors");
        assert!(src.is_file(), "missing {}", src.display());
        assert!(reference.is_file(), "missing {}", reference.display());
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("triposr_f16.safetensors");
        let t = std::time::Instant::now();
        convert_ckpt(&src, &out).expect("Rust conversion");
        println!("converted in {:.1}s -> {} bytes", t.elapsed().as_secs_f64(), std::fs::metadata(&out).unwrap().len());

        let ours_bytes = std::fs::read(&out).unwrap();
        let ref_bytes = std::fs::read(&reference).unwrap();
        let ours = safetensors::SafeTensors::deserialize(&ours_bytes).unwrap();
        let reference = safetensors::SafeTensors::deserialize(&ref_bytes).unwrap();
        let mut our_names: Vec<String> = ours.names().into_iter().map(str::to_string).collect();
        let mut ref_names: Vec<String> = reference.names().into_iter().map(str::to_string).collect();
        our_names.sort();
        ref_names.sort();
        assert_eq!(our_names, ref_names, "tensor name sets differ");

        let mut worst = (0f64, String::new());
        let mut mismatched_f32 = 0usize;
        for name in &our_names {
            let a = ours.tensor(name).unwrap();
            let b = reference.tensor(name).unwrap();
            assert_eq!(a.dtype(), b.dtype(), "{name}: dtype");
            assert_eq!(a.shape(), b.shape(), "{name}: shape");
            let is_f32 = a.dtype() == safetensors::Dtype::F32;
            let av = decode_safetensors(a);
            let bv = decode_safetensors(b);
            assert_eq!(av.len(), bv.len(), "{name}: element count");
            let mut max = 0f64;
            for (x, y) in av.iter().zip(&bv) {
                let d = (*x as f64 - *y as f64).abs();
                if d > max { max = d; }
                if d != 0.0 && is_f32 { mismatched_f32 += 1; }
            }
            if max > worst.0 { worst = (max, name.clone()); }
        }
        println!("largest absolute difference: {:.3e} on {}", worst.0, worst.1);
        assert_eq!(mismatched_f32, 0, "f32 tensors must be bit-identical");
        assert!(worst.0 < 1e-3, "worst tensor {} differs by {:.3e}", worst.1, worst.0);
        // dino_pos512 is the only tensor whose values are recomputed, so report it.
        let a = ours.tensor("dino_pos512").unwrap();
        let b = reference.tensor("dino_pos512").unwrap();
        let av = decode_safetensors(a);
        let bv = decode_safetensors(b);
        let max = av.iter().zip(&bv).map(|(x, y)| (*x - *y).abs()).fold(0f32, f32::max);
        let diffs = av.iter().zip(&bv).filter(|(x, y)| x != y).count();
        println!("dino_pos512: max |diff| {max:.3e}, {diffs}/{} values differ", av.len());
    }

    fn decode_safetensors(v: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
        match v.dtype() {
            safetensors::Dtype::F32 => v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            safetensors::Dtype::F16 => v.data().chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            d => panic!("unexpected dtype {d:?}"),
        }
    }
}
