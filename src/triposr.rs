//! TripoSR (stabilityai/TripoSR, MIT): single image -> triplane NeRF -> marching-cubes mesh with vertex colours.
//! Native Burn port. Weights: `<store>/cabin/triposr/triposr_f16.safetensors` (converted from `model.ckpt` by
//! convert.py: f16 storage, decoder f32, DINO position table pre-interpolated to the 32x32 grid of 512 px).
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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const RADIUS: f32 = 0.87;
pub fn weights_dir(store_root: &Path) -> PathBuf { std::env::var("PYRE_TRIPOSR_DIR").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("triposr")) }
fn file(dir: &Path) -> PathBuf { dir.join("triposr_f16.safetensors") }
pub fn available(dir: &Path) -> bool { file(dir).is_file() }

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

/// Binary glTF 2.0: POSITION, NORMAL (area-weighted), COLOR_0 (unsigned byte normalised), u32 indices.
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
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &m.pos { for q in 0..3 { mn[q] = mn[q].min(p[q]); mx[q] = mx[q].max(p[q]); } }
    let mut bin = Vec::new();
    for p in &m.pos { for q in 0..3 { bin.extend(p[q].to_le_bytes()); } }
    let nor_off = bin.len(); for p in &nor { for q in 0..3 { bin.extend(p[q].to_le_bytes()); } }
    let col_off = bin.len(); for c in &m.col { for q in 0..3 { bin.push((c[q].clamp(0.0, 1.0) * 255.0).round() as u8); } bin.push(255); }
    let idx_off = bin.len(); for t in &m.tri { for &i in t { bin.extend(i.to_le_bytes()); } }
    while bin.len() % 4 != 0 { bin.push(0); }
    let json = serde_json::json!({
        "asset": {"version": "2.0", "generator": "pyre cabin (TripoSR native)"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}, "doubleSided": true}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": nor_off, "target": 34962},
            {"buffer": 0, "byteOffset": nor_off, "byteLength": col_off - nor_off, "target": 34962},
            {"buffer": 0, "byteOffset": col_off, "byteLength": idx_off - col_off, "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": m.tri.len() * 12, "target": 34963}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": true, "count": n, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": m.tri.len() * 3, "type": "SCALAR"}]
    });
    let mut js = serde_json::to_vec(&json).unwrap(); while js.len() % 4 != 0 { js.push(b' '); }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(0x4654_6C67u32.to_le_bytes()); out.extend(2u32.to_le_bytes()); out.extend((total as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes()); out.extend(0x4E4F_534Au32.to_le_bytes()); out.extend(js);
    out.extend((bin.len() as u32).to_le_bytes()); out.extend(0x004E_4942u32.to_le_bytes()); out.extend(bin);
    out
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
