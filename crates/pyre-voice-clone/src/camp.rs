//! CAMPPlus, the 192-d speaker embedding the mean-flow UNet conditions on.
//!
//! This is not the voice encoder. T3 still takes the 256-d `ve.safetensors`
//! vector. The flow's `spk_embed_affine_layer` takes this 192-d vector, L2
//! normalized. Weights live under `speaker_encoder.*` in
//! `s3gen_meanflow.safetensors` (3D-Speaker / CosyVoice, Apache-2.0).
//!
//! Input is [`crate::frontend::kaldi_fbank`], already mean-subtracted, shaped
//! `[frames, 80]`. The head sees it as a 1-channel image `[1, 80, frames]`.

use crate::frontend::kaldi_fbank;
use crate::s3::discover_s3;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const EMB: usize = 192;

pub struct CampPlus {
    w: HashMap<String, Vec<f32>>,
}

impl CampPlus {
    pub fn discover() -> Option<PathBuf> {
        discover_s3()
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes).context("safetensors")?;
        let mut w = HashMap::new();
        for name in st.names() {
            if let Some(rest) = name.strip_prefix("speaker_encoder.") {
                if rest.ends_with("num_batches_tracked") {
                    continue;
                }
                w.insert(rest.to_string(), read_f32(&st, name)?);
            }
        }
        if w.is_empty() {
            bail!("no speaker_encoder tensors in {}", path.display());
        }
        Ok(Self { w })
    }

    /// 192-d embedding of a 16 kHz clip. Dither is off, so this is a function
    /// of the samples.
    pub fn embed(&self, wav16: &[f32]) -> Result<Vec<f32>> {
        let fbank = kaldi_fbank(wav16);
        if fbank.is_empty() {
            bail!("reference is shorter than one 25 ms frame");
        }
        let frames = fbank.len() / 80;
        // (T, F) -> (F, T)
        let mut x = vec![0f32; 80 * frames];
        for t in 0..frames {
            for f in 0..80 {
                x[f * frames + t] = fbank[t * 80 + f];
            }
        }
        let mut h = self.head(&x, frames)?;
        let t = h.len() / 320;
        h = self.tdnn(&h, 320, 128, t, 5, 2, 1)?;
        let mut t = h.len() / 128;
        h = self.dense_block(&h, 128, t, 12, 1, "xvector.block1")?;
        t = h.len() / (128 + 12 * 32);
        h = self.transit(&h, 128 + 12 * 32, 256, t, "xvector.transit1")?;
        t = h.len() / 256;
        h = self.dense_block(&h, 256, t, 24, 2, "xvector.block2")?;
        t = h.len() / (256 + 24 * 32);
        h = self.transit(&h, 256 + 24 * 32, 512, t, "xvector.transit2")?;
        t = h.len() / 512;
        h = self.dense_block(&h, 512, t, 16, 2, "xvector.block3")?;
        t = h.len() / (512 + 16 * 32);
        h = self.transit(&h, 512 + 16 * 32, 512, t, "xvector.transit3")?;
        t = h.len() / 512;
        bn1d(
            &mut h,
            512,
            t,
            self.affine("xvector.out_nonlinear.batchnorm")?,
            1e-5,
        );
        relu(&mut h);
        let mut stats = stats_pool(&h, 512, t);
        // DenseLayer sees [C, 1].
        stats = conv1d(
            self.req("xvector.dense.linear.weight")?,
            None,
            &stats,
            1024,
            EMB,
            1,
            1,
            1,
            0,
            1,
        );
        bn1d_affine_false(
            &mut stats,
            EMB,
            1,
            self.mean_var("xvector.dense.nonlinear.batchnorm")?,
            1e-5,
        );
        Ok(stats)
    }

    fn head(&self, freq_time: &[f32], frames: usize) -> Result<Vec<f32>> {
        // [1, 80, T]
        let (mut x, h, w) = conv2d(
            self.req("head.conv1.weight")?,
            &unsqueeze1(freq_time, 80, frames),
            1,
            32,
            80,
            frames,
            3,
            3,
            1,
            1,
            1,
        );
        bn2d(&mut x, 32, h * w, self.affine("head.bn1")?, 1e-5);
        relu(&mut x);
        let (x, h, w) = self.res_layer(&x, 32, 32, h, w, 2, "head.layer1")?;
        let (x, h, w) = self.res_layer(&x, 32, 32, h, w, 2, "head.layer2")?;
        let (mut x, h, w) = conv2d(
            self.req("head.conv2.weight")?,
            &x,
            32,
            32,
            h,
            w,
            3,
            3,
            2,
            1,
            1,
        );
        bn2d(&mut x, 32, h * w, self.affine("head.bn2")?, 1e-5);
        relu(&mut x);
        // [32, F, T] -> [32*F, T]
        let mut flat = vec![0f32; 32 * h * w];
        for c in 0..32 {
            for f in 0..h {
                for t in 0..w {
                    flat[(c * h + f) * w + t] = x[(c * h + f) * w + t];
                }
            }
        }
        let _ = flat.len();
        Ok(x_as_channels(&x, 32, h, w))
    }

    fn res_layer(
        &self,
        x: &[f32],
        mut cin: usize,
        planes: usize,
        mut h: usize,
        mut w: usize,
        first_stride: usize,
        prefix: &str,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let mut cur = x.to_vec();
        for i in 0..2 {
            let stride = if i == 0 { first_stride } else { 1 };
            let name = format!("{prefix}.{i}");
            let (y, ho, wo) = self.res_block(&cur, cin, planes, h, w, stride, &name)?;
            cur = y;
            h = ho;
            w = wo;
            cin = planes;
        }
        Ok((cur, h, w))
    }

    fn res_block(
        &self,
        x: &[f32],
        cin: usize,
        planes: usize,
        h: usize,
        w: usize,
        stride: usize,
        prefix: &str,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let (mut y, ho, wo) = conv2d(
            self.req(&format!("{prefix}.conv1.weight"))?,
            x,
            cin,
            planes,
            h,
            w,
            3,
            3,
            stride,
            1,
            1,
        );
        bn2d(
            &mut y,
            planes,
            ho * wo,
            self.affine(&format!("{prefix}.bn1"))?,
            1e-5,
        );
        relu(&mut y);
        let (mut y, ho, wo) = conv2d(
            self.req(&format!("{prefix}.conv2.weight"))?,
            &y,
            planes,
            planes,
            ho,
            wo,
            3,
            3,
            1,
            1,
            1,
        );
        bn2d(
            &mut y,
            planes,
            ho * wo,
            self.affine(&format!("{prefix}.bn2"))?,
            1e-5,
        );
        let skip = if stride != 1 || cin != planes {
            let (mut s, sho, swo) = conv2d(
                self.req(&format!("{prefix}.shortcut.0.weight"))?,
                x,
                cin,
                planes,
                h,
                w,
                1,
                1,
                stride,
                1,
                0,
            );
            bn2d(
                &mut s,
                planes,
                sho * swo,
                self.affine(&format!("{prefix}.shortcut.1"))?,
                1e-5,
            );
            let _ = (sho, swo);
            s
        } else {
            x.to_vec()
        };
        for (a, b) in y.iter_mut().zip(&skip) {
            *a += *b;
        }
        relu(&mut y);
        Ok((y, ho, wo))
    }

    fn tdnn(
        &self,
        x: &[f32],
        cin: usize,
        cout: usize,
        t: usize,
        k: usize,
        stride: usize,
        dil: usize,
    ) -> Result<Vec<f32>> {
        let pad = (k - 1) / 2 * dil;
        let mut y = conv1d(
            self.req("xvector.tdnn.linear.weight")?,
            None,
            x,
            cin,
            cout,
            t,
            k,
            stride,
            pad,
            dil,
        );
        let to = y.len() / cout;
        bn1d(
            &mut y,
            cout,
            to,
            self.affine("xvector.tdnn.nonlinear.batchnorm")?,
            1e-5,
        );
        relu(&mut y);
        Ok(y)
    }

    fn dense_block(
        &self,
        x: &[f32],
        mut cin: usize,
        t: usize,
        layers: usize,
        dil: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let mut cur = x.to_vec();
        for i in 1..=layers {
            let name = format!("{prefix}.tdnnd{i}");
            let y = self.cam_layer(&cur, cin, 32, 128, t, 3, dil, &name)?;
            cur = cat_channels(&cur, cin, &y, 32, t);
            cin += 32;
        }
        Ok(cur)
    }

    fn cam_layer(
        &self,
        x: &[f32],
        cin: usize,
        growth: usize,
        bn_ch: usize,
        t: usize,
        k: usize,
        dil: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        bn1d(
            &mut h,
            cin,
            t,
            self.affine(&format!("{prefix}.nonlinear1.batchnorm"))?,
            1e-5,
        );
        relu(&mut h);
        h = conv1d(
            self.req(&format!("{prefix}.linear1.weight"))?,
            None,
            &h,
            cin,
            bn_ch,
            t,
            1,
            1,
            0,
            1,
        );
        bn1d(
            &mut h,
            bn_ch,
            t,
            self.affine(&format!("{prefix}.nonlinear2.batchnorm"))?,
            1e-5,
        );
        relu(&mut h);
        let pad = (k - 1) / 2 * dil;
        let local = conv1d(
            self.req(&format!("{prefix}.cam_layer.linear_local.weight"))?,
            None,
            &h,
            bn_ch,
            growth,
            t,
            k,
            1,
            pad,
            dil,
        );
        let context = context_map(&h, bn_ch, t);
        let mut g = conv1d(
            self.req(&format!("{prefix}.cam_layer.linear1.weight"))?,
            Some(self.req(&format!("{prefix}.cam_layer.linear1.bias"))?),
            &context,
            bn_ch,
            bn_ch / 2,
            t,
            1,
            1,
            0,
            1,
        );
        relu(&mut g);
        let mut m = conv1d(
            self.req(&format!("{prefix}.cam_layer.linear2.weight"))?,
            Some(self.req(&format!("{prefix}.cam_layer.linear2.bias"))?),
            &g,
            bn_ch / 2,
            growth,
            t,
            1,
            1,
            0,
            1,
        );
        for v in &mut m {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        Ok(local.iter().zip(&m).map(|(a, b)| a * b).collect())
    }

    fn transit(
        &self,
        x: &[f32],
        cin: usize,
        cout: usize,
        t: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        bn1d(
            &mut h,
            cin,
            t,
            self.affine(&format!("{prefix}.nonlinear.batchnorm"))?,
            1e-5,
        );
        relu(&mut h);
        Ok(conv1d(
            self.req(&format!("{prefix}.linear.weight"))?,
            None,
            &h,
            cin,
            cout,
            t,
            1,
            1,
            0,
            1,
        ))
    }

    fn req(&self, name: &str) -> Result<&[f32]> {
        self.w
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| name.to_string())
    }

    fn affine(&self, prefix: &str) -> Result<Affine<'_>> {
        Ok(Affine {
            w: self.req(&format!("{prefix}.weight"))?,
            b: self.req(&format!("{prefix}.bias"))?,
            mean: self.req(&format!("{prefix}.running_mean"))?,
            var: self.req(&format!("{prefix}.running_var"))?,
        })
    }

    fn mean_var(&self, prefix: &str) -> Result<(&[f32], &[f32])> {
        Ok((
            self.req(&format!("{prefix}.running_mean"))?,
            self.req(&format!("{prefix}.running_var"))?,
        ))
    }
}

struct Affine<'a> {
    w: &'a [f32],
    b: &'a [f32],
    mean: &'a [f32],
    var: &'a [f32],
}

fn read_f32(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = st.tensor(name)?;
    if view.dtype() != safetensors::Dtype::F32 {
        bail!("{name} is not f32");
    }
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn unsqueeze1(freq_time: &[f32], f: usize, t: usize) -> Vec<f32> {
    let mut x = vec![0f32; f * t];
    x.copy_from_slice(&freq_time[..f * t]);
    x
}

fn x_as_channels(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                out[(ci * h + hi) * w + wi] = x[(ci * h + hi) * w + wi];
            }
        }
    }
    out
}

fn cat_channels(a: &[f32], ca: usize, b: &[f32], cb: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; (ca + cb) * t];
    out[..ca * t].copy_from_slice(&a[..ca * t]);
    out[ca * t..].copy_from_slice(&b[..cb * t]);
    out
}

fn relu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn bn1d(x: &mut [f32], c: usize, t: usize, a: Affine<'_>, eps: f32) {
    for ch in 0..c {
        let scale = a.w[ch] / (a.var[ch] + eps).sqrt();
        let shift = a.b[ch] - a.mean[ch] * scale;
        for i in 0..t {
            let idx = ch * t + i;
            x[idx] = x[idx] * scale + shift;
        }
    }
}

fn bn1d_affine_false(x: &mut [f32], c: usize, t: usize, stats: (&[f32], &[f32]), eps: f32) {
    let (mean, var) = stats;
    for ch in 0..c {
        let scale = 1.0 / (var[ch] + eps).sqrt();
        let shift = -mean[ch] * scale;
        for i in 0..t {
            let idx = ch * t + i;
            x[idx] = x[idx] * scale + shift;
        }
    }
}

fn bn2d(x: &mut [f32], c: usize, hw: usize, a: Affine<'_>, eps: f32) {
    bn1d(x, c, hw, a, eps);
}

fn conv1d(
    w: &[f32],
    b: Option<&[f32]>,
    x: &[f32],
    cin: usize,
    cout: usize,
    t: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
) -> Vec<f32> {
    let t_out = (t + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    let mut y = vec![0f32; cout * t_out];
    for oc in 0..cout {
        for ot in 0..t_out {
            let mut acc = b.map(|b| b[oc]).unwrap_or(0.0);
            for ic in 0..cin {
                for kk in 0..k {
                    let it =
                        ot as isize * stride as isize + kk as isize * dil as isize - pad as isize;
                    if it < 0 || it >= t as isize {
                        continue;
                    }
                    acc += x[ic * t + it as usize] * w[(oc * cin + ic) * k + kk];
                }
            }
            y[oc * t_out + ot] = acc;
        }
    }
    y
}

fn conv2d(
    w: &[f32],
    x: &[f32],
    cin: usize,
    cout: usize,
    h: usize,
    width: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let ho = (h + 2 * pad - kh) / stride_h + 1;
    let wo = (width + 2 * pad - kw) / stride_w + 1;
    let mut y = vec![0f32; cout * ho * wo];
    for oc in 0..cout {
        for iy in 0..ho {
            for ix in 0..wo {
                let mut acc = 0.0f32;
                for ic in 0..cin {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = iy as isize * stride_h as isize + ky as isize - pad as isize;
                            let in_x = ix as isize * stride_w as isize + kx as isize - pad as isize;
                            if in_y < 0 || in_x < 0 || in_y >= h as isize || in_x >= width as isize
                            {
                                continue;
                            }
                            let xv = x[(ic * h + in_y as usize) * width + in_x as usize];
                            let wv = w[(((oc * cin + ic) * kh + ky) * kw) + kx];
                            acc += xv * wv;
                        }
                    }
                }
                y[(oc * ho + iy) * wo + ix] = acc;
            }
        }
    }
    (y, ho, wo)
}

fn context_map(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let seg = seg_pool(x, c, t);
    let mut out = vec![0f32; c * t];
    for ch in 0..c {
        let mut mean = 0.0f32;
        for i in 0..t {
            mean += x[ch * t + i];
        }
        mean /= t as f32;
        for i in 0..t {
            out[ch * t + i] = mean + seg[ch * t + i];
        }
    }
    out
}

fn seg_pool(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let seg_len = 100usize;
    let nseg = (t + seg_len - 1) / seg_len;
    let mut pooled = vec![0f32; c * nseg];
    for ch in 0..c {
        for s in 0..nseg {
            let start = s * seg_len;
            let end = (start + seg_len).min(t);
            let mut sum = 0.0f32;
            for i in start..end {
                sum += x[ch * t + i];
            }
            pooled[ch * nseg + s] = sum / (end - start) as f32;
        }
    }
    let mut out = vec![0f32; c * t];
    for ch in 0..c {
        for i in 0..t {
            out[ch * t + i] = pooled[ch * nseg + i / seg_len];
        }
    }
    out
}

fn stats_pool(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * 2];
    for ch in 0..c {
        let mut mean = 0.0f64;
        for i in 0..t {
            mean += x[ch * t + i] as f64;
        }
        mean /= t as f64;
        let mut var = 0.0f64;
        for i in 0..t {
            let d = x[ch * t + i] as f64 - mean;
            var += d * d;
        }
        var /= (t - 1) as f64;
        out[ch] = mean as f32;
        out[c + ch] = var.sqrt() as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn campplus_matches_torch_on_a_tone() {
        let path = std::env::var("PYRE_CHATTERBOX_S3").unwrap_or_else(|_| String::new());
        if path.is_empty() {
            eprintln!("set PYRE_CHATTERBOX_S3");
            return;
        }
        let camp = CampPlus::load(Path::new(&path)).unwrap();
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("testdata")
                .join("camp_tone.f32"),
        )
        .unwrap();
        let wav: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want_bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("testdata")
                .join("camp_tone_emb.f32"),
        )
        .unwrap();
        let want: Vec<f32> = want_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let got = camp.embed(&wav).unwrap();
        assert_eq!(got.len(), 192);
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max < 1e-3,
            "camp max abs {max} got {:?} want {:?}",
            &got[..4],
            &want[..4]
        );
    }
}
