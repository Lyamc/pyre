//! WeSpeaker-style ResNet34 embedding (fused conv+BN, host f32).

use crate::model::vad::weights::{load_f32, load_named, TensorMap};

const N_MELS: usize = 80;
const SAMPLE_RATE: usize = 16_000;

struct ConvBn {
    w: Vec<f32>,
    b: Vec<f32>,
    out_c: usize,
    k: usize,
    stride: usize,
}

struct Block {
    conv1: ConvBn,
    conv2: ConvBn,
    down: Option<ConvBn>,
}

pub struct SpeakerEmbedder {
    stem: Option<ConvBn>,
    layers: Vec<Vec<Block>>,
    embed_w: Vec<f32>,
    embed_b: Vec<f32>,
    embed_in: usize,
    embed_out: usize,
}

impl SpeakerEmbedder {
    pub fn from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(dir.join("embedding.safetensors"))
            .or_else(|_| std::fs::read(dir.join("model.safetensors")))
            .map_err(|e| format!("embedding weights: {e}"))?;
        let map = load_f32(&bytes)?;
        Self::from_tensors(&map)
    }

    pub fn from_tensors(map: &TensorMap) -> Result<Self, String> {
        let stem = conv_bn(map, "stem.conv", 32, 1, 3, 1).ok();
        // WeSpeaker ResNet34 groups are named layer1..layer4 (not 0-based).
        let layout = [
            (1usize, 3usize, 32usize, 32usize, 1usize),
            (2, 4, 32, 64, 2),
            (3, 6, 64, 128, 2),
            (4, 3, 128, 256, 2),
        ];
        let mut layers = Vec::new();
        for &(li, n, in_c, out_c, stride) in &layout {
            let mut blocks = Vec::new();
            for bi in 0..n {
                let p = format!("layer{li}.{bi}");
                let ic = if bi == 0 { in_c } else { out_c };
                let st = if bi == 0 { stride } else { 1 };
                let Ok(conv1) = conv_bn(map, &format!("{p}.conv1"), out_c, ic, 3, st) else {
                    break;
                };
                let Ok(conv2) = conv_bn(map, &format!("{p}.conv2"), out_c, out_c, 3, 1) else {
                    break;
                };
                let down = if ic != out_c || st != 1 {
                    conv_bn(map, &format!("{p}.down"), out_c, ic, 1, st).ok()
                } else {
                    None
                };
                blocks.push(Block { conv1, conv2, down });
            }
            if !blocks.is_empty() {
                layers.push(blocks);
            }
        }
        let (embed_w, shape) = load_named(map, "embed.weight")?;
        let embed_b = load_named(map, "embed.bias")
            .map(|(v, _)| v)
            .unwrap_or_else(|_| vec![0.0; shape[0]]);
        let (embed_out, embed_in) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            (embed_b.len().max(1), embed_w.len() / embed_b.len().max(1))
        };
        Ok(Self {
            stem,
            layers,
            embed_w,
            embed_b,
            embed_in,
            embed_out,
        })
    }

    pub fn embed(&self, samples: &[f32], sample_rate: usize) -> Result<Vec<f32>, String> {
        let audio = if sample_rate == SAMPLE_RATE {
            samples.to_vec()
        } else {
            pyre_whisper_audio::resample_mono(samples, sample_rate, SAMPLE_RATE)
                .unwrap_or_else(|_| samples.to_vec())
        };
        let mel = log_mel_fbank(&audio, SAMPLE_RATE, N_MELS);
        if mel.is_empty() {
            return Ok(vec![0.0; self.embed_out]);
        }
        let t = mel.len() / N_MELS;
        // x: [1, 80, T]
        let mut x = mel;
        let mut c = 1usize;
        let mut h = N_MELS;
        let mut w = t;
        if let Some(stem) = &self.stem {
            let (y, oc, oh, ow) = conv2d(&x, c, h, w, stem, 1);
            x = relu_vec(y);
            c = oc;
            h = oh;
            w = ow;
        }
        for layer in &self.layers {
            for block in layer {
                let residual = if let Some(down) = &block.down {
                    conv2d(&x, c, h, w, down, 0).0
                } else {
                    x.clone()
                };
                let (y1, c1, h1, w1) = conv2d(&x, c, h, w, &block.conv1, 1);
                let y1 = relu_vec(y1);
                let (y2, c2, h2, w2) = conv2d(&y1, c1, h1, w1, &block.conv2, 0);
                let mut residual = residual;
                residual.resize(y2.len(), 0.0);
                x = y2
                    .iter()
                    .zip(residual.iter())
                    .map(|(a, b)| (a + b).max(0.0))
                    .collect();
                c = c2;
                h = h2;
                w = w2;
            }
        }
        // Stats pooling over time+freq → [2C]
        let mut mean = vec![0f32; c];
        let mut m2 = vec![0f32; c];
        let n = (h * w).max(1) as f32;
        for ch in 0..c {
            let mut s = 0.0;
            for i in 0..(h * w) {
                s += x[ch * h * w + i];
            }
            mean[ch] = s / n;
        }
        for ch in 0..c {
            let mut s = 0.0;
            for i in 0..(h * w) {
                let d = x[ch * h * w + i] - mean[ch];
                s += d * d;
            }
            m2[ch] = (s / n).sqrt();
        }
        let mut pooled = mean;
        pooled.extend_from_slice(&m2);
        if pooled.len() != self.embed_in {
            pooled.resize(self.embed_in, 0.0);
        }
        let mut out = vec![0f32; self.embed_out];
        for o in 0..self.embed_out {
            let mut acc = self.embed_b.get(o).copied().unwrap_or(0.0);
            let row = o * self.embed_in;
            for i in 0..self.embed_in {
                acc += pooled[i] * self.embed_w[row + i];
            }
            out[o] = acc;
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out {
            *v /= norm;
        }
        Ok(out)
    }
}

fn conv_bn(
    map: &TensorMap,
    prefix: &str,
    out_c: usize,
    in_c: usize,
    k: usize,
    stride: usize,
) -> Result<ConvBn, String> {
    let (w, shape) = load_named(map, &format!("{prefix}.weight"))?;
    let b = load_named(map, &format!("{prefix}.bias"))
        .map(|(v, _)| v)
        .unwrap_or_else(|_| vec![0.0; out_c]);
    let w = reshape_conv2d(w, &shape, out_c, in_c, k)?;
    Ok(ConvBn {
        w,
        b,
        out_c,
        k,
        stride,
    })
}

fn reshape_conv2d(
    data: Vec<f32>,
    shape: &[usize],
    out_c: usize,
    in_c: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if data.len() == out_c * in_c * k * k {
        return Ok(data);
    }
    Err(format!(
        "conv weight len {} != {} (shape {shape:?})",
        data.len(),
        out_c * in_c * k * k
    ))
}

fn conv2d(
    x: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    layer: &ConvBn,
    pad: usize,
) -> (Vec<f32>, usize, usize, usize) {
    let k = layer.k;
    let stride = layer.stride.max(1);
    let oh = (h + 2 * pad).saturating_sub(k) / stride + 1;
    let ow = (w + 2 * pad).saturating_sub(k) / stride + 1;
    let mut y = vec![0f32; layer.out_c * oh * ow];
    for oc in 0..layer.out_c {
        let bias = layer.b.get(oc).copied().unwrap_or(0.0);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias;
                for ic in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            let sy = iy as isize - pad as isize;
                            let sx = ix as isize - pad as isize;
                            if sy >= 0 && sx >= 0 && (sy as usize) < h && (sx as usize) < w {
                                let xv = x[ic * h * w + sy as usize * w + sx as usize];
                                let wv = layer.w[oc * in_c * k * k + ic * k * k + ky * k + kx];
                                acc += xv * wv;
                            }
                        }
                    }
                }
                y[oc * oh * ow + oy * ow + ox] = acc;
            }
        }
    }
    (y, layer.out_c, oh, ow)
}

fn relu_vec(mut x: Vec<f32>) -> Vec<f32> {
    for v in &mut x {
        *v = v.max(0.0);
    }
    x
}

fn log_mel_fbank(samples: &[f32], sr: usize, n_mels: usize) -> Vec<f32> {
    let n_fft = 512;
    let hop = 320; // 20 ms @ 16 kHz (clustering windows are short)
    let win = hann(n_fft);
    let n_freq = n_fft / 2 + 1;
    let banks = mel_banks(sr as f32, n_fft, n_mels);
    let mut frames = Vec::new();
    let mut i = 0usize;
    while i + n_fft <= samples.len() {
        let mut re = vec![0f32; n_fft];
        let mut im = vec![0f32; n_fft];
        for n in 0..n_fft {
            re[n] = samples[i + n] * win[n];
        }
        fft_radix2(&mut re, &mut im);
        let mut spec = vec![0f32; n_freq];
        for f in 0..n_freq {
            spec[f] = re[f] * re[f] + im[f] * im[f];
        }
        let mut mel = vec![0f32; n_mels];
        for m in 0..n_mels {
            let mut acc = 0.0;
            for f in 0..n_freq {
                acc += spec[f] * banks[m * n_freq + f];
            }
            mel[m] = (acc.max(1e-10)).ln();
        }
        frames.push(mel);
        i += hop;
    }
    if frames.is_empty() {
        return Vec::new();
    }
    // CMN
    let t = frames.len();
    let mut mean = vec![0f32; n_mels];
    for fr in &frames {
        for m in 0..n_mels {
            mean[m] += fr[m];
        }
    }
    for m in mean.iter_mut() {
        *m /= t as f32;
    }
    let mut out = vec![0f32; n_mels * t];
    for (ti, fr) in frames.iter().enumerate() {
        for m in 0..n_mels {
            out[m * t + ti] = fr[m] - mean[m];
        }
    }
    out
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn mel_banks(sr: f32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let fmax = sr / 2.0;
    let mut pts = vec![0f32; n_mels + 2];
    for (i, p) in pts.iter_mut().enumerate() {
        let hz = fmax * i as f32 / (n_mels + 1) as f32;
        *p = 2595.0 * (1.0 + hz / 700.0).log10();
    }
    let hz: Vec<f32> = pts
        .iter()
        .map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0))
        .collect();
    let mut w = vec![0f32; n_mels * n_freq];
    for m in 0..n_mels {
        let (left, center, right) = (hz[m], hz[m + 1], hz[m + 2]);
        for f in 0..n_freq {
            let freq = sr * f as f32 / n_fft as f32;
            let lo = (freq - left) / (center - left).max(1e-8);
            let hi = (right - freq) / (right - center).max(1e-8);
            w[m * n_freq + f] = lo.min(hi).max(0.0);
        }
    }
    w
}

fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wlen_re, wlen_im) = (ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for k in 0..half {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + half] * w_re - im[i + k + half] * w_im;
                let vi = re[i + k + half] * w_im + im[i + k + half] * w_re;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
                let nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
        len *= 2;
    }
}
