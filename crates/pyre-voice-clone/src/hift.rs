//! HiFT vocoder: 80-bin mel at 50 Hz to a 24 kHz waveform.
//!
//! Upstream is `HiFTGenerator` in Resemble chatterbox `s3gen/hifigan.py`
//! (CosyVoice, Apache-2.0). Nano builds it with upsample rates `[8, 5, 3]`,
//! kernels `[16, 11, 7]`, source kernels `[7, 7, 11]`, dilations `[1, 3, 5]`,
//! iSTFT `n_fft=16` hop `4`, 8 harmonics, voiced threshold 10.
//!
//! Conv weights that were trained under `weight_norm` are stored as
//! `parametrizations.weight.original0` (the gain, `[C, 1, 1]`) and
//! `original1` (the direction). The fused kernel is `v * g / ||v||`, with the
//! norm over every axis except 0. `Conv1d` kernels are `[out, in, k]`.
//! `ConvTranspose1d` kernels are `[in, out, k]`. `source_downs` are plain
//! convs, not weight-normalized.
//!
//! The harmonic source draws one uniform phase per overtone and gaussian
//! noise per sample. Those draws use SplitMix64 here so two calls with the
//! same seed match. They are not PyTorch's RNG. [`Hift::decode`] given a
//! saved excitation is the check against `torch.istft`.
//!
//! This stage does not turn S3 or T3 token ids into a mel. Mean-flow
//! (`flow.*`, CAMPPlus, two Euler steps) is still required, and
//! [`crate::text_to_cloned_speech_ready`] stays false. A reconstruction of a
//! reference mel is not a cloned voice.

use crate::s3::{discover_s3, S3_FILENAME};
use anyhow::{bail, Context, Result};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};

pub const S3GEN_SR: u32 = 24_000;
pub const MEL_BINS: usize = 80;
pub const MEL_HOP: usize = 480;
const N_FFT: usize = 16;
const ISTFT_HOP: usize = 4;
const N_FREQ: usize = N_FFT / 2 + 1;
const BASE: usize = 512;
const NB_HARM: usize = 8;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_F0: f32 = 10.0;
const AUDIO_LIMIT: f32 = 0.99;
const LRELU: f32 = 0.1;
const SNAKE_EPS: f32 = 1e-9;
const UP_RATES: [usize; 3] = [8, 5, 3];
const UP_KERNELS: [usize; 3] = [16, 11, 7];
const RES_KERNELS: [usize; 3] = [3, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];
const SRC_KERNELS: [usize; 3] = [7, 7, 11];

struct Conv1d {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
}

struct ConvTr {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    alpha1: Vec<Vec<f32>>,
    alpha2: Vec<Vec<f32>>,
}

/// HiFT weights from `mel2wav.*` inside `s3gen_meanflow.safetensors`.
pub struct Hift {
    conv_pre: Conv1d,
    ups: Vec<ConvTr>,
    source_downs: Vec<Conv1d>,
    source_res: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    f0_convs: Vec<Conv1d>,
    f0_w: Vec<f32>,
    f0_b: f32,
    source_w: Vec<f32>,
    source_b: f32,
}

impl Hift {
    pub fn discover() -> Option<PathBuf> {
        discover_s3()
    }

    pub fn filename() -> &'static str {
        S3_FILENAME
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes).context("safetensors")?;
        let conv_pre = conv_wn(&st, "mel2wav.conv_pre", 1, 3, 1)?;
        let mut ups = Vec::new();
        for i in 0..3 {
            let pad = (UP_KERNELS[i] - UP_RATES[i]) / 2;
            ups.push(conv_tr_wn(
                &st,
                &format!("mel2wav.ups.{i}"),
                UP_RATES[i],
                pad,
            )?);
        }
        let src_strides = [15usize, 3, 1];
        let src_kernels = [30usize, 6, 1];
        let src_pads = [7usize, 1, 0];
        let mut source_downs = Vec::new();
        for i in 0..3 {
            source_downs.push(conv_plain(
                &st,
                &format!("mel2wav.source_downs.{i}"),
                src_strides[i],
                src_pads[i],
                1,
            )?);
            let shape = tensor_shape(&st, &format!("mel2wav.source_downs.{i}.weight"))?;
            if shape != [source_downs[i].cout, N_FFT + 2, src_kernels[i]] {
                bail!("source_downs.{i} shape {shape:?}");
            }
        }
        let source_res = (0..3)
            .map(|i| {
                resblock(
                    &st,
                    &format!("mel2wav.source_resblocks.{i}"),
                    SRC_KERNELS[i],
                    BASE >> (i + 1),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut resblocks = Vec::new();
        for i in 0..3 {
            let ch = BASE >> (i + 1);
            for (j, &k) in RES_KERNELS.iter().enumerate() {
                resblocks.push(resblock(
                    &st,
                    &format!("mel2wav.resblocks.{}", i * 3 + j),
                    k,
                    ch,
                )?);
            }
        }
        let f0_convs = (0..5)
            .map(|i| {
                conv_wn(
                    &st,
                    &format!("mel2wav.f0_predictor.condnet.{}", i * 2),
                    1,
                    1,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let f0_w = named(&st, "mel2wav.f0_predictor.classifier.weight", &[1, 512])?;
        let f0_b = named(&st, "mel2wav.f0_predictor.classifier.bias", &[1])?[0];
        let source_w = named(&st, "mel2wav.m_source.l_linear.weight", &[1, NB_HARM + 1])?;
        let source_b = named(&st, "mel2wav.m_source.l_linear.bias", &[1])?[0];
        Ok(Self {
            conv_pre,
            ups,
            source_downs,
            source_res,
            resblocks,
            conv_post: conv_wn(&st, "mel2wav.conv_post", 1, 3, 1)?,
            f0_convs,
            f0_w,
            f0_b,
            source_w,
            source_b,
        })
    }

    /// Absolute F0 in Hz, one value per mel frame. `[T]`.
    pub fn predict_f0(&self, mel: &[f32]) -> Result<Vec<f32>> {
        let frames = mel_frames(mel)?;
        let mut x = mel.to_vec();
        for conv in &self.f0_convs {
            x = conv.forward(&x);
            elu_inplace(&mut x);
        }
        let mut f0 = vec![0.0; frames];
        let ch = self.f0_convs.last().map(|c| c.cout).unwrap_or(0);
        for t in 0..frames {
            let mut acc = self.f0_b;
            for c in 0..ch {
                acc += x[c * frames + t] * self.f0_w[c];
            }
            f0[t] = acc.abs();
        }
        Ok(f0)
    }

    /// Harmonic-plus-noise excitation at 24 kHz. Same seed, same samples.
    pub fn source_from_f0(&self, f0: &[f32], seed: u64) -> Vec<f32> {
        let mut up = Vec::with_capacity(f0.len() * MEL_HOP);
        for &hz in f0 {
            up.extend(std::iter::repeat(hz).take(MEL_HOP));
        }
        sine_source(&up, &self.source_w, self.source_b, seed)
    }

    /// Mel `[80, T]` plus excitation `[T * 480]` to a 24 kHz waveform.
    pub fn decode(&self, mel: &[f32], source: &[f32]) -> Result<Vec<f32>> {
        let frames = mel_frames(mel)?;
        if source.len() != frames * MEL_HOP {
            bail!(
                "source length {} is not {} mel frames * {}",
                source.len(),
                frames,
                MEL_HOP
            );
        }
        let spec = stft_source(source);
        let mut x = self.conv_pre.forward(mel);
        for i in 0..3 {
            leaky_relu(&mut x, LRELU);
            x = self.ups[i].forward(&x);
            if i + 1 == 3 {
                x = reflect_left(&x, self.ups[i].cout);
            }
            let mut si = self.source_downs[i].forward(&spec);
            si = self.source_res[i].forward(&si);
            if si.len() != x.len() {
                bail!(
                    "source fusion {i} length {} != upsample {}",
                    si.len() / self.source_downs[i].cout.max(1),
                    x.len() / self.ups[i].cout.max(1)
                );
            }
            for (a, b) in x.iter_mut().zip(&si) {
                *a += *b;
            }
            let mut acc = vec![0.0; x.len()];
            for j in 0..3 {
                let y = self.resblocks[i * 3 + j].forward(&x);
                for (a, b) in acc.iter_mut().zip(&y) {
                    *a += *b;
                }
            }
            for v in &mut acc {
                *v /= 3.0;
            }
            x = acc;
        }
        leaky_relu(&mut x, 0.01);
        x = self.conv_post.forward(&x);
        let time = x.len() / (N_FFT + 2);
        // conv_post emits n_fft+2 channels. The first half is a log magnitude
        // and the second half is a phase that is passed through sin.
        let (log_mag, raw_phase) = x.split_at(N_FREQ * time);
        let magnitude: Vec<f32> = log_mag.iter().map(|v| v.exp()).collect();
        let angle: Vec<f32> = raw_phase.iter().map(|v| v.sin()).collect();
        let mut wav = istft(&magnitude, &angle, time);
        for s in &mut wav {
            if !s.is_finite() {
                bail!("HiFT iSTFT produced a non-finite sample");
            }
            *s = s.clamp(-AUDIO_LIMIT, AUDIO_LIMIT);
        }
        Ok(wav)
    }

    /// F0, seeded source, then [`decode`](Self::decode).
    pub fn vocode(&self, mel: &[f32], seed: u64) -> Result<Vec<f32>> {
        let f0 = self.predict_f0(mel)?;
        let source = self.source_from_f0(&f0, seed);
        self.decode(mel, &source)
    }
}

fn mel_frames(mel: &[f32]) -> Result<usize> {
    if mel.len() % MEL_BINS != 0 || mel.is_empty() {
        bail!("mel length {} is not a multiple of {MEL_BINS}", mel.len());
    }
    Ok(mel.len() / MEL_BINS)
}

fn sine_source(f0: &[f32], weight: &[f32], bias: f32, seed: u64) -> Vec<f32> {
    let n = f0.len();
    let h = NB_HARM + 1;
    let mut rng = SplitMix::new(seed);
    let mut phase = [0.0f32; NB_HARM + 1];
    for p in phase.iter_mut().skip(1) {
        *p = rng.uniform(-PI, PI);
    }
    let mut cum = vec![0.0f32; h];
    let mut waves = vec![0.0f32; h * n];
    for t in 0..n {
        let voiced = f0[t] > VOICED_F0;
        let uv = if voiced { 1.0 } else { 0.0 };
        let noise_amp = if voiced { NOISE_STD } else { SINE_AMP / 3.0 };
        for i in 0..h {
            cum[i] = (cum[i] + f0[t] * (i + 1) as f32 / S3GEN_SR as f32) % 1.0;
            let theta = 2.0 * PI * cum[i];
            let mut s = SINE_AMP * (theta + phase[i]).sin();
            s = s * uv + noise_amp * rng.normal();
            waves[i * n + t] = s;
        }
    }
    let mut out = vec![0.0; n];
    for t in 0..n {
        let mut acc = bias;
        for i in 0..h {
            acc += waves[i * n + t] * weight[i];
        }
        out[t] = acc.tanh();
    }
    out
}

struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.uniform(1e-7, 1.0);
        let u2 = self.uniform(0.0, 1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

impl Conv1d {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let t_in = x.len() / self.cin;
        let numer = t_in + 2 * self.pad - self.dilation * (self.k - 1) - 1;
        let t_out = numer / self.stride + 1;
        let mut y = vec![0.0; self.cout * t_out];
        for oc in 0..self.cout {
            for ot in 0..t_out {
                let mut acc = self.b[oc];
                let origin = ot * self.stride;
                for ic in 0..self.cin {
                    let xrow = &x[ic * t_in..(ic + 1) * t_in];
                    let wrow = &self.w[((oc * self.cin + ic) * self.k)..][..self.k];
                    for (kk, &wv) in wrow.iter().enumerate() {
                        let it =
                            origin as isize + (kk * self.dilation) as isize - self.pad as isize;
                        if it >= 0 && (it as usize) < t_in {
                            acc += xrow[it as usize] * wv;
                        }
                    }
                }
                y[oc * t_out + ot] = acc;
            }
        }
        y
    }
}

impl ConvTr {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let t_in = x.len() / self.cin;
        let t_out = (t_in - 1) * self.stride + self.k - 2 * self.pad;
        let mut y = vec![0.0; self.cout * t_out];
        for oc in 0..self.cout {
            for t in 0..t_out {
                y[oc * t_out + t] = self.b[oc];
            }
        }
        for ic in 0..self.cin {
            let xrow = &x[ic * t_in..(ic + 1) * t_in];
            for it in 0..t_in {
                let xv = xrow[it];
                if xv == 0.0 {
                    continue;
                }
                for kk in 0..self.k {
                    let ot = it * self.stride + kk;
                    if ot < self.pad {
                        continue;
                    }
                    let ot = ot - self.pad;
                    if ot >= t_out {
                        continue;
                    }
                    for oc in 0..self.cout {
                        let wv = self.w[((ic * self.cout + oc) * self.k) + kk];
                        y[oc * t_out + ot] += xv * wv;
                    }
                }
            }
        }
        y
    }
}

impl ResBlock {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for i in 0..self.convs1.len() {
            let mut y = cur.clone();
            snake(&mut y, self.convs1[i].cout, &self.alpha1[i]);
            y = self.convs1[i].forward(&y);
            snake(&mut y, self.convs2[i].cout, &self.alpha2[i]);
            y = self.convs2[i].forward(&y);
            for (a, b) in cur.iter_mut().zip(&y) {
                *a += *b;
            }
        }
        cur
    }
}

fn snake(x: &mut [f32], channels: usize, alpha: &[f32]) {
    let time = x.len() / channels;
    for c in 0..channels {
        let a = alpha[c];
        let inv = 1.0 / (a + SNAKE_EPS);
        for v in &mut x[c * time..(c + 1) * time] {
            let s = (*v * a).sin();
            *v += inv * s * s;
        }
    }
}

fn elu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

fn reflect_left(x: &[f32], channels: usize) -> Vec<f32> {
    let time = x.len() / channels;
    let mut y = vec![0.0; channels * (time + 1)];
    for c in 0..channels {
        let src = &x[c * time..(c + 1) * time];
        let dst = &mut y[c * (time + 1)..(c + 1) * (time + 1)];
        dst[0] = src[1];
        dst[1..].copy_from_slice(src);
    }
    y
}

fn hann16() -> [f32; N_FFT] {
    let mut w = [0.0; N_FFT];
    for (i, s) in w.iter_mut().enumerate() {
        *s = 0.5 - 0.5 * (2.0 * PI * i as f32 / N_FFT as f32).cos();
    }
    w
}

fn stft_source(source: &[f32]) -> Vec<f32> {
    let pad = N_FFT / 2;
    let mut padded = vec![0.0; source.len() + N_FFT];
    for i in 0..pad {
        padded[i] = source[pad - i];
    }
    padded[pad..pad + source.len()].copy_from_slice(source);
    for i in 0..pad {
        padded[pad + source.len() + i] = source[source.len() - 2 - i];
    }
    let frames = (padded.len() - N_FFT) / ISTFT_HOP + 1;
    let window = hann16();
    let mut out = vec![0.0; (N_FFT + 2) * frames];
    let mut frame = [0.0f32; N_FFT];
    for f in 0..frames {
        for n in 0..N_FFT {
            frame[n] = padded[f * ISTFT_HOP + n] * window[n];
        }
        for k in 0..N_FREQ {
            let mut re = 0.0;
            let mut im = 0.0;
            for n in 0..N_FFT {
                let ang = -2.0 * PI * (n * k) as f32 / N_FFT as f32;
                let (s, c) = ang.sin_cos();
                re += frame[n] * c;
                im += frame[n] * s;
            }
            out[k * frames + f] = re;
            out[(N_FREQ + k) * frames + f] = im;
        }
    }
    out
}

fn istft(mag: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    let window = hann16();
    let buf_len = (frames - 1) * ISTFT_HOP + N_FFT;
    let mut acc = vec![0.0f32; buf_len];
    let mut env = vec![0.0f32; buf_len];
    for f in 0..frames {
        let mut re = [0.0f32; N_FREQ];
        let mut im = [0.0f32; N_FREQ];
        for k in 0..N_FREQ {
            let m = mag[k * frames + f].min(1e2);
            let p = phase[k * frames + f];
            re[k] = m * p.cos();
            im[k] = m * p.sin();
        }
        let frame = irfft16(&re, &im);
        for n in 0..N_FFT {
            let i = f * ISTFT_HOP + n;
            acc[i] += frame[n] * window[n];
            env[i] += window[n] * window[n];
        }
    }
    for i in 0..buf_len {
        if env[i] > 1e-11 {
            acc[i] /= env[i];
        }
    }
    acc[N_FFT / 2..buf_len - N_FFT / 2].to_vec()
}

fn irfft16(re: &[f32; N_FREQ], im: &[f32; N_FREQ]) -> [f32; N_FFT] {
    let mut fre = [0.0f32; N_FFT];
    let mut fim = [0.0f32; N_FFT];
    for k in 0..N_FREQ {
        fre[k] = re[k];
        fim[k] = im[k];
    }
    for k in 1..N_FFT / 2 {
        fre[N_FFT - k] = re[k];
        fim[N_FFT - k] = -im[k];
    }
    let mut y = [0.0f32; N_FFT];
    for n in 0..N_FFT {
        let mut sum = 0.0;
        for k in 0..N_FFT {
            let ang = 2.0 * PI * (n * k) as f32 / N_FFT as f32;
            let (s, c) = ang.sin_cos();
            sum += fre[k] * c - fim[k] * s;
        }
        y[n] = sum / N_FFT as f32;
    }
    y
}

fn named(st: &safetensors::SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = st.tensor(name).with_context(|| name.to_string())?;
    if view.shape() != shape {
        bail!("{name} shape {:?} != {shape:?}", view.shape());
    }
    f32s(view)
}

fn tensor_shape(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<usize>> {
    Ok(st
        .tensor(name)
        .with_context(|| name.to_string())?
        .shape()
        .to_vec())
}

fn f32s(view: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    if view.dtype() != safetensors::Dtype::F32 {
        bail!("expected f32");
    }
    let bytes = view.data();
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn fuse(g: &[f32], v: &[f32], cout_or_cin: usize) -> Vec<f32> {
    let rest = v.len() / cout_or_cin;
    let mut w = vec![0.0; v.len()];
    for c in 0..cout_or_cin {
        let mut n2 = 0.0;
        for i in 0..rest {
            let x = v[c * rest + i];
            n2 += x * x;
        }
        let scale = g[c] / n2.sqrt().max(1e-12);
        for i in 0..rest {
            w[c * rest + i] = v[c * rest + i] * scale;
        }
    }
    w
}

fn conv_wn(
    st: &safetensors::SafeTensors<'_>,
    prefix: &str,
    stride: usize,
    pad: usize,
    dilation: usize,
) -> Result<Conv1d> {
    let g = f32s(st.tensor(&format!("{prefix}.parametrizations.weight.original0"))?)?;
    let v = f32s(st.tensor(&format!("{prefix}.parametrizations.weight.original1"))?)?;
    let shape = st
        .tensor(&format!("{prefix}.parametrizations.weight.original1"))?
        .shape()
        .to_vec();
    let b = f32s(st.tensor(&format!("{prefix}.bias"))?)?;
    let (cout, cin, k) = (shape[0], shape[1], shape[2]);
    Ok(Conv1d {
        w: fuse(&g, &v, cout),
        b,
        cin,
        cout,
        k,
        stride,
        pad,
        dilation,
    })
}

fn conv_plain(
    st: &safetensors::SafeTensors<'_>,
    prefix: &str,
    stride: usize,
    pad: usize,
    dilation: usize,
) -> Result<Conv1d> {
    let shape = st.tensor(&format!("{prefix}.weight"))?.shape().to_vec();
    let (cout, cin, k) = (shape[0], shape[1], shape[2]);
    Ok(Conv1d {
        w: named(st, &format!("{prefix}.weight"), &[cout, cin, k])?,
        b: named(st, &format!("{prefix}.bias"), &[cout])?,
        cin,
        cout,
        k,
        stride,
        pad,
        dilation,
    })
}

fn conv_tr_wn(
    st: &safetensors::SafeTensors<'_>,
    prefix: &str,
    stride: usize,
    pad: usize,
) -> Result<ConvTr> {
    let g = f32s(st.tensor(&format!("{prefix}.parametrizations.weight.original0"))?)?;
    let v = f32s(st.tensor(&format!("{prefix}.parametrizations.weight.original1"))?)?;
    let shape = st
        .tensor(&format!("{prefix}.parametrizations.weight.original1"))?
        .shape()
        .to_vec();
    let (cin, cout, k) = (shape[0], shape[1], shape[2]);
    Ok(ConvTr {
        w: fuse(&g, &v, cin),
        b: f32s(st.tensor(&format!("{prefix}.bias"))?)?,
        cin,
        cout,
        k,
        stride,
        pad,
    })
}

fn resblock(
    st: &safetensors::SafeTensors<'_>,
    prefix: &str,
    kernel: usize,
    channels: usize,
) -> Result<ResBlock> {
    let mut convs1 = Vec::new();
    let mut convs2 = Vec::new();
    let mut alpha1 = Vec::new();
    let mut alpha2 = Vec::new();
    for i in 0..3 {
        let dil = DILATIONS[i];
        let pad = dil * (kernel - 1) / 2;
        let c1 = conv_wn(&st, &format!("{prefix}.convs1.{i}"), 1, pad, dil)?;
        if c1.k != kernel || c1.cout != channels {
            bail!(
                "{prefix}.convs1.{i} is {}x{} k={} not {channels} k={kernel}",
                c1.cout,
                c1.cin,
                c1.k
            );
        }
        convs1.push(c1);
        let pad2 = (kernel - 1) / 2;
        convs2.push(conv_wn(&st, &format!("{prefix}.convs2.{i}"), 1, pad2, 1)?);
        alpha1.push(named(
            st,
            &format!("{prefix}.activations1.{i}.alpha"),
            &[channels],
        )?);
        alpha2.push(named(
            st,
            &format!("{prefix}.activations2.{i}.alpha"),
            &[channels],
        )?);
    }
    Ok(ResBlock {
        convs1,
        convs2,
        alpha1,
        alpha2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> serde_json::Value {
        serde_json::from_str(include_str!("../testdata/hift_tiny.json")).unwrap()
    }

    #[test]
    fn snake_conv_transpose_and_istft_match_torch() {
        let g = tiny();
        let mut sx = g["snake_in"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        snake(&mut sx, 1, &[g["snake_alpha"].as_f64().unwrap() as f32]);
        for (a, b) in sx.iter().zip(g["snake_out"].as_array().unwrap()) {
            let b = b.as_f64().unwrap() as f32;
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let conv = ConvTr {
            w: g["conv_t_w"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect(),
            b: vec![0.0, 0.0],
            cin: 1,
            cout: 2,
            k: 3,
            stride: 2,
            pad: 1,
        };
        let x: Vec<f32> = g["conv_t_in"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let y = conv.forward(&x);
        let expect: Vec<f32> = g["conv_t_out"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(y.len(), expect.len());
        for (a, b) in y.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let mut mag = vec![0.0f32; 9 * 3];
        let mut phase = vec![0.0f32; 9 * 3];
        for f in 0..3 {
            mag[1 * 3 + f] = 0.5;
        }
        phase[1 * 3] = 0.3;
        phase[1 * 3 + 1] = -0.2;
        phase[1 * 3 + 2] = 0.7;
        let wav = istft(&mag, &phase, 3);
        let expect: Vec<f32> = g["istft_wav"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(wav.len(), expect.len());
        for (a, b) in wav.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-5, "istft {a} vs {b}");
        }
    }

    #[test]
    fn seeded_excitation_repeats_and_changes_with_f0() {
        let w = vec![0.1; 9];
        let a = sine_source(&[100.0, 120.0, 80.0], &w, 0.0, 7);
        let b = sine_source(&[100.0, 120.0, 80.0], &w, 0.0, 7);
        let c = sine_source(&[100.0, 120.0, 90.0], &w, 0.0, 7);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        assert_eq!(a.len(), 3);
    }

    /// Real `s3gen_meanflow.safetensors` against the LDC93S1 mel in testdata.
    /// The wav is a vocoder reconstruction, not a clone.
    #[test]
    #[ignore = "set PYRE_CHATTERBOX_S3 to s3gen_meanflow.safetensors and pass --ignored"]
    fn real_reference_mel_matches_torch_waveform() {
        let Some(path) = std::env::var_os("PYRE_CHATTERBOX_S3") else {
            eprintln!("PYRE_CHATTERBOX_S3 unset");
            return;
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
        let mel = f32_file(&root.join("hift_mel.f32"));
        let f0_ref = f32_file(&root.join("hift_f0.f32"));
        let source = f32_file(&root.join("hift_source.f32"));
        let wav_ref = f32_file(&root.join("hift_wav.f32"));
        let meta: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/hift_golden.json")).unwrap();
        assert!(meta["note"].as_str().unwrap().contains("Not a voice clone"));
        let model = Hift::load(Path::new(&path)).unwrap();
        let f0 = model.predict_f0(&mel).unwrap();
        let f0_err = max_abs(&f0, &f0_ref);
        assert!(f0_err < 1e-3, "f0 max abs {f0_err}");
        let wav = model.decode(&mel, &source).unwrap();
        assert_eq!(wav.len(), wav_ref.len());
        assert_eq!(wav.len(), meta["wav_len"].as_u64().unwrap() as usize);
        assert!(wav.iter().all(|v| v.is_finite()));
        let err = max_abs(&wav, &wav_ref);
        assert!(err < 1e-4, "waveform max abs {err}");
        let again = model.vocode(&mel, 11).unwrap();
        let again2 = model.vocode(&mel, 11).unwrap();
        assert_eq!(again, again2);
        assert_eq!(again.len(), mel.len() / MEL_BINS * MEL_HOP);
        let mut shifted = mel.clone();
        for v in &mut shifted {
            *v *= 1.15;
        }
        let other = model.vocode(&shifted, 11).unwrap();
        let moved = max_abs(&again, &other);
        assert!(
            moved > 1e-3,
            "a different mel did not move the waveform ({moved})"
        );
    }

    fn f32_file(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
}
