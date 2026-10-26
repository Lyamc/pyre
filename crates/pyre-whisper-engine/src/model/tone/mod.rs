//! T-one (t-tech) streaming Conformer CTC — offline greedy path.
//! 8 kHz, 64-bin log-mel, character CTC (Russian alphabet + space + blank).

pub mod weights;

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::{Tensor};

use crate::audio::mel::get_mel_filters;
use crate::model::conformer::RMSNorm;

use self::weights::ToneRuntimeConfig;

const LOG_ZERO_GUARD: f32 = 1.0 / 16_777_216.0;
const PREEMPH: f32 = 0.97;
const TONE_SR: usize = 8_000;
const N_FFT: usize = 160;
const HOP: usize = 80;
const N_MELS: usize = 64;

#[derive(Config, Debug)]
pub struct TONEConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_mels: usize,
    pub n_vocab: usize,
    pub conv_kernel: usize,
    pub ff_mult: usize,
    pub rope_dim: usize,
    pub chunk_size: usize,
    pub reduction_position: usize,
    pub upsample_position: usize,
    pub reduction_factor: usize,
    pub reduction_kernel: usize,
    pub mhsa_left: usize,
    pub mhsa_stateless: usize,
}

impl TONEConfig {
    pub fn from_runtime(rt: &ToneRuntimeConfig) -> Self {
        Self {
            d_model: rt.d_model,
            n_heads: rt.n_heads,
            n_layers: rt.n_layers,
            n_mels: rt.n_mels,
            n_vocab: rt.n_vocab,
            conv_kernel: rt.conv_kernel,
            ff_mult: rt.ff_mult,
            rope_dim: rt.rope_dim,
            chunk_size: rt.chunk_size,
            reduction_position: rt.reduction_position,
            upsample_position: rt.upsample_position,
            reduction_factor: rt.reduction_factor,
            reduction_kernel: rt.reduction_kernel,
            mhsa_left: rt.mhsa_left,
            mhsa_stateless: rt.mhsa_stateless,
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> TONE {
        let recompute = default_recompute(self.n_layers);
        let feat_hidden = pre_encode_feat_out(self.n_mels);
        TONE {
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_mels: self.n_mels,
            n_vocab: self.n_vocab,
            rope_dim: self.rope_dim,
            chunk_size: self.chunk_size,
            reduction_position: self.reduction_position,
            upsample_position: self.upsample_position,
            mhsa_left: self.mhsa_left,
            mhsa_stateless: self.mhsa_stateless.min(self.n_layers),
            pre_norm: RMSNorm::with_eps(self.n_mels, 1e-8, device),
            conv0: Conv2dConfig::new([1, 32], [11, 21]).init(device),
            bn0: BatchNormConfig::new(32).init(device),
            conv1: Conv2dConfig::new([32, 64], [11, 11])
                .with_stride([3, 1])
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            pre_out: LinearConfig::new(64 * feat_hidden, self.d_model)
                .with_bias(false)
                .init(device),
            out_norm: RMSNorm::with_eps(self.d_model, 1e-8, device),
            layers: (0..self.n_layers)
                .map(|i| ToneLayer::new(self, recompute[i.min(recompute.len() - 1)], device))
                .collect(),
            red_conv: Conv1dConfig::new(self.d_model, self.d_model * 4, self.reduction_kernel)
                .with_stride(self.reduction_factor)
                .with_groups(self.d_model)
                .init(device),
            red_pw: Conv1dConfig::new(self.d_model * 4, self.d_model, 1).init(device),
            ctc: Conv1dConfig::new(self.d_model, self.n_vocab, 1).init(device),
        }
    }
}

fn default_recompute(n: usize) -> Vec<bool> {
    let mut v = vec![false; n];
    for i in [0usize, 7, 14, 15] {
        if i < n {
            v[i] = true;
        }
    }
    v
}

fn pre_encode_feat_out(n_mels: usize) -> usize {
    let mut f = n_mels as i64;
    for &(k, s) in &[(21i64, 1i64), (11, 1)] {
        f = (f - k) / s + 1;
    }
    f.max(1) as usize
}

#[derive(Module, Debug)]
pub struct TONE {
    #[module(skip)]
    d_model: usize,
    #[module(skip)]
    n_heads: usize,
    #[module(skip)]
    n_mels: usize,
    #[module(skip)]
    n_vocab: usize,
    #[module(skip)]
    rope_dim: usize,
    #[module(skip)]
    chunk_size: usize,
    #[module(skip)]
    reduction_position: usize,
    #[module(skip)]
    upsample_position: usize,
    #[module(skip)]
    mhsa_left: usize,
    #[module(skip)]
    mhsa_stateless: usize,
    pub pre_norm: RMSNorm,
    pub conv0: Conv2d,
    pub bn0: BatchNorm,
    pub conv1: Conv2d,
    pub bn1: BatchNorm,
    pub pre_out: Linear,
    pub out_norm: RMSNorm,
    pub layers: Vec<ToneLayer>,
    pub red_conv: Conv1d,
    pub red_pw: Conv1d,
    pub ctc: Conv1d,
}

impl TONE {
    pub fn encoder_ctx_size(&self) -> usize {
        24_000
    }

    pub fn encoder_mel_size(&self) -> usize {
        self.n_mels
    }

    pub fn device(&self) -> burn::tensor::Device {
        self.ctc.weight.device()
    }

    pub fn encode_mel(&self, mel: Tensor<3>) -> Tensor<3> {
        let x = mel.swap_dims(1, 2);
        let x = tone_rms(&self.pre_norm, x);
        let [b, t, f] = x.dims();
        let mut x = x.reshape([b, 1, t, f]);
        x = pad_time(x, 10);
        x = silu(self.bn0.forward(self.conv0.forward(x)));
        x = pad_time(x, 8);
        x = silu(self.bn1.forward(self.conv1.forward(x)));
        let [b, c, t2, f2] = x.dims();
        let x = x.swap_dims(1, 2).reshape([b, t2, c * f2]);
        let mut x = tone_rms(&self.out_norm, self.pre_out.forward(x));

        let mut residual_for_up: Option<Tensor<3>> = None;
        let mut att_scores: Option<Tensor<4>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (chunk, left) = if i <= self.reduction_position || i > self.upsample_position {
                let left = if i < self.mhsa_stateless {
                    0
                } else {
                    self.mhsa_left
                };
                (self.chunk_size, left)
            } else {
                (self.chunk_size / 2, self.mhsa_left / 2)
            };
            let (next, scores) = layer.forward(
                x,
                chunk.max(1),
                left,
                self.n_heads,
                self.rope_dim,
                att_scores,
            );
            x = next;
            att_scores = scores;
            if i == self.reduction_position {
                residual_for_up = Some(x.clone());
                x = self.temporal_reduce(x);
            }
            if i == self.upsample_position {
                if let Some(res) = residual_for_up.take() {
                    x = upsample_add(x, res, 2);
                }
            }
        }
        x
    }

    fn temporal_reduce(&self, x: Tensor<3>) -> Tensor<3> {
        let [_b, t, _d] = x.dims();
        let x = x.swap_dims(1, 2);
        let right = if t % 2 == 0 { 0 } else { 1 };
        let x = pad_conv1d(x, 1, right);
        let x = self.red_pw.forward(self.red_conv.forward(x));
        x.swap_dims(1, 2)
    }

    pub fn logits(&self, mel: Tensor<3>) -> Tensor<3> {
        let enc = self.encode_mel(mel);
        let x = enc.swap_dims(1, 2);
        self.ctc.forward(x).swap_dims(1, 2)
    }
}

fn tone_rms<const D: usize>(norm: &RMSNorm, x: Tensor<D>) -> Tensor<D> {
    let dim = x.dims()[D - 1] as f64;
    let l2 = x.clone().powf_scalar(2.0).sum_dim(D - 1).sqrt();
    let rms = l2.div_scalar(dim.sqrt());
    (x / rms.add_scalar(norm.epsilon)) * norm.gamma.val().unsqueeze()
}

fn pad_time(x: Tensor<4>, left: usize) -> Tensor<4> {
    if left == 0 {
        return x;
    }
    let [b, c, _t, f] = x.dims();
    let z = Tensor::zeros([b, c, left, f], &x.device());
    Tensor::cat(vec![z, x], 2)
}

fn pad_conv1d(x: Tensor<3>, left: usize, right: usize) -> Tensor<3> {
    let [b, c, _t] = x.dims();
    let device = x.device();
    let mut parts = Vec::new();
    if left > 0 {
        parts.push(Tensor::zeros([b, c, left], &device));
    }
    parts.push(x);
    if right > 0 {
        parts.push(Tensor::zeros([b, c, right], &device));
    }
    if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Tensor::cat(parts, 2)
    }
}

fn upsample_add(x: Tensor<3>, residual: Tensor<3>, factor: usize) -> Tensor<3> {
    let [b, t, d] = x.dims();
    let target = residual.dims()[1];
    let up = x
        .reshape([b, t, 1, d])
        .repeat(&[1, 1, factor, 1])
        .reshape([b, t * factor, d]);
    let up_t = up.dims()[1].min(target);
    up.slice([0..b, 0..up_t, 0..d]) + residual.slice([0..b, 0..up_t, 0..d])
}

#[derive(Module, Debug)]
pub struct ToneLayer {
    #[module(skip)]
    recompute: bool,
    pub norm_ff1: RMSNorm,
    pub ff1_g: Linear,
    pub ff1_v: Linear,
    pub ff1_o: Linear,
    pub norm_att: RMSNorm,
    pub q: Option<Linear>,
    pub k: Option<Linear>,
    pub q_ln: Option<burn::nn::LayerNorm>,
    pub k_ln: Option<burn::nn::LayerNorm>,
    pub v: Linear,
    pub o: Linear,
    pub norm_conv: RMSNorm,
    pub pw1: Conv1d,
    pub dw: Conv1d,
    pub bn: BatchNorm,
    pub pw2: Conv1d,
    pub norm_ff2: RMSNorm,
    pub ff2_g: Linear,
    pub ff2_v: Linear,
    pub ff2_o: Linear,
    pub norm_out: RMSNorm,
}

impl ToneLayer {
    fn new(cfg: &TONEConfig, recompute: bool, device: &burn::tensor::Device) -> Self {
        let d = cfg.d_model;
        let ff = d * cfg.ff_mult;
        let head = d / cfg.n_heads.max(1);
        let qk = if recompute {
            (
                Some(LinearConfig::new(d, d).init(device)),
                Some(LinearConfig::new(d, d).init(device)),
                Some(burn::nn::LayerNormConfig::new(head).init(device)),
                Some(burn::nn::LayerNormConfig::new(head).init(device)),
            )
        } else {
            (None, None, None, None)
        };
        Self {
            recompute,
            norm_ff1: RMSNorm::with_eps(d, 1e-8, device),
            ff1_g: LinearConfig::new(d, ff).init(device),
            ff1_v: LinearConfig::new(d, ff).init(device),
            ff1_o: LinearConfig::new(ff, d).init(device),
            norm_att: RMSNorm::with_eps(d, 1e-8, device),
            q: qk.0,
            k: qk.1,
            q_ln: qk.2,
            k_ln: qk.3,
            v: LinearConfig::new(d, d).init(device),
            o: LinearConfig::new(d, d).init(device),
            norm_conv: RMSNorm::with_eps(d, 1e-8, device),
            pw1: Conv1dConfig::new(d, d * 2, 1).init(device),
            dw: Conv1dConfig::new(d, d, cfg.conv_kernel)
                .with_groups(d)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            bn: BatchNormConfig::new(d).init(device),
            pw2: Conv1dConfig::new(d, d, 1).init(device),
            norm_ff2: RMSNorm::with_eps(d, 1e-8, device),
            ff2_g: LinearConfig::new(d, ff).init(device),
            ff2_v: LinearConfig::new(d, ff).init(device),
            ff2_o: LinearConfig::new(ff, d).init(device),
            norm_out: RMSNorm::with_eps(d, 1e-8, device),
        }
    }

    fn ff(g: &Linear, v: &Linear, o: &Linear, x: Tensor<3>) -> Tensor<3> {
        o.forward(silu(g.forward(x.clone())) * v.forward(x))
    }

    fn forward(
        &self,
        x: Tensor<3>,
        chunk: usize,
        left: usize,
        n_heads: usize,
        rope_dim: usize,
        prev_scores: Option<Tensor<4>>,
    ) -> (Tensor<3>, Option<Tensor<4>>) {
        let mut residual = x.clone();
        residual = residual
            + Self::ff(&self.ff1_g, &self.ff1_v, &self.ff1_o, tone_rms(&self.norm_ff1, x))
                .mul_scalar(0.5);

        let attn_in = tone_rms(&self.norm_att, residual.clone());
        let (attn, scores) = self.attention(attn_in, chunk, left, n_heads, rope_dim, prev_scores);
        residual = residual + attn;

        let conv_in = tone_rms(&self.norm_conv, residual.clone());
        residual = residual + self.conv(conv_in);

        residual = residual.clone()
            + Self::ff(
                &self.ff2_g,
                &self.ff2_v,
                &self.ff2_o,
                tone_rms(&self.norm_ff2, residual),
            )
            .mul_scalar(0.5);
        (tone_rms(&self.norm_out, residual), scores)
    }

    fn conv(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.swap_dims(1, 2);
        let x = self.pw1.forward(x);
        let [b, c2, t] = x.dims();
        let half = c2 / 2;
        let a = x.clone().slice([0..b, 0..half, 0..t]);
        let g = x.slice([0..b, half..c2, 0..t]);
        let x = a * sigmoid(g);
        let x = pad_conv1d(x, 30, 0);
        let x = self.dw.forward(x);
        let x = silu(self.bn.forward(x));
        let x = self.pw2.forward(x);
        x.swap_dims(1, 2)
    }

    fn attention(
        &self,
        x: Tensor<3>,
        chunk: usize,
        left: usize,
        n_heads: usize,
        rope_dim: usize,
        prev_scores: Option<Tensor<4>>,
    ) -> (Tensor<3>, Option<Tensor<4>>) {
        let [b, t, d] = x.dims();
        let h = n_heads.max(1);
        let dh = d / h;
        let v = reshape_heads(self.v.forward(x.clone()), b, t, h, dh);
        let (scores, keep) = if let (Some(q), Some(k), Some(qln), Some(kln)) =
            (&self.q, &self.k, &self.q_ln, &self.k_ln)
        {
            let q = apply_head_ln(reshape_heads(q.forward(x.clone()), b, t, h, dh), qln);
            let k = apply_head_ln(reshape_heads(k.forward(x.clone()), b, t, h, dh), kln);
            let q = apply_rope_prefix(q, rope_dim);
            let k = apply_rope_prefix(k, rope_dim);
            let scale = (dh as f64).sqrt().recip();
            let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
            (scores.clone(), Some(scores))
        } else if let Some(prev) = prev_scores {
            (prev.clone(), Some(prev))
        } else {
            let zeros = Tensor::zeros([b, h, t, t], &x.device());
            (zeros.clone(), Some(zeros))
        };
        let masked = scores + chunk_mask(t, chunk.max(1), left, &x.device());
        let ctx = softmax(masked, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, t, d]);
        (self.o.forward(ctx), keep)
    }
}

fn reshape_heads(x: Tensor<3>, b: usize, t: usize, h: usize, dh: usize) -> Tensor<4> {
    x.reshape([b, t, h, dh]).swap_dims(1, 2)
}

fn apply_head_ln(x: Tensor<4>, ln: &burn::nn::LayerNorm) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    ln.forward(x.reshape([b * h * t, d])).reshape([b, h, t, d])
}

fn apply_rope_prefix(x: Tensor<4>, rope_dim: usize) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    if rope_dim == 0 || rope_dim > d {
        return x;
    }
    let rot = x.clone().slice([0..b, 0..h, 0..t, 0..rope_dim]);
    let pass = x.slice([0..b, 0..h, 0..t, rope_dim..d]);
    let half = rope_dim / 2;
    let x1 = rot.clone().slice([0..b, 0..h, 0..t, 0..half]);
    let x2 = rot.clone().slice([0..b, 0..h, 0..t, half..rope_dim]);
    let rotated = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);
    let (cos, sin) = rope_cos_sin(t, rope_dim, 10_000.0, &rot.device());
    let cos = cos.unsqueeze_dim::<4>(1);
    let sin = sin.unsqueeze_dim::<4>(1);
    Tensor::cat(vec![rot * cos + rotated * sin, pass], 3)
}

fn rope_cos_sin(
    seq: usize,
    dim: usize,
    base: f64,
    device: &burn::tensor::Device,
) -> (Tensor<3>, Tensor<3>) {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (base.powf(-(2.0 * i as f64) / dim as f64)) as f32)
        .collect();
    let inv = Tensor::<1>::from_floats(inv.as_slice(), device);
    let t = Tensor::<1, burn::tensor::Int>::arange(0..seq as i64, device).float();
    let freqs = t.unsqueeze::<2>().transpose().matmul(inv.unsqueeze::<2>());
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);
    (emb.clone().cos().unsqueeze::<3>(), emb.sin().unsqueeze::<3>())
}

fn chunk_mask(t: usize, chunk: usize, left: usize, device: &burn::tensor::Device) -> Tensor<4> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        let start = (i / chunk) * chunk;
        let lo = start.saturating_sub(left);
        let hi = (start + chunk).min(t);
        for j in 0..t {
            if j < lo || j >= hi {
                data[i * t + j] = -10_000.0;
            }
        }
    }
    Tensor::<1>::from_floats(data.as_slice(), device).reshape([1, 1, t, t])
}

fn dft_kernels() -> (Vec<f32>, Vec<f32>) {
    let n_freq = N_FFT / 2 + 1;
    let cols = 2 * n_freq;
    let win: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (N_FFT - 1) as f64).cos())
        .collect();
    let mut basis = vec![0f64; N_FFT * cols];
    for k in 0..n_freq {
        for n in 0..N_FFT {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / (N_FFT as f64);
            basis[n * cols + k] = angle.cos() * win[n];
            basis[n * cols + n_freq + k] = angle.sin() * win[n];
        }
    }
    let a = PREEMPH as f64;
    let mut out = vec![0f64; N_FFT * cols];
    for bin in 0..cols {
        out[bin] = (1.0 - a) * basis[bin] - a * basis[cols + bin];
        for n in 1..N_FFT - 1 {
            out[n * cols + bin] = basis[n * cols + bin] - a * basis[(n + 1) * cols + bin];
        }
        out[(N_FFT - 1) * cols + bin] = basis[(N_FFT - 1) * cols + bin];
    }
    let mut re = vec![0f32; n_freq * N_FFT];
    let mut im = vec![0f32; n_freq * N_FFT];
    for k in 0..n_freq {
        for n in 0..N_FFT {
            re[k * N_FFT + n] = out[n * cols + k] as f32;
            im[k * N_FFT + n] = out[n * cols + n_freq + k] as f32;
        }
    }
    (re, im)
}

pub fn tone_log_mel(waveform: &[f32], sample_rate: usize, device: &burn::tensor::Device) -> Tensor<3> {
    let wav = if sample_rate == TONE_SR {
        waveform.to_vec()
    } else {
        pyre_whisper_audio::resample_mono(waveform, sample_rate, TONE_SR).unwrap_or_else(|_| waveform.to_vec())
    };
    let mut padded = vec![0f32; N_FFT - HOP];
    padded.extend_from_slice(&wav);
    if padded.len() < N_FFT {
        padded.resize(N_FFT, 0.0);
    }
    let n_freq = N_FFT / 2 + 1;
    let n_frames = (padded.len() - N_FFT) / HOP + 1;
    let (re_k, im_k) = dft_kernels();
    let mut spec = vec![0f32; n_frames * n_freq];
    for f in 0..n_frames {
        let start = f * HOP;
        let frame = &padded[start..start + N_FFT];
        for k in 0..n_freq {
            let mut re = 0f32;
            let mut im = 0f32;
            let off = k * N_FFT;
            for n in 0..N_FFT {
                re += frame[n] * re_k[off + n];
                im += frame[n] * im_k[off + n];
            }
            spec[f * n_freq + k] = re * re + im * im;
        }
    }
    let spec = Tensor::<1>::from_floats(spec.as_slice(), device).reshape([1, n_frames, n_freq]);
    let filters = get_mel_filters(TONE_SR as f64, N_FFT, N_MELS, false, device, true);
    let mels = spec.matmul(filters.transpose().unsqueeze());
    let mels = mels.swap_dims(1, 2);
    (mels + LOG_ZERO_GUARD).log()
}

pub fn greedy_ctc_text(logits: Tensor<3>, vocab: &[String], blank: usize) -> (String, Vec<usize>) {
    let data = logits.argmax(2).into_data();
    let ids = data
        .clone()
        .to_vec::<i32>()
        .or_else(|_| {
            data.to_vec::<i64>()
                .map(|v| v.into_iter().map(|x| x as i32).collect())
        })
        .unwrap_or_default();
    let mut out = String::new();
    let mut kept = Vec::new();
    let mut prev = None;
    for id in ids {
        let t = id as usize;
        if prev == Some(t) {
            continue;
        }
        prev = Some(t);
        if t != blank {
            kept.push(t);
            if let Some(ch) = vocab.get(t) {
                out.push_str(ch);
            }
        }
    }
    (out, kept)
}
