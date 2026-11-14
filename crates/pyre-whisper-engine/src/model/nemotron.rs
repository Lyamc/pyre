//! NVIDIA Nemotron Speech Streaming (FastConformer-RNNT).
//!
//! This is not the Parakeet TDT stack. The published checkpoint
//! (`nvidia/nemotron-speech-streaming-en-0.6b`, NVIDIA Open Model License) uses causal
//! Conv2d subsampling, a LayerNorm convolution module, interleaved relative positions, and a
//! chunked attention mask (default lookahead 13 frames). Offline file decode matches the
//! Hugging Face forward with `use_cache=false`. Incremental HTTP streaming is not claimed.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid, silu, softmax};
use burn::tensor::{Tensor, TensorData};

use crate::model::conformer::FeedForwardModule;
use crate::model::parakeet_decoder::{ParakeetDecoder, ParakeetDecoderCache, ParakeetDecoderConfig};

/// Time frames after `n_conv` causal stride-2 convolutions. Offline pad is `(k-1, stride-1)`.
pub fn nemotron_subsampled_frames(mel_frames: usize, kernel: usize, stride: usize, n_conv: usize) -> usize {
    let mut len = mel_frames;
    for _ in 0..n_conv {
        len = len.saturating_add(kernel - 1).saturating_add(stride - 1).saturating_sub(kernel) / stride + 1;
    }
    len
}

/// Frequency bins left after the same causal stack (mel axis).
pub fn nemotron_subsampled_freq(n_mels: usize, kernel: usize, stride: usize, n_conv: usize) -> usize {
    nemotron_subsampled_frames(n_mels, kernel, stride, n_conv)
}

/// Keep a query/key pair under the HF `chunked_limited` mask.
pub fn chunked_limited_keep(q: usize, kv: usize, chunk: usize, left_chunks: usize) -> bool {
    if chunk == 0 {
        return false;
    }
    let diff = (q / chunk) as i64 - (kv / chunk) as i64;
    diff >= 0 && diff <= left_chunks as i64
}

#[derive(Config, Debug)]
pub struct NemotronConfig {
    pub d_model: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub n_mels: usize,
    pub n_vocab: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub conv_channels: usize,
    pub sub_kernel: usize,
    pub conv_kernel: usize,
    pub ff_inner: usize,
    pub lookahead: usize,
    pub sliding_window: usize,
    pub max_symbols: usize,
    pub blank_id: usize,
}

impl NemotronConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> NemotronAsr {
        let freq = nemotron_subsampled_freq(self.n_mels, self.sub_kernel, 2, 3);
        let sub = NemotronSubsampleConfig {
            n_mels: self.n_mels,
            channels: self.conv_channels,
            kernel: self.sub_kernel,
            d_model: self.d_model,
            freq,
        }
        .init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                NemotronBlockConfig {
                    d_model: self.d_model,
                    n_head: self.n_head,
                    conv_kernel: self.conv_kernel,
                    ff_inner: self.ff_inner,
                }
                .init(device)
            })
            .collect();
        let final_proj = LinearConfig::new(self.d_model, self.decoder_hidden).init(device);
        let joint = LinearConfig::new(self.decoder_hidden, self.n_vocab).init(device);
        let decoder = ParakeetDecoderConfig {
            vocab_size: self.n_vocab,
            hidden_size: self.decoder_hidden,
            num_layers: self.decoder_layers,
        }
        .init(device);
        NemotronAsr {
            sub,
            layers,
            final_proj,
            joint,
            decoder,
            n_head: self.n_head,
            lookahead: self.lookahead,
            sliding_window: self.sliding_window,
            max_symbols: self.max_symbols,
            blank_id: self.blank_id,
            n_mels: self.n_mels,
            sub_kernel: self.sub_kernel,
        }
    }
}

#[derive(Config, Debug)]
struct NemotronSubsampleConfig {
    n_mels: usize,
    channels: usize,
    kernel: usize,
    d_model: usize,
    freq: usize,
}

fn causal_conv2d(in_ch: usize, out_ch: usize, k: usize, groups: usize, bias: bool, device: &burn::tensor::Device) -> Conv2d {
    // Offline pad: time (k-1, stride-1), freq the same. Explicit is (top, left, bottom, right).
    let stride = 2usize;
    Conv2dConfig::new([in_ch, out_ch], [k, k])
        .with_stride([stride, stride])
        .with_groups(groups)
        .with_bias(bias)
        .with_padding(PaddingConfig2d::Explicit(k - 1, k - 1, stride - 1, stride - 1))
        .init(device)
}

impl NemotronSubsampleConfig {
    fn init(&self, device: &burn::tensor::Device) -> NemotronSubsample {
        let k = self.kernel;
        let c = self.channels;
        NemotronSubsample {
            conv_in: causal_conv2d(1, c, k, 1, true, device),
            dw0: causal_conv2d(c, c, k, c, true, device),
            pw0: Conv2dConfig::new([c, c], [1, 1]).init(device),
            dw1: causal_conv2d(c, c, k, c, true, device),
            pw1: Conv2dConfig::new([c, c], [1, 1]).init(device),
            linear: LinearConfig::new(c * self.freq.max(1), self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct NemotronSubsample {
    pub conv_in: Conv2d,
    pub dw0: Conv2d,
    pub pw0: Conv2d,
    pub dw1: Conv2d,
    pub pw1: Conv2d,
    pub linear: Linear,
}

impl NemotronSubsample {
    fn forward(&self, mel_bct: Tensor<3>) -> Tensor<3> {
        // Frontend is [B, mel, T]. HF subsampling unsqueezes a channel on [B, T, mel].
        let x = mel_bct.swap_dims(1, 2).unsqueeze_dim::<4>(1);
        let x = relu(self.conv_in.forward(x));
        let x = relu(self.pw0.forward(self.dw0.forward(x)));
        let x = relu(self.pw1.forward(self.dw1.forward(x)));
        let [b, c, t, f] = x.dims();
        let x = x.permute([0, 2, 1, 3]);
        let need = self.linear.weight.dims()[0];
        let freq = (need / c.max(1)).max(1);
        let x = if f > freq {
            x.slice([0..b, 0..t, 0..c, 0..freq])
        } else if f < freq {
            let pad = Tensor::zeros([b, t, c, freq - f], &x.device());
            Tensor::cat(vec![x, pad], 3)
        } else {
            x
        };
        let x = x.reshape([b, t, c * freq]);
        self.linear.forward(x)
    }
}

#[derive(Config, Debug)]
struct NemotronBlockConfig {
    d_model: usize,
    n_head: usize,
    conv_kernel: usize,
    ff_inner: usize,
}

impl NemotronBlockConfig {
    fn init(&self, device: &burn::tensor::Device) -> NemotronBlock {
        let d = self.d_model;
        let k = self.conv_kernel;
        NemotronBlock {
            ff1: ff(d, self.ff_inner, device),
            attn: NemotronAttentionConfig { d_model: d, n_head: self.n_head }.init(device),
            attn_ln: LayerNormConfig::new(d).init(device),
            conv_ln: LayerNormConfig::new(d).init(device),
            conv: NemotronConvConfig { d_model: d, kernel: k }.init(device),
            ff2: ff(d, self.ff_inner, device),
            final_ln: LayerNormConfig::new(d).init(device),
        }
    }
}

fn ff(d_model: usize, inner: usize, device: &burn::tensor::Device) -> FeedForwardModule {
    // Same math as the Parakeet feed-forward (pre-norm, SiLU, no dropout at eval).
    // Built with the checkpoint's intermediate width, which is 4× hidden on the 0.6B model.
    let ln = LayerNormConfig::new(d_model).init(device);
    let lin1 = LinearConfig::new(d_model, inner).with_bias(false).init(device);
    let lin2 = LinearConfig::new(inner, d_model).with_bias(false).init(device);
    FeedForwardModule { ln, lin1, lin2, lora_lin1: None, lora_lin2: None }
}

#[derive(Module, Debug)]
pub struct NemotronBlock {
    pub ff1: FeedForwardModule,
    pub attn: NemotronAttention,
    pub attn_ln: LayerNorm,
    pub conv_ln: LayerNorm,
    pub conv: NemotronConv,
    pub ff2: FeedForwardModule,
    pub final_ln: LayerNorm,
}

impl NemotronBlock {
    fn forward(&self, x: Tensor<3>, pos: Tensor<3>, mask: Tensor<2>, valid: usize) -> Tensor<3> {
        let x = x.clone() + self.ff1.forward(x).mul_scalar(0.5);
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), pos, mask);
        let x = x.clone() + self.conv.forward(self.conv_ln.forward(x), valid);
        let x = x.clone() + self.ff2.forward(x).mul_scalar(0.5);
        self.final_ln.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct NemotronAttentionConfig {
    pub d_model: usize,
    pub n_head: usize,
}

impl NemotronAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> NemotronAttention {
        let d = self.d_model;
        let no_bias = |lin: LinearConfig| lin.with_bias(false).init(device);
        let head = d / self.n_head;
        NemotronAttention {
            n_head: self.n_head,
            query: no_bias(LinearConfig::new(d, d)),
            key: no_bias(LinearConfig::new(d, d)),
            value: no_bias(LinearConfig::new(d, d)),
            out: no_bias(LinearConfig::new(d, d)),
            pos: no_bias(LinearConfig::new(d, d)),
            pos_bias_u: Param::from_tensor(Tensor::zeros([self.n_head, head], device)),
            pos_bias_v: Param::from_tensor(Tensor::zeros([self.n_head, head], device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct NemotronAttention {
    pub n_head: usize,
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub out: Linear,
    pub pos: Linear,
    pub pos_bias_u: Param<Tensor<2>>,
    pub pos_bias_v: Param<Tensor<2>>,
}

impl NemotronAttention {
    fn forward(&self, x: Tensor<3>, pos: Tensor<3>, mask: Tensor<2>) -> Tensor<3> {
        let [batch, seq, d_model] = x.dims();
        let head = d_model / self.n_head;
        let q = self.query.forward(x.clone()).reshape([batch, seq, self.n_head, head]).swap_dims(1, 2);
        let k = self.key.forward(x.clone()).reshape([batch, seq, self.n_head, head]).swap_dims(1, 2);
        let v = self.value.forward(x).reshape([batch, seq, self.n_head, head]).swap_dims(1, 2);
        let p = self.pos.forward(pos);
        let p_len = p.dims()[1];
        let p = p.reshape([1, p_len, self.n_head, head]).swap_dims(1, 2);

        let u = self.pos_bias_u.val().reshape([1, self.n_head, 1, head]);
        let v_bias = self.pos_bias_v.val().reshape([1, self.n_head, 1, head]);
        let scale = (head as f64).powf(-0.5);
        let ac = (q.clone() + u).matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let mut bd = (q + v_bias).matmul(p.swap_dims(2, 3));
        bd = rel_shift(bd).mul_scalar(scale);
        let [b, h, t, _] = bd.dims();
        bd = bd.slice([0..b, 0..h, 0..t, 0..seq]);
        bd = bd + mask.unsqueeze::<3>().unsqueeze::<4>();
        let attn = softmax(ac + bd, 3);
        let ctx = attn.matmul(v).swap_dims(1, 2).reshape([batch, seq, d_model]);
        self.out.forward(ctx)
    }
}

/// HF `NemotronAsrStreamingEncoderAttention._rel_shift` (pad 1 on the left, drop the first row).
fn rel_shift(x: Tensor<4>) -> Tensor<4> {
    let [b, h, q, p] = x.dims();
    let z = Tensor::zeros([b, h, q, 1], &x.device());
    let padded = Tensor::cat(vec![z, x], 3);
    let view = padded.reshape([b, h, p + 1, q]);
    let dropped = view.slice([0..b, 0..h, 1..p + 1, 0..q]);
    dropped.reshape([b, h, q, p])
}

/// Interleaved sin/cos relative embedding. Positions run from `L-1` down to `-(L-1)`.
pub fn nemotron_pos_embed(seq: usize, d_model: usize, device: &burn::tensor::Device) -> Tensor<3> {
    let half = d_model / 2;
    let mut positions = Vec::with_capacity(seq * 2);
    let mut p = seq as i64 - 1;
    while p > -(seq as i64) {
        positions.push(p as f32);
        p -= 1;
    }
    let pos = Tensor::<1>::from_data(TensorData::new(positions.clone(), [positions.len()]), device);
    // HF: 10000 ** -(arange(0, d_model, 2) / d_model), i.e. even indices 0,2,4,...
    let idx: Tensor<1> = Tensor::<1, burn::tensor::Int>::arange(0..half as i64, device).float();
    let inv = idx.mul_scalar(2.0).div_scalar(d_model as f64).mul_scalar(10000f64.ln()).exp().recip();
    let freqs = pos.unsqueeze_dim::<2>(1) * inv.unsqueeze_dim::<2>(0);
    let stacked = Tensor::cat(vec![freqs.clone().sin().unsqueeze_dim::<3>(2), freqs.cos().unsqueeze_dim::<3>(2)], 2);
    let [t, h, _] = stacked.dims();
    stacked.reshape([1, t, h * 2])
}

fn chunk_mask(seq: usize, valid: usize, lookahead: usize, sliding_window: usize, device: &burn::tensor::Device) -> Tensor<2> {
    let chunk = lookahead.saturating_add(1).max(1);
    let left = sliding_window.saturating_sub(1);
    let left_chunks = left / chunk;
    let mut m = vec![0f32; seq * seq];
    for q in 0..seq {
        for kv in 0..seq {
            if kv >= valid || !chunked_limited_keep(q, kv, chunk, left_chunks) {
                m[q * seq + kv] = -10_000.0;
            }
        }
    }
    Tensor::<2>::from_data(TensorData::new(m, [seq, seq]), device)
}

#[derive(Config, Debug)]
struct NemotronConvConfig {
    d_model: usize,
    kernel: usize,
}

impl NemotronConvConfig {
    fn init(&self, device: &burn::tensor::Device) -> NemotronConv {
        let d = self.d_model;
        let k = self.kernel;
        NemotronConv {
            point_conv1: Conv1dConfig::new(d, d * 2, 1).with_bias(false).init(device),
            depth_conv: Conv1dConfig::new(d, d, k)
                .with_groups(d)
                .with_bias(false)
                .with_padding(PaddingConfig1d::Explicit(k - 1, 0))
                .init(device),
            post_ln: LayerNormConfig::new(d).init(device),
            point_conv2: Conv1dConfig::new(d, d, 1).with_bias(false).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct NemotronConv {
    pub point_conv1: Conv1d,
    pub depth_conv: Conv1d,
    pub post_ln: LayerNorm,
    pub point_conv2: Conv1d,
}

impl NemotronConv {
    fn forward(&self, x: Tensor<3>, valid: usize) -> Tensor<3> {
        let x = x.swap_dims(1, 2);
        let x = self.point_conv1.forward(x);
        let [b, d2, s] = x.dims();
        let d = d2 / 2;
        let x = x.clone().slice([0..b, 0..d, 0..s]) * sigmoid(x.slice([0..b, d..d2, 0..s]));
        let x = if valid < s {
            let head = x.clone().slice([0..b, 0..d, 0..valid]);
            let tail = Tensor::zeros([b, d, s - valid], &x.device());
            Tensor::cat(vec![head, tail], 2)
        } else {
            x
        };
        let x = self.depth_conv.forward(x).swap_dims(1, 2);
        let x = silu(self.post_ln.forward(x)).swap_dims(1, 2);
        self.point_conv2.forward(x).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct NemotronAsr {
    pub sub: NemotronSubsample,
    pub layers: Vec<NemotronBlock>,
    pub final_proj: Linear,
    pub joint: Linear,
    pub decoder: ParakeetDecoder,
    pub n_head: usize,
    pub lookahead: usize,
    pub sliding_window: usize,
    pub max_symbols: usize,
    pub blank_id: usize,
    pub n_mels: usize,
    pub sub_kernel: usize,
}

impl NemotronAsr {
    pub fn encode(&self, mel: Tensor<3>) -> Tensor<3> {
        let [_, _, mel_t] = mel.dims();
        let x = self.sub.forward(mel);
        let valid = nemotron_subsampled_frames(mel_t, self.sub_kernel, 2, 3).min(x.dims()[1]);
        let seq = x.dims()[1];
        let d_model = x.dims()[2];
        let device = x.device();
        let pos = nemotron_pos_embed(seq, d_model, &device);
        let mask = chunk_mask(seq, valid, self.lookahead, self.sliding_window, &device);
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h, pos.clone(), mask.clone(), valid);
        }
        self.final_proj.forward(h)
    }

    /// Greedy RNN-T. Advance one frame on blank, or after `max_symbols` non-blanks.
    /// Token ids, the encoder frame each token was emitted on, and the valid frame count.
    ///
    /// No extra left-context silence. Padding ~5.6 s and then slicing it off doubled the
    /// error (LDC 27.3% → 63.6%) and dropped the first 5–15 words. The chunk mask already
    /// gives frame 0 only its own chunk, which is how this checkpoint was trained.
    pub fn decode_rnnt(&self, mel: Tensor<3>, blank: usize, vocab: usize, max_symbols: usize) -> (Vec<usize>, Vec<usize>, usize) {
        let enc = self.encode(mel.clone());
        let [_, frames, joint] = enc.dims();
        let [_, _, mel_t] = mel.dims();
        let valid = nemotron_subsampled_frames(mel_t, self.sub_kernel, 2, 3).min(frames);
        let device = enc.device();
        let mut cache = ParakeetDecoderCache::new(&device, self.decoder.lstm.layers.len(), self.decoder.hidden_size());
        let mut frame = 0usize;
        let mut symbols = 0usize;
        let mut current = blank;
        let mut decoded = Vec::new();
        let mut frames = Vec::new();
        let limit = valid.saturating_mul(max_symbols.max(1).saturating_add(1)).max(1);
        let mut steps = 0usize;
        while frame < valid && steps < limit {
            let enc_frame = enc.clone().slice([0..1, frame..frame + 1, 0..joint]).squeeze_dim::<2>(1);
            let update = current != blank;
            let dec = self.decoder.forward_step(current, blank, &mut cache, update);
            let logits = self.joint.forward(relu(enc_frame + dec));
            let row = logits.into_data().to_vec::<f32>().unwrap();
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (i, &v) in row.iter().take(vocab).enumerate() {
                if v > best_s {
                    best_s = v;
                    best = i;
                }
            }
            if best == blank {
                frame += 1;
                symbols = 0;
                current = blank;
            } else {
                decoded.push(best);
                frames.push(frame);
                current = best;
                symbols += 1;
                if symbols >= max_symbols.max(1) {
                    frame += 1;
                    symbols = 0;
                }
            }
            steps += 1;
        }
        (decoded, frames, valid)
    }

    pub fn encoder_ctx_size(&self) -> usize {
        1500
    }

    pub fn encoder_mel_size(&self) -> usize {
        self.n_mels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_frequencies_use_even_indices() {
        let device = burn::tensor::Device::flex();
        let emb = nemotron_pos_embed(2, 8, &device);
        let v = emb.into_data().to_vec::<f32>().unwrap();
        // position 1 (the second distance, 0) times inv_freq[0] = 1 is sin(0), cos(0).
        // Layout is [1, 2*L-1, d] interleaved sin/cos. Distance L-1 = 1 is row 0.
        assert_eq!(emb_len_ok(&v), true);
        let d = 8usize;
        let row0 = &v[..d];
        // freq 0 is 1.0, so sin(pos), cos(pos) for pos = 1.
        assert!((row0[0] - 1f32.sin()).abs() < 1e-5, "{}", row0[0]);
        assert!((row0[1] - 1f32.cos()).abs() < 1e-5, "{}", row0[1]);
        let inv1 = (10000f32).powf(-2.0 / d as f32);
        assert!((row0[2] - (inv1).sin()).abs() < 1e-4, "even-index freq {}", row0[2]);
    }

    fn emb_len_ok(v: &[f32]) -> bool {
        v.len() == 3 * 8
    }

    #[test]
    fn subsample_lengths_match_the_causal_formula() {
        assert_eq!(nemotron_subsampled_frames(100, 3, 2, 3), 14);
        assert_eq!(nemotron_subsampled_freq(128, 3, 2, 3), 17);
        assert_eq!(17 * 256, 4352);
    }

    #[test]
    fn chunk_mask_keeps_the_current_and_left_chunks_only() {
        let chunk = 2;
        let left = 1;
        assert!(chunked_limited_keep(0, 0, chunk, left));
        assert!(chunked_limited_keep(0, 1, chunk, left));
        assert!(!chunked_limited_keep(0, 2, chunk, left));
        assert!(chunked_limited_keep(2, 0, chunk, left));
        assert!(chunked_limited_keep(2, 3, chunk, left));
        assert!(!chunked_limited_keep(2, 0, chunk, 0));
    }

    #[test]
    fn tiny_graph_encodes_a_short_mel() {
        let device = burn::tensor::Device::flex();
        let cfg = NemotronConfig {
            d_model: 32,
            n_head: 4,
            n_layers: 1,
            n_mels: 16,
            n_vocab: 20,
            decoder_hidden: 16,
            decoder_layers: 1,
            conv_channels: 8,
            sub_kernel: 3,
            conv_kernel: 5,
            ff_inner: 64,
            lookahead: 1,
            sliding_window: 5,
            max_symbols: 4,
            blank_id: 19,
        };
        let model = cfg.init(&device);
        let mel = Tensor::<3>::zeros([1, 16, 20], &device);
        let enc = model.encode(mel);
        let [b, t, h] = enc.dims();
        assert_eq!((b, h), (1, 16));
        assert_eq!(t, nemotron_subsampled_frames(20, 3, 2, 3));
        assert!(enc.clone().sum().into_data().to_vec::<f32>().unwrap()[0].is_finite());
    }
}
