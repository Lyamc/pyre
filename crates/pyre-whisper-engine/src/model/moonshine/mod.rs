//! Moonshine encoder-decoder ASR (UsefulSensors/moonshine-tiny).
//!
//! Raw 16 kHz waveform → conv frontend → RoPE transformer encoder/decoder → greedy tokens.

pub mod weights;

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, silu, softmax, tanh};
use burn::tensor::{Int, Tensor};

use crate::model::attention::attn_decoder_mask;
use crate::token::Gpt2Tokenizer;

use self::weights::MoonshineRuntimeConfig;

#[derive(Config, Debug)]
pub struct MoonshineASRConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub pad_head_dim_to_multiple_of: usize,
    pub decoder_start_token_id: usize,
    pub eos_token_id: usize,
    pub max_new_tokens: usize,
}

impl MoonshineASRConfig {
    pub fn from_runtime(runtime: &MoonshineRuntimeConfig) -> Self {
        Self {
            hidden_size: runtime.hidden_size,
            intermediate_size: runtime.intermediate_size,
            encoder_layers: runtime.encoder_layers,
            decoder_layers: runtime.decoder_layers,
            n_heads: runtime.n_heads,
            n_kv_heads: runtime.n_kv_heads,
            vocab_size: runtime.vocab_size,
            rope_theta: runtime.rope_theta,
            partial_rotary_factor: runtime.partial_rotary_factor,
            pad_head_dim_to_multiple_of: runtime.pad_head_dim_to_multiple_of,
            decoder_start_token_id: runtime.decoder_start_token_id,
            eos_token_id: runtime.eos_token_id,
            max_new_tokens: runtime.max_new_tokens,
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> MoonshineASR {
        let head_dim = self.hidden_size / self.n_heads.max(1);
        let rotary_dim = ((head_dim as f64) * self.partial_rotary_factor).round() as usize;
        let head_pad = if self.pad_head_dim_to_multiple_of > 0 {
            let m = self.pad_head_dim_to_multiple_of;
            (m - (head_dim % m)) % m
        } else {
            0
        };
        MoonshineASR {
            hidden_size: self.hidden_size,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads.max(1),
            head_dim,
            rotary_dim: rotary_dim.max(2),
            head_pad,
            rope_theta: self.rope_theta,
            vocab_size: self.vocab_size,
            decoder_start_token_id: self.decoder_start_token_id,
            eos_token_id: self.eos_token_id,
            max_new_tokens: self.max_new_tokens.max(1),
            conv1: Conv1dConfig::new(1, self.hidden_size, 127)
                .with_stride(64)
                .with_bias(false)
                .init(device),
            conv2: Conv1dConfig::new(self.hidden_size, 2 * self.hidden_size, 7)
                .with_stride(3)
                .init(device),
            conv3: Conv1dConfig::new(2 * self.hidden_size, self.hidden_size, 3)
                .with_stride(2)
                .init(device),
            groupnorm_weight: Param::from_tensor(Tensor::ones([self.hidden_size], device)),
            groupnorm_bias: Param::from_tensor(Tensor::zeros([self.hidden_size], device)),
            encoder_layers: (0..self.encoder_layers)
                .map(|_| MoonshineEncoderLayer::new(self, device))
                .collect(),
            encoder_norm: ln(self.hidden_size, device),
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            decoder_layers: (0..self.decoder_layers)
                .map(|_| MoonshineDecoderLayer::new(self, device))
                .collect(),
            decoder_norm: ln(self.hidden_size, device),
            proj_out: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

fn ln(n: usize, device: &burn::tensor::Device) -> LayerNorm {
    LayerNormConfig::new(n).init(device)
}

#[derive(Module, Debug)]
pub struct MoonshineASR {
    #[module(skip)]
    hidden_size: usize,
    #[module(skip)]
    n_heads: usize,
    #[module(skip)]
    n_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rotary_dim: usize,
    #[module(skip)]
    head_pad: usize,
    #[module(skip)]
    rope_theta: f64,
    #[module(skip)]
    vocab_size: usize,
    #[module(skip)]
    decoder_start_token_id: usize,
    #[module(skip)]
    eos_token_id: usize,
    #[module(skip)]
    max_new_tokens: usize,
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub conv3: Conv1d,
    pub groupnorm_weight: Param<Tensor<1>>,
    pub groupnorm_bias: Param<Tensor<1>>,
    pub encoder_layers: Vec<MoonshineEncoderLayer>,
    pub encoder_norm: LayerNorm,
    pub embed_tokens: Embedding,
    pub decoder_layers: Vec<MoonshineDecoderLayer>,
    pub decoder_norm: LayerNorm,
    pub proj_out: Linear,
}

impl MoonshineASR {
    pub fn device(&self) -> burn::tensor::Device {
        self.proj_out.weight.device()
    }

    pub fn encoder_ctx_size(&self) -> usize {
        480_000
    }

    pub fn encoder_mel_size(&self) -> usize {
        1
    }

    pub fn encode(&self, waveform: Tensor<2>) -> Tensor<3> {
        let [b, n] = waveform.dims();
        let x = waveform.reshape([b, 1, n]);
        let x = tanh(self.conv1.forward(x));
        let x = group_norm_1(x, &self.groupnorm_weight.val(), &self.groupnorm_bias.val());
        let x = gelu(self.conv2.forward(x));
        let x = gelu(self.conv3.forward(x));
        let x = x.swap_dims(1, 2);
        let seq = x.dims()[1];
        let (cos, sin) = rope_embeddings(seq, self.rotary_dim, self.rope_theta, &x.device());
        let mut hidden = x;
        for layer in &self.encoder_layers {
            hidden = layer.forward(hidden, &cos, &sin, self);
        }
        self.encoder_norm.forward(hidden)
    }

    pub fn generate_greedy(&self, encoder_hidden: Tensor<3>, verbose: bool) -> Vec<usize> {
        let device = encoder_hidden.device();
        let mut tokens = vec![self.decoder_start_token_id];
        for step in 0..self.max_new_tokens {
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let input = Tensor::<1, Int>::from_ints(ids.as_slice(), &device).reshape([1, tokens.len()]);
            let logits = self.decode_step(input, encoder_hidden.clone());
            let [_, seq, vocab] = logits.dims();
            let last = logits.slice([0..1, seq - 1..seq, 0..vocab]).reshape([vocab]);
            let next = argmax_i32(&last) as usize;
            if verbose && step < 8 {
                println!("DEBUG Moonshine step {step}: token={next}");
            }
            if next == self.eos_token_id {
                break;
            }
            tokens.push(next);
        }
        tokens
    }

    fn decode_step(&self, input_ids: Tensor<2, Int>, encoder_hidden: Tensor<3>) -> Tensor<3> {
        let hidden = self.embed_tokens.forward(input_ids);
        let seq = hidden.dims()[1];
        let device = hidden.device();
        let (cos, sin) = rope_embeddings(seq, self.rotary_dim, self.rope_theta, &device);
        let mask = attn_decoder_mask(seq, &device);
        let mut x = hidden;
        for layer in &self.decoder_layers {
            x = layer.forward(x, encoder_hidden.clone(), &cos, &sin, &mask, self);
        }
        self.proj_out.forward(self.decoder_norm.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct MoonshineEncoderLayer {
    pub input_layernorm: LayerNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub post_attention_layernorm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MoonshineEncoderLayer {
    fn new(cfg: &MoonshineASRConfig, device: &burn::tensor::Device) -> Self {
        let h = cfg.hidden_size;
        let q = cfg.n_heads * (h / cfg.n_heads);
        Self {
            input_layernorm: ln(h, device),
            q_proj: LinearConfig::new(h, q).with_bias(false).init(device),
            k_proj: LinearConfig::new(h, q).with_bias(false).init(device),
            v_proj: LinearConfig::new(h, q).with_bias(false).init(device),
            o_proj: LinearConfig::new(q, h).with_bias(false).init(device),
            post_attention_layernorm: ln(h, device),
            fc1: LinearConfig::new(h, cfg.intermediate_size).init(device),
            fc2: LinearConfig::new(cfg.intermediate_size, h).init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        model: &MoonshineASR,
    ) -> Tensor<3> {
        let attn = moonshine_self_attn(
            self.input_layernorm.forward(x.clone()),
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            Some((cos, sin)),
            None,
            None,
            model,
        );
        let x = x + attn;
        let mlp = self.fc2.forward(gelu(self.fc1.forward(self.post_attention_layernorm.forward(x.clone()))));
        x + mlp
    }
}

#[derive(Module, Debug)]
pub struct MoonshineDecoderLayer {
    pub input_layernorm: LayerNorm,
    pub self_q: Linear,
    pub self_k: Linear,
    pub self_v: Linear,
    pub self_o: Linear,
    pub post_attention_layernorm: LayerNorm,
    pub cross_q: Linear,
    pub cross_k: Linear,
    pub cross_v: Linear,
    pub cross_o: Linear,
    pub final_layernorm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MoonshineDecoderLayer {
    fn new(cfg: &MoonshineASRConfig, device: &burn::tensor::Device) -> Self {
        let h = cfg.hidden_size;
        let q = cfg.n_heads * (h / cfg.n_heads);
        Self {
            input_layernorm: ln(h, device),
            self_q: LinearConfig::new(h, q).with_bias(false).init(device),
            self_k: LinearConfig::new(h, q).with_bias(false).init(device),
            self_v: LinearConfig::new(h, q).with_bias(false).init(device),
            self_o: LinearConfig::new(q, h).with_bias(false).init(device),
            post_attention_layernorm: ln(h, device),
            cross_q: LinearConfig::new(h, q).with_bias(false).init(device),
            cross_k: LinearConfig::new(h, q).with_bias(false).init(device),
            cross_v: LinearConfig::new(h, q).with_bias(false).init(device),
            cross_o: LinearConfig::new(q, h).with_bias(false).init(device),
            final_layernorm: ln(h, device),
            fc1: LinearConfig::new(h, cfg.intermediate_size * 2).init(device),
            fc2: LinearConfig::new(cfg.intermediate_size, h).init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        encoder_hidden: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        mask: &Tensor<2>,
        model: &MoonshineASR,
    ) -> Tensor<3> {
        let attn = moonshine_self_attn(
            self.input_layernorm.forward(x.clone()),
            &self.self_q,
            &self.self_k,
            &self.self_v,
            &self.self_o,
            Some((cos, sin)),
            Some(mask),
            None,
            model,
        );
        let x = x + attn;
        let cross = moonshine_self_attn(
            self.post_attention_layernorm.forward(x.clone()),
            &self.cross_q,
            &self.cross_k,
            &self.cross_v,
            &self.cross_o,
            None,
            None,
            Some(encoder_hidden),
            model,
        );
        let x = x + cross;
        let h = self.final_layernorm.forward(x.clone());
        let gated = self.fc1.forward(h);
        let [b, s, d] = gated.dims();
        let half = d / 2;
        let a = gated.clone().slice([0..b, 0..s, 0..half]);
        let gate = gated.slice([0..b, 0..s, half..d]);
        let mlp = self.fc2.forward(a * silu(gate));
        x + mlp
    }
}

fn moonshine_self_attn(
    x: Tensor<3>,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    o_proj: &Linear,
    rope: Option<(&Tensor<3>, &Tensor<3>)>,
    mask: Option<&Tensor<2>>,
    kv_states: Option<Tensor<3>>,
    model: &MoonshineASR,
) -> Tensor<3> {
    let [b, q_len, _] = x.dims();
    let n_heads = model.n_heads;
    let head_dim = model.head_dim;
    let kv_src = kv_states.unwrap_or_else(|| x.clone());
    let kv_len = kv_src.dims()[1];

    let q = reshape_heads(q_proj.forward(x), b, q_len, n_heads, head_dim);
    let k = reshape_heads(k_proj.forward(kv_src.clone()), b, kv_len, n_heads, head_dim);
    let v = reshape_heads(v_proj.forward(kv_src), b, kv_len, n_heads, head_dim);

    let (q, k) = if let Some((cos, sin)) = rope {
        (apply_moonshine_rope(q, cos, sin, model.rotary_dim), apply_moonshine_rope(k, cos, sin, model.rotary_dim))
    } else {
        (q, k)
    };

    let q = pad_head(q, model.head_pad);
    let k = pad_head(k, model.head_pad);
    let v = pad_head(v, model.head_pad);
    let scale = (head_dim as f64).sqrt().recip();
    let mut scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
    if let Some(mask) = mask {
        scores = scores + mask.clone().unsqueeze::<3>().unsqueeze::<4>();
    }
    let weights = softmax(scores, 3);
    let mut ctx = weights.matmul(v).swap_dims(1, 2);
    if model.head_pad > 0 {
        let [bb, ss, hh, dd] = ctx.dims();
        ctx = ctx.slice([0..bb, 0..ss, 0..hh, 0..dd - model.head_pad]);
    }
    let [bb, ss, hh, dd] = ctx.dims();
    o_proj.forward(ctx.reshape([bb, ss, hh * dd]))
}

fn reshape_heads(x: Tensor<3>, b: usize, seq: usize, n_heads: usize, head_dim: usize) -> Tensor<4> {
    x.reshape([b, seq, n_heads, head_dim]).swap_dims(1, 2)
}

fn pad_head(x: Tensor<4>, pad: usize) -> Tensor<4> {
    if pad == 0 {
        return x;
    }
    let [b, h, s, _] = x.dims();
    let zeros = Tensor::zeros([b, h, s, pad], &x.device());
    Tensor::cat(vec![x, zeros], 3)
}

/// GroupNorm(num_groups=1) over [B, C, T].
fn group_norm_1(x: Tensor<3>, weight: &Tensor<1>, bias: &Tensor<1>) -> Tensor<3> {
    let [b, c, t] = x.dims();
    let flat = x.clone().reshape([b, c * t]);
    let mean = flat.clone().mean_dim(1).reshape([b, 1, 1]);
    let var = (flat.clone() - flat.mean_dim(1))
        .powf_scalar(2.0)
        .mean_dim(1)
        .reshape([b, 1, 1]);
    let normed = (x - mean) / (var.add_scalar(1e-5)).sqrt();
    let w = weight.clone().reshape([1, c, 1]);
    let bias = bias.clone().reshape([1, c, 1]);
    normed * w + bias
}

fn rope_embeddings(
    seq: usize,
    rotary_dim: usize,
    theta: f64,
    device: &burn::tensor::Device,
) -> (Tensor<3>, Tensor<3>) {
    let half = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            let exp = (2.0 * i as f64) / rotary_dim as f64;
            (theta.powf(-exp)) as f32
        })
        .collect();
    let inv = Tensor::<1>::from_floats(inv_freq.as_slice(), device);
    let t = Tensor::<1, Int>::arange(0..seq as i64, device).float();
    let freqs = t.unsqueeze::<2>().transpose().matmul(inv.unsqueeze::<2>());
    let cos = freqs.clone().cos();
    let sin = freqs.sin();
    let cos = interleave_pair(cos);
    let sin = interleave_pair(sin);
    (cos.unsqueeze::<3>(), sin.unsqueeze::<3>())
}

fn interleave_pair(x: Tensor<2>) -> Tensor<2> {
    let [s, half] = x.dims();
    let x = x.reshape([s, half, 1]);
    Tensor::cat(vec![x.clone(), x], 2).reshape([s, half * 2])
}

fn apply_moonshine_rope(
    x: Tensor<4>,
    cos: &Tensor<3>,
    sin: &Tensor<3>,
    rotary_dim: usize,
) -> Tensor<4> {
    let [b, h, s, d] = x.dims();
    let rot = x.clone().slice([0..b, 0..h, 0..s, 0..rotary_dim]);
    let pass = if d > rotary_dim {
        Some(x.clone().slice([0..b, 0..h, 0..s, rotary_dim..d]))
    } else {
        None
    };
    let rotated = rotate_interleaved(rot);
    let cos = cos.clone().unsqueeze_dim::<4>(1);
    let sin = sin.clone().unsqueeze_dim::<4>(1);
    let [cb, _, cs, cd] = cos.dims();
    let cos = cos.slice([0..cb, 0..1, 0..cs, 0..cd.min(rotary_dim)]);
    let sin = sin.slice([0..1, 0..1, 0..s, 0..rotary_dim]);
    let embedded = rot_mul(rotated, &x, &cos, &sin, b, h, s, rotary_dim);
    match pass {
        Some(p) => Tensor::cat(vec![embedded, p], 3),
        None => embedded,
    }
}

fn rot_mul(
    rotated: Tensor<4>,
    orig: &Tensor<4>,
    cos: &Tensor<4>,
    sin: &Tensor<4>,
    b: usize,
    h: usize,
    s: usize,
    rotary_dim: usize,
) -> Tensor<4> {
    let x_rot = orig.clone().slice([0..b, 0..h, 0..s, 0..rotary_dim]);
    x_rot * cos.clone() + rotated * sin.clone()
}

fn rotate_interleaved(x: Tensor<4>) -> Tensor<4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x = x.reshape([b, h, s, half, 2]);
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half, 0..1]).reshape([b, h, s, half]);
    let x2 = x.slice([0..b, 0..h, 0..s, 0..half, 1..2]).reshape([b, h, s, half]);
    let stacked = Tensor::cat(
        vec![
            x2.mul_scalar(-1.0).reshape([b, h, s, half, 1]),
            x1.reshape([b, h, s, half, 1]),
        ],
        4,
    );
    stacked.reshape([b, h, s, d])
}

fn argmax_i32(x: &Tensor<1>) -> i32 {
    let idx = x.clone().argmax(0);
    if let Ok(v) = idx.clone().into_data().to_vec::<i32>() {
        return v[0];
    }
    idx.into_data().to_vec::<i64>().unwrap()[0] as i32
}

pub fn decode_moonshine_tokens(bpe: &Gpt2Tokenizer, tokens: &[usize], eos: usize, bos: usize) -> String {
    let filtered: Vec<usize> = tokens
        .iter()
        .copied()
        .filter(|&t| t != eos && t != bos)
        .collect();
    bpe.decode(&filtered, true).unwrap_or_default()
}
