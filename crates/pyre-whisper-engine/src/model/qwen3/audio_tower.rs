use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::{activation::gelu, Tensor};

use super::mel::cnn_output_length;

#[derive(Config, Debug)]
pub struct Qwen3AudioTowerConfig {
    pub d_model: usize,
    pub output_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_mels: usize,
    pub ffn_dim: usize,
    pub downsample_hidden_size: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub conv_chunksize: usize,
    pub max_source_positions: usize,
}

impl Qwen3AudioTowerConfig {
    pub fn mel_after_conv(&self) -> usize {
        let m = self.n_mels;
        (((m + 1) / 2 + 1) / 2 + 1) / 2
    }

    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3AudioTower {
        let ds = self.downsample_hidden_size;
        let conv_cfg = |in_ch, out_ch| {
            Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        };
        let layers = (0..self.n_layers)
            .map(|_| Qwen3AudioEncoderLayerConfig::new(self.d_model, self.n_heads, self.ffn_dim).init(device))
            .collect();
        Qwen3AudioTower {
            n_mels: self.n_mels,
            output_dim: self.output_dim,
            n_window: self.n_window,
            n_window_infer: self.n_window_infer,
            conv_chunksize: self.conv_chunksize,
            conv2d1: conv_cfg(1, ds).init(device),
            conv2d2: conv_cfg(ds, ds).init(device),
            conv2d3: conv_cfg(ds, ds).init(device),
            conv_out: LinearConfig::new(ds * self.mel_after_conv(), self.d_model)
                .with_bias(false)
                .init(device),
            layers,
            ln_post: LayerNormConfig::new(self.d_model).init(device),
            gelu: Gelu::new(),
            proj1: LinearConfig::new(self.d_model, self.d_model).init(device),
            proj2: LinearConfig::new(self.d_model, self.output_dim).init(device),
            max_source_positions: self.max_source_positions,
            d_model: self.d_model,
        }
    }
}

#[derive(Debug)]
struct Qwen3AudioEncoderLayerConfig {
    d_model: usize,
    n_heads: usize,
    ffn_dim: usize,
}

impl Qwen3AudioEncoderLayerConfig {
    fn new(d_model: usize, n_heads: usize, ffn_dim: usize) -> Self {
        Self { d_model, n_heads, ffn_dim }
    }

    fn init(&self, device: &burn::tensor::Device) -> Qwen3AudioEncoderLayer {
        Qwen3AudioEncoderLayer {
            self_attn_layer_norm: LayerNormConfig::new(self.d_model).init(device),
            self_attn: Qwen3AudioAttentionConfig::new(self.d_model, self.n_heads).init(device),
            final_layer_norm: LayerNormConfig::new(self.d_model).init(device),
            fc1: LinearConfig::new(self.d_model, self.ffn_dim).init(device),
            fc2: LinearConfig::new(self.ffn_dim, self.d_model).init(device),
            gelu: Gelu::new(),
        }
    }
}

#[derive(Debug)]
struct Qwen3AudioAttentionConfig {
    d_model: usize,
    n_heads: usize,
}

impl Qwen3AudioAttentionConfig {
    fn new(d_model: usize, n_heads: usize) -> Self {
        Self { d_model, n_heads }
    }

    fn init(&self, device: &burn::tensor::Device) -> Qwen3AudioAttention {
        Qwen3AudioAttention {
            n_heads: self.n_heads,
            head_dim: self.d_model / self.n_heads,
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen3AudioAttention {
    #[module(skip)]
    n_heads: usize,
    #[module(skip)]
    head_dim: usize,
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) out_proj: Linear,
}

impl Qwen3AudioAttention {
    fn forward(&self, x: Tensor<2>, mask: &Tensor<4>) -> Tensor<2> {
        let [seq, d_model] = x.dims();
        let scale = (self.head_dim as f64).powf(-0.5);

        let q = self.q_proj.forward(x.clone()).reshape([seq, self.n_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone()).reshape([seq, self.n_heads, self.head_dim]);
        let v = self.v_proj.forward(x).reshape([seq, self.n_heads, self.head_dim]);

        let q = q.swap_dims(0, 1).unsqueeze::<4>();
        let k = k.swap_dims(0, 1).unsqueeze::<4>().swap_dims(2, 3);
        let v = v.swap_dims(0, 1).unsqueeze::<4>();

        let mut scores = q.matmul(k).mul_scalar(scale) + mask.clone();
        scores = burn::tensor::activation::softmax(scores, 3);
        // HF: attn_output.transpose(1, 2) on [1, heads, seq, head_dim]
        let out = scores.matmul(v).swap_dims(1, 2).reshape([seq, d_model]);
        self.out_proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen3AudioEncoderLayer {
    pub(crate) self_attn_layer_norm: LayerNorm,
    pub(crate) self_attn: Qwen3AudioAttention,
    pub(crate) final_layer_norm: LayerNorm,
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
    gelu: Gelu,
}

impl Qwen3AudioEncoderLayer {
    fn forward(&self, x: Tensor<2>, mask: &Tensor<4>) -> Tensor<2> {
        let residual = x.clone();
        let x = self.self_attn.forward(self.self_attn_layer_norm.forward(x), mask);
        let x = residual + x;
        let residual = x.clone();
        let x = self.final_layer_norm.forward(x);
        let x = self.fc2.forward(self.gelu.forward(self.fc1.forward(x)));
        residual + x
    }
}

#[derive(Module, Debug)]
pub struct Qwen3AudioTower {
    #[module(skip)]
    pub n_mels: usize,
    #[module(skip)]
    pub output_dim: usize,
    #[module(skip)]
    n_window: usize,
    #[module(skip)]
    n_window_infer: usize,
    #[module(skip)]
    conv_chunksize: usize,
    #[module(skip)]
    max_source_positions: usize,
    #[module(skip)]
    d_model: usize,
    pub(crate) conv2d1: Conv2d,
    pub(crate) conv2d2: Conv2d,
    pub(crate) conv2d3: Conv2d,
    pub(crate) conv_out: Linear,
    pub(crate) layers: Vec<Qwen3AudioEncoderLayer>,
    pub(crate) ln_post: LayerNorm,
    gelu: Gelu,
    pub(crate) proj1: Linear,
    pub(crate) proj2: Linear,
}

impl Qwen3AudioTower {
    pub fn forward(&self, mel: Tensor<3>) -> Tensor<3> {
        let [batch, n_mels, frames] = mel.dims();
        debug_assert_eq!(batch, 1, "Qwen3 audio tower supports batch=1");
        debug_assert_eq!(n_mels, self.n_mels);

        if frames == 0 {
            return Tensor::zeros([1, 0, self.output_dim], &mel.device());
        }

        let device = mel.device();
        let mel_data = mel.into_data().to_vec::<f32>().unwrap();
        let hidden = self.encode_frames(&mel_data, n_mels, frames, &device);
        let n_tokens = hidden.len() / self.output_dim;
        Tensor::<1>::from_floats(hidden.as_slice(), &device).reshape([1, n_tokens, self.output_dim])
    }

    fn encode_frames(
        &self,
        mel: &[f32],
        n_mels: usize,
        frames: usize,
        device: &burn::tensor::Device,
    ) -> Vec<f32> {
        let (flat, valid_count) = self.build_conv_sequence(mel, n_mels, frames, device);
        let hidden = Tensor::<1>::from_floats(flat.as_slice(), device).reshape([valid_count, self.d_model]);
        // HF SDPA/eager audio encoder attends over the full flattened sequence (mask=None),
        // not cu_seqlens blocks — see modeling_qwen3_asr.py Qwen3ASRAudioEncoder.forward.
        let attn_mask = Tensor::<4>::zeros([1, 1, valid_count, valid_count], device);

        let mut x = hidden;
        for layer in &self.layers {
            x = layer.forward(x, &attn_mask);
        }

        let x = self.ln_post.forward(x);
        let x = self.proj2.forward(self.gelu.forward(self.proj1.forward(x)));
        x.into_data().to_vec::<f32>().unwrap()
    }

    fn build_conv_sequence(
        &self,
        mel: &[f32],
        n_mels: usize,
        frames: usize,
        device: &burn::tensor::Device,
    ) -> (Vec<f32>, usize) {
        let chunk_size = self.n_window * 2;
        let chunk_num = (frames + chunk_size - 1) / chunk_size;
        let mut chunk_lengths = vec![chunk_size; chunk_num];
        let rem = frames % chunk_size;
        if rem != 0 {
            *chunk_lengths.last_mut().unwrap() = rem;
        } else if !chunk_lengths.is_empty() {
            *chunk_lengths.last_mut().unwrap() = chunk_size;
        }

        let mut chunks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(chunk_num);
        let mut offset = 0usize;
        for &len in &chunk_lengths {
            let mut chunk = vec![vec![0.0f32; len]; n_mels];
            for m in 0..n_mels {
                for t in 0..len {
                    chunk[m][t] = mel[m * frames + offset + t];
                }
            }
            chunks.push(chunk);
            offset += len;
        }

        let max_chunk = *chunk_lengths.iter().max().unwrap_or(&0);
        let cnn_lens: Vec<usize> = chunk_lengths.iter().map(|&l| cnn_output_length(l)).collect();
        let max_cnn = *cnn_lens.iter().max().unwrap_or(&0);
        let ds = self.conv2d1.weight.dims()[0];
        let mel_after = self.conv_out.weight.dims()[0] / ds;

        let mut padded = vec![0.0f32; chunks.len() * max_cnn * self.d_model];
        let mut mask = vec![false; chunks.len() * max_cnn];

        for (ci, chunk) in chunks.iter().enumerate() {
            let t = chunk_lengths[ci];
            let input = self.chunk_to_conv_tensor(chunk, n_mels, t, max_chunk, device);
            let embed = self.run_conv_stack(input, mel_after, device);
            let [_, seq, d] = embed.dims();
            let data = embed.into_data().to_vec::<f32>().unwrap();
            let pe = sinusoidal_pe(seq, self.d_model, self.max_source_positions, device);
            let pe_data = pe.into_data().to_vec::<f32>().unwrap();

            for s in 0..seq {
                for d_idx in 0..d {
                    let base = (ci * max_cnn + s) * self.d_model + d_idx;
                    padded[base] = data[s * d + d_idx] + pe_data[s * d + d_idx];
                }
                if s < cnn_lens[ci] {
                    mask[ci * max_cnn + s] = true;
                }
            }
        }

        let valid_count = mask.iter().filter(|&&m| m).count();
        let mut flat = vec![0.0f32; valid_count * self.d_model];
        let mut vi = 0usize;
        for i in 0..chunks.len() * max_cnn {
            if mask[i] {
                let src = i * self.d_model;
                let dst = vi * self.d_model;
                flat[dst..dst + self.d_model].copy_from_slice(&padded[src..src + self.d_model]);
                vi += 1;
            }
        }

        (flat, valid_count)
    }

    fn chunk_to_conv_tensor(
        &self,
        chunk: &[Vec<f32>],
        n_mels: usize,
        t: usize,
        max_t: usize,
        device: &burn::tensor::Device,
    ) -> Tensor<4> {
        let mut data = vec![0.0f32; n_mels * max_t];
        for m in 0..n_mels {
            for ti in 0..t {
                data[m * max_t + ti] = chunk[m][ti];
            }
        }
        Tensor::<1>::from_floats(data.as_slice(), device)
            .reshape([1, 1, n_mels, max_t])
    }

    fn run_conv_stack(&self, input: Tensor<4>, _mel_after: usize, _device: &burn::tensor::Device) -> Tensor<3> {
        let mut parts = Vec::new();
        let [n, _, _, _] = input.dims();
        let step = self.conv_chunksize.max(1);
        for start in (0..n).step_by(step) {
            let end = (start + step).min(n);
            let chunk = input.clone().slice([start..end, 0..1, 0..input.dims()[2], 0..input.dims()[3]]);
            let x = gelu(self.conv2d1.forward(chunk));
            let x = gelu(self.conv2d2.forward(x));
            let x = gelu(self.conv2d3.forward(x));
            parts.push(x);
        }
        let x = Tensor::cat(parts, 0);
        let [b, c, f, t] = x.dims();
        // HF: padded_embed.permute(0, 3, 1, 2).view(b, t, c * f)
        let flat = x.swap_dims(1, 3).swap_dims(2, 3).reshape([b, t, c * f]);
        self.conv_out.forward(flat)
    }
}

fn sinusoidal_pe(seq: usize, dim: usize, max_len: usize, device: &burn::tensor::Device) -> Tensor<2> {
    let len = seq.min(max_len);
    let half = dim / 2;
    let log_step = (10000.0f64).ln() / (half as f64 - 1.0);
    let mut data = vec![0.0f32; seq * dim];
    for pos in 0..len {
        for i in 0..half {
            let freq = (pos as f64) * (-log_step * i as f64).exp();
            data[pos * dim + i] = freq.sin() as f32;
            data[pos * dim + half + i] = freq.cos() as f32;
        }
    }
    Tensor::<1>::from_floats(data.as_slice(), device).reshape([seq, dim])
}