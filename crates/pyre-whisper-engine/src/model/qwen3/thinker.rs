use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::{activation::silu, Tensor, Int};

use crate::model::attention::attn_decoder_mask;
use crate::model::conformer::RMSNorm;

#[derive(Config, Debug)]
pub struct Qwen3ThinkerConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
}

impl Qwen3ThinkerConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3Thinker {
        let layers = (0..self.n_layers)
            .map(|_| {
                Qwen3ThinkerLayerConfig::new(
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    self.intermediate_size,
                    self.rms_norm_eps,
                )
                .init(device)
            })
            .collect();
        Qwen3Thinker {
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            mrope_section: self.mrope_section.clone(),
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RMSNorm::with_eps(self.hidden_size, self.rms_norm_eps, device),
            lm_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Debug)]
struct Qwen3ThinkerLayerConfig {
    hidden_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    rms_norm_eps: f64,
}

impl Qwen3ThinkerLayerConfig {
    fn new(
        hidden_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
    ) -> Self {
        Self {
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
        }
    }

    fn init(&self, device: &burn::tensor::Device) -> Qwen3ThinkerLayer {
        let q_out = self.n_heads * self.head_dim;
        let kv_out = self.n_kv_heads * self.head_dim;
        Qwen3ThinkerLayer {
            input_layernorm: RMSNorm::with_eps(self.hidden_size, self.rms_norm_eps, device),
            self_attn: Qwen3ThinkerAttention {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                q_proj: LinearConfig::new(self.hidden_size, q_out).with_bias(false).init(device),
                k_proj: LinearConfig::new(self.hidden_size, kv_out).with_bias(false).init(device),
                v_proj: LinearConfig::new(self.hidden_size, kv_out).with_bias(false).init(device),
                o_proj: LinearConfig::new(q_out, self.hidden_size).with_bias(false).init(device),
                q_norm: RMSNorm::with_eps(self.head_dim, self.rms_norm_eps, device),
                k_norm: RMSNorm::with_eps(self.head_dim, self.rms_norm_eps, device),
            },
            post_attention_layernorm: RMSNorm::with_eps(self.hidden_size, self.rms_norm_eps, device),
            mlp: Qwen3ThinkerMlp {
                gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
            },
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen3ThinkerAttention {
    #[module(skip)]
    n_heads: usize,
    #[module(skip)]
    n_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) q_norm: RMSNorm,
    pub(crate) k_norm: RMSNorm,
}

type KvLayerCache = (Tensor<4>, Tensor<4>);

impl Qwen3ThinkerAttention {
    fn forward_with_cache(
        &self,
        x: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        mask: &Tensor<2>,
        cache: Option<KvLayerCache>,
    ) -> (Tensor<3>, KvLayerCache) {
        let [batch, seq, _] = x.dims();
        let scale = (self.head_dim as f64).powf(-0.5);
        let n_rep = self.n_heads / self.n_kv_heads;

        let q = self
            .q_norm
            .forward(self.q_proj.forward(x.clone()).reshape([batch, seq, self.n_heads, self.head_dim]));
        let k = self
            .k_norm
            .forward(self.k_proj.forward(x.clone()).reshape([batch, seq, self.n_kv_heads, self.head_dim]));
        let v = self.v_proj.forward(x).reshape([batch, seq, self.n_kv_heads, self.head_dim]);

        let q = apply_rope(q.swap_dims(1, 2), cos, sin);
        let k_new = apply_rope(k.swap_dims(1, 2), cos, sin);
        let v_new = v.swap_dims(1, 2);

        let (k_full, v_full) = match cache {
            Some((k_cache, v_cache)) => (
                Tensor::cat(vec![k_cache, k_new], 2),
                Tensor::cat(vec![v_cache, v_new], 2),
            ),
            None => (k_new, v_new),
        };

        let k = repeat_kv(k_full.clone(), n_rep);
        let v = repeat_kv(v_full.clone(), n_rep);

        let mut scores =
            q.matmul(k.swap_dims(2, 3)).mul_scalar(scale) + mask.clone().unsqueeze::<3>().unsqueeze::<4>();
        scores = burn::tensor::activation::softmax(scores, 3);
        let out = scores.matmul(v).swap_dims(1, 2).reshape([batch, seq, self.n_heads * self.head_dim]);
        (self.o_proj.forward(out), (k_full, v_full))
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen3ThinkerMlp {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
}

impl Qwen3ThinkerMlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(x))
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen3ThinkerLayer {
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) self_attn: Qwen3ThinkerAttention,
    pub(crate) post_attention_layernorm: RMSNorm,
    pub(crate) mlp: Qwen3ThinkerMlp,
}

impl Qwen3ThinkerLayer {
    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        mask: &Tensor<2>,
    ) -> Tensor<3> {
        self.forward_with_cache(x, cos, sin, mask, None).0
    }

    fn forward_with_cache(
        &self,
        x: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        mask: &Tensor<2>,
        cache: Option<KvLayerCache>,
    ) -> (Tensor<3>, KvLayerCache) {
        let residual = x.clone();
        let (attn, cache) = self.self_attn.forward_with_cache(
            self.input_layernorm.forward(x),
            cos,
            sin,
            mask,
            cache,
        );
        let x = residual + attn;
        let residual = x.clone();
        let x = self.mlp.forward(self.post_attention_layernorm.forward(x));
        (residual + x, cache)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3Thinker {
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rope_theta: f64,
    #[module(skip)]
    mrope_section: Vec<usize>,
    pub embed_tokens: Embedding,
    pub(crate) layers: Vec<Qwen3ThinkerLayer>,
    pub(crate) norm: RMSNorm,
    pub lm_head: Linear,
}

impl Qwen3Thinker {
    pub fn forward_hidden(
        &self,
        inputs_embeds: Tensor<3>,
    ) -> Tensor<3> {
        let [_, seq, _] = inputs_embeds.dims();
        let device = inputs_embeds.device();
        let (cos, sin) = mrope_embeddings(
            seq,
            self.head_dim,
            self.rope_theta,
            &self.mrope_section,
            &device,
        );
        let mask = attn_decoder_mask(seq, &device);
        let mut x = inputs_embeds;
        for layer in &self.layers {
            x = layer.forward(x, &cos, &sin, &mask);
        }
        self.norm.forward(x)
    }

    pub fn forward_logits(
        &self,
        inputs_embeds: Tensor<3>,
    ) -> Tensor<3> {
        self.lm_head.forward(self.forward_hidden(inputs_embeds))
    }

    pub fn merge_audio_embeds(
        &self,
        input_ids: &[usize],
        audio_features: Tensor<3>,
        pad_token_id: usize,
        device: &burn::tensor::Device,
    ) -> Tensor<3> {
        let ids: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
        let ids_tensor = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, input_ids.len()]);
        let mut hidden = self.embed_tokens.forward(ids_tensor);
        let [_, _seq, h] = hidden.dims();
        let pad_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == pad_token_id)
            .map(|(i, _)| i)
            .collect();
        for (ai, &pos) in pad_positions.iter().enumerate() {
            let audio_slice = audio_features.clone().slice([0..1, ai..ai + 1, 0..h]);
            hidden = hidden.slice_assign([0..1, pos..pos + 1, 0..h], audio_slice);
        }
        hidden
    }

    pub fn generate_greedy(
        &self,
        prefix_ids: &[usize],
        audio_features: Tensor<3>,
        pad_token_id: usize,
        eos_ids: &[usize],
        max_new_tokens: usize,
        device: &burn::tensor::Device,
    ) -> Vec<usize> {
        let prefix_len = prefix_ids.len();
        if prefix_len == 0 {
            return Vec::new();
        }

        let hidden = self.merge_audio_embeds(prefix_ids, audio_features, pad_token_id, device);
        let (mut hidden, mut caches) = self.forward_hidden_with_cache(hidden, device, None);
        let mut generated = Vec::new();
        let mut pos = prefix_len;

        for _ in 0..max_new_tokens {
            let logits = self.lm_head.forward(hidden.clone());
            let [_, seq, vocab] = logits.dims();
            let last = logits.slice([0..1, seq - 1..seq, 0..vocab]);
            let next = argmax_id(last);
            if eos_ids.contains(&next) {
                break;
            }
            generated.push(next);

            let next_id = vec![next as i32];
            let ids_tensor = Tensor::<1, Int>::from_ints(next_id.as_slice(), device).reshape([1, 1]);
            let token_hidden = self.embed_tokens.forward(ids_tensor);
            let (cos, sin) = mrope_embeddings_at_pos(pos, self.head_dim, self.rope_theta, &self.mrope_section, device);
            let mask = Tensor::<2>::zeros([1, pos + 1], device);
            let mut x = token_hidden;
            let mut layer_caches = caches;
            for (layer, cache) in self.layers.iter().zip(layer_caches.iter_mut()) {
                let (out, new_cache) = layer.forward_with_cache(x, &cos, &sin, &mask, cache.take());
                *cache = Some(new_cache);
                x = out;
            }
            hidden = self.norm.forward(x);
            caches = layer_caches;
            pos += 1;
        }

        generated
    }

    /// Text-only greedy decode (no audio embeddings). Used by the offline summarizer.
    pub fn generate_greedy_text(
        &self,
        prefix_ids: &[usize],
        eos_ids: &[usize],
        max_new_tokens: usize,
        device: &burn::tensor::Device,
    ) -> Vec<usize> {
        if prefix_ids.is_empty() {
            return Vec::new();
        }
        let ids: Vec<i32> = prefix_ids.iter().map(|&id| id as i32).collect();
        let ids_tensor =
            Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, prefix_ids.len()]);
        let hidden = self.embed_tokens.forward(ids_tensor);
        let (mut hidden, mut caches) = self.forward_hidden_with_cache(hidden, device, None);
        let mut generated = Vec::new();
        let mut pos = prefix_ids.len();

        for _ in 0..max_new_tokens {
            let logits = self.lm_head.forward(hidden.clone());
            let [_, seq, vocab] = logits.dims();
            let last = logits.slice([0..1, seq - 1..seq, 0..vocab]);
            let next = argmax_id(last);
            if eos_ids.contains(&next) {
                break;
            }
            generated.push(next);

            let next_id = vec![next as i32];
            let ids_tensor =
                Tensor::<1, Int>::from_ints(next_id.as_slice(), device).reshape([1, 1]);
            let token_hidden = self.embed_tokens.forward(ids_tensor);
            let (cos, sin) =
                mrope_embeddings_at_pos(pos, self.head_dim, self.rope_theta, &self.mrope_section, device);
            let mask = Tensor::<2>::zeros([1, pos + 1], device);
            let mut x = token_hidden;
            let mut layer_caches = caches;
            for (layer, cache) in self.layers.iter().zip(layer_caches.iter_mut()) {
                let (out, new_cache) = layer.forward_with_cache(x, &cos, &sin, &mask, cache.take());
                *cache = Some(new_cache);
                x = out;
            }
            hidden = self.norm.forward(x);
            caches = layer_caches;
            pos += 1;
        }

        generated
    }

    fn forward_hidden_with_cache(
        &self,
        inputs_embeds: Tensor<3>,
        device: &burn::tensor::Device,
        caches: Option<Vec<Option<KvLayerCache>>>,
    ) -> (Tensor<3>, Vec<Option<KvLayerCache>>) {
        let [_, seq, _] = inputs_embeds.dims();
        let (cos, sin) = mrope_embeddings(seq, self.head_dim, self.rope_theta, &self.mrope_section, device);
        let mask = attn_decoder_mask(seq, device);
        let mut x = inputs_embeds;
        let mut out_caches = caches.unwrap_or_else(|| vec![None; self.layers.len()]);
        for (layer, cache) in self.layers.iter().zip(out_caches.iter_mut()) {
            let (out, new_cache) = layer.forward_with_cache(x, &cos, &sin, &mask, cache.take());
            *cache = Some(new_cache);
            x = out;
        }
        (self.norm.forward(x), out_caches)
    }
}

fn argmax_id(last: Tensor<3>) -> usize {
    let data = last.argmax(2).into_data();
    data.clone()
        .to_vec::<i64>()
        .or_else(|_| {
            data.to_vec::<i32>()
                .map(|v| v.into_iter().map(i64::from).collect())
        })
        .expect("argmax ids")[0] as usize
}

fn apply_rope(x: Tensor<4>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.clone().slice([0..b, 0..h, 0..s, half..d]);
    let neg_x2 = x2.mul_scalar(-1.0);
    let rotated = Tensor::cat(vec![neg_x2, x1], 3);
    // HF: cos/sin [batch, seq, head_dim] -> unsqueeze(1) -> [batch, 1, seq, head_dim]
    let cos = cos.clone().unsqueeze_dim::<4>(1);
    let sin = sin.clone().unsqueeze_dim::<4>(1);
    x * cos + rotated * sin
}

fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    // HF GQA: repeat each KV head `n_rep` times consecutively (not block-concat all heads).
    let [b, n_kv, s, d] = x.dims();
    let mut heads = Vec::with_capacity(n_kv * n_rep);
    for i in 0..n_kv {
        let head = x.clone().slice([0..b, i..i + 1, 0..s, 0..d]);
        for _ in 0..n_rep {
            heads.push(head.clone());
        }
    }
    Tensor::cat(heads, 1)
}

fn mrope_embeddings_at_pos(
    pos: usize,
    head_dim: usize,
    theta: f64,
    mrope_section: &[usize],
    device: &burn::tensor::Device,
) -> (Tensor<3>, Tensor<3>) {
    let (cos, sin) = mrope_embeddings(pos + 1, head_dim, theta, mrope_section, device);
    (
        cos.slice([0..1, pos..pos + 1, 0..head_dim]),
        sin.slice([0..1, pos..pos + 1, 0..head_dim]),
    )
}

fn mrope_embeddings(
    seq: usize,
    head_dim: usize,
    theta: f64,
    mrope_section: &[usize],
    device: &burn::tensor::Device,
) -> (Tensor<3>, Tensor<3>) {
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f64; half];
    // HF default RoPE: arange(0, head_dim, 2) / head_dim (not 0..half).
    for i in 0..half {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
    }

    let mut freqs_t = vec![vec![0.0f64; half]; seq];
    let mut freqs_h = freqs_t.clone();
    let mut freqs_w = freqs_t.clone();
    for pos in 0..seq {
        for i in 0..half {
            let f = pos as f64 * inv_freq[i];
            freqs_t[pos][i] = f;
            freqs_h[pos][i] = f;
            freqs_w[pos][i] = f;
        }
    }

    let section_h = mrope_section.get(1).copied().unwrap_or(20);
    let section_w = mrope_section.get(2).copied().unwrap_or(20);
    for pos in 0..seq {
        for dim in 1..=2 {
            let section = if dim == 1 { section_h } else { section_w };
            let length = section * 3;
            let src = if dim == 1 { &freqs_h } else { &freqs_w };
            let mut idx = dim;
            while idx < length {
                freqs_t[pos][idx] = src[pos][idx];
                idx += 3;
            }
        }
    }

    let mut cos_data = vec![0.0f32; seq * head_dim];
    let mut sin_data = vec![0.0f32; seq * head_dim];
    for pos in 0..seq {
        for i in 0..half {
            let f = freqs_t[pos][i];
            cos_data[pos * head_dim + i] = f.cos() as f32;
            cos_data[pos * head_dim + half + i] = f.cos() as f32;
            sin_data[pos * head_dim + i] = f.sin() as f32;
            sin_data[pos * head_dim + half + i] = f.sin() as f32;
        }
    }

    let cos = Tensor::<1>::from_floats(cos_data.as_slice(), device)
        .reshape([1, seq, head_dim]);
    let sin = Tensor::<1>::from_floats(sin_data.as_slice(), device)
        .reshape([1, seq, head_dim]);
    (cos, sin)
}

#[cfg(test)]
mod repeat_kv_tests {
    use super::repeat_kv;
        use burn::tensor::Tensor;

    #[test]
    fn interleaves_kv_heads_like_hf() {
        let device = burn::tensor::Device::default();
        // [B=1, n_kv=2, S=1, D=2]
        let x = Tensor::<1>::from_floats([1.0f32, 10.0, 2.0, 20.0], &device)
            .reshape([1, 2, 1, 2]);
        let y = repeat_kv(x, 2);
        let data = y.into_data().to_vec::<f32>().unwrap();
        // kv0, kv0, kv1, kv1 — not kv0, kv1, kv0, kv1
        assert_eq!(data, vec![1.0, 10.0, 1.0, 10.0, 2.0, 20.0, 2.0, 20.0]);
    }
}

#[cfg(test)]
mod mrope_tests {
    use super::mrope_embeddings;
    
    #[test]
    fn inv_freq_matches_hf_even_indices() {
        let device = burn::tensor::Device::default();
        let head_dim = 128;
        let theta = 1_000_000.0;
        let mrope_section = [24usize, 20, 20];
        let (cos, sin) = mrope_embeddings(2, head_dim, theta, &mrope_section, &device);
        let cos_data = cos.into_data().to_vec::<f32>().unwrap();
        let sin_data = sin.into_data().to_vec::<f32>().unwrap();

        let inv0 = 1.0f64;
        let inv3 = 1.0 / theta.powf(6.0 / head_dim as f64);
        let expected_cos0 = (0.0f64 * inv0).cos() as f32;
        let expected_cos1 = (1.0f64 * inv0).cos() as f32;
        let expected_sin1 = (1.0f64 * inv0).sin() as f32;

        assert!((cos_data[0] - expected_cos0).abs() < 1e-5);
        assert!((cos_data[head_dim] - expected_cos1).abs() < 1e-5, "pos1 dim0");
        assert!((sin_data[head_dim] - expected_sin1).abs() < 1e-5, "pos1 dim0 sin");
        // Temporal index 3 uses inv_freq[3] = theta^(-6/head_dim), not theta^(-3/head_dim).
        assert!((cos_data[3] - (0.0f64 * inv3).cos() as f32).abs() < 1e-5, "pos0 dim3");
        assert!(
            (cos_data[head_dim + 3] - (1.0f64 * inv3).cos() as f32).abs() < 1e-5,
            "pos1 dim3"
        );
    }
}