use burn::config::Config;
use burn::module::Module;
use burn::tensor::{activation::silu, Tensor, Int};

use crate::model::attention::attn_decoder_mask;
use crate::model::conformer::RMSNorm;

use super::host_embed::HostEmbedding;
use super::quant::QuantLinear;
use super::runtime::VibeVoiceRuntimeConfig;

type KvLayerCache = (Tensor<4>, Tensor<4>);

#[derive(Config, Debug)]
pub struct Qwen2DecoderConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Qwen2DecoderConfig {
    pub fn from_runtime(runtime: &VibeVoiceRuntimeConfig) -> Self {
        let n_heads = runtime.num_attention_heads.max(1);
        let head_dim = runtime.hidden_size / n_heads;
        Self {
            hidden_size: runtime.hidden_size,
            n_layers: runtime.num_hidden_layers,
            n_heads,
            n_kv_heads: runtime.num_key_value_heads.max(1),
            head_dim,
            intermediate_size: runtime.intermediate_size.max(runtime.hidden_size * 4),
            vocab_size: runtime.vocab_size,
            rms_norm_eps: runtime.rms_norm_eps,
            rope_theta: runtime.rope_theta,
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> Qwen2Decoder {
        let layers = (0..self.n_layers)
            .map(|_| {
                Qwen2DecoderLayer::new(
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    self.intermediate_size,
                    self.rms_norm_eps,
                    device,
                )
            })
            .collect();
        Qwen2Decoder {
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            embed_tokens: HostEmbedding::empty(self.vocab_size, self.hidden_size, device),
            layers,
            norm: RMSNorm::with_eps(self.hidden_size, self.rms_norm_eps, device),
            lm_head: QuantLinear::empty(self.hidden_size, self.vocab_size, false, device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen2Attention {
    #[module(skip)]
    pub(crate) n_heads: usize,
    #[module(skip)]
    pub(crate) n_kv_heads: usize,
    #[module(skip)]
    pub(crate) head_dim: usize,
    pub(crate) q_proj: QuantLinear,
    pub(crate) k_proj: QuantLinear,
    pub(crate) v_proj: QuantLinear,
    pub(crate) o_proj: QuantLinear,
}

impl Qwen2Attention {
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
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq, self.n_heads, self.head_dim]);
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, seq, self.n_kv_heads, self.head_dim]);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq, self.n_kv_heads, self.head_dim]);

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
        let out = scores
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq, self.n_heads * self.head_dim]);
        (self.o_proj.forward(out), (k_full, v_full))
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen2Mlp {
    pub(crate) gate_proj: QuantLinear,
    pub(crate) up_proj: QuantLinear,
    pub(crate) down_proj: QuantLinear,
}

impl Qwen2Mlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(x))
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen2DecoderLayer {
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) self_attn: Qwen2Attention,
    pub(crate) post_attention_layernorm: RMSNorm,
    pub(crate) mlp: Qwen2Mlp,
}

impl Qwen2DecoderLayer {
    fn new(
        hidden_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        device: &burn::tensor::Device,
    ) -> Self {
        let q_out = n_heads * head_dim;
        let kv_out = n_kv_heads * head_dim;
        Self {
            input_layernorm: RMSNorm::with_eps(hidden_size, rms_norm_eps, device),
            self_attn: Qwen2Attention {
                n_heads,
                n_kv_heads,
                head_dim,
                q_proj: QuantLinear::empty(hidden_size, q_out, true, device),
                k_proj: QuantLinear::empty(hidden_size, kv_out, true, device),
                v_proj: QuantLinear::empty(hidden_size, kv_out, true, device),
                o_proj: QuantLinear::empty(q_out, hidden_size, false, device),
            },
            post_attention_layernorm: RMSNorm::with_eps(hidden_size, rms_norm_eps, device),
            mlp: Qwen2Mlp {
                gate_proj: QuantLinear::empty(hidden_size, intermediate_size, false, device),
                up_proj: QuantLinear::empty(hidden_size, intermediate_size, false, device),
                down_proj: QuantLinear::empty(intermediate_size, hidden_size, false, device),
            },
        }
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
pub struct Qwen2Decoder {
    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub(crate) head_dim: usize,
    #[module(skip)]
    pub(crate) rope_theta: f64,
    pub embed_tokens: HostEmbedding,
    pub(crate) layers: Vec<Qwen2DecoderLayer>,
    pub(crate) norm: RMSNorm,
    pub lm_head: QuantLinear,
}

impl Qwen2Decoder {
    pub fn merge_audio_embeds(
        &self,
        input_ids: &[usize],
        audio_features: Tensor<3>,
        pad_token_id: usize,
        device: &burn::tensor::Device,
    ) -> Tensor<3> {
        let ids: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
        let ids_tensor =
            Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, input_ids.len()]);
        let mut hidden = self.embed_tokens.forward(ids_tensor);
        let [_, _, h] = hidden.dims();
        let pad_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == pad_token_id)
            .map(|(i, _)| i)
            .collect();
        let [_, audio_tokens, _] = audio_features.dims();
        for (ai, &pos) in pad_positions.iter().enumerate() {
            if ai >= audio_tokens {
                break;
            }
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
        banned_ids: &[usize],
        max_new_tokens: usize,
        device: &burn::tensor::Device,
        mut should_stop: impl FnMut(&[usize]) -> bool,
    ) -> Vec<usize> {
        let prefix_len = prefix_ids.len();
        if prefix_len == 0 {
            return Vec::new();
        }

        let hidden = self.merge_audio_embeds(prefix_ids, audio_features, pad_token_id, device);
        let (hidden, mut caches) = self.forward_hidden_with_cache(hidden, device);
        let [_, seq, h] = hidden.dims();
        // Only the last position is scored. Running lm_head on the full prefix
        // (146 × 152064) was the main reason JFK never finished.
        let mut last_h = hidden.slice([0..1, seq - 1..seq, 0..h]);
        let mut generated = Vec::new();
        let mut pos = prefix_len;

        for _ in 0..max_new_tokens {
            let logits = self.lm_head.forward(last_h);
            let next = greedy_id(logits, banned_ids);
            if eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            if should_stop(&generated) {
                break;
            }

            let next_id = vec![next as i32];
            let ids_tensor =
                Tensor::<1, Int>::from_ints(next_id.as_slice(), device).reshape([1, 1]);
            let token_hidden = self.embed_tokens.forward(ids_tensor);
            let (cos, sin) = rope_embeddings_at_pos(pos, self.head_dim, self.rope_theta, device);
            let mask = Tensor::<2>::zeros([1, pos + 1], device);
            let mut x = token_hidden;
            for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                let (out, new_cache) = layer.forward_with_cache(x, &cos, &sin, &mask, cache.take());
                *cache = Some(new_cache);
                x = out;
            }
            last_h = self.norm.forward(x);
            pos += 1;
        }

        generated
    }

    fn forward_hidden_with_cache(
        &self,
        inputs_embeds: Tensor<3>,
        device: &burn::tensor::Device,
    ) -> (Tensor<3>, Vec<Option<KvLayerCache>>) {
        let [_, seq, _] = inputs_embeds.dims();
        let (cos, sin) = rope_embeddings(seq, self.head_dim, self.rope_theta, device);
        let mask = attn_decoder_mask(seq, device);
        let mut x = inputs_embeds;
        let mut caches = vec![None; self.layers.len()];
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let (out, new_cache) = layer.forward_with_cache(x, &cos, &sin, &mask, None);
            *cache = Some(new_cache);
            x = out;
        }
        (self.norm.forward(x), caches)
    }
}

fn greedy_id(logits: Tensor<3>, banned: &[usize]) -> usize {
    let data = logits.into_data();
    let mut v = data.to_vec::<f32>().expect("lm_head logits f32");
    for &b in banned {
        if b < v.len() {
            v[b] = f32::NEG_INFINITY;
        }
    }
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn apply_rope(x: Tensor<4>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.clone().slice([0..b, 0..h, 0..s, half..d]);
    let rotated = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);
    let cos = cos.clone().unsqueeze_dim::<4>(1);
    let sin = sin.clone().unsqueeze_dim::<4>(1);
    x * cos + rotated * sin
}

fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
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

fn rope_embeddings(
    seq: usize,
    head_dim: usize,
    theta: f64,
    device: &burn::tensor::Device,
) -> (Tensor<3>, Tensor<3>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (theta.powf(-(2.0 * i as f64) / head_dim as f64)) as f32)
        .collect();
    let inv = Tensor::<1>::from_floats(inv_freq.as_slice(), device);
    let t = Tensor::<1, Int>::arange(0..seq as i64, device).float();
    let freqs = t.unsqueeze::<2>().transpose().matmul(inv.unsqueeze::<2>());
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);
    let cos = emb.clone().cos().unsqueeze::<3>();
    let sin = emb.sin().unsqueeze::<3>();
    (cos, sin)
}

fn rope_embeddings_at_pos(
    pos: usize,
    head_dim: usize,
    theta: f64,
    device: &burn::tensor::Device,
) -> (Tensor<3>, Tensor<3>) {
    let (cos, sin) = rope_embeddings(pos + 1, head_dim, theta, device);
    (
        cos.slice([0..1, pos..pos + 1, 0..head_dim]),
        sin.slice([0..1, pos..pos + 1, 0..head_dim]),
    )
}
