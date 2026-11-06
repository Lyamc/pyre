use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{module::embedding, Distribution, Int, Tensor},
};

use crate::model::attention::{
    attn_decoder_mask, MultiHeadCrossAttention, MultiHeadCrossAttentionConfig,
    MultiHeadAttention, MultiHeadAttentionConfig,
};
use crate::model::mlp::{MLP, MLPConfig};

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init(&self, tensor_device_ref: &burn::tensor::Device) -> TextDecoder {
        let token_embedding = Param::from_tensor(Tensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let blocks: Vec<_> = (0..self.n_text_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(tensor_device_ref)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(tensor_device_ref);

        let mask = Param::from_tensor(attn_decoder_mask(self.n_text_ctx, tensor_device_ref));

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder {
    pub token_embedding: Param<Tensor<2>>,
    pub positional_embedding: Param<Tensor<2>>,
    pub blocks: Vec<ResidualDecoderAttentionBlock>,
    pub ln: nn::LayerNorm,
    pub mask: Param<Tensor<2>>,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
}

impl TextDecoder {
    pub fn forward(&self, x: Tensor<2, Int>, xa: Tensor<3>) -> Tensor<3> {
        let dims = x.dims();
        let [_, seq_len] = dims;

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mask = self.mask.val().slice([0..seq_len, 0..seq_len]);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, xa.clone(), mask.clone());
        }

        let x = self.ln.forward(x);
        return x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>());
    }

    /// One decoder step. `tokens` are only the new positions; `past` holds K/V
    /// for everything already consumed. Logits are the last new position, `[B, 1, vocab]`.
    pub fn forward_cached(
        &self,
        tokens: Tensor<2, Int>,
        xa: Tensor<3>,
        past: Option<DecoderKv>,
    ) -> (Tensor<3>, DecoderKv) {
        let past_len = past.as_ref().map(|p| p.len()).unwrap_or(0);
        let [_, seq_len] = tokens.dims();
        let abs_end = past_len + seq_len;
        assert!(
            abs_end <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            abs_end,
            self.n_text_ctx
        );
        assert!(
            past_len == 0 || seq_len == 1,
            "cached decode prefill is the whole prompt; later steps are one token"
        );

        let x = embedding(self.token_embedding.val(), tokens)
            + self
                .positional_embedding
                .val()
                .slice([past_len..abs_end])
                .unsqueeze::<3>();

        let mask = if past_len == 0 {
            Some(self.mask.val().slice([0..seq_len, 0..seq_len]))
        } else {
            None
        };

        let mut x = x;
        let mut self_k = Vec::with_capacity(self.blocks.len());
        let mut self_v = Vec::with_capacity(self.blocks.len());
        let mut cross_k = Vec::with_capacity(self.blocks.len());
        let mut cross_v = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let self_past = past
                .as_ref()
                .map(|p| (p.self_k[i].clone(), p.self_v[i].clone()));
            let cross_past = past
                .as_ref()
                .map(|p| (p.cross_k[i].clone(), p.cross_v[i].clone()));
            let (y, sk, sv, ck, cv) =
                block.forward_cached(x, &xa, self_past, cross_past, mask.clone());
            x = y;
            self_k.push(sk);
            self_v.push(sv);
            cross_k.push(ck);
            cross_v.push(cv);
        }

        let x = self.ln.forward(x);
        let [b, s, d] = x.dims();
        let x = x.slice([0..b, s - 1..s, 0..d]);
        let logits = x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>());
        (logits, DecoderKv { self_k, self_v, cross_k, cross_v })
    }

    pub fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
}

/// Per-layer K/V for autoregressive Whisper decoding.
/// Self entries grow by one token each step. Cross entries are the encoder
/// projections and stay fixed for a chunk.
#[derive(Clone, Debug)]
pub struct DecoderKv {
    pub self_k: Vec<Tensor<3>>,
    pub self_v: Vec<Tensor<3>>,
    pub cross_k: Vec<Tensor<3>>,
    pub cross_v: Vec<Tensor<3>>,
}

impl DecoderKv {
    pub fn len(&self) -> usize {
        self.self_k.first().map(|k| k.dims()[1]).unwrap_or(0)
    }

    pub fn batch(&self) -> usize {
        self.self_k.first().map(|k| k.dims()[0]).unwrap_or(1)
    }

    pub fn concat_batch(parts: &[DecoderKv]) -> DecoderKv {
        assert!(!parts.is_empty());
        let n_layers = parts[0].self_k.len();
        let mut self_k = Vec::with_capacity(n_layers);
        let mut self_v = Vec::with_capacity(n_layers);
        let mut cross_k = Vec::with_capacity(n_layers);
        let mut cross_v = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            self_k.push(Tensor::cat(parts.iter().map(|p| p.self_k[layer].clone()).collect::<Vec<_>>(), 0));
            self_v.push(Tensor::cat(parts.iter().map(|p| p.self_v[layer].clone()).collect::<Vec<_>>(), 0));
            cross_k.push(Tensor::cat(parts.iter().map(|p| p.cross_k[layer].clone()).collect::<Vec<_>>(), 0));
            cross_v.push(Tensor::cat(parts.iter().map(|p| p.cross_v[layer].clone()).collect::<Vec<_>>(), 0));
        }
        DecoderKv { self_k, self_v, cross_k, cross_v }
    }

    pub fn split_batch(&self) -> Vec<DecoderKv> {
        let b = self.batch();
        (0..b)
            .map(|i| DecoderKv {
                self_k: split_row(&self.self_k, i),
                self_v: split_row(&self.self_v, i),
                cross_k: split_row(&self.cross_k, i),
                cross_v: split_row(&self.cross_v, i),
            })
            .collect()
    }
}

fn split_row(layers: &[Tensor<3>], index: usize) -> Vec<Tensor<3>> {
    layers
        .iter()
        .map(|t| {
            let [_, time, dim] = t.dims();
            t.clone().slice([index..index + 1, 0..time, 0..dim])
        })
        .collect()
}

#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init(
        &self,
        tensor_device_ref: &burn::tensor::Device,
    ) -> ResidualDecoderAttentionBlock {
        let attn =
            MultiHeadAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let cross_attn =
            MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: nn::LayerNorm,
    pub cross_attn: MultiHeadCrossAttention,
    pub cross_attn_ln: nn::LayerNorm,
    pub mlp: MLP,
    pub mlp_ln: nn::LayerNorm,
}

impl ResidualDecoderAttentionBlock {
    pub fn forward(
        &self,
        x: Tensor<3>,
        xa: Tensor<3>,
        mask: Tensor<2>,
    ) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        let x = x.clone() + self.cross_attn.forward(self.cross_attn_ln.forward(x), xa);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        x
    }

    fn forward_cached(
        &self,
        x: Tensor<3>,
        xa: &Tensor<3>,
        self_past: Option<(Tensor<3>, Tensor<3>)>,
        cross_past: Option<(Tensor<3>, Tensor<3>)>,
        self_mask: Option<Tensor<2>>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>, Tensor<3>, Tensor<3>) {
        let (attn_out, sk, sv) =
            self.attn.forward_self_step(self.attn_ln.forward(x.clone()), self_past, self_mask);
        let x = x + attn_out;
        let (cross_out, ck, cv) = self
            .cross_attn
            .forward_cross_step(self.cross_attn_ln.forward(x.clone()), xa, cross_past);
        let x = x + cross_out;
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        (x, sk, sv, ck, cv)
    }
}
