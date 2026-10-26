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

    pub fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
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
}
