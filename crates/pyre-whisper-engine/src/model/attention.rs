use burn::{
    nn,
    tensor::{activation::softmax, Tensor, Int},
};
use burn::config::Config;
use burn::module::{Module, Param};

// --- Standard Attention (Whisper, RedHat) ---

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl MultiHeadAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> MultiHeadAttention {
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state).with_bias(false).init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadAttention {
            n_head: self.n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    pub n_head: usize,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    pub out: nn::Linear,
}

impl MultiHeadAttention {
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<2>>) -> Tensor<3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);
        qkv_attention(q, k, v, mask, self.n_head, &self.out)
    }
}

// --- Relative Positional Attention (Parakeet, FastConformer) ---

#[derive(Config, Debug)]
pub struct RelPosMultiHeadAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl RelPosMultiHeadAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> RelPosMultiHeadAttention {
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let pos = nn::LinearConfig::new(self.n_state, self.n_state).with_bias(false).init(device);

        let n_hstate = self.n_state / self.n_head;
        let pos_bias_u: Tensor<4> = Tensor::zeros([1, self.n_head, 1, n_hstate], device);
        let pos_bias_v: Tensor<4> = Tensor::zeros([1, self.n_head, 1, n_hstate], device);

        RelPosMultiHeadAttention {
            n_head: self.n_head,
            query,
            key,
            value,
            out,
            pos,
            pos_bias_u: Param::from_tensor(pos_bias_u.squeeze_dim::<3>(0).squeeze_dim::<2>(1)),
            pos_bias_v: Param::from_tensor(pos_bias_v.squeeze_dim::<3>(0).squeeze_dim::<2>(1)),
        }
    }
}

#[derive(Module, Debug)]
pub struct RelPosMultiHeadAttention {
    pub n_head: usize,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    pub out: nn::Linear,
    pub pos: nn::Linear,
    pub pos_bias_u: Param<Tensor<2>>,
    pub pos_bias_v: Param<Tensor<2>>,
}

impl RelPosMultiHeadAttention {
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<2>>) -> Tensor<3> {
        let [batch, seq_len, d_model] = x.dims();
        let n_hstate = d_model / self.n_head;

        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x.clone());

        let pos_emb = self.generate_rel_pos(seq_len, &x.device());
        let p = self.pos.forward(pos_emb);

        let q = q.reshape([batch, seq_len, self.n_head, n_hstate]).swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, self.n_head, n_hstate]).swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, self.n_head, n_hstate]).swap_dims(1, 2);
        let p = p.reshape([batch, seq_len * 2 - 1, self.n_head, n_hstate]).swap_dims(1, 2);

        // Transformer-XL style relative positional attention
        let u = self.pos_bias_u.val().reshape([1, self.n_head, 1, n_hstate]);
        let ac = (q.clone() + u).matmul(k.swap_dims(2, 3));

        let v_bias = self.pos_bias_v.val().reshape([1, self.n_head, 1, n_hstate]);
        let bd = (q.clone() + v_bias).matmul(p.swap_dims(2, 3));
        let bd = self.rel_shift(bd);

        let scale = (n_hstate as f64).powf(-0.5);
        let mut scores = (ac + bd).mul_scalar(scale);

        if let Some(mask) = mask {
            let m_dims = mask.dims();
            if m_dims.len() == 2 {
                scores = scores + mask.unsqueeze::<3>().unsqueeze::<4>();
            } else {
                scores = scores + mask.unsqueeze::<4>();
            }
        }

        let attn = softmax(scores, 3);
        let context = attn.matmul(v);
        
        let out = context.swap_dims(1, 2).reshape([batch, seq_len, d_model]);
        self.out.forward(out)
    }

    fn generate_rel_pos(&self, seq_len: usize, device: &burn::tensor::Device) -> Tensor<3> {
        let [_, d_model] = self.pos.weight.val().dims();
        let half_d = d_model / 2;
        let log_step = (10000.0f64).ln() / (half_d - 1) as f64;
        let inv_freq = Tensor::<1, Int>::arange(0..half_d as i64, device).float().mul_scalar(-log_step).exp();
        
        let start = -(seq_len as i64 - 1);
        let end = seq_len as i64;
        let range = Tensor::<1, Int>::arange(start..end, device).float();
        let sin_inp = range.unsqueeze::<2>().transpose().matmul(inv_freq.unsqueeze::<2>());
        
        let sin = sin_inp.clone().sin();
        let cos = sin_inp.cos();
        
        Tensor::cat(vec![sin, cos], 1).unsqueeze()
    }

    fn rel_shift(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, heads, time1, time2] = x.dims();
        // x: [B, H, L, 2*L-1]
        let x = x.reshape([batch, heads, time2, time1]);
        let x = x.slice([0..batch, 0..heads, 1..time2, 0..time1]);
        let x = x.reshape([batch, heads, time1, time2 - 1]);
        x.slice([0..batch, 0..heads, 0..time1, 0..time1])
    }
}

// --- Cross Attention (Whisper) ---

#[derive(Config, Debug)]
pub struct MultiHeadCrossAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> MultiHeadCrossAttention {
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state).with_bias(false).init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadCrossAttention {
            n_head: self.n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention {
    pub n_head: usize,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    pub out: nn::Linear,
}

impl MultiHeadCrossAttention {
    pub fn forward(&self, x: Tensor<3>, xa: Tensor<3>) -> Tensor<3> {
        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);
        qkv_attention(q, k, v, None, self.n_head, &self.out)
    }
}

// --- Utils ---

pub fn qkv_attention(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    mask: Option<Tensor<2>>,
    n_head: usize,
    out_linear: &nn::Linear,
) -> Tensor<3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();
    let scale = (n_state as f64 / n_head as f64).powf(-0.5);
    let n_hstate = n_state / n_head;

    let q = q.reshape([n_batch, n_qctx, n_head, n_hstate]).swap_dims(1, 2) * scale;
    let k = k.reshape([n_batch, n_ctx, n_head, n_hstate]).swap_dims(1, 2).swap_dims(2, 3);
    let v = v.reshape([n_batch, n_ctx, n_head, n_hstate]).swap_dims(1, 2);

    let mut qk = q.matmul(k);
    if let Some(mask) = mask {
        qk = qk + mask.unsqueeze::<3>().unsqueeze::<4>();
    }
    let w = softmax(qk, 3);
    let context = w.matmul(v).swap_dims(1, 2).flatten(2, 3);
    out_linear.forward(context)
}

pub fn attn_decoder_mask(
    seq_length: usize,
    device: &burn::tensor::Device,
) -> Tensor<2> {
    let mut mask = Tensor::<2>::zeros([seq_length, seq_length], device);
    for i in 0..(seq_length - 1) {
        let values = Tensor::<2>::zeros([1, seq_length - (i + 1)], device)
            .add_scalar(-1e9);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }
    mask
}
