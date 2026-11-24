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
        MultiHeadAttention {
            n_head: self.n_head,
            query: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            key: nn::LinearConfig::new(self.n_state, self.n_state).with_bias(false).init(device),
            value: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            qkv: None,
            out: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    pub n_head: usize,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    /// Set after weight loading: q/k/v packed into a single matmul (order q, k, v).
    /// Optional so existing burnpack bundles (which name the three separately) still load.
    pub qkv: Option<nn::Linear>,
    pub out: nn::Linear,
}

impl MultiHeadAttention {
    /// One projection for all three, then split the last dimension into q/k/v.
    pub fn qkv_forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        if let Some(qkv) = &self.qkv {
            let y = qkv.forward(x);
            let [b, t, d3] = y.dims();
            let d = d3 / 3;
            (
                y.clone().slice([0..b, 0..t, 0..d]),
                y.clone().slice([0..b, 0..t, d..2 * d]),
                y.slice([0..b, 0..t, 2 * d..3 * d]),
            )
        } else {
            (
                self.query.forward(x.clone()),
                self.key.forward(x.clone()),
                self.value.forward(x),
            )
        }
    }

    /// Pack the loaded q/k/v weights into one Linear. Call once after loading.
    pub fn fuse_qkv(&mut self, device: &burn::tensor::Device) {
        let n = self.query.weight.dims()[0];
        let w = Tensor::cat(
            vec![
                self.query.weight.val().clone(),
                self.key.weight.val().clone(),
                self.value.weight.val().clone(),
            ],
            1,
        );
        let zero = || Tensor::zeros([n], device);
        let b = Tensor::cat(
            vec![
                self.query.bias.as_ref().map(|p| p.val().clone()).unwrap_or_else(zero),
                self.key.bias.as_ref().map(|p| p.val().clone()).unwrap_or_else(zero),
                self.value.bias.as_ref().map(|p| p.val().clone()).unwrap_or_else(zero),
            ],
            0,
        );
        let mut fused = nn::LinearConfig::new(n, n * 3).with_bias(true).init(device);
        fused.weight = Param::from_tensor(w);
        fused.bias = Some(Param::from_tensor(b));
        self.qkv = Some(fused);
    }

    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<2>>) -> Tensor<3> {
        let (q, k, v) = self.qkv_forward(x);
        qkv_attention(q, k, v, mask, self.n_head, &self.out)
    }

    /// Self-attention over `x_ln` (new positions only). `past` is K/V for the
    /// tokens already decoded; both are returned covering past + new.
    pub fn forward_self_step(
        &self,
        x_ln: Tensor<3>,
        past: Option<(Tensor<3>, Tensor<3>)>,
        mask: Option<Tensor<2>>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let (q, k_new, v_new) = self.qkv_forward(x_ln);
        let (k, v) = match past {
            Some((pk, pv)) => (
                Tensor::cat(vec![pk, k_new], 1),
                Tensor::cat(vec![pv, v_new], 1),
            ),
            None => (k_new, v_new),
        };
        let y = qkv_attention(q, k.clone(), v.clone(), mask, self.n_head, &self.out);
        (y, k, v)
    }
}

// --- Relative Positional Attention (Parakeet, FastConformer) ---

#[derive(Config, Debug)]
pub struct RelPosMultiHeadAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
    /// `attention_bias` from the checkpoint. Published Parakeet v2/v3 are false.
    #[config(default = false)]
    pub bias: bool,
}

impl RelPosMultiHeadAttentionConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> RelPosMultiHeadAttention {
        let query = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(self.bias)
            .init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(self.bias)
            .init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(self.bias)
            .init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(self.bias)
            .init(device);
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
            lora_q: None,
            lora_k: None,
            lora_v: None,
            lora_o: None,
            lora_pos: None,
        }
    }
}

#[cfg(test)]
mod rel_pos_tests {
    use super::RelPosMultiHeadAttentionConfig;

    #[test]
    fn relative_attention_has_no_projection_bias() {
        let device = burn::tensor::Device::flex();
        let attn = RelPosMultiHeadAttentionConfig::new(8, 2).init(&device);
        assert!(attn.query.bias.is_none());
        assert!(attn.key.bias.is_none());
        assert!(attn.value.bias.is_none());
        assert!(attn.out.bias.is_none());
        assert!(attn.pos.bias.is_none());
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
    #[module(skip)]
    pub lora_q: Option<crate::model::fluid::LoraAB>,
    #[module(skip)]
    pub lora_k: Option<crate::model::fluid::LoraAB>,
    #[module(skip)]
    pub lora_v: Option<crate::model::fluid::LoraAB>,
    #[module(skip)]
    pub lora_o: Option<crate::model::fluid::LoraAB>,
    #[module(skip)]
    pub lora_pos: Option<crate::model::fluid::LoraAB>,
}

impl RelPosMultiHeadAttention {
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<2>>) -> Tensor<3> {
        let [batch, seq_len, d_model] = x.dims();
        let n_hstate = d_model / self.n_head;

        let q = crate::model::fluid::linear_with_lora(&self.query, x.clone(), self.lora_q.as_ref());
        let k = crate::model::fluid::linear_with_lora(&self.key, x.clone(), self.lora_k.as_ref());
        let v = crate::model::fluid::linear_with_lora(&self.value, x.clone(), self.lora_v.as_ref());

        let pos_emb = self.generate_rel_pos(seq_len, &x.device());
        let p = crate::model::fluid::linear_with_lora(&self.pos, pos_emb, self.lora_pos.as_ref());

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
        crate::model::fluid::linear_with_lora(&self.out, out, self.lora_o.as_ref())
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
        MultiHeadCrossAttention {
            n_head: self.n_head,
            query: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            key: nn::LinearConfig::new(self.n_state, self.n_state).with_bias(false).init(device),
            value: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            qkv: None,
            out: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention {
    pub n_head: usize,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    /// Set after weight loading; optional so old bundles with separate q/k/v still load.
    pub qkv: Option<nn::Linear>,
    pub out: nn::Linear,
}

impl MultiHeadCrossAttention {
    fn qkv_forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        if let Some(qkv) = &self.qkv {
            let y = qkv.forward(x);
            let [b, t, d3] = y.dims();
            let d = d3 / 3;
            (
                y.clone().slice([0..b, 0..t, 0..d]),
                y.clone().slice([0..b, 0..t, d..2 * d]),
                y.slice([0..b, 0..t, 2 * d..3 * d]),
            )
        } else {
            (
                self.query.forward(x.clone()),
                self.key.forward(x.clone()),
                self.value.forward(x),
            )
        }
    }

    /// Pack the loaded q/k/v weights into one Linear. Call once after loading.
    pub fn fuse_qkv(&mut self, device: &burn::tensor::Device) {
        let n = self.query.weight.dims()[0];
        let w = Tensor::cat(
            vec![
                self.query.weight.val().clone(),
                self.key.weight.val().clone(),
                self.value.weight.val().clone(),
            ],
            1,
        );
        let zero = || Tensor::zeros([n], device);
        let b = Tensor::cat(
            vec![
                self.query.bias.as_ref().map(|p| p.val().clone()).unwrap_or_else(zero),
                self.key.bias.as_ref().map(|p| p.val().clone()).unwrap_or_else(zero),
                self.value.bias.as_ref().map(|p| p.val().clone()).unwrap_or_else(zero),
            ],
            0,
        );
        let mut fused = nn::LinearConfig::new(n, n * 3).with_bias(true).init(device);
        fused.weight = Param::from_tensor(w);
        fused.bias = Some(Param::from_tensor(b));
        self.qkv = Some(fused);
    }

    pub fn forward(&self, x: Tensor<3>, xa: Tensor<3>) -> Tensor<3> {
        let (q, _, _) = self.qkv_forward(x);
        let (_, k, v) = self.qkv_forward(xa);
        qkv_attention(q, k, v, None, self.n_head, &self.out)
    }

    /// Cross-attention. Encoder K/V are projected once and reused for every token.
    pub fn forward_cross_step(
        &self,
        x_ln: Tensor<3>,
        xa: &Tensor<3>,
        past: Option<(Tensor<3>, Tensor<3>)>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let (k, v) = match past {
            Some(kv) => kv,
            None => {
                let (_, k, v) = self.qkv_forward(xa.clone());
                (k, v)
            }
        };
        let (q, _, _) = self.qkv_forward(x_ln);
        let y = qkv_attention(q, k.clone(), v.clone(), None, self.n_head, &self.out);
        (y, k, v)
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
    let [k_batch, n_ctx, _] = k.dims();
    let scale = (n_state as f64 / n_head as f64).powf(-0.5);
    let n_hstate = n_state / n_head;

    // One new token against a shared encoder K/V: put heads in front so the
    // matmul is [H, B, 1, D] @ [H, 1, D, T], the skinny GEMV shape.
    if n_qctx == 1 && k_batch == 1 && n_batch > 1 {
        let q = q.reshape([n_batch, 1, n_head, n_hstate]).swap_dims(1, 2).swap_dims(0, 1) * scale;
        let k = k
            .reshape([1, n_ctx, n_head, n_hstate])
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .swap_dims(0, 1);
        let v = v
            .reshape([1, n_ctx, n_head, n_hstate])
            .swap_dims(1, 2)
            .swap_dims(0, 1);
        let qk = q.matmul(k);
        let w = softmax(qk, 3);
        let context = w.matmul(v).swap_dims(0, 1).reshape([n_batch, 1, n_state]);
        return out_linear.forward(context);
    }

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

#[cfg(test)]
mod whisper_qkv_tests {
    use super::{MultiHeadAttentionConfig, MultiHeadCrossAttentionConfig};
    use burn::tensor::{Distribution, Tensor};

    fn close(a: Tensor<3>, b: Tensor<3>) {
        let a = a.into_data().to_vec::<f32>().unwrap();
        let b = b.into_data().to_vec::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn fused_qkv_matches_separate_projections() {
        let device = burn::tensor::Device::flex();
        let x = Tensor::<3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device);

        let mut attn = MultiHeadAttentionConfig::new(8, 2).init(&device);
        let q = attn.query.forward(x.clone());
        let k = attn.key.forward(x.clone());
        let v = attn.value.forward(x.clone());
        attn.fuse_qkv(&device);
        let (qf, kf, vf) = attn.qkv_forward(x.clone());
        close(q, qf);
        close(k, kf);
        close(v, vf);

        let mut cross = MultiHeadCrossAttentionConfig::new(8, 2).init(&device);
        let q = cross.query.forward(x.clone());
        let k = cross.key.forward(x.clone());
        let v = cross.value.forward(x.clone());
        cross.fuse_qkv(&device);
        let (qf, kf, vf) = cross.qkv_forward(x);
        close(q, qf);
        close(k, kf);
        close(v, vf);
    }
}
