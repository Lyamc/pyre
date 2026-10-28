//! Stable Diffusion 3 / 3.5 MMDiT. [`SD3Config::medium`] is SD 3.5 Medium:
//! 24 layers, hidden size 1536, QK RMSNorm, and a second self-attention on
//! layers 0 through 12. Positional embeddings are a center crop of a 384 grid.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::{FloatDType, Tensor};

use crate::models::embeddings::{
    get_timestep_embedding, TimestepEmbedding, TimestepEmbeddingConfig,
};
use crate::models::ops::attention;

const EPS: f64 = 1e-6;

/// SD 3.5 Medium, and smaller configs used by tests.
#[derive(Clone, Debug)]
pub struct SD3Config {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub joint_dim: usize,
    pub pooled_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch: usize,
    /// Latent grid covered by the learned positional table.
    pub pos_max: usize,
    /// Layers `0..dual_until` have the extra image self-attention.
    pub dual_until: usize,
}

impl SD3Config {
    pub fn medium() -> Self {
        Self {
            layers: 24,
            heads: 24,
            head_dim: 64,
            joint_dim: 4096,
            pooled_dim: 2048,
            in_channels: 16,
            out_channels: 16,
            patch: 2,
            pos_max: 384,
            dual_until: 13,
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> SD3Transformer {
        let dim = self.heads * self.head_dim;
        let blocks = (0..self.layers)
            .map(|i| {
                Block::new(
                    dim,
                    self.heads,
                    self.head_dim,
                    i + 1 == self.layers,
                    i < self.dual_until,
                    device,
                )
            })
            .collect();
        SD3Transformer {
            pos_embed: PatchEmbed::new(self.in_channels, dim, self.patch, self.pos_max, device),
            time_text_embed: TimeText::new(dim, self.pooled_dim, device),
            context_embedder: LinearConfig::new(self.joint_dim, dim).init(device),
            transformer_blocks: blocks,
            norm_out: Ada::new(dim, dim * 2, device),
            proj_out: LinearConfig::new(dim, self.patch * self.patch * self.out_channels)
                .init(device),
            patch: self.patch,
            out_channels: self.out_channels,
        }
    }
}

#[derive(Module, Debug)]
pub struct SD3Transformer {
    pos_embed: PatchEmbed,
    time_text_embed: TimeText,
    context_embedder: Linear,
    transformer_blocks: Vec<Block>,
    norm_out: Ada,
    proj_out: Linear,
    #[module(skip)]
    patch: usize,
    #[module(skip)]
    out_channels: usize,
}

impl SD3Transformer {
    /// `hidden` is `[batch, channels, h, w]`. `context` is `[batch, tokens, joint]`.
    /// `pooled` is `[batch, pooled]`. `timestep` is `sigma * 1000`.
    pub fn forward(
        &self,
        hidden: Tensor<4>,
        context: Tensor<3>,
        pooled: Tensor<2>,
        timestep: f64,
    ) -> Tensor<4> {
        let [_, _, height, width] = hidden.dims();
        let mut hidden = self.pos_embed.forward(hidden);
        let temb = self.time_text_embed.forward(timestep, pooled);
        let mut context = self.context_embedder.forward(context);
        for block in &self.transformer_blocks {
            (context, hidden) = block.forward(hidden, context, temb.clone());
        }
        hidden = self.norm_out.continuous(hidden, temb);
        hidden = self.proj_out.forward(hidden);
        unpatchify(hidden, height, width, self.patch, self.out_channels)
    }
}

#[derive(Module, Debug)]
struct PatchEmbed {
    proj: Conv2d,
    pos_embed: Param<Tensor<3>>,
    #[module(skip)]
    patch: usize,
    #[module(skip)]
    pos_max: usize,
}

impl PatchEmbed {
    fn new(in_ch: usize, dim: usize, patch: usize, pos_max: usize, device: &burn::tensor::Device) -> Self {
        PatchEmbed {
            proj: Conv2dConfig::new([in_ch, dim], [patch, patch])
                .with_stride([patch, patch])
                .init(device),
            pos_embed: Param::from_tensor(Tensor::zeros([1, pos_max * pos_max, dim], device)),
            patch,
            pos_max,
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<3> {
        let [b, _, h, w] = x.dims();
        let gh = h / self.patch;
        let gw = w / self.patch;
        let x = self.proj.forward(x);
        let channels = x.dims()[1];
        let x = x.reshape([b, channels, gh * gw]).swap_dims(1, 2);
        let table = self
            .pos_embed
            .val()
            .reshape([self.pos_max, self.pos_max, channels]);
        let top = (self.pos_max - gh) / 2;
        let left = (self.pos_max - gw) / 2;
        let pos = table
            .slice([top..top + gh, left..left + gw, 0..channels])
            .reshape([1, gh * gw, channels]);
        x + pos
    }
}

#[derive(Module, Debug)]
struct TimeText {
    timestep_embedder: TimestepEmbedding,
    text_embedder: TimestepEmbedding,
}

impl TimeText {
    fn new(dim: usize, pooled: usize, device: &burn::tensor::Device) -> Self {
        Self {
            timestep_embedder: TimestepEmbeddingConfig::new(256, dim).init(device),
            text_embedder: TimestepEmbeddingConfig::new(pooled, dim).init(device),
        }
    }

    fn forward(&self, timestep: f64, pooled: Tensor<2>) -> Tensor<2> {
        let t = Tensor::<1>::from_floats([timestep as f32], &pooled.device());
        let emb = get_timestep_embedding(t, 256, true, 0.0);
        self.timestep_embedder.forward(emb) + self.text_embedder.forward(pooled)
    }
}

#[derive(Module, Debug)]
struct Ada {
    linear: Linear,
}

impl Ada {
    fn new(input: usize, output: usize, device: &burn::tensor::Device) -> Self {
        Self {
            linear: LinearConfig::new(input, output).init(device),
        }
    }

    fn chunks(&self, emb: Tensor<2>, n: usize) -> Vec<Tensor<2>> {
        self.linear.forward(silu(emb)).chunk(n, 1)
    }

    fn continuous(&self, x: Tensor<3>, emb: Tensor<2>) -> Tensor<3> {
        let mut parts = self.chunks(emb, 2);
        let shift = parts.pop().unwrap();
        let scale = parts.pop().unwrap();
        modulate(layernorm(x), shift, scale)
    }
}

#[derive(Module, Debug)]
struct Rms {
    weight: Param<Tensor<1>>,
}

impl Rms {
    fn new(dim: usize, device: &burn::tensor::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let dtype = x.dtype();
        let xf = x.cast(FloatDType::F32);
        let var = xf.clone().square().mean_dim(3);
        let head = xf.dims()[3];
        let y = xf / var.add_scalar(EPS).sqrt();
        y.cast(dtype) * self.weight.val().reshape([1, 1, 1, head])
    }
}

#[derive(Module, Debug)]
struct Gelu {
    proj: Linear,
}

#[derive(Module, Debug)]
struct FeedForward {
    gelu: Gelu,
    out: Linear,
}

impl FeedForward {
    fn new(dim: usize, device: &burn::tensor::Device) -> Self {
        let inner = dim * 4;
        Self {
            gelu: Gelu {
                proj: LinearConfig::new(dim, inner).init(device),
            },
            out: LinearConfig::new(inner, dim).init(device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.out.forward(gelu_tanh(self.gelu.proj.forward(x)))
    }
}

#[derive(Module, Debug)]
struct JointAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    to_add_out: Option<Linear>,
    norm_q: Rms,
    norm_k: Rms,
    norm_added_q: Rms,
    norm_added_k: Rms,
    #[module(skip)]
    heads: usize,
}

impl JointAttn {
    fn new(dim: usize, heads: usize, head: usize, pre_only: bool, device: &burn::tensor::Device) -> Self {
        let linear = |a, b| LinearConfig::new(a, b).init(device);
        Self {
            to_q: linear(dim, dim),
            to_k: linear(dim, dim),
            to_v: linear(dim, dim),
            to_out: linear(dim, dim),
            add_q_proj: linear(dim, dim),
            add_k_proj: linear(dim, dim),
            add_v_proj: linear(dim, dim),
            to_add_out: (!pre_only).then(|| linear(dim, dim)),
            norm_q: Rms::new(head, device),
            norm_k: Rms::new(head, device),
            norm_added_q: Rms::new(head, device),
            norm_added_k: Rms::new(head, device),
            heads,
        }
    }

    fn forward(&self, hidden: Tensor<3>, context: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let image_len = hidden.dims()[1];
        let q = self
            .norm_q
            .forward(to_heads(self.to_q.forward(hidden.clone()), self.heads));
        let k = self
            .norm_k
            .forward(to_heads(self.to_k.forward(hidden.clone()), self.heads));
        let v = to_heads(self.to_v.forward(hidden), self.heads);
        let qc = self.norm_added_q.forward(to_heads(
            self.add_q_proj.forward(context.clone()),
            self.heads,
        ));
        let kc = self.norm_added_k.forward(to_heads(
            self.add_k_proj.forward(context.clone()),
            self.heads,
        ));
        let vc = to_heads(self.add_v_proj.forward(context), self.heads);
        let q = Tensor::cat(vec![q, qc], 2);
        let k = Tensor::cat(vec![k, kc], 2);
        let v = Tensor::cat(vec![v, vc], 2);
        let scale = 1.0 / (q.dims()[3] as f64).sqrt();
        let out = from_heads(attention(q, k, v, scale));
        let [batch, seq, width] = out.dims();
        let image = out.clone().slice([0..batch, 0..image_len, 0..width]);
        let ctx = out.slice([0..batch, image_len..seq, 0..width]);
        let image = self.to_out.forward(image);
        let ctx = match &self.to_add_out {
            Some(proj) => proj.forward(ctx),
            None => ctx,
        };
        (image, ctx)
    }
}

#[derive(Module, Debug)]
struct SelfAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Rms,
    norm_k: Rms,
    #[module(skip)]
    heads: usize,
}

impl SelfAttn {
    fn new(dim: usize, heads: usize, head: usize, device: &burn::tensor::Device) -> Self {
        let linear = |a, b| LinearConfig::new(a, b).init(device);
        Self {
            to_q: linear(dim, dim),
            to_k: linear(dim, dim),
            to_v: linear(dim, dim),
            to_out: linear(dim, dim),
            norm_q: Rms::new(head, device),
            norm_k: Rms::new(head, device),
            heads,
        }
    }

    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let q = self
            .norm_q
            .forward(to_heads(self.to_q.forward(hidden.clone()), self.heads));
        let k = self
            .norm_k
            .forward(to_heads(self.to_k.forward(hidden.clone()), self.heads));
        let v = to_heads(self.to_v.forward(hidden), self.heads);
        let scale = 1.0 / (q.dims()[3] as f64).sqrt();
        self.to_out.forward(from_heads(attention(q, k, v, scale)))
    }
}

#[derive(Module, Debug)]
struct Block {
    norm1: Ada,
    norm1_context: Ada,
    attn: JointAttn,
    attn2: Option<SelfAttn>,
    ff: FeedForward,
    ff_context: Option<FeedForward>,
    #[module(skip)]
    dual: bool,
    #[module(skip)]
    pre_only: bool,
}

impl Block {
    fn new(
        dim: usize,
        heads: usize,
        head: usize,
        pre_only: bool,
        dual: bool,
        device: &burn::tensor::Device,
    ) -> Self {
        Self {
            norm1: Ada::new(dim, if dual { dim * 9 } else { dim * 6 }, device),
            norm1_context: Ada::new(dim, if pre_only { dim * 2 } else { dim * 6 }, device),
            attn: JointAttn::new(dim, heads, head, pre_only, device),
            attn2: dual.then(|| SelfAttn::new(dim, heads, head, device)),
            ff: FeedForward::new(dim, device),
            ff_context: (!pre_only).then(|| FeedForward::new(dim, device)),
            dual,
            pre_only,
        }
    }

    fn forward(
        &self,
        hidden: Tensor<3>,
        context: Tensor<3>,
        temb: Tensor<2>,
    ) -> (Tensor<3>, Tensor<3>) {
        let (normed, gate_msa, shift_mlp, scale_mlp, gate_mlp, normed2, gate_msa2) = if self.dual {
            let mut p = self.norm1.chunks(temb.clone(), 9);
            let gate_msa2 = p.pop().unwrap();
            let scale_msa2 = p.pop().unwrap();
            let shift_msa2 = p.pop().unwrap();
            let gate_mlp = p.pop().unwrap();
            let scale_mlp = p.pop().unwrap();
            let shift_mlp = p.pop().unwrap();
            let gate_msa = p.pop().unwrap();
            let scale_msa = p.pop().unwrap();
            let shift_msa = p.pop().unwrap();
            let base = layernorm(hidden.clone());
            (
                modulate(base.clone(), shift_msa, scale_msa),
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                Some(modulate(base, shift_msa2, scale_msa2)),
                Some(gate_msa2),
            )
        } else {
            let mut p = self.norm1.chunks(temb.clone(), 6);
            let gate_mlp = p.pop().unwrap();
            let scale_mlp = p.pop().unwrap();
            let shift_mlp = p.pop().unwrap();
            let gate_msa = p.pop().unwrap();
            let scale_msa = p.pop().unwrap();
            let shift_msa = p.pop().unwrap();
            (
                modulate(layernorm(hidden.clone()), shift_msa, scale_msa),
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                None,
                None,
            )
        };

        let (image, ctx_attn) = if self.pre_only {
            let norm_c = self.norm1_context.continuous(context.clone(), temb);
            self.attn.forward(normed, norm_c)
        } else {
            let mut p = self.norm1_context.chunks(temb, 6);
            let c_gate_mlp = p.pop().unwrap();
            let c_scale_mlp = p.pop().unwrap();
            let c_shift_mlp = p.pop().unwrap();
            let c_gate_msa = p.pop().unwrap();
            let c_scale_msa = p.pop().unwrap();
            let c_shift_msa = p.pop().unwrap();
            let norm_c = modulate(layernorm(context.clone()), c_shift_msa, c_scale_msa);
            let (image, ctx_attn) = self.attn.forward(normed, norm_c);
            let ctx_attn = c_gate_msa.unsqueeze_dim::<3>(1) * ctx_attn;
            let context = context + ctx_attn;
            let norm_c = modulate(layernorm(context.clone()), c_shift_mlp, c_scale_mlp);
            let context = context
                + c_gate_mlp.unsqueeze_dim::<3>(1)
                    * self.ff_context.as_ref().unwrap().forward(norm_c);
            return finish(
                self,
                hidden,
                image,
                normed2,
                gate_msa,
                gate_msa2,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                Some(context),
            );
        };
        let _ = ctx_attn;
        finish(
            self, hidden, image, normed2, gate_msa, gate_msa2, shift_mlp, scale_mlp, gate_mlp, None,
        )
    }
}

fn finish(
    block: &Block,
    hidden: Tensor<3>,
    image: Tensor<3>,
    normed2: Option<Tensor<3>>,
    gate_msa: Tensor<2>,
    gate_msa2: Option<Tensor<2>>,
    shift_mlp: Tensor<2>,
    scale_mlp: Tensor<2>,
    gate_mlp: Tensor<2>,
    context: Option<Tensor<3>>,
) -> (Tensor<3>, Tensor<3>) {
    let mut hidden = hidden + gate_msa.unsqueeze_dim::<3>(1) * image;
    if let (Some(attn2), Some(normed2), Some(gate2)) = (&block.attn2, normed2, gate_msa2) {
        hidden = hidden + gate2.unsqueeze_dim::<3>(1) * attn2.forward(normed2);
    }
    let ff = gate_mlp.unsqueeze_dim::<3>(1)
        * block
            .ff
            .forward(modulate(layernorm(hidden.clone()), shift_mlp, scale_mlp));
    hidden = hidden + ff;
    (
        context.unwrap_or_else(|| Tensor::zeros(hidden.dims(), &hidden.device())),
        hidden,
    )
}

fn to_heads(x: Tensor<3>, heads: usize) -> Tensor<4> {
    let [b, s, d] = x.dims();
    x.reshape([b, s, heads, d / heads]).swap_dims(1, 2)
}

fn from_heads(x: Tensor<4>) -> Tensor<3> {
    let [b, h, s, d] = x.dims();
    x.swap_dims(1, 2).reshape([b, s, h * d])
}

fn layernorm(x: Tensor<3>) -> Tensor<3> {
    let dtype = x.dtype();
    let xf = x.cast(FloatDType::F32);
    let mean = xf.clone().mean_dim(2);
    let centered = xf - mean;
    let var = centered.clone().square().mean_dim(2);
    (centered / var.add_scalar(EPS).sqrt()).cast(dtype)
}

fn modulate(x: Tensor<3>, shift: Tensor<2>, scale: Tensor<2>) -> Tensor<3> {
    x * scale.unsqueeze_dim::<3>(1).add_scalar(1.0) + shift.unsqueeze_dim::<3>(1)
}

fn gelu_tanh<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let cube = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cube * 0.044715) * (2.0 / std::f64::consts::PI).sqrt();
    x * (inner.tanh().add_scalar(1.0)) * 0.5
}

fn unpatchify(
    x: Tensor<3>,
    height: usize,
    width: usize,
    patch: usize,
    channels: usize,
) -> Tensor<4> {
    let gh = height / patch;
    let gw = width / patch;
    let [b, _, _] = x.dims();
    x.reshape([b, gh, gw, patch, patch, channels])
        .permute([0, 5, 1, 3, 2, 4])
        .reshape([b, channels, height, width])
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::TestBackend;

    type B = TestBackend;

    #[test]
    fn a_tiny_mmdit_keeps_the_latent_shape() {
        let device = Default::default();
        let cfg = SD3Config {
            layers: 2,
            heads: 2,
            head_dim: 4,
            joint_dim: 8,
            pooled_dim: 4,
            in_channels: 4,
            out_channels: 4,
            patch: 2,
            pos_max: 8,
            dual_until: 1,
        };
        let model = cfg.init(&device);
        let latent = Tensor::<4>::zeros([1, 4, 4, 4], &device);
        let context = Tensor::<3>::zeros([1, 3, 8], &device);
        let pooled = Tensor::<2>::zeros([1, 4], &device);
        let out = model.forward(latent, context, pooled, 500.0);
        assert_eq!(out.dims(), [1, 4, 4, 4]);
    }
}
