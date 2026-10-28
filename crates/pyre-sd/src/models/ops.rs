//! Precision-aware building blocks shared by the models.
//!
//! * Normalisation in half precision: the variance sums squares over a whole
//!   group (about a million values per group in the VAE at 512x512), which
//!   overflows f16. [`group_norm`] and [`layer_norm`] compute the statistics
//!   in f32 when the input is f16/bf16 and are plain Burn modules otherwise.
//! * Attention: [`attention`] runs either Burn's fused `attention` op (flash
//!   attention kernels on cubecl backends, autotuned against a fallback) or
//!   the original matmul/softmax/matmul, selected by [`set_attention_mode`].

use core::sync::atomic::{AtomicU8, Ordering};

use burn::nn::{GroupNorm, LayerNorm};
use burn::tensor::activation::softmax;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, FloatDType, Tensor};

/// Epsilon of every LayerNorm in SD 1.x (CLIP and the transformer blocks).
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// Whether `dtype` needs f32 statistics for normalisation.
pub fn is_half(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

/// GroupNorm that computes mean and variance in f32 for half inputs.
pub fn group_norm<const D: usize>(
    norm: &GroupNorm,
    x: Tensor<D>,
) -> Tensor<D> {
    if is_half(x.dtype()) {
        group_norm_f32(norm, x)
    } else {
        norm.forward(x)
    }
}

/// The f32-statistics GroupNorm, for any input dtype (public for tests).
pub fn group_norm_f32<const D: usize>(
    norm: &GroupNorm,
    x: Tensor<D>,
) -> Tensor<D> {
    let dtype = x.dtype();
    let shape = x.shape();
    let (batch, channels) = (shape[0], shape[1]);
    assert_eq!(channels, norm.num_channels, "GroupNorm channel count");
    let hidden = shape.num_elements() / batch / norm.num_groups;
    let x = x
        .cast(FloatDType::F32)
        .reshape([batch, norm.num_groups, hidden]);
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(2);
    let normed = (centered / var.add_scalar(norm.epsilon).sqrt())
        .reshape(shape)
        .cast(dtype);
    if !norm.affine {
        return normed;
    }
    let mut affine = [1usize; D];
    affine[1] = channels;
    let gamma = norm
        .gamma
        .as_ref()
        .expect("affine GroupNorm has gamma")
        .val();
    let beta = norm.beta.as_ref().expect("affine GroupNorm has beta").val();
    normed * gamma.reshape(affine) + beta.reshape(affine)
}

/// LayerNorm over the last dimension with f32 statistics for half inputs.
/// `eps` must match the module's (Burn keeps it private);
/// [`LAYER_NORM_EPS`] for SD 1.x.
pub fn layer_norm<const D: usize>(
    norm: &LayerNorm,
    x: Tensor<D>,
    eps: f64,
) -> Tensor<D> {
    if is_half(x.dtype()) {
        layer_norm_f32(norm, x, eps)
    } else {
        norm.forward(x)
    }
}

/// The f32-statistics LayerNorm, for any input dtype (public for tests).
pub fn layer_norm_f32<const D: usize>(
    norm: &LayerNorm,
    x: Tensor<D>,
    eps: f64,
) -> Tensor<D> {
    let dtype = x.dtype();
    let x = x.cast(FloatDType::F32);
    let mean = x.clone().mean_dim(D - 1);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(D - 1);
    let normed = (centered / var.add_scalar(eps).sqrt()).cast(dtype);
    let out = normed * norm.gamma.val().unsqueeze::<D>();
    match &norm.beta {
        Some(beta) => out + beta.val().unsqueeze::<D>(),
        None => out,
    }
}

/// How attention is computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionMode {
    /// [`Naive`](AttentionMode::Naive) while the score matrix is small (it is
    /// faster there), [`Fused`](AttentionMode::Fused) once it would take more
    /// than [`AUTO_FUSED_ABOVE`] bytes.
    #[default]
    Auto,
    /// Burn's fused attention op (flash attention where the backend has it).
    Fused,
    /// Explicit QK^T, softmax, matmul with V (materialises the score matrix).
    Naive,
}

/// Score-matrix size above which [`AttentionMode::Auto`] switches to the fused
/// op. SD 1.5 at 512x512 has a 512 MiB score matrix in f16 (naive is about 12%
/// faster per UNet step there on an RTX 3060) and 1 GiB in f32 (same speed, but
/// fused saves 2 GB of VRAM).
pub const AUTO_FUSED_ABOVE: usize = 768 << 20;

/// Largest score matrix the fused op is asked to cover in a single pass.
///
/// Flash attention never needs the matrix, and on a backend with a real flash
/// kernel this would not matter. It matters here because cubecl autotunes the
/// kernel per shape and some candidates do materialise it: an SDXL refiner at
/// 1024 px has a 6.4 GB widest block, and that ran a 12 GB card out of memory
/// while the same model on the tiled naive path finished in 2.1 s/step at
/// 10.7 GB. Tiling the query sequence bounds the peak either way, and is
/// exact — a query row's output depends only on that row, never on the rows
/// beside it — so the result is bit-for-bit the same as one big pass.
pub const FUSED_TILE_ABOVE: usize = 1 << 30;

impl AttentionMode {
    pub fn name(self) -> &'static str {
        match self {
            AttentionMode::Auto => "auto",
            AttentionMode::Fused => "fused",
            AttentionMode::Naive => "naive",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(AttentionMode::Auto),
            "fused" | "flash" => Some(AttentionMode::Fused),
            "naive" | "matmul" => Some(AttentionMode::Naive),
            _ => None,
        }
    }

    /// The implementation to use for a `[batch, heads, q_len, k_len]` score
    /// matrix of `elem_bytes`-sized elements.
    pub fn resolve(
        self,
        batch: usize,
        heads: usize,
        q_len: usize,
        k_len: usize,
        elem_bytes: usize,
    ) -> Self {
        match self {
            AttentionMode::Auto => {
                let bytes = batch
                    .saturating_mul(heads)
                    .saturating_mul(q_len)
                    .saturating_mul(k_len)
                    .saturating_mul(elem_bytes);
                if bytes > AUTO_FUSED_ABOVE {
                    AttentionMode::Fused
                } else {
                    AttentionMode::Naive
                }
            }
            m => m,
        }
    }
}

static ATTENTION_MODE: AtomicU8 = AtomicU8::new(0);

/// Select the attention implementation for all models in this process.
pub fn set_attention_mode(mode: AttentionMode) {
    ATTENTION_MODE.store(mode as u8, Ordering::Relaxed);
}

pub fn attention_mode() -> AttentionMode {
    match ATTENTION_MODE.load(Ordering::Relaxed) {
        1 => AttentionMode::Fused,
        2 => AttentionMode::Naive,
        _ => AttentionMode::Auto,
    }
}

/// How many query rows to score at once so the matrix stays under `limit` bytes.
pub fn naive_query_tile(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    elem_bytes: usize,
    limit: usize,
) -> usize {
    let row = batch
        .saturating_mul(heads)
        .saturating_mul(k_len)
        .saturating_mul(elem_bytes)
        .max(1);
    (limit / row).clamp(1, q_len.max(1))
}

/// How many query rows the fused op scores per pass, under `limit` bytes.
pub fn fused_query_tile(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    elem_bytes: usize,
    limit: usize,
) -> usize {
    naive_query_tile(batch, heads, q_len, k_len, elem_bytes, limit)
}

/// Explicit QK^T, sliced along the query sequence when the full matrix would
/// pass [`AUTO_FUSED_ABOVE`]. Softmax is over the key axis, so the slices match
/// one full matmul.
fn naive_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f64,
) -> Tensor<4> {
    let [batch, heads, q_len, _] = q.dims();
    let k_len = k.dims()[2];
    let tile = naive_query_tile(
        batch,
        heads,
        q_len,
        k_len,
        q.dtype().size(),
        AUTO_FUSED_ABOVE,
    );
    if tile >= q_len {
        return naive_attention_tile(q, k, v, scale);
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < q_len {
        let end = (start + tile).min(q_len);
        let q_slice = q
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..q.dims()[3]]);
        parts.push(naive_attention_tile(q_slice, k.clone(), v.clone(), scale));
        start = end;
    }
    Tensor::cat(parts, 2)
}

fn naive_attention_tile(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f64,
) -> Tensor<4> {
    // Scale q and k by sqrt(scale) each, as diffusers does, which keeps
    // the scores in range for f16.
    let s = scale.sqrt();
    let scores = (q * s).matmul(k.swap_dims(2, 3) * s);
    softmax(scores, 3).matmul(v)
}

/// softmax(q k^T * scale) v for `[batch, heads, seq, head_dim]` tensors,
/// using the current [`attention_mode`].
pub fn attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f64,
) -> Tensor<4> {
    attention_with(attention_mode(), q, k, v, scale)
}

/// [`attention`] with an explicit mode.
pub fn attention_with(
    mode: AttentionMode,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f64,
) -> Tensor<4> {
    let [batch, heads, q_len, _] = q.dims();
    let k_len = k.dims()[2];
    let mode = mode.resolve(batch, heads, q_len, k_len, q.dtype().size());
    match mode {
        AttentionMode::Auto | AttentionMode::Fused => {
            // Burn runs the flash kernel only when `scale` is None, and then
            // uses 1/sqrt(head_dim). An explicit scale takes the fallback that
            // materialises QK^T. Fold `scale` into Q so the kernel still
            // computes softmax(Q K^T * scale).
            let tile = fused_query_tile(
                batch,
                heads,
                q_len,
                k_len,
                q.dtype().size(),
                FUSED_TILE_ABOVE,
            );
            if tile >= q_len {
                return fused_attention(q, k, v, scale);
            }
            let head = q.dims()[3];
            let mut parts = Vec::with_capacity(q_len.div_ceil(tile));
            let mut start = 0;
            while start < q_len {
                let end = (start + tile).min(q_len);
                let rows = q.clone().slice([0..batch, 0..heads, start..end, 0..head]);
                parts.push(fused_attention(rows, k.clone(), v.clone(), scale));
                start = end;
            }
            Tensor::cat(parts, 2)
        }
        AttentionMode::Naive => naive_attention(q, k, v, scale),
    }
}

/// One pass of the fused attention op, with `scale` folded into `q`.
fn fused_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f64,
) -> Tensor<4> {
    let head = q.dims()[3] as f64;
    let factor = scale * head.sqrt();
    let q = if (factor - 1.0).abs() < 1e-8 {
        q
    } else {
        q * factor
    };
    burn::tensor::module::attention(
        q,
        k,
        v,
        None,
        None,
        AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
            causal_alignment: Default::default(),
        },
    )
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::nn::{GroupNormConfig, LayerNormConfig};
    use burn::tensor::{Distribution, Tolerance};

    type B = TestBackend;

    fn random<const D: usize>(shape: [usize; D], seed: u64) -> Tensor<D> {
        let device = Default::default();
        device.seed(seed);
        Tensor::random(shape, Distribution::Normal(0.5, 3.0), &device)
    }

    #[test]
    fn group_norm_f32_matches_burn() {
        let device = Default::default();
        let mut norm = GroupNormConfig::new(4, 8).init(&device);
        norm.gamma = Some(burn::module::Param::from_tensor(random([8], 1)));
        norm.beta = Some(burn::module::Param::from_tensor(random([8], 2)));
        let x = random([2, 8, 5, 3], 3);
        group_norm_f32(&norm, x.clone())
            .into_data()
            .assert_approx_eq::<f32>(&norm.forward(x).into_data(), Tolerance::rel_abs(1e-4, 1e-5));
    }

    #[test]
    fn layer_norm_f32_matches_burn() {
        let device = Default::default();
        let mut norm = LayerNormConfig::new(6).init(&device);
        norm.gamma = burn::module::Param::from_tensor(random([6], 4));
        let x = random([2, 3, 6], 5);
        layer_norm_f32(&norm, x.clone(), LAYER_NORM_EPS)
            .into_data()
            .assert_approx_eq::<f32>(&norm.forward(x).into_data(), Tolerance::rel_abs(1e-4, 1e-5));
    }

    #[test]
    fn fused_attention_matches_naive() {
        let (q, k, v) = (
            random([2, 3, 7, 4], 6),
            random([2, 3, 5, 4], 7),
            random([2, 3, 5, 6], 8),
        );
        let fused = attention_with(AttentionMode::Fused, q.clone(), k.clone(), v.clone(), 0.5);
        let naive = attention_with(AttentionMode::Naive, q, k, v, 0.5);
        assert_eq!(fused.dims(), [2, 3, 7, 6]);
        fused
            .into_data()
            .assert_approx_eq::<f32>(&naive.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
    }

    #[test]
    fn fused_attention_honours_a_scale_other_than_one_over_sqrt_dim() {
        // head dim 4, so 1/sqrt(d) is 0.5. 0.25 must still match the explicit scores.
        let (q, k, v) = (
            random([1, 2, 5, 4], 9),
            random([1, 2, 6, 4], 10),
            random([1, 2, 6, 4], 11),
        );
        let fused = attention_with(AttentionMode::Fused, q.clone(), k.clone(), v.clone(), 0.25);
        let naive = attention_with(AttentionMode::Naive, q, k, v, 0.25);
        fused
            .into_data()
            .assert_approx_eq::<f32>(&naive.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
    }

    #[test]
    fn auto_attention_picks_by_score_size() {
        // SD 1.5 at 512x512: 4096 tokens, 8 heads, CFG batch 2.
        assert_eq!(
            AttentionMode::Auto.resolve(2, 8, 4096, 4096, 2),
            AttentionMode::Naive
        );
        assert_eq!(
            AttentionMode::Auto.resolve(2, 8, 4096, 4096, 4),
            AttentionMode::Fused
        );
        // 768x768 in f16.
        assert_eq!(
            AttentionMode::Auto.resolve(2, 8, 9216, 9216, 2),
            AttentionMode::Fused
        );
        // Cross attention against 77 text tokens stays naive.
        assert_eq!(
            AttentionMode::Auto.resolve(2, 8, 16384, 77, 4),
            AttentionMode::Naive
        );
        // 1024 px self-attention (16384 tokens) must not materialise scores.
        assert_eq!(
            AttentionMode::Auto.resolve(2, 8, 16384, 16384, 2),
            AttentionMode::Fused
        );
        // If it does (Attention set to naive), score it in tiles under the same cap.
        let tile = naive_query_tile(2, 8, 16384, 16384, 2, AUTO_FUSED_ABOVE);
        assert!(tile < 16384);
        assert!(2 * 8 * tile * 16384 * 2 <= AUTO_FUSED_ABOVE);
        assert_eq!(
            AttentionMode::Fused.resolve(1, 1, 1, 1, 2),
            AttentionMode::Fused
        );
        assert_eq!(
            AttentionMode::Naive.resolve(64, 64, 1 << 16, 1 << 16, 4),
            AttentionMode::Naive
        );
    }

    #[test]
    fn attention_mode_round_trips() {
        for m in [
            AttentionMode::Auto,
            AttentionMode::Fused,
            AttentionMode::Naive,
        ] {
            assert_eq!(AttentionMode::parse(m.name()), Some(m));
            set_attention_mode(m);
            assert_eq!(attention_mode(), m);
        }
        set_attention_mode(AttentionMode::default());
        assert_eq!(AttentionMode::parse("flash"), Some(AttentionMode::Fused));
        assert_eq!(AttentionMode::parse("x"), None);
        assert!(is_half(DType::F16) && is_half(DType::BF16) && !is_half(DType::F32));
    }
}
