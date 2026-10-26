//! Packed linear layers for VibeVoice Qwen2.5 decoders.
//!
//! * 7B (`vibevoice-asr`): per-output-channel INT8 for attn/MLP; `lm_head` stays
//!   host f32 (Q4_K → f32, no extra INT8) and is still tiled so the 2.18 GB
//!   matrix never lands on the GPU in one piece.
//! * 1.5B BitNet (`bitnet-asr`): per-tensor absmean ternary `{-1,0,+1}` (I2_S).
//!
//! Packed `i8` (or host f32) stays on the host. Each forward copies **one tile**
//! into f32, runs the matmul, then drops the tile.

use std::fmt;
use std::sync::Arc;

use burn::module::{Module, Param};
use burn::tensor::{Tensor};

#[derive(Module)]
pub struct QuantLinear {
    #[module(skip)]
    qweight: Arc<Vec<i8>>,
    #[module(skip)]
    scale_host: Arc<Vec<f32>>,
    #[module(skip)]
    f32_weight: Option<Arc<Vec<f32>>>,
    #[module(skip)]
    d_in: usize,
    #[module(skip)]
    d_out: usize,
    pub scale: Param<Tensor<1>>,
    pub bias: Option<Param<Tensor<1>>>,
}

impl fmt::Debug for QuantLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantLinear")
            .field("d_in", &self.d_in)
            .field("d_out", &self.d_out)
            .field("packed_bytes", &self.qweight.len())
            .finish()
    }
}

impl QuantLinear {
    pub fn empty(d_in: usize, d_out: usize, with_bias: bool, device: &burn::tensor::Device) -> Self {
        Self {
            qweight: Arc::new(Vec::new()),
            scale_host: Arc::new(vec![1.0; d_out]),
            f32_weight: None,
            d_in,
            d_out,
            scale: Param::from_tensor(Tensor::<1>::ones([d_out], device)),
            bias: with_bias.then(|| Param::from_tensor(Tensor::<1>::zeros([d_out], device))),
        }
    }

    pub fn from_quantized(
        qweight: Vec<i8>,
        scale: Vec<f32>,
        bias: Option<Vec<f32>>,
        d_in: usize,
        d_out: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        debug_assert_eq!(qweight.len(), d_in * d_out);
        debug_assert_eq!(scale.len(), d_out);
        let scale_t = Tensor::<1>::from_floats(scale.as_slice(), device);
        let bias = bias.map(|b| Param::from_tensor(Tensor::<1>::from_floats(b.as_slice(), device)));
        Self {
            qweight: Arc::new(qweight),
            scale_host: Arc::new(scale),
            f32_weight: None,
            d_in,
            d_out,
            scale: Param::from_tensor(scale_t),
            bias,
        }
    }

    /// Host f32 weights, still tiled on forward (used for 7B `lm_head`).
    pub fn from_f32(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        d_in: usize,
        d_out: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        debug_assert_eq!(weight.len(), d_in * d_out);
        let bias =
            bias.map(|b| Param::from_tensor(Tensor::<1>::from_floats(b.as_slice(), device)));
        Self {
            qweight: Arc::new(Vec::new()),
            scale_host: Arc::new(vec![1.0; d_out]),
            f32_weight: Some(Arc::new(weight)),
            d_in,
            d_out,
            scale: Param::from_tensor(Tensor::<1>::ones([d_out], device)),
            bias,
        }
    }

    pub fn device(&self) -> burn::tensor::Device {
        self.scale.val().device()
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, s, d] = x.dims();
        debug_assert_eq!(d, self.d_in);
        let device = x.device();
        let x2 = x.reshape([b * s, d]);
        // Cap tiles at ~128 MiB. Full lm_head is 2.18 GB (OOM); MLP 271 MB fits in one tile.
        const MAX_F32: usize = 32 * 1024 * 1024;
        let chunk_out = (MAX_F32 / self.d_in.max(1)).max(1).min(self.d_out.max(1));
        let mut parts = Vec::new();
        let mut col = 0usize;
        while col < self.d_out {
            let col1 = (col + chunk_out).min(self.d_out);
            let w = self.dequant_cols(col, col1, &device);
            parts.push(x2.clone().matmul(w));
            col = col1;
        }
        let mut y = Tensor::cat(parts, 1).reshape([b, s, self.d_out]);
        if let Some(bias) = &self.bias {
            y = y + bias.val().unsqueeze::<2>().unsqueeze::<3>();
        }
        y
    }

    fn dequant_cols(&self, col0: usize, col1: usize, device: &burn::tensor::Device) -> Tensor<2> {
        let w_out = col1 - col0;
        let q = self.qweight.as_slice();
        let sc = self.scale_host.as_slice();
        let mut w = vec![0f32; self.d_in * w_out];
        if let Some(full) = &self.f32_weight {
            let table = full.as_slice();
            if table.len() == self.d_in * self.d_out {
                for i in 0..self.d_in {
                    let src = i * self.d_out + col0;
                    let dst = i * w_out;
                    w[dst..dst + w_out].copy_from_slice(&table[src..src + w_out]);
                }
            }
        } else if q.len() == self.d_in * self.d_out {
            for i in 0..self.d_in {
                let src = i * self.d_out + col0;
                let dst = i * w_out;
                for j in 0..w_out {
                    w[dst + j] = q[src + j] as f32 * sc[col0 + j];
                }
            }
        }
        Tensor::<1>::from_floats(w.as_slice(), device).reshape([self.d_in, w_out])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearQuant {
    /// Symmetric INT8 with a scale per output channel.
    Int8,
    /// BitNet I2_S: per-tensor absmean, values in `{-1,0,+1}`.
    Ternary,
}

/// Symmetric per-output-channel INT8 (Burn layout `[d_in, d_out]`).
pub fn quantize_per_out_channel(weight: &[f32], d_in: usize, d_out: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; d_in * d_out];
    let mut scale = vec![0f32; d_out];
    for j in 0..d_out {
        let mut max = 0f32;
        for i in 0..d_in {
            max = max.max(weight[i * d_out + j].abs());
        }
        let s = (max / 127.0).max(1e-8);
        scale[j] = s;
        for i in 0..d_in {
            let v = (weight[i * d_out + j] / s).round().clamp(-127.0, 127.0);
            q[i * d_out + j] = v as i8;
        }
    }
    (q, scale)
}

/// BitNet absmean ternary used by VibeVoice-ASR-BitNet / VibeASR.cpp.
///
/// `scale = mean(|W|)`, `q = round(W / scale).clamp(-1, 1)`, reconstruct `q * scale`.
/// The scale vector is length `d_out` (identical values) so dequant stays per-column.
pub fn quantize_ternary_absmean(weight: &[f32], d_in: usize, d_out: usize) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(weight.len(), d_in * d_out);
    let n = weight.len().max(1) as f32;
    let mean_abs = weight.iter().map(|w| w.abs()).sum::<f32>() / n;
    let scale = mean_abs.max(1e-5);
    let mut q = vec![0i8; d_in * d_out];
    for (i, &w) in weight.iter().enumerate() {
        q[i] = (w / scale).round().clamp(-1.0, 1.0) as i8;
    }
    (q, vec![scale; d_out])
}

pub fn quantize_linear(
    scheme: LinearQuant,
    weight: &[f32],
    d_in: usize,
    d_out: usize,
) -> (Vec<i8>, Vec<f32>) {
    match scheme {
        LinearQuant::Int8 => quantize_per_out_channel(weight, d_in, d_out),
        LinearQuant::Ternary => quantize_ternary_absmean(weight, d_in, d_out),
    }
}

#[cfg(test)]
mod tests {
    use super::{quantize_per_out_channel, quantize_ternary_absmean};

    #[test]
    fn f32_forward_matches_matmul() {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        use burn::tensor::Tensor;
        let device = NdArrayDevice::Cpu;
        let d_in = 3;
        let d_out = 4;
        // Burn layout [d_in, d_out]
        let w: Vec<f32> = (0..d_in * d_out).map(|i| i as f32 * 0.5).collect();
        let layer = super::QuantLinear::<NdArray>::from_f32(w.clone(), None, d_in, d_out, &device);
        let x = Tensor::<NdArray, 1>::from_floats([1.0f32, 0.0, 2.0].as_slice(), &device).reshape([1, 1, d_in]);
        let y = layer.forward(x).into_data().to_vec::<f32>().unwrap();
        // y[j] = x0*w[0,j] + x2*w[2,j]
        let mut expect = vec![0f32; d_out];
        for j in 0..d_out {
            expect[j] = 1.0 * w[j] + 2.0 * w[2 * d_out + j];
        }
        for (a, b) in y.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "{y:?} vs {expect:?}");
        }
    }

    #[test]
    fn roundtrip_stays_close() {
        let d_in = 8;
        let d_out = 4;
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| (i as f32 - 10.0) * 0.13)
            .collect();
        let (q, scale) = quantize_per_out_channel(&w, d_in, d_out);
        let mut max_err = 0f32;
        for i in 0..d_in {
            for j in 0..d_out {
                let rec = q[i * d_out + j] as f32 * scale[j];
                max_err = max_err.max((rec - w[i * d_out + j]).abs());
            }
        }
        assert!(max_err < 0.02, "max dequant error {max_err}");
    }

    #[test]
    fn ternary_absmean_matches_vibeasr() {
        let d_in = 6;
        let d_out = 4;
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| (i as f32 - 11.0) * 0.07)
            .collect();
        let mean = w.iter().map(|x| x.abs()).sum::<f32>() / w.len() as f32;
        let (q, scale) = quantize_ternary_absmean(&w, d_in, d_out);
        assert!(q.iter().all(|&v| v == -1 || v == 0 || v == 1));
        for s in &scale {
            assert!((s - mean.max(1e-5)).abs() < 1e-6);
        }
        let mut max_err = 0f32;
        for i in 0..w.len() {
            let rec = q[i] as f32 * scale[0];
            max_err = max_err.max((rec - w[i]).abs());
        }
        assert!(max_err < mean + 1e-5, "ternary reconstruction error {max_err}");
    }
}
