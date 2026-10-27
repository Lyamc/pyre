//! 3x3 convolutions as im2col + matmul.
//!
//! On backends without tensor-core MMA (wgpu), Burn's conv autotuner only has its direct kernel for
//! k>1, which made SD 1.5 spend ~97% of its time in `DirectConv2dKernel` (~25 s/step). Unfolding
//! the input into `[B, C*kh*kw, H*W]` columns and calling the (tiled) matmul is far faster there.
//! Output rows are processed in chunks so the column buffer stays under ~256 MiB.
use burn::nn::conv::Conv2d;
use burn::nn::PaddingConfig2d;
use burn::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

static IM2COL: AtomicBool = AtomicBool::new(false);
/// Route eligible convolutions through im2col (the host enables this on wgpu/vulkan).
pub fn set_im2col(on: bool) { IM2COL.store(on, Ordering::Relaxed); }

pub trait ConvFwd { fn fwd(&self, x: Tensor<4>) -> Tensor<4>; }

impl ConvFwd for Conv2d {
    fn fwd(&self, x: Tensor<4>) -> Tensor<4> {
        let [kh, kw] = self.kernel_size;
        let pad = match &self.padding { PaddingConfig2d::Explicit(t, l, b, r) if t == b && l == r && t == l => Some(*t), PaddingConfig2d::Valid => Some(0), _ => None };
        let ok = IM2COL.load(Ordering::Relaxed) && self.groups == 1 && self.dilation == [1, 1] && self.stride == [1, 1] && kh * kw > 1;
        match (ok, pad) { (true, Some(p)) => im2col_conv(self, x, p), _ => self.forward(x) }
    }
}

fn pad_hw(x: Tensor<4>, p: usize) -> Tensor<4> {
    if p == 0 { return x; }
    let [b, c, h, w] = x.dims();
    let zr = x.clone().slice([0..b, 0..c, 0..p, 0..w]).zeros_like();
    let x = Tensor::cat(vec![zr.clone(), x, zr], 2);
    let zc = x.clone().slice([0..b, 0..c, 0..h + 2 * p, 0..p]).zeros_like();
    Tensor::cat(vec![zc.clone(), x, zc], 3)
}

fn im2col_conv(conv: &Conv2d, x: Tensor<4>, p: usize) -> Tensor<4> {
    let [kh, kw] = conv.kernel_size;
    let [b, c, h, w] = x.dims();
    let xp = pad_hw(x, p);
    let (ho, wo) = (h + 2 * p - kh + 1, w + 2 * p - kw + 1);
    let wt = conv.weight.val();
    let o = wt.dims()[0];
    let wm: Tensor<3> = wt.reshape([1, o, c * kh * kw]);
    let bytes_per_row = (b * c * kh * kw * wo * 4).max(1);
    let rows = ((256usize << 20) / bytes_per_row).clamp(1, ho);
    let mut outs = vec![];
    let mut r0 = 0;
    while r0 < ho {
        let r1 = (r0 + rows).min(ho);
        let n = r1 - r0;
        let mut cols = Vec::with_capacity(kh * kw);
        for i in 0..kh { for j in 0..kw {
            cols.push(xp.clone().slice([0..b, 0..c, r0 + i..r0 + i + n, j..j + wo]).reshape([b, c, 1, n * wo]));
        } }
        let cols: Tensor<3> = Tensor::cat(cols, 2).reshape([b, c * kh * kw, n * wo]);
        outs.push(wm.clone().matmul(cols).reshape([b, o, n, wo]));
        r0 = r1;
    }
    let mut y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(outs, 2) };
    if let Some(bias) = &conv.bias { y = y + bias.val().reshape([1, o, 1, 1]); }
    y
}
