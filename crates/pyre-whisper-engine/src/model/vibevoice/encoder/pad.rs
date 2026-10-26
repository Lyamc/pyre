pub fn padding_total(kernel_size: usize, stride: usize, dilation: usize) -> usize {
    (kernel_size - 1) * dilation - (stride - 1)
}

pub fn extra_padding_for_conv1d(length: usize, kernel_size: usize, stride: usize, padding_total: usize) -> usize {
    let n_frames = (length as f64 - kernel_size as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal = (n_frames.ceil() as usize - 1) * stride + (kernel_size - padding_total);
    ideal.saturating_sub(length)
}

/// Zero padding on the time axis of `[batch, channels, time]`.
pub fn zero_pad1d(
    x: burn::tensor::Tensor<3>,
    left: usize,
    right: usize,
) -> burn::tensor::Tensor<3> {
    if left == 0 && right == 0 {
        return x;
    }
    let [batch, channels, time] = x.dims();
    let device = x.device();
    let total = left + time + right;
    let mut out = burn::tensor::Tensor::zeros([batch, channels, total], &device);
    if time > 0 {
        out = out.slice_assign([0..batch, 0..channels, left..left + time], x);
    }
    out
}

/// Output time length of a causal SConv1d (padding_total left + extra right, then stride).
#[cfg(test)]
pub fn conv_out_len(length: usize, kernel_size: usize, stride: usize) -> usize {
    let pad = padding_total(kernel_size, stride, 1);
    let extra = extra_padding_for_conv1d(length, kernel_size, stride, pad);
    let padded = length + pad + extra;
    if padded < kernel_size {
        0
    } else {
        (padded - kernel_size) / stride + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jfk_vae_tokens_match_ceil_compress() {
        // 11s @ 24 kHz after the same resample the CLI uses on samples/jfk.wav.
        let mut len = 264_064;
        len = conv_out_len(len, 7, 1); // stem
        for ratio in [2, 2, 4, 5, 5, 8] {
            len = conv_out_len(len, ratio * 2, ratio);
        }
        len = conv_out_len(len, 7, 1); // head
        let expected = (264_064 + 3200 - 1) / 3200;
        assert_eq!(len, expected, "encoder frames {len} vs ceil(samples/3200)={expected}");
    }
}