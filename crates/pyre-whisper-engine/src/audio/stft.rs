use burn::tensor::{Tensor};
use crate::helper::*;

pub fn hann_window(window_length: usize, device: &burn::tensor::Device) -> Tensor<1> {
    Tensor::arange(0..window_length as i64, device)
        .float()
        .mul_scalar(std::f64::consts::PI / window_length as f64)
        .sin()
        .powf_scalar(2.0)
}

/// Periodic=false Hann window used by HF Parakeet (torch.hann_window(..., periodic=False)).
pub fn hann_window_symmetric(window_length: usize, device: &burn::tensor::Device) -> Tensor<1> {
    if window_length <= 1 {
        return Tensor::ones([window_length.max(1)], device);
    }
    let denom = (window_length - 1) as f64;
    Tensor::arange(0..window_length as i64, device)
        .float()
        .mul_scalar(std::f64::consts::PI * 2.0 / denom)
        .cos()
        .mul_scalar(-0.5)
        .add_scalar(0.5)
}

/// torch.stft(..., center=True, pad_mode="constant") style STFT for HF Parakeet.
pub fn stfft_centered_constant(
    input: Tensor<2>,
    n_fft: usize,
    hop_length: usize,
    window: Tensor<1>,
) -> (Tensor<3>, Tensor<3>) {
    let [n_batch, _] = input.dims();
    let device = input.device();
    let pad = n_fft / 2;
    let left = Tensor::zeros([n_batch, pad], &device);
    let right = Tensor::zeros([n_batch, pad], &device);
    let input = Tensor::cat(vec![left, input, right], 1);
    stfft_frames(input, n_fft, hop_length, window)
}

/// HF `WhisperFeatureExtractor` STFT (`center=False`): no edge padding, last mel frame dropped later.
pub fn stfft_whisper_hf(
    input: Tensor<2>,
    n_fft: usize,
    hop_length: usize,
    window: Tensor<1>,
) -> (Tensor<3>, Tensor<3>) {
    stfft_frames(input, n_fft, hop_length, window)
}

pub fn stfft(
    input: Tensor<2>,
    n_fft: usize,
    hop_length: usize,
    window: Tensor<1>,
) -> (Tensor<3>, Tensor<3>) {
    let [n_batch, orig_input_size] = input.dims();
    // Short clips (or a short final mel chunk) used to assert and poison the STT mutex.
    let input = if orig_input_size < n_fft {
        let pad_right = Tensor::zeros([n_batch, n_fft - orig_input_size], &input.device());
        Tensor::cat(vec![input, pad_right], 1)
    } else {
        input
    };
    let [_, orig_input_size] = input.dims();

    let pad = n_fft / 2;
    let left_pad = reverse(input.clone().slice([0..input.dims()[0], 1..(pad + 1)]), 1);
    let right_pad = reverse(
        input.clone().slice([
            0..input.dims()[0],
            (orig_input_size - pad - 1)..(orig_input_size - 1),
        ]),
        1,
    );
    let input = Tensor::cat(vec![left_pad, input, right_pad], 1);
    stfft_frames(input, n_fft, hop_length, window)
}

fn stfft_frames(
    input: Tensor<2>,
    n_fft: usize,
    hop_length: usize,
    window: Tensor<1>,
) -> (Tensor<3>, Tensor<3>) {
    let [n_batch, input_size] = input.dims();
    let device = input.device();

    let [orig_window_length] = window.dims();
    let window = if orig_window_length < n_fft {
        let left_pad = (n_fft - orig_window_length) / 2;
        let right_pad = n_fft - orig_window_length - left_pad;
        Tensor::cat(
            vec![
                Tensor::zeros([left_pad], &device),
                window,
                Tensor::zeros([right_pad], &device),
            ],
            0,
        )
    } else {
        window
    };

    let n_frame = (input_size - n_fft) / hop_length + 1;
    let n_freq = n_fft / 2 + 1;

    let mut windows_vec = Vec::with_capacity(n_frame);
    for i in 0..n_frame {
        let start = i * hop_length;
        let end = start + n_fft;
        windows_vec.push(input.clone().slice([0..n_batch, start..end]).reshape([n_batch, n_fft, 1]));
    }
    let input_windows = Tensor::cat(windows_vec, 2);

    let coe = std::f64::consts::PI * 2.0 / n_fft as f64;
    let b = Tensor::arange(0..n_freq as i64, &device)
        .float()
        .mul_scalar(coe)
        .unsqueeze::<2>()
        .transpose()
        .repeat(&[1, n_fft])
        * Tensor::arange(0..n_fft as i64, &device)
            .float()
            .unsqueeze::<2>();

    let real_part = (b.clone().cos() * window.clone().unsqueeze())
        .unsqueeze()
        .matmul(input_windows.clone());
    let imaginary_part = (b.sin() * (-window).unsqueeze())
        .unsqueeze()
        .matmul(input_windows);

    (real_part, imaginary_part)
}
