pub mod mel;
pub mod qwen3_chunk;
pub mod stft;

use burn::tensor::cast::ToElement;
use burn::tensor::{Tensor};
use crate::helper::*;
use self::mel::*;
use self::stft::{hann_window, hann_window_symmetric, stfft, stfft_centered_constant};

/// Mel frames after HF Whisper feature extraction (`stft[..., :-1]`).
pub fn qwen3_valid_mel_frames(mel_frames: usize) -> usize {
    mel_frames.saturating_sub(1)
}

/// Valid mel frames for HF Parakeet (attention_mask length, excludes STFT tail frame).
pub fn parakeet_valid_mel_frames(mel_frames: usize) -> usize {
    // torch.stft(center=True): n_mel = 1 + floor(audio_samples / hop); mask uses floor(audio/hop).
    mel_frames.saturating_sub(1).max(1)
}

/// HF FastConformer subsampling: three stride-2 conv layers (8× on time).
pub fn parakeet_subsampled_frames(mel_frames: usize) -> usize {
    let mut len = parakeet_valid_mel_frames(mel_frames) as i64;
    for _ in 0..3 {
        len = (len - 1) / 2 + 1;
    }
    len.max(1) as usize
}

const WHISPER_N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const PARAKEET_LOG_ZERO_GUARD: f32 = 1.0 / 16777216.0; // 2^-24
const PARAKEET_PREEMPHASIS: f32 = 0.97;

pub struct AudioProcessor {
    pub mel_filters: Tensor<2>,
    pub window: Tensor<1>,
    pub n_mels: usize,
    pub n_fft: usize,
    pub is_nemo: bool,
    pub is_parakeet: bool,
    pub is_qwen3: bool,
    pub is_tone: bool,
    pub mean: Option<Tensor<2>>,
    pub std: Option<Tensor<2>>,
}

impl AudioProcessor {
    pub fn new(device: &burn::tensor::Device, sample_rate: f64, n_mels: usize, is_nemo: bool, model_name: &str) -> Self {
        let is_tone = model_name.contains("t-one");
        let is_parakeet = model_name.contains("parakeet");
        let is_qwen3 = model_name.contains("qwen3");
        let n_fft = if is_nemo || is_tone || is_parakeet { 512 } else { WHISPER_N_FFT };
        let win_length = 400;
        // HF Parakeet uses Slaney-normalized mel filters (librosa-style).
        let mel_filters = get_mel_filters(
            sample_rate,
            n_fft,
            n_mels,
            is_tone,
            device,
            !is_nemo || is_parakeet,
        );
        let window = if is_parakeet {
            hann_window_symmetric(win_length, device)
        } else {
            hann_window(win_length, device)
        };

        AudioProcessor {
            mel_filters,
            window,
            n_mels,
            n_fft,
            is_nemo,
            is_parakeet,
            is_qwen3,
            is_tone,
            mean: None,
            std: None,
        }
    }

    pub fn with_stats(mut self, mean: Tensor<2>, std: Tensor<2>) -> Self {
        self.mean = Some(mean);
        self.std = Some(std);
        self
    }

    pub fn prep_audio(&self, waveform: Tensor<2>) -> Tensor<3> {
        // Whisper: float [-1, 1]. Legacy NeMo/T-ONE: int16-scale. Parakeet HF: float + preemphasis.
        let waveform = if self.is_parakeet {
            waveform
        } else if self.is_nemo || self.is_tone {
            waveform.mul_scalar(32768.0)
        } else {
            waveform
        };
        let [b, n] = waveform.dims();
        let waveform = if self.is_parakeet && n > 1 {
            let x0 = waveform.clone().slice([0..b, 0..(n - 1)]);
            let x1 = waveform.clone().slice([0..b, 1..n]);
            let y = x1 - x0.mul_scalar(PARAKEET_PREEMPHASIS);
            Tensor::cat(vec![waveform.clone().slice([0..b, 0..1]), y], 1)
        } else {
            waveform
        };

        let (real, imag) = if self.is_parakeet {
            stfft_centered_constant(waveform, self.n_fft, HOP_LENGTH, self.window.clone())
        } else {
            stfft(waveform, self.n_fft, HOP_LENGTH, self.window.clone())
        };
        let spec = real.powf_scalar(2.0) + imag.powf_scalar(2.0);
        let spec = spec.swap_dims(1, 2);
        let mels = spec.matmul(self.mel_filters.clone().transpose().unsqueeze());
        let mels = mels.swap_dims(1, 2);

        if self.is_tone {
            // T-one style: natural log + standard instance normalization
            let mels = (mels + 1e-5).log();
            let mean = mels.clone().mean_dim(2);
            let diff = mels.clone() - mean.clone();
            let var = diff.powf_scalar(2.0).mean_dim(2);
            let std = var.sqrt().add_scalar(1e-5);
            (mels - mean) / std
        } else if self.is_parakeet {
            let mels = (mels + PARAKEET_LOG_ZERO_GUARD).log();
            let [b, c, t] = mels.dims();
            let valid_t = parakeet_valid_mel_frames(t).min(t);
            let valid = mels.clone().slice([0..b, 0..c, 0..valid_t]);
            let mean = valid.clone().mean_dim(2);
            let diff = valid.clone() - mean.clone();
            let denom = (valid_t.saturating_sub(1).max(1)) as f64;
            let var = diff.powf_scalar(2.0).sum_dim(2).div_scalar(denom);
            let std = var.sqrt().add_scalar(1e-5);
            let device = mels.device();
            let normed = (mels - mean) / std;
            let mut mask_vals = vec![0f32; t];
            for v in mask_vals.iter_mut().take(valid_t) {
                *v = 1.0;
            }
            let frame_mask = Tensor::<1>::from_floats(mask_vals.as_slice(), &device)
                .reshape([1, 1, t]);
            normed * frame_mask
        } else if self.is_nemo {
            let mels = (mels + 1e-5).log();

            if let (Some(mean), Some(std)) = (&self.mean, &self.std) {
                (mels - mean.clone().unsqueeze()) / std.clone().unsqueeze()
            } else {
                mels
            }
        } else {
            // Standard Whisper style: log10, dynamic-range clamp, global scale
            let mels = tensor_log10(tensor_max_scalar(mels, 1e-10));
            let peak = mels.clone().max().into_scalar::<f64>();
            let mels = tensor_max_scalar(mels, peak - 8.0);
            let mels = (mels + 4.0).div_scalar(4.0);
            if self.is_qwen3 {
                let [b, c, t] = mels.dims();
                if t > 0 {
                    mels.slice([0..b, 0..c, 0..t - 1])
                } else {
                    mels
                }
            } else {
                mels
            }
        }
    }
}

pub fn max_waveform_samples(n_ctx: usize) -> usize {
    n_ctx * HOP_LENGTH
}

pub fn prep_audio(waveform: Tensor<2>, sample_rate: f64, n_mels: usize, is_nemo: bool, model_name: &str) -> Tensor<3> {
    let processor = AudioProcessor::new(&waveform.device(), sample_rate, n_mels, is_nemo, model_name);
    processor.prep_audio(waveform)
}
