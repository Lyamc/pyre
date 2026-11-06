use burn::tensor::{Tensor};
use crate::audio::AudioProcessor;
use std::ops::Div;
use std::sync::Arc;

use crate::model::load::helpers::load_tensor;

pub fn slice_to_mel(
    waveform: &[f32],
    start: usize,
    end: usize,
    processor: &AudioProcessor,
    device: &burn::tensor::Device,
) -> Tensor<3> {
    let end = end.min(waveform.len()).max(start);
    let mut chunk = waveform[start..end].to_vec();
    // Whisper STFT reflection pad needs >= n_fft (400) samples.
    if chunk.len() < 400 {
        chunk.resize(400, 0.0);
    }
    let waveform_tensor = Tensor::<1>::from_floats(chunk.as_slice(), device);
    processor.prep_audio(waveform_tensor.unsqueeze())
}

pub fn waveform_to_mel_tensor(
    waveform: Vec<f32>,
    sample_rate: usize,
    window_length_samples: usize,
    device: &burn::tensor::Device,
    n_mels: usize,
    is_nemo: bool,
    model_name: &str,
) -> impl Iterator<Item = Tensor<3>> {
    let chunk_overlap = sample_rate * 3;
    let n_samples_per_tensor = window_length_samples.max(1);
    let shift = n_samples_per_tensor.saturating_sub(chunk_overlap).max(1);
    let iter_len = if waveform.is_empty() {
        0
    } else {
        waveform.len().saturating_sub(1).div(shift) + 1
    };
    let device_clone = device.clone();

    let mut processor = AudioProcessor::new(&device_clone, sample_rate as f64, n_mels, is_nemo, model_name);
    
    if is_nemo {
        let dir = pyre_whisper_core::resolve_model_dir(model_name);
        let mean_path = dir.join("mean.npy");
        let std_path = dir.join("std.npy");
        if mean_path.exists() && std_path.exists() {
            let dir_str = dir.to_string_lossy();
            let mean = load_tensor::<2>("mean", dir_str.as_ref(), &device_clone).unwrap();
            let std = load_tensor::<2>("std", dir_str.as_ref(), &device_clone).unwrap();
            processor = processor.with_stats(mean, std);
        }
    }

    let processor_arc = Arc::new(processor);

    (0..iter_len).map(move |i| {
        let processor = processor_arc.clone();
        let start = i * shift;
        let end = (start + n_samples_per_tensor).min(waveform.len());
        slice_to_mel(&waveform, start, end, &processor, &device_clone)
    })
}
