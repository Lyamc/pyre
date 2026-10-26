use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use pyre_whisper_core::WhisburnError;

pub fn resample_mono(
    input: &[f32],
    from_rate: usize,
    to_rate: usize,
) -> Result<Vec<f32>, WhisburnError> {
    if from_rate == to_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut resampler = Fft::<f32>::new(from_rate, to_rate, 1024, 1, FixedSync::Both)
        .map_err(|e| WhisburnError::Audio(e.to_string()))?;
    let adapter = InterleavedSlice::new(input, 1, input.len())
        .map_err(|e| WhisburnError::Audio(e.to_string()))?;
    let owned = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|e| WhisburnError::Audio(e.to_string()))?;
    Ok(owned.take_data())
}