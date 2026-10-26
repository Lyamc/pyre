/// Resample-free mono waveform prep for VibeVoice (expects 24 kHz input).
pub fn normalize_dbfs(samples: &[f32], target_dbfs: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 1e-8 {
        return samples.to_vec();
    }
    let current_dbfs = 20.0 * rms.log10();
    let gain = 10f32.powf((target_dbfs - current_dbfs) / 20.0);
    samples.iter().map(|s| s * gain).collect()
}

pub fn vae_token_length(num_samples: usize, compress_ratio: usize) -> usize {
    (num_samples + compress_ratio - 1) / compress_ratio.max(1)
}

pub fn audio_duration_secs(num_samples: usize, sample_rate: usize) -> f32 {
    num_samples as f32 / sample_rate.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_len_matches_hf_ratio() {
        assert_eq!(vae_token_length(24_000, 3200), 8);
        assert_eq!(vae_token_length(240_000, 3200), 75);
    }

    #[test]
    fn normalize_increases_quiet_signal() {
        let quiet = vec![0.001f32; 1000];
        let loud = normalize_dbfs(&quiet, -25.0);
        let rms = (loud.iter().map(|x| x * x).sum::<f32>() / loud.len() as f32).sqrt();
        assert!(rms > 0.01);
    }
}