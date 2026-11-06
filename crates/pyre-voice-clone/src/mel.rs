//! 16 kHz mel front-end for the Chatterbox voice encoder.
//!
//! This follows `chatterbox/models/voice_encoder/melspec.py` against librosa 0.10:
//! periodic Hann, `n_fft = 400`, hop 160, reflect-centered STFT, power spectrum
//! (`mel_power = 2`), Slaney mel basis with `n_mels = 40`, `fmin = 0`, `fmax = 8000`.
//! `normalized_mels` is false and `mel_type` is `"amp"`, so the values are linear
//! power mels, not decibels and not scaled to `[0, 1]`.
//!
//! The same Slaney scale and periodic Hann already exist in `pyre-whisper-engine`
//! (`audio::mel`, `audio::stft`). Those functions are behind the `stt` feature and
//! use Whisper's 80-bin, 400-hop filterbank, so this crate keeps its own 40-bin
//! contract instead of taking that dependency.
//!
//! Silence trimming matches `librosa.effects.trim(..., top_db=20)` (the default
//! `embeds_from_wavs` passes). Resampling anything other than 16 kHz is linear.
//! Upstream uses `librosa.resample(..., res_type="kaiser_fast")`. Callers that need
//! a clone match should resample to 16 kHz before calling.

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_MELS: usize = 40;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_FREQ: usize = N_FFT / 2 + 1;
pub const PARTIAL_FRAMES: usize = 160;
/// `embeds_from_wavs` sets this when the caller does not. It is a partial-utterance
/// rate in partials per second, not a playback speed.
pub const DEFAULT_PARTIAL_RATE: f64 = 1.3;
pub const MIN_COVERAGE: f64 = 0.8;

pub fn partial_frame_step(rate: f64) -> usize {
    let step = ((SAMPLE_RATE as f64 / rate) / PARTIAL_FRAMES as f64).round() as usize;
    step.clamp(1, PARTIAL_FRAMES)
}

/// `(n_partials, target_frames)` from `get_num_wins`.
pub fn partial_windows(n_frames: usize, frame_step: usize) -> (usize, usize) {
    let span = (n_frames as isize - PARTIAL_FRAMES as isize + frame_step as isize).max(0) as usize;
    let (mut n_wins, remainder) = (span / frame_step, span % frame_step);
    if n_wins == 0
        || (remainder + (PARTIAL_FRAMES - frame_step)) as f64 / PARTIAL_FRAMES as f64
            >= MIN_COVERAGE
    {
        n_wins += 1;
    }
    let target = PARTIAL_FRAMES + frame_step * n_wins.saturating_sub(1);
    (n_wins.max(1), target)
}

/// Linear resampler. Prefer feeding 16 kHz audio; see the module note.
pub fn linear_resample(wav: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == 0 || to == 0 || wav.is_empty() || from == to {
        return wav.to_vec();
    }
    let n = ((wav.len() as u64 * to as u64) / from as u64).max(1) as usize;
    let mut out = Vec::with_capacity(n);
    let scale = from as f64 / to as f64;
    for i in 0..n {
        let x = i as f64 * scale;
        let i0 = (x.floor() as usize).min(wav.len() - 1);
        let i1 = (i0 + 1).min(wav.len() - 1);
        let f = (x - i0 as f64) as f32;
        out.push(wav[i0] * (1.0 - f) + wav[i1] * f);
    }
    out
}

/// Drop leading and trailing frames whose RMS is more than `top_db` below the peak RMS.
pub fn trim_top_db(wav: &[f32], top_db: f64) -> Vec<f32> {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    if wav.is_empty() {
        return Vec::new();
    }
    let pad = FRAME / 2;
    let n_frames = 1 + wav.len() / HOP;
    let mut rms = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * HOP;
        let mut acc = 0f32;
        for n in 0..FRAME {
            let idx = start as isize + n as isize - pad as isize;
            let s = if idx < 0 || idx >= wav.len() as isize {
                0.0
            } else {
                wav[idx as usize]
            };
            acc += s * s;
        }
        rms.push((acc / FRAME as f32).sqrt());
    }
    let peak = rms.iter().copied().fold(0f32, f32::max);
    if peak == 0.0 {
        return Vec::new();
    }
    let floor = peak * 10f32.powf((-top_db / 20.0) as f32);
    let mut first = None;
    let mut last = 0;
    for (i, r) in rms.iter().enumerate() {
        if *r > floor {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
        }
    }
    let Some(first) = first else {
        return Vec::new();
    };
    let start = first * HOP;
    let end = ((last + 1) * HOP).min(wav.len());
    if start >= end {
        return Vec::new();
    }
    wav[start..end].to_vec()
}

fn reflect(wav: &[f32], index: isize) -> f32 {
    let n = wav.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return wav[0];
    }
    let period = ((n - 1) * 2) as isize;
    let mut i = index.rem_euclid(period);
    if i >= n as isize {
        i = period - i;
    }
    wav[i as usize]
}

fn hann() -> [f32; N_FFT] {
    let mut w = [0f32; N_FFT];
    for (n, slot) in w.iter_mut().enumerate() {
        let s = (std::f64::consts::PI * n as f64 / N_FFT as f64).sin();
        *slot = (s * s) as f32;
    }
    w
}

fn hz_to_mel(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        mel * f_sp
    }
}

fn mel_basis() -> [[f32; N_FREQ]; N_MELS] {
    let mut mel_f = [0f64; N_MELS + 2];
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(8_000.0);
    for (i, slot) in mel_f.iter_mut().enumerate() {
        let t = i as f64 / (N_MELS + 1) as f64;
        *slot = mel_to_hz(mel_min + (mel_max - mel_min) * t);
    }
    let mut weights = [[0f32; N_FREQ]; N_MELS];
    for i in 0..N_MELS {
        let fdiff_lo = mel_f[i + 1] - mel_f[i];
        let fdiff_hi = mel_f[i + 2] - mel_f[i + 1];
        let enorm = (2.0 / (mel_f[i + 2] - mel_f[i])) as f32;
        for k in 0..N_FREQ {
            let fft = k as f64 * SAMPLE_RATE as f64 / N_FFT as f64;
            let lower = -(mel_f[i] - fft) / fdiff_lo;
            let upper = (mel_f[i + 2] - fft) / fdiff_hi;
            let w = lower.min(upper).max(0.0) as f32 * enorm;
            weights[i][k] = w;
        }
    }
    weights
}

/// Power mel spectrogram, shape `(frames, N_MELS)`, time-major. Matches
/// `melspectrogram(wav).T` in the voice encoder.
pub fn melspectrogram(wav: &[f32]) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let window = hann();
    let basis = mel_basis();
    let n_frames = 1 + wav.len() / HOP;
    let mut mel = vec![0f32; n_frames * N_MELS];
    for t in 0..n_frames {
        let start = t as isize * HOP as isize - (N_FFT / 2) as isize;
        let mut re = [0f64; N_FREQ];
        let mut im = [0f64; N_FREQ];
        for n in 0..N_FFT {
            let s = reflect(wav, start + n as isize) as f64 * window[n] as f64;
            for k in 0..N_FREQ {
                let ang = std::f64::consts::TAU * (k * n) as f64 / N_FFT as f64;
                re[k] += s * ang.cos();
                im[k] -= s * ang.sin();
            }
        }
        for m in 0..N_MELS {
            let mut acc = 0f64;
            for k in 0..N_FREQ {
                let power = re[k] * re[k] + im[k] * im[k];
                acc += basis[m][k] as f64 * power;
            }
            mel[t * N_MELS + m] = acc as f32;
        }
    }
    mel
}

/// Overlapping partials, each `PARTIAL_FRAMES * N_MELS`, row-major time-major.
/// Pads with zeros or trims, the same way `VoiceEncoder.inference` does.
pub fn partials(mel_tm: &[f32], rate: f64) -> Vec<Vec<f32>> {
    let n_frames = mel_tm.len() / N_MELS;
    if n_frames == 0 {
        return Vec::new();
    }
    let step = partial_frame_step(rate);
    let (n_partials, target) = partial_windows(n_frames, step);
    let mut padded = vec![0f32; target * N_MELS];
    let copy_frames = n_frames.min(target);
    padded[..copy_frames * N_MELS].copy_from_slice(&mel_tm[..copy_frames * N_MELS]);
    let mut out = Vec::with_capacity(n_partials);
    for i in 0..n_partials {
        let start = i * step * N_MELS;
        out.push(padded[start..start + PARTIAL_FRAMES * N_MELS].to_vec());
    }
    out
}

/// 16 kHz waveform in, partial mels out. Applies the 20 dB trim upstream uses.
pub fn prepare_wav(wav: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
    let wav = if sample_rate == SAMPLE_RATE {
        wav.to_vec()
    } else {
        linear_resample(wav, sample_rate, SAMPLE_RATE)
    };
    let wav = trim_top_db(&wav, 20.0);
    if wav.is_empty() {
        return Vec::new();
    }
    partials(&melspectrogram(&wav), DEFAULT_PARTIAL_RATE)
}

/// Harmonic stack used by the CPU tests and by `tools/ve_reference.py`.
pub fn synth_harmonic(f0: f32, seconds: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE as f32).round() as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let mut s = 0.0;
        for k in 1..=8 {
            s += (1.0 / k as f32) * (std::f32::consts::TAU * f0 * k as f32 * t).sin();
        }
        out.push(s * 0.15);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_step_matches_numpy_round() {
        assert_eq!(partial_frame_step(1.3), 77);
    }

    #[test]
    fn short_clip_is_one_padded_partial() {
        let (n, target) = partial_windows(10, 77);
        assert_eq!((n, target), (1, 160));
    }

    #[test]
    fn three_second_harmonic_has_the_librosa_frame_count() {
        let wav = synth_harmonic(140.0, 3.0);
        assert_eq!(wav.len(), 48_000);
        let mel = melspectrogram(&wav);
        let frames = mel.len() / N_MELS;
        assert_eq!(frames, 1 + wav.len() / HOP);
        let parts = partials(&mel, DEFAULT_PARTIAL_RATE);
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().all(|p| p.len() == PARTIAL_FRAMES * N_MELS));
        assert!(mel.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn trim_drops_leading_silence() {
        let mut wav = vec![0f32; 8_000];
        wav.extend(synth_harmonic(180.0, 1.0));
        let trimmed = trim_top_db(&wav, 20.0);
        assert!(trimmed.len() < wav.len());
        assert!(trimmed.len() > 8_000);
    }

    #[test]
    fn reflect_pad_matches_numpy() {
        let wav = [1f32, 2.0, 3.0, 4.0];
        assert_eq!(reflect(&wav, -1), 2.0);
        assert_eq!(reflect(&wav, 4), 3.0);
        assert_eq!(reflect(&wav, 0), 1.0);
    }
}
