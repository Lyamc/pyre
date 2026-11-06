//! Audio front ends the mean-flow decoder consumes.
//!
//! Two spectrograms, and they are not interchangeable:
//!
//! * [`kaldi_fbank`] is the 16 kHz filterbank CAMPPlus embeds. It follows
//!   torchaudio's Kaldi recipe with the defaults `extract_feature` actually
//!   passes: 25 ms / 10 ms, snip edges, per-frame mean removal, preemphasis
//!   0.97, a Povey window (Hann to the power 0.85), 80 mel bins, log, then a
//!   mean over time. Dither stays 0, which is what torchaudio 2.11 defaults
//!   to and what makes the embedding a function of the clip.
//! * [`prompt_mel_24k`] is the CosyVoice / chatterbox prompt mel. 24 kHz,
//!   n_fft 1920, hop 480, periodic Hann, reflect pad of 720, magnitude with
//!   a 1e-9 floor, Slaney 80-bin filters (`fmin` 0, `fmax` 8000), log clamp
//!   1e-5. The flow model wants this time-major, shape `(frames, 80)`, and
//!   the frame count is `samples / 480`.

const FBANK_SR: f64 = 16_000.0;
const FBANK_WIN: usize = 400;
const FBANK_SHIFT: usize = 160;
const FBANK_FFT: usize = 512;
const FBANK_BINS: usize = 80;

const MEL_SR: f64 = 24_000.0;
const MEL_FFT: usize = 1920;
const MEL_HOP: usize = 480;
const MEL_BINS: usize = 80;
const MEL_PAD: usize = (MEL_FFT - MEL_HOP) / 2;

/// Kaldi fbank, time-major `(frames, 80)`, mean subtracted over time.
pub fn kaldi_fbank(wav: &[f32]) -> Vec<f32> {
    if wav.len() < FBANK_WIN {
        return Vec::new();
    }
    let frames = 1 + (wav.len() - FBANK_WIN) / FBANK_SHIFT;
    let banks = kaldi_banks();
    let mut out = vec![0f32; frames * FBANK_BINS];
    let mut windowed = vec![0f32; FBANK_FFT];
    for t in 0..frames {
        let start = t * FBANK_SHIFT;
        let frame = &wav[start..start + FBANK_WIN];
        let mean = frame.iter().sum::<f32>() / FBANK_WIN as f32;
        // Replicate-pad the first sample, then preemphasize.
        let mut prev = frame[0] - mean;
        for n in 0..FBANK_WIN {
            let x = frame[n] - mean;
            let y = x - 0.97 * prev;
            prev = x;
            let w = {
                let s = (std::f64::consts::PI * n as f64 / (FBANK_WIN as f64 - 1.0)).sin();
                (s * s).powf(0.85)
            };
            windowed[n] = y * w as f32;
        }
        for n in FBANK_WIN..FBANK_FFT {
            windowed[n] = 0.0;
        }
        let power = rfft_power(&windowed);
        let eps = f32::EPSILON;
        for bin in 0..FBANK_BINS {
            let mut e = 0.0f32;
            for k in 0..FBANK_FFT / 2 {
                e += power[k] * banks[bin][k];
            }
            out[t * FBANK_BINS + bin] = e.max(eps).ln();
        }
    }
    let mut mean = vec![0f32; FBANK_BINS];
    for t in 0..frames {
        for bin in 0..FBANK_BINS {
            mean[bin] += out[t * FBANK_BINS + bin];
        }
    }
    for bin in 0..FBANK_BINS {
        mean[bin] /= frames as f32;
    }
    for t in 0..frames {
        for bin in 0..FBANK_BINS {
            out[t * FBANK_BINS + bin] -= mean[bin];
        }
    }
    out
}

/// 24 kHz prompt mel, time-major `(frames, 80)`. `frames = wav.len() / 480`
/// when the length is a multiple of the hop.
pub fn prompt_mel_24k(wav: &[f32]) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let padded_len = wav.len() + 2 * MEL_PAD;
    let frames = 1 + (padded_len - MEL_FFT) / MEL_HOP;
    let basis = slaney_80();
    let mut out = vec![0f32; frames * MEL_BINS];
    let mut frame = vec![0f32; MEL_FFT];
    for t in 0..frames {
        let start = t * MEL_HOP;
        for n in 0..MEL_FFT {
            let idx = start as isize + n as isize - MEL_PAD as isize;
            let s = sample_reflect(wav, idx);
            let w = {
                let s = (std::f64::consts::PI * n as f64 / MEL_FFT as f64).sin();
                (s * s) as f32
            };
            frame[n] = s * w;
        }
        let power = rfft_mag(&frame);
        for bin in 0..MEL_BINS {
            let mut e = 0.0f32;
            for k in 0..MEL_FFT / 2 + 1 {
                e += power[k] * basis[bin][k];
            }
            out[t * MEL_BINS + bin] = e.max(1e-5).ln();
        }
    }
    out
}

/// `F.pad(..., mode="reflect")` does not repeat the edge sample.
fn sample_reflect(wav: &[f32], i: isize) -> f32 {
    let n = wav.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return wav[0];
    }
    let period = (2 * (n - 1)) as isize;
    let mut j = i.rem_euclid(period);
    if j >= n as isize {
        j = period - j;
    }
    wav[j as usize]
}

fn rfft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let bins = n / 2;
    let mut out = vec![0f32; bins];
    for k in 0..bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, s) in frame.iter().enumerate() {
            let ang = std::f64::consts::TAU * (t * k) as f64 / n as f64;
            re += *s as f64 * ang.cos();
            im -= *s as f64 * ang.sin();
        }
        out[k] = (re * re + im * im) as f32;
    }
    out
}

fn rfft_mag(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let bins = n / 2 + 1;
    let mut out = vec![0f32; bins];
    for k in 0..bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, s) in frame.iter().enumerate() {
            let ang = std::f64::consts::TAU * (t * k) as f64 / n as f64;
            re += *s as f64 * ang.cos();
            im -= *s as f64 * ang.sin();
        }
        out[k] = (re * re + im * im + 1e-9).sqrt() as f32;
    }
    out
}

fn kaldi_banks() -> [[f32; FBANK_FFT / 2]; FBANK_BINS] {
    // Kaldi mel: 1127 ln(1 + f/700). num_fft_bins is padded/2 (the Nyquist
    // column is padded with zeros and never contributes).
    let n_fft = FBANK_FFT;
    let num_fft_bins = n_fft / 2;
    let low = 20.0f64;
    let high = FBANK_SR / 2.0;
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let mel_low = mel(low);
    let mel_high = mel(high);
    let delta = (mel_high - mel_low) / (FBANK_BINS as f64 + 1.0);
    let bin_width = FBANK_SR / n_fft as f64;
    let mut banks = [[0f32; FBANK_FFT / 2]; FBANK_BINS];
    for b in 0..FBANK_BINS {
        let left = mel_low + b as f64 * delta;
        let center = mel_low + (b as f64 + 1.0) * delta;
        let right = mel_low + (b as f64 + 2.0) * delta;
        for k in 0..num_fft_bins {
            let m = mel(bin_width * k as f64);
            let up = (m - left) / (center - left);
            let down = (right - m) / (right - center);
            banks[b][k] = up.min(down).max(0.0) as f32;
        }
    }
    banks
}

fn slaney_80() -> [[f32; MEL_FFT / 2 + 1]; MEL_BINS] {
    let n_freq = MEL_FFT / 2 + 1;
    let mut mel_f = [0f64; MEL_BINS + 2];
    let mel_min = slaney_hz_to_mel(0.0);
    let mel_max = slaney_hz_to_mel(8_000.0);
    for (i, slot) in mel_f.iter_mut().enumerate() {
        let t = i as f64 / (MEL_BINS + 1) as f64;
        *slot = slaney_mel_to_hz(mel_min + (mel_max - mel_min) * t);
    }
    let mut weights = [[0f32; MEL_FFT / 2 + 1]; MEL_BINS];
    for i in 0..MEL_BINS {
        let fdiff_lo = mel_f[i + 1] - mel_f[i];
        let fdiff_hi = mel_f[i + 2] - mel_f[i + 1];
        let enorm = (2.0 / (mel_f[i + 2] - mel_f[i])) as f32;
        for k in 0..n_freq {
            let fft = k as f64 * MEL_SR / MEL_FFT as f64;
            let lower = -(mel_f[i] - fft) / fdiff_lo;
            let upper = (mel_f[i + 2] - fft) / fdiff_hi;
            weights[i][k] = lower.min(upper).max(0.0) as f32 * enorm;
        }
    }
    weights
}

fn slaney_hz_to_mel(freq: f64) -> f64 {
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

fn slaney_mel_to_hz(mel: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(name: &str) -> Vec<f32> {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("testdata")
                .join(name),
        )
        .unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn kaldi_fbank_matches_torchaudio_on_a_tone() {
        let wav = f32s("camp_tone.f32");
        let want = f32s("camp_tone_fbank.f32");
        let got = kaldi_fbank(&wav);
        assert_eq!(got.len(), want.len());
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-4, "fbank max abs {max}");
    }

    #[test]
    fn prompt_mel_matches_librosa_on_a_tone() {
        let wav = f32s("prompt_tone.f32");
        let want = f32s("prompt_tone_mel.f32");
        let got = prompt_mel_24k(&wav);
        assert_eq!(got.len(), want.len());
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let loud = got
            .iter()
            .zip(&want)
            .filter(|(_, b)| **b > -5.0)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(loud < 1e-4, "prompt mel loud-bin max abs {loud}");
        assert!(max < 5e-3, "prompt mel max abs {max}");
    }
}
