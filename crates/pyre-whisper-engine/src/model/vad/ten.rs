//! TEN-VAD: 40-bin log-mel (+ pitch) over 3 frames → 2-layer LSTM(64) → sigmoid.

use super::host::{lstm_step, relu, sigmoid};
use super::segments::{probs_to_segments, VadSegment, VadSettings};
use super::weights::{load_f32, load_named, TensorMap};
use serde::{Deserialize, Serialize};

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 1024;
const N_MELS: usize = 40;
const FEAT: usize = 41;
const HIDDEN: usize = 64;
const DEFAULT_HOP: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenFeatures {
    pub mean: Vec<f32>,
    pub inv_stddev: Vec<f32>,
    pub window: Vec<f32>,
    #[serde(default = "default_hop")]
    pub hop_size: usize,
}

fn default_hop() -> usize {
    DEFAULT_HOP
}

pub struct TenVad {
    features: TenFeatures,
    mel: Vec<f32>, // [40, n_fft/2+1]
    lstm0_w_ih: Vec<f32>,
    lstm0_w_hh: Vec<f32>,
    lstm0_b_ih: Vec<f32>,
    lstm0_b_hh: Vec<f32>,
    lstm1_w_ih: Vec<f32>,
    lstm1_w_hh: Vec<f32>,
    lstm1_b_ih: Vec<f32>,
    lstm1_b_hh: Vec<f32>,
    dense_w: Vec<f32>,
    dense_b: Vec<f32>,
    fc_w: Vec<f32>,
    fc_b: f32,
}

impl TenVad {
    pub fn from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let feat: TenFeatures = serde_json::from_str(
            &std::fs::read_to_string(dir.join("ten_features.json"))
                .map_err(|e| format!("ten_features.json: {e}"))?,
        )
        .map_err(|e| format!("parse ten_features.json: {e}"))?;
        let bytes = std::fs::read(dir.join("model.safetensors"))
            .map_err(|e| format!("ten model.safetensors: {e}"))?;
        let map = load_f32(&bytes)?;
        Self::from_tensors(feat, &map)
    }

    pub fn from_tensors(features: TenFeatures, map: &TensorMap) -> Result<Self, String> {
        let (fc_w, _) = load_named(map, "fc.weight")?;
        let fc_b = load_named(map, "fc.bias").map(|(v, _)| v[0]).unwrap_or(0.0);
        let (dense_w, _) = load_named(map, "dense.weight").unwrap_or((vec![0f32; 32 * 128], vec![]));
        let dense_b = load_named(map, "dense.bias")
            .map(|(v, _)| v)
            .unwrap_or_else(|_| vec![0f32; 32]);
        Ok(Self {
            mel: slaney_mel_banks(SAMPLE_RATE as f32, N_FFT, N_MELS),
            lstm0_w_ih: lstm_w(map, 0, true, 80)?,
            lstm0_w_hh: lstm_w(map, 0, false, 64)?,
            lstm0_b_ih: lstm_b(map, 0, true)?,
            lstm0_b_hh: lstm_b(map, 0, false)?,
            lstm1_w_ih: lstm_w(map, 1, true, 64)?,
            lstm1_w_hh: lstm_w(map, 1, false, 64)?,
            lstm1_b_ih: lstm_b(map, 1, true)?,
            lstm1_b_hh: lstm_b(map, 1, false)?,
            dense_w,
            dense_b,
            fc_w,
            fc_b,
            features,
        })
    }

    fn frame_feat(&self, samples: &[f32], last_sample: &mut f32) -> [f32; FEAT] {
        let hop = self.features.hop_size.max(1).min(768);
        let mut buf = vec![0f32; N_FFT];
        let n = samples.len().min(hop).min(N_FFT);
        for i in 0..n {
            buf[i] = samples[i] * 32768.0;
        }
        let orig_last = buf[n.saturating_sub(1)];
        for i in (1..n).rev() {
            buf[i] -= 0.97 * buf[i - 1];
        }
        if n > 0 {
            buf[0] -= 0.97 * *last_sample;
        }
        *last_sample = orig_last;
        let win_n = self.features.window.len().min(n);
        for i in 0..win_n {
            buf[i] *= self.features.window[i];
        }
        let spec = rfft_power(&buf);
        let mut mel = [0f32; FEAT];
        let n_freq = N_FFT / 2 + 1;
        for m in 0..N_MELS {
            let mut acc = 0.0;
            for f in 0..n_freq {
                acc += spec[f] * self.mel[m * n_freq + f];
            }
            mel[m] = (acc + 1e-10).ln() - 20.79441541679836;
        }
        mel[40] = 0.0;
        for i in 0..FEAT {
            let mean = self.features.mean.get(i).copied().unwrap_or(0.0);
            let inv = self.features.inv_stddev.get(i).copied().unwrap_or(1.0);
            mel[i] = (mel[i] - mean) * inv;
        }
        mel
    }

    pub fn speech_probs(&self, samples: &[f32], sample_rate: usize) -> Vec<f32> {
        let audio = if sample_rate == SAMPLE_RATE {
            samples.to_vec()
        } else {
            pyre_whisper_audio::resample_mono(samples, sample_rate, SAMPLE_RATE)
                .unwrap_or_else(|_| samples.to_vec())
        };
        let hop = self.features.hop_size.max(1);
        let mut last_sample = 0.0f32;
        let mut hist = vec![[0f32; FEAT]; 3];
        let mut h0 = vec![0f32; HIDDEN];
        let mut c0 = vec![0f32; HIDDEN];
        let mut h1 = vec![0f32; HIDDEN];
        let mut c1 = vec![0f32; HIDDEN];
        let mut probs = Vec::new();
        let mut i = 0usize;
        while i < audio.len() {
            let n = (audio.len() - i).min(hop);
            let feat = self.frame_feat(&audio[i..i + n], &mut last_sample);
            hist.remove(0);
            hist.push(feat);
            let mut x80 = vec![0f32; 80];
            x80[..FEAT.min(80)].copy_from_slice(&hist[2][..FEAT.min(80)]);
            lstm_step(
                &x80,
                &mut h0,
                &mut c0,
                &self.lstm0_w_ih,
                &self.lstm0_w_hh,
                &self.lstm0_b_ih,
                &self.lstm0_b_hh,
            );
            lstm_step(
                &h0,
                &mut h1,
                &mut c1,
                &self.lstm1_w_ih,
                &self.lstm1_w_hh,
                &self.lstm1_b_ih,
                &self.lstm1_b_hh,
            );
            let mut cat = h0.clone();
            cat.extend_from_slice(&h1);
            let mut hid = vec![0f32; 32];
            let din = 128.min(cat.len());
            for o in 0..32 {
                let mut acc = self.dense_b.get(o).copied().unwrap_or(0.0);
                let row = o * 128;
                if row + din <= self.dense_w.len() {
                    for i in 0..din {
                        acc += cat[i] * self.dense_w[row + i];
                    }
                }
                hid[o] = acc;
            }
            relu(&mut hid);
            let mut logit = self.fc_b;
            for j in 0..32.min(self.fc_w.len()) {
                logit += hid[j] * self.fc_w[j];
            }
            probs.push(sigmoid(logit));
            i += hop;
        }
        probs
    }

    pub fn segments(&self, samples: &[f32], sample_rate: usize) -> Vec<VadSegment> {
        let probs = self.speech_probs(samples, sample_rate);
        let hop = self.features.hop_size.max(1) as f64 / SAMPLE_RATE as f64;
        probs_to_segments(&probs, hop, VadSettings::ten())
    }
}

fn lstm_w(map: &TensorMap, layer: usize, ih: bool, expected_in: usize) -> Result<Vec<f32>, String> {
    let kind = if ih { "weight_ih_l" } else { "weight_hh_l" };
    let name = format!("lstm.{kind}{layer}");
    load_named(map, &name)
        .or_else(|_| load_named(map, &format!("rnn.{kind}{layer}")))
        .or_else(|_| find_lstm_weight(map, layer, ih, expected_in))
        .map(|(d, _)| d)
}

fn lstm_b(map: &TensorMap, layer: usize, ih: bool) -> Result<Vec<f32>, String> {
    let kind = if ih { "bias_ih_l" } else { "bias_hh_l" };
    load_named(map, &format!("lstm.{kind}{layer}"))
        .or_else(|_| load_named(map, &format!("rnn.{kind}{layer}")))
        .or_else(|_| Ok((vec![0f32; 4 * HIDDEN], vec![4 * HIDDEN])))
        .map(|(d, _)| {
            if d.len() == 4 * HIDDEN {
                d
            } else {
                vec![0f32; 4 * HIDDEN]
            }
        })
}

fn find_lstm_weight(
    map: &TensorMap,
    _layer: usize,
    ih: bool,
    expected_in: usize,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let want = [4 * HIDDEN, expected_in];
    find_by_shape(map, &want).or_else(|_| {
        if ih {
            Err("lstm weight_ih not found".into())
        } else {
            find_by_shape(map, &[4 * HIDDEN, HIDDEN])
        }
    })
}

fn find_by_shape(map: &TensorMap, shape: &[usize]) -> Result<(Vec<f32>, Vec<usize>), String> {
    map.values()
        .find(|(_, s)| s.as_slice() == shape)
        .cloned()
        .ok_or_else(|| format!("no tensor with shape {shape:?}"))
}

fn slaney_mel_banks(sr: f32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let fmax = sr / 2.0;
    let mut mel = vec![0f32; n_mels + 2];
    for (i, m) in mel.iter_mut().enumerate() {
        *m = hz_to_mel(fmax * i as f32 / (n_mels + 1) as f32);
    }
    let hz: Vec<f32> = mel.iter().map(|&m| mel_to_hz(m)).collect();
    let mut w = vec![0f32; n_mels * n_freq];
    for m in 0..n_mels {
        let left = hz[m];
        let center = hz[m + 1];
        let right = hz[m + 2];
        let enorm = 2.0 / (right - left).max(1e-8);
        for f in 0..n_freq {
            let freq = sr * f as f32 / n_fft as f32;
            let lo = (freq - left) / (center - left).max(1e-8);
            let hi = (right - freq) / (right - center).max(1e-8);
            w[m * n_freq + f] = lo.min(hi).max(0.0) * enorm;
        }
    }
    w
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

fn rfft_power(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut re: Vec<f32> = x.to_vec();
    let mut im = vec![0f32; n];
    fft_radix2(&mut re, &mut im);
    let mut p = vec![0f32; n / 2 + 1];
    p[0] = re[0] * re[0];
    p[n / 2] = re[n / 2] * re[n / 2];
    for i in 1..n / 2 {
        p[i] = re[i] * re[i] + im[i] * im[i];
    }
    p
}

fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wlen_re, wlen_im) = (ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for k in 0..half {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + half] * w_re - im[i + k + half] * w_im;
                let vi = re[i + k + half] * w_im + im[i + k + half] * w_re;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
                let nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
        len *= 2;
    }
}
