//! Silero VAD v5: learned STFT → 4×Conv1d+ReLU → LSTM(128) → sigmoid.

use super::host::{conv1d, lstm_step, reflect_pad_right, relu, sigmoid};
use super::segments::{probs_to_segments, VadSegment, VadSettings};
use crate::model::vad::weights::{load_f32, load_named, TensorMap};

const WINDOW: usize = 512;
const CONTEXT: usize = 64;
const HIDDEN: usize = 128;
const SAMPLE_RATE: usize = 16_000;

pub struct SileroVad {
    stft_w: Vec<f32>,
    enc0_w: Vec<f32>,
    enc0_b: Vec<f32>,
    enc1_w: Vec<f32>,
    enc1_b: Vec<f32>,
    enc2_w: Vec<f32>,
    enc2_b: Vec<f32>,
    enc3_w: Vec<f32>,
    enc3_b: Vec<f32>,
    lstm_w_ih: Vec<f32>,
    lstm_w_hh: Vec<f32>,
    lstm_b_ih: Vec<f32>,
    lstm_b_hh: Vec<f32>,
    dec_w: Vec<f32>,
    dec_b: f32,
}

impl SileroVad {
    pub fn from_tensors(map: &TensorMap) -> Result<Self, String> {
        let stft_w = conv_weight(map, "stft.conv.weight", 258, 1, 256)?;
        Ok(Self {
            stft_w,
            enc0_w: conv_weight(map, "encoder.0.conv.weight", 128, 129, 3)?,
            enc0_b: load_named(map, "encoder.0.conv.bias")?.0,
            enc1_w: conv_weight(map, "encoder.1.conv.weight", 64, 128, 3)?,
            enc1_b: load_named(map, "encoder.1.conv.bias")?.0,
            enc2_w: conv_weight(map, "encoder.2.conv.weight", 64, 64, 3)?,
            enc2_b: load_named(map, "encoder.2.conv.bias")?.0,
            enc3_w: conv_weight(map, "encoder.3.conv.weight", 128, 64, 3)?,
            enc3_b: load_named(map, "encoder.3.conv.bias")?.0,
            lstm_w_ih: load_named(map, "decoder.lstm.weight_ih_l0")?.0,
            lstm_w_hh: load_named(map, "decoder.lstm.weight_hh_l0")?.0,
            lstm_b_ih: load_named(map, "decoder.lstm.bias_ih_l0")?.0,
            lstm_b_hh: load_named(map, "decoder.lstm.bias_hh_l0")?.0,
            dec_w: load_named(map, "decoder.output.weight")?.0,
            dec_b: load_named(map, "decoder.output.bias")?.0[0],
        })
    }

    pub fn from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let path = dir.join("model.safetensors");
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let map = load_f32(&bytes)?;
        Self::from_tensors(&map)
    }

    fn forward_chunk(&self, chunk576: &[f32], h: &mut [f32], c: &mut [f32]) -> f32 {
        let padded = reflect_pad_right(chunk576, CONTEXT);
        let (stft, t) = conv1d(&padded, 1, padded.len(), &self.stft_w, None, 258, 256, 128, 0);
        let mut mag = vec![0f32; 129 * t];
        for ti in 0..t {
            for f in 0..129 {
                let re = stft[f * t + ti];
                let im = stft[(f + 129) * t + ti];
                mag[f * t + ti] = (re * re + im * im).sqrt();
            }
        }
        let (mut x, t) = conv1d(&mag, 129, t, &self.enc0_w, Some(&self.enc0_b), 128, 3, 1, 1);
        relu(&mut x);
        let (mut x, t) = conv1d(&x, 128, t, &self.enc1_w, Some(&self.enc1_b), 64, 3, 2, 1);
        relu(&mut x);
        let (mut x, t) = conv1d(&x, 64, t, &self.enc2_w, Some(&self.enc2_b), 64, 3, 2, 1);
        relu(&mut x);
        let (mut x, t) = conv1d(&x, 64, t, &self.enc3_w, Some(&self.enc3_b), 128, 3, 1, 1);
        relu(&mut x);
        let mut feat = vec![0f32; 128];
        for ch in 0..128 {
            let mut acc = 0.0;
            for ti in 0..t.max(1) {
                acc += x[ch * t.max(1) + ti.min(t.saturating_sub(1))];
            }
            feat[ch] = acc / t.max(1) as f32;
        }
        lstm_step(
            &feat,
            h,
            c,
            &self.lstm_w_ih,
            &self.lstm_w_hh,
            &self.lstm_b_ih,
            &self.lstm_b_hh,
        );
        let mut act = h.to_vec();
        relu(&mut act);
        let mut logit = self.dec_b;
        let n = HIDDEN.min(self.dec_w.len());
        for i in 0..n {
            logit += act[i] * self.dec_w[i];
        }
        sigmoid(logit)
    }

    pub fn speech_probs(&self, samples: &[f32], sample_rate: usize) -> Vec<f32> {
        let audio = if sample_rate == SAMPLE_RATE {
            samples.to_vec()
        } else {
            pyre_whisper_audio::resample_mono(samples, sample_rate, SAMPLE_RATE)
                .unwrap_or_else(|_| samples.to_vec())
        };
        let mut context = vec![0f32; CONTEXT];
        let mut h = vec![0f32; HIDDEN];
        let mut c = vec![0f32; HIDDEN];
        let mut probs = Vec::new();
        let mut i = 0usize;
        while i < audio.len() {
            let mut chunk = vec![0f32; WINDOW];
            let n = (audio.len() - i).min(WINDOW);
            chunk[..n].copy_from_slice(&audio[i..i + n]);
            let mut packed = Vec::with_capacity(CONTEXT + WINDOW);
            packed.extend_from_slice(&context);
            packed.extend_from_slice(&chunk);
            let p = self.forward_chunk(&packed, &mut h, &mut c);
            probs.push(p);
            context.copy_from_slice(&chunk[WINDOW - CONTEXT..]);
            i += WINDOW;
        }
        probs
    }

    pub fn segments(&self, samples: &[f32], sample_rate: usize) -> Vec<VadSegment> {
        let probs = self.speech_probs(samples, sample_rate);
        let hop = WINDOW as f64 / SAMPLE_RATE as f64;
        probs_to_segments(&probs, hop, VadSettings::silero())
    }
}

fn conv_weight(
    map: &TensorMap,
    name: &str,
    out_c: usize,
    in_c: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let (data, shape) = load_named(map, name)?;
    // Accept [O,I,K], [O,K,I], or [O,I*K] flattened.
    if shape == [out_c, in_c, k] {
        return Ok(data);
    }
    if shape == [out_c, k, in_c] {
        let mut w = vec![0f32; out_c * in_c * k];
        for o in 0..out_c {
            for i in 0..in_c {
                for kk in 0..k {
                    w[o * in_c * k + i * k + kk] = data[o * k * in_c + kk * in_c + i];
                }
            }
        }
        return Ok(w);
    }
    if data.len() == out_c * in_c * k {
        return Ok(data);
    }
    Err(format!("{name}: expected conv [{out_c},{in_c},{k}], got {shape:?}"))
}
