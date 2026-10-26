use serde::{Deserialize, Serialize};

use pyre_whisper_core::WhisburnError;

pub const TARGET_SAMPLE_RATE: usize = 16_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Waveform {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
}

impl Waveform {
    pub fn new(samples: Vec<f32>, sample_rate: usize) -> Self {
        Self { samples, sample_rate }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate as f64
        }
    }

    pub fn ensure_sample_rate(self, target: usize) -> Result<Self, WhisburnError> {
        if self.sample_rate == target {
            return Ok(self);
        }
        let samples = crate::resample::resample_mono(&self.samples, self.sample_rate, target)?;
        Ok(Self {
            samples,
            sample_rate: target,
        })
    }

    pub fn slice(&self, start_secs: f64, end_secs: f64) -> Result<Self, WhisburnError> {
        let start = (start_secs * self.sample_rate as f64).max(0.0) as usize;
        let end = (end_secs * self.sample_rate as f64).min(self.samples.len() as f64) as usize;
        if start >= end {
            return Err(WhisburnError::Audio("invalid slice range".into()));
        }
        Ok(Self {
            samples: self.samples[start..end].to_vec(),
            sample_rate: self.sample_rate,
        })
    }
}