use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeVoiceRuntimeConfig {
    pub burn_bundle_version: String,
    pub sample_rate: usize,
    pub speech_compress_ratio: usize,
    pub target_dbfs: f32,
    pub acoustic_vae_dim: usize,
    pub semantic_vae_dim: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub audio_bos_token: String,
    pub audio_eos_token: String,
    pub audio_pad_token: String,
    pub inference_status: String,
}

fn default_intermediate_size() -> usize {
    18_944
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}

impl Default for VibeVoiceRuntimeConfig {
    fn default() -> Self {
        Self {
            burn_bundle_version: "0.21.0-vibevoice-stt".to_string(),
            sample_rate: 24_000,
            speech_compress_ratio: 3200,
            target_dbfs: -25.0,
            acoustic_vae_dim: 64,
            semantic_vae_dim: 128,
            hidden_size: 3584,
            vocab_size: 152_064,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 131_072,
            intermediate_size: 18_944,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            audio_bos_token: "<|object_ref_start|>".to_string(),
            audio_eos_token: "<|object_ref_end|>".to_string(),
            audio_pad_token: "<|box_start|>".to_string(),
            inference_status: "ready".to_string(),
        }
    }
}

pub fn load_vibevoice_runtime(model_name: &str) -> VibeVoiceRuntimeConfig {
    let path = pyre_whisper_core::resolve_model_file(model_name, "vibevoice_runtime.json");
    Path::new(&path)
        .exists()
        .then(|| std::fs::read_to_string(&path).ok())
        .flatten()
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}