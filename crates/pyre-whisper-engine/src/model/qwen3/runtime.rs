use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3RuntimeConfig {
    pub burn_bundle_version: String,
    pub sample_rate: usize,
    pub num_mel_bins: usize,
    pub audio_d_model: usize,
    pub audio_layers: usize,
    pub audio_heads: usize,
    pub audio_ffn_dim: usize,
    pub audio_output_dim: usize,
    pub downsample_hidden_size: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub conv_chunksize: usize,
    pub max_source_positions: usize,
    pub text_hidden_size: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub text_kv_heads: usize,
    pub text_head_dim: usize,
    pub text_intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    pub audio_start_token_id: usize,
    pub audio_end_token_id: usize,
    pub audio_pad_token_id: usize,
    pub im_start_token_id: usize,
    pub im_end_token_id: usize,
    pub asr_text_token_id: usize,
    pub eos_token_ids: Vec<usize>,
    pub inference_status: String,
    #[serde(default)]
    pub text_only: bool,
}

impl Default for Qwen3RuntimeConfig {
    fn default() -> Self {
        Self {
            burn_bundle_version: "0.21.0-qwen3".to_string(),
            sample_rate: 16_000,
            num_mel_bins: 128,
            audio_d_model: 896,
            audio_layers: 18,
            audio_heads: 14,
            audio_ffn_dim: 3584,
            audio_output_dim: 1024,
            downsample_hidden_size: 480,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            max_source_positions: 1500,
            text_hidden_size: 1024,
            text_layers: 28,
            text_heads: 16,
            text_kv_heads: 8,
            text_head_dim: 128,
            text_intermediate_size: 3072,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            mrope_section: vec![24, 20, 20],
            audio_start_token_id: 151_669,
            audio_end_token_id: 151_670,
            audio_pad_token_id: 151_676,
            im_start_token_id: 151_644,
            im_end_token_id: 151_645,
            asr_text_token_id: 151_704,
            eos_token_ids: vec![151_643, 151_645],
            inference_status: "ready".to_string(),
            text_only: false,
        }
    }
}

pub fn load_qwen3_runtime(model_name: &str) -> Qwen3RuntimeConfig {
    let path = pyre_whisper_core::resolve_model_file(model_name, "qwen3_runtime.json");
    Path::new(&path)
        .exists()
        .then(|| std::fs::read_to_string(&path).ok())
        .flatten()
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}