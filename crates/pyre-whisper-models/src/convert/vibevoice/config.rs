use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct VibeVoiceHfConfig {
    #[serde(default)]
    pub acoustic_vae_dim: usize,
    #[serde(default)]
    pub semantic_vae_dim: usize,
    pub decoder_config: Option<VibeVoiceDecoderConfig>,
    pub text_config: Option<VibeVoiceDecoderConfig>,
    pub acoustic_tokenizer_config: Option<VibeVoiceTokenizerConfig>,
    pub semantic_tokenizer_config: Option<VibeVoiceTokenizerConfig>,
    pub acoustic_tokenizer_encoder_config: Option<VibeVoiceEncoderSectionConfig>,
    pub semantic_tokenizer_encoder_config: Option<VibeVoiceEncoderSectionConfig>,
}

#[derive(Debug, Deserialize)]
pub struct VibeVoiceEncoderSectionConfig {
    pub downsampling_ratios: Vec<usize>,
    pub hidden_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct VibeVoiceDecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_hf_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_hf_rms")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hf_rope")]
    pub rope_theta: f64,
}

fn default_hf_intermediate() -> usize {
    18_944
}
fn default_hf_rms() -> f64 {
    1e-6
}
fn default_hf_rope() -> f64 {
    1_000_000.0
}
fn default_runtime_intermediate() -> usize {
    18_944
}
fn default_runtime_rms() -> f64 {
    1e-6
}
fn default_runtime_rope() -> f64 {
    1_000_000.0
}

#[derive(Debug, Deserialize)]
pub struct VibeVoiceTokenizerConfig {
    pub vae_dim: usize,
    pub encoder_ratios: Vec<usize>,
    pub fix_std: f32,
}

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
    #[serde(default = "default_runtime_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_runtime_rms")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_runtime_rope")]
    pub rope_theta: f64,
    pub audio_bos_token: String,
    pub audio_eos_token: String,
    pub audio_pad_token: String,
    pub inference_status: String,
}

impl VibeVoiceRuntimeConfig {
    pub fn from_hf(
        hf: &VibeVoiceHfConfig,
        processor: &VibeVoiceProcessorConfig,
    ) -> Self {
        let decoder = hf.decoder_config.as_ref().or(hf.text_config.as_ref());
        let compress_ratio = hf
            .acoustic_tokenizer_encoder_config
            .as_ref()
            .map(|c| c.downsampling_ratios.iter().product())
            .or_else(|| {
                hf.acoustic_tokenizer_config
                    .as_ref()
                    .map(|c| c.encoder_ratios.iter().product())
            })
            .unwrap_or(3200);
        let acoustic_vae = hf
            .acoustic_vae_dim
            .max(
                hf.acoustic_tokenizer_encoder_config
                    .as_ref()
                    .map(|c| c.hidden_size)
                    .unwrap_or(0),
            )
            .max(
                hf.acoustic_tokenizer_config
                    .as_ref()
                    .map(|c| c.vae_dim)
                    .unwrap_or(0),
            );
        let semantic_vae = hf
            .semantic_vae_dim
            .max(
                hf.semantic_tokenizer_encoder_config
                    .as_ref()
                    .map(|c| c.hidden_size)
                    .unwrap_or(0),
            )
            .max(
                hf.semantic_tokenizer_config
                    .as_ref()
                    .map(|c| c.vae_dim)
                    .unwrap_or(0),
            );
        Self {
            burn_bundle_version: super::BURN_BUNDLE_VERSION.to_string(),
            sample_rate: processor.feature_extractor.sampling_rate,
            speech_compress_ratio: compress_ratio,
            target_dbfs: processor.feature_extractor.target_dbfs,
            acoustic_vae_dim: acoustic_vae,
            semantic_vae_dim: semantic_vae,
            hidden_size: decoder.map(|d| d.hidden_size).unwrap_or(3584),
            vocab_size: decoder.map(|d| d.vocab_size).unwrap_or(152_064),
            num_hidden_layers: decoder.map(|d| d.num_hidden_layers).unwrap_or(28),
            num_attention_heads: decoder.map(|d| d.num_attention_heads).unwrap_or(28),
            num_key_value_heads: decoder.map(|d| d.num_key_value_heads).unwrap_or(4),
            max_position_embeddings: decoder.map(|d| d.max_position_embeddings).unwrap_or(131_072),
            intermediate_size: decoder.map(|d| d.intermediate_size).unwrap_or(18_944),
            rms_norm_eps: decoder.map(|d| d.rms_norm_eps).unwrap_or(1e-6),
            rope_theta: decoder.map(|d| d.rope_theta).unwrap_or(1_000_000.0),
            audio_bos_token: processor.audio_bos_token.clone(),
            audio_eos_token: processor.audio_eos_token.clone(),
            audio_pad_token: processor.audio_token.clone(),
            inference_status: "ready".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VibeVoiceProcessorConfig {
    pub audio_bos_token: String,
    pub audio_eos_token: String,
    pub audio_token: String,
    pub feature_extractor: VibeVoiceFeatureExtractorConfig,
}

#[derive(Debug, Deserialize)]
pub struct VibeVoiceFeatureExtractorConfig {
    pub sampling_rate: usize,
    #[serde(rename = "target_dB_FS")]
    pub target_dbfs: f32,
}