pub mod load;
pub mod registry;
pub mod attention;
pub mod mlp;
pub mod encoder;
pub mod decoder;
pub mod whisper;
pub mod conformer;
pub mod parakeet;
pub mod parakeet_decoder;
pub mod tone;
pub mod moonshine;
pub mod qwen3;
pub mod vibevoice;
pub mod vad;
pub mod diarize;

pub use load::*;
pub use attention::*;
pub use mlp::*;
pub use encoder::*;
pub use decoder::*;
pub use whisper::{Whisper, WhisperConfig};
pub use parakeet::*;
pub use tone::{greedy_ctc_text, tone_log_mel, TONE, TONEConfig};
pub use qwen3::{
    build_asr_prompt, flatten_prompt, load_qwen3_lm_weights, load_qwen3_runtime, load_qwen3_weights,
    parse_qwen3_asr_output, parse_qwen3_asr_output_parts, parse_qwen3_asr_output_parts_with_forced,
    parse_qwen3_asr_output_with_forced, qwen3_forced_language_from_code, qwen3_language_name,
    Qwen3ASR, Qwen3ASRConfig, Qwen3AsrPrompt, Qwen3AudioTower, Qwen3AudioTowerConfig,
    Qwen3RuntimeConfig, Qwen3Thinker, Qwen3ThinkerConfig, QWEN3_DEFAULT_MAX_NEW_TOKENS,
};
pub use vibevoice::{
    audio_duration_secs, build_vibevoice_config, load_vibevoice_runtime, normalize_dbfs,
    pad_placeholders, vae_token_length, Qwen2Decoder, Qwen2DecoderConfig, SpeechConnector,
    SpeechConnectorConfig, TokenizerEncoder, TokenizerEncoderConfig, VibeVoiceASR,
    VibeVoiceASRConfig, VibeVoicePrompt, VibeVoiceRuntimeConfig,
};

use burn::{
    config::Config,
    module::Module
    };

#[derive(Config, Debug)]
pub enum ModelConfig {
    Whisper(whisper::WhisperConfig),
    Parakeet(parakeet::ParakeetConfig),
    TONE(tone::TONEConfig),
    Qwen3(qwen3::Qwen3ASRConfig),
    VibeVoice(vibevoice::VibeVoiceASRConfig),
    Moonshine(moonshine::MoonshineASRConfig),
}

impl ModelConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Model {
        match self {
            ModelConfig::Whisper(config) => Model::Whisper(config.init(device)),
            ModelConfig::Parakeet(config) => Model::Parakeet(config.init(device)),
            ModelConfig::TONE(config) => Model::TONE(config.init(device)),
            ModelConfig::Qwen3(config) => Model::Qwen3(config.init(device)),
            ModelConfig::VibeVoice(config) => Model::VibeVoice(config.init(device)),
            ModelConfig::Moonshine(config) => Model::Moonshine(config.init(device)),
        }
    }
}

#[derive(Module, Debug)]
pub enum Model {
    Whisper(whisper::Whisper),
    Parakeet(parakeet::Parakeet),
    TONE(tone::TONE),
    Qwen3(qwen3::Qwen3ASR),
    VibeVoice(vibevoice::VibeVoiceASR),
    Moonshine(moonshine::MoonshineASR),
}

impl Model {
    pub fn encoder_ctx_size(&self) -> usize {
        match self {
            Model::Whisper(m) => m.encoder_ctx_size(),
            Model::Parakeet(m) => m.encoder_ctx_size(),
            Model::TONE(m) => m.encoder_ctx_size(),
            Model::Qwen3(m) => m.encoder_ctx_size(),
            Model::VibeVoice(m) => m.encoder_ctx_size(),
            Model::Moonshine(m) => m.encoder_ctx_size(),
        }
    }

    pub fn encoder_mel_size(&self) -> usize {
        match self {
            Model::Whisper(m) => m.encoder_mel_size(),
            Model::Parakeet(m) => m.encoder_mel_size(),
            Model::TONE(m) => m.encoder_mel_size(),
            Model::Qwen3(m) => m.encoder_mel_size(),
            Model::VibeVoice(m) => m.encoder_mel_size(),
            Model::Moonshine(m) => m.encoder_mel_size(),
        }
    }
}