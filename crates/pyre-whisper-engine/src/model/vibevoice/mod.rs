mod connector;
pub mod decoder;
pub mod encoder;
pub mod host_embed;
pub mod quant;
mod hf_config;
mod prompt;
mod runtime;
mod waveform;
pub mod weights;

use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Tensor};

pub use connector::{SpeechConnector, SpeechConnectorConfig};
pub use decoder::{Qwen2Decoder, Qwen2DecoderConfig};
pub use encoder::{TokenizerEncoder, TokenizerEncoderConfig};
pub use prompt::{build_asr_prompt, pad_placeholders, VibeVoicePrompt};
pub use runtime::{load_vibevoice_runtime, VibeVoiceRuntimeConfig};
pub use waveform::{audio_duration_secs, normalize_dbfs, vae_token_length};

#[derive(Config, Debug)]
pub struct VibeVoiceASRConfig {
    pub acoustic_encoder: TokenizerEncoderConfig,
    pub semantic_encoder: TokenizerEncoderConfig,
    pub acoustic_connector: SpeechConnectorConfig,
    pub semantic_connector: SpeechConnectorConfig,
    pub decoder: Qwen2DecoderConfig,
}

impl VibeVoiceASRConfig {
    pub fn from_runtime_and_encoders(
        runtime: &VibeVoiceRuntimeConfig,
        acoustic_encoder: TokenizerEncoderConfig,
        semantic_encoder: TokenizerEncoderConfig,
    ) -> Self {
        Self {
            acoustic_encoder,
            semantic_encoder,
            acoustic_connector: SpeechConnectorConfig {
                input_dim: runtime.acoustic_vae_dim,
                hidden_size: runtime.hidden_size,
            },
            semantic_connector: SpeechConnectorConfig {
                input_dim: runtime.semantic_vae_dim,
                hidden_size: runtime.hidden_size,
            },
            decoder: Qwen2DecoderConfig::from_runtime(runtime),
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> VibeVoiceASR {
        VibeVoiceASR {
            acoustic_encoder: encoder::init_encoder(&self.acoustic_encoder, device),
            semantic_encoder: encoder::init_encoder(&self.semantic_encoder, device),
            acoustic_connector: self.acoustic_connector.init(device),
            semantic_connector: self.semantic_connector.init(device),
            decoder: self.decoder.init(device),
            acoustic_fix_std: self.acoustic_encoder.fix_std,
        }
    }
}

#[derive(Module, Debug)]
pub struct VibeVoiceASR {
    pub acoustic_encoder: TokenizerEncoder,
    pub semantic_encoder: TokenizerEncoder,
    pub acoustic_connector: SpeechConnector,
    pub semantic_connector: SpeechConnector,
    pub decoder: Qwen2Decoder,
    #[module(skip)]
    pub acoustic_fix_std: f32,
}

impl VibeVoiceASR {
    pub fn encoder_ctx_size(&self) -> usize {
        VibeVoiceRuntimeConfig::default().max_position_embeddings / 4
    }

    pub fn encoder_mel_size(&self) -> usize {
        1
    }

    pub fn sample_rate(&self) -> usize {
        VibeVoiceRuntimeConfig::default().sample_rate
    }

    /// Encode waveform `[batch, 1, samples]` into combined speech features `[batch, tokens, hidden]`.
    pub fn encode_speech(&self, audio: Tensor<3>) -> Tensor<3> {
        let acoustic = self
            .acoustic_connector
            .forward(self.acoustic_encoder.encode_mean(audio.clone()));
        let semantic = self
            .semantic_connector
            .forward(self.semantic_encoder.encode_mean(audio));
        acoustic + semantic
    }
}

pub fn build_vibevoice_config(
    model_name: &str,
    runtime: &VibeVoiceRuntimeConfig,
) -> Result<VibeVoiceASRConfig, String> {
    let model_dir = pyre_whisper_core::resolve_model_dir(model_name);
    let (acoustic, semantic) = hf_config::load_encoder_configs(&model_dir.to_string_lossy())?;
    Ok(VibeVoiceASRConfig::from_runtime_and_encoders(
        runtime, acoustic, semantic,
    ))
}