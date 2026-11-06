use burn::{
    config::Config,
    module::Module,
    tensor::{Int, Tensor},
};

use crate::model::encoder::{AudioEncoder, AudioEncoderConfig};
use crate::model::decoder::{TextDecoder, TextDecoderConfig};

#[derive(Config, Debug)]
pub struct WhisperConfig {
    pub audio_encoder_config: AudioEncoderConfig,
    pub text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    pub fn init(&self, tensor_device_ref: &burn::tensor::Device) -> Whisper {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert!(
            n_audio_state == n_text_state,
            "Audio encoder state size {} must be equal to text decoder state size {}.",
            n_audio_state,
            n_text_state
        );

        let encoder = self.audio_encoder_config.init(tensor_device_ref);
        let decoder = self.text_decoder_config.init(tensor_device_ref);

        Whisper { encoder, decoder }
    }
}

#[derive(Module, Debug)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
}

impl Whisper {
    pub fn forward(&self, mel: Tensor<3>, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.decoder.forward(tokens, self.encoder.forward(mel))
    }

    pub fn forward_encoder(&self, mel: Tensor<3>) -> Tensor<3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<2, Int>,
        encoder_output: Tensor<3>,
    ) -> Tensor<3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn forward_decoder_cached(
        &self,
        tokens: Tensor<2, Int>,
        encoder_output: Tensor<3>,
        past: Option<crate::model::decoder::DecoderKv>,
    ) -> (Tensor<3>, crate::model::decoder::DecoderKv) {
        self.decoder.forward_cached(tokens, encoder_output, past)
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }
    pub fn encoder_mel_size(&self) -> usize {
        self.encoder.n_mels
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }
}
