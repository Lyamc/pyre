use burn::{
    module::Param,
    nn::{self, conv::Conv1dConfig, PaddingConfig1d}
    };
use crate::model::*;
use crate::model::load::helpers::*;
use crate::model::attention::attn_decoder_mask;
use std::error::Error;
use std::path::Path;

pub fn load_audio_encoder(
    path: &str,
    device: &burn::tensor::Device,
) -> Result<(AudioEncoder, AudioEncoderConfig), Box<dyn Error>> {
    let n_mels = load_usize("n_mels", path, device)?;
    let n_audio_state = load_usize("n_audio_state", path, device)?;

    let conv1_config =
        Conv1dConfig::new(n_mels, n_audio_state, 3).with_padding(PaddingConfig1d::Explicit(1, 1));
    let conv2_config = Conv1dConfig::new(n_audio_state, n_audio_state, 3)
        .with_padding(PaddingConfig1d::Explicit(1, 1))
        .with_stride(2);

    let conv1 = if tensor_exists("weight", &format!("{}/{}", path, "conv1")) {
        load_conv1d(&format!("{}/{}", path, "conv1"), conv1_config, device)?
    } else {
        nn::conv::Conv1dConfig::new(n_mels, n_audio_state, 1).init(device)
    };

    let conv2 = if tensor_exists("weight", &format!("{}/{}", path, "conv2")) {
        load_conv1d(&format!("{}/{}", path, "conv2"), conv2_config, device)?
    } else {
        nn::conv::Conv1dConfig::new(n_audio_state, n_audio_state, 1).init(device)
    };

    let n_layer = load_usize("n_layer", path, device)?;

    let blocks: Vec<ResidualEncoderAttentionBlock> = (0..n_layer)
        .map(|i| load_residual_encoder_attention_block(&format!("{}/block_{}", path, i), n_audio_state, device))
        .collect::<Result<_, _>>()?;

    let ln_post = load_layer_norm(&format!("{}/{}", path, "ln_post"), n_audio_state, device)?;

    let n_head = blocks[0].attn.n_head;

    let positional_embedding = if Path::new(&format!("{}/positional_embedding.npy", path)).exists() {
        Some(Param::from_tensor(load_tensor::<2>("positional_embedding", path, device)?))
    } else {
        None
    };

    let n_audio_ctx = if let Some(ref pos_emb) = positional_embedding {
        pos_emb.val().dims()[0]
    } else {
        1500
    };

    let audio_encoder = AudioEncoder {
        conv1,
        gelu1: nn::Gelu::new(),
        conv2,
        gelu2: nn::Gelu::new(),
        blocks,
        ln_post,
        positional_embedding,
        n_mels,
        n_audio_ctx,
    };

    let config = AudioEncoderConfig {
        n_mels,
        n_audio_ctx,
        n_audio_state,
        n_audio_head: n_head,
        n_audio_layer: n_layer,
    };

    Ok((audio_encoder, config))
}

pub fn load_text_decoder(
    path: &str,
    device: &burn::tensor::Device,
) -> Result<(TextDecoder, TextDecoderConfig), Box<dyn Error>> {
    let token_embedding_tensor = load_tensor::<2>("token_embedding/weight", path, device)?;
    let positional_embedding_tensor = load_tensor::<2>("positional_embedding", path, device)?;
    let tensor_device_ref = token_embedding_tensor.device();

    let [n_text_ctx, n_text_state] = positional_embedding_tensor.dims();
    let [n_vocab, _] = token_embedding_tensor.dims();

    let n_layer = load_usize("n_layer", path, device)?;
    let blocks: Vec<ResidualDecoderAttentionBlock> = (0..n_layer)
        .map(|i| load_residual_decoder_attention_block(&format!("{}/block_{}", path, i), n_text_state, device))
        .collect::<Result<_, _>>()?;

    let n_text_head = blocks[0].attn.n_head;

    let ln = load_layer_norm(&format!("{}/{}", path, "ln"), n_text_state, device)?;

    let mask = attn_decoder_mask(n_text_ctx, &tensor_device_ref);

    let text_decoder = TextDecoder {
        token_embedding: Param::from_tensor(token_embedding_tensor),
        positional_embedding: Param::from_tensor(positional_embedding_tensor),
        blocks,
        ln,
        mask: Param::from_tensor(mask),
        n_text_ctx,
        n_vocab,
    };

    let config = TextDecoderConfig {
        n_vocab,
        n_text_ctx,
        n_text_state,
        n_text_head: n_text_head,
        n_text_layer: n_layer,
    };

    Ok((text_decoder, config))
}

pub fn load_whisper(path: &str, device: &burn::tensor::Device) -> Result<(Whisper, WhisperConfig), Box<dyn Error>> {
    let (encoder, encoder_config) = load_audio_encoder(&format!("{}/{}", path, "encoder"), device)?;
    let (decoder, decoder_config) = load_text_decoder(&format!("{}/{}", path, "decoder"), device)?;
    let whisper = Whisper {
        encoder,
        decoder,
    };

    let config = WhisperConfig {
        audio_encoder_config: encoder_config,
        text_decoder_config: decoder_config,
    };

    Ok((whisper, config))
}
