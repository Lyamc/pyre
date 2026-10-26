use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor};
use std::error::Error;

use super::dtype::transpose_linear_weight;
use super::store::VibeVoiceWeightStore;
use crate::model::vibevoice::encoder::block::Block1D;
use crate::model::vibevoice::encoder::level::EncoderLevel;
use crate::model::vibevoice::encoder::norm::ConvRmsNorm;
use crate::model::vibevoice::encoder::sconv;
use crate::model::vibevoice::encoder::TokenizerEncoder;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderLayout {
    /// `acoustic_tokenizer_encoder.{stem,conv_layers,head}` (VibeVoice-ASR-HF).
    Hf,
    /// `model.acoustic_tokenizer.encoder.{downsample_layers,stages,head}` (BitNet / original).
    Original,
}

pub fn detect_encoder_layout(
    store: &VibeVoiceWeightStore,
    kind: &str,
) -> Option<(String, EncoderLayout)> {
    let hf = format!("{kind}_tokenizer_encoder.stem.conv.conv.weight");
    if store.has_key(&hf) {
        return Some((format!("{kind}_tokenizer_encoder"), EncoderLayout::Hf));
    }
    let original = format!("model.{kind}_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight");
    if store.has_key(&original) {
        return Some((
            format!("model.{kind}_tokenizer.encoder"),
            EncoderLayout::Original,
        ));
    }
    let original_stripped = format!("{kind}_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight");
    if store.has_key(&original_stripped) {
        return Some((format!("{kind}_tokenizer.encoder"), EncoderLayout::Original));
    }
    None
}

pub fn load_encoder(
    store: &mut VibeVoiceWeightStore,
    prefix: &str,
    layout: EncoderLayout,
    encoder: &TokenizerEncoder,
    device: &burn::tensor::Device,
) -> Result<TokenizerEncoder, Box<dyn Error>> {
    let (stem_conv, stem_blocks) = match layout {
        EncoderLayout::Hf => (format!("{prefix}.stem"), format!("{prefix}.stem.stage")),
        EncoderLayout::Original => (
            format!("{prefix}.downsample_layers.0.0"),
            format!("{prefix}.stages.0"),
        ),
    };
    let stem = load_level(store, &stem_conv, &stem_blocks, &encoder.stem, device)?;
    let conv_layers = encoder
        .conv_layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let (conv_prefix, block_prefix) = match layout {
                EncoderLayout::Hf => (
                    format!("{prefix}.conv_layers.{i}"),
                    format!("{prefix}.conv_layers.{i}.stage"),
                ),
                EncoderLayout::Original => (
                    format!("{prefix}.downsample_layers.{}.0", i + 1),
                    format!("{prefix}.stages.{}", i + 1),
                ),
            };
            load_level(store, &conv_prefix, &block_prefix, layer, device)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let head = load_sconv_pair(store, &format!("{prefix}.head.conv"), &encoder.head, device)?;
    let mut record = encoder.clone();
    record.stem = stem;
    record.conv_layers = conv_layers.into_iter().collect();
    record.head = head;
    Ok(record)
}

fn load_level(
    store: &mut VibeVoiceWeightStore,
    conv_prefix: &str,
    block_prefix: &str,
    level: &EncoderLevel,
    device: &burn::tensor::Device,
) -> Result<EncoderLevel, Box<dyn Error>> {
    let conv = load_sconv_pair(store, &format!("{conv_prefix}.conv"), &level.conv, device)?;
    let blocks = level
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| load_block(store, &format!("{block_prefix}.{i}"), block, device))
        .collect::<Result<Vec<_>, _>>()?;
    let mut record = level.clone();
    record.conv = conv;
    record.blocks = blocks.into_iter().collect();
    Ok(record)
}

fn load_block(
    store: &mut VibeVoiceWeightStore,
    prefix: &str,
    block: &Block1D,
    device: &burn::tensor::Device,
) -> Result<Block1D, Box<dyn Error>> {
    let dim = block.norm.weight.dims()[0];
    let norm = load_rms(store, &format!("{prefix}.norm.weight"), block.norm.eps, dim, device)?;
    let mixer = load_sconv_pair(store, &format!("{prefix}.mixer.conv"), &block.mixer, device)?;
    let gamma = load_vec(store, &format!("{prefix}.gamma"), dim, device)?;
    let ffn_norm = load_rms(store, &format!("{prefix}.ffn_norm.weight"), block.ffn_norm.eps, dim, device)?;
    let ffn_gamma = load_vec(store, &format!("{prefix}.ffn_gamma"), dim, device)?;
    let linear1 = load_linear(store, &format!("{prefix}.ffn.linear1"), device)?;
    let linear2 = load_linear(store, &format!("{prefix}.ffn.linear2"), device)?;

    let mut record = block.clone();
    record.norm = norm;
    record.mixer = mixer;
    record.gamma = Param::from_tensor(gamma);
    record.ffn_norm = ffn_norm;
    record.ffn.linear1 = linear1;
    record.ffn.linear2 = linear2;
    record.ffn_gamma = Param::from_tensor(ffn_gamma);
    Ok(record)
}

fn load_sconv_pair(
    store: &mut VibeVoiceWeightStore,
    prefix: &str,
    layer: &sconv::SConv1d,
    device: &burn::tensor::Device,
) -> Result<sconv::SConv1d, Box<dyn Error>> {
    let weight_key = sconv_weight_key(store, prefix)?;
    let (w, w_shape) = store.tensor_f32(&weight_key)?;
    if w_shape.len() != 3 {
        return Err(format!("{prefix}.weight: expected rank-3 conv, got {w_shape:?}").into());
    }
    let weight = Tensor::<1>::from_floats(w.as_slice(), device).reshape([
        w_shape[0],
        w_shape[1],
        w_shape[2],
    ]);
    let bias_key = weight_key
        .strip_suffix(".weight")
        .map(|p| format!("{p}.bias"))
        .unwrap_or_else(|| format!("{prefix}.bias"));
    let bias = store
        .tensor_f32(&bias_key)
        .ok()
        .map(|(b, _)| Tensor::<1>::from_floats(b.as_slice(), device));
    let [out_ch, in_per_group, kernel] = weight.dims();
    let groups = layer.groups.max(1);
    let in_ch = in_per_group * groups;
    Ok(sconv::load_sconv(
        weight,
        bias,
        in_ch,
        out_ch,
        kernel,
        layer.stride,
        groups,
        device,
    ))
}

fn sconv_weight_key(store: &VibeVoiceWeightStore, prefix: &str) -> Result<String, Box<dyn Error>> {
    let candidates = [
        format!("{prefix}.weight"),
        format!("{prefix}.conv.weight"),
        format!("{prefix}.conv.conv.weight"),
    ];
    for key in &candidates {
        if store.has_key(key) {
            return Ok(key.clone());
        }
    }
    Err(format!("{prefix}.weight: conv weight not in index (tried nested conv wrappers)").into())
}

fn load_rms(
    store: &mut VibeVoiceWeightStore,
    key: &str,
    eps: f64,
    dim: usize,
    device: &burn::tensor::Device,
) -> Result<ConvRmsNorm, Box<dyn Error>> {
    let weight = load_vec(store, key, dim, device)?;
    Ok(ConvRmsNorm {
        weight: Param::from_tensor(weight),
        eps,
    })
}

fn load_vec(
    store: &mut VibeVoiceWeightStore,
    key: &str,
    dim: usize,
    device: &burn::tensor::Device,
) -> Result<Tensor<1>, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    if shape != [dim] {
        return Err(format!("{key}: expected [{dim}], got {shape:?}").into());
    }
    Ok(Tensor::<1>::from_floats(data.as_slice(), device))
}

fn load_linear(
    store: &mut VibeVoiceWeightStore,
    prefix: &str,
    device: &burn::tensor::Device,
) -> Result<Linear, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    let weight = Tensor::<1>::from_floats(weight_data.as_slice(), device).reshape([d_in, d_out]);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<1>::from_floats(b.as_slice(), device));
    let linear = LinearConfig::new(d_in, d_out)
        .with_bias(bias.is_some())
        .init(device);
    let mut record = linear.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(record)
}