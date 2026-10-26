use burn::module::Param;
use burn::tensor::{Tensor};
use std::error::Error;

use super::dtype::transpose_linear_weight;
use super::store::VibeVoiceWeightStore;
use crate::model::conformer::RMSNorm;
use crate::model::vibevoice::decoder::{Qwen2Attention, Qwen2Decoder, Qwen2DecoderLayer, Qwen2Mlp};
use crate::model::vibevoice::host_embed::HostEmbedding;
use crate::model::vibevoice::quant::{quantize_linear, LinearQuant, QuantLinear};

const LM: &str = "language_model.model";

pub fn load_decoder(
    store: &mut VibeVoiceWeightStore,
    decoder: &Qwen2Decoder,
    proj_quant: LinearQuant,
    device: &burn::tensor::Device,
) -> Result<Qwen2Decoder, Box<dyn Error>> {
    let embed = load_embedding(store, &format!("{LM}.embed_tokens.weight"), device)?;
    let layers = decoder
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| load_layer(store, i, layer, proj_quant, device))
        .collect::<Result<Vec<_>, _>>()?;
    let dim = decoder.hidden_size;
    let norm = load_rms(store, &format!("{LM}.norm.weight"), decoder.norm.epsilon, dim, device)?;
    // lm_head is not BitNet-trained (I2_S is q/k/v/o/gate/up/down only).
    // 7B keeps the dequantized f32 head (tiled); BitNet still packs INT8.
    let lm_head = if store.has_key("language_model.lm_head.weight") {
        if matches!(proj_quant, LinearQuant::Ternary) {
            load_quant_linear(store, "language_model.lm_head", LinearQuant::Int8, device)?
        } else {
            load_f32_linear(store, "language_model.lm_head", device)?
        }
    } else {
        load_quant_linear(store, &format!("{LM}.embed_tokens"), LinearQuant::Int8, device)?
    };

    Ok(Qwen2Decoder {
        hidden_size: decoder.hidden_size,
        vocab_size: decoder.vocab_size,
        head_dim: decoder.head_dim,
        rope_theta: decoder.rope_theta,
        embed_tokens: embed,
        layers,
        norm,
        lm_head,
    })
}

fn load_layer(
    store: &mut VibeVoiceWeightStore,
    i: usize,
    layer: &Qwen2DecoderLayer,
    proj_quant: LinearQuant,
    device: &burn::tensor::Device,
) -> Result<Qwen2DecoderLayer, Box<dyn Error>> {
    let p = format!("{LM}.layers.{i}");
    let dim = layer.input_layernorm.gamma.val().dims()[0];
    let input_ln = load_rms(store, &format!("{p}.input_layernorm.weight"), layer.input_layernorm.epsilon, dim, device)?;
    let post_ln = load_rms(
        store,
        &format!("{p}.post_attention_layernorm.weight"),
        layer.post_attention_layernorm.epsilon,
        dim,
        device,
    )?;
    Ok(Qwen2DecoderLayer {
        input_layernorm: input_ln,
        self_attn: Qwen2Attention {
            n_heads: layer.self_attn.n_heads,
            n_kv_heads: layer.self_attn.n_kv_heads,
            head_dim: layer.self_attn.head_dim,
            q_proj: load_quant_linear(store, &format!("{p}.self_attn.q_proj"), proj_quant, device)?,
            k_proj: load_quant_linear(store, &format!("{p}.self_attn.k_proj"), proj_quant, device)?,
            v_proj: load_quant_linear(store, &format!("{p}.self_attn.v_proj"), proj_quant, device)?,
            o_proj: load_quant_linear(store, &format!("{p}.self_attn.o_proj"), proj_quant, device)?,
        },
        post_attention_layernorm: post_ln,
        mlp: Qwen2Mlp {
            gate_proj: load_quant_linear(store, &format!("{p}.mlp.gate_proj"), proj_quant, device)?,
            up_proj: load_quant_linear(store, &format!("{p}.mlp.up_proj"), proj_quant, device)?,
            down_proj: load_quant_linear(store, &format!("{p}.mlp.down_proj"), proj_quant, device)?,
        },
    })
}

fn load_embedding(
    store: &mut VibeVoiceWeightStore,
    key: &str,
    device: &burn::tensor::Device,
) -> Result<HostEmbedding, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(format!("{key}: expected rank-2 embedding, got {shape:?}").into());
    }
    Ok(HostEmbedding::from_weight(data, shape[0], shape[1], device))
}

fn load_rms(
    store: &mut VibeVoiceWeightStore,
    key: &str,
    eps: f64,
    dim: usize,
    device: &burn::tensor::Device,
) -> Result<RMSNorm, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    if shape != [dim] && data.len() != dim {
        return Err(format!("{key}: expected [{dim}], got {shape:?}").into());
    }
    Ok(RMSNorm {
        gamma: Param::from_tensor(Tensor::<1>::from_floats(data.as_slice(), device)),
        epsilon: eps,
    })
}

fn load_quant_linear(
    store: &mut VibeVoiceWeightStore,
    prefix: &str,
    scheme: LinearQuant,
    device: &burn::tensor::Device,
) -> Result<QuantLinear, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    drop(data);
    let (q, scale) = quantize_linear(scheme, &weight_data, d_in, d_out);
    drop(weight_data);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| b);
    Ok(QuantLinear::from_quantized(q, scale, bias, d_in, d_out, device))
}

fn load_f32_linear(
    store: &mut VibeVoiceWeightStore,
    prefix: &str,
    device: &burn::tensor::Device,
) -> Result<QuantLinear, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    drop(data);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| b);
    Ok(QuantLinear::from_f32(weight_data, bias, d_in, d_out, device))
}
