use burn::module::{Module, Param};
use burn::nn::conv::Conv2d;
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::{Tensor};
use std::error::Error;
use std::path::Path;

use crate::model::conformer::RMSNorm;
use crate::model::vibevoice::weights::dtype::transpose_linear_weight;
use crate::model::qwen3::audio_tower::Qwen3AudioEncoderLayer;
use crate::model::qwen3::thinker::Qwen3ThinkerLayer;
use crate::model::qwen3::{Qwen3ASR, Qwen3AudioTower, Qwen3Thinker};
use super::Qwen3WeightStore;

pub fn load_qwen3_weights(
    model: &mut Qwen3ASR,
    model_dir: &Path,
    device: &burn::tensor::Device,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    if !weights_present(model_dir) {
        return Err(format!(
            "Qwen3 safetensors missing audio/thinker keys in {}",
            model_dir.display()
        )
        .into());
    }

    let store = Qwen3WeightStore::open(model_dir).map_err(|e| format!("{e}"))?;
    let mut store = store;

    model.audio_tower = load_audio_tower(&mut store, &model.audio_tower, device)?;
    model.thinker = load_thinker(&mut store, &model.thinker, device)?;

    if verbose {
        println!("Qwen3: loaded audio_tower + thinker weights from safetensors");
    }
    Ok(())
}

pub fn load_qwen3_lm_weights(
    thinker: &mut Qwen3Thinker,
    model_dir: &Path,
    device: &burn::tensor::Device,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let mut store = Qwen3WeightStore::open(model_dir).map_err(|e| format!("{e}"))?;
    if !store.has_key("thinker.model.layers.0.self_attn.q_proj.weight") {
        return Err("Qwen3 text weights not found (expected model.layers.* in safetensors)".into());
    }
    *thinker = load_thinker(&mut store, thinker, device)?;
    if verbose {
        println!("Qwen3 LM: loaded thinker weights from safetensors");
    }
    Ok(())
}

fn weights_present(model_dir: &Path) -> bool {
    super::weights_present(model_dir)
}

fn load_audio_tower(
    store: &mut Qwen3WeightStore,
    tower: &Qwen3AudioTower,
    device: &burn::tensor::Device,
) -> Result<Qwen3AudioTower, Box<dyn Error>> {
    let p = "thinker.audio_tower";
    let conv2d1 = load_conv2d(store, &format!("{p}.conv2d1"), tower.conv2d1.clone(), device)?;
    let conv2d2 = load_conv2d(store, &format!("{p}.conv2d2"), tower.conv2d2.clone(), device)?;
    let conv2d3 = load_conv2d(store, &format!("{p}.conv2d3"), tower.conv2d3.clone(), device)?;
    let conv_out = load_linear(store, &format!("{p}.conv_out"), tower.conv_out.clone(), device)?;
    let layers = tower
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| load_audio_layer(store, &format!("{p}.layers.{i}"), layer, device))
        .collect::<Result<Vec<_>, _>>()?;
    let ln_post = load_layer_norm(store, &format!("{p}.ln_post"), tower.ln_post.clone(), device)?;
    let proj1 = load_linear(store, &format!("{p}.proj1"), tower.proj1.clone(), device)?;
    let proj2 = load_linear(store, &format!("{p}.proj2"), tower.proj2.clone(), device)?;

    let mut record = tower.clone();
    record.conv2d1 = conv2d1;
    record.conv2d2 = conv2d2;
    record.conv2d3 = conv2d3;
    record.conv_out = conv_out;
    record.layers = layers.into_iter().collect();
    record.ln_post = ln_post;
    record.proj1 = proj1;
    record.proj2 = proj2;
    Ok(record)
}

fn load_audio_layer(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    layer: &Qwen3AudioEncoderLayer,
    device: &burn::tensor::Device,
) -> Result<super::super::audio_tower::Qwen3AudioEncoderLayer, Box<dyn Error>> {
    let attn_prefix = format!("{prefix}.self_attn");
    let q_proj = load_linear(store, &format!("{attn_prefix}.q_proj"), layer.self_attn.q_proj.clone(), device)?;
    let k_proj = load_linear(store, &format!("{attn_prefix}.k_proj"), layer.self_attn.k_proj.clone(), device)?;
    let v_proj = load_linear(store, &format!("{attn_prefix}.v_proj"), layer.self_attn.v_proj.clone(), device)?;
    let out_proj = load_linear(store, &format!("{attn_prefix}.out_proj"), layer.self_attn.out_proj.clone(), device)?;
    let self_attn_layer_norm =
        load_layer_norm(store, &format!("{prefix}.self_attn_layer_norm"), layer.self_attn_layer_norm.clone(), device)?;
    let final_layer_norm =
        load_layer_norm(store, &format!("{prefix}.final_layer_norm"), layer.final_layer_norm.clone(), device)?;
    let fc1 = load_linear(store, &format!("{prefix}.fc1"), layer.fc1.clone(), device)?;
    let fc2 = load_linear(store, &format!("{prefix}.fc2"), layer.fc2.clone(), device)?;

    let mut record = layer.clone();
    record.self_attn.q_proj = q_proj;
    record.self_attn.k_proj = k_proj;
    record.self_attn.v_proj = v_proj;
    record.self_attn.out_proj = out_proj;
    record.self_attn_layer_norm = self_attn_layer_norm;
    record.final_layer_norm = final_layer_norm;
    record.fc1 = fc1;
    record.fc2 = fc2;
    Ok(record)
}

fn load_thinker(
    store: &mut Qwen3WeightStore,
    thinker: &Qwen3Thinker,
    device: &burn::tensor::Device,
) -> Result<Qwen3Thinker, Box<dyn Error>> {
    let embed = load_embedding(store, "thinker.model.embed_tokens", thinker.embed_tokens.clone(), device)?;
    let layers = thinker
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| load_thinker_layer(store, &format!("thinker.model.layers.{i}"), layer, device))
        .collect::<Result<Vec<_>, _>>()?;
    let norm = load_rms(store, "thinker.model.norm.weight", thinker.norm.clone(), device)?;
    let lm_head = load_linear(store, "thinker.lm_head", thinker.lm_head.clone(), device)?;

    let mut record = thinker.clone();
    record.embed_tokens = embed;
    record.layers = layers.into_iter().collect();
    record.norm = norm;
    record.lm_head = lm_head;
    Ok(record)
}

fn load_thinker_layer(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    layer: &Qwen3ThinkerLayer,
    device: &burn::tensor::Device,
) -> Result<super::super::thinker::Qwen3ThinkerLayer, Box<dyn Error>> {
    let attn = format!("{prefix}.self_attn");
    let q_proj = load_linear(store, &format!("{attn}.q_proj"), layer.self_attn.q_proj.clone(), device)?;
    let k_proj = load_linear(store, &format!("{attn}.k_proj"), layer.self_attn.k_proj.clone(), device)?;
    let v_proj = load_linear(store, &format!("{attn}.v_proj"), layer.self_attn.v_proj.clone(), device)?;
    let o_proj = load_linear(store, &format!("{attn}.o_proj"), layer.self_attn.o_proj.clone(), device)?;
    let q_norm = load_rms(store, &format!("{attn}.q_norm.weight"), layer.self_attn.q_norm.clone(), device)?;
    let k_norm = load_rms(store, &format!("{attn}.k_norm.weight"), layer.self_attn.k_norm.clone(), device)?;
    let input_layernorm =
        load_rms(store, &format!("{prefix}.input_layernorm.weight"), layer.input_layernorm.clone(), device)?;
    let post_attention_layernorm = load_rms(
        store,
        &format!("{prefix}.post_attention_layernorm.weight"),
        layer.post_attention_layernorm.clone(),
        device,
    )?;
    let gate_proj = load_linear(store, &format!("{prefix}.mlp.gate_proj"), layer.mlp.gate_proj.clone(), device)?;
    let up_proj = load_linear(store, &format!("{prefix}.mlp.up_proj"), layer.mlp.up_proj.clone(), device)?;
    let down_proj = load_linear(store, &format!("{prefix}.mlp.down_proj"), layer.mlp.down_proj.clone(), device)?;

    let mut record = layer.clone();
    record.self_attn.q_proj = q_proj;
    record.self_attn.k_proj = k_proj;
    record.self_attn.v_proj = v_proj;
    record.self_attn.o_proj = o_proj;
    record.self_attn.q_norm = q_norm;
    record.self_attn.k_norm = k_norm;
    record.input_layernorm = input_layernorm;
    record.post_attention_layernorm = post_attention_layernorm;
    record.mlp.gate_proj = gate_proj;
    record.mlp.up_proj = up_proj;
    record.mlp.down_proj = down_proj;
    Ok(record)
}

fn load_linear(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    linear: Linear,
    device: &burn::tensor::Device,
) -> Result<Linear, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    let weight = Tensor::<1>::from_floats(weight_data.as_slice(), device).reshape([d_in, d_out]);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<1>::from_floats(b.as_slice(), device));
    let mut record = linear.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(record)
}

fn load_conv2d(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    conv: Conv2d,
    device: &burn::tensor::Device,
) -> Result<Conv2d, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let shape4: [usize; 4] = shape.try_into().map_err(|_| "conv2d weight must be 4D")?;
    let weight = Tensor::<1>::from_floats(data.as_slice(), device).reshape(shape4);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<1>::from_floats(b.as_slice(), device));
    let mut record = conv.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(record)
}

fn load_layer_norm(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    ln: LayerNorm,
    device: &burn::tensor::Device,
) -> Result<LayerNorm, Box<dyn Error>> {
    let (w, _) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::<1>::from_floats(w.as_slice(), device);
    let bias = match store.tensor_f32(&format!("{prefix}.bias")) {
        Ok((b, _)) => Tensor::<1>::from_floats(b.as_slice(), device),
        Err(_) => Tensor::<1>::zeros([weight.dims()[0]], device),
    };
    let mut record = ln.clone();
    record.gamma = Param::from_tensor(weight);
    record.beta = Some(Param::from_tensor(bias));
    Ok(record)
}

fn load_rms(
    store: &mut Qwen3WeightStore,
    key: &str,
    norm: RMSNorm,
    device: &burn::tensor::Device,
) -> Result<RMSNorm, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    let dim = shape[0];
    let weight = Tensor::<1>::from_floats(data.as_slice(), device);
    debug_assert_eq!(dim, norm.gamma.dims()[0]);
    let mut record = norm.clone();
    record.gamma = Param::from_tensor(weight);
    Ok(record)
}

fn load_embedding(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    embed: Embedding,
    device: &burn::tensor::Device,
) -> Result<Embedding, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::<1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
    let mut record = embed.clone();
    record.weight = Param::from_tensor(weight);
    Ok(record)
}