use std::collections::HashMap;
use std::fs;
use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv2d};
use burn::nn::{BatchNorm, LayerNorm, Linear};
use burn::tensor::{Tensor};
use serde::{Deserialize, Serialize};

use super::{ToneLayer, TONE};
use crate::model::conformer::RMSNorm;
use crate::model::vibevoice::weights::dtype::{bytes_to_f32, transpose_linear_weight};

pub const BURN_BUNDLE_VERSION: &str = "0.21.0-tone";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneRuntimeConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_mels: usize,
    pub n_vocab: usize,
    pub conv_kernel: usize,
    pub ff_mult: usize,
    pub rope_dim: usize,
    pub chunk_size: usize,
    pub reduction_position: usize,
    pub upsample_position: usize,
    pub reduction_factor: usize,
    pub reduction_kernel: usize,
    pub mhsa_left: usize,
    pub mhsa_stateless: usize,
    pub sample_rate: usize,
    pub blank_id: usize,
    pub vocab: Vec<String>,
    pub inference_status: String,
}

impl ToneRuntimeConfig {
    pub fn from_hf(hf: &ToneHfConfig) -> Self {
        let enc = &hf.encoder_params;
        let vocab = hf
            .decoder_params
            .vocabulary
            .clone()
            .unwrap_or_else(default_russian_vocab);
        let n_vocab = vocab.len() + 1;
        Self {
            d_model: enc.d_model.unwrap_or(384),
            n_heads: enc.n_heads.unwrap_or(8),
            n_layers: enc.n_layers.unwrap_or(16),
            n_mels: hf
                .feature_extraction_params
                .as_ref()
                .and_then(|f| f.n_mels)
                .unwrap_or(64),
            n_vocab,
            conv_kernel: enc.conv_kernel_size.unwrap_or(31),
            ff_mult: enc.ff_expansion_factor.unwrap_or(4),
            rope_dim: enc.rope_dim.unwrap_or(32),
            chunk_size: enc.chunk_size.unwrap_or(10),
            reduction_position: enc.reduction_position.unwrap_or(6),
            upsample_position: enc.upsample_position.unwrap_or(14),
            reduction_factor: enc.reduction_factor.unwrap_or(2),
            reduction_kernel: enc.reduction_kernel_size.unwrap_or(3),
            mhsa_left: enc.mhsa_state_size.unwrap_or(30),
            mhsa_stateless: enc.mhsa_stateless_layers.unwrap_or(14),
            sample_rate: hf
                .feature_extraction_params
                .as_ref()
                .and_then(|f| f.sample_rate)
                .unwrap_or(8_000),
            blank_id: hf.pad_token_id.unwrap_or(vocab.len()),
            vocab,
            inference_status: "ready".into(),
        }
    }
}

fn default_russian_vocab() -> Vec<String> {
    [
        "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р",
        "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", " ",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[derive(Debug, Deserialize)]
pub struct ToneHfConfig {
    pub encoder_params: ToneEncoderParams,
    pub decoder_params: ToneDecoderParams,
    pub feature_extraction_params: Option<ToneFeatParams>,
    pub pad_token_id: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ToneEncoderParams {
    pub d_model: Option<usize>,
    pub n_heads: Option<usize>,
    pub n_layers: Option<usize>,
    pub conv_kernel_size: Option<usize>,
    pub ff_expansion_factor: Option<usize>,
    pub rope_dim: Option<usize>,
    pub chunk_size: Option<usize>,
    pub reduction_position: Option<usize>,
    pub upsample_position: Option<usize>,
    pub reduction_factor: Option<usize>,
    pub reduction_kernel_size: Option<usize>,
    pub mhsa_state_size: Option<usize>,
    pub mhsa_stateless_layers: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ToneDecoderParams {
    pub vocabulary: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct ToneFeatParams {
    pub sample_rate: Option<usize>,
    pub n_mels: Option<usize>,
}

pub fn load_tone_runtime(model_name: &str) -> ToneRuntimeConfig {
    let path = pyre_whisper_core::resolve_model_file(model_name, "tone_runtime.json");
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| ToneRuntimeConfig::from_hf(&ToneHfConfig {
            encoder_params: ToneEncoderParams {
                d_model: Some(384),
                n_heads: Some(8),
                n_layers: Some(16),
                conv_kernel_size: Some(31),
                ff_expansion_factor: Some(4),
                rope_dim: Some(32),
                chunk_size: Some(10),
                reduction_position: Some(6),
                upsample_position: Some(14),
                reduction_factor: Some(2),
                reduction_kernel_size: Some(3),
                mhsa_state_size: Some(30),
                mhsa_stateless_layers: Some(14),
            },
            decoder_params: ToneDecoderParams {
                vocabulary: Some(default_russian_vocab()),
            },
            feature_extraction_params: Some(ToneFeatParams {
                sample_rate: Some(8_000),
                n_mels: Some(64),
            }),
            pad_token_id: Some(34),
        }))
}

pub fn weights_present(model_dir: &Path) -> bool {
    model_dir.join("model.safetensors").exists()
}

struct Store {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Store {
    fn open(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("model.safetensors");
        let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let mut tensors = HashMap::new();
        for key in st.names() {
            let tensor = st.tensor(key).map_err(|e| format!("{key}: {e}"))?;
            let Ok(floats) = bytes_to_f32(tensor.data(), tensor.dtype()) else {
                continue;
            };
            let shape: Vec<usize> = tensor.shape().iter().copied().collect();
            let stripped = strip_prefix(key);
            tensors.insert(stripped, (floats, shape));
        }
        Ok(Self { tensors })
    }

    fn tensor_f32(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| format!("tensor key not found: {key}"))
    }

    fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }
}

fn strip_prefix(key: &str) -> String {
    let k = key.strip_prefix("model.").unwrap_or(key);
    let k = k.strip_prefix("tone.").unwrap_or(k);
    k.strip_prefix("encoder.").unwrap_or(k).to_string()
}

pub fn load_tone_weights(
    model: &mut TONE,
    model_dir: &Path,
    device: &burn::tensor::Device,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !weights_present(model_dir) {
        return Err("T-one: model.safetensors missing".into());
    }
    let store = Store::open(model_dir)?;
    model.pre_norm = load_rms(&store, "pre_encode.pre_norm", model.pre_norm.clone(), device)?;
    model.conv0 = load_conv2d(&store, "pre_encode.conv.0.0", model.conv0.clone(), device)?;
    model.bn0 = load_bn2d(&store, "pre_encode.conv.0.1", model.bn0.clone(), device)?;
    model.conv1 = load_conv2d(&store, "pre_encode.conv.1.0", model.conv1.clone(), device)?;
    model.bn1 = load_bn2d(&store, "pre_encode.conv.1.1", model.bn1.clone(), device)?;
    model.pre_out = load_linear(&store, "pre_encode.out", model.pre_out.clone(), device)?;
    model.out_norm = load_rms(&store, "pre_encode.out_norm", model.out_norm.clone(), device)?;
    for (i, layer) in model.layers.iter_mut().enumerate() {
        *layer = load_layer(&store, i, layer, device)?;
    }
    model.red_conv = load_conv1d(&store, "temportal_reduction.conv", model.red_conv.clone(), device)?;
    model.red_pw = load_conv1d(&store, "temportal_reduction.conv_pw", model.red_pw.clone(), device)?;
    let ctc_key = if store.has("decoder.decoder_layers.0.weight") {
        "decoder.decoder_layers.0"
    } else {
        "decoder.decoder_layers.0.0"
    };
    model.ctc = load_conv1d(&store, ctc_key, model.ctc.clone(), device)?;
    if verbose {
        println!("T-one: loaded encoder + CTC head from safetensors ({} tensors)", store.tensors.len());
    }
    Ok(())
}

fn load_layer(
    store: &Store,
    i: usize,
    layer: &ToneLayer,
    device: &burn::tensor::Device,
) -> Result<ToneLayer, Box<dyn std::error::Error>> {
    let p = format!("layers.{i}");
    let mut record = layer.clone();
    record.norm_ff1 = load_rms(store, &format!("{p}.norm_feed_forward1"), layer.norm_ff1.clone(), device)?;
    record.ff1_g = load_linear(store, &format!("{p}.feed_forward1.linear1"), layer.ff1_g.clone(), device)?;
    record.ff1_v = load_linear(store, &format!("{p}.feed_forward1.linearv"), layer.ff1_v.clone(), device)?;
    record.ff1_o = load_linear(store, &format!("{p}.feed_forward1.linear2"), layer.ff1_o.clone(), device)?;
    record.norm_att = load_rms(store, &format!("{p}.norm_self_att"), layer.norm_att.clone(), device)?;
    if let Some(q) = &layer.q {
        record.q = Some(load_linear(store, &format!("{p}.self_attn.linear_q"), q.clone(), device)?);
    }
    if let Some(k) = &layer.k {
        record.k = Some(load_linear(store, &format!("{p}.self_attn.linear_k"), k.clone(), device)?);
    }
    if let Some(qln) = &layer.q_ln {
        record.q_ln = Some(load_ln(store, &format!("{p}.self_attn.q_ln"), qln.clone(), device)?);
    }
    if let Some(kln) = &layer.k_ln {
        record.k_ln = Some(load_ln(store, &format!("{p}.self_attn.k_ln"), kln.clone(), device)?);
    }
    record.v = load_linear(store, &format!("{p}.self_attn.linear_v"), layer.v.clone(), device)?;
    record.o = load_linear(store, &format!("{p}.self_attn.linear_out"), layer.o.clone(), device)?;
    record.norm_conv = load_rms(store, &format!("{p}.norm_conv"), layer.norm_conv.clone(), device)?;
    record.pw1 = load_conv1d(store, &format!("{p}.conv.pointwise_conv1"), layer.pw1.clone(), device)?;
    record.dw = load_conv1d(store, &format!("{p}.conv.depthwise_conv.conv"), layer.dw.clone(), device)?;
    record.bn = load_bn1d(store, &format!("{p}.conv.batch_norm"), layer.bn.clone(), device)?;
    record.pw2 = load_conv1d(store, &format!("{p}.conv.pointwise_conv2"), layer.pw2.clone(), device)?;
    record.norm_ff2 = load_rms(store, &format!("{p}.norm_feed_forward2"), layer.norm_ff2.clone(), device)?;
    record.ff2_g = load_linear(store, &format!("{p}.feed_forward2.linear1"), layer.ff2_g.clone(), device)?;
    record.ff2_v = load_linear(store, &format!("{p}.feed_forward2.linearv"), layer.ff2_v.clone(), device)?;
    record.ff2_o = load_linear(store, &format!("{p}.feed_forward2.linear2"), layer.ff2_o.clone(), device)?;
    record.norm_out = load_rms(store, &format!("{p}.norm_out"), layer.norm_out.clone(), device)?;
    Ok(record)
}

fn load_linear(
    store: &Store,
    prefix: &str,
    linear: Linear,
    device: &burn::tensor::Device,
) -> Result<Linear, Box<dyn std::error::Error>> {
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

fn load_conv1d(
    store: &Store,
    prefix: &str,
    conv: Conv1d,
    device: &burn::tensor::Device,
) -> Result<Conv1d, Box<dyn std::error::Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let shape3: [usize; 3] = shape
        .clone()
        .try_into()
        .map_err(|_| format!("{prefix}.weight must be 3D, got {shape:?}"))?;
    let weight = Tensor::<1>::from_floats(data.as_slice(), device).reshape(shape3);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<1>::from_floats(b.as_slice(), device));
    let mut record = conv.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(record)
}

fn load_conv2d(
    store: &Store,
    prefix: &str,
    conv: Conv2d,
    device: &burn::tensor::Device,
) -> Result<Conv2d, Box<dyn std::error::Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let shape4: [usize; 4] = shape
        .clone()
        .try_into()
        .map_err(|_| format!("{prefix}.weight must be 4D, got {shape:?}"))?;
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

fn load_rms(
    store: &Store,
    prefix: &str,
    rms: RMSNorm,
    device: &burn::tensor::Device,
) -> Result<RMSNorm, Box<dyn std::error::Error>> {
    let (w, _) = store
        .tensor_f32(&format!("{prefix}.weight"))
        .or_else(|_| store.tensor_f32(&format!("{prefix}.gamma")))?;
    let mut record = rms.clone();
    record.gamma = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), device));
    Ok(record)
}

fn load_ln(
    store: &Store,
    prefix: &str,
    ln: LayerNorm,
    device: &burn::tensor::Device,
) -> Result<LayerNorm, Box<dyn std::error::Error>> {
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

fn load_bn1d(
    store: &Store,
    prefix: &str,
    bn: BatchNorm,
    device: &burn::tensor::Device,
) -> Result<BatchNorm, Box<dyn std::error::Error>> {
    load_bn(store, prefix, bn, device)
}

fn load_bn2d(
    store: &Store,
    prefix: &str,
    bn: BatchNorm,
    device: &burn::tensor::Device,
) -> Result<BatchNorm, Box<dyn std::error::Error>> {
    load_bn(store, prefix, bn, device)
}

fn load_bn(
    store: &Store,
    prefix: &str,
    bn: BatchNorm,
    device: &burn::tensor::Device,
) -> Result<BatchNorm, Box<dyn std::error::Error>> {
    let n = bn.gamma.val().dims()[0];
    let gamma = store
        .tensor_f32(&format!("{prefix}.weight"))
        .map(|(w, _)| w)
        .unwrap_or_else(|_| vec![1.0; n]);
    let beta = store
        .tensor_f32(&format!("{prefix}.bias"))
        .map(|(w, _)| w)
        .unwrap_or_else(|_| vec![0.0; n]);
    let mean = store
        .tensor_f32(&format!("{prefix}.running_mean"))
        .map(|(w, _)| w)
        .unwrap_or_else(|_| vec![0.0; n]);
    let var = store
        .tensor_f32(&format!("{prefix}.running_var"))
        .map(|(w, _)| w)
        .unwrap_or_else(|_| vec![1.0; n]);
    let mut record = bn.clone();
    record.gamma = Param::from_tensor(Tensor::<1>::from_floats(gamma.as_slice(), device));
    record.beta = Param::from_tensor(Tensor::<1>::from_floats(beta.as_slice(), device));
    record.running_mean = burn::module::RunningState::new(Tensor::<1>::from_floats(mean.as_slice(), device));
    record.running_var = burn::module::RunningState::new(Tensor::<1>::from_floats(var.as_slice(), device));
    Ok(record)
}
