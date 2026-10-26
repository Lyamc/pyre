use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor};
use std::error::Error;

use super::dtype::transpose_linear_weight;
use super::store::VibeVoiceWeightStore;
use crate::model::vibevoice::SpeechConnector;

const ACOUSTIC_PREFIX: &str = "multi_modal_projector.acoustic";
const SEMANTIC_PREFIX: &str = "multi_modal_projector.semantic";

enum ConnectorKeys {
    Hf,
    Original,
}

fn connector_layout(store: &VibeVoiceWeightStore) -> ConnectorKeys {
    if store.has_key("multi_modal_projector.acoustic_linear_1.weight") {
        ConnectorKeys::Hf
    } else {
        ConnectorKeys::Original
    }
}

pub fn load_speech_connector(
    store: &mut VibeVoiceWeightStore,
    name: &str,
    connector: &SpeechConnector,
    device: &burn::tensor::Device,
) -> Result<SpeechConnector, Box<dyn Error>> {
    let (fc1_prefix, norm_key, fc2_prefix) = match connector_layout(store) {
        ConnectorKeys::Hf => {
            let hf = if name == "acoustic" {
                ACOUSTIC_PREFIX
            } else {
                SEMANTIC_PREFIX
            };
            (
                format!("{hf}_linear_1"),
                format!("{hf}_norm.weight"),
                format!("{hf}_linear_2"),
            )
        }
        ConnectorKeys::Original => {
            let orig = if store.has_key(&format!("model.{name}_connector.fc1.weight")) {
                format!("model.{name}_connector")
            } else {
                format!("{name}_connector")
            };
            (
                format!("{orig}.fc1"),
                format!("{orig}.norm.weight"),
                format!("{orig}.fc2"),
            )
        }
    };
    let fc1 = load_linear(store, &fc1_prefix, connector.fc1.weight.dims(), device)?;
    let norm_gamma = load_vector(store, &norm_key, connector.norm_gamma.dims()[0], device)?;
    let fc2 = load_linear(store, &fc2_prefix, connector.fc2.weight.dims(), device)?;

    let mut record = connector.clone();
    record.fc1 = fc1;
    record.norm_gamma = Param::from_tensor(norm_gamma);
    record.fc2 = fc2;
    Ok(record)
}

pub fn load_connectors(
    store: &mut VibeVoiceWeightStore,
    model: &super::super::VibeVoiceASR,
    device: &burn::tensor::Device,
) -> Result<(SpeechConnector, SpeechConnector), Box<dyn Error>> {
    let acoustic = load_speech_connector(store, "acoustic", &model.acoustic_connector, device)?;
    let semantic = load_speech_connector(store, "semantic", &model.semantic_connector, device)?;
    Ok((acoustic, semantic))
}

fn load_linear(
    store: &mut VibeVoiceWeightStore,
    hf_prefix: &str,
    expected: [usize; 2],
    device: &burn::tensor::Device,
) -> Result<Linear, Box<dyn Error>> {
    let weight_key = format!("{hf_prefix}.weight");
    let (data, shape) = store.tensor_f32(&weight_key)?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    if [d_in, d_out] != expected {
        return Err(format!(
            "{weight_key}: expected {:?}, got [{d_in}, {d_out}]",
            expected
        )
        .into());
    }

    let weight = Tensor::<1>::from_floats(weight_data.as_slice(), device).reshape([d_in, d_out]);
    let bias_key = format!("{hf_prefix}.bias");
    let bias = store
        .tensor_f32(&bias_key)
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

fn load_vector(
    store: &mut VibeVoiceWeightStore,
    key: &str,
    expected_len: usize,
    device: &burn::tensor::Device,
) -> Result<Tensor<1>, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    if shape != [expected_len] {
        return Err(format!("{key}: expected [{expected_len}], got {shape:?}").into());
    }
    Ok(Tensor::<1>::from_floats(data.as_slice(), device))
}