use std::error::Error;
use std::path::Path;

use crate::model::moonshine::weights::{load_moonshine_runtime, load_moonshine_weights, weights_present};
use crate::model::moonshine::MoonshineASRConfig;
use crate::model::*;
use crate::token::Gpt2Tokenizer;

pub fn load_moonshine_model(
    model_name: &str,
    device: &burn::tensor::Device,
    verbose: bool,
) -> Result<(Gpt2Tokenizer, ModelConfig, Model), Box<dyn Error>> {
    let model_dir = pyre_whisper_core::resolve_model_dir(model_name).display().to_string();
    let runtime_path = format!("{model_dir}/moonshine_runtime.json");
    if !Path::new(&runtime_path).exists() {
        return Err(format!(
            "Moonshine runtime config missing at {runtime_path}. Run: \
             cargo run -p whisburn-cli -- models download {model_name} --verbose"
        )
        .into());
    }

    let runtime = load_moonshine_runtime(model_name);
    let model_config = ModelConfig::Moonshine(MoonshineASRConfig::from_runtime(&runtime));
    let bpe = Gpt2Tokenizer::new(model_name)
        .map_err(|e| format!("Failed to load tokenizer for {model_name}: {e}"))?;
    let mut model = model_config.init(device).to_device(device);

    if let Model::Moonshine(ref mut m) = model {
        load_moonshine_weights(m, Path::new(&model_dir), device, verbose)?;
    }

    if verbose {
        println!(
            "Moonshine loaded (safetensors present: {}, status: {})",
            weights_present(Path::new(&model_dir)),
            runtime.inference_status
        );
    }

    Ok((bpe, model_config, model))
}
