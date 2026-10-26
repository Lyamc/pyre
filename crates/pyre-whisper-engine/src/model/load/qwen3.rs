use std::error::Error;
use std::path::Path;

use crate::model::*;
use crate::model::qwen3::weights::weights_present;
use crate::model::qwen3::{build_qwen3_config, load_qwen3_runtime, load_qwen3_weights};
use crate::token::Gpt2Tokenizer;

pub fn load_qwen3_model(
    model_name: &str,
    device: &burn::tensor::Device,
    verbose: bool,
) -> Result<(Gpt2Tokenizer, ModelConfig, Model), Box<dyn Error>> {
    let model_dir = pyre_whisper_core::resolve_model_dir(model_name).display().to_string();
    let runtime_path = format!("{model_dir}/qwen3_runtime.json");
    if !Path::new(&runtime_path).exists() {
        return Err(format!(
            "Qwen3 runtime config missing at {runtime_path}. Run: \
             cargo run -p whisburn-cli -- models download {model_name} --verbose"
        )
        .into());
    }

    let runtime = load_qwen3_runtime(model_name);
    let model_config = ModelConfig::Qwen3(build_qwen3_config(model_name)?);
    let bpe = Gpt2Tokenizer::new(model_name)
        .map_err(|e| format!("Failed to load tokenizer for {model_name}: {e}"))?;
    let mut model = model_config.init(device).to_device(device);

    if let Model::Qwen3(ref mut qwen) = model {
        load_qwen3_weights(qwen, Path::new(&model_dir), device, verbose)?;
    }

    if verbose {
        let loaded = weights_present(Path::new(&model_dir));
        println!(
            "Qwen3-ASR loaded (safetensors present: {loaded}, status: {})",
            runtime.inference_status
        );
    }

    Ok((bpe, model_config, model))
}