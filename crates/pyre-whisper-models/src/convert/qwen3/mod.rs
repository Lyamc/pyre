mod config;
mod mapping;

use std::fs;
use std::path::Path;

use anyhow::Context;

use crate::download::DownloadOptions;

pub use config::Qwen3RuntimeConfig;
pub use mapping::{map_hf_tensor, should_skip_key};

pub const BURN_BUNDLE_VERSION: &str = "0.21.0-qwen3";

pub fn prepare_qwen3_bundle(
    model_dir: &Path,
    _model_name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");

    if !weights_path.exists() {
        anyhow::bail!(
            "missing model.safetensors in {} — download from HuggingFace first",
            model_dir.display()
        );
    }

    let config_raw = fs::read_to_string(&config_path).context("read config.json")?;
    options.report(crate::download::PrepProgress::convert(
        "Writing Qwen3 runtime config",
        0.9,
    ));
    let runtime = if let Ok(hf) = serde_json::from_str::<config::Qwen3HfConfig>(&config_raw) {
        Qwen3RuntimeConfig::from_hf(&hf)
    } else if let Ok(flat) = serde_json::from_str::<config::Qwen3AsrFlatHfConfig>(&config_raw) {
        Qwen3RuntimeConfig::from_flat_hf(&flat)
    } else {
        let text: config::Qwen3TextOnlyHfConfig =
            serde_json::from_str(&config_raw).context("parse Qwen3 text config.json")?;
        Qwen3RuntimeConfig::from_text_hf(&text)
    };
    fs::write(
        model_dir.join("qwen3_runtime.json"),
        serde_json::to_string_pretty(&runtime)?,
    )?;
    fs::write(model_dir.join(".burn_version"), BURN_BUNDLE_VERSION)?;

    if options.verbose {
        let bytes = fs::read(&weights_path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let mapped = tensors.names().iter().filter(|k| map_hf_tensor(k).is_some()).count();
        let skipped = tensors.names().iter().filter(|k| should_skip_key(k)).count();
        tracing::info!(
            "qwen3 bundle ready: {} tensors, {} mappable, {} skipped",
            tensors.names().len(),
            mapped,
            skipped
        );
        tracing::info!("inference status: {}", runtime.inference_status);
    }

    Ok(())
}

pub fn is_qwen3_model(name: &str) -> bool {
    matches!(name, "qwen3-asr-0.6b" | "qwen3-asr-1.7b" | "qwen3-0.6b")
}

pub fn needs_qwen3_reconversion(model_dir: &Path) -> bool {
    match fs::read_to_string(model_dir.join(".burn_version")) {
        Ok(v) => v.trim() != BURN_BUNDLE_VERSION,
        Err(_) => true,
    }
}

