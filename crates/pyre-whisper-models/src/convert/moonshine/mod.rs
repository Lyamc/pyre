use std::fs;
use std::path::Path;

use anyhow::Context;
use pyre_whisper_engine::model::moonshine::weights::{MoonshineHfConfig, MoonshineRuntimeConfig, BURN_BUNDLE_VERSION};

use crate::download::DownloadOptions;

pub fn prepare_moonshine_bundle(
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
    if !model_dir.join("tokenizer.json").exists() {
        anyhow::bail!("missing tokenizer.json in {}", model_dir.display());
    }

    let hf: MoonshineHfConfig =
        serde_json::from_str(&fs::read_to_string(&config_path).context("read config.json")?)?;
    let runtime = MoonshineRuntimeConfig::from_hf(&hf);
    fs::write(
        model_dir.join("moonshine_runtime.json"),
        serde_json::to_string_pretty(&runtime)?,
    )?;
    fs::write(model_dir.join(".burn_version"), BURN_BUNDLE_VERSION)?;

    if options.verbose {
        tracing::info!(
            "moonshine bundle ready ({}d, {} enc / {} dec layers)",
            runtime.hidden_size,
            runtime.encoder_layers,
            runtime.decoder_layers
        );
    }
    Ok(())
}

pub fn is_moonshine_model(name: &str) -> bool {
    name.starts_with("moonshine")
}

pub fn needs_moonshine_reconversion(model_dir: &Path) -> bool {
    match fs::read_to_string(model_dir.join(".burn_version")) {
        Ok(v) => v.trim() != BURN_BUNDLE_VERSION,
        Err(_) => true,
    }
}
