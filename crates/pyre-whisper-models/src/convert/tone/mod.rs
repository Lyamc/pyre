use std::fs;
use std::path::Path;

use anyhow::Context;
use pyre_whisper_engine::model::tone::weights::{ToneHfConfig, ToneRuntimeConfig, BURN_BUNDLE_VERSION};
use serde_json::{json, Value};

use crate::download::DownloadOptions;

pub fn prepare_tone_bundle(
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

    let hf: ToneHfConfig =
        serde_json::from_str(&fs::read_to_string(&config_path).context("read config.json")?)?;
    let runtime = ToneRuntimeConfig::from_hf(&hf);
    fs::write(
        model_dir.join("tone_runtime.json"),
        serde_json::to_string_pretty(&runtime)?,
    )?;
    fs::write(model_dir.join(".burn_version"), BURN_BUNDLE_VERSION)?;
    write_tokenizer_json(model_dir, &runtime)?;

    if options.verbose {
        tracing::info!(
            "T-one bundle ready (d={}, {} layers, vocab {}, blank {})",
            runtime.d_model,
            runtime.n_layers,
            runtime.n_vocab,
            runtime.blank_id
        );
    }
    Ok(())
}

fn write_tokenizer_json(model_dir: &Path, runtime: &ToneRuntimeConfig) -> anyhow::Result<()> {
    let path = model_dir.join("tokenizer.json");
    if path.exists() {
        return Ok(());
    }
    let mut vocab = serde_json::Map::new();
    for (i, token) in runtime.vocab.iter().enumerate() {
        vocab.insert(token.clone(), json!(i));
    }
    vocab.insert("[PAD]".into(), json!(runtime.blank_id));
    vocab.insert("<unk>".into(), json!(runtime.blank_id.saturating_add(3).max(runtime.n_vocab)));
    let doc = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": [],
        "normalizer": Value::Null,
        "pre_tokenizer": Value::Null,
        "post_processor": Value::Null,
        "decoder": Value::Null,
        "model": {
            "type": "WordLevel",
            "unk_token": "<unk>",
            "vocab": vocab
        }
    });
    fs::write(path, serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}

pub fn is_tone_model(name: &str) -> bool {
    name == "t-one" || name.starts_with("t-one")
}

pub fn needs_tone_reconversion(model_dir: &Path) -> bool {
    match fs::read_to_string(model_dir.join(".burn_version")) {
        Ok(v) => v.trim() != BURN_BUNDLE_VERSION,
        Err(_) => true,
    }
}
