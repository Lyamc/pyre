use std::fs;
use std::path::Path;

use hf_hub::api::sync::Api;

use crate::convert::prepare_qwen3_bundle;
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::options::DownloadOptions;

const QWEN3_REQUIRED: &[&str] = &["config.json", "model.safetensors"];

/// Layout differs across Qwen3-ASR repos (`Qwen3-ASR-0.6B` vs `Qwen3-ASR-1.7B-hf`).
const QWEN3_OPTIONAL: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
];

const DEFAULT_PREPROCESSOR: &str = r#"{
  "chunk_length": 30,
  "dither": 0.0,
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": 128,
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 480000,
  "nb_max_frames": 3000,
  "padding_side": "right",
  "padding_value": 0.0,
  "processor_class": "Qwen3ASRProcessor",
  "return_attention_mask": true
}
"#;

pub fn download_qwen3_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());

    if options.verbose {
        tracing::info!("downloading Qwen3 weights from {}", source.repo_id());
    }

    for file in QWEN3_REQUIRED {
        fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    for file in QWEN3_OPTIONAL {
        fetch_optional_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }

    let has_tokenizer_json = dir.join("tokenizer.json").exists();
    let has_bpe_files = dir.join("vocab.json").exists() && dir.join("merges.txt").exists();
    if !has_tokenizer_json && !has_bpe_files {
        anyhow::bail!(
            "Qwen3 '{name}' is missing tokenizer.json and vocab.json/merges.txt from {}",
            source.repo_id()
        );
    }

    let preprocessor = dir.join("preprocessor_config.json");
    if !preprocessor.exists() {
        fs::write(&preprocessor, DEFAULT_PREPROCESSOR)?;
        if options.verbose {
            tracing::info!(
                "wrote default preprocessor_config.json (repo had processor_config.json or none)"
            );
        }
    }

    prepare_qwen3_bundle(dir, name, options)
}