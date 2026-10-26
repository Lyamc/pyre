mod config;
mod mapping;

use std::fs;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

use crate::download::DownloadOptions;

pub use config::VibeVoiceRuntimeConfig;
pub use mapping::{map_hf_tensor, should_skip_key};

pub const BURN_BUNDLE_VERSION: &str = "0.21.0-vibevoice-stt-2";

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

pub fn prepare_vibevoice_bundle(
    model_dir: &Path,
    _model_name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let config_path = model_dir.join("config.json");
    let processor_path = model_dir.join("processor_config.json");
    let index_path = model_dir.join("model.safetensors.index.json");

    let has_gguf = model_dir.join("vibevoice-asr-q4_k.gguf").exists()
        || std::fs::read_dir(model_dir).ok().is_some_and(|it| {
            it.flatten()
                .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"))
        });

    if !index_path.exists() && !has_gguf {
        anyhow::bail!(
            "missing model.safetensors.index.json or GGUF in {} — download weights first",
            model_dir.display()
        );
    }

    if has_gguf && !index_path.exists() {
        let hf_config: config::VibeVoiceHfConfig =
            serde_json::from_str(&fs::read_to_string(&config_path).context("read config.json")?)?;
        let processor: config::VibeVoiceProcessorConfig = fs::read_to_string(&processor_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(default_processor_config);
        let runtime = VibeVoiceRuntimeConfig::from_hf(&hf_config, &processor);
        fs::write(
            model_dir.join("vibevoice_runtime.json"),
            serde_json::to_string_pretty(&runtime)?,
        )?;
        ensure_qwen_audio_special_tokens(model_dir)?;
        fs::write(model_dir.join(".burn_version"), BURN_BUNDLE_VERSION)?;
        if options.verbose {
            tracing::info!("vibevoice bundle ready from GGUF Q4 (Burn INT8 at load)");
        }
        return Ok(());
    }

    let hf_config: config::VibeVoiceHfConfig =
        serde_json::from_str(&fs::read_to_string(&config_path).context("read config.json")?)?;
    let processor: config::VibeVoiceProcessorConfig = fs::read_to_string(&processor_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(default_processor_config);

    let index: SafetensorsIndex =
        serde_json::from_str(&fs::read_to_string(&index_path).context("read index")?)?;

    let shard_files: Vec<String> = index
        .weight_map
        .values()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .cloned()
        .collect();

    let missing: Vec<_> = shard_files
        .iter()
        .filter(|shard| !model_dir.join(shard).exists())
        .collect();

    if !missing.is_empty() {
        anyhow::bail!(
            "missing safetensor shards: {:?}",
            missing.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    let runtime = VibeVoiceRuntimeConfig::from_hf(&hf_config, &processor);
    fs::write(
        model_dir.join("vibevoice_runtime.json"),
        serde_json::to_string_pretty(&runtime)?,
    )?;
    ensure_qwen_audio_special_tokens(model_dir)?;
    fs::write(model_dir.join(".burn_version"), BURN_BUNDLE_VERSION)?;

    let mapped_keys = index
        .weight_map
        .keys()
        .filter(|k| map_hf_tensor(k).is_some())
        .count();
    let skipped_keys = index
        .weight_map
        .keys()
        .filter(|k| should_skip_key(k))
        .count();

    if options.verbose {
        tracing::info!(
            "vibevoice bundle ready: {} shards, {} mappable keys, {} skipped",
            shard_files.len(),
            mapped_keys,
            skipped_keys
        );
        tracing::info!(
            "inference status: {} (Qwen2.5 greedy STT)",
            runtime.inference_status
        );
    }

    Ok(())
}

/// Qwen2.5-1.5B/7B audio + vision specials. BitNet's HF `tokenizer.json` only
/// ships chat tokens (`<|im_start|>` / `<|im_end|>`); without these IDs the
/// prompt pads never match and speech features are dropped.
const QWEN_EXTRA_SPECIALS: &[(u32, &str, bool)] = &[
    (151646, "<|object_ref_start|>", true),
    (151647, "<|object_ref_end|>", true),
    (151648, "<|box_start|>", true),
    (151649, "<|box_end|>", true),
    (151650, "<|quad_start|>", true),
    (151651, "<|quad_end|>", true),
    (151652, "<|vision_start|>", true),
    (151653, "<|vision_end|>", true),
    (151654, "<|vision_pad|>", true),
    (151655, "<|image_pad|>", true),
    (151656, "<|video_pad|>", true),
    (151657, "<tool_call>", false),
    (151658, "</tool_call>", false),
    (151659, "<|fim_prefix|>", false),
    (151660, "<|fim_middle|>", false),
    (151661, "<|fim_suffix|>", false),
    (151662, "<|fim_pad|>", false),
    (151663, "<|repo_name|>", false),
    (151664, "<|file_sep|>", false),
];

fn ensure_qwen_audio_special_tokens(model_dir: &Path) -> anyhow::Result<()> {
    let path = model_dir.join("tokenizer.json");
    if !path.exists() {
        return Ok(());
    }
    let raw = fs::read_to_string(&path).context("read tokenizer.json")?;
    let mut value: serde_json::Value = serde_json::from_str(&raw).context("parse tokenizer.json")?;
    let Some(added) = value.get_mut("added_tokens").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    let existing: std::collections::HashSet<String> = added
        .iter()
        .filter_map(|t| t.get("content").and_then(|c| c.as_str()).map(str::to_string))
        .collect();
    let mut inserted = 0;
    for &(id, content, special) in QWEN_EXTRA_SPECIALS {
        if existing.contains(content) {
            continue;
        }
        added.push(serde_json::json!({
            "id": id,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": special,
        }));
        inserted += 1;
    }
    if inserted > 0 {
        fs::write(&path, serde_json::to_string(&value)?)?;
        tracing::info!(
            "patched {} Qwen audio/vision special tokens into {}",
            inserted,
            path.display()
        );
    }
    Ok(())
}

fn default_processor_config() -> config::VibeVoiceProcessorConfig {
    config::VibeVoiceProcessorConfig {
        audio_bos_token: "<|object_ref_start|>".to_string(),
        audio_eos_token: "<|object_ref_end|>".to_string(),
        audio_token: "<|box_start|>".to_string(),
        feature_extractor: config::VibeVoiceFeatureExtractorConfig {
            sampling_rate: 24_000,
            target_dbfs: -25.0,
        },
    }
}

pub fn is_vibevoice_model(name: &str) -> bool {
    matches!(name, "vibevoice-asr" | "bitnet-asr")
}

pub fn needs_vibevoice_reconversion(model_dir: &Path) -> bool {
    match fs::read_to_string(model_dir.join(".burn_version")) {
        Ok(v) => v.trim() != BURN_BUNDLE_VERSION,
        Err(_) => true,
    }
}

