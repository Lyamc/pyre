use std::collections::HashSet;
use std::fs;
use std::path::Path;

use hf_hub::api::sync::Api;
use serde::Deserialize;

use crate::convert::prepare_vibevoice_bundle;
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::options::DownloadOptions;

const VIBEVOICE_REQUIRED_META: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
];

const VIBEVOICE_OPTIONAL_META: &[&str] = &[
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
];

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

pub fn download_vibevoice_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());

    if options.verbose {
        let hint = if name == "bitnet-asr" {
            "VibeVoice-ASR-BitNet (~11 GB safetensors; GGUF files are skipped — Burn ternary path)"
        } else {
            "VibeVoice-ASR 7B: prefer GGUF Q4_K (~5 GB) then convert to Burn INT8"
        };
        tracing::info!("downloading {hint} — this may take a while");
    }

    const GGUF_META: &[&str] = &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ];
    if name != "bitnet-asr" {
        for file in GGUF_META {
            if file == &"config.json" || file == &"tokenizer.json" {
                fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
            } else {
                let _ = fetch_optional_hf_file(&repo, source.repo_id(), file, dir, options)?;
            }
        }
        let gguf_ok = super::http::fetch_hf_http(
            "cstr/vibevoice-asr-GGUF",
            "vibevoice-asr-q4_k.gguf",
            dir,
            options,
        )
        .is_ok();
        if gguf_ok {
            if options.verbose {
                tracing::info!("using cstr/vibevoice-asr-GGUF Q4_K; skipping 17 GB f32 shards");
            }
            for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
                let p = entry.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.ends_with(".safetensors") || name == "model.safetensors.index.json" {
                    let _ = fs::remove_file(&p);
                }
            }
            return prepare_vibevoice_bundle(dir, name, options);
        }
        tracing::warn!("GGUF Q4_K download failed; falling back to safetensor shards");
    }

    for file in VIBEVOICE_REQUIRED_META {
        fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    for file in VIBEVOICE_OPTIONAL_META {
        let _ = fetch_optional_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }

    let index: SafetensorsIndex = serde_json::from_str(&fs::read_to_string(
        dir.join("model.safetensors.index.json"),
    )?)?;
    let mut shards: Vec<String> = index
        .weight_map
        .values()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    shards.sort();

    for (i, file) in shards.iter().enumerate() {
        if options.verbose {
            tracing::info!("fetching shard {}/{}: {file}", i + 1, shards.len());
        }
        fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }

    prepare_vibevoice_bundle(dir, name, options)
}