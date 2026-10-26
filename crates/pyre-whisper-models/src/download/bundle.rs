use std::fs;
use std::path::Path;

use hf_hub::api::sync::{Api, ApiRepo};

use crate::convert::save_model_from_npy;
use crate::paths::has_burn_bundle;
use crate::sources::DownloadSource;

use super::cleanup::decompress_mpk_gz_variants;
use super::fetch::{fetch_repo_file, fetch_required_hf_file};
use super::http::{fetch_hf_http, list_hf_repo_files};
use super::options::DownloadOptions;

/// Fetch a pre-converted Burn 0.21 folder from `lyamc/whisburn/{name}/`.
///
/// Lists that folder on the Hub and downloads whatever is there (`.bpk` Whisper/Parakeet
/// bundles, Qwen3 safetensors, VibeVoice shards, …). Incomplete folders error so the
/// caller can fall back to upstream convert.
pub fn download_published_bundle(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let files = list_hf_repo_files(source.repo_id(), name, &options.hf_token)?;
    if files.is_empty() {
        anyhow::bail!(
            "published folder '{name}' on {} has no files",
            source.repo_id()
        );
    }

    options.report(crate::download::options::PrepProgress::download(
        format!("Fetching published '{name}' from {}", source.repo_id()),
        0.0,
        None,
        None,
    ));
    if options.verbose {
        tracing::info!(
            "published bundle: {} files in {}/{name}",
            files.len(),
            source.repo_id()
        );
    }

    let repo = api.model(source.repo_id().to_string());
    for remote in &files {
        fetch_required_hf_file(&repo, source.repo_id(), remote, dir, options)?;
    }

    decompress_mpk_gz_variants(dir, name, options.verbose)?;
    if !has_burn_bundle(dir, name) {
        anyhow::bail!(
            "published bundle for '{name}' on {} is incomplete after download",
            source.repo_id()
        );
    }
    Ok(())
}

pub fn download_generic_bundle(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());
    let bundle_candidates = [
        format!("{name}/config.cfg"),
        format!("{name}/tokenizer.json"),
        format!("{name}/model.bpk"),
        format!("{name}/model.bpk"),
        "config.cfg".to_string(),
        "tokenizer.json".to_string(),
        "model.bpk".to_string(),
        "model.bpk".to_string(),
    ];

    let fetched = bundle_candidates
        .iter()
        .filter(|remote| {
            fetch_repo_file(&repo, remote, dir, options)
                .or_else(|_| {
                    fetch_hf_http(source.repo_id(), remote, dir, options)
                })
                .is_ok()
        })
        .count();

    if fetched == 0 {
        download_hf_primitives(&repo, dir, options)?;
        save_model_from_npy(name, options)?;
    } else {
        decompress_mpk_gz_variants(dir, name, options.verbose)?;
    }

    Ok(())
}

fn download_hf_primitives(
    repo: &ApiRepo,
    dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    const CANDIDATES: &[&str] = &[
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ];

    CANDIDATES
        .iter()
        .filter_map(|file| repo.get(file).ok())
        .try_for_each(|path| {
            let dest = dir.join(
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown"),
            );
            fs::copy(&path, &dest)?;
            if options.verbose {
                tracing::info!("cached primitive {}", dest.display());
            }
            Ok(())
        })
}