use std::path::Path;

use hf_hub::api::sync::Api;

use crate::convert::{convert_parakeet_from_hf, extract_nemo_to_hf};
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::http::fetch_hf_http;
use super::options::DownloadOptions;

const PARAKEET_FILES: &[&str] = &["tokenizer.json", "config.json", "model.safetensors"];
const V2_NEMO: &str = "parakeet-tdt-0.6b-v2.nemo";
const V2_ONNX_VOCAB_REPO: &str = "istupakov/parakeet-tdt-0.6b-v2-onnx";

pub fn download_parakeet_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());

    if dir.join("shapes.json").exists() && dir.join("encoder").is_dir() {
        if options.verbose {
            tracing::info!("using existing npy dump in {}", dir.display());
        }
        return convert_parakeet_from_hf(dir, name, options);
    }

    if name == "fluid-speech-pro" {
        download_fluid(&repo, source, dir, options)?;
    } else if name == "parakeet-tdt-0.6b-v2" {
        download_parakeet_v2(&repo, source, dir, options)?;
    } else {
        for file in PARAKEET_FILES {
            fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
        }
        let _ = fetch_optional_hf_file(
            &repo,
            source.repo_id(),
            "preprocessor_config.json",
            dir,
            options,
        )?;
        let _ = fetch_optional_hf_file(&repo, source.repo_id(), "generation_config.json", dir, options)?;
    }

    convert_parakeet_from_hf(dir, name, options)
}

fn download_parakeet_v2(
    repo: &hf_hub::api::sync::ApiRepo,
    source: &DownloadSource,
    dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    // Prefer a transformers-style bundle when NVIDIA publishes one.
    let has_safetensors = fetch_optional_hf_file(
        repo,
        source.repo_id(),
        "model.safetensors",
        dir,
        options,
    )?
    .is_some();
    if has_safetensors {
        for file in ["config.json", "tokenizer.json"] {
            fetch_optional_hf_file(repo, source.repo_id(), file, dir, options)?;
        }
        return Ok(());
    }

    if options.verbose {
        tracing::info!(
            "v2 is a NeMo archive (not transformers safetensors); converting to Burn, not ONNX Runtime"
        );
    }

    fetch_required_hf_file(repo, source.repo_id(), V2_NEMO, dir, options)?;
    let _ = fetch_hf_http(V2_ONNX_VOCAB_REPO, "vocab.txt", dir, options);

    if dir.join("model.safetensors").exists() && dir.join("config.json").exists() {
        return Ok(());
    }

    extract_nemo_archive(dir, options)
}

fn download_fluid(
    repo: &hf_hub::api::sync::ApiRepo,
    source: &DownloadSource,
    dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    // The NeMo archive is the reference: it holds `encoder.fv_insert` and the clean-mode
    // factors. The Transformers safetensors omit that block.
    const NEMO: &str = "fluid-speech-pro.nemo";
    fetch_required_hf_file(repo, source.repo_id(), NEMO, dir, options)?;
    for file in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json"] {
        let _ = fetch_optional_hf_file(repo, source.repo_id(), file, dir, options)?;
    }
    if options.verbose {
        tracing::info!("extracting {NEMO} in Rust (frame insert + clean mode)");
    }
    extract_nemo_to_hf(&dir.join(NEMO), dir)?;
    Ok(())
}

fn extract_nemo_archive(dir: &Path, options: &DownloadOptions) -> anyhow::Result<()> {
    let nemo = dir.join(V2_NEMO);
    if options.verbose {
        tracing::info!("extracting {} in Rust to HF-style safetensors", nemo.display());
    }
    extract_nemo_to_hf(&nemo, dir)?;
    if !dir.join("model.safetensors").exists() {
        anyhow::bail!("nemo extract did not produce model.safetensors in {}", dir.display());
    }
    Ok(())
}