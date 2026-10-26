use std::path::Path;

use hf_hub::api::sync::Api;

use crate::convert::convert_whisper_from_hf;
use crate::sources::DownloadSource;

use super::fetch::fetch_required_hf_file;
use super::options::DownloadOptions;

const WHISPER_FILES: &[&str] = &[
    "tokenizer.json",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
];

pub fn download_whisper_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());
    for file in WHISPER_FILES {
        fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    convert_whisper_from_hf(dir, name, options)
}