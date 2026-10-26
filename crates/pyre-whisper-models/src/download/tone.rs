use std::path::Path;

use hf_hub::api::sync::Api;

use crate::convert::prepare_tone_bundle;
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::options::DownloadOptions;

const REQUIRED: &[&str] = &["config.json", "model.safetensors"];
const OPTIONAL: &[&str] = &[
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
];

pub fn download_tone_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());
    for file in REQUIRED {
        fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    for file in OPTIONAL {
        let _ = fetch_optional_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    prepare_tone_bundle(dir, name, options)
}
