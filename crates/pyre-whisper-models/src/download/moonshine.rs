use std::path::Path;

use hf_hub::api::sync::Api;

use crate::convert::prepare_moonshine_bundle;
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::options::DownloadOptions;

const REQUIRED: &[&str] = &["config.json", "model.safetensors", "tokenizer.json"];
const OPTIONAL: &[&str] = &["generation_config.json", "preprocessor_config.json", "tokenizer_config.json"];

pub fn download_moonshine_hf(
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
    prepare_moonshine_bundle(dir, name, options)
}
