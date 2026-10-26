use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};

use super::http::fetch_hf_http;
use super::options::DownloadOptions;

pub fn build_api(token: &Option<String>) -> anyhow::Result<Api> {
    let mut builder = ApiBuilder::new();
    if let Some(token) = token {
        builder = builder.with_token(Some(token.clone()));
    }
    Ok(builder.build()?)
}

pub fn fetch_repo_file(
    repo: &ApiRepo,
    remote_path: &str,
    dest_dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<PathBuf> {
    let downloaded = repo.get(remote_path)?;
    let file_name = Path::new(remote_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(remote_path);
    let dest = dest_dir.join(file_name);

    if options.verbose {
        tracing::info!("copying {} -> {}", downloaded.display(), dest.display());
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&downloaded, &dest)?;
    Ok(dest)
}

pub fn fetch_required_hf_file(
    repo: &ApiRepo,
    repo_id: &str,
    remote_path: &str,
    dest_dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<PathBuf> {
    let dest = dest_dir.join(
        Path::new(remote_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(remote_path),
    );

    if dest.exists() {
        let len = dest.metadata()?.len();
        let min_bytes = if remote_path.ends_with(".safetensors") {
            1_000_000
        } else if remote_path.ends_with(".nemo") {
            2_000_000_000
        } else {
            1
        };
        if len >= min_bytes {
            if options.verbose {
                tracing::info!("using cached {} ({} bytes)", dest.display(), len);
            }
            options.report(crate::download::options::PrepProgress::download(
                format!("Using cached {remote_path}"),
                1.0,
                Some(len),
                Some(len),
            ));
            return Ok(dest);
        }
    }

    options.report(crate::download::options::PrepProgress::download(
        format!("Fetching {remote_path}"),
        0.0,
        None,
        None,
    ));

    let via_http = || {
        fetch_hf_http(repo_id, remote_path, dest_dir, options)
    };

    // HTTP first so byte progress (and serve logs) stay live; hf-hub is a silent fallback.
    via_http().or_else(|e1| {
        let msg = e1.to_string();
        if msg.contains("404") || msg.contains("curl exited") && msg.contains("22") {
            return Err(e1);
        }
        fetch_repo_file(repo, remote_path, dest_dir, options)
            .map_err(|e2| anyhow::anyhow!("http: {e1}; hub: {e2}"))
    })
}

/// Fetch a file if the repo has it. Missing remotes (404) are not errors.
pub fn fetch_optional_hf_file(
    repo: &ApiRepo,
    repo_id: &str,
    remote_path: &str,
    dest_dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<Option<PathBuf>> {
    match fetch_required_hf_file(repo, repo_id, remote_path, dest_dir, options) {
        Ok(path) => Ok(Some(path)),
        Err(err) => {
            if options.verbose {
                tracing::info!("optional {remote_path} not in {repo_id}: {err}");
            }
            Ok(None)
        }
    }
}