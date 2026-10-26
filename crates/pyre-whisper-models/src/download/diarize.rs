use std::fs;
use std::path::Path;

use hf_hub::api::sync::Api;

use crate::sources::DownloadSource;

use super::fetch::fetch_repo_file;
use super::options::DownloadOptions;
use super::python::{run_script, script_path};

pub fn is_diarize_download(name: &str) -> bool {
    matches!(name, "diarization-3.1" | "nemo-diarization")
}

pub fn download_diarize(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let repo = api.model(source.repo_id().to_string());
    let candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "embedding.safetensors",
        "speakerverification_en_titanet_large.nemo",
        "titanet-l.nemo",
    ];
    let mut ckpt = None;
    for remote in candidates {
        if let Ok(p) = fetch_repo_file(&repo, remote, dir, options) {
            ckpt = Some(p);
            break;
        }
    }
    let ckpt = ckpt.ok_or_else(|| {
        anyhow::anyhow!(
            "could not fetch speaker embedding weights from {} (need HF_TOKEN for gated pyannote models)",
            source.repo_id()
        )
    })?;
    run_script(
        &script_path("pack_wespeaker.py"),
        &[ckpt.display().to_string(), dir.display().to_string()],
    )?;
    let _ = name;
    Ok(())
}
