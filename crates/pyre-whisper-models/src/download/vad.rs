use std::fs;
use std::path::Path;

use hf_hub::api::sync::Api;

use crate::sources::DownloadSource;

use super::fetch::fetch_repo_file;
use super::http::fetch_url_bytes;
use super::options::DownloadOptions;
use super::python::{run_script, script_path};

pub fn is_vad_download(name: &str) -> bool {
    matches!(name, "silero-vad" | "ten-vad")
}

pub fn download_vad(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let repo = api.model(source.repo_id().to_string());
    match name {
        "silero-vad" => {
            let onnx = [
                "silero_vad.onnx",
                "src/silero_vad/data/silero_vad.onnx",
                "files/silero_vad.onnx",
            ];
            let mut got = None;
            for remote in onnx {
                if let Ok(p) = fetch_repo_file(&repo, remote, dir, options) {
                    got = Some(p);
                    break;
                }
            }
            if got.is_none() {
                let dest = dir.join("silero_vad.onnx");
                for url in [
                    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
                    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                ] {
                    if fetch_http(url, &dest).is_ok() && dest.is_file() {
                        got = Some(dest);
                        break;
                    }
                }
            }
            let onnx_path = got.ok_or_else(|| {
                anyhow::anyhow!("could not fetch silero_vad.onnx from {}", source.repo_id())
            })?;
            run_script(
                &script_path("pack_silero_vad.py"),
                &[
                    onnx_path.display().to_string(),
                    dir.display().to_string(),
                ],
            )?;
        }
        "ten-vad" => {
            let mut onnx_path = None;
            for remote in [
                "ten-vad.onnx",
                "src/onnx_model/ten-vad.onnx",
                "onnx_model/ten-vad.onnx",
            ] {
                if let Ok(p) = fetch_repo_file(&repo, remote, dir, options) {
                    onnx_path = Some(p);
                    break;
                }
            }
            if onnx_path.is_none() {
                // GitHub raw fallback
                let url = "https://github.com/TEN-framework/ten-vad/raw/main/src/onnx_model/ten-vad.onnx";
                let dest = dir.join("ten-vad.onnx");
                fetch_http(url, &dest)?;
                onnx_path = Some(dest);
            }
            let onnx_path = onnx_path.unwrap();
            run_script(
                &script_path("pack_ten_vad.py"),
                &[
                    onnx_path.display().to_string(),
                    dir.display().to_string(),
                ],
            )?;
        }
        _ => anyhow::bail!("not a VAD model: {name}"),
    }
    Ok(())
}

fn fetch_http(url: &str, dest: &Path) -> anyhow::Result<()> {
    let bytes = fetch_url_bytes(url)?;
    if bytes.len() < 10_000 {
        anyhow::bail!("download too small ({} bytes) from {url}", bytes.len());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(dest, bytes)?;
    Ok(())
}
