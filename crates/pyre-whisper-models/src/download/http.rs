use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::options::{format_bytes, DownloadOptions, PrepProgress};

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
}

/// List files under `prefix/` in a Hugging Face model repo (`tree/main/{prefix}`).
pub fn list_hf_repo_files(
    repo_id: &str,
    prefix: &str,
    token: &Option<String>,
) -> anyhow::Result<Vec<String>> {
    let url = format!(
        "https://huggingface.co/api/models/{repo_id}/tree/main/{prefix}?limit=1000"
    );
    let mut request = native_agent().get(&url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let response = match request.call() {
        Ok(r) => r,
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("404") {
                anyhow::bail!("no published folder '{prefix}' on {repo_id}");
            }
            anyhow::bail!("list {url}: {err}");
        }
    };
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| anyhow::anyhow!("read tree {url}: {e}"))?;
    let entries: Vec<HfTreeEntry> = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("parse tree {url}: {e}"))?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind == "file")
        .map(|e| e.path)
        .collect())
}

pub fn fetch_url_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    let resp = native_agent()
        .get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    resp.into_body()
        .read_to_vec()
        .map_err(|e| anyhow::anyhow!("read {url}: {e}"))
}

fn native_agent() -> ureq::Agent {
    use std::sync::OnceLock;
    use ureq::tls::{TlsConfig, TlsProvider};
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT
        .get_or_init(|| {
            let config = ureq::config::Config::builder()
                .tls_config(
                    TlsConfig::builder()
                        .provider(TlsProvider::NativeTls)
                        .build(),
                )
                .build();
            ureq::Agent::new_with_config(config)
        })
        .clone()
}

pub fn fetch_hf_http(
    repo_id: &str,
    remote_path: &str,
    dest_dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<PathBuf> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{remote_path}");
    let file_name = Path::new(remote_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(remote_path);
    let dest = dest_dir.join(file_name);

    tracing::info!("http GET {url}");

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    options.report(PrepProgress::download(
        format!("Downloading {file_name}"),
        0.0,
        Some(0),
        None,
    ));

    let ureq_err = match fetch_hf_http_ureq(&url, &dest, options, file_name) {
        Ok(()) => {
            if let Ok(meta) = dest.metadata() {
                options.report(PrepProgress::download(
                    format!("Downloaded {file_name} ({})", format_bytes(meta.len())),
                    1.0,
                    Some(meta.len()),
                    Some(meta.len()),
                ));
            }
            return Ok(dest);
        }
        Err(e) => e,
    };

    if cfg!(windows) {
        fetch_hf_http_curl(&url, &dest, options, file_name)
            .map_err(|curl_err| anyhow::anyhow!("ureq: {ureq_err}; curl: {curl_err}"))?;
        return Ok(dest);
    }

    Err(ureq_err)
}

fn fetch_hf_http_ureq(
    url: &str,
    dest: &Path,
    options: &DownloadOptions,
    file_name: &str,
) -> anyhow::Result<()> {
    let mut request = native_agent().get(url);
    if let Some(token) = &options.hf_token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }

    let response = request.call().map_err(|e| anyhow::anyhow!("{e}"))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|n| *n > 0);
    let mut reader = response.into_body().into_reader();
    let mut file = std::fs::File::create(dest)?;
    let mut buf = [0u8; 64 * 1024];
    let mut copied = 0u64;
    let mut last_emit = Instant::now() - Duration::from_secs(1);

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        copied += n as u64;
        if last_emit.elapsed() >= Duration::from_millis(250) {
            let fraction = total.map(|t| copied as f64 / t as f64).unwrap_or(0.0);
            let label = match total {
                Some(t) => format!(
                    "Downloading {file_name} ({} / {})",
                    format_bytes(copied),
                    format_bytes(t)
                ),
                None => format!("Downloading {file_name} ({})", format_bytes(copied)),
            };
            options.report(PrepProgress::download(label, fraction, Some(copied), total));
            last_emit = Instant::now();
        }
    }
    Ok(())
}

fn fetch_hf_http_curl(
    url: &str,
    dest: &Path,
    options: &DownloadOptions,
    file_name: &str,
) -> anyhow::Result<()> {
    tracing::info!("curl fallback: {url}");

    let total = head_content_length(url, &options.hf_token);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = stop.clone();
    let dest_watch = dest.to_path_buf();
    let options_watch = options.clone();
    let name = file_name.to_string();
    let watcher = thread::spawn(move || {
        let mut last = 0u64;
        let mut last_emit = Instant::now() - Duration::from_secs(1);
        while !stop_flag.load(Ordering::Relaxed) {
            if let Ok(meta) = fs::metadata(&dest_watch) {
                let len = meta.len();
                if len != last && last_emit.elapsed() >= Duration::from_millis(300) {
                    last = len;
                    last_emit = Instant::now();
                    let fraction = total.map(|t| len as f64 / t as f64).unwrap_or(0.0);
                    let label = match total {
                        Some(t) => format!(
                            "Downloading {name} ({} / {})",
                            format_bytes(len),
                            format_bytes(t)
                        ),
                        None => format!("Downloading {name} ({})", format_bytes(len)),
                    };
                    options_watch.report(PrepProgress::download(
                        label,
                        fraction,
                        Some(len),
                        total,
                    ));
                }
            }
            thread::sleep(Duration::from_millis(300));
        }
    });

    let mut cmd = Command::new("curl");
    cmd.arg("-fL")
        .arg("--retry")
        .arg("5")
        .arg("--retry-delay")
        .arg("3")
        .arg("-C")
        .arg("-")
        .arg("-o")
        .arg(dest);

    if let Some(token) = &options.hf_token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {token}"));
    }

    cmd.arg(url);

    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("curl spawn failed: {e}"))?;
    stop.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    if !status.success() {
        anyhow::bail!("curl exited with {status}");
    }
    Ok(())
}

fn head_content_length(url: &str, token: &Option<String>) -> Option<u64> {
    let mut request = native_agent().head(url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    request
        .call()
        .ok()?
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .filter(|n: &u64| *n > 0)
}
