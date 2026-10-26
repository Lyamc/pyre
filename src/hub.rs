//! Resolve `-m` to a local directory: either an existing path or a HF repo id downloaded via hf-hub.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn resolve(model: &str, cache: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(model);
    if p.is_dir() { return Ok(p.to_path_buf()); }
    let (owner, name) = model.split_once('/').context("model must be a directory or an HF repo id like Qwen/Qwen2.5-0.5B-Instruct")?;
    let mut b = hf_hub::HFClient::builder();
    // Default cache: $PYRE_CACHE, else $HF_HUB_CACHE, else <LOCALAPPDATA or ~/.cache>/pyre/hub
    let dflt = std::env::var_os("PYRE_CACHE").map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HUB_CACHE").map(PathBuf::from))
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("pyre").join("hub")))
        .or_else(|| std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".cache").join("pyre").join("hub")));
    let cdir = cache.map(Path::to_path_buf).or(dflt);
    if let Some(c) = cdir.clone() { b = b.cache_dir(c); }
    let client = b.build_sync()?;
    let repo = client.model(owner, name);
    let get = |f: &str| repo.download_file().filename(f).send();
    tracing::info!("downloading {model} via hf-hub (cache: {:?})", cdir);
    let cfg = get("config.json").context("config.json")?;
    let dir = cfg.parent().unwrap().to_path_buf();
    get("tokenizer.json").context("tokenizer.json")?;
    for f in ["tokenizer_config.json", "generation_config.json", "chat_template.jinja"] { let _ = get(f); }
    match get("model.safetensors.index.json") {
        Ok(idx) => {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(idx)?)?;
            let mut files: Vec<String> = j["weight_map"].as_object().context("weight_map")?.values()
                .filter_map(|v| v.as_str().map(String::from)).collect();
            files.sort(); files.dedup();
            for f in files { tracing::info!("fetching {f}"); get(&f)?; }
        }
        Err(_) => { tracing::info!("fetching model.safetensors"); get("model.safetensors").context("model.safetensors")?; }
    }
    Ok(dir)
}
