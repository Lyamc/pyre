//! Locating a Qwen3.8-Flash-Next GGUF and its tokenizer/chat template.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// `p` is a .gguf shard, or a folder (e.g. the HF snapshot of ISTA-DASLab/...-GGUF) holding one
/// quant sub-folder. `PYRE_QN_QUANT` (e.g. IQ2_XS) picks one when several are present.
pub fn find_gguf(p: &Path) -> Option<PathBuf> {
    if p.is_file() { return (p.extension()?.to_str()? == "gguf").then(|| p.to_path_buf()); }
    let want = std::env::var("PYRE_QN_QUANT").ok();
    let mut found = vec![];
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let q = e.path();
            if q.is_dir() { stack.push(q); continue; }
            let n = q.file_name()?.to_string_lossy().to_string();
            if n.ends_with(".gguf") && !n.starts_with("mmproj") && (n.contains("-00001-of-") || !n.contains("-of-")) { found.push(q); }
        }
    }
    found.sort();
    if let Some(w) = want { if let Some(f) = found.iter().find(|f| f.to_string_lossy().contains(&w)) { return Some(f.clone()); } }
    found.into_iter().next()
}

pub fn is_qwen4exp(p: &Path) -> bool {
    let Some(f) = find_gguf(p) else { return false };
    let Ok(mut fh) = std::fs::File::open(&f) else { return false };
    let mut buf = vec![0u8; 4096];
    use std::io::Read;
    let n = fh.read(&mut buf).unwrap_or(0);
    buf[..n].windows(8).any(|w| w == b"qwen4exp")
}

fn hf_hub() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") { return p.into(); }
    if let Ok(p) = std::env::var("HF_HOME") { return Path::new(&p).join("hub"); }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
    Path::new(&home).join(".cache").join("huggingface").join("hub")
}

/// Folder with tokenizer.json (+ chat_template.jinja): next to the GGUF, `PYRE_QN_TOKENIZER`,
/// or the base model's HF snapshot named in the GGUF metadata (general.base_model.0.repo_url).
pub fn tokenizer_dir(gguf: &Path, base_repo_url: Option<&str>) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("PYRE_QN_TOKENIZER") { return Ok(p.into()); }
    for d in gguf.ancestors().skip(1).take(3) { if d.join("tokenizer.json").exists() { return Ok(d.to_path_buf()); } }
    let repo = base_repo_url.and_then(|u| u.strip_prefix("https://huggingface.co/")).unwrap_or("Qwen/Qwen3.8-Flash-Next");
    let snaps = hf_hub().join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    for e in std::fs::read_dir(&snaps).with_context(|| format!("no tokenizer: download {repo} tokenizer.json/chat_template.jinja (hf download {repo} tokenizer.json tokenizer_config.json chat_template.jinja) or set PYRE_QN_TOKENIZER"))?.flatten() {
        if e.path().join("tokenizer.json").exists() { return Ok(e.path()); }
    }
    anyhow::bail!("no tokenizer.json in {}", snaps.display())
}
