//! Read-only view of the standard Hugging Face hub cache (`huggingface_hub` / `huggingface-cli`
//! compatible): `models--org--name/{blobs,refs,snapshots/<commit>}`. Snapshot entries are
//! symlinks into `blobs/` on Unix (and Windows with symlink rights) or plain copies/hardlinks
//! when symlinks are unavailable; both are handled.
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// `HF_HUB_CACHE` > `$HF_HOME/hub` > `$XDG_CACHE_HOME/huggingface/hub` > `~/.cache/huggingface/hub`.
pub fn cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") { return PathBuf::from(p); }
    if let Some(p) = std::env::var_os("HF_HOME") { return PathBuf::from(p).join("hub"); }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") { return PathBuf::from(p).join("huggingface").join("hub"); }
    home().join(".cache").join("huggingface").join("hub")
}

pub fn home() -> PathBuf {
    std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

pub fn repo_folder(repo: &str) -> String { format!("models--{}", repo.replace('/', "--")) }

pub fn split_rev(spec: &str) -> (&str, Option<&str>) {
    match spec.split_once('@') { Some((r, v)) => (r, Some(v)), None => (spec, None) }
}

/// A snapshot holds a model: config.json / model_index.json, a Danbooru tagger (selected_tags.csv +
/// weights), or GGUF files (top level or one quant sub-folder).
fn has_model(snap: &Path) -> bool {
    if snap.join("config.json").exists() || snap.join("model_index.json").exists() { return true; }
    if snap.join("selected_tags.csv").exists() && ["weights.pt", "model.safetensors"].iter().any(|f| snap.join(f).exists()) { return true; }
    let gguf = |d: &Path| std::fs::read_dir(d).map(|r| r.flatten().any(|e| e.path().extension().is_some_and(|x| x == "gguf"))).unwrap_or(false);
    gguf(snap) || std::fs::read_dir(snap).map(|r| r.flatten().any(|e| e.path().is_dir() && gguf(&e.path()))).unwrap_or(false)
}

/// Resolve `repo[@rev]` to a snapshot directory if it is present in the cache.
pub fn find(cache: &Path, repo: &str, rev: Option<&str>) -> Option<PathBuf> {
    let root = cache.join(repo_folder(repo));
    let rev = rev.unwrap_or("main");
    let commit = std::fs::read_to_string(root.join("refs").join(rev)).map(|s| s.trim().to_string()).unwrap_or_else(|_| rev.to_string());
    let snap = root.join("snapshots").join(&commit);
    if snap.is_dir() && has_model(&snap) { return Some(snap); }
    // a ref-less cache (e.g. copied around): take the newest snapshot when asking for "main"
    if rev == "main" {
        let mut snaps: Vec<_> = std::fs::read_dir(root.join("snapshots")).ok()?.flatten()
            .filter(|e| has_model(&e.path()))
            .collect();
        snaps.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
        return snaps.last().map(|e| e.path());
    }
    None
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CachedModel {
    pub repo: String,
    pub revision: String,
    pub refs: Vec<String>,
    pub path: PathBuf,
    pub size: u64,
    /// "symlink" (entries point into blobs/), "hardlink", "copy" or "mixed"
    pub mode: String,
}

pub type Blobs = Vec<(u64, PathBuf)>;

pub fn blobs_of(snap: &Path) -> Blobs {
    let Some(dir) = snap.parent().and_then(|p| p.parent()).map(|r| r.join("blobs")) else { return vec![] };
    std::fs::read_dir(dir).map(|rd| rd.flatten().filter_map(|e| Some((e.metadata().ok()?.len(), e.path()))).collect()).unwrap_or_default()
}

/// File in a snapshot -> (real file, link mode, blob it shares storage with).
pub fn resolve_entry(p: &Path, blobs: &Blobs) -> (PathBuf, &'static str, Option<PathBuf>) {
    if let Ok(md) = std::fs::symlink_metadata(p) {
        if md.file_type().is_symlink() {
            let real = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            return (real.clone(), "symlink", Some(real));
        }
        let len = md.len();
        for (l, b) in blobs { if *l == len && same_file::is_same_file(p, b).unwrap_or(false) { return (p.to_path_buf(), "hardlink", Some(b.clone())); } }
    }
    (p.to_path_buf(), "copy", None)
}

pub fn list(cache: &Path) -> Vec<CachedModel> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(cache) else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix("models--") else { continue };
        let repo = rest.replacen("--", "/", 1);
        let root = e.path();
        let mut refs: Vec<(String, String)> = vec![];
        if let Ok(r) = std::fs::read_dir(root.join("refs")) {
            for f in r.flatten() {
                if let Ok(c) = std::fs::read_to_string(f.path()) { refs.push((f.file_name().to_string_lossy().into(), c.trim().into())); }
            }
        }
        let Ok(snaps) = std::fs::read_dir(root.join("snapshots")) else { continue };
        for s in snaps.flatten() {
            let rev = s.file_name().to_string_lossy().to_string();
            let mut size = 0;
            let mut modes = std::collections::BTreeSet::new();
            let blobs = blobs_of(&s.path());
            for f in super::walk(&s.path()) {
                let (real, m, _) = resolve_entry(&f, &blobs);
                size += std::fs::metadata(&real).map(|m| m.len()).unwrap_or(0);
                modes.insert(m);
            }
            let mode = if modes.len() == 1 { modes.into_iter().next().unwrap().to_string() } else if modes.is_empty() { "empty".into() } else { "mixed".into() };
            out.push(CachedModel { repo: repo.clone(), refs: refs.iter().filter(|r| r.1 == rev).map(|r| r.0.clone()).collect(), revision: rev, path: s.path(), size, mode });
        }
    }
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    out
}

pub fn offline() -> bool {
    matches!(std::env::var("HF_HUB_OFFLINE").as_deref(), Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes"))
}

/// Download `repo[@rev]` into the standard cache with hf-hub (token from HF_TOKEN / token file,
/// handled by hf-hub; never logged). Afterwards, snapshot copies are replaced by hardlinks to
/// their blob so the cache uses no duplicate space on filesystems without symlinks.
pub fn download(cache: &Path, repo: &str, rev: Option<&str>, progress: &dyn Fn(&str, usize, usize)) -> Result<PathBuf> {
    if offline() { bail!("{repo} is not in the HF cache ({}) and HF_HUB_OFFLINE / --offline is set", cache.display()); }
    let (owner, name) = repo.split_once('/').context("repo id must look like org/name")?;
    let client = hf_hub::HFClient::builder().cache_dir(cache).user_agent(concat!("pyre/", env!("CARGO_PKG_VERSION"))).build_sync()?;
    let h = client.model(owner, name);
    let get = |f: &str| -> Result<PathBuf> {
        let b = h.download_file().filename(f);
        Ok(match rev { Some(r) => b.revision(r).send()?, None => b.send()? })
    };
    // diffusers-style repos have model_index.json instead of config.json; Danbooru taggers
    // (Kagami / WD) have selected_tags.csv + timm weights (model.safetensors or weights.pt).
    let (first, is_sd, is_tagger) = match get("config.json") {
        Ok(p) => (p, false, false),
        Err(_) => match get("model_index.json") {
            Ok(p) => (p, true, false),
            Err(_) => (get("selected_tags.csv").context("none of config.json, model_index.json or selected_tags.csv found")?, false, true),
        },
    };
    let dir = first.parent().unwrap().to_path_buf();
    let no_tok = !is_sd && std::fs::read_to_string(&first).map(|s| s.contains("\"audio_config\"") && !s.contains("\"text_config\"")).unwrap_or(false);
    let mut files: Vec<String> = vec![];
    if is_tagger {
        let w = get("model.safetensors").or_else(|_| get("weights.pt")).context("tagger repo has neither model.safetensors nor weights.pt")?;
        tracing::info!("tagger weights: {}", w.display());
        progress("done", 1, 1);
        dedupe_snapshot(&dir);
        return Ok(dir);
    }
    if is_sd {
        for sub in ["text_encoder", "unet", "vae", "tokenizer", "scheduler"] {
            for f in ["config.json", "model.safetensors", "diffusion_pytorch_model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json", "scheduler_config.json"] {
                files.push(format!("{sub}/{f}"));
            }
        }
    } else {
        files.extend(["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "special_tokens_map.json", "preprocessor_config.json"].map(String::from));
        match get("model.safetensors.index.json") {
            Ok(idx) => {
                let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(idx)?)?;
                let mut s: Vec<String> = j["weight_map"].as_object().context("weight_map")?.values().filter_map(|v| v.as_str().map(String::from)).collect();
                s.sort(); s.dedup(); files.extend(s);
            }
            Err(_) => files.push("model.safetensors".into()),
        }
    }
    let n = files.len();
    for (i, f) in files.iter().enumerate() {
        progress(f, i, n);
        // tokenizer.json is required for language models only (audio models such as diarizers have none)
        let required = f.ends_with(".safetensors") && !is_sd || (f == "tokenizer.json" && !no_tok);
        match get(f) { Ok(_) => {}, Err(e) if required => return Err(e.context(format!("downloading {f}"))), Err(_) => {} }
    }
    progress("done", n, n);
    dedupe_snapshot(&dir);
    Ok(dir)
}

/// Replace snapshot files that are byte-for-byte copies of a blob with hardlinks to it.
pub fn dedupe_snapshot(snap: &Path) {
    let blobs = blobs_of(snap);
    for f in super::walk(snap) {
        if resolve_entry(&f, &blobs).1 != "copy" { continue; }
        let Ok(len) = std::fs::metadata(&f).map(|m| m.len()) else { continue };
        for (_, b) in blobs.iter().filter(|(l, _)| *l == len) {
            if files_equal(&f, b) {
                let tmp = f.with_extension("pyre-tmp");
                if std::fs::hard_link(b, &tmp).is_ok() && std::fs::rename(&tmp, &f).is_ok() {
                    tracing::info!("dedup: {} is now a hardlink to its blob", f.display());
                } else { let _ = std::fs::remove_file(&tmp); }
                break;
            }
        }
    }
}

fn read_full(f: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::io::Read;
    let mut n = 0;
    while n < buf.len() { let k = f.read(&mut buf[n..])?; if k == 0 { break; } n += k; }
    Ok(n)
}

pub fn files_equal(a: &Path, b: &Path) -> bool {
    let (Ok(mut fa), Ok(mut fb)) = (std::fs::File::open(a), std::fs::File::open(b)) else { return false };
    let mut ba = vec![0u8; 1 << 20];
    let mut bb = vec![0u8; 1 << 20];
    loop {
        let (Ok(na), Ok(nb)) = (read_full(&mut fa, &mut ba), read_full(&mut fb, &mut bb)) else { return false };
        if na != nb || ba[..na] != bb[..nb] { return false; }
        if na == 0 { return true; }
    }
}
